from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from bgr.curve_estimators import IsotonicCurveEstimator
from bgr.envs.grid_recovery import GridRecoveryBenchmark
from bgr.metrics import critical_radius, recovery_auc
from bgr.priorities import BGRPriorityScorer
from bgr.records import LevelRecord
from bgr.samplers import mixed_priority_probs, sample_boundary_radius


@dataclass(frozen=True, slots=True)
class GridResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


def run_method(config: dict, method: str, seed: int) -> GridResult:
    exp = config["experiment"]
    bgr_cfg = config.get("bgr", {})
    rng = np.random.default_rng(seed + 20_000)
    bench = GridRecoveryBenchmark(
        num_tasks=int(exp["num_tasks"]),
        grid_size=int(exp["grid_size"]),
        obstacle_prob=float(exp["obstacle_prob"]),
        replay_states_per_task=int(exp["replay_states_per_task"]),
        max_offset=int(exp["max_offset"]),
        horizon=int(exp["horizon"]),
        learning_rate=float(exp["learning_rate"]),
        seed=seed,
    )
    _pretrain_clean_suffixes(bench, int(exp.get("clean_pretrain_steps", 0)), rng)
    alpha = float(exp.get("alpha", 0.8))
    eval_grid = np.linspace(0.0, 1.0, int(exp.get("eval_grid_size", 9)))
    records = _init_records(bench, bgr_cfg, alpha, rng)
    scorer = BGRPriorityScorer(target_radius=float(exp.get("target_margin", 0.45)))

    history: list[dict[str, float]] = []
    for step in range(int(exp["iterations"]) + 1):
        if step % int(exp["eval_every"]) == 0:
            metrics = evaluate(bench, eval_grid, alpha, int(exp.get("eval_trials_per_radius", 5)), rng)
            metrics["step"] = float(step)
            history.append(metrics)
            if _uses_bgr_records(method):
                _refresh_records(bench, records, bgr_cfg, alpha, rng, step)
        if step == int(exp["iterations"]):
            break
        for _ in range(int(exp["train_batch_size"])):
            state_idx, sigma = _sample_training_pair(method, bench, records, scorer, config, rng, step)
            bench.train_step(state_idx, sigma, rng)
            if _uses_bgr_records(method):
                success = bench.rollout(state_idx, sigma, rng)
                records[state_idx].add_observation(sigma, success)
                records[state_idx].replay_count += 1

    final = history[-1]
    return GridResult(
        method=method,
        seed=seed,
        final_clean=final["clean"],
        final_rauc=final["rauc"],
        final_median_r80=final["median_r80"],
        rauc_aulc=_history_aulc(history, "rauc"),
        best_rauc=max(item["rauc"] for item in history),
        history=history,
    )


def evaluate(
    bench: GridRecoveryBenchmark,
    eval_grid: np.ndarray,
    alpha: float,
    trials_per_radius: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    clean: list[float] = []
    raucs: list[float] = []
    radii: list[float] = []
    for replay_idx in range(len(bench.replay_states)):
        curve = []
        for sigma in eval_grid:
            successes = sum(bench.rollout(replay_idx, float(sigma), rng) for _ in range(trials_per_radius))
            curve.append(successes / trials_per_radius)
        curve_arr = np.array(curve, dtype=float)
        clean.append(float(curve_arr[0]))
        raucs.append(recovery_auc(eval_grid, curve_arr, sigma_max=1.0))
        radii.append(critical_radius(eval_grid, curve_arr, alpha=alpha))
    return {
        "clean": float(np.mean(clean)),
        "rauc": float(np.mean(raucs)),
        "median_r80": float(np.median(radii)),
        "p25_r80": float(np.quantile(radii, 0.25)),
        "p75_r80": float(np.quantile(radii, 0.75)),
    }


def serialize_result(result: GridResult) -> dict:
    return asdict(result)


def _pretrain_clean_suffixes(
    bench: GridRecoveryBenchmark,
    steps: int,
    rng: np.random.Generator,
) -> None:
    for _ in range(max(0, steps)):
        for replay_idx in range(len(bench.replay_states)):
            bench.train_step(replay_idx, 0.0, rng)


def _history_aulc(history: list[dict[str, float]], key: str) -> float:
    if len(history) == 1:
        return float(history[0][key])
    area = 0.0
    for left, right in zip(history[:-1], history[1:], strict=True):
        width = float(right["step"] - left["step"])
        area += width * 0.5 * (float(left[key]) + float(right[key]))
    horizon = float(history[-1]["step"] - history[0]["step"])
    return area / max(horizon, 1e-9)


def _init_records(
    bench: GridRecoveryBenchmark,
    bgr_cfg: dict,
    alpha: float,
    rng: np.random.Generator,
) -> list[LevelRecord]:
    initial = [float(x) for x in bgr_cfg.get("initial_probes", [0.0, 0.25, 0.5, 0.75, 1.0])]
    records: list[LevelRecord] = []
    for replay_idx, replay in enumerate(bench.replay_states):
        record = LevelRecord(
            id=f"grid_{replay_idx}",
            domain="grid_recovery",
            task_id=str(replay.task_id),
            clean_success_hat=0.0,
            feasibility_hat=bench.feasibility(replay_idx, 1.0),
            sigma_grid=initial,
        )
        estimator = IsotonicCurveEstimator(1.0, alpha)
        for sigma in initial:
            for _ in range(int(bgr_cfg.get("min_trials", 2))):
                success = bench.rollout(replay_idx, sigma, rng)
                record.add_observation(sigma, success)
                estimator.update_bernoulli(sigma, success)
        estimate = estimator.fit()
        record.clean_success_hat = record.success_rate(0.0)
        record.r_alpha_hat = estimate.r_alpha
        record.sharpness_hat = estimate.sharpness
        record.uncertainty_hat = estimate.r_uncertainty
        record.recovery_curve_hat = estimate.recovery.tolist()
        records.append(record)
    return records


def _refresh_records(
    bench: GridRecoveryBenchmark,
    records: list[LevelRecord],
    bgr_cfg: dict,
    alpha: float,
    rng: np.random.Generator,
    step: int,
) -> None:
    count = min(len(records), int(bgr_cfg.get("refresh_per_eval", 64)))
    priorities = np.array([1.0 + record.uncertainty_hat for record in records], dtype=float)
    probs = priorities / np.sum(priorities)
    for idx in rng.choice(len(records), size=count, replace=False, p=probs):
        record = records[int(idx)]
        estimator = IsotonicCurveEstimator(1.0, alpha)
        for sigma, trials in record.trials.items():
            estimator.update(sigma, record.successes.get(sigma, 0), trials)
        sigma = estimator.next_probe(rng, jitter=0.08)
        for _ in range(2):
            success = bench.rollout(int(idx), sigma, rng)
            record.add_observation(sigma, success)
            estimator.update_bernoulli(sigma, success)
        estimate = estimator.fit()
        record.clean_success_hat = record.success_rate(0.0)
        record.feasibility_hat = bench.feasibility(int(idx), estimate.r_alpha)
        record.r_alpha_hat = estimate.r_alpha
        record.sharpness_hat = estimate.sharpness
        record.uncertainty_hat = estimate.r_uncertainty
        record.recovery_curve_hat = estimate.recovery.tolist()
        record.last_evaluated_step = step


def _sample_training_pair(
    method: str,
    bench: GridRecoveryBenchmark,
    records: list[LevelRecord],
    scorer: BGRPriorityScorer,
    config: dict,
    rng: np.random.Generator,
    step: int,
) -> tuple[int, float]:
    exp = config["experiment"]
    if method == "uniform":
        return int(rng.integers(len(records))), float(rng.uniform(0.0, 1.0))
    if method == "fixed":
        return int(rng.integers(len(records))), float(exp.get("fixed_radius", 0.75))
    if method == "failure_only":
        candidates = rng.choice(
            len(records),
            size=min(int(exp.get("baseline_candidates", 32)), len(records)),
            replace=False,
        )
        scores = []
        sigmas = rng.uniform(0.0, 1.0, size=len(candidates))
        for idx, sigma in zip(candidates, sigmas, strict=True):
            scores.append(
                1.0
                - _quick_success_rate(
                    bench,
                    int(idx),
                    float(sigma),
                    rng,
                    trials=int(exp.get("failure_probe_trials", 3)),
                )
            )
        selected = int(np.argmax(scores))
        return int(candidates[selected]), float(sigmas[selected])
    if method == "plr_loss":
        candidates = rng.choice(
            len(records),
            size=min(int(exp.get("baseline_candidates", 32)), len(records)),
            replace=False,
        )
        sigmas = rng.uniform(0.0, 1.0, size=len(candidates))
        scores = [bench.loss_proxy(int(idx), float(sigma), rng) for idx, sigma in zip(candidates, sigmas, strict=True)]
        selected = int(np.argmax(scores))
        return int(candidates[selected]), float(sigmas[selected])
    if _uses_bgr_records(method):
        target = float(exp.get("target_margin", 0.45))
        adaptive_scorer = BGRPriorityScorer(
            clean_threshold=0.05,
            feasibility_threshold=scorer.feasibility_threshold,
            target_radius=target,
            radius_bandwidth=scorer.radius_bandwidth,
            sharpness_power=scorer.sharpness_power,
            uncertainty_power=scorer.uncertainty_power,
            staleness_weight=scorer.staleness_weight,
            min_priority=scorer.min_priority,
        )
        priorities = np.array([adaptive_scorer.score(record, step) for record in records], dtype=float)
        probs = mixed_priority_probs(
            priorities,
            temperature=float(config.get("bgr", {}).get("priority_temperature", 0.8)),
            uniform_mix=float(config.get("bgr", {}).get("uniform_mix", 0.10)),
        )
        replay_idx = int(rng.choice(len(records), p=probs))
        sigma = _sample_grid_bgr_radius(method, rng, records[replay_idx].r_alpha_hat, config.get("bgr", {}))
        return replay_idx, sigma
    raise ValueError(f"unknown method: {method}")


def _uses_bgr_records(method: str) -> bool:
    return method in {"bgr", "bgr_mixed"}


def _sample_grid_bgr_radius(
    method: str,
    rng: np.random.Generator,
    r_alpha: float,
    bgr_cfg: dict,
) -> float:
    if method == "bgr_mixed":
        clean_prob = float(bgr_cfg.get("mixed_clean_radius_prob", 0.20))
        uniform_prob = float(bgr_cfg.get("mixed_uniform_radius_prob", 0.45))
        draw = float(rng.random())
        if draw < clean_prob:
            return 0.0
        if draw < clean_prob + uniform_prob:
            return float(rng.uniform(0.0, 1.0))
    return sample_boundary_radius(
        rng,
        r_alpha,
        1.0,
        radius_noise=float(bgr_cfg.get("radius_noise", 0.08)),
    )


def _quick_success_rate(
    bench: GridRecoveryBenchmark,
    replay_idx: int,
    sigma: float,
    rng: np.random.Generator,
    trials: int = 3,
) -> float:
    return sum(bench.rollout(replay_idx, sigma, rng) for _ in range(trials)) / trials
