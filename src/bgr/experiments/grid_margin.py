from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from bgr.curve_estimators import IsotonicCurveEstimator
from bgr.envs.grid_recovery import GridMarginRecoveryBenchmark
from bgr.metrics import critical_radius, recovery_auc
from bgr.priorities import BGRPriorityScorer
from bgr.records import LevelRecord
from bgr.samplers import mixed_priority_probs, sample_boundary_radius


@dataclass(frozen=True, slots=True)
class GridMarginResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


def run_method(config: dict, method: str, seed: int) -> GridMarginResult:
    exp = config["experiment"]
    bgr_cfg = config.get("bgr", {})
    rng = np.random.default_rng(seed + 30_000)
    bench = GridMarginRecoveryBenchmark(
        num_tasks=int(exp["num_tasks"]),
        grid_size=int(exp["grid_size"]),
        obstacle_prob=float(exp["obstacle_prob"]),
        replay_states_per_task=int(exp["replay_states_per_task"]),
        max_offset=int(exp["max_offset"]),
        learning_rate=float(exp["learning_rate"]),
        seed=seed,
    )
    alpha = float(exp.get("alpha", 0.8))
    eval_grid = np.linspace(0.0, 1.0, int(exp.get("eval_grid_size", 9)))
    records = _init_records(bench, bgr_cfg, alpha, rng)
    scorer = BGRPriorityScorer(target_radius=float(exp.get("target_margin", 0.38)))

    history: list[dict[str, float]] = []
    for step in range(int(exp["iterations"]) + 1):
        if step % int(exp["eval_every"]) == 0:
            metrics = evaluate(bench, eval_grid, alpha)
            metrics["step"] = float(step)
            history.append(metrics)
            if _uses_bgr_records(method):
                _refresh_records(bench, records, bgr_cfg, alpha, rng, step)
        if step == int(exp["iterations"]):
            break
        for _ in range(int(exp["train_batch_size"])):
            replay_idx, sigma = _sample_training_pair(method, bench, records, scorer, config, rng, step)
            bench.train_step(replay_idx, sigma, rng)
            if _uses_bgr_records(method):
                success = bench.rollout(replay_idx, sigma, rng)
                records[replay_idx].add_observation(sigma, success)
                records[replay_idx].replay_count += 1

    final = history[-1]
    return GridMarginResult(
        method=method,
        seed=seed,
        final_clean=final["clean"],
        final_rauc=final["rauc"],
        final_median_r80=final["median_r80"],
        rauc_aulc=_history_aulc(history, "rauc"),
        best_rauc=max(item["rauc"] for item in history),
        history=history,
    )


def evaluate(bench: GridMarginRecoveryBenchmark, eval_grid: np.ndarray, alpha: float) -> dict[str, float]:
    clean: list[float] = []
    raucs: list[float] = []
    radii: list[float] = []
    for replay_idx in range(len(bench.states)):
        curve = np.array([bench.success_prob(replay_idx, float(sigma)) for sigma in eval_grid], dtype=float)
        clean.append(float(curve[0]))
        raucs.append(recovery_auc(eval_grid, curve, sigma_max=1.0))
        radii.append(critical_radius(eval_grid, curve, alpha=alpha))
    return {
        "clean": float(np.mean(clean)),
        "rauc": float(np.mean(raucs)),
        "median_r80": float(np.median(radii)),
        "p25_r80": float(np.quantile(radii, 0.25)),
        "p75_r80": float(np.quantile(radii, 0.75)),
    }


def serialize_result(result: GridMarginResult) -> dict:
    return asdict(result)


def _init_records(
    bench: GridMarginRecoveryBenchmark,
    bgr_cfg: dict,
    alpha: float,
    rng: np.random.Generator,
) -> list[LevelRecord]:
    initial = [float(x) for x in bgr_cfg.get("initial_probes", [0.0, 0.25, 0.5, 0.75, 1.0])]
    records: list[LevelRecord] = []
    for replay_idx, state in enumerate(bench.states):
        record = LevelRecord(
            id=f"grid_margin_{replay_idx}",
            domain="grid_margin_recovery",
            task_id=str(state.replay.task_id),
            clean_success_hat=bench.success_prob(replay_idx, 0.0),
            feasibility_hat=bench.feasibility(replay_idx, state.margin),
            sigma_grid=initial,
        )
        estimator = IsotonicCurveEstimator(1.0, alpha)
        for sigma in initial:
            for _ in range(int(bgr_cfg.get("min_trials", 3))):
                success = bench.rollout(replay_idx, sigma, rng)
                record.add_observation(sigma, success)
                estimator.update_bernoulli(sigma, success)
        _write_estimate(record, estimator.fit())
        records.append(record)
    return records


def _refresh_records(
    bench: GridMarginRecoveryBenchmark,
    records: list[LevelRecord],
    bgr_cfg: dict,
    alpha: float,
    rng: np.random.Generator,
    step: int,
) -> None:
    count = min(len(records), int(bgr_cfg.get("refresh_per_eval", 64)))
    stale_unc = np.array([1.0 + record.uncertainty_hat + 0.005 * (step - record.last_evaluated_step) for record in records])
    probs = stale_unc / np.sum(stale_unc)
    for idx in rng.choice(len(records), size=count, replace=False, p=probs):
        record = records[int(idx)]
        estimator = IsotonicCurveEstimator(1.0, alpha)
        for sigma, trials in record.trials.items():
            estimator.update(sigma, record.successes.get(sigma, 0), trials)
        sigma = estimator.next_probe(rng, jitter=0.07)
        for _ in range(2):
            success = bench.rollout(int(idx), sigma, rng)
            record.add_observation(sigma, success)
            estimator.update_bernoulli(sigma, success)
        estimate = estimator.fit()
        record.clean_success_hat = bench.success_prob(int(idx), 0.0)
        record.feasibility_hat = bench.feasibility(int(idx), estimate.r_alpha)
        _write_estimate(record, estimate)
        record.last_evaluated_step = step


def _write_estimate(record: LevelRecord, estimate) -> None:
    record.r_alpha_hat = estimate.r_alpha
    record.sharpness_hat = estimate.sharpness
    record.uncertainty_hat = estimate.r_uncertainty
    record.recovery_curve_hat = estimate.recovery.tolist()


def _sample_training_pair(
    method: str,
    bench: GridMarginRecoveryBenchmark,
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
        candidates = rng.choice(len(records), size=min(int(exp.get("baseline_candidates", 12)), len(records)), replace=False)
        sigmas = rng.uniform(0.0, 1.0, size=len(candidates))
        scores = [1.0 - _quick_success_rate(bench, int(idx), float(sigma), rng, int(exp.get("failure_probe_trials", 1))) for idx, sigma in zip(candidates, sigmas, strict=True)]
        selected = int(np.argmax(scores))
        return int(candidates[selected]), float(sigmas[selected])
    if method == "plr_loss":
        candidates = rng.choice(len(records), size=min(int(exp.get("baseline_candidates", 12)), len(records)), replace=False)
        sigmas = rng.uniform(0.0, 1.0, size=len(candidates))
        scores = [bench.loss_proxy(int(idx), float(sigma), rng) for idx, sigma in zip(candidates, sigmas, strict=True)]
        selected = int(np.argmax(scores))
        return int(candidates[selected]), float(sigmas[selected])
    if _uses_bgr_records(method):
        return _sample_bgr_pair(method, records, scorer, config, rng, step)
    raise ValueError(f"unknown method: {method}")


def _uses_bgr_records(method: str) -> bool:
    return method in {"bgr", "bgr_no_uncertainty", "bgr_no_sharpness", "bgr_uniform_radius"}


def _sample_bgr_pair(
    method: str,
    records: list[LevelRecord],
    scorer: BGRPriorityScorer,
    config: dict,
    rng: np.random.Generator,
    step: int,
) -> tuple[int, float]:
    exp = config["experiment"]
    uncertainty_power = 0.0 if method == "bgr_no_uncertainty" else scorer.uncertainty_power
    sharpness_power = 0.0 if method == "bgr_no_sharpness" else scorer.sharpness_power
    adaptive_scorer = BGRPriorityScorer(
        clean_threshold=scorer.clean_threshold,
        feasibility_threshold=scorer.feasibility_threshold,
        target_radius=float(exp.get("target_margin", scorer.target_radius)),
        radius_bandwidth=scorer.radius_bandwidth,
        sharpness_power=sharpness_power,
        uncertainty_power=uncertainty_power,
        staleness_weight=scorer.staleness_weight,
        min_priority=scorer.min_priority,
    )
    priorities = np.array([adaptive_scorer.score(record, step) for record in records], dtype=float)
    probs = mixed_priority_probs(
        priorities,
        temperature=float(config.get("bgr", {}).get("priority_temperature", 0.7)),
        uniform_mix=float(config.get("bgr", {}).get("uniform_mix", 0.08)),
    )
    replay_idx = int(rng.choice(len(records), p=probs))
    if method == "bgr_uniform_radius":
        sigma = float(rng.uniform(0.0, 1.0))
    else:
        sigma = sample_boundary_radius(
            rng,
            records[replay_idx].r_alpha_hat,
            1.0,
            radius_noise=float(config.get("bgr", {}).get("radius_noise", 0.07)),
        )
    return replay_idx, sigma


def _quick_success_rate(
    bench: GridMarginRecoveryBenchmark,
    replay_idx: int,
    sigma: float,
    rng: np.random.Generator,
    trials: int,
) -> float:
    return sum(bench.rollout(replay_idx, sigma, rng) for _ in range(max(1, trials))) / max(1, trials)


def _history_aulc(history: list[dict[str, float]], key: str) -> float:
    if len(history) == 1:
        return float(history[0][key])
    area = 0.0
    for left, right in zip(history[:-1], history[1:], strict=True):
        width = float(right["step"] - left["step"])
        area += width * 0.5 * (float(left[key]) + float(right[key]))
    horizon = float(history[-1]["step"] - history[0]["step"])
    return area / max(horizon, 1e-9)
