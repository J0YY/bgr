from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from bgr.curve_estimators import IsotonicCurveEstimator
from bgr.envs.robot_suffix import RobotSuffixRecoveryBenchmark
from bgr.metrics import critical_radius, recovery_auc
from bgr.priorities import BGRPriorityScorer
from bgr.records import LevelRecord
from bgr.samplers import mixed_priority_probs


@dataclass(frozen=True, slots=True)
class SuffixResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    final_transfer_rauc: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


def run_method(config: dict, method: str, seed: int) -> SuffixResult:
    exp = config["experiment"]
    bgr_cfg = config.get("bgr", {})
    rng = np.random.default_rng(seed + 40_000)
    bench = RobotSuffixRecoveryBenchmark(
        num_tasks=int(exp["num_tasks"]),
        suffixes_per_task=int(exp["suffixes_per_task"]),
        learning_rate=float(exp["learning_rate"]),
        seed=seed,
    )
    alpha = float(exp.get("alpha", 0.8))
    eval_grid = np.linspace(0.0, 1.0, int(exp.get("eval_grid_size", 9)))
    records = _init_records(bench, bgr_cfg, alpha, rng)
    scorer = BGRPriorityScorer(target_radius=float(exp.get("target_margin", 0.36)))

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
            state_idx, sigma = _sample_training_pair(method, bench, records, scorer, config, rng, step)
            bench.train_step(state_idx, sigma, rng, perturbation_family="object")
            if _uses_bgr_records(method):
                success = bench.rollout(state_idx, sigma, rng, perturbation_family="object")
                records[state_idx].add_observation(sigma, success)
                records[state_idx].replay_count += 1

    final = history[-1]
    return SuffixResult(
        method=method,
        seed=seed,
        final_clean=final["clean"],
        final_rauc=final["rauc"],
        final_median_r80=final["median_r80"],
        final_transfer_rauc=final["transfer_rauc"],
        rauc_aulc=_history_aulc(history, "rauc"),
        best_rauc=max(item["rauc"] for item in history),
        history=history,
    )


def evaluate(bench: RobotSuffixRecoveryBenchmark, eval_grid: np.ndarray, alpha: float) -> dict[str, float]:
    clean: list[float] = []
    raucs: list[float] = []
    transfer_raucs: list[float] = []
    radii: list[float] = []
    for state_idx in range(len(bench.states)):
        curve = np.array([bench.success_prob(state_idx, float(sigma), "object") for sigma in eval_grid], dtype=float)
        transfer = np.array([bench.success_prob(state_idx, float(sigma), "ee") for sigma in eval_grid], dtype=float)
        clean.append(float(curve[0]))
        raucs.append(recovery_auc(eval_grid, curve, sigma_max=1.0))
        transfer_raucs.append(recovery_auc(eval_grid, transfer, sigma_max=1.0))
        radii.append(critical_radius(eval_grid, curve, alpha=alpha))
    return {
        "clean": float(np.mean(clean)),
        "rauc": float(np.mean(raucs)),
        "transfer_rauc": float(np.mean(transfer_raucs)),
        "median_r80": float(np.median(radii)),
        "p25_r80": float(np.quantile(radii, 0.25)),
        "p75_r80": float(np.quantile(radii, 0.75)),
    }


def serialize_result(result: SuffixResult) -> dict:
    return asdict(result)


def _init_records(
    bench: RobotSuffixRecoveryBenchmark,
    bgr_cfg: dict,
    alpha: float,
    rng: np.random.Generator,
) -> list[LevelRecord]:
    initial = [float(x) for x in bgr_cfg.get("initial_probes", [0.0, 0.25, 0.5, 0.75, 1.0])]
    records: list[LevelRecord] = []
    for state_idx, state in enumerate(bench.states):
        record = LevelRecord(
            id=f"suffix_{state_idx}",
            domain="robot_suffix",
            task_id=state.task_id,
            clean_success_hat=bench.success_prob(state_idx, 0.0, "object"),
            feasibility_hat=bench.feasibility(state_idx, state.margin_object, "object"),
            sigma_grid=initial,
        )
        estimator = IsotonicCurveEstimator(1.0, alpha)
        for sigma in initial:
            for _ in range(int(bgr_cfg.get("min_trials", 3))):
                success = bench.rollout(state_idx, sigma, rng, "object")
                record.add_observation(sigma, success)
                estimator.update_bernoulli(sigma, success)
        _write_estimate(record, estimator.fit())
        records.append(record)
    return records


def _refresh_records(
    bench: RobotSuffixRecoveryBenchmark,
    records: list[LevelRecord],
    bgr_cfg: dict,
    alpha: float,
    rng: np.random.Generator,
    step: int,
) -> None:
    count = min(len(records), int(bgr_cfg.get("refresh_per_eval", 64)))
    weights = np.array([1.0 + record.uncertainty_hat + 0.004 * (step - record.last_evaluated_step) for record in records])
    probs = weights / np.sum(weights)
    for idx in rng.choice(len(records), size=count, replace=False, p=probs):
        record = records[int(idx)]
        estimator = IsotonicCurveEstimator(1.0, alpha)
        for sigma, trials in record.trials.items():
            estimator.update(sigma, record.successes.get(sigma, 0), trials)
        sigma = estimator.next_probe(rng, jitter=0.07)
        for _ in range(2):
            success = bench.rollout(int(idx), sigma, rng, "object")
            record.add_observation(sigma, success)
            estimator.update_bernoulli(sigma, success)
        estimate = estimator.fit()
        record.clean_success_hat = bench.success_prob(int(idx), 0.0, "object")
        record.feasibility_hat = bench.feasibility(int(idx), estimate.r_alpha, "object")
        _write_estimate(record, estimate)
        record.last_evaluated_step = step


def _sample_training_pair(
    method: str,
    bench: RobotSuffixRecoveryBenchmark,
    records: list[LevelRecord],
    scorer: BGRPriorityScorer,
    config: dict,
    rng: np.random.Generator,
    step: int,
) -> tuple[int, float]:
    exp = config["experiment"]
    if method == "clean_ft":
        return int(rng.integers(len(records))), 0.0
    if method == "uniform":
        return int(rng.integers(len(records))), float(rng.uniform(0.0, 1.0))
    if method == "fixed":
        return int(rng.integers(len(records))), float(exp.get("fixed_radius", 0.70))
    if method == "failure_only":
        candidates = rng.choice(len(records), size=min(int(exp.get("baseline_candidates", 12)), len(records)), replace=False)
        sigmas = rng.uniform(0.0, 1.0, size=len(candidates))
        scores = [1.0 - _quick_success_rate(bench, int(idx), float(sigma), rng, int(exp.get("failure_probe_trials", 1))) for idx, sigma in zip(candidates, sigmas, strict=True)]
        selected = int(np.argmax(scores))
        return int(candidates[selected]), float(sigmas[selected])
    if method == "loss_priority":
        candidates = rng.choice(len(records), size=min(int(exp.get("baseline_candidates", 12)), len(records)), replace=False)
        sigmas = rng.uniform(0.0, 1.0, size=len(candidates))
        scores = [bench.loss_proxy(int(idx), float(sigma), rng, "object") for idx, sigma in zip(candidates, sigmas, strict=True)]
        selected = int(np.argmax(scores))
        return int(candidates[selected]), float(sigmas[selected])
    if _uses_bgr_records(method):
        bgr_scorer = BGRPriorityScorer(
            clean_threshold=scorer.clean_threshold,
            feasibility_threshold=scorer.feasibility_threshold,
            target_radius=float(exp.get("target_margin", scorer.target_radius)),
            radius_bandwidth=scorer.radius_bandwidth,
            sharpness_power=scorer.sharpness_power,
            uncertainty_power=scorer.uncertainty_power,
            staleness_weight=scorer.staleness_weight,
            min_priority=scorer.min_priority,
        )
        priorities = np.array([bgr_scorer.score(record, step) for record in records], dtype=float)
        probs = mixed_priority_probs(
            priorities,
            temperature=float(config.get("bgr", {}).get("priority_temperature", 0.7)),
            uniform_mix=float(config.get("bgr", {}).get("uniform_mix", 0.10)),
        )
        state_idx = int(rng.choice(len(records), p=probs))
        sigma = _sample_suffix_bgr_radius(
            method,
            rng,
            records[state_idx].r_alpha_hat,
            config.get("bgr", {}),
        )
        return state_idx, sigma
    raise ValueError(f"unknown method: {method}")


def _uses_bgr_records(method: str) -> bool:
    return method in {"bgr", "bgr_boundary", "bgr_broad", "bgr_hard"}


def _write_estimate(record: LevelRecord, estimate) -> None:
    record.r_alpha_hat = estimate.r_alpha
    record.sharpness_hat = estimate.sharpness
    record.uncertainty_hat = estimate.r_uncertainty
    record.recovery_curve_hat = estimate.recovery.tolist()


def _sample_suffix_bgr_radius(
    method: str,
    rng: np.random.Generator,
    r_alpha: float,
    bgr_cfg: dict,
) -> float:
    if method == "bgr_boundary":
        clean_prob = float(bgr_cfg.get("boundary_clean_radius_prob", 0.08))
        uniform_prob = float(bgr_cfg.get("boundary_uniform_radius_prob", 0.05))
        mode_probs = [0.75, 0.15, 0.10]
    elif method == "bgr_broad":
        clean_prob = float(bgr_cfg.get("broad_clean_radius_prob", 0.08))
        uniform_prob = float(bgr_cfg.get("broad_uniform_radius_prob", 0.60))
        mode_probs = [0.35, 0.05, 0.60]
    elif method == "bgr_hard":
        clean_prob = float(bgr_cfg.get("hard_clean_radius_prob", 0.06))
        uniform_prob = float(bgr_cfg.get("hard_uniform_radius_prob", 0.20))
        mode_probs = [0.30, 0.05, 0.65]
    else:
        clean_prob = float(bgr_cfg.get("clean_radius_prob", 0.15))
        uniform_prob = float(bgr_cfg.get("uniform_radius_prob", 0.25))
        mode_probs = [0.58, 0.17, 0.25]
    draw = float(rng.random())
    if draw < clean_prob:
        return 0.0
    if draw < clean_prob + uniform_prob:
        return float(rng.uniform(0.0, 1.0))
    mode = rng.choice(["boundary", "easy", "hard"], p=mode_probs)
    center = float(r_alpha)
    if mode == "easy":
        center *= 0.7
    elif mode == "hard":
        center = min(1.0, center * 1.35 + 0.03)
    sigma = rng.normal(center, float(bgr_cfg.get("radius_noise", 0.07)))
    return float(np.clip(sigma, 0.0, 1.0))


def _quick_success_rate(
    bench: RobotSuffixRecoveryBenchmark,
    state_idx: int,
    sigma: float,
    rng: np.random.Generator,
    trials: int,
) -> float:
    return sum(bench.rollout(state_idx, sigma, rng, "object") for _ in range(max(1, trials))) / max(1, trials)


def _history_aulc(history: list[dict[str, float]], key: str) -> float:
    if len(history) == 1:
        return float(history[0][key])
    area = 0.0
    for left, right in zip(history[:-1], history[1:], strict=True):
        width = float(right["step"] - left["step"])
        area += width * 0.5 * (float(left[key]) + float(right[key]))
    horizon = float(history[-1]["step"] - history[0]["step"])
    return area / max(horizon, 1e-9)
