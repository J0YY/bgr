from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from bgr.curve_estimators import IsotonicCurveEstimator
from bgr.envs.frozenlake_recovery import FrozenLakeRecoveryBenchmark
from bgr.metrics import critical_radius, recovery_auc
from bgr.priorities import BGRPriorityScorer
from bgr.records import LevelRecord
from bgr.samplers import mixed_priority_probs, sample_boundary_radius


@dataclass(frozen=True, slots=True)
class FrozenLakeRecoveryResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


def run_method(config: dict, method: str, seed: int) -> FrozenLakeRecoveryResult:
    exp = config["experiment"]
    bgr_cfg = config.get("bgr", {})
    rng = np.random.default_rng(seed + 41_000)
    bench = FrozenLakeRecoveryBenchmark(
        replay_state_count=int(exp.get("replay_state_count", 36)),
        max_radius=int(exp.get("max_radius", 6)),
        learning_rate=float(exp.get("learning_rate", 0.35)),
        discount=float(exp.get("discount", 0.97)),
        epsilon=float(exp.get("epsilon", 0.08)),
        q_init_blend=float(exp.get("q_init_blend", 0.45)),
        q_init_noise=float(exp.get("q_init_noise", 0.03)),
        seed=seed,
    )
    alpha = float(exp.get("alpha", 0.8))
    eval_grid = np.linspace(0.0, 1.0, int(exp.get("eval_grid_size", 7)))
    records = _init_records(bench, bgr_cfg, alpha, rng)
    scorer = BGRPriorityScorer(target_radius=float(exp.get("target_margin", 0.45)))

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
        for _ in range(int(exp.get("train_batch_size", 16))):
            replay_idx, sigma = _sample_training_pair(method, bench, records, scorer, config, rng, step)
            bench.train_step(replay_idx, sigma, rng, max_steps=int(exp.get("episode_max_steps", 96)))
            if _uses_bgr_records(method):
                success = bench.rollout(replay_idx, sigma, rng, max_steps=int(exp.get("episode_max_steps", 96)))
                records[replay_idx].add_observation(sigma, success)
                records[replay_idx].replay_count += 1

    final = history[-1]
    return FrozenLakeRecoveryResult(
        method=method,
        seed=seed,
        final_clean=final["clean"],
        final_rauc=final["rauc"],
        final_median_r80=final["median_r80"],
        rauc_aulc=_history_aulc(history, "rauc"),
        best_rauc=max(item["rauc"] for item in history),
        history=history,
    )


def evaluate(bench: FrozenLakeRecoveryBenchmark, eval_grid: np.ndarray, alpha: float) -> dict[str, float]:
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


def serialize_result(result: FrozenLakeRecoveryResult) -> dict:
    return asdict(result)


def _init_records(
    bench: FrozenLakeRecoveryBenchmark,
    bgr_cfg: dict,
    alpha: float,
    rng: np.random.Generator,
) -> list[LevelRecord]:
    initial = [float(x) for x in bgr_cfg.get("initial_probes", [0.0, 0.25, 0.5, 0.75, 1.0])]
    records: list[LevelRecord] = []
    for replay_idx, state in enumerate(bench.states):
        record = LevelRecord(
            id=f"frozenlake_{state.state}",
            domain="gym_frozenlake_8x8_recovery",
            task_id="FrozenLake8x8-v1",
            clean_success_hat=bench.clean_success(replay_idx),
            feasibility_hat=bench.feasibility(replay_idx, 1.0),
            perturbation_family="safe_manhattan_restart",
            sigma_grid=initial,
        )
        estimator = IsotonicCurveEstimator(1.0, alpha)
        for sigma in initial:
            for _ in range(int(bgr_cfg.get("min_trials", 4))):
                success = bench.rollout(replay_idx, sigma, rng)
                record.add_observation(sigma, success)
                estimator.update_bernoulli(sigma, success)
        _write_estimate(record, estimator.fit())
        records.append(record)
    return records


def _refresh_records(
    bench: FrozenLakeRecoveryBenchmark,
    records: list[LevelRecord],
    bgr_cfg: dict,
    alpha: float,
    rng: np.random.Generator,
    step: int,
) -> None:
    count = min(len(records), int(bgr_cfg.get("refresh_per_eval", 18)))
    stale_unc = np.array([1.0 + record.uncertainty_hat + 0.005 * (step - record.last_evaluated_step) for record in records])
    probs = stale_unc / np.sum(stale_unc)
    for idx in rng.choice(len(records), size=count, replace=False, p=probs):
        record = records[int(idx)]
        estimator = IsotonicCurveEstimator(1.0, alpha)
        for sigma, trials in record.trials.items():
            estimator.update(sigma, record.successes.get(sigma, 0), trials)
        sigma = estimator.next_probe(rng, jitter=0.08)
        for _ in range(int(bgr_cfg.get("refresh_trials", 2))):
            success = bench.rollout(int(idx), sigma, rng)
            record.add_observation(sigma, success)
            estimator.update_bernoulli(sigma, success)
        estimate = estimator.fit()
        record.clean_success_hat = bench.clean_success(int(idx))
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
    bench: FrozenLakeRecoveryBenchmark,
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
        return int(rng.integers(len(records))), float(exp.get("fixed_radius", 0.65))
    if method == "failure_only":
        candidates = rng.choice(len(records), size=min(int(exp.get("baseline_candidates", 12)), len(records)), replace=False)
        sigmas = rng.uniform(0.0, 1.0, size=len(candidates))
        scores = [1.0 - bench.success_prob(int(idx), float(sigma)) for idx, sigma in zip(candidates, sigmas, strict=True)]
        selected = int(np.argmax(scores))
        return int(candidates[selected]), float(sigmas[selected])
    if method == "td_loss":
        candidates = rng.choice(len(records), size=min(int(exp.get("baseline_candidates", 12)), len(records)), replace=False)
        sigmas = rng.uniform(0.0, 1.0, size=len(candidates))
        scores = [bench.loss_proxy(int(idx), float(sigma), rng) for idx, sigma in zip(candidates, sigmas, strict=True)]
        selected = int(np.argmax(scores))
        return int(candidates[selected]), float(sigmas[selected])
    if _uses_bgr_records(method):
        return _sample_bgr_pair(method, records, scorer, config, rng, step)
    raise ValueError(f"unknown method: {method}")


def _uses_bgr_records(method: str) -> bool:
    return method in {"bgr", "bgr_coverage", "bgr_uniform_radius"}


def _sample_bgr_pair(
    method: str,
    records: list[LevelRecord],
    scorer: BGRPriorityScorer,
    config: dict,
    rng: np.random.Generator,
    step: int,
) -> tuple[int, float]:
    exp = config["experiment"]
    adaptive_scorer = BGRPriorityScorer(
        clean_threshold=float(exp.get("clean_threshold", scorer.clean_threshold)),
        feasibility_threshold=0.0,
        target_radius=float(exp.get("target_margin", scorer.target_radius)),
        radius_bandwidth=float(exp.get("radius_bandwidth", scorer.radius_bandwidth)),
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
    if method == "bgr_uniform_radius":
        sigma = float(rng.uniform(0.0, 1.0))
    elif method == "bgr_coverage" and rng.random() < float(config.get("bgr", {}).get("radius_uniform_mix", 0.45)):
        sigma = float(rng.uniform(0.0, 1.0))
    else:
        sigma = sample_boundary_radius(
            rng,
            records[replay_idx].r_alpha_hat,
            1.0,
            radius_noise=float(config.get("bgr", {}).get("radius_noise", 0.08)),
        )
    return replay_idx, sigma


def _quick_success_rate(
    bench: FrozenLakeRecoveryBenchmark,
    replay_idx: int,
    sigma: float,
    rng: np.random.Generator,
    trials: int,
) -> float:
    return sum(bench.rollout(replay_idx, sigma, rng) for _ in range(max(1, trials))) / max(1, trials)


def _history_aulc(history: list[dict[str, float]], key: str) -> float:
    if len(history) == 1:
        return float(history[0][key])
    xs = np.array([row["step"] for row in history], dtype=float)
    ys = np.array([row[key] for row in history], dtype=float)
    try:
        area = np.trapezoid(ys, xs)
    except AttributeError:
        area = np.trapz(ys, xs)
    return float(area / (xs[-1] - xs[0]))
