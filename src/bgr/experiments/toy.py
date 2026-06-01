from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from bgr.curve_estimators import IsotonicCurveEstimator
from bgr.envs.synthetic_margin import SyntheticMarginBenchmark
from bgr.metrics import critical_radius, recovery_auc
from bgr.priorities import BGRPriorityScorer
from bgr.records import LevelRecord
from bgr.samplers import mixed_priority_probs, sample_boundary_radius


@dataclass(frozen=True, slots=True)
class ToyResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    best_rauc: float
    history: list[dict[str, float]]


def run_method(config: dict, method: str, seed: int) -> ToyResult:
    exp = config["experiment"]
    bgr_cfg = config.get("bgr", {})
    rng = np.random.default_rng(seed + 10_000)
    bench = SyntheticMarginBenchmark(
        num_states=int(exp["num_states"]),
        sigma_max=float(exp["sigma_max"]),
        learning_rate=float(exp["learning_rate"]),
        seed=seed,
    )
    sigma_max = float(exp["sigma_max"])
    alpha = float(exp.get("alpha", 0.8))
    eval_grid = np.linspace(0.0, sigma_max, int(exp.get("eval_grid_size", 11)))
    records = _init_records(bench, bgr_cfg, alpha, rng)
    scorer = BGRPriorityScorer(target_radius=float(exp.get("target_margin", 0.42)))

    history: list[dict[str, float]] = []
    for step in range(int(exp["iterations"]) + 1):
        if step % int(exp["eval_every"]) == 0:
            metrics = evaluate(bench, eval_grid, alpha)
            metrics["step"] = float(step)
            history.append(metrics)
            if method == "bgr":
                _refresh_records(bench, records, bgr_cfg, alpha, rng, step)

        if step == int(exp["iterations"]):
            break

        for _ in range(int(exp["train_batch_size"])):
            state_idx, sigma = _sample_training_pair(method, bench, records, scorer, config, rng, step)
            bench.train_step(state_idx, sigma)
            if method == "bgr":
                success = bench.rollout(state_idx, sigma, rng)
                records[state_idx].add_observation(sigma, success)
                records[state_idx].replay_count += 1

    final = history[-1]
    return ToyResult(
        method=method,
        seed=seed,
        final_clean=final["clean"],
        final_rauc=final["rauc"],
        final_median_r80=final["median_r80"],
        best_rauc=max(item["rauc"] for item in history),
        history=history,
    )


def evaluate(bench: SyntheticMarginBenchmark, eval_grid: np.ndarray, alpha: float) -> dict[str, float]:
    clean: list[float] = []
    raucs: list[float] = []
    radii: list[float] = []
    for idx, _state in enumerate(bench.states):
        curve = np.array([bench.success_prob(idx, sigma) for sigma in eval_grid], dtype=float)
        clean.append(float(curve[0]))
        raucs.append(recovery_auc(eval_grid, curve, sigma_max=float(eval_grid[-1])))
        radii.append(critical_radius(eval_grid, curve, alpha=alpha))
    return {
        "clean": float(np.mean(clean)),
        "rauc": float(np.mean(raucs)),
        "median_r80": float(np.median(radii)),
        "p25_r80": float(np.quantile(radii, 0.25)),
        "p75_r80": float(np.quantile(radii, 0.75)),
    }


def serialize_result(result: ToyResult) -> dict:
    return asdict(result)


def _init_records(
    bench: SyntheticMarginBenchmark,
    bgr_cfg: dict,
    alpha: float,
    rng: np.random.Generator,
) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    initial = [float(x) for x in bgr_cfg.get("initial_probes", [0.0, 0.25, 0.5, 0.75, 1.0])]
    for idx, state in enumerate(bench.states):
        record = LevelRecord(
            id=f"synthetic_{idx}",
            domain="synthetic_margin",
            task_id=state.task_id,
            clean_success_hat=bench.success_prob(idx, 0.0),
            feasibility_hat=min(1.0, state.feasible_radius / bench.sigma_max),
            sigma_grid=initial,
        )
        estimator = IsotonicCurveEstimator(bench.sigma_max, alpha)
        for sigma in initial:
            for _ in range(int(bgr_cfg.get("min_trials", 3))):
                success = bench.rollout(idx, sigma, rng)
                record.add_observation(sigma, success)
                estimator.update_bernoulli(sigma, success)
        estimate = estimator.fit()
        record.r_alpha_hat = estimate.r_alpha
        record.sharpness_hat = estimate.sharpness
        record.uncertainty_hat = estimate.r_uncertainty
        record.recovery_curve_hat = estimate.recovery.tolist()
        records.append(record)
    return records


def _refresh_records(
    bench: SyntheticMarginBenchmark,
    records: list[LevelRecord],
    bgr_cfg: dict,
    alpha: float,
    rng: np.random.Generator,
    step: int,
) -> None:
    count = min(len(records), int(bgr_cfg.get("refresh_per_eval", 64)))
    for idx in rng.choice(len(records), size=count, replace=False):
        record = records[int(idx)]
        estimator = IsotonicCurveEstimator(bench.sigma_max, alpha)
        for sigma, trials in record.trials.items():
            estimator.update(sigma, record.successes.get(sigma, 0), trials)
        sigma = estimator.next_probe(rng, jitter=0.07)
        for _ in range(2):
            success = bench.rollout(int(idx), sigma, rng)
            record.add_observation(sigma, success)
            estimator.update_bernoulli(sigma, success)
        estimate = estimator.fit()
        record.clean_success_hat = bench.success_prob(int(idx), 0.0)
        record.feasibility_hat = min(1.0, bench.states[int(idx)].feasible_radius / bench.sigma_max)
        record.r_alpha_hat = estimate.r_alpha
        record.sharpness_hat = estimate.sharpness
        record.uncertainty_hat = estimate.r_uncertainty
        record.recovery_curve_hat = estimate.recovery.tolist()
        record.last_evaluated_step = step


def _sample_training_pair(
    method: str,
    bench: SyntheticMarginBenchmark,
    records: list[LevelRecord],
    scorer: BGRPriorityScorer,
    config: dict,
    rng: np.random.Generator,
    step: int,
) -> tuple[int, float]:
    exp = config["experiment"]
    sigma_max = float(exp["sigma_max"])
    if method == "uniform":
        return int(rng.integers(len(bench.states))), float(rng.uniform(0.0, sigma_max))
    if method == "fixed":
        return int(rng.integers(len(bench.states))), float(exp.get("fixed_radius", 0.5))
    if method == "failure_only":
        candidates = rng.choice(len(bench.states), size=min(32, len(bench.states)), replace=False)
        scores = []
        for idx in candidates:
            sigma = float(rng.uniform(0.0, sigma_max))
            scores.append(1.0 - bench.success_prob(int(idx), sigma))
        state_idx = int(candidates[int(np.argmax(scores))])
        sigma = float(rng.uniform(0.45 * sigma_max, sigma_max))
        return state_idx, sigma
    if method == "plr_loss":
        candidates = rng.choice(len(bench.states), size=min(32, len(bench.states)), replace=False)
        sigmas = rng.uniform(0.0, sigma_max, size=len(candidates))
        scores = [bench.imitation_loss_proxy(int(idx), float(sig)) for idx, sig in zip(candidates, sigmas, strict=True)]
        selected = int(np.argmax(scores))
        return int(candidates[selected]), float(sigmas[selected])
    if method == "bgr":
        target = float(np.quantile([record.r_alpha_hat for record in records], 0.60))
        adaptive_scorer = BGRPriorityScorer(
            clean_threshold=scorer.clean_threshold,
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
            temperature=float(config.get("bgr", {}).get("priority_temperature", 0.7)),
            uniform_mix=float(config.get("bgr", {}).get("uniform_mix", 0.08)),
        )
        state_idx = int(rng.choice(len(records), p=probs))
        sigma = sample_boundary_radius(
            rng,
            records[state_idx].r_alpha_hat,
            sigma_max,
            radius_noise=float(config.get("bgr", {}).get("radius_noise", 0.07)),
        )
        return state_idx, sigma
    raise ValueError(f"unknown method: {method}")
