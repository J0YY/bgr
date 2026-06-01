from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from bgr.curve_estimators import IsotonicCurveEstimator
from bgr.envs.synthetic_margin import SyntheticMarginBenchmark
from bgr.metrics import critical_radius, recovery_auc


@dataclass(frozen=True, slots=True)
class EstimatorResult:
    method: str
    seed: int
    probes_per_state: int
    r80_mae: float
    r80_bias: float
    rauc_mae: float
    boundary_hit_rate: float
    mean_uncertainty: float


def run_method(config: dict, method: str, seed: int) -> EstimatorResult:
    exp = config["experiment"]
    sigma_max = float(exp["sigma_max"])
    alpha = float(exp.get("alpha", 0.8))
    probes = int(exp["probes_per_state"])
    rng = np.random.default_rng(seed + 30_000)
    bench = SyntheticMarginBenchmark(
        num_states=int(exp["num_states"]),
        sigma_max=sigma_max,
        learning_rate=0.0,
        seed=seed,
    )
    true_grid = np.linspace(0.0, sigma_max, int(exp.get("true_grid_size", 101)))

    r_errors: list[float] = []
    r_biases: list[float] = []
    rauc_errors: list[float] = []
    hits: list[float] = []
    uncertainties: list[float] = []

    for state_idx in range(len(bench.states)):
        true_curve = np.array([bench.success_prob(state_idx, sigma) for sigma in true_grid], dtype=float)
        true_r = critical_radius(true_grid, true_curve, alpha=alpha)
        true_rauc = recovery_auc(true_grid, true_curve, sigma_max=sigma_max)

        estimator = IsotonicCurveEstimator(sigma_max=sigma_max, alpha=alpha)
        for probe_idx in range(probes):
            sigma = _select_sigma(method, estimator, config, rng, probe_idx)
            success = bench.rollout(state_idx, sigma, rng)
            estimator.update_bernoulli(sigma, success)

        estimate = estimator.fit()
        estimated_curve = np.interp(true_grid, estimate.sigmas, estimate.recovery)
        est_rauc = recovery_auc(true_grid, estimated_curve, sigma_max=sigma_max)
        radius_error = estimate.r_alpha - true_r
        r_errors.append(abs(radius_error))
        r_biases.append(radius_error)
        rauc_errors.append(abs(est_rauc - true_rauc))
        hits.append(float(abs(radius_error) <= float(exp.get("hit_tolerance", 0.08))))
        uncertainties.append(estimate.r_uncertainty)

    return EstimatorResult(
        method=method,
        seed=seed,
        probes_per_state=probes,
        r80_mae=float(np.mean(r_errors)),
        r80_bias=float(np.mean(r_biases)),
        rauc_mae=float(np.mean(rauc_errors)),
        boundary_hit_rate=float(np.mean(hits)),
        mean_uncertainty=float(np.mean(uncertainties)),
    )


def serialize_result(result: EstimatorResult) -> dict:
    return asdict(result)


def _select_sigma(
    method: str,
    estimator: IsotonicCurveEstimator,
    config: dict,
    rng: np.random.Generator,
    probe_idx: int,
) -> float:
    exp = config["experiment"]
    sigma_max = float(exp["sigma_max"])
    initial = [float(sigma) for sigma in config.get("active", {}).get("initial_probes", [0.0, 0.5, 1.0])]
    if method == "uniform":
        return float(rng.uniform(0.0, sigma_max))
    if method == "coarse":
        grid_size = int(exp.get("coarse_grid_size", 5))
        grid = np.linspace(0.0, sigma_max, grid_size)
        return float(grid[probe_idx % grid_size])
    if method == "active":
        initial_trials = int(config.get("active", {}).get("initial_trials", 1))
        initial_budget = min(len(initial) * initial_trials, int(exp["probes_per_state"]))
        if probe_idx < initial_budget:
            return float(np.clip(initial[probe_idx % len(initial)], 0.0, sigma_max))
        sigma = estimator.next_probe(rng, jitter=float(config.get("active", {}).get("jitter", 0.06)))
        sigma += float(config.get("active", {}).get("probe_bias", 0.0)) * sigma_max
        return float(np.clip(sigma, 0.0, sigma_max))
    raise ValueError(f"unknown estimator method: {method}")
