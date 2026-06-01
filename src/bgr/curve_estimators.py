from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bgr.metrics import critical_radius, finite_difference_sharpness, recovery_auc


@dataclass(frozen=True, slots=True)
class CurveEstimate:
    sigmas: np.ndarray
    recovery: np.ndarray
    trials: np.ndarray
    r_alpha: float
    r_uncertainty: float
    sharpness: float
    rauc: float


def _pava_nonincreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted PAVA projection onto non-increasing sequences."""

    y = -np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    blocks: list[tuple[float, float, int]] = []
    for value, weight in zip(y, w, strict=True):
        blocks.append((float(value), float(max(weight, 1e-9)), 1))
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            v1, w1, n1 = blocks.pop()
            v0, w0, n0 = blocks.pop()
            merged_w = w0 + w1
            merged_v = (v0 * w0 + v1 * w1) / merged_w
            blocks.append((merged_v, merged_w, n0 + n1))
    fitted: list[float] = []
    for value, _weight, count in blocks:
        fitted.extend([-value] * count)
    return np.clip(np.asarray(fitted, dtype=float), 0.0, 1.0)


class IsotonicCurveEstimator:
    """Assumption-light estimator for monotone recovery curves."""

    def __init__(self, sigma_max: float = 1.0, alpha: float = 0.8) -> None:
        self.sigma_max = float(sigma_max)
        self.alpha = float(alpha)
        self._successes: dict[float, int] = {}
        self._trials: dict[float, int] = {}

    def update(self, sigma: float, successes: int, trials: int = 1) -> None:
        key = round(float(sigma), 6)
        if trials <= 0:
            return
        self._successes[key] = self._successes.get(key, 0) + int(successes)
        self._trials[key] = self._trials.get(key, 0) + int(trials)

    def update_bernoulli(self, sigma: float, success: bool) -> None:
        self.update(sigma, int(success), 1)

    def fit(self) -> CurveEstimate:
        if not self._trials:
            sigmas = np.array([0.0, self.sigma_max], dtype=float)
            recovery = np.array([0.0, 0.0], dtype=float)
            trials = np.zeros(2, dtype=float)
        else:
            sigmas = np.array(sorted(self._trials), dtype=float)
            trials = np.array([self._trials[float(round(s, 6))] for s in sigmas], dtype=float)
            rates = np.array(
                [self._successes.get(float(round(s, 6)), 0) / max(1, self._trials[float(round(s, 6))]) for s in sigmas],
                dtype=float,
            )
            recovery = _pava_nonincreasing(rates, trials)
        r_alpha = critical_radius(sigmas, recovery, alpha=self.alpha)
        uncertainty = self._radius_uncertainty(sigmas, recovery, trials)
        sharpness = finite_difference_sharpness(sigmas, recovery, r_alpha)
        rauc = recovery_auc(sigmas, recovery, sigma_max=self.sigma_max)
        return CurveEstimate(sigmas, recovery, trials, r_alpha, uncertainty, sharpness, rauc)

    def next_probe(self, rng: np.random.Generator, jitter: float = 0.05) -> float:
        estimate = self.fit()
        if np.sum(estimate.trials) == 0:
            return float(self.sigma_max / 2)
        sigma = estimate.r_alpha + rng.normal(0.0, jitter * self.sigma_max)
        return float(np.clip(sigma, 0.0, self.sigma_max))

    def _radius_uncertainty(self, sigmas: np.ndarray, recovery: np.ndarray, trials: np.ndarray) -> float:
        if sigmas.size < 2:
            return self.sigma_max
        total = max(1.0, float(np.sum(trials)))
        local_trials = float(np.interp(critical_radius(sigmas, recovery, self.alpha), sigmas, trials))
        probe_term = 1.0 / np.sqrt(1.0 + local_trials)
        coverage_term = 1.0 / np.sqrt(1.0 + total / max(1, sigmas.size))
        return float(self.sigma_max * (0.5 * probe_term + 0.5 * coverage_term))
