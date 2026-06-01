from __future__ import annotations

import numpy as np


def recovery_auc(sigmas: np.ndarray, recovery: np.ndarray, sigma_max: float | None = None) -> float:
    """Normalize trapezoidal area under a recovery curve."""

    sigmas = np.asarray(sigmas, dtype=float)
    recovery = np.asarray(recovery, dtype=float)
    if sigmas.ndim != 1 or recovery.ndim != 1 or sigmas.size != recovery.size:
        raise ValueError("sigmas and recovery must be same-length vectors")
    if sigmas.size < 2:
        return float(recovery[0]) if sigmas.size == 1 else 0.0
    order = np.argsort(sigmas)
    sigmas = sigmas[order]
    recovery = np.clip(recovery[order], 0.0, 1.0)
    width = float(sigma_max if sigma_max is not None else sigmas[-1] - sigmas[0])
    if width <= 0:
        return float(np.mean(recovery))
    try:
        area = np.trapezoid(recovery, sigmas)
    except AttributeError:  # NumPy < 2.0 compatibility.
        area = np.trapz(recovery, sigmas)
    return float(area / width)


def critical_radius(
    sigmas: np.ndarray,
    recovery: np.ndarray,
    alpha: float = 0.8,
    relative_to_clean: bool = True,
) -> float:
    """Return largest sigma with recovery above the alpha threshold.

    Linear interpolation is used on the first downward crossing.
    """

    sigmas = np.asarray(sigmas, dtype=float)
    recovery = np.asarray(recovery, dtype=float)
    order = np.argsort(sigmas)
    sigmas = sigmas[order]
    recovery = np.clip(recovery[order], 0.0, 1.0)
    if sigmas.size == 0:
        return 0.0
    clean = float(recovery[0])
    threshold = alpha * clean if relative_to_clean else alpha
    above = recovery >= threshold
    if bool(np.all(above)):
        return float(sigmas[-1])
    if not bool(above[0]):
        return float(sigmas[0])
    below_idx = int(np.argmax(~above))
    lo_idx = below_idx - 1
    x0, y0 = sigmas[lo_idx], recovery[lo_idx]
    x1, y1 = sigmas[below_idx], recovery[below_idx]
    if y0 == y1:
        return float(x0)
    frac = (threshold - y0) / (y1 - y0)
    return float(np.clip(x0 + frac * (x1 - x0), x0, x1))


def finite_difference_sharpness(sigmas: np.ndarray, recovery: np.ndarray, radius: float) -> float:
    sigmas = np.asarray(sigmas, dtype=float)
    recovery = np.asarray(recovery, dtype=float)
    if sigmas.size < 2:
        return 0.0
    order = np.argsort(sigmas)
    sigmas = sigmas[order]
    recovery = recovery[order]
    left = np.interp(max(float(sigmas[0]), radius - 0.08), sigmas, recovery)
    right = np.interp(min(float(sigmas[-1]), radius + 0.08), sigmas, recovery)
    return float(max(0.0, (left - right) / max(1e-9, 0.16)))
