from __future__ import annotations

import numpy as np


def mixed_priority_probs(priorities: np.ndarray, temperature: float = 1.0, uniform_mix: float = 0.05) -> np.ndarray:
    priorities = np.asarray(priorities, dtype=float)
    if priorities.ndim != 1 or priorities.size == 0:
        raise ValueError("priorities must be a non-empty vector")
    scaled = np.maximum(priorities, 1e-12) ** (1.0 / max(temperature, 1e-9))
    probs = scaled / np.sum(scaled)
    mix = float(np.clip(uniform_mix, 0.0, 1.0))
    return (1.0 - mix) * probs + mix / priorities.size


def sample_boundary_radius(
    rng: np.random.Generator,
    r_alpha: float,
    sigma_max: float,
    radius_noise: float = 0.07,
) -> float:
    mode = rng.choice(["boundary", "easy", "hard", "clean"], p=[0.6, 0.15, 0.15, 0.10])
    if mode == "clean":
        return 0.0
    center = float(r_alpha)
    if mode == "easy":
        center *= 0.7
    elif mode == "hard":
        center = min(float(sigma_max), center * 1.3)
    sigma = rng.normal(center, radius_noise * sigma_max)
    return float(np.clip(sigma, 0.0, sigma_max))
