from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SyntheticState:
    id: int
    task_id: str
    feasible_radius: float
    margin: float
    clean_success: float
    temperature: float
    loss_bias: float


class SyntheticMarginBenchmark:
    """Analytic recovery-margin benchmark for Tier-0 BGR validation.

    Each replayable state has a policy margin that training can expand. Success
    probability drops sigmoidally as perturbation radius exceeds that margin.
    Updates near the current margin yield the largest useful improvement.
    """

    def __init__(
        self,
        num_states: int,
        sigma_max: float,
        learning_rate: float,
        seed: int,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.sigma_max = float(sigma_max)
        self.learning_rate = float(learning_rate)
        self.states = self._make_states(num_states)

    def _make_states(self, num_states: int) -> list[SyntheticState]:
        states: list[SyntheticState] = []
        for idx in range(num_states):
            feasible = float(self.rng.uniform(0.45, 1.05))
            margin = float(np.clip(self.rng.normal(0.22, 0.08), 0.04, 0.45))
            clean = float(self.rng.uniform(0.82, 0.99))
            temp = float(self.rng.uniform(0.045, 0.11))
            loss_bias = float(self.rng.uniform(0.0, 0.25))
            states.append(
                SyntheticState(
                    id=idx,
                    task_id=f"task_{idx % 8}",
                    feasible_radius=feasible,
                    margin=margin,
                    clean_success=clean,
                    temperature=temp,
                    loss_bias=loss_bias,
                )
            )
        return states

    def success_prob(self, state_idx: int, sigma: float) -> float:
        state = self.states[state_idx]
        if sigma > state.feasible_radius:
            feasibility_decay = np.exp(-8.0 * (sigma - state.feasible_radius))
        else:
            feasibility_decay = 1.0
        transition = 1.0 / (1.0 + np.exp((sigma - state.margin) / state.temperature))
        floor = 0.03
        return float(np.clip(floor + (state.clean_success - floor) * transition * feasibility_decay, 0.0, 1.0))

    def rollout(self, state_idx: int, sigma: float, rng: np.random.Generator) -> bool:
        return bool(rng.random() < self.success_prob(state_idx, sigma))

    def train_step(self, state_idx: int, sigma: float) -> float:
        state = self.states[state_idx]
        feasible = 1.0 if sigma <= state.feasible_radius else np.exp(-10.0 * (sigma - state.feasible_radius))
        boundary_signal = np.exp(-((sigma - state.margin) / 0.16) ** 2)
        saturation = max(0.0, 1.0 - state.margin / self.sigma_max)
        gain = self.learning_rate * feasible * boundary_signal * saturation
        state.margin = float(np.clip(state.margin + gain, 0.0, min(self.sigma_max, state.feasible_radius)))
        state.clean_success = float(np.clip(state.clean_success - 0.0015 * max(0.0, sigma - state.margin), 0.65, 1.0))
        return float(gain)

    def imitation_loss_proxy(self, state_idx: int, sigma: float) -> float:
        state = self.states[state_idx]
        p = self.success_prob(state_idx, sigma)
        return float((1.0 - p) + state.loss_bias)
