from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RobotSuffixState:
    id: int
    task_id: str
    family: str
    suffix_phase: float
    feasible_object_radius: float
    feasible_ee_radius: float
    margin_object: float
    margin_ee: float
    clean_success: float
    temperature: float
    teacher_quality: float
    clutter: float


class RobotSuffixRecoveryBenchmark:
    """Lightweight robotics-style suffix recovery benchmark.

    States mimic language-conditioned manipulation suffixes. Perturbation
    families are object-pose offsets and end-effector offsets, normalized to
    [0, 1]. A teacher/feasibility witness limits useful learning beyond each
    state's recoverable radius.
    """

    FAMILIES = ("pick_place", "drawer", "button", "stack")

    def __init__(
        self,
        num_tasks: int,
        suffixes_per_task: int,
        learning_rate: float,
        seed: int,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.learning_rate = float(learning_rate)
        self.states = self._make_states(num_tasks, suffixes_per_task)

    def success_prob(self, state_idx: int, sigma: float, perturbation_family: str = "object") -> float:
        state = self.states[state_idx]
        margin = state.margin_object if perturbation_family == "object" else state.margin_ee
        feasible = self.feasibility(state_idx, sigma, perturbation_family)
        transition = 1.0 / (1.0 + np.exp((sigma - margin) / state.temperature))
        floor = 0.01 * feasible
        visual_penalty = 1.0 - 0.08 * state.clutter * max(0.0, sigma - margin)
        return float(np.clip((floor + (state.clean_success - floor) * transition * feasible) * visual_penalty, 0.0, 1.0))

    def feasibility(self, state_idx: int, sigma: float, perturbation_family: str = "object") -> float:
        state = self.states[state_idx]
        radius = state.feasible_object_radius if perturbation_family == "object" else state.feasible_ee_radius
        if sigma <= radius:
            return state.teacher_quality
        return float(state.teacher_quality * np.exp(-9.0 * (sigma - radius)))

    def rollout(self, state_idx: int, sigma: float, rng: np.random.Generator, perturbation_family: str = "object") -> bool:
        return bool(rng.random() < self.success_prob(state_idx, sigma, perturbation_family))

    def train_step(self, state_idx: int, sigma: float, rng: np.random.Generator, perturbation_family: str = "object") -> float:
        state = self.states[state_idx]
        if perturbation_family == "object":
            margin = state.margin_object
        else:
            margin = state.margin_ee
        feasible = self.feasibility(state_idx, sigma, perturbation_family)
        boundary_signal = np.exp(-((sigma - margin) / 0.14) ** 2)
        clean_signal = 0.20 * np.exp(-(sigma / 0.10) ** 2)
        teacher_signal = feasible * (0.75 + 0.25 * state.teacher_quality)
        gain = self.learning_rate * teacher_signal * (boundary_signal + clean_signal)

        if perturbation_family == "object":
            cap = state.feasible_object_radius
            state.margin_object = float(np.clip(state.margin_object + gain, 0.02, cap))
            state.margin_ee = float(np.clip(state.margin_ee + 0.35 * gain, 0.02, state.feasible_ee_radius))
        else:
            cap = state.feasible_ee_radius
            state.margin_ee = float(np.clip(state.margin_ee + gain, 0.02, cap))
            state.margin_object = float(np.clip(state.margin_object + 0.25 * gain, 0.02, state.feasible_object_radius))

        state.clean_success = float(np.clip(state.clean_success + 0.012 * clean_signal - 0.0015 * max(0.0, sigma - margin), 0.70, 0.995))
        return float(gain)

    def loss_proxy(self, state_idx: int, sigma: float, rng: np.random.Generator, perturbation_family: str = "object") -> float:
        state = self.states[state_idx]
        return float((1.0 - self.success_prob(state_idx, sigma, perturbation_family)) + 0.15 * state.clutter)

    def _make_states(self, num_tasks: int, suffixes_per_task: int) -> list[RobotSuffixState]:
        states: list[RobotSuffixState] = []
        for task_idx in range(num_tasks):
            family = self.FAMILIES[task_idx % len(self.FAMILIES)]
            family_bias = {
                "pick_place": 0.04,
                "drawer": -0.02,
                "button": 0.00,
                "stack": -0.04,
            }[family]
            for suffix_idx in range(suffixes_per_task):
                phase = (suffix_idx + 1) / (suffixes_per_task + 1)
                feasible_object = float(np.clip(0.48 + 0.25 * phase + family_bias + self.rng.normal(0.0, 0.05), 0.35, 0.92))
                feasible_ee = float(np.clip(feasible_object - 0.07 + self.rng.normal(0.0, 0.04), 0.30, 0.88))
                margin_object = float(np.clip(0.12 + 0.18 * phase + 0.5 * family_bias + self.rng.normal(0.0, 0.04), 0.04, 0.38))
                margin_ee = float(np.clip(margin_object - 0.03 + self.rng.normal(0.0, 0.03), 0.03, 0.34))
                clean = float(self.rng.uniform(0.82, 0.97))
                temp = float(self.rng.uniform(0.04, 0.09))
                teacher = float(self.rng.uniform(0.78, 0.98))
                clutter = float(self.rng.uniform(0.0, 1.0))
                states.append(
                    RobotSuffixState(
                        id=len(states),
                        task_id=f"{family}_{task_idx}",
                        family=family,
                        suffix_phase=phase,
                        feasible_object_radius=feasible_object,
                        feasible_ee_radius=feasible_ee,
                        margin_object=margin_object,
                        margin_ee=margin_ee,
                        clean_success=clean,
                        temperature=temp,
                        teacher_quality=teacher,
                        clutter=clutter,
                    )
                )
        return states
