from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LevelRecord:
    """Training-time metadata for one replayable decision state."""

    id: str
    domain: str
    task_id: str
    state_ref: object | None = None
    source: str = "unknown"
    source_return: float = 0.0
    clean_success_hat: float = 0.0
    feasibility_hat: float = 1.0
    perturbation_family: str = "synthetic_radius"
    sigma_grid: list[float] = field(default_factory=list)
    trials: dict[float, int] = field(default_factory=dict)
    successes: dict[float, int] = field(default_factory=dict)
    recovery_curve_hat: list[float] = field(default_factory=list)
    r_alpha_hat: float = 0.0
    sharpness_hat: float = 0.0
    uncertainty_hat: float = 1.0
    diversity_key: str = "default"
    last_evaluated_step: int = 0
    priority: float = 1.0
    replay_count: int = 0

    def add_observation(self, sigma: float, success: bool, ndigits: int = 6) -> None:
        key = round(float(sigma), ndigits)
        self.trials[key] = self.trials.get(key, 0) + 1
        self.successes[key] = self.successes.get(key, 0) + int(success)

    @property
    def num_trials(self) -> int:
        return sum(self.trials.values())

    def success_rate(self, sigma: float) -> float:
        key = round(float(sigma), 6)
        n = self.trials.get(key, 0)
        if n == 0:
            return 0.0
        return self.successes.get(key, 0) / n
