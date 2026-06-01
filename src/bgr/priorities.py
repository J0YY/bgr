from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bgr.records import LevelRecord


@dataclass(frozen=True, slots=True)
class BGRPriorityScorer:
    clean_threshold: float = 0.45
    feasibility_threshold: float = 0.55
    target_radius: float = 0.4
    radius_bandwidth: float = 0.22
    sharpness_power: float = 0.35
    uncertainty_power: float = 0.2
    staleness_weight: float = 0.2
    min_priority: float = 1e-6

    def score(self, record: LevelRecord, step: int = 0) -> float:
        if record.clean_success_hat < self.clean_threshold:
            return self.min_priority
        if record.feasibility_hat < self.feasibility_threshold:
            return self.min_priority

        radius_delta = (record.r_alpha_hat - self.target_radius) / max(self.radius_bandwidth, 1e-9)
        boundary = float(np.exp(-0.5 * radius_delta * radius_delta))
        sharp = max(0.05, record.sharpness_hat) ** self.sharpness_power
        unc = max(0.02, record.uncertainty_hat) ** self.uncertainty_power
        stale = 1.0 + self.staleness_weight * min(1.0, max(0, step - record.last_evaluated_step) / 100.0)
        replay_penalty = 1.0 / np.sqrt(1.0 + record.replay_count / 25.0)
        priority = record.feasibility_hat * boundary * sharp * unc * stale * replay_penalty
        return float(max(self.min_priority, priority))
