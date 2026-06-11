#!/usr/bin/env python3
"""Generate a quantitative scorecard for the internal AAAI-readiness gap."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from scripts.check_acceptance_readiness import OPENVLA_LEGACY_COMPLETE
from scripts.check_acceptance_readiness import OPENVLA_NON_IDENTITY_PERTURBATIONS
from scripts.check_acceptance_readiness import OPENVLA_PERTURB_ONLY_ANCHOR_COMPLETE
from scripts.check_acceptance_readiness import OPENVLA_PERTURB_ONLY_ANCHOR_MARKER
from scripts.check_acceptance_readiness import OPENVLA_PROXIMAL_ANCHOR_COMPLETE
from scripts.check_acceptance_readiness import OPENVLA_WEIGHTED_COMPLETE
from scripts.check_acceptance_readiness import partial_openvla_failure_detail

OPENVLA_OCCLUSION_BOTTLENECK_MARKER = "scripts/queue_openvla_oft_preregistered_occlusion_bottleneck.sh"
OPENVLA_OCCLUSION_BOTTLENECK_COMPLETE = (
    "results/openvla_oft_perturb_eval_cleanmix_p2048unique_occlusion_bottleneck_prereg_"
    "proxanchor_l2_5em0_step50400_lr2em7_identitylora_imageaug_officialtrainstats_"
    "fullgoal10x10_perturb_v1/summary.csv"
)
OPENVLA_HARD_OCCLUSION_TRANSFER_COMPLETE = (
    "results/openvla_oft_perturb_eval_occlusion_bottleneck_hardocc065_transfer_step50400_lr2em7_v1/summary.csv"
)
OPENVLA_HARD_OCCLUSION080_TRANSFER_COMPLETE = (
    "results/openvla_oft_perturb_eval_occlusion_bottleneck_hardocc080_transfer_step50400_lr2em7_v1/summary.csv"
)
OPENVLA_HARD_OCCLUSION080_TRANSFER_HEADINTERP000_LORAFULL_NOVIDEO_COMPLETE = (
    "results/openvla_oft_perturb_eval_occlusion_bottleneck_hardocc080_transfer_headinterp000_lorafull_novideo_v1/"
    "summary.csv"
)
OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_COMPLETE = (
    "results/openvla_oft_perturb_eval_cleanmix_p2048unique_hardocc080_identityanchor_micro_prereg_proxanchor_l2_1em2_"
    "step50050_lr5em8_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1/summary.csv"
)
OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_A40_COMPLETE = (
    "results/openvla_oft_perturb_eval_cleanmix_p2048unique_hardocc080_identityanchor_micro_a40_prereg_proxanchor_l2_1em2_"
    "step50050_lr5em8_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1/summary.csv"
)
OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_COMPLETE = (
    "results/openvla_oft_perturb_eval_cleanmix_p2048unique_hardocc080_identityanchor_prereg_proxanchor_l2_2em1_"
    "step50200_lr1em7_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1/summary.csv"
)
OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_A40_COMPLETE = (
    "results/openvla_oft_perturb_eval_cleanmix_p2048unique_hardocc080_identityanchor_a40_prereg_proxanchor_l2_2em1_"
    "step50200_lr1em7_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1/summary.csv"
)
OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_STRICT_COMPLETE = (
    "results/openvla_oft_perturb_eval_cleanmix_p2048unique_hardocc080_identityanchor_strict_prereg_proxanchor_l2_5em1_"
    "step50100_lr5em8_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1/summary.csv"
)
OPENVLA_HARD_OCCLUSION090_IDENTITY_ANCHOR_STRICT_COMPLETE = (
    "results/openvla_oft_perturb_eval_cleanmix_p2048unique_hardocc090_identityanchor_strict_prereg_proxanchor_l2_5em1_"
    "step50100_lr5em8_identitylora_imageaug_officialtrainstats_hardocc090_fullgoal10x40_v1/summary.csv"
)
OPENVLA_HARD_OCCLUSION_ADAPT_COMPLETE = (
    "results/openvla_oft_perturb_eval_hardocc065_adapt_step50400_lr2em7_v1/summary.csv"
)
OPENVLA_HARD_OCCLUSION_ADAPT_A40_COMPLETE = (
    "results/openvla_oft_perturb_eval_hardocc065_a40_adapt_step50400_lr2em7_v1/summary.csv"
)
OPENVLA_HARD_OCCLUSION_TRANSFER_MARKER = "sync_openvla_oft_hard_occlusion_transfer_results.sh"
OPENVLA_HARD_OCCLUSION080_TRANSFER_MARKER = "sync_openvla_oft_hard_occlusion080_transfer_results.sh"
OPENVLA_HARD_OCCLUSION080_TRANSFER_HEADINTERP000_LORAFULL_NOVIDEO_MARKER = (
    "headinterp000_lorafull_novideo_v1"
)
OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_MARKER = (
    "sync_openvla_oft_hard_occlusion080_identityanchor_micro_results.sh"
)
OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_A40_MARKER = (
    "sync_openvla_oft_hard_occlusion080_identityanchor_micro_a40_results.sh"
)
OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MARKER = "sync_openvla_oft_hard_occlusion080_identityanchor_results.sh"
OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_A40_MARKER = (
    "sync_openvla_oft_hard_occlusion080_identityanchor_a40_results.sh"
)
OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_STRICT_MARKER = (
    "sync_openvla_oft_hard_occlusion080_identityanchor_strict_results.sh"
)
OPENVLA_HARD_OCCLUSION090_IDENTITY_ANCHOR_STRICT_MARKER = (
    "sync_openvla_oft_hard_occlusion090_identityanchor_strict_results.sh"
)
OPENVLA_HARD_OCCLUSION_ADAPT_MARKER = "sync_openvla_oft_hard_occlusion_adapt_results.sh"
OPENVLA_HARD_OCCLUSION_ADAPT_A40_MARKER = "sync_openvla_oft_hard_occlusion_adapt_a40_results.sh"

COMPLETED_METHOD_SCREEN_BY_CALIBRATION = {
    "FetchPush-v4 object-state calibration": (
        "results/fetchpush_object_state_recovery_probe_densecommon_bgr_v1_778106/summary.csv"
    ),
    "Gymnasium MuJoCo Reacher-v5 calibration": "results/reacher_recovery_probe_12seed_v1/summary.csv",
    "Gymnasium MuJoCo HalfCheetah-v5 calibration": "results/halfcheetah_recovery_probe_4seed_v1_784535/summary.csv",
    "Gymnasium MuJoCo InvertedPendulum-v5 calibration": "results/inverted_pendulum_recovery_probe_4seed_v1/summary.csv",
    "Gymnasium MuJoCo InvertedDoublePendulum-v5 calibration": "results/inverted_double_pendulum_recovery_probe_4seed_v1/summary.csv",
    "Gymnasium Box2D LunarLander-v3 calibration": "results/lunarlander_recovery_probe_30seed_v3_782056_782062/summary.csv",
    "Gymnasium Box2D LunarLanderContinuous-v3 calibration": (
        "results/lunarlander_continuous_recovery_probe_4seed_v1_784643/summary.csv"
    ),
    "MinAtar Breakout calibration": "results/minatar_breakout_recovery_probe_4seed_v1/summary.csv",
    "MinAtar Asterix calibration": "results/minatar_asterix_recovery_probe_4seed_v1/summary.csv",
    "MinAtar Freeway calibration": "results/minatar_freeway_recovery_probe_4seed_v1/summary.csv",
    "MinAtar Space Invaders calibration": "results/minatar_space_invaders_recovery_probe_4seed_v1/summary.csv",
}


@dataclass(frozen=True)
class PairedComparison:
    treatment: str
    baseline: str
    treatment_mean: float
    baseline_mean: float
    delta: float
    wins: int
    losses: int
    ties: int


@dataclass(frozen=True)
class BenchmarkCandidate:
    name: str
    path: str
    treatment: str
    seeds: int
    radius_metric: str
    vs_uniform: PairedComparison | None
    worst_required: PairedComparison | None
    vs_ablation: PairedComparison | None
    radius_vs_uniform: PairedComparison | None

    @property
    def clears_uniform_gate(self) -> bool:
        return (
            self.vs_uniform is not None
            and self.seeds >= 4
            and self.vs_uniform.delta >= 0.01
            and self.vs_uniform.wins >= min(3, self.seeds)
        )

    @property
    def clears_required_baselines(self) -> bool:
        return self.worst_required is not None and self.worst_required.delta > 0.0

    @property
    def clears_ablation_gate(self) -> bool:
        return self.vs_ablation is None or self.vs_ablation.delta > 0.0

    @property
    def clears_radius_gate(self) -> bool:
        return (
            self.radius_vs_uniform is None
            or (
                self.radius_vs_uniform.delta >= 0.0
                and not self.radius_ceiling_saturated
                and not self.radius_floor_saturated
            )
        )

    @property
    def radius_ceiling_saturated(self) -> bool:
        return (
            self.radius_vs_uniform is not None
            and self.radius_vs_uniform.treatment_mean >= 0.99
            and self.radius_vs_uniform.baseline_mean >= 0.99
            and abs(self.radius_vs_uniform.delta) < 1e-12
        )

    @property
    def radius_floor_saturated(self) -> bool:
        return (
            self.radius_vs_uniform is not None
            and self.radius_vs_uniform.treatment_mean <= 0.01
            and self.radius_vs_uniform.baseline_mean <= 0.01
            and abs(self.radius_vs_uniform.delta) < 1e-12
        )

    @property
    def promotable_screen(self) -> bool:
        return (
            self.clears_uniform_gate
            and self.clears_required_baselines
            and self.clears_ablation_gate
            and self.clears_radius_gate
        )

    @property
    def cleared_gate_count(self) -> int:
        return sum(
            [
                self.clears_uniform_gate,
                self.clears_required_baselines,
                self.clears_ablation_gate,
                self.clears_radius_gate,
            ]
        )

    @property
    def failure_reasons(self) -> list[str]:
        reasons: list[str] = []
        if not self.clears_uniform_gate:
            reasons.append("uniform-gate")
        if not self.clears_required_baselines:
            reasons.append("required-baseline")
        if not self.clears_ablation_gate:
            reasons.append("state-priority-ablation")
        if self.radius_ceiling_saturated:
            reasons.append(f"{self.radius_metric}-ceiling-saturated")
        elif self.radius_floor_saturated:
            reasons.append(f"{self.radius_metric}-floor-saturated")
        elif not self.clears_radius_gate:
            reasons.append(f"{self.radius_metric}-contradiction")
        return reasons

    @property
    def priority_key(self) -> tuple[int, int, float, float]:
        uniform_delta = self.vs_uniform.delta if self.vs_uniform is not None else -999.0
        worst_delta = self.worst_required.delta if self.worst_required is not None else -999.0
        return int(self.promotable_screen), self.cleared_gate_count, uniform_delta, worst_delta


@dataclass(frozen=True)
class CalibrationScreen:
    name: str
    path: str
    clean_success: float
    min_recovery: float
    max_recovery: float
    r80: float

    @property
    def recovery_range(self) -> float:
        return self.max_recovery - self.min_recovery

    @property
    def usable(self) -> bool:
        return self.clean_success >= 0.80 and self.recovery_range >= 0.20

    @property
    def decision(self) -> str:
        return "usable-calibration" if self.usable else "reject-calibration"


@dataclass(frozen=True)
class RouteScout:
    name: str
    path: str
    target_radius: float
    treatment_mean: float
    uniform_mean: float
    delta: float
    wins: int
    losses: int
    ties: int
    seeds: int
    best_fixed_mean: float
    best_fixed_target: float

    @property
    def fixed_delta(self) -> float:
        return self.treatment_mean - self.best_fixed_mean

    @property
    def decision(self) -> str:
        if self.seeds >= 30:
            if self.delta >= 0.03 and self.wins >= 20 and self.fixed_delta >= 0.03:
                if (
                    "openml_numeric_external_fixed_target2_30seed_v1" in self.path
                    and "blood-transfusion-service-center" not in self.name
                    and "phoneme" not in self.name
                ) or "openml_broad_numeric_target2_30seed_v1" in self.path:
                    return "candidate-for-replication"
                return "positive-follow-up"
            return "reject-follow-up"
        if self.delta >= 0.03 and self.wins >= 3 and self.losses == 0:
            return "candidate-for-preregistration"
        return "reject-scout"

    @property
    def needs_preregistration(self) -> bool:
        return self.decision in {"candidate-for-preregistration", "candidate-for-replication"}

    @property
    def positive_followup(self) -> bool:
        return self.decision == "positive-follow-up"

    @property
    def rejected(self) -> bool:
        return self.decision.startswith("reject")


def route_scout_evidence_key(scout: RouteScout) -> str:
    """Group exploratory scouts with their fixed follow-up evidence."""
    if scout.name == "OpenML margin replay (diabetes)" or scout.name.startswith("OpenML diabetes margin"):
        return "openml_diabetes_margin"
    if "blood-transfusion-service-center" in scout.name or scout.name.startswith("OpenML blood transfusion margin"):
        return "openml_blood_transfusion_margin"
    if "phoneme" in scout.name or scout.name.startswith("OpenML phoneme margin"):
        return "openml_phoneme_margin"
    if "MagicTelescope" in scout.name:
        return "openml_magic_telescope_margin"
    if "haberman" in scout.name:
        return "openml_haberman_margin"
    if "heart-statlog" in scout.name:
        return "openml_heart_statlog_margin"
    return scout.name


BENCHMARK_SCREENS = [
    (
        "FrozenLake8x8",
        "results/frozenlake_recovery_focused_30seed_v1/summary.csv",
        ["bgr"],
        ["failure_only"],
        None,
        "final_median_r80",
    ),
    (
        "MiniGrid FourRooms official-package",
        "results/minigrid_fourrooms_recovery_probe_4seed_v1/summary.csv",
        ["bgr_coverage", "bgr"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "MiniGrid FourRooms abs-r10 follow-up",
        "results/minigrid_fourrooms_recovery_probe_absr10_4seed_v1/summary.csv",
        ["bgr_coverage", "bgr"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_abs_r10",
    ),
    (
        "MiniGrid FourRooms max-radius-10 follow-up",
        "results/minigrid_fourrooms_recovery_probe_maxr10_4seed_v1/summary.csv",
        ["bgr_coverage", "bgr"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "MiniGrid FourRooms midband",
        "results/minigrid_fourrooms_recovery_probe_midband_4seed_v1/summary.csv",
        ["bgr_coverage", "bgr"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "MiniGrid FourRooms mid2-5",
        "results/minigrid_fourrooms_recovery_probe_mid2_5_4seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "MiniGrid DoorKey",
        "results/minigrid_doorkey_recovery_probe_4seed_v1/summary.csv",
        ["bgr_coverage", "bgr"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_abs_r10",
    ),
    (
        "MiniGrid LavaCrossing",
        "results/minigrid_lavacrossing_recovery_probe_4seed_v1/summary.csv",
        ["bgr_coverage", "bgr"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_abs_r10",
    ),
    (
        "MiniGrid LavaGapS7",
        "results/minigrid_lavagap_s7_recovery_probe_4seed_v1/summary.csv",
        ["bgr_coverage", "bgr"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_abs_r10",
    ),
    (
        "MiniGrid DynamicObstacles",
        "results/minigrid_dynamic_obstacles_recovery_probe_4seed_v1_779232/summary.csv",
        ["bgr_coverage", "bgr"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "MiniGrid DynamicObstacles clean-shield",
        "results/minigrid_dynamic_obstacles_clean_shield_probe_4seed_v1_779412/summary.csv",
        ["bgr_clean_shield", "bgr_coverage", "bgr"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "PointMaze U-Maze clean-shield",
        "results/pointmaze_umaze_clean_shield_probe_4seed_v1/summary.csv",
        ["bgr_clean_shield", "bgr_coverage", "bgr"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_abs_r20",
    ),
    (
        "FetchReach-v4 goal recovery",
        "results/fetchreach_goal_recovery_probe_4seed_v1/summary.csv",
        ["bgr_coverage", "bgr"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "FetchReach-v4 hard-budget goal recovery",
        "results/fetchreach_goal_recovery_hard_probe_4seed_v1/summary.csv",
        ["bgr_coverage", "bgr"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "Gymnasium Taxi-v3 default budget",
        "results/taxi_recovery_probe_4seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "Gymnasium Taxi-v3 hard budget",
        "results/taxi_recovery_hard_probe_4seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "Gymnasium MuJoCo Reacher-v5",
        "results/reacher_recovery_probe_12seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "Gymnasium MuJoCo InvertedPendulum-v5",
        "results/inverted_pendulum_recovery_probe_4seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "Gymnasium MuJoCo InvertedDoublePendulum-v5",
        "results/inverted_double_pendulum_recovery_probe_4seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "Gymnasium Box2D LunarLander-v3 30-seed",
        "results/lunarlander_recovery_probe_30seed_v3_782056_782062/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "Gymnasium Box2D LunarLander-v3 4-seed",
        "results/lunarlander_recovery_probe_4seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "Gymnasium Acrobot package-state",
        "results/acrobot_package_recovery_probe_4seed_v1_783971/summary.csv",
        ["bgr_coverage", "bgr"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "bsuite DeepSea",
        "results/bsuite_deepsea_recovery_probe_4seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "bsuite Catch 30-seed",
        "results/bsuite_catch_recovery_probe_30seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "bsuite MountainCar",
        "results/bsuite_mountaincar_recovery_probe_4seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "bsuite Cartpole",
        "results/bsuite_cartpole_recovery_probe_4seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "bsuite Cartpole Swingup",
        "results/bsuite_cartpole_swingup_recovery_probe_4seed_v1_782844/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "MinAtar Breakout",
        "results/minatar_breakout_recovery_probe_4seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "MinAtar Asterix",
        "results/minatar_asterix_recovery_probe_4seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "MinAtar Freeway",
        "results/minatar_freeway_recovery_probe_4seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
    (
        "MinAtar Space Invaders",
        "results/minatar_space_invaders_recovery_probe_4seed_v1/summary.csv",
        ["bgr", "bgr_coverage"],
        ["fixed", "failure_only", "td_loss"],
        "bgr_uniform_radius",
        "final_median_r80",
    ),
]

CALIBRATION_SCREENS = [
    (
        "FetchPush-v4 object-goal calibration",
        "results/fetchpush_object_goal_calibration_2seed_v1/summary.json",
    ),
    (
        "FetchPush-v4 far-push object-goal calibration",
        "results/fetchpush_object_goal_calibration_far_push_2seed_v1/summary.json",
    ),
    (
        "FetchPush-v4 object-state calibration",
        "results/fetchpush_object_state_calibration_sweep_g8_h250_2seed_v1/summary.json",
    ),
    (
        "FetchSlide-v4 object-goal calibration",
        "results/fetchslide_object_goal_calibration_2seed_v1/summary.json",
    ),
    (
        "FetchPickAndPlace-v4 object-goal calibration",
        "results/fetchpickplace_object_goal_calibration_2seed_v1/summary.json",
    ),
    (
        "Gymnasium-Robotics HandReach-v3 calibration",
        "results/handreach_recovery_calibration_8seed_v1/summary.json",
    ),
    (
        "highway-env parking-v0 calibration",
        "results/highway_parking_recovery_calibration_12seed_v1/summary.json",
    ),
    (
        "highway-env highway-fast-v0 lane calibration",
        "results/highway_lane_recovery_calibration_12seed_v1/summary.json",
    ),
    (
        "MinAtar Breakout calibration",
        "results/minatar_breakout_recovery_calibration_12seed_v1/summary.json",
    ),
    (
        "MinAtar Asterix calibration",
        "results/minatar_asterix_recovery_calibration_12seed_v1/summary.json",
    ),
    (
        "MinAtar Freeway calibration",
        "results/minatar_freeway_recovery_calibration_20seed_v1/summary.json",
    ),
    (
        "MinAtar Space Invaders calibration",
        "results/minatar_space_invaders_recovery_calibration_20seed_v1/summary.json",
    ),
    (
        "Gymnasium MuJoCo Reacher-v5 calibration",
        "results/reacher_recovery_calibration_12seed_v1/summary.json",
    ),
    (
        "Gymnasium MuJoCo Swimmer-v5 calibration",
        "results/swimmer_recovery_calibration_12seed_v1_784458/summary.json",
    ),
    (
        "Gymnasium MuJoCo HalfCheetah-v5 calibration",
        "results/halfcheetah_recovery_calibration_12seed_v1_784535/summary.json",
    ),
    (
        "Gymnasium MuJoCo InvertedPendulum-v5 calibration",
        "results/inverted_pendulum_recovery_calibration_12seed_v1/summary.json",
    ),
    (
        "Gymnasium MuJoCo InvertedDoublePendulum-v5 calibration",
        "results/inverted_double_pendulum_recovery_calibration_12seed_v1/summary.json",
    ),
    (
        "Gymnasium Box2D LunarLander-v3 calibration",
        "results/lunarlander_recovery_calibration_12seed_v1/summary.json",
    ),
    (
        "Gymnasium Box2D LunarLanderContinuous-v3 calibration",
        "results/lunarlander_continuous_recovery_calibration_12seed_v1_784643/summary.json",
    ),
]

ROUTE_SCOUTS = [
    (
        "sklearn digits margin replay",
        "results/sklearn_digits_margin_scout_v0/summary.csv",
    ),
    (
        "sklearn tabular margin replay",
        "results/sklearn_tabular_margin_scout_v0/summary.csv",
    ),
    (
        "OpenML margin replay",
        "results/openml_margin_scout_v0/summary.csv",
    ),
    (
        "OpenML diabetes margin 30-seed",
        "results/openml_diabetes_margin_30seed_v1/summary.csv",
    ),
    (
        "OpenML diabetes margin replication 30-seed",
        "results/openml_diabetes_margin_replication_30seed_v1/summary.csv",
    ),
    (
        "OpenML numeric external fixed target2 30-seed",
        "results/openml_numeric_external_fixed_target2_30seed_v1/summary.csv",
    ),
    (
        "OpenML numeric external fixed target2 replication 30-seed",
        "results/openml_numeric_external_fixed_target2_replication_30seed_v1/summary.csv",
    ),
    (
        "OpenML broad numeric fixed target2 30-seed",
        "results/openml_broad_numeric_target2_30seed_v1/summary.csv",
    ),
    (
        "OpenML broad numeric fixed target2 replication 30-seed",
        "results/openml_broad_numeric_target2_replication_30seed_v1/summary.csv",
    ),
    (
        "OpenML heart-statlog fixed target2 extension 60-seed",
        "results/openml_heart_statlog_target2_extension_60seed_v1/summary.csv",
    ),
    (
        "OpenML multiclass numeric fixed target2 30-seed",
        "results/openml_multiclass_numeric_target2_30seed_v1/summary.csv",
    ),
    (
        "OpenML multiclass numeric fixed target2 replication 30-seed",
        "results/openml_multiclass_numeric_target2_replication_30seed_v1/summary.csv",
    ),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def grouped(rows: list[dict[str, str]]) -> dict[str, dict[int, dict[str, str]]]:
    result: dict[str, dict[int, dict[str, str]]] = {}
    for row in rows:
        result.setdefault(row["method"], {})[int(float(row["seed"]))] = row
    return result


def compare(
    rows_by_method: dict[str, dict[int, dict[str, str]]],
    *,
    treatment: str,
    baseline: str,
    metric: str,
) -> PairedComparison | None:
    if treatment not in rows_by_method or baseline not in rows_by_method:
        return None
    seeds = sorted(set(rows_by_method[treatment]) & set(rows_by_method[baseline]))
    if not seeds:
        return None
    treatment_values = [float(rows_by_method[treatment][seed][metric]) for seed in seeds]
    baseline_values = [float(rows_by_method[baseline][seed][metric]) for seed in seeds]
    diffs = [treatment_value - baseline_value for treatment_value, baseline_value in zip(treatment_values, baseline_values, strict=True)]
    return PairedComparison(
        treatment=treatment,
        baseline=baseline,
        treatment_mean=mean(treatment_values),
        baseline_mean=mean(baseline_values),
        delta=mean(diffs),
        wins=sum(diff > 0.0 for diff in diffs),
        losses=sum(diff < 0.0 for diff in diffs),
        ties=sum(diff == 0.0 for diff in diffs),
    )


def strongest_negative(comparisons: list[PairedComparison]) -> PairedComparison | None:
    if not comparisons:
        return None
    return min(comparisons, key=lambda comparison: comparison.delta)


def candidate_from_rows(
    *,
    name: str,
    path: str,
    treatment: str,
    rows: list[dict[str, str]],
    required_baselines: list[str],
    ablation: str | None,
    radius_metric: str,
) -> BenchmarkCandidate:
    rows_by_method = grouped(rows)
    seeds = len(rows_by_method.get(treatment, {}))
    vs_uniform = compare(rows_by_method, treatment=treatment, baseline="uniform", metric="final_rauc")
    required = [
        comparison
        for baseline in required_baselines
        if (comparison := compare(rows_by_method, treatment=treatment, baseline=baseline, metric="final_rauc")) is not None
    ]
    vs_ablation = compare(rows_by_method, treatment=treatment, baseline=ablation, metric="final_rauc") if ablation else None
    radius_vs_uniform = (
        compare(rows_by_method, treatment=treatment, baseline="uniform", metric=radius_metric)
        if rows and radius_metric in rows[0]
        else None
    )
    return BenchmarkCandidate(
        name=name,
        path=path,
        treatment=treatment,
        seeds=seeds,
        vs_uniform=vs_uniform,
        worst_required=strongest_negative(required),
        vs_ablation=vs_ablation,
        radius_vs_uniform=radius_vs_uniform,
        radius_metric=radius_metric,
    )


def benchmark_candidates(root: Path) -> list[BenchmarkCandidate]:
    candidates: list[BenchmarkCandidate] = []
    for name, rel_path, treatments, required_baselines, ablation, radius_metric in BENCHMARK_SCREENS:
        path = root / rel_path
        if not path.exists():
            continue
        rows = read_rows(path)
        for treatment in treatments:
            if any(row["method"] == treatment for row in rows):
                candidates.append(
                    candidate_from_rows(
                        name=name,
                        path=rel_path,
                        treatment=treatment,
                        rows=rows,
                        required_baselines=required_baselines,
                        ablation=ablation,
                        radius_metric=radius_metric,
                    )
                )
    return candidates


def calibration_screens(root: Path) -> list[CalibrationScreen]:
    screens: list[CalibrationScreen] = []
    for name, rel_path in CALIBRATION_SCREENS:
        path = root / rel_path
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        screens.append(
            CalibrationScreen(
                name=name,
                path=rel_path,
                clean_success=float(summary["clean_success"]),
                min_recovery=float(summary["min_recovery"]),
                max_recovery=float(summary["max_recovery"]),
                r80=float(summary["r80"]),
            )
        )
    return screens


def route_scouts(root: Path) -> list[RouteScout]:
    scouts: list[RouteScout] = []
    for name, rel_path in ROUTE_SCOUTS:
        path = root / rel_path
        if not path.exists():
            continue
        rows = read_rows(path)
        if rows and "dataset" in rows[0]:
            grouped_rows: dict[str, list[dict[str, str]]] = {}
            for row in rows:
                grouped_rows.setdefault(row["dataset"], []).append(row)
        else:
            grouped_rows = {"": rows}
        for dataset, scout_rows in grouped_rows.items():
            bgr_rows = [row for row in scout_rows if row["method"] == "bgr"]
            uniform_rows = {
                float(row["target_radius"]): float(row["final_rauc_mean"])
                for row in scout_rows
                if row["method"] == "uniform"
            }
            fixed_rows = [row for row in scout_rows if row["method"] == "fixed"]
            if not bgr_rows or not uniform_rows or not fixed_rows:
                continue
            best_bgr = max(bgr_rows, key=lambda row: float(row["final_rauc_mean"]))
            best_fixed = max(fixed_rows, key=lambda row: float(row["final_rauc_mean"]))
            target = float(best_bgr["target_radius"])
            treatment_mean = float(best_bgr["final_rauc_mean"])
            uniform_mean = uniform_rows[target]
            scout_name = f"{name} ({dataset})" if dataset else name
            scouts.append(
                RouteScout(
                    name=scout_name,
                    path=rel_path,
                    target_radius=target,
                    treatment_mean=treatment_mean,
                    uniform_mean=uniform_mean,
                    delta=treatment_mean - uniform_mean,
                    wins=int(best_bgr["wins_vs_uniform"]),
                    losses=int(best_bgr["losses_vs_uniform"]),
                    ties=int(best_bgr["ties_vs_uniform"]),
                    seeds=int(best_bgr["n"]),
                    best_fixed_mean=float(best_fixed["final_rauc_mean"]),
                    best_fixed_target=float(best_fixed["target_radius"]),
                )
            )
    return scouts


def grid_summary(root: Path) -> str:
    rows = read_rows(root / "results/grid_margin_full_30seed_v1/summary.csv") + read_rows(
        root / "results/grid_margin_full_replication_30seed_v1/summary.csv"
    )
    rows_by_method = grouped(rows)
    comparison = compare(rows_by_method, treatment="bgr", baseline="uniform", metric="final_rauc")
    if comparison is None:
        raise RuntimeError("missing grid BGR/uniform comparison")
    clearance = comparison.delta - 0.03
    return (
        f"Grid-margin mechanism clears the internal mechanism threshold: "
        f"BGR {comparison.treatment_mean:.4f} vs uniform {comparison.baseline_mean:.4f}, "
        f"delta {comparison.delta:+.4f}; margin above +0.03 threshold {clearance:+.4f}."
    )


def perturbation_total(rows: list[dict[str, str]], method: str, perturbations: set[str]) -> tuple[int, int]:
    successes = 0
    episodes = 0
    for row in rows:
        if row.get("method") == method and row.get("perturbation") in perturbations:
            successes += int(float(row["successes"]))
            episodes += int(float(row["episodes"]))
    return successes, episodes


def identity_total(rows: list[dict[str, str]], method: str) -> tuple[int, int]:
    successes = 0
    episodes = 0
    for row in rows:
        if row.get("method") == method and row.get("perturbation") == "identity":
            successes += int(float(row["successes"]))
            episodes += int(float(row["episodes"]))
    return successes, episodes


def missing_openvla_rows(rows: list[dict[str, str]], perturbations: set[str] | None = None) -> list[str]:
    methods = ("bgr", "official", "random")
    required_perturbations = tuple(sorted((perturbations or OPENVLA_NON_IDENTITY_PERTURBATIONS) | {"identity"}))
    present = {(row.get("method"), row.get("perturbation")) for row in rows}
    return [
        f"{method}/{perturbation}"
        for method in methods
        for perturbation in required_perturbations
        if (method, perturbation) not in present
    ]


def openvla_gate_summary(
    label: str,
    rows: list[dict[str, str]],
    *,
    non_identity_perturbations: set[str] | None = None,
) -> str:
    required_non_identity = non_identity_perturbations or OPENVLA_NON_IDENTITY_PERTURBATIONS
    missing = missing_openvla_rows(rows, required_non_identity)
    if missing:
        return f"{label} OpenVLA audit summary is incomplete; missing {', '.join(missing)}."
    bgr_success, bgr_episodes = perturbation_total(rows, "bgr", required_non_identity)
    official_success, official_episodes = perturbation_total(rows, "official", required_non_identity)
    random_success, random_episodes = perturbation_total(rows, "random", required_non_identity)
    bgr_identity, identity_episodes = identity_total(rows, "bgr")
    official_identity, _ = identity_total(rows, "official")
    random_identity, _ = identity_total(rows, "random")
    official_gap = bgr_success - official_success
    random_gap = bgr_success - random_success
    official_rate_gap = bgr_success / bgr_episodes - official_success / official_episodes
    random_rate_gap = bgr_success / bgr_episodes - random_success / random_episodes
    clean_deficit = max(official_identity, random_identity) - bgr_identity
    passed = (
        bgr_episodes == 400
        and official_episodes == 400
        and random_episodes == 400
        and official_gap >= 10
        and random_gap >= 10
        and official_rate_gap >= 0.02
        and random_rate_gap >= 0.02
        and clean_deficit <= 1
    )
    status = "clears" if passed else "does not clear"
    return (
        f"{label} OpenVLA audit {status} the learned-policy promotion gate: "
        f"BGR {bgr_success}/{bgr_episodes}, official {official_success}/{official_episodes}, "
        f"random {random_success}/{random_episodes}; identity BGR {bgr_identity}/{identity_episodes}, "
        f"official {official_identity}/{identity_episodes}, random {random_identity}/{identity_episodes}; "
        f"official gap {official_gap:+d} ({official_rate_gap:+.4f}), "
        f"random gap {random_gap:+d} ({random_rate_gap:+.4f}), clean deficit {clean_deficit}."
    )


def learned_policy_summary(root: Path) -> str:
    hard_occ_summaries = [
        (
            OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_COMPLETE,
            "Hard-occlusion 0.80 micro identity-anchored adaptation",
        ),
        (
            OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_A40_COMPLETE,
            "Hard-occlusion 0.80 micro identity-anchored A40 adaptation",
        ),
        (
            OPENVLA_HARD_OCCLUSION090_IDENTITY_ANCHOR_STRICT_COMPLETE,
            "Hard-occlusion 0.90 strict identity-anchored adaptation",
        ),
        (
            OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_COMPLETE,
            "Hard-occlusion 0.80 identity-anchored adaptation",
        ),
        (
            OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_A40_COMPLETE,
            "Hard-occlusion 0.80 identity-anchored A40 adaptation",
        ),
        (
            OPENVLA_HARD_OCCLUSION080_TRANSFER_COMPLETE,
            "Hard-occlusion 0.80 transfer",
        ),
        (
            OPENVLA_HARD_OCCLUSION_TRANSFER_COMPLETE,
            "Hard-occlusion 0.65 transfer",
        ),
        (
            OPENVLA_HARD_OCCLUSION_ADAPT_A40_COMPLETE,
            "Hard-occlusion 0.65 A40 adaptation",
        ),
        (
            OPENVLA_HARD_OCCLUSION_ADAPT_COMPLETE,
            "Hard-occlusion 0.65 adaptation",
        ),
    ]
    hard_occ_readouts = [
        openvla_gate_summary(label, read_rows(root / relative_path), non_identity_perturbations={"occlusion"})
        for relative_path, label in hard_occ_summaries
        if (root / relative_path).exists()
    ]
    if hard_occ_readouts:
        return hard_occ_readouts[0]

    occlusion_bottleneck = root / OPENVLA_OCCLUSION_BOTTLENECK_COMPLETE
    if occlusion_bottleneck.exists():
        return openvla_gate_summary("Occlusion-bottleneck", read_rows(occlusion_bottleneck))

    perturb_only = root / OPENVLA_PERTURB_ONLY_ANCHOR_COMPLETE
    if perturb_only.exists():
        return openvla_gate_summary("Perturb-only anchored", read_rows(perturb_only))

    proximal = root / OPENVLA_PROXIMAL_ANCHOR_COMPLETE
    if proximal.exists():
        return openvla_gate_summary("Proximal-anchor", read_rows(proximal))

    weighted = root / OPENVLA_WEIGHTED_COMPLETE
    if weighted.exists():
        rows = read_rows(weighted)
        bgr_success, bgr_episodes = perturbation_total(rows, "bgr", OPENVLA_NON_IDENTITY_PERTURBATIONS)
        official_success, official_episodes = perturbation_total(rows, "official", OPENVLA_NON_IDENTITY_PERTURBATIONS)
        random_success, random_episodes = perturbation_total(rows, "random", OPENVLA_NON_IDENTITY_PERTURBATIONS)
        bgr_identity, identity_episodes = identity_total(rows, "bgr")
        official_gap = bgr_success - official_success
        random_gap = bgr_success - random_success
        best_gap = bgr_success - max(official_success, random_success)
        needed = max(0, 10 - best_gap)
        random_detail = f"random {random_success}/{random_episodes}"
        status = "non-promotable"
        if bgr_episodes == 400 and official_episodes == 400 and official_gap < 10:
            status = "complete and fails the learned-policy gate"
        return (
            f"Weighted OpenVLA audit is {status}: "
            f"BGR {bgr_success}/{bgr_episodes}, official {official_success}/{official_episodes}, "
            f"{random_detail}; identity BGR {bgr_identity}/{identity_episodes}. "
            f"Episode margins are official {official_gap:+d} and random {random_gap:+d}, "
            f"short of the best-comparator +10 gate by {needed} episodes."
        )

    legacy = root / OPENVLA_LEGACY_COMPLETE
    rows = read_rows(legacy)
    bgr_success, bgr_episodes = perturbation_total(rows, "bgr", OPENVLA_NON_IDENTITY_PERTURBATIONS)
    official_success, _ = perturbation_total(rows, "official", OPENVLA_NON_IDENTITY_PERTURBATIONS)
    random_success, _ = perturbation_total(rows, "random", OPENVLA_NON_IDENTITY_PERTURBATIONS)
    best = max(official_success, random_success)
    margin = bgr_success - best
    return (
        f"Latest complete OpenVLA audit is negative: BGR {bgr_success}/{bgr_episodes}, "
        f"official {official_success}/{bgr_episodes}, random {random_success}/{bgr_episodes}; "
        f"margin versus best comparator {margin:+d}, short of +10 by {max(0, 10 - margin)} episodes."
    )


def learned_policy_inflight_summary(root: Path) -> str | None:
    """Report preregistered learned-policy runs that are failed or unsummarized."""
    ledger_paths = [root / "AGENTS.md", root / "results/README.md", root / "docs/aaai_acceptance_gap.md"]
    ledger_text = "\n".join(path.read_text(encoding="utf-8") for path in ledger_paths if path.exists())
    active: list[str] = []
    no_active_cluster_jobs = "No active learned-policy cluster jobs remain queued" in ledger_text

    def append_openvla_route(
        marker: str,
        complete_path: str,
        label: str,
        pending_detail: str,
        closed_marker: str | None = None,
        closed_detail: str | None = None,
    ) -> None:
        if marker not in ledger_text or (root / complete_path).exists():
            return
        partial_failure = partial_openvla_failure_detail(
            root,
            complete_path,
            label=label,
            non_identity_perturbations={"occlusion"},
        )
        if partial_failure is not None:
            active.append(partial_failure)
        elif closed_marker is not None and closed_marker in ledger_text:
            if closed_detail is not None:
                active.append(closed_detail)
        elif not no_active_cluster_jobs:
            active.append(pending_detail)

    append_openvla_route(
        OPENVLA_HARD_OCCLUSION080_TRANSFER_HEADINTERP000_LORAFULL_NOVIDEO_MARKER,
        OPENVLA_HARD_OCCLUSION080_TRANSFER_HEADINTERP000_LORAFULL_NOVIDEO_COMPLETE,
        "hard-occlusion 0.80 alpha0 official-head/full-LoRA no-video OpenVLA repair",
        "hard-occlusion 0.80 alpha0 official-head/full-LoRA no-video repair is queued/running and missing a complete summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_MARKER,
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_COMPLETE,
        "hard-occlusion 0.80 micro identity-anchored OpenVLA adaptation",
        "hard-occlusion 0.80 micro identity-anchored adaptation is queued/running and missing a complete summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_A40_MARKER,
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_A40_COMPLETE,
        "hard-occlusion 0.80 micro identity-anchored A40 OpenVLA adaptation",
        "hard-occlusion 0.80 micro identity-anchored A40 adaptation is queued/running and missing a complete summary",
        closed_marker="micro A40 route jobs",
        closed_detail=(
            "hard-occlusion 0.80 micro identity-anchored A40 adaptation "
            "closed incomplete after failed/cancelled Slurm jobs; only official identity "
            "393/400 is summarized, so there is no gateable BGR/random comparison"
        ),
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MARKER,
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_COMPLETE,
        "hard-occlusion 0.80 identity-anchored OpenVLA adaptation",
        "hard-occlusion 0.80 identity-anchored adaptation is queued/running and missing a complete summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_A40_MARKER,
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_A40_COMPLETE,
        "hard-occlusion 0.80 identity-anchored A40 OpenVLA adaptation",
        "hard-occlusion 0.80 identity-anchored A40 adaptation is queued/running and missing a complete summary",
        closed_marker="identity-anchor A40 route jobs",
        closed_detail=(
            "hard-occlusion 0.80 identity-anchored A40 adaptation "
            "closed incomplete after failed/cancelled Slurm jobs; only official identity "
            "393/400 and partial official occlusion 241/342 are summarized"
        ),
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_STRICT_MARKER,
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_STRICT_COMPLETE,
        "hard-occlusion 0.80 strict identity-anchored OpenVLA adaptation",
        "hard-occlusion 0.80 strict identity-anchored adaptation is queued/running and missing a complete summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION090_IDENTITY_ANCHOR_STRICT_MARKER,
        OPENVLA_HARD_OCCLUSION090_IDENTITY_ANCHOR_STRICT_COMPLETE,
        "hard-occlusion 0.90 strict identity-anchored OpenVLA adaptation",
        "hard-occlusion 0.90 strict identity-anchored adaptation is queued/running and missing a complete summary",
        closed_marker="0.90 strict A40 route jobs",
        closed_detail=(
            "hard-occlusion 0.90 strict identity-anchored adaptation "
            "closed incomplete after failed/cancelled Slurm jobs; only official identity "
            "393/400 is summarized, so there is no gateable BGR/random comparison"
        ),
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION080_TRANSFER_MARKER,
        OPENVLA_HARD_OCCLUSION080_TRANSFER_COMPLETE,
        "hard-occlusion 0.80 transfer OpenVLA route",
        "hard-occlusion 0.80 transfer eval is queued/running and missing a complete summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION_TRANSFER_MARKER,
        OPENVLA_HARD_OCCLUSION_TRANSFER_COMPLETE,
        "hard-occlusion transfer OpenVLA route",
        "hard-occlusion transfer eval is queued/running and missing a complete summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION_ADAPT_MARKER,
        OPENVLA_HARD_OCCLUSION_ADAPT_COMPLETE,
        "hard-occlusion adaptation OpenVLA route",
        "hard-occlusion adaptation is queued and missing logs/summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION_ADAPT_A40_MARKER,
        OPENVLA_HARD_OCCLUSION_ADAPT_A40_COMPLETE,
        "hard-occlusion A40 fallback OpenVLA adaptation",
        "hard-occlusion A40 fallback adaptation is queued/running and missing a complete summary",
    )
    if active:
        return "; ".join(active)

    occlusion_bottleneck_summary = root / OPENVLA_OCCLUSION_BOTTLENECK_COMPLETE
    if not occlusion_bottleneck_summary.exists():
        if OPENVLA_OCCLUSION_BOTTLENECK_MARKER in ledger_text:
            return (
                "Occlusion-bottleneck OpenVLA route is preregistered, not yet evidence: "
                "the clean-plus-occlusion TFDS prep, proximal-anchor BGR/random adaptation, "
                "and official/BGR/random 10-task x 10-trial perturbation summaries must finish "
                "before the +10/400 and +0.02 learned-policy gate can be checked."
            )

    perturb_only_summary = root / OPENVLA_PERTURB_ONLY_ANCHOR_COMPLETE
    if not perturb_only_summary.exists():
        ledger_paths = [root / "AGENTS.md", root / "results/README.md", root / "docs/aaai_acceptance_gap.md"]
        ledger_text = "\n".join(path.read_text(encoding="utf-8") for path in ledger_paths if path.exists())
        if OPENVLA_PERTURB_ONLY_ANCHOR_MARKER in ledger_text:
            return (
                "Perturb-only anchored OpenVLA route is preregistered, not yet evidence: "
                "the fixed perturb-only TFDS prep, proximal-anchor BGR/random adaptation, "
                "and official/BGR/random 10-task x 10-trial perturbation summaries must finish "
                "before the +10/400 and +0.02 learned-policy gate can be checked."
            )

    proximal_job_ids = {
        "bgr_train": "767657",
        "bgr_merge": "767658",
        "bgr_clean_eval": "767659",
        "random_train": "767660",
        "random_merge": "767661",
        "random_clean_eval": "767662",
        "official_first": "767663",
        "official_last": "767667",
        "bgr_perturb_first": "767674",
        "bgr_perturb_last": "767678",
        "random_perturb_first": "767681",
        "random_perturb_last": "767685",
    }
    proximal_summary = (
        root
        / "results/openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_proxanchor_l2_1em0_ddpgradfix_v1_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv"
    )
    if proximal_summary.exists():
        return None

    ledger_paths = [root / "AGENTS.md", root / "results/README.md", root / "docs/aaai_acceptance_gap.md"]
    ledger_text = "\n".join(path.read_text(encoding="utf-8") for path in ledger_paths if path.exists())
    if (
        proximal_job_ids["bgr_train"] not in ledger_text
        or proximal_job_ids["random_perturb_last"] not in ledger_text
    ):
        return None

    bgr_train = proximal_job_ids["bgr_train"]
    if f"{bgr_train}` failed" in ledger_text or f"{bgr_train} failed" in ledger_text:
        return (
            "Proximal-anchor OpenVLA route failed before producing evidence: "
            f"BGR train job {bgr_train} exited 1:0 with a PyTorch DDP ready-twice error, "
            "leaving downstream BGR/random jobs dependency-held; repair or retire this route "
            "before applying the +10/400 and +0.02 learned-policy gate."
        )

    return (
        "Proximal-anchor OpenVLA route is unsummarized, not yet evidence: "
        f"adaptation jobs BGR {proximal_job_ids['bgr_train']}/{proximal_job_ids['bgr_merge']}/"
        f"{proximal_job_ids['bgr_clean_eval']} and random {proximal_job_ids['random_train']}/"
        f"{proximal_job_ids['random_merge']}/{proximal_job_ids['random_clean_eval']} "
        f"do not have complete fixed summaries; fixed perturbation jobs official "
        f"{proximal_job_ids['official_first']}--{proximal_job_ids['official_last']}, "
        f"BGR {proximal_job_ids['bgr_perturb_first']}--{proximal_job_ids['bgr_perturb_last']}, "
        f"and random {proximal_job_ids['random_perturb_first']}--{proximal_job_ids['random_perturb_last']} must finish before the +10/400 "
        "and +0.02 learned-policy gate can be checked."
    )


def fmt_comparison(comparison: PairedComparison | None) -> str:
    if comparison is None:
        return "n/a"
    return f"{comparison.delta:+.4f} ({comparison.wins}/{comparison.losses}/{comparison.ties})"


def fmt_reasons(candidate: BenchmarkCandidate) -> str:
    if candidate.promotable_screen:
        return "none"
    return ", ".join(candidate.failure_reasons)


def render_markdown(root: Path) -> str:
    candidates = sorted(benchmark_candidates(root), key=lambda candidate: candidate.priority_key, reverse=True)
    calibrations = calibration_screens(root)
    scouts = route_scouts(root)
    best = candidates[0] if candidates else None
    learned_summary = learned_policy_summary(root)
    lines = [
        "# Acceptance Scorecard",
        "",
        "This scorecard is generated from local result artifacts. It is not an acceptance probability; it quantifies distance to the internal promotion gates.",
        "",
        "## Gate Summary",
        "",
        f"- {grid_summary(root)}",
        f"- {learned_summary}",
    ]
    inflight = learned_policy_inflight_summary(root)
    if inflight is not None:
        lines.append(f"- {inflight}")
    if best is not None:
        lines.append(
            f"- Closest independent benchmark screen is `{best.name}` with treatment `{best.treatment}`: "
            f"delta vs uniform {fmt_comparison(best.vs_uniform)}, worst required-baseline delta {fmt_comparison(best.worst_required)}, "
            f"ablation delta {fmt_comparison(best.vs_ablation)}, radius delta {fmt_comparison(best.radius_vs_uniform)}, "
            f"failure reason(s): {fmt_reasons(best)}."
        )
    rejected_calibrations = [screen for screen in calibrations if not screen.usable]
    usable_calibrations = [screen for screen in calibrations if screen.usable]
    retired_calibrations = [
        screen
        for screen in usable_calibrations
        if (screen.name in COMPLETED_METHOD_SCREEN_BY_CALIBRATION)
        and (root / COMPLETED_METHOD_SCREEN_BY_CALIBRATION[screen.name]).exists()
    ]
    retired_calibration_names = {screen.name for screen in retired_calibrations}
    active_calibrations = [
        screen
        for screen in usable_calibrations
        if screen.name not in retired_calibration_names
    ]
    if rejected_calibrations:
        names = ", ".join(f"`{screen.name}`" for screen in rejected_calibrations)
        lines.append(f"- Rejected pre-method calibration route(s): {names}.")
    if retired_calibrations:
        names = ", ".join(
            f"`{screen.name}` clean {screen.clean_success:.4f}, range {screen.min_recovery:.4f}--{screen.max_recovery:.4f}, r80 {screen.r80:.4f}"
            for screen in retired_calibrations
        )
        lines.append(f"- Retired calibrated route(s) that cleared pre-method calibration: {names}.")
    positive_followups = [scout for scout in scouts if scout.positive_followup]
    positive_followup_keys = {route_scout_evidence_key(scout) for scout in positive_followups}
    rejected_scouts = [scout for scout in scouts if scout.rejected]
    rejected_followup_keys = {
        route_scout_evidence_key(scout)
        for scout in rejected_scouts
        if scout.seeds >= 30
    }
    superseded_scouts = [
        scout
        for scout in scouts
        if scout.needs_preregistration and route_scout_evidence_key(scout) in positive_followup_keys
    ]
    candidate_scouts = [
        scout
        for scout in scouts
        if scout.needs_preregistration
        and route_scout_evidence_key(scout) not in positive_followup_keys
        and route_scout_evidence_key(scout) not in rejected_followup_keys
    ]
    if rejected_scouts:
        names = ", ".join(
            f"`{scout.name}` best BGR {scout.treatment_mean:.4f} vs uniform {scout.uniform_mean:.4f} "
            f"(W/L/T={scout.wins}/{scout.losses}/{scout.ties})"
            for scout in rejected_scouts
        )
        lines.append(f"- Rejected route scout(s): {names}.")
    if candidate_scouts:
        names = ", ".join(
            f"`{scout.name}` best BGR {scout.treatment_mean:.4f} vs uniform {scout.uniform_mean:.4f} "
            f"(W/L/T={scout.wins}/{scout.losses}/{scout.ties})"
            for scout in candidate_scouts
        )
        lines.append(f"- Route scout(s) requiring preregistration before promotion: {names}.")
    if superseded_scouts:
        names = ", ".join(f"`{scout.name}`" for scout in superseded_scouts)
        lines.append(
            f"- Superseded route scout(s): {names} already has fixed positive follow-up evidence; no preregistration action remains."
        )
    if positive_followups:
        names = ", ".join(
            f"`{scout.name}` BGR {scout.treatment_mean:.4f} vs uniform {scout.uniform_mean:.4f} "
            f"(W/L/T={scout.wins}/{scout.losses}/{scout.ties}), fixed gap {scout.fixed_delta:+.4f}"
            for scout in positive_followups
        )
        lines.append(f"- Positive pre-existing-dataset follow-up(s): {names}.")
    if active_calibrations:
        names = ", ".join(
            f"`{screen.name}` clean {screen.clean_success:.4f}, range {screen.min_recovery:.4f}--{screen.max_recovery:.4f}, r80 {screen.r80:.4f}"
            for screen in active_calibrations
        )
        lines.append(f"- Active pre-method calibration route(s): {names}.")
    lines.extend(
        [
            "",
            "## Promotion Deficits",
            "",
        ]
    )
    if best is None:
        lines.append("- Independent benchmark: no completed screen artifacts were found.")
    else:
        lines.append(
            f"- Independent benchmark: no screen clears the 4/4 promotion screen. "
            f"The closest screen, `{best.name}` with `{best.treatment}`, clears "
            f"{best.cleared_gate_count}/4 gates and fails on {fmt_reasons(best)}."
        )
    if learned_summary.startswith("Weighted OpenVLA audit"):
        lines.append(
            "- Learned policy: the latest weighted OpenVLA audit is short of the "
            "best-comparator promotion gate by 13/400 non-identity episodes "
            "and 0.0275 absolute success."
        )
    elif "does not clear" in learned_summary or "negative" in learned_summary:
        lines.append(f"- Learned policy: {learned_summary}")
    elif "incomplete" in learned_summary:
        lines.append(f"- Learned policy: {learned_summary}")
    else:
        lines.append(
            "- Learned policy: the latest summarized OpenVLA audit clears the "
            "internal learned-policy gate; paper incorporation still requires "
            "claim and package checks."
        )
    if retired_calibrations:
        names = ", ".join(f"`{screen.name}`" for screen in retired_calibrations)
        lines.append(
            f"- Retired calibrated route(s): {names} cleared pre-method calibration, but the corresponding fixed method screen is negative or tied; not active acceptance evidence."
        )
    if rejected_scouts:
        names = ", ".join(f"`{scout.name}`" for scout in rejected_scouts)
        lines.append(
            f"- Rejected route scout(s): {names} did not clear the +0.03 and 3/4 paired pre-registration screen; not active acceptance evidence."
        )
    if candidate_scouts:
        names = ", ".join(f"`{scout.name}`" for scout in candidate_scouts)
        lines.append(
            f"- Route scout(s): {names} need a fixed preregistered comparison before any manuscript claim."
        )
    if superseded_scouts:
        names = ", ".join(f"`{scout.name}`" for scout in superseded_scouts)
        lines.append(
            f"- Superseded route scout(s): {names} have fixed positive follow-up evidence; the scout itself is not an active action item."
        )
    if positive_followups:
        names = ", ".join(f"`{scout.name}`" for scout in positive_followups)
        lines.append(
            f"- Positive pre-existing-dataset follow-up(s): {names} clear the internal 30-seed margin-replay follow-up gate and are available only as supervised margin-replay evidence."
        )
    if inflight is None and active_calibrations:
        names = ", ".join(f"`{screen.name}`" for screen in active_calibrations)
        lines.append(
            f"- Active route: {names} cleared pre-method calibration; run only the fixed preregistered all-method screen before interpreting it."
        )
    elif inflight is None:
        lines.append(
            "- Active route: no queued learned-policy route is recorded in the local ledgers."
        )
    else:
        lines.append(
            f"- Learned-policy route status: {inflight}"
        )
    lines.extend(
        [
            "",
            "## Independent Benchmark Screens",
            "",
            "| Screen | Treatment | Seeds | dRAUC vs uniform (W/L/T) | Worst required baseline dRAUC | Ablation dRAUC | Radius metric | Radius delta | Cleared gates | Screen gate | Failure reason(s) |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for candidate in candidates:
        decision = "pass-4seed-screen" if candidate.promotable_screen else "fail"
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate.name,
                    candidate.treatment,
                    str(candidate.seeds),
                    fmt_comparison(candidate.vs_uniform),
                    fmt_comparison(candidate.worst_required),
                    fmt_comparison(candidate.vs_ablation),
                    candidate.radius_metric if candidate.radius_vs_uniform is not None else "n/a",
                    fmt_comparison(candidate.radius_vs_uniform),
                    f"{candidate.cleared_gate_count}/4",
                    decision,
                    fmt_reasons(candidate),
                ]
            )
            + " |"
        )
    if scouts:
        lines.extend(
            [
                "",
                "## Route Scouts",
                "",
                "| Scout | Best BGR target | BGR RAUC | Uniform RAUC | dRAUC vs uniform (W/L/T) | Best fixed-radius RAUC | Decision |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for scout in scouts:
            scout_decision = scout.decision
            if scout in superseded_scouts:
                scout_decision = "superseded-by-positive-follow-up"
            elif scout.needs_preregistration and route_scout_evidence_key(scout) in rejected_followup_keys:
                scout_decision = "closed-by-rejected-follow-up"
            lines.append(
                "| "
                + " | ".join(
                    [
                        scout.name,
                        f"{scout.target_radius:.4f}",
                        f"{scout.treatment_mean:.4f}",
                        f"{scout.uniform_mean:.4f}",
                        f"{scout.delta:+.4f} ({scout.wins}/{scout.losses}/{scout.ties})",
                        f"{scout.best_fixed_mean:.4f} @ {scout.best_fixed_target:.4f}",
                        scout_decision,
                    ]
                )
                + " |"
            )
    if calibrations:
        lines.extend(
            [
                "",
                "## Pre-Method Calibrations",
                "",
                "| Calibration | Clean | Recovery range | Median r80 | Decision |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for screen in calibrations:
            lines.append(
                "| "
                + " | ".join(
                    [
                        screen.name,
                        f"{screen.clean_success:.4f}",
                        f"{screen.min_recovery:.4f}--{screen.max_recovery:.4f}",
                        f"{screen.r80:.4f}",
                        screen.decision,
                    ]
                )
                + " |"
            )
    if learned_summary.startswith("Proximal-anchor OpenVLA audit"):
        if "does not clear" in learned_summary:
            detail = learned_summary.removeprefix(
                "Proximal-anchor OpenVLA audit does not clear the learned-policy promotion gate: "
            )
            learned_priority = (
                "- The latest proximal-anchor OpenVLA audit is complete and fails the learned-policy gate: "
                f"{detail}"
            )
        elif "clears" in learned_summary:
            learned_priority = (
                "- The latest proximal-anchor OpenVLA audit clears the learned-policy gate; "
                "verify paper claim wording and package checks before promoting it."
            )
        else:
            learned_priority = f"- {learned_summary}"
    elif learned_summary.startswith("Weighted OpenVLA audit"):
        learned_priority = (
            "- The latest OpenVLA weighted audit is complete and fails the learned-policy gate: "
            "BGR ties the official checkpoint at 367/400 non-identity successes and trails matched random by 3/400."
        )
    else:
        learned_priority = f"- {learned_summary}"

    priority_lines = [
        "- The controlled grid mechanism is above its internal effect threshold, but it is still a constructed mechanism benchmark.",
        "- The standard-environment recovery route still has not produced a promotable screen: early promising screens either fail paired/radius checks or do not survive scale-up, and later non-saturated screens trail uniform, stronger baselines, or the state-priority/uniform-radius ablation.",
        learned_priority,
    ]
    if rejected_scouts:
        priority_lines.insert(
            2,
            "- Most pre-existing-dataset route scouts are rejected before preregistration: their best BGR rows stay below the +0.03 screen even when paired signs are favorable, or fixed-radius replay is competitive.",
        )
    if candidate_scouts:
        names = ", ".join(f"`{scout.name}`" for scout in candidate_scouts)
        priority_lines.insert(
            2,
            f"- {names} cleared an exploratory route screen only; it needs the matching fixed preregistered or held-out comparison before any manuscript claim.",
        )
    if superseded_scouts:
        names = ", ".join(f"`{scout.name}`" for scout in superseded_scouts)
        priority_lines.insert(
            2,
            f"- {names} is superseded by fixed positive OpenML follow-ups, so it is no longer a pending preregistration item.",
        )
    if positive_followups:
        names = ", ".join(f"`{scout.name}`" for scout in positive_followups)
        priority_lines.insert(
            2,
            f"- {names} now give a replicated positive pre-existing-dataset signal, but it must be framed as supervised margin-replay evidence rather than robotics or standard-environment recovery.",
        )
    if any(screen.name == "Gymnasium MuJoCo Reacher-v5 calibration" for screen in usable_calibrations):
        priority_lines.insert(
            2,
            "- The usable Reacher-v5 calibration is pre-method evidence only; the fixed full all-method comparison is now negative and should not be promoted.",
        )
    if any(screen.name == "Gymnasium MuJoCo InvertedPendulum-v5 calibration" for screen in retired_calibrations):
        priority_lines.insert(
            3,
            "- The InvertedPendulum-v5 calibration also cleared pre-method checks, but its fixed 4-seed method screen ties all methods on final RAUC and median-r80; do not scale or promote it.",
        )
    if any(screen.name == "Gymnasium MuJoCo InvertedDoublePendulum-v5 calibration" for screen in retired_calibrations):
        priority_lines.insert(
            4,
            "- The InvertedDoublePendulum-v5 calibration cleared pre-method checks, but its fixed 4-seed method screen collapses clean success; the small BGR RAUC edge is not acceptance evidence.",
        )
    if active_calibrations:
        names = ", ".join(f"`{screen.name}`" for screen in active_calibrations)
        priority_lines.insert(
            3,
            f"- Active pre-method calibration route(s) awaiting fixed comparison result: {names}.",
        )
    if any(screen.name == "Gymnasium MuJoCo HalfCheetah-v5 calibration" for screen in active_calibrations):
        priority_lines.insert(
            4,
            "- The active HalfCheetah-v5 calibration has a usable reset interface, but the first local linear/phase imitation scaffold had zero clean recovery and was not queued; a better fixed learner/controller must pass a local clean-recovery smoke before the all-method screen.",
        )
    priority_lines.insert(
        2 if not usable_calibrations else 4,
        "- Rejected pre-method calibrations should not be scaled into BGR comparisons until the reset interface and controller first produce clean, non-saturated recovery curves.",
    )
    if inflight is None and active_calibrations:
        if any(screen.name == "Gymnasium MuJoCo HalfCheetah-v5 calibration" for screen in active_calibrations):
            priority_lines.append(
                "- The next acceptance-moving HalfCheetah work is a preregistered clean-viable learner/controller for the calibrated reset interface, followed by the fixed all-method screen if that viability smoke passes."
            )
        else:
            priority_lines.append(
                "- The next acceptance-moving work is the fixed all-method screen for the active pre-method calibration route; do not tune the protocol after seeing its method results."
            )
    elif inflight is None and usable_calibrations:
        priority_lines.append(
            "- The next acceptance-moving work must find a genuinely different independent route, change the learned-policy intervention, or strengthen theory/presentation; retired calibrated routes are scope evidence, not acceptance evidence."
        )
    elif inflight is None:
        priority_lines.append(
            "- The next acceptance-moving work should change the learned-policy intervention or materially strengthen theory/presentation; another same-protocol MiniGrid/classic-control screen is unlikely to move the gate."
        )
    else:
        priority_lines.append(
            f"- Learned-policy route status: {inflight}"
        )
        priority_lines.append(
            "- Do not start another same-protocol MiniGrid, classic-control, PointMaze, or FetchReach screen while this is pending; existing screens already show saturated radius checks, stronger-baseline losses, or state-priority-only ablation failures."
        )

    lines.extend(
        [
            "",
            "## Priority Read",
            "",
            *priority_lines,
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    text = render_markdown(args.root)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
