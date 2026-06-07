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
from scripts.check_acceptance_readiness import OPENVLA_PROXIMAL_ANCHOR_COMPLETE
from scripts.check_acceptance_readiness import OPENVLA_WEIGHTED_COMPLETE


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
        return self.clean_success >= 0.75 and self.recovery_range >= 0.05

    @property
    def decision(self) -> str:
        return "usable-calibration" if self.usable else "reject-calibration"


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
]

CALIBRATION_SCREENS = [
    (
        "FetchPush-v4 object-goal calibration",
        "results/fetchpush_object_goal_calibration_2seed_v1/summary.json",
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
        "highway-env parking-v0 calibration",
        "results/highway_parking_recovery_calibration_12seed_v1/summary.json",
    ),
    (
        "Gymnasium MuJoCo Reacher-v5 calibration",
        "results/reacher_recovery_calibration_12seed_v1/summary.json",
    ),
    (
        "Gymnasium MuJoCo InvertedPendulum-v5 calibration",
        "results/inverted_pendulum_recovery_calibration_12seed_v1/summary.json",
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


def missing_openvla_rows(rows: list[dict[str, str]]) -> list[str]:
    methods = ("bgr", "official", "random")
    perturbations = tuple(sorted(OPENVLA_NON_IDENTITY_PERTURBATIONS | {"identity"}))
    present = {(row.get("method"), row.get("perturbation")) for row in rows}
    return [
        f"{method}/{perturbation}"
        for method in methods
        for perturbation in perturbations
        if (method, perturbation) not in present
    ]


def learned_policy_summary(root: Path) -> str:
    proximal = root / OPENVLA_PROXIMAL_ANCHOR_COMPLETE
    if proximal.exists():
        rows = read_rows(proximal)
        missing = missing_openvla_rows(rows)
        if missing:
            return f"Proximal-anchor OpenVLA audit summary is incomplete; missing {', '.join(missing)}."
        bgr_success, bgr_episodes = perturbation_total(rows, "bgr", OPENVLA_NON_IDENTITY_PERTURBATIONS)
        official_success, official_episodes = perturbation_total(rows, "official", OPENVLA_NON_IDENTITY_PERTURBATIONS)
        random_success, random_episodes = perturbation_total(rows, "random", OPENVLA_NON_IDENTITY_PERTURBATIONS)
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
            f"Proximal-anchor OpenVLA audit {status} the learned-policy promotion gate: "
            f"BGR {bgr_success}/{bgr_episodes}, official {official_success}/{official_episodes}, "
            f"random {random_success}/{random_episodes}; identity BGR {bgr_identity}/{identity_episodes}, "
            f"official {official_identity}/{identity_episodes}, random {random_identity}/{identity_episodes}; "
            f"official gap {official_gap:+d} ({official_rate_gap:+.4f}), "
            f"random gap {random_gap:+d} ({random_rate_gap:+.4f}), clean deficit {clean_deficit}."
        )

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
    if rejected_calibrations:
        names = ", ".join(f"`{screen.name}`" for screen in rejected_calibrations)
        lines.append(f"- Rejected pre-method calibration route(s): {names}.")
    if usable_calibrations:
        names = ", ".join(
            f"`{screen.name}` clean {screen.clean_success:.4f}, range {screen.min_recovery:.4f}--{screen.max_recovery:.4f}, r80 {screen.r80:.4f}"
            for screen in usable_calibrations
        )
        lines.append(f"- Usable pre-method calibration route(s): {names}.")
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
    completed_negative_calibration_names = {"Gymnasium MuJoCo Reacher-v5 calibration"}
    active_calibrations = [
        screen
        for screen in usable_calibrations
        if screen.name not in completed_negative_calibration_names
    ]
    if inflight is None and active_calibrations:
        names = ", ".join(f"`{screen.name}`" for screen in active_calibrations)
        lines.append(
            f"- Active route: {names} cleared pre-method calibration; run only the fixed preregistered all-method screen before interpreting it."
        )
    elif inflight is None and usable_calibrations:
        names = ", ".join(f"`{screen.name}`" for screen in usable_calibrations)
        lines.append(f"- Active route: {names} cleared pre-method calibration, but all corresponding completed method screens are negative or absent.")
    elif inflight is None:
        lines.append(
            "- Active route: no queued learned-policy route is recorded in the local ledgers."
        )
    else:
        lines.append(
            f"- Active route: {inflight}"
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
        "- The independent-benchmark route has not produced a promotable screen: the closest external-package screen with a visible RAUC lead fails because the radius metric is saturated, while later non-saturated screens trail uniform, stronger baselines, or the state-priority/uniform-radius ablation.",
        learned_priority,
    ]
    if any(screen.name == "Gymnasium MuJoCo Reacher-v5 calibration" for screen in usable_calibrations):
        priority_lines.insert(
            2,
            "- The usable Reacher-v5 calibration is pre-method evidence only; the fixed full all-method comparison is now negative and should not be promoted.",
        )
    if active_calibrations:
        names = ", ".join(f"`{screen.name}`" for screen in active_calibrations)
        priority_lines.insert(
            3,
            f"- Active pre-method calibration route(s) awaiting fixed comparison result: {names}.",
        )
    priority_lines.insert(
        2 if not usable_calibrations else 4,
        "- Rejected pre-method calibrations should not be scaled into BGR comparisons until the reset interface and controller first produce clean, non-saturated recovery curves.",
    )
    if inflight is None and active_calibrations:
        priority_lines.append(
            "- The next acceptance-moving work is the fixed InvertedPendulum-v5 all-method screen; do not tune the protocol after seeing its method results."
        )
    elif inflight is None and usable_calibrations:
        priority_lines.append(
            "- The next acceptance-moving work must find a genuinely different independent route, change the learned-policy intervention, or strengthen theory/presentation; the Reacher route is now scope evidence, not acceptance evidence."
        )
    elif inflight is None:
        priority_lines.append(
            "- The next acceptance-moving work should change the learned-policy intervention or materially strengthen theory/presentation; another same-protocol MiniGrid/classic-control screen is unlikely to move the gate."
        )
    else:
        priority_lines.append(
            "- The current acceptance-moving learned-policy work is the repaired proximal-anchor OpenVLA route; do not treat it as evidence until compact summaries exist and clear the fixed +10/400 and +0.02 gate."
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
