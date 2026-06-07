"""Summarize whether the current evidence clears the internal AAAI-readiness gate."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

OPENVLA_NON_IDENTITY_PERTURBATIONS = {"blur", "brightness", "occlusion", "shift"}
OPENVLA_WEIGHTED_COMPLETE = (
    "results/openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_step50500_lr5em7_identitylora_"
    "imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv"
)
OPENVLA_LEGACY_COMPLETE = (
    "results/openvla_oft_perturb_eval_cleanmix_p4096_commonavail_step50500_lr5em7_identitylora_imageaug_"
    "officialtrainstats_prereg_fullgoal10x10_v1/summary.csv"
)
OPENVLA_PROXIMAL_ANCHOR_COMPLETE = (
    "results/openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_proxanchor_l2_1em0_ddpgradfix_v1_"
    "step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv"
)
OPENVLA_PERTURB_ONLY_ANCHOR_COMPLETE = (
    "results/openvla_oft_perturb_eval_p2048unique_perturbonly_anchor_prereg_perturbonly_proxanchor_l2_5em0_"
    "step50300_lr2em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv"
)
OPENVLA_PERTURB_ONLY_ANCHOR_MARKER = "queue_openvla_oft_preregistered_perturb_only_anchor.sh"
OPENVLA_PROXIMAL_ANCHOR_JOB_IDS = {
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
CALIBRATION_SUMMARIES = [
    ("FetchPush calibration", "results/fetchpush_object_goal_calibration_2seed_v1/summary.json"),
    ("FetchSlide calibration", "results/fetchslide_object_goal_calibration_2seed_v1/summary.json"),
    ("FetchPickAndPlace calibration", "results/fetchpickplace_object_goal_calibration_2seed_v1/summary.json"),
    ("Highway parking calibration", "results/highway_parking_recovery_calibration_12seed_v1/summary.json"),
    ("Reacher-v5 calibration", "results/reacher_recovery_calibration_12seed_v1/summary.json"),
    ("InvertedPendulum-v5 calibration", "results/inverted_pendulum_recovery_calibration_12seed_v1/summary.json"),
    (
        "InvertedDoublePendulum-v5 calibration",
        "results/inverted_double_pendulum_recovery_calibration_12seed_v1/summary.json",
    ),
    ("LunarLander-v3 calibration", "results/lunarlander_recovery_calibration_12seed_v1/summary.json"),
]
ROADMAP_DOCS = [
    "AGENTS.md",
    "docs/aaai_acceptance_gap.md",
    "docs/review_weakness_response.md",
]
STALE_ROADMAP_SNIPPETS = [
    "Treat official MiniGrid-FourRooms as the next",
    "is now the next fixed",
    "The next preregistered external-package screen is official",
    "The next preregistered PointMaze",
    "The next preregistered learned-policy intervention is weighted perturbation",
    "The current learned-policy follow-up is the preregistered weighted",
    "The next independent-benchmark route is Gymnasium-Robotics FetchReach",
    "The next harder Gymnasium-Robotics calibration route",
    "The next Gymnasium-Robotics object calibration",
    "Official MiniGrid-LavaCrossingS9N3 is the next",
]


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_metric(rows: list[dict[str, str]], method: str, metric: str) -> float:
    values = [float(row[metric]) for row in rows if row.get("method") == method]
    if not values:
        raise ValueError(f"{method=} {metric=} not found")
    return mean(values)


def paired_wins(rows: list[dict[str, str]], treatment: str, baseline: str, metric: str) -> tuple[int, int, int]:
    seeds = sorted({int(float(row["seed"])) for row in rows if row.get("method") in {treatment, baseline}})
    wins = losses = ties = 0
    for seed in seeds:
        seed_rows = [row for row in rows if int(float(row["seed"])) == seed]
        delta = mean_metric(seed_rows, treatment, metric) - mean_metric(seed_rows, baseline, metric)
        if abs(delta) < 1e-12:
            ties += 1
        elif delta > 0.0:
            wins += 1
        else:
            losses += 1
    return wins, losses, ties


def success_total(rows: list[dict[str, str]], method: str, *, exclude_identity: bool) -> tuple[int, int]:
    successes = 0
    episodes = 0
    for row in rows:
        if row.get("method") != method:
            continue
        if exclude_identity and row.get("perturbation") == "identity":
            continue
        successes += int(row["successes"])
        episodes += int(row["episodes"])
    if episodes == 0:
        raise ValueError(f"no OpenVLA rows for {method=}")
    return successes, episodes


def perturbation_total(rows: list[dict[str, str]], method: str, perturbations: set[str]) -> tuple[int, int]:
    successes = 0
    episodes = 0
    for row in rows:
        if row.get("method") != method or row.get("perturbation") not in perturbations:
            continue
        successes += int(float(row["successes"]))
        episodes += int(float(row["episodes"]))
    if episodes == 0:
        raise ValueError(f"no OpenVLA rows for {method=} {perturbations=}")
    return successes, episodes


def missing_openvla_rows(rows: list[dict[str, str]], methods: set[str], perturbations: set[str]) -> list[str]:
    present = {(row.get("method"), row.get("perturbation")) for row in rows}
    return [
        f"{method}/{perturbation}"
        for method in sorted(methods)
        for perturbation in sorted(perturbations)
        if (method, perturbation) not in present
    ]


def learned_policy_inflight_detail(root: Path) -> str | None:
    if (root / OPENVLA_PERTURB_ONLY_ANCHOR_COMPLETE).exists():
        return None

    ledger_text = ""
    for relative_path in ["AGENTS.md", "results/README.md", "docs/aaai_acceptance_gap.md"]:
        path = root / relative_path
        if path.exists():
            ledger_text += "\n" + path.read_text(encoding="utf-8")

    if OPENVLA_PERTURB_ONLY_ANCHOR_MARKER in ledger_text:
        return (
            "perturb-only anchored OpenVLA route preregistered, not yet evidence: "
            "prepare perturb-only BGR/random TFDS roots, adapt both branches with the stronger "
            "official-checkpoint proximal anchor, and run the fixed official/BGR/random "
            "10-task x 10-trial perturbation eval before applying the +10/400 and +0.02 "
            "learned-policy gate"
        )

    if (root / OPENVLA_PROXIMAL_ANCHOR_COMPLETE).exists():
        return None

    if (
        OPENVLA_PROXIMAL_ANCHOR_JOB_IDS["bgr_train"] not in ledger_text
        or OPENVLA_PROXIMAL_ANCHOR_JOB_IDS["random_perturb_last"] not in ledger_text
    ):
        return None

    bgr_train = OPENVLA_PROXIMAL_ANCHOR_JOB_IDS["bgr_train"]
    if f"{bgr_train}` failed" in ledger_text or f"{bgr_train} failed" in ledger_text:
        return (
            "proximal-anchor route failed before producing evidence: BGR train job "
            f"{bgr_train} exited 1:0 with a PyTorch DDP ready-twice error in the proximal-anchor "
            "wrapper, leaving downstream BGR/random jobs dependency-held; repair or retire "
            "the route before applying the learned-policy gate"
        )

    ids = OPENVLA_PROXIMAL_ANCHOR_JOB_IDS
    return (
        "proximal-anchor route unsummarized, not yet evidence: adaptation jobs "
        f"BGR {ids['bgr_train']}/{ids['bgr_merge']}/{ids['bgr_clean_eval']} and random "
        f"{ids['random_train']}/{ids['random_merge']}/{ids['random_clean_eval']} have no complete "
        f"fixed summaries; fixed perturbation jobs official {ids['official_first']}--{ids['official_last']}, "
        f"BGR {ids['bgr_perturb_first']}--{ids['bgr_perturb_last']}, and random "
        f"{ids['random_perturb_first']}--{ids['random_perturb_last']} must finish before the +10/400 "
        "and +0.02 learned-policy gate can be checked"
    )


def learned_policy_summary_gate(root: Path, relative_path: str, *, label: str) -> GateResult:
    rows = read_rows(root / relative_path)
    methods = {"bgr", "official", "random"}
    required_perturbations = OPENVLA_NON_IDENTITY_PERTURBATIONS | {"identity"}
    missing = missing_openvla_rows(rows, methods, required_perturbations)
    if missing:
        return GateResult(
            "learned-policy OpenVLA/LIBERO",
            False,
            f"{label} incomplete; missing {', '.join(missing)}",
        )

    bgr_success, bgr_episodes = perturbation_total(rows, "bgr", OPENVLA_NON_IDENTITY_PERTURBATIONS)
    official_success, official_episodes = perturbation_total(rows, "official", OPENVLA_NON_IDENTITY_PERTURBATIONS)
    random_success, random_episodes = perturbation_total(rows, "random", OPENVLA_NON_IDENTITY_PERTURBATIONS)
    bgr_identity, identity_eps = success_total(
        [row for row in rows if row.get("perturbation") == "identity"], "bgr", exclude_identity=False
    )
    official_identity, _ = success_total(
        [row for row in rows if row.get("perturbation") == "identity"], "official", exclude_identity=False
    )
    random_identity, _ = success_total(
        [row for row in rows if row.get("perturbation") == "identity"], "random", exclude_identity=False
    )

    official_margin = bgr_success - official_success
    random_margin = bgr_success - random_success
    official_rate_margin = bgr_success / bgr_episodes - official_success / official_episodes
    random_rate_margin = bgr_success / bgr_episodes - random_success / random_episodes
    clean_floor = max(official_identity, random_identity) - 1
    passed = (
        bgr_episodes == 400
        and official_episodes == 400
        and random_episodes == 400
        and official_margin >= 10
        and random_margin >= 10
        and official_rate_margin >= 0.02
        and random_rate_margin >= 0.02
        and bgr_identity >= clean_floor
    )
    return GateResult(
        "learned-policy OpenVLA/LIBERO",
        passed,
        (
            f"{label} non-identity BGR {bgr_success}/{bgr_episodes}, "
            f"official {official_success}/{official_episodes}, random {random_success}/{random_episodes}; "
            f"identity BGR {bgr_identity}/{identity_eps}, official {official_identity}/{identity_eps}, "
            f"random {random_identity}/{identity_eps}; official_margin={official_margin}, "
            f"random_margin={random_margin}, official_rate_margin={official_rate_margin:+.4f}, "
            f"random_rate_margin={random_rate_margin:+.4f}"
        ),
    )


def grid_mechanism_gate(root: Path) -> GateResult:
    original = read_rows(root / "results/grid_margin_full_30seed_v1/summary.csv")
    replication = read_rows(root / "results/grid_margin_full_replication_30seed_v1/summary.csv")
    pooled = original + replication
    bgr = mean_metric(pooled, "bgr", "final_rauc")
    uniform = mean_metric(pooled, "uniform", "final_rauc")
    wins = paired_wins(original, "bgr", "uniform", "final_rauc")
    rep_wins = paired_wins(replication, "bgr", "uniform", "final_rauc")
    passed = (bgr - uniform) >= 0.03 and wins == (30, 0, 0) and rep_wins == (30, 0, 0)
    return GateResult(
        "controlled grid mechanism",
        passed,
        f"pooled RAUC {bgr:.4f} vs {uniform:.4f}; original W/L/T={wins}; held-out W/L/T={rep_wins}",
    )


def independent_benchmark_gate(root: Path) -> GateResult:
    failures: list[str] = []

    frozen = read_rows(root / "results/frozenlake_recovery_focused_30seed_v1/summary.csv")
    frozen_bgr = mean_metric(frozen, "bgr", "final_rauc")
    frozen_uniform = mean_metric(frozen, "uniform", "final_rauc")
    frozen_failure = mean_metric(frozen, "failure_only", "final_rauc")
    if not (frozen_bgr > frozen_uniform and frozen_bgr > frozen_failure):
        failures.append(
            f"FrozenLake not promotable: BGR {frozen_bgr:.4f}, uniform {frozen_uniform:.4f}, failure-only {frozen_failure:.4f}"
        )

    fourrooms = read_rows(root / "results/minigrid_fourrooms_recovery_probe_midband_4seed_v1/summary.csv")
    four_coverage = mean_metric(fourrooms, "bgr_coverage", "final_rauc")
    four_uniform = mean_metric(fourrooms, "uniform", "final_rauc")
    four_failure = mean_metric(fourrooms, "failure_only", "final_rauc")
    if not (four_coverage > four_uniform and four_coverage > four_failure):
        failures.append(
            f"MiniGrid-FourRooms negative: BGR-Coverage {four_coverage:.4f}, uniform {four_uniform:.4f}, failure-only {four_failure:.4f}"
        )

    four_mid25_path = root / "results/minigrid_fourrooms_recovery_probe_mid2_5_4seed_v1/summary.csv"
    if four_mid25_path.exists():
        four_mid25 = read_rows(four_mid25_path)
        mid25_bgr = mean_metric(four_mid25, "bgr", "final_rauc")
        mid25_coverage = mean_metric(four_mid25, "bgr_coverage", "final_rauc")
        mid25_uniform = mean_metric(four_mid25, "uniform", "final_rauc")
        mid25_fixed = mean_metric(four_mid25, "fixed", "final_rauc")
        mid25_failure = mean_metric(four_mid25, "failure_only", "final_rauc")
        mid25_r80 = mean_metric(four_mid25, "bgr", "final_median_r80")
        mid25_uniform_r80 = mean_metric(four_mid25, "uniform", "final_median_r80")
        mid25_wins = paired_wins(four_mid25, "bgr", "uniform", "final_rauc")
        if not (
            mid25_bgr > mid25_uniform
            and mid25_bgr > mid25_fixed
            and mid25_bgr > mid25_failure
            and mid25_wins[0] >= 3
            and mid25_r80 >= mid25_uniform_r80
        ):
            failures.append(
                f"MiniGrid-FourRooms mid2-5 negative: BGR {mid25_bgr:.4f}, BGR-Coverage {mid25_coverage:.4f}, "
                f"uniform {mid25_uniform:.4f}, fixed {mid25_fixed:.4f}, failure-only {mid25_failure:.4f}, "
                f"W/L/T={mid25_wins}, median-r80 {mid25_r80:.4f} vs uniform {mid25_uniform_r80:.4f}"
            )

    doorkey = read_rows(root / "results/minigrid_doorkey_recovery_probe_4seed_v1/summary.csv")
    doorkey_coverage = mean_metric(doorkey, "bgr_coverage", "final_rauc")
    doorkey_uniform = mean_metric(doorkey, "uniform", "final_rauc")
    doorkey_failure = mean_metric(doorkey, "failure_only", "final_rauc")
    if not (doorkey_coverage > doorkey_uniform and doorkey_coverage > doorkey_failure):
        failures.append(
            f"MiniGrid-DoorKey negative: BGR-Coverage {doorkey_coverage:.4f}, uniform {doorkey_uniform:.4f}, failure-only {doorkey_failure:.4f}"
        )

    lava = read_rows(root / "results/minigrid_lavacrossing_recovery_probe_4seed_v1/summary.csv")
    lava_coverage = mean_metric(lava, "bgr_coverage", "final_rauc")
    lava_uniform = mean_metric(lava, "uniform", "final_rauc")
    lava_ablation = mean_metric(lava, "bgr_uniform_radius", "final_rauc")
    if not (lava_coverage > lava_uniform and lava_coverage > lava_ablation):
        failures.append(
            f"MiniGrid-LavaCrossing negative: BGR-Coverage {lava_coverage:.4f}, uniform {lava_uniform:.4f}, uniform-radius {lava_ablation:.4f}"
        )

    lavagap_path = root / "results/minigrid_lavagap_s7_recovery_probe_4seed_v1/summary.csv"
    if lavagap_path.exists():
        lavagap = read_rows(lavagap_path)
        gap_coverage = mean_metric(lavagap, "bgr_coverage", "final_rauc")
        gap_uniform = mean_metric(lavagap, "uniform", "final_rauc")
        gap_ablation = mean_metric(lavagap, "bgr_uniform_radius", "final_rauc")
        if not (gap_coverage > gap_uniform and gap_coverage > gap_ablation):
            failures.append(
                f"MiniGrid-LavaGap negative: BGR-Coverage {gap_coverage:.4f}, uniform {gap_uniform:.4f}, uniform-radius {gap_ablation:.4f}"
            )

    pointmaze = read_rows(root / "results/pointmaze_umaze_clean_shield_probe_4seed_v1/summary.csv")
    shield = mean_metric(pointmaze, "bgr_clean_shield", "final_rauc")
    point_uniform = mean_metric(pointmaze, "uniform", "final_rauc")
    point_failure = mean_metric(pointmaze, "failure_only", "final_rauc")
    shield_wins = paired_wins(pointmaze, "bgr_clean_shield", "uniform", "final_rauc")
    if not (shield > point_uniform and shield > point_failure and shield_wins[0] >= 3):
        failures.append(
            f"PointMaze negative: BGR-Clean-Shield {shield:.4f}, uniform {point_uniform:.4f}, failure-only {point_failure:.4f}, W/L/T={shield_wins}"
        )

    fetch_path = root / "results/fetchreach_goal_recovery_probe_4seed_v1/summary.csv"
    if fetch_path.exists():
        fetch = read_rows(fetch_path)
        fetch_coverage = mean_metric(fetch, "bgr_coverage", "final_rauc")
        fetch_bgr = mean_metric(fetch, "bgr", "final_rauc")
        fetch_uniform = mean_metric(fetch, "uniform", "final_rauc")
        fetch_failure = mean_metric(fetch, "failure_only", "final_rauc")
        fetch_r80 = mean_metric(fetch, "bgr_coverage", "final_median_r80")
        if not (fetch_coverage > fetch_uniform and fetch_coverage > fetch_failure and fetch_r80 < 0.15):
            failures.append(
                f"FetchReach negative/saturated: BGR-Coverage {fetch_coverage:.4f}, BGR {fetch_bgr:.4f}, "
                f"uniform {fetch_uniform:.4f}, failure-only {fetch_failure:.4f}, median-r80 {fetch_r80:.4f}"
            )

    fetch_hard_path = root / "results/fetchreach_goal_recovery_hard_probe_4seed_v1/summary.csv"
    if fetch_hard_path.exists():
        fetch_hard = read_rows(fetch_hard_path)
        hard_coverage = mean_metric(fetch_hard, "bgr_coverage", "final_rauc")
        hard_bgr = mean_metric(fetch_hard, "bgr", "final_rauc")
        hard_uniform = mean_metric(fetch_hard, "uniform", "final_rauc")
        hard_failure = mean_metric(fetch_hard, "failure_only", "final_rauc")
        hard_ablation = mean_metric(fetch_hard, "bgr_uniform_radius", "final_rauc")
        hard_r80 = mean_metric(fetch_hard, "bgr_coverage", "final_median_r80")
        hard_wins = paired_wins(fetch_hard, "bgr_coverage", "uniform", "final_rauc")
        if not (
            hard_coverage > hard_uniform
            and hard_coverage > hard_failure
            and hard_coverage > hard_ablation
            and hard_wins[0] >= 3
            and hard_r80 < 0.15
        ):
            failures.append(
                f"FetchReach hard-budget negative/saturated: BGR-Coverage {hard_coverage:.4f}, BGR {hard_bgr:.4f}, "
                f"uniform {hard_uniform:.4f}, failure-only {hard_failure:.4f}, uniform-radius {hard_ablation:.4f}, "
                f"W/L/T={hard_wins}, median-r80 {hard_r80:.4f}"
            )

    reacher_path = root / "results/reacher_recovery_probe_12seed_v1/summary.csv"
    if reacher_path.exists():
        reacher = read_rows(reacher_path)
        reacher_bgr = mean_metric(reacher, "bgr", "final_rauc")
        reacher_coverage = mean_metric(reacher, "bgr_coverage", "final_rauc")
        reacher_uniform = mean_metric(reacher, "uniform", "final_rauc")
        reacher_failure = mean_metric(reacher, "failure_only", "final_rauc")
        reacher_fixed = mean_metric(reacher, "fixed", "final_rauc")
        reacher_td = mean_metric(reacher, "td_loss", "final_rauc")
        reacher_ablation = mean_metric(reacher, "bgr_uniform_radius", "final_rauc")
        reacher_bgr_wins = paired_wins(reacher, "bgr", "uniform", "final_rauc")
        reacher_coverage_wins = paired_wins(reacher, "bgr_coverage", "uniform", "final_rauc")
        best_reacher_wins = reacher_bgr_wins if reacher_bgr >= reacher_coverage else reacher_coverage_wins
        if not (
            (reacher_bgr > reacher_uniform or reacher_coverage > reacher_uniform)
            and max(reacher_bgr, reacher_coverage) > max(reacher_failure, reacher_fixed, reacher_td, reacher_ablation)
            and best_reacher_wins[0] >= 9
        ):
            failures.append(
                f"Reacher-v5 negative: BGR {reacher_bgr:.4f}, BGR-Coverage {reacher_coverage:.4f}, "
                f"uniform {reacher_uniform:.4f}, failure-only {reacher_failure:.4f}, fixed {reacher_fixed:.4f}, "
                f"TD-loss {reacher_td:.4f}, uniform-radius {reacher_ablation:.4f}, W/L/T={best_reacher_wins}"
            )

    inverted_path = root / "results/inverted_pendulum_recovery_probe_4seed_v1/summary.csv"
    if inverted_path.exists():
        inverted = read_rows(inverted_path)
        inverted_bgr = mean_metric(inverted, "bgr", "final_rauc")
        inverted_coverage = mean_metric(inverted, "bgr_coverage", "final_rauc")
        inverted_uniform = mean_metric(inverted, "uniform", "final_rauc")
        inverted_failure = mean_metric(inverted, "failure_only", "final_rauc")
        inverted_fixed = mean_metric(inverted, "fixed", "final_rauc")
        inverted_td = mean_metric(inverted, "td_loss", "final_rauc")
        inverted_ablation = mean_metric(inverted, "bgr_uniform_radius", "final_rauc")
        inverted_bgr_wins = paired_wins(inverted, "bgr", "uniform", "final_rauc")
        inverted_coverage_wins = paired_wins(inverted, "bgr_coverage", "uniform", "final_rauc")
        best_inverted_wins = inverted_bgr_wins if inverted_bgr >= inverted_coverage else inverted_coverage_wins
        if not (
            (inverted_bgr > inverted_uniform or inverted_coverage > inverted_uniform)
            and max(inverted_bgr, inverted_coverage) > max(
                inverted_failure,
                inverted_fixed,
                inverted_td,
                inverted_ablation,
            )
            and best_inverted_wins[0] >= 3
        ):
            failures.append(
                f"InvertedPendulum-v5 negative/tied: BGR {inverted_bgr:.4f}, "
                f"BGR-Coverage {inverted_coverage:.4f}, uniform {inverted_uniform:.4f}, "
                f"failure-only {inverted_failure:.4f}, fixed {inverted_fixed:.4f}, "
                f"TD-loss {inverted_td:.4f}, uniform-radius {inverted_ablation:.4f}, W/L/T={best_inverted_wins}"
            )

    inverted_double_path = root / "results/inverted_double_pendulum_recovery_probe_4seed_v1/summary.csv"
    if inverted_double_path.exists():
        inverted_double = read_rows(inverted_double_path)
        double_bgr = mean_metric(inverted_double, "bgr", "final_rauc")
        double_coverage = mean_metric(inverted_double, "bgr_coverage", "final_rauc")
        double_uniform = mean_metric(inverted_double, "uniform", "final_rauc")
        double_failure = mean_metric(inverted_double, "failure_only", "final_rauc")
        double_fixed = mean_metric(inverted_double, "fixed", "final_rauc")
        double_td = mean_metric(inverted_double, "td_loss", "final_rauc")
        double_ablation = mean_metric(inverted_double, "bgr_uniform_radius", "final_rauc")
        double_bgr_clean = mean_metric(inverted_double, "bgr", "final_clean")
        double_coverage_clean = mean_metric(inverted_double, "bgr_coverage", "final_clean")
        double_bgr_wins = paired_wins(inverted_double, "bgr", "uniform", "final_rauc")
        double_coverage_wins = paired_wins(inverted_double, "bgr_coverage", "uniform", "final_rauc")
        best_double_wins = double_bgr_wins if double_bgr >= double_coverage else double_coverage_wins
        if not (
            (double_bgr > double_uniform or double_coverage > double_uniform)
            and max(double_bgr, double_coverage) > max(
                double_failure,
                double_fixed,
                double_td,
                double_ablation,
            )
            and max(double_bgr_clean, double_coverage_clean) >= 0.75
            and best_double_wins[0] >= 3
        ):
            failures.append(
                f"InvertedDoublePendulum-v5 negative/collapsed: BGR {double_bgr:.4f}, "
                f"BGR-Coverage {double_coverage:.4f}, uniform {double_uniform:.4f}, "
                f"failure-only {double_failure:.4f}, fixed {double_fixed:.4f}, "
                f"TD-loss {double_td:.4f}, uniform-radius {double_ablation:.4f}, "
                f"BGR clean {double_bgr_clean:.4f}, BGR-Coverage clean {double_coverage_clean:.4f}, "
                f"W/L/T={best_double_wins}"
            )

    lunar_path = root / "results/lunarlander_recovery_probe_4seed_v1/summary.csv"
    if lunar_path.exists():
        lunar = read_rows(lunar_path)
        lunar_bgr = mean_metric(lunar, "bgr", "final_rauc")
        lunar_coverage = mean_metric(lunar, "bgr_coverage", "final_rauc")
        lunar_uniform = mean_metric(lunar, "uniform", "final_rauc")
        lunar_failure = mean_metric(lunar, "failure_only", "final_rauc")
        lunar_fixed = mean_metric(lunar, "fixed", "final_rauc")
        lunar_td = mean_metric(lunar, "td_loss", "final_rauc")
        lunar_ablation = mean_metric(lunar, "bgr_uniform_radius", "final_rauc")
        lunar_bgr_r80 = mean_metric(lunar, "bgr", "final_median_r80")
        lunar_coverage_r80 = mean_metric(lunar, "bgr_coverage", "final_median_r80")
        lunar_uniform_r80 = mean_metric(lunar, "uniform", "final_median_r80")
        lunar_bgr_wins = paired_wins(lunar, "bgr", "uniform", "final_rauc")
        lunar_coverage_wins = paired_wins(lunar, "bgr_coverage", "uniform", "final_rauc")
        best_lunar = lunar_bgr if lunar_bgr >= lunar_coverage else lunar_coverage
        best_lunar_r80 = lunar_bgr_r80 if lunar_bgr >= lunar_coverage else lunar_coverage_r80
        best_lunar_wins = lunar_bgr_wins if lunar_bgr >= lunar_coverage else lunar_coverage_wins
        if not (
            best_lunar > lunar_uniform
            and best_lunar > max(lunar_failure, lunar_fixed, lunar_td, lunar_ablation)
            and best_lunar_wins[0] >= 3
            and best_lunar_r80 >= lunar_uniform_r80
            and not (best_lunar_r80 >= 0.99 and lunar_uniform_r80 >= 0.99)
            and not (best_lunar_r80 <= 0.01 and lunar_uniform_r80 <= 0.01)
        ):
            failures.append(
                f"LunarLander-v3 negative: BGR {lunar_bgr:.4f}, BGR-Coverage {lunar_coverage:.4f}, "
                f"uniform {lunar_uniform:.4f}, failure-only {lunar_failure:.4f}, fixed {lunar_fixed:.4f}, "
                f"TD-loss {lunar_td:.4f}, uniform-radius {lunar_ablation:.4f}, "
                f"best-r80 {best_lunar_r80:.4f} vs uniform {lunar_uniform_r80:.4f}, W/L/T={best_lunar_wins}"
            )

    for label, relative_path in CALIBRATION_SUMMARIES:
        calibration_path = root / relative_path
        if not calibration_path.exists():
            continue
        summary = json.loads(calibration_path.read_text(encoding="utf-8"))
        clean = float(summary["clean_success"])
        min_recovery = float(summary["min_recovery"])
        max_recovery = float(summary["max_recovery"])
        r80 = float(summary["r80"])
        if clean < 0.80 or abs(max_recovery - min_recovery) < 0.20:
            failures.append(
                f"{label} invalid: clean {clean:.4f}, recovery range "
                f"{min_recovery:.4f}--{max_recovery:.4f}, median-r80 {r80:.4f}"
            )

    return GateResult(
        "independent/pre-existing benchmark",
        not failures,
        "; ".join(failures) if failures else "at least one independent benchmark clears the promotion criteria",
    )


def learned_policy_gate(root: Path) -> GateResult:
    inflight_detail = learned_policy_inflight_detail(root)
    if (root / OPENVLA_PERTURB_ONLY_ANCHOR_COMPLETE).exists():
        return learned_policy_summary_gate(
            root,
            OPENVLA_PERTURB_ONLY_ANCHOR_COMPLETE,
            label="latest perturb-only anchored audit",
        )
    if (root / OPENVLA_PROXIMAL_ANCHOR_COMPLETE).exists():
        gate = learned_policy_summary_gate(root, OPENVLA_PROXIMAL_ANCHOR_COMPLETE, label="latest proximal-anchor audit")
        if inflight_detail:
            return GateResult(gate.name, gate.passed, f"{gate.detail}; {inflight_detail}")
        return gate

    weighted_path = root / OPENVLA_WEIGHTED_COMPLETE
    if weighted_path.exists():
        rows = read_rows(weighted_path)
        bgr_success, bgr_episodes = perturbation_total(rows, "bgr", OPENVLA_NON_IDENTITY_PERTURBATIONS)
        official_success, official_episodes = perturbation_total(rows, "official", OPENVLA_NON_IDENTITY_PERTURBATIONS)
        random_success, random_episodes = perturbation_total(rows, "random", OPENVLA_NON_IDENTITY_PERTURBATIONS)
        bgr_identity, identity_eps = success_total(
            [row for row in rows if row.get("perturbation") == "identity"], "bgr", exclude_identity=False
        )
        official_identity, _ = success_total(
            [row for row in rows if row.get("perturbation") == "identity"], "official", exclude_identity=False
        )
        random_identity, _ = success_total(
            [row for row in rows if row.get("perturbation") == "identity"], "random", exclude_identity=False
        )
        official_margin = bgr_success - official_success
        official_rate_margin = bgr_success / bgr_episodes - official_success / official_episodes
        random_detail = f"random {random_success}/{random_episodes}"
        official_gate_impossible = (
            bgr_episodes == 400
            and official_episodes == 400
            and (official_margin < 10 or official_rate_margin < 0.02)
        )
        gate_detail = ""
        if official_gate_impossible:
            gate_detail = "; weighted audit complete and negative against the preregistered gate"
        if inflight_detail:
            gate_detail = f"{gate_detail}; {inflight_detail}"
        clean_floor = max(official_identity, random_identity) - 1
        passed = (
            bgr_episodes == 400
            and official_episodes == 400
            and random_episodes == 400
            and official_margin >= 10
            and official_rate_margin >= 0.02
            and bgr_success - random_success >= 10
            and bgr_success / bgr_episodes - random_success / random_episodes >= 0.02
            and bgr_identity >= clean_floor
        )
        return GateResult(
            "learned-policy OpenVLA/LIBERO",
            passed,
            (
                f"latest weighted audit non-identity BGR {bgr_success}/{bgr_episodes}, "
                f"official {official_success}/{official_episodes}, {random_detail}; "
                f"identity BGR {bgr_identity}/{identity_eps}, official {official_identity}/{identity_eps}, "
                f"random {random_identity}/{identity_eps}; official_margin={official_margin}, "
                f"official_rate_margin={official_rate_margin:+.4f}{gate_detail}"
            ),
        )

    rows = read_rows(root / OPENVLA_LEGACY_COMPLETE)
    bgr_success, non_identity = success_total(rows, "bgr", exclude_identity=True)
    random_success, _ = success_total(rows, "random", exclude_identity=True)
    official_success, _ = success_total(rows, "official", exclude_identity=True)
    bgr_identity, identity_eps = success_total([row for row in rows if row.get("perturbation") == "identity"], "bgr", exclude_identity=False)
    official_identity, _ = success_total([row for row in rows if row.get("perturbation") == "identity"], "official", exclude_identity=False)
    random_identity, _ = success_total([row for row in rows if row.get("perturbation") == "identity"], "random", exclude_identity=False)

    margin_vs_best = bgr_success - max(random_success, official_success)
    clean_floor = max(official_identity, random_identity) - 1
    passed = margin_vs_best >= 10 and (bgr_success / non_identity - max(random_success, official_success) / non_identity) >= 0.02 and bgr_identity >= clean_floor
    gate_detail = f"; {inflight_detail}" if inflight_detail else ""
    return GateResult(
        "learned-policy OpenVLA/LIBERO",
        passed,
        (
            f"non-identity successes BGR {bgr_success}/{non_identity}, official {official_success}/{non_identity}, "
            f"random {random_success}/{non_identity}; identity BGR {bgr_identity}/{identity_eps}, "
            f"official {official_identity}/{identity_eps}, random {random_identity}/{identity_eps}; "
            f"margin_vs_best={margin_vs_best}{gate_detail}"
        ),
    )


def roadmap_hygiene_gate(root: Path) -> GateResult:
    offenders: list[str] = []
    for relative in ROADMAP_DOCS:
        path = root / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in STALE_ROADMAP_SNIPPETS:
            if snippet in text:
                offenders.append(f"{relative}: {snippet}")
    return GateResult(
        "acceptance roadmap hygiene",
        not offenders,
        "active roadmap avoids stale next-step instructions" if not offenders else "; ".join(offenders),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--require-ready", action="store_true", help="exit nonzero if the AAAI-readiness gate is not cleared")
    args = parser.parse_args()

    gates = [
        grid_mechanism_gate(args.root),
        independent_benchmark_gate(args.root),
        learned_policy_gate(args.root),
        roadmap_hygiene_gate(args.root),
    ]
    ready = all(gate.passed for gate in gates)
    for gate in gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"[{status}] {gate.name}: {gate.detail}")
    print(f"[decision] {'READY_FOR_90P_AAAI_CLAIM' if ready else 'NOT_READY_FOR_90P_AAAI_CLAIM'}")
    return 1 if args.require_ready and not ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
