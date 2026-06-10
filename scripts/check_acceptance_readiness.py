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
OPENVLA_OCCLUSION_BOTTLENECK_COMPLETE = (
    "results/openvla_oft_perturb_eval_cleanmix_p2048unique_occlusion_bottleneck_prereg_proxanchor_l2_5em0_"
    "step50400_lr2em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv"
)
OPENVLA_HARD_OCCLUSION_TRANSFER_COMPLETE = (
    "results/openvla_oft_perturb_eval_occlusion_bottleneck_hardocc065_transfer_step50400_lr2em7_v1/summary.csv"
)
OPENVLA_HARD_OCCLUSION080_TRANSFER_COMPLETE = (
    "results/openvla_oft_perturb_eval_occlusion_bottleneck_hardocc080_transfer_step50400_lr2em7_v1/summary.csv"
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
OPENVLA_PERTURB_ONLY_ANCHOR_MARKER = "queue_openvla_oft_preregistered_perturb_only_anchor.sh"
OPENVLA_HARD_OCCLUSION_TRANSFER_MARKER = "sync_openvla_oft_hard_occlusion_transfer_results.sh"
OPENVLA_HARD_OCCLUSION080_TRANSFER_MARKER = "sync_openvla_oft_hard_occlusion080_transfer_results.sh"
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
    ("FetchPush far-push calibration", "results/fetchpush_object_goal_calibration_far_push_2seed_v1/summary.json"),
    ("FetchPush object-state calibration", "results/fetchpush_object_state_calibration_sweep_g8_h250_2seed_v1/summary.json"),
    ("FetchSlide calibration", "results/fetchslide_object_goal_calibration_2seed_v1/summary.json"),
    ("FetchPickAndPlace calibration", "results/fetchpickplace_object_goal_calibration_2seed_v1/summary.json"),
    ("HandReach-v3 calibration", "results/handreach_recovery_calibration_8seed_v1/summary.json"),
    ("Highway parking calibration", "results/highway_parking_recovery_calibration_12seed_v1/summary.json"),
    ("Highway lane calibration", "results/highway_lane_recovery_calibration_12seed_v1/summary.json"),
    ("MinAtar Breakout calibration", "results/minatar_breakout_recovery_calibration_12seed_v1/summary.json"),
    ("MinAtar Asterix calibration", "results/minatar_asterix_recovery_calibration_12seed_v1/summary.json"),
    ("MinAtar Freeway calibration", "results/minatar_freeway_recovery_calibration_20seed_v1/summary.json"),
    (
        "MinAtar Space Invaders calibration",
        "results/minatar_space_invaders_recovery_calibration_20seed_v1/summary.json",
    ),
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
    "The current newly opened independent route",
    "The active acceptance-moving work is now the fixed all-method LunarLander screen",
    "The current preregistered learned-policy route is perturb-only anchored",
    "The latest preregistered learned-policy intervention is",
    "The next preregistered learned-policy route",
    "is now the active pre-method calibration route",
    "is the next independent pre-method",
    "route remains in flight",
    "route remains unevaluated",
    "route still cannot be scored",
    "Until compact summaries exist",
    "As of the 2026-06-05 refresh",
    "Next independent-benchmark route",
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


def partial_openvla_failure_detail(
    root: Path,
    relative_path: str,
    *,
    label: str,
    non_identity_perturbations: set[str],
    min_episode_margin: int = 10,
    min_rate_margin: float = 0.02,
    max_identity_deficit: int = 1,
) -> str | None:
    available_path = root / Path(relative_path).with_name("summary_available.csv")
    if not available_path.exists():
        return None
    rows = read_rows(available_path)
    present = {(row.get("method"), row.get("perturbation")): row for row in rows}
    reasons: list[str] = []

    bgr_identity = present.get(("bgr", "identity"))
    if bgr_identity is not None:
        bgr_identity_success = int(float(bgr_identity["successes"]))
        for comparator in ("official", "random"):
            comparator_identity = present.get((comparator, "identity"))
            if comparator_identity is None:
                continue
            comparator_success = int(float(comparator_identity["successes"]))
            identity_deficit = comparator_success - bgr_identity_success
            if identity_deficit > max_identity_deficit:
                reasons.append(
                    f"identity BGR {bgr_identity_success}/{int(float(bgr_identity['episodes']))} "
                    f"trails {comparator} {comparator_success}/{int(float(comparator_identity['episodes']))} "
                    f"by {identity_deficit} > {max_identity_deficit}"
                )

    for comparator in ("official", "random"):
        try:
            bgr_success, bgr_episodes = perturbation_total(rows, "bgr", non_identity_perturbations)
            comparator_success, comparator_episodes = perturbation_total(rows, comparator, non_identity_perturbations)
        except ValueError:
            continue
        if bgr_episodes != comparator_episodes:
            continue
        episode_margin = bgr_success - comparator_success
        rate_margin = bgr_success / bgr_episodes - comparator_success / comparator_episodes
        failed_thresholds: list[str] = []
        if episode_margin < min_episode_margin:
            failed_thresholds.append(f"margin {episode_margin} < +{min_episode_margin}")
        if rate_margin < min_rate_margin:
            failed_thresholds.append(f"rate {rate_margin:+.4f} < +{min_rate_margin:.2f}")
        if failed_thresholds:
            reasons.append(
                f"non-identity BGR {bgr_success}/{bgr_episodes} vs {comparator} "
                f"{comparator_success}/{comparator_episodes} gives margin {episode_margin} "
                f"and rate {rate_margin:+.4f}; fails {' and '.join(failed_thresholds)}"
            )

    if not reasons:
        return None
    return f"{label} closed negative on partial summary: {'; '.join(reasons)}"


def learned_policy_inflight_detail(root: Path) -> str | None:
    ledger_text = ""
    for relative_path in ["AGENTS.md", "results/README.md", "docs/aaai_acceptance_gap.md"]:
        path = root / relative_path
        if path.exists():
            ledger_text += "\n" + path.read_text(encoding="utf-8")

    inflight: list[str] = []

    def append_openvla_route(marker: str, complete_path: str, label: str, pending_detail: str) -> None:
        if marker not in ledger_text or (root / complete_path).exists():
            return
        partial_failure = partial_openvla_failure_detail(
            root,
            complete_path,
            label=label,
            non_identity_perturbations={"occlusion"},
        )
        inflight.append(partial_failure or pending_detail)

    append_openvla_route(
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_MARKER,
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_COMPLETE,
        "hard-occlusion 0.80 micro identity-anchored OpenVLA adaptation",
        "hard-occlusion 0.80 micro identity-anchored OpenVLA adaptation route is queued/running and still missing a complete summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_A40_MARKER,
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_A40_COMPLETE,
        "hard-occlusion 0.80 micro identity-anchored A40 OpenVLA adaptation",
        "hard-occlusion 0.80 micro identity-anchored A40 OpenVLA adaptation route is queued/running and still missing a complete summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MARKER,
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_COMPLETE,
        "hard-occlusion 0.80 identity-anchored OpenVLA adaptation",
        "hard-occlusion 0.80 identity-anchored OpenVLA adaptation route is queued/running and still missing a complete summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_A40_MARKER,
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_A40_COMPLETE,
        "hard-occlusion 0.80 identity-anchored A40 OpenVLA adaptation",
        "hard-occlusion 0.80 identity-anchored A40 OpenVLA adaptation route is queued/running and still missing a complete summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_STRICT_MARKER,
        OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_STRICT_COMPLETE,
        "hard-occlusion 0.80 strict identity-anchored OpenVLA adaptation",
        "hard-occlusion 0.80 strict identity-anchored OpenVLA adaptation route is queued/running and still missing a complete summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION090_IDENTITY_ANCHOR_STRICT_MARKER,
        OPENVLA_HARD_OCCLUSION090_IDENTITY_ANCHOR_STRICT_COMPLETE,
        "hard-occlusion 0.90 strict identity-anchored OpenVLA adaptation",
        "hard-occlusion 0.90 strict identity-anchored OpenVLA adaptation route is queued/running and still missing a complete summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION080_TRANSFER_MARKER,
        OPENVLA_HARD_OCCLUSION080_TRANSFER_COMPLETE,
        "hard-occlusion 0.80 transfer OpenVLA route",
        "hard-occlusion 0.80 transfer OpenVLA route is queued/running and still missing a complete summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION_TRANSFER_MARKER,
        OPENVLA_HARD_OCCLUSION_TRANSFER_COMPLETE,
        "hard-occlusion transfer OpenVLA route",
        "hard-occlusion transfer OpenVLA route is queued/running and still missing a complete summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION_ADAPT_MARKER,
        OPENVLA_HARD_OCCLUSION_ADAPT_COMPLETE,
        "hard-occlusion adaptation OpenVLA route",
        "hard-occlusion adaptation OpenVLA route is queued and still missing logs/summary",
    )
    append_openvla_route(
        OPENVLA_HARD_OCCLUSION_ADAPT_A40_MARKER,
        OPENVLA_HARD_OCCLUSION_ADAPT_A40_COMPLETE,
        "hard-occlusion A40 fallback OpenVLA adaptation",
        "hard-occlusion A40 fallback OpenVLA adaptation route is queued/running and still missing a complete summary",
    )
    if inflight:
        return "; ".join(inflight)

    if (root / OPENVLA_PERTURB_ONLY_ANCHOR_COMPLETE).exists():
        return None

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


def learned_policy_summary_gate(
    root: Path,
    relative_path: str,
    *,
    label: str,
    non_identity_perturbations: set[str] | None = None,
) -> GateResult:
    rows = read_rows(root / relative_path)
    methods = {"bgr", "official", "random"}
    required_non_identity = non_identity_perturbations or OPENVLA_NON_IDENTITY_PERTURBATIONS
    required_perturbations = required_non_identity | {"identity"}
    missing = missing_openvla_rows(rows, methods, required_perturbations)
    if missing:
        return GateResult(
            "learned-policy OpenVLA/LIBERO",
            False,
            f"{label} incomplete; missing {', '.join(missing)}",
        )

    bgr_success, bgr_episodes = perturbation_total(rows, "bgr", required_non_identity)
    official_success, official_episodes = perturbation_total(rows, "official", required_non_identity)
    random_success, random_episodes = perturbation_total(rows, "random", required_non_identity)
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


def openml_replicated_positive_detail(
    root: Path,
    *,
    label: str,
    original_path: Path,
    replication_path: Path,
    dataset: str | None = None,
    min_uniform_wins: int = 20,
    min_fixed_wins: int = 18,
) -> str | None:
    if not original_path.exists() or not replication_path.exists():
        return None

    original = read_rows(original_path)
    replication = read_rows(replication_path)
    if dataset is not None:
        original = [row for row in original if row.get("dataset") == dataset]
        replication = [row for row in replication if row.get("dataset") == dataset]
    pooled = original + replication

    def comparison_detail(rows: list[dict[str, str]]) -> tuple[float, float, tuple[int, int, int], tuple[int, int, int]]:
        bgr = mean_metric(rows, "bgr", "final_rauc")
        uniform = mean_metric(rows, "uniform", "final_rauc")
        fixed = mean_metric(rows, "fixed", "final_rauc")
        return (
            bgr - uniform,
            bgr - fixed,
            paired_wins(rows, "bgr", "uniform", "final_rauc"),
            paired_wins(rows, "bgr", "fixed", "final_rauc"),
        )

    original_uniform_delta, original_fixed_delta, original_uniform_wins, original_fixed_wins = comparison_detail(original)
    replication_uniform_delta, replication_fixed_delta, replication_uniform_wins, replication_fixed_wins = comparison_detail(
        replication
    )
    pooled_uniform_delta, pooled_fixed_delta, pooled_uniform_wins, pooled_fixed_wins = comparison_detail(pooled)
    passed = (
        original_uniform_delta >= 0.03
        and original_fixed_delta >= 0.03
        and original_uniform_wins[0] >= min_uniform_wins
        and original_fixed_wins[0] >= min_fixed_wins
        and replication_uniform_delta >= 0.03
        and replication_fixed_delta >= 0.03
        and replication_uniform_wins[0] >= min_uniform_wins
        and replication_fixed_wins[0] >= min_fixed_wins
    )
    if not passed:
        return None
    return (
        f"{label} margin replay positive and replicated: "
        f"original dRAUC vs uniform {original_uniform_delta:+.4f} W/L/T={original_uniform_wins}, "
        f"vs fixed {original_fixed_delta:+.4f} W/L/T={original_fixed_wins}; "
        f"held-out dRAUC vs uniform {replication_uniform_delta:+.4f} W/L/T={replication_uniform_wins}, "
        f"vs fixed {replication_fixed_delta:+.4f} W/L/T={replication_fixed_wins}; "
        f"pooled dRAUC vs uniform {pooled_uniform_delta:+.4f} W/L/T={pooled_uniform_wins}, "
        f"vs fixed {pooled_fixed_delta:+.4f} W/L/T={pooled_fixed_wins}"
    )


def openml_positive_details(root: Path) -> list[str]:
    details: list[str] = []
    diabetes = openml_replicated_positive_detail(
        root,
        label="OpenML diabetes",
        original_path=root / "results/openml_diabetes_margin_30seed_v1/per_seed.csv",
        replication_path=root / "results/openml_diabetes_margin_replication_30seed_v1/per_seed.csv",
    )
    if diabetes:
        details.append(diabetes)
    blood = openml_replicated_positive_detail(
        root,
        label="OpenML blood-transfusion",
        original_path=root / "results/openml_numeric_external_fixed_target2_30seed_v1/per_seed.csv",
        replication_path=root / "results/openml_blood_transfusion_margin_replication_30seed_v1/per_seed.csv",
        dataset="blood-transfusion-service-center",
    )
    if blood:
        details.append(blood)
    phoneme = openml_replicated_positive_detail(
        root,
        label="OpenML phoneme",
        original_path=root / "results/openml_numeric_external_fixed_target2_30seed_v1/per_seed.csv",
        replication_path=root / "results/openml_phoneme_margin_replication_30seed_v1/per_seed.csv",
        dataset="phoneme",
    )
    if phoneme:
        details.append(phoneme)
    magic = openml_replicated_positive_detail(
        root,
        label="OpenML MagicTelescope",
        original_path=root / "results/openml_broad_numeric_target2_30seed_v1/per_seed.csv",
        replication_path=root / "results/openml_broad_numeric_target2_replication_30seed_v1/per_seed.csv",
        dataset="MagicTelescope",
    )
    if magic:
        details.append(magic)
    haberman = openml_replicated_positive_detail(
        root,
        label="OpenML haberman",
        original_path=root / "results/openml_broad_numeric_target2_30seed_v1/per_seed.csv",
        replication_path=root / "results/openml_broad_numeric_target2_replication_30seed_v1/per_seed.csv",
        dataset="haberman",
    )
    if haberman:
        details.append(haberman)
    jm1 = openml_replicated_positive_detail(
        root,
        label="OpenML jm1",
        original_path=root / "results/openml_secondary_numeric_target2_30seed_v1/per_seed.csv",
        replication_path=root / "results/openml_secondary_numeric_target2_replication_30seed_v1/per_seed.csv",
        dataset="jm1",
    )
    if jm1:
        details.append(jm1)
    all_binary = openml_macro_suite_detail(
        root,
        label="OpenML all-binary target-1.5",
        original_path=root / "results/openml_all_binary_numeric_target15_30seed_v1_780049/per_seed.csv",
        replication_path=root / "results/openml_all_binary_numeric_target15_replication_30seed_v1_780050/per_seed.csv",
        third_path=root / "results/openml_all_binary_numeric_target15_thirdsplit_30seed_v1_781682_781685/per_seed.csv",
    )
    if all_binary:
        details.append(all_binary)
    return details


def openml_macro_suite_detail(
    root: Path,
    *,
    label: str,
    original_path: Path,
    replication_path: Path,
    third_path: Path | None = None,
) -> str | None:
    del root
    if not original_path.exists() or not replication_path.exists():
        return None

    def suite_stats(rows: list[dict[str, str]]) -> tuple[float, float, float, int, int, int]:
        datasets = sorted({row["dataset"] for row in rows})
        uniform_means = []
        fixed_means = []
        bgr_means = []
        wins_uniform = 0
        wins_fixed = 0
        for dataset in datasets:
            dataset_rows = [row for row in rows if row["dataset"] == dataset]
            uniform = mean_metric(dataset_rows, "uniform", "final_rauc")
            fixed = mean_metric(dataset_rows, "fixed", "final_rauc")
            bgr = mean_metric(dataset_rows, "bgr", "final_rauc")
            uniform_means.append(uniform)
            fixed_means.append(fixed)
            bgr_means.append(bgr)
            wins_uniform += int(bgr > uniform)
            wins_fixed += int(bgr > fixed)
        return mean(uniform_means), mean(fixed_means), mean(bgr_means), wins_uniform, wins_fixed, len(datasets)

    original = read_rows(original_path)
    replication = read_rows(replication_path)
    third = read_rows(third_path) if third_path is not None and third_path.exists() else []
    pooled = original + replication + third
    orig_uniform, orig_fixed, orig_bgr, orig_wins_uniform, orig_wins_fixed, orig_n = suite_stats(original)
    rep_uniform, rep_fixed, rep_bgr, rep_wins_uniform, rep_wins_fixed, rep_n = suite_stats(replication)
    third_summary = ""
    if third:
        third_uniform, third_fixed, third_bgr, third_wins_uniform, third_wins_fixed, third_n = suite_stats(third)
        third_summary = (
            f"; third-block macro BGR {third_bgr:.4f} vs uniform {third_uniform:.4f} "
            f"and fixed {third_fixed:.4f} ({third_wins_uniform}/{third_n}, {third_wins_fixed}/{third_n})"
        )
    pooled_uniform, pooled_fixed, pooled_bgr, pooled_wins_uniform, pooled_wins_fixed, pooled_n = suite_stats(pooled)
    if not (pooled_bgr > pooled_uniform and pooled_bgr > pooled_fixed):
        return None
    return (
        f"{label} broad macro check: original macro BGR {orig_bgr:.4f} vs uniform {orig_uniform:.4f} "
        f"and fixed {orig_fixed:.4f} ({orig_wins_uniform}/{orig_n} dataset means vs uniform, "
        f"{orig_wins_fixed}/{orig_n} vs fixed); held-out macro BGR {rep_bgr:.4f} vs uniform "
        f"{rep_uniform:.4f} and fixed {rep_fixed:.4f} ({rep_wins_uniform}/{rep_n}, "
        f"{rep_wins_fixed}/{rep_n}){third_summary}; pooled macro BGR {pooled_bgr:.4f} vs uniform "
        f"{pooled_uniform:.4f} and fixed {pooled_fixed:.4f} ({pooled_wins_uniform}/{pooled_n}, "
        f"{pooled_wins_fixed}/{pooled_n})"
    )


def independent_benchmark_gate(root: Path) -> GateResult:
    failures: list[str] = []
    positive_details = openml_positive_details(root)

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

    four_maxr10_path = root / "results/minigrid_fourrooms_recovery_probe_maxr10_4seed_v1/summary.csv"
    if four_maxr10_path.exists():
        four_maxr10 = read_rows(four_maxr10_path)
        maxr10_coverage = mean_metric(four_maxr10, "bgr_coverage", "final_rauc")
        maxr10_uniform = mean_metric(four_maxr10, "uniform", "final_rauc")
        maxr10_ablation = mean_metric(four_maxr10, "bgr_uniform_radius", "final_rauc")
        maxr10_r80 = mean_metric(four_maxr10, "bgr_coverage", "final_median_r80")
        maxr10_uniform_r80 = mean_metric(four_maxr10, "uniform", "final_median_r80")
        maxr10_wins = paired_wins(four_maxr10, "bgr_coverage", "uniform", "final_rauc")
        if not (
            maxr10_coverage - maxr10_uniform >= 0.01
            and maxr10_coverage > maxr10_ablation
            and maxr10_wins[0] >= 3
            and not (maxr10_r80 >= 0.99 and maxr10_uniform_r80 >= 0.99)
        ):
            failures.append(
                f"MiniGrid-FourRooms max-r10 negative: BGR-Coverage {maxr10_coverage:.4f}, "
                f"uniform {maxr10_uniform:.4f}, uniform-radius {maxr10_ablation:.4f}, "
                f"W/L/T={maxr10_wins}, median-r80 {maxr10_r80:.4f} vs uniform {maxr10_uniform_r80:.4f}"
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

    dynamic_path = root / "results/minigrid_dynamic_obstacles_recovery_probe_4seed_v1_779232/summary.csv"
    if dynamic_path.exists():
        dynamic = read_rows(dynamic_path)
        dynamic_bgr = mean_metric(dynamic, "bgr", "final_rauc")
        dynamic_coverage = mean_metric(dynamic, "bgr_coverage", "final_rauc")
        dynamic_uniform = mean_metric(dynamic, "uniform", "final_rauc")
        dynamic_failure = mean_metric(dynamic, "failure_only", "final_rauc")
        dynamic_fixed = mean_metric(dynamic, "fixed", "final_rauc")
        dynamic_td = mean_metric(dynamic, "td_loss", "final_rauc")
        dynamic_ablation = mean_metric(dynamic, "bgr_uniform_radius", "final_rauc")
        dynamic_bgr_r80 = mean_metric(dynamic, "bgr", "final_median_r80")
        dynamic_coverage_r80 = mean_metric(dynamic, "bgr_coverage", "final_median_r80")
        dynamic_uniform_r80 = mean_metric(dynamic, "uniform", "final_median_r80")
        dynamic_bgr_wins = paired_wins(dynamic, "bgr", "uniform", "final_rauc")
        dynamic_coverage_wins = paired_wins(dynamic, "bgr_coverage", "uniform", "final_rauc")
        best_dynamic = dynamic_bgr if dynamic_bgr >= dynamic_coverage else dynamic_coverage
        best_dynamic_r80 = dynamic_bgr_r80 if dynamic_bgr >= dynamic_coverage else dynamic_coverage_r80
        best_dynamic_wins = dynamic_bgr_wins if dynamic_bgr >= dynamic_coverage else dynamic_coverage_wins
        if not (
            best_dynamic > dynamic_uniform
            and best_dynamic > max(dynamic_failure, dynamic_fixed, dynamic_td, dynamic_ablation)
            and best_dynamic_wins[0] >= 3
            and best_dynamic_r80 >= dynamic_uniform_r80
            and not (best_dynamic_r80 >= 0.99 and dynamic_uniform_r80 >= 0.99)
            and not (best_dynamic_r80 <= 0.01 and dynamic_uniform_r80 <= 0.01)
        ):
            failures.append(
                f"MiniGrid DynamicObstacles negative: BGR {dynamic_bgr:.4f}, "
                f"BGR-Coverage {dynamic_coverage:.4f}, uniform {dynamic_uniform:.4f}, "
                f"failure-only {dynamic_failure:.4f}, fixed {dynamic_fixed:.4f}, "
                f"TD-loss {dynamic_td:.4f}, uniform-radius {dynamic_ablation:.4f}, "
                f"best-r80 {best_dynamic_r80:.4f} vs uniform {dynamic_uniform_r80:.4f}, "
                f"W/L/T={best_dynamic_wins}"
            )

    dynamic_clean_shield_path = root / "results/minigrid_dynamic_obstacles_clean_shield_probe_4seed_v1_779412/summary.csv"
    if dynamic_clean_shield_path.exists():
        dynamic = read_rows(dynamic_clean_shield_path)
        dynamic_bgr = mean_metric(dynamic, "bgr", "final_rauc")
        dynamic_clean_shield = mean_metric(dynamic, "bgr_clean_shield", "final_rauc")
        dynamic_coverage = mean_metric(dynamic, "bgr_coverage", "final_rauc")
        dynamic_uniform = mean_metric(dynamic, "uniform", "final_rauc")
        dynamic_failure = mean_metric(dynamic, "failure_only", "final_rauc")
        dynamic_fixed = mean_metric(dynamic, "fixed", "final_rauc")
        dynamic_td = mean_metric(dynamic, "td_loss", "final_rauc")
        dynamic_ablation = mean_metric(dynamic, "bgr_uniform_radius", "final_rauc")
        dynamic_bgr_r80 = mean_metric(dynamic, "bgr", "final_median_r80")
        dynamic_clean_shield_r80 = mean_metric(dynamic, "bgr_clean_shield", "final_median_r80")
        dynamic_coverage_r80 = mean_metric(dynamic, "bgr_coverage", "final_median_r80")
        dynamic_uniform_r80 = mean_metric(dynamic, "uniform", "final_median_r80")
        dynamic_bgr_wins = paired_wins(dynamic, "bgr", "uniform", "final_rauc")
        dynamic_clean_shield_wins = paired_wins(dynamic, "bgr_clean_shield", "uniform", "final_rauc")
        dynamic_coverage_wins = paired_wins(dynamic, "bgr_coverage", "uniform", "final_rauc")
        candidates = [
            (dynamic_bgr, dynamic_bgr_r80, dynamic_bgr_wins),
            (dynamic_clean_shield, dynamic_clean_shield_r80, dynamic_clean_shield_wins),
            (dynamic_coverage, dynamic_coverage_r80, dynamic_coverage_wins),
        ]
        best_dynamic, best_dynamic_r80, best_dynamic_wins = max(candidates, key=lambda item: (item[0], item[2][0]))
        if not (
            best_dynamic > dynamic_uniform
            and best_dynamic > max(dynamic_failure, dynamic_fixed, dynamic_td, dynamic_ablation)
            and best_dynamic_wins[0] >= 3
            and best_dynamic_r80 >= dynamic_uniform_r80
            and not (best_dynamic_r80 >= 0.99 and dynamic_uniform_r80 >= 0.99)
            and not (best_dynamic_r80 <= 0.01 and dynamic_uniform_r80 <= 0.01)
        ):
            failures.append(
                f"MiniGrid DynamicObstacles clean-shield negative: BGR {dynamic_bgr:.4f}, "
                f"BGR-Clean-Shield {dynamic_clean_shield:.4f}, BGR-Coverage {dynamic_coverage:.4f}, "
                f"uniform {dynamic_uniform:.4f}, failure-only {dynamic_failure:.4f}, fixed {dynamic_fixed:.4f}, "
                f"TD-loss {dynamic_td:.4f}, uniform-radius {dynamic_ablation:.4f}, "
                f"best-r80 {best_dynamic_r80:.4f} vs uniform {dynamic_uniform_r80:.4f}, "
                f"W/L/T={best_dynamic_wins}"
            )

    deepsea_path = root / "results/bsuite_deepsea_recovery_probe_4seed_v1/summary.csv"
    if deepsea_path.exists():
        deepsea = read_rows(deepsea_path)
        deepsea_bgr = mean_metric(deepsea, "bgr", "final_rauc")
        deepsea_coverage = mean_metric(deepsea, "bgr_coverage", "final_rauc")
        deepsea_uniform = mean_metric(deepsea, "uniform", "final_rauc")
        deepsea_failure = mean_metric(deepsea, "failure_only", "final_rauc")
        deepsea_fixed = mean_metric(deepsea, "fixed", "final_rauc")
        deepsea_td = mean_metric(deepsea, "td_loss", "final_rauc")
        deepsea_ablation = mean_metric(deepsea, "bgr_uniform_radius", "final_rauc")
        deepsea_bgr_r80 = mean_metric(deepsea, "bgr", "final_median_r80")
        deepsea_coverage_r80 = mean_metric(deepsea, "bgr_coverage", "final_median_r80")
        deepsea_uniform_r80 = mean_metric(deepsea, "uniform", "final_median_r80")
        deepsea_bgr_wins = paired_wins(deepsea, "bgr", "uniform", "final_rauc")
        deepsea_coverage_wins = paired_wins(deepsea, "bgr_coverage", "uniform", "final_rauc")
        best_deepsea = deepsea_bgr if deepsea_bgr >= deepsea_coverage else deepsea_coverage
        best_deepsea_r80 = deepsea_bgr_r80 if deepsea_bgr >= deepsea_coverage else deepsea_coverage_r80
        best_deepsea_wins = deepsea_bgr_wins if deepsea_bgr >= deepsea_coverage else deepsea_coverage_wins
        if not (
            best_deepsea > deepsea_uniform
            and best_deepsea > max(deepsea_failure, deepsea_fixed, deepsea_td, deepsea_ablation)
            and best_deepsea_wins[0] >= 3
            and best_deepsea_r80 >= deepsea_uniform_r80
            and not (best_deepsea_r80 >= 0.99 and deepsea_uniform_r80 >= 0.99)
            and not (best_deepsea_r80 <= 0.01 and deepsea_uniform_r80 <= 0.01)
        ):
            failures.append(
                f"bsuite DeepSea negative: BGR {deepsea_bgr:.4f}, BGR-Coverage {deepsea_coverage:.4f}, "
                f"uniform {deepsea_uniform:.4f}, failure-only {deepsea_failure:.4f}, fixed {deepsea_fixed:.4f}, "
                f"TD-loss {deepsea_td:.4f}, uniform-radius {deepsea_ablation:.4f}, "
                f"best-r80 {best_deepsea_r80:.4f} vs uniform {deepsea_uniform_r80:.4f}, W/L/T={best_deepsea_wins}"
            )

    catch_path = root / "results/bsuite_catch_recovery_probe_30seed_v1/summary.csv"
    if catch_path.exists():
        catch = read_rows(catch_path)
        catch_bgr = mean_metric(catch, "bgr", "final_rauc")
        catch_coverage = mean_metric(catch, "bgr_coverage", "final_rauc")
        catch_uniform = mean_metric(catch, "uniform", "final_rauc")
        catch_failure = mean_metric(catch, "failure_only", "final_rauc")
        catch_fixed = mean_metric(catch, "fixed", "final_rauc")
        catch_td = mean_metric(catch, "td_loss", "final_rauc")
        catch_ablation = mean_metric(catch, "bgr_uniform_radius", "final_rauc")
        catch_bgr_r80 = mean_metric(catch, "bgr", "final_median_r80")
        catch_coverage_r80 = mean_metric(catch, "bgr_coverage", "final_median_r80")
        catch_uniform_r80 = mean_metric(catch, "uniform", "final_median_r80")
        catch_bgr_wins = paired_wins(catch, "bgr", "uniform", "final_rauc")
        catch_coverage_wins = paired_wins(catch, "bgr_coverage", "uniform", "final_rauc")
        best_catch = catch_bgr if catch_bgr >= catch_coverage else catch_coverage
        best_catch_r80 = catch_bgr_r80 if catch_bgr >= catch_coverage else catch_coverage_r80
        best_catch_wins = catch_bgr_wins if catch_bgr >= catch_coverage else catch_coverage_wins
        if not (
            best_catch > catch_uniform
            and best_catch > max(catch_failure, catch_fixed, catch_td, catch_ablation)
            and best_catch_wins[0] >= 24
            and best_catch_r80 >= catch_uniform_r80
            and not (best_catch_r80 >= 0.99 and catch_uniform_r80 >= 0.99)
            and not (best_catch_r80 <= 0.01 and catch_uniform_r80 <= 0.01)
        ):
            failures.append(
                f"bsuite Catch 30-seed negative: BGR {catch_bgr:.4f}, "
                f"BGR-Coverage {catch_coverage:.4f}, uniform {catch_uniform:.4f}, "
                f"failure-only {catch_failure:.4f}, fixed {catch_fixed:.4f}, "
                f"TD-loss {catch_td:.4f}, uniform-radius {catch_ablation:.4f}, "
                f"best-r80 {best_catch_r80:.4f} vs uniform {catch_uniform_r80:.4f}, "
                f"W/L/T={best_catch_wins}"
            )

    mountaincar_path = root / "results/bsuite_mountaincar_recovery_probe_4seed_v1/summary.csv"
    if mountaincar_path.exists():
        mountaincar = read_rows(mountaincar_path)
        mountain_bgr = mean_metric(mountaincar, "bgr", "final_rauc")
        mountain_coverage = mean_metric(mountaincar, "bgr_coverage", "final_rauc")
        mountain_uniform = mean_metric(mountaincar, "uniform", "final_rauc")
        mountain_failure = mean_metric(mountaincar, "failure_only", "final_rauc")
        mountain_fixed = mean_metric(mountaincar, "fixed", "final_rauc")
        mountain_td = mean_metric(mountaincar, "td_loss", "final_rauc")
        mountain_ablation = mean_metric(mountaincar, "bgr_uniform_radius", "final_rauc")
        mountain_bgr_r80 = mean_metric(mountaincar, "bgr", "final_median_r80")
        mountain_coverage_r80 = mean_metric(mountaincar, "bgr_coverage", "final_median_r80")
        mountain_uniform_r80 = mean_metric(mountaincar, "uniform", "final_median_r80")
        mountain_bgr_wins = paired_wins(mountaincar, "bgr", "uniform", "final_rauc")
        mountain_coverage_wins = paired_wins(mountaincar, "bgr_coverage", "uniform", "final_rauc")
        best_mountain = mountain_bgr if mountain_bgr >= mountain_coverage else mountain_coverage
        best_mountain_r80 = mountain_bgr_r80 if mountain_bgr >= mountain_coverage else mountain_coverage_r80
        best_mountain_wins = mountain_bgr_wins if mountain_bgr >= mountain_coverage else mountain_coverage_wins
        if not (
            best_mountain - mountain_uniform >= 0.01
            and best_mountain > max(mountain_failure, mountain_fixed, mountain_td, mountain_ablation)
            and best_mountain_wins[0] >= 3
            and best_mountain_r80 >= mountain_uniform_r80
            and not (best_mountain_r80 >= 0.99 and mountain_uniform_r80 >= 0.99)
            and not (best_mountain_r80 <= 0.01 and mountain_uniform_r80 <= 0.01)
        ):
            failures.append(
                f"bsuite MountainCar negative/saturated: BGR {mountain_bgr:.4f}, "
                f"BGR-Coverage {mountain_coverage:.4f}, uniform {mountain_uniform:.4f}, "
                f"failure-only {mountain_failure:.4f}, fixed {mountain_fixed:.4f}, "
                f"TD-loss {mountain_td:.4f}, uniform-radius {mountain_ablation:.4f}, "
                f"best-r80 {best_mountain_r80:.4f} vs uniform {mountain_uniform_r80:.4f}, "
                f"W/L/T={best_mountain_wins}"
            )

    cartpole_path = root / "results/bsuite_cartpole_recovery_probe_4seed_v1/summary.csv"
    if cartpole_path.exists():
        cartpole = read_rows(cartpole_path)
        cartpole_bgr = mean_metric(cartpole, "bgr", "final_rauc")
        cartpole_coverage = mean_metric(cartpole, "bgr_coverage", "final_rauc")
        cartpole_uniform = mean_metric(cartpole, "uniform", "final_rauc")
        cartpole_failure = mean_metric(cartpole, "failure_only", "final_rauc")
        cartpole_fixed = mean_metric(cartpole, "fixed", "final_rauc")
        cartpole_td = mean_metric(cartpole, "td_loss", "final_rauc")
        cartpole_ablation = mean_metric(cartpole, "bgr_uniform_radius", "final_rauc")
        cartpole_bgr_r80 = mean_metric(cartpole, "bgr", "final_median_r80")
        cartpole_coverage_r80 = mean_metric(cartpole, "bgr_coverage", "final_median_r80")
        cartpole_uniform_r80 = mean_metric(cartpole, "uniform", "final_median_r80")
        cartpole_bgr_wins = paired_wins(cartpole, "bgr", "uniform", "final_rauc")
        cartpole_coverage_wins = paired_wins(cartpole, "bgr_coverage", "uniform", "final_rauc")
        best_cartpole = cartpole_bgr if cartpole_bgr >= cartpole_coverage else cartpole_coverage
        best_cartpole_r80 = cartpole_bgr_r80 if cartpole_bgr >= cartpole_coverage else cartpole_coverage_r80
        best_cartpole_wins = cartpole_bgr_wins if cartpole_bgr >= cartpole_coverage else cartpole_coverage_wins
        if not (
            best_cartpole - cartpole_uniform >= 0.01
            and best_cartpole > max(cartpole_failure, cartpole_fixed, cartpole_td, cartpole_ablation)
            and best_cartpole_wins[0] >= 3
            and best_cartpole_r80 >= cartpole_uniform_r80
            and not (best_cartpole_r80 >= 0.99 and cartpole_uniform_r80 >= 0.99)
            and not (best_cartpole_r80 <= 0.01 and cartpole_uniform_r80 <= 0.01)
        ):
            failures.append(
                f"bsuite Cartpole negative: BGR {cartpole_bgr:.4f}, "
                f"BGR-Coverage {cartpole_coverage:.4f}, uniform {cartpole_uniform:.4f}, "
                f"failure-only {cartpole_failure:.4f}, fixed {cartpole_fixed:.4f}, "
                f"TD-loss {cartpole_td:.4f}, uniform-radius {cartpole_ablation:.4f}, "
                f"best-r80 {best_cartpole_r80:.4f} vs uniform {cartpole_uniform_r80:.4f}, "
                f"W/L/T={best_cartpole_wins}"
            )

    minatar_path = root / "results/minatar_breakout_recovery_probe_4seed_v1/summary.csv"
    if minatar_path.exists():
        minatar = read_rows(minatar_path)
        minatar_bgr = mean_metric(minatar, "bgr", "final_rauc")
        minatar_coverage = mean_metric(minatar, "bgr_coverage", "final_rauc")
        minatar_uniform = mean_metric(minatar, "uniform", "final_rauc")
        minatar_failure = mean_metric(minatar, "failure_only", "final_rauc")
        minatar_fixed = mean_metric(minatar, "fixed", "final_rauc")
        minatar_td = mean_metric(minatar, "td_loss", "final_rauc")
        minatar_ablation = mean_metric(minatar, "bgr_uniform_radius", "final_rauc")
        minatar_bgr_r80 = mean_metric(minatar, "bgr", "final_median_r80")
        minatar_coverage_r80 = mean_metric(minatar, "bgr_coverage", "final_median_r80")
        minatar_uniform_r80 = mean_metric(minatar, "uniform", "final_median_r80")
        minatar_bgr_wins = paired_wins(minatar, "bgr", "uniform", "final_rauc")
        minatar_coverage_wins = paired_wins(minatar, "bgr_coverage", "uniform", "final_rauc")
        best_minatar = minatar_bgr if minatar_bgr >= minatar_coverage else minatar_coverage
        best_minatar_r80 = minatar_bgr_r80 if minatar_bgr >= minatar_coverage else minatar_coverage_r80
        best_minatar_wins = minatar_bgr_wins if minatar_bgr >= minatar_coverage else minatar_coverage_wins
        if not (
            best_minatar - minatar_uniform >= 0.01
            and best_minatar > max(minatar_failure, minatar_fixed, minatar_td, minatar_ablation)
            and best_minatar_wins[0] >= 3
            and best_minatar_r80 >= minatar_uniform_r80
            and not (best_minatar_r80 >= 0.99 and minatar_uniform_r80 >= 0.99)
            and not (best_minatar_r80 <= 0.01 and minatar_uniform_r80 <= 0.01)
        ):
            failures.append(
                f"MinAtar Breakout negative/saturated: BGR {minatar_bgr:.4f}, "
                f"BGR-Coverage {minatar_coverage:.4f}, uniform {minatar_uniform:.4f}, "
                f"failure-only {minatar_failure:.4f}, fixed {minatar_fixed:.4f}, "
                f"TD-loss {minatar_td:.4f}, uniform-radius {minatar_ablation:.4f}, "
                f"best-r80 {best_minatar_r80:.4f} vs uniform {minatar_uniform_r80:.4f}, "
                f"W/L/T={best_minatar_wins}"
            )

    asterix_path = root / "results/minatar_asterix_recovery_probe_4seed_v1/summary.csv"
    if asterix_path.exists():
        asterix = read_rows(asterix_path)
        asterix_bgr = mean_metric(asterix, "bgr", "final_rauc")
        asterix_coverage = mean_metric(asterix, "bgr_coverage", "final_rauc")
        asterix_uniform = mean_metric(asterix, "uniform", "final_rauc")
        asterix_failure = mean_metric(asterix, "failure_only", "final_rauc")
        asterix_fixed = mean_metric(asterix, "fixed", "final_rauc")
        asterix_td = mean_metric(asterix, "td_loss", "final_rauc")
        asterix_ablation = mean_metric(asterix, "bgr_uniform_radius", "final_rauc")
        asterix_bgr_r80 = mean_metric(asterix, "bgr", "final_median_r80")
        asterix_coverage_r80 = mean_metric(asterix, "bgr_coverage", "final_median_r80")
        asterix_uniform_r80 = mean_metric(asterix, "uniform", "final_median_r80")
        asterix_bgr_wins = paired_wins(asterix, "bgr", "uniform", "final_rauc")
        asterix_coverage_wins = paired_wins(asterix, "bgr_coverage", "uniform", "final_rauc")
        best_asterix = asterix_bgr if asterix_bgr >= asterix_coverage else asterix_coverage
        best_asterix_r80 = asterix_bgr_r80 if asterix_bgr >= asterix_coverage else asterix_coverage_r80
        best_asterix_wins = asterix_bgr_wins if asterix_bgr >= asterix_coverage else asterix_coverage_wins
        if not (
            best_asterix - asterix_uniform >= 0.01
            and best_asterix > max(asterix_failure, asterix_fixed, asterix_td, asterix_ablation)
            and best_asterix_wins[0] >= 3
            and best_asterix_r80 >= asterix_uniform_r80
            and not (best_asterix_r80 >= 7.99 and asterix_uniform_r80 >= 7.99)
            and not (best_asterix_r80 <= 0.01 and asterix_uniform_r80 <= 0.01)
        ):
            failures.append(
                f"MinAtar Asterix negative: BGR {asterix_bgr:.4f}, "
                f"BGR-Coverage {asterix_coverage:.4f}, uniform {asterix_uniform:.4f}, "
                f"failure-only {asterix_failure:.4f}, fixed {asterix_fixed:.4f}, "
                f"TD-loss {asterix_td:.4f}, uniform-radius {asterix_ablation:.4f}, "
                f"best-r80 {best_asterix_r80:.4f} vs uniform {asterix_uniform_r80:.4f}, "
                f"W/L/T={best_asterix_wins}"
            )

    for label, relative_path, radius_ceiling in [
        ("MinAtar Freeway", "results/minatar_freeway_recovery_probe_4seed_v1/summary.csv", 9.0),
        ("MinAtar Space Invaders", "results/minatar_space_invaders_recovery_probe_4seed_v1/summary.csv", 6.0),
    ]:
        minatar_route_path = root / relative_path
        if not minatar_route_path.exists():
            continue
        minatar_route = read_rows(minatar_route_path)
        route_bgr = mean_metric(minatar_route, "bgr", "final_rauc")
        route_coverage = mean_metric(minatar_route, "bgr_coverage", "final_rauc")
        route_uniform = mean_metric(minatar_route, "uniform", "final_rauc")
        route_failure = mean_metric(minatar_route, "failure_only", "final_rauc")
        route_fixed = mean_metric(minatar_route, "fixed", "final_rauc")
        route_td = mean_metric(minatar_route, "td_loss", "final_rauc")
        route_ablation = mean_metric(minatar_route, "bgr_uniform_radius", "final_rauc")
        route_bgr_r80 = mean_metric(minatar_route, "bgr", "final_median_r80")
        route_coverage_r80 = mean_metric(minatar_route, "bgr_coverage", "final_median_r80")
        route_uniform_r80 = mean_metric(minatar_route, "uniform", "final_median_r80")
        route_bgr_wins = paired_wins(minatar_route, "bgr", "uniform", "final_rauc")
        route_coverage_wins = paired_wins(minatar_route, "bgr_coverage", "uniform", "final_rauc")
        best_route = route_bgr if route_bgr >= route_coverage else route_coverage
        best_route_r80 = route_bgr_r80 if route_bgr >= route_coverage else route_coverage_r80
        best_route_wins = route_bgr_wins if route_bgr >= route_coverage else route_coverage_wins
        if not (
            best_route - route_uniform >= 0.01
            and best_route > max(route_failure, route_fixed, route_td, route_ablation)
            and best_route_wins[0] >= 3
            and best_route_r80 >= route_uniform_r80
            and not (best_route_r80 >= radius_ceiling - 1e-12 and route_uniform_r80 >= radius_ceiling - 1e-12)
            and not (best_route_r80 <= 0.01 and route_uniform_r80 <= 0.01)
        ):
            failures.append(
                f"{label} negative/tied: BGR {route_bgr:.4f}, "
                f"BGR-Coverage {route_coverage:.4f}, uniform {route_uniform:.4f}, "
                f"failure-only {route_failure:.4f}, fixed {route_fixed:.4f}, "
                f"TD-loss {route_td:.4f}, uniform-radius {route_ablation:.4f}, "
                f"best-r80 {best_route_r80:.4f} vs uniform {route_uniform_r80:.4f}, "
                f"W/L/T={best_route_wins}"
            )

    for label, relative_path in [
        ("Gymnasium Taxi-v3 default budget", "results/taxi_recovery_probe_4seed_v1/summary.csv"),
        ("Gymnasium Taxi-v3 hard budget", "results/taxi_recovery_hard_probe_4seed_v1/summary.csv"),
    ]:
        taxi_path = root / relative_path
        if not taxi_path.exists():
            continue
        taxi = read_rows(taxi_path)
        taxi_bgr = mean_metric(taxi, "bgr", "final_rauc")
        taxi_coverage = mean_metric(taxi, "bgr_coverage", "final_rauc")
        taxi_uniform = mean_metric(taxi, "uniform", "final_rauc")
        taxi_failure = mean_metric(taxi, "failure_only", "final_rauc")
        taxi_fixed = mean_metric(taxi, "fixed", "final_rauc")
        taxi_td = mean_metric(taxi, "td_loss", "final_rauc")
        taxi_ablation = mean_metric(taxi, "bgr_uniform_radius", "final_rauc")
        taxi_bgr_r80 = mean_metric(taxi, "bgr", "final_median_r80")
        taxi_coverage_r80 = mean_metric(taxi, "bgr_coverage", "final_median_r80")
        taxi_uniform_r80 = mean_metric(taxi, "uniform", "final_median_r80")
        taxi_bgr_wins = paired_wins(taxi, "bgr", "uniform", "final_rauc")
        taxi_coverage_wins = paired_wins(taxi, "bgr_coverage", "uniform", "final_rauc")
        best_taxi = taxi_bgr if taxi_bgr >= taxi_coverage else taxi_coverage
        best_taxi_r80 = taxi_bgr_r80 if taxi_bgr >= taxi_coverage else taxi_coverage_r80
        best_taxi_wins = taxi_bgr_wins if taxi_bgr >= taxi_coverage else taxi_coverage_wins
        if not (
            best_taxi - taxi_uniform >= 0.01
            and best_taxi > max(taxi_failure, taxi_fixed, taxi_td, taxi_ablation)
            and best_taxi_wins[0] >= 3
            and best_taxi_r80 >= taxi_uniform_r80
            and not (best_taxi_r80 >= 0.99 and taxi_uniform_r80 >= 0.99)
            and not (best_taxi_r80 <= 0.01 and taxi_uniform_r80 <= 0.01)
        ):
            failures.append(
                f"{label} negative: BGR {taxi_bgr:.4f}, "
                f"BGR-Coverage {taxi_coverage:.4f}, uniform {taxi_uniform:.4f}, "
                f"failure-only {taxi_failure:.4f}, fixed {taxi_fixed:.4f}, "
                f"TD-loss {taxi_td:.4f}, uniform-radius {taxi_ablation:.4f}, "
                f"best-r80 {best_taxi_r80:.4f} vs uniform {taxi_uniform_r80:.4f}, "
                f"W/L/T={best_taxi_wins}"
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
        bool(positive_details) or not failures,
        (
            f"{'; '.join(positive_details)}; remaining standard-environment negatives: {'; '.join(failures)}"
            if positive_details and failures
            else "; ".join(positive_details)
            if positive_details
            else "; ".join(failures)
            if failures
            else "at least one independent benchmark clears the promotion criteria"
        ),
    )


def learned_policy_gate(root: Path) -> GateResult:
    inflight_detail = learned_policy_inflight_detail(root)
    hard_occ_summaries = [
        (
            OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_COMPLETE,
            "hard-occlusion 0.80 micro identity-anchored adaptation audit",
        ),
        (
            OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_MICRO_A40_COMPLETE,
            "hard-occlusion 0.80 micro identity-anchored A40 adaptation audit",
        ),
        (
            OPENVLA_HARD_OCCLUSION090_IDENTITY_ANCHOR_STRICT_COMPLETE,
            "hard-occlusion 0.90 strict identity-anchored adaptation audit",
        ),
        (
            OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_COMPLETE,
            "hard-occlusion 0.80 identity-anchored adaptation audit",
        ),
        (
            OPENVLA_HARD_OCCLUSION080_IDENTITY_ANCHOR_A40_COMPLETE,
            "hard-occlusion 0.80 identity-anchored A40 adaptation audit",
        ),
        (
            OPENVLA_HARD_OCCLUSION080_TRANSFER_COMPLETE,
            "hard-occlusion 0.80 transfer audit",
        ),
        (
            OPENVLA_HARD_OCCLUSION_TRANSFER_COMPLETE,
            "hard-occlusion 0.65 transfer audit",
        ),
        (
            OPENVLA_HARD_OCCLUSION_ADAPT_A40_COMPLETE,
            "hard-occlusion 0.65 A40 adaptation audit",
        ),
        (
            OPENVLA_HARD_OCCLUSION_ADAPT_COMPLETE,
            "hard-occlusion 0.65 adaptation audit",
        ),
    ]
    hard_occ_gates = [
        learned_policy_summary_gate(
            root,
            relative_path,
            label=label,
            non_identity_perturbations={"occlusion"},
        )
        for relative_path, label in hard_occ_summaries
        if (root / relative_path).exists()
    ]
    passed_hard_occ = next((gate for gate in hard_occ_gates if gate.passed), None)
    if passed_hard_occ is not None:
        if inflight_detail:
            return GateResult(
                passed_hard_occ.name,
                passed_hard_occ.passed,
                f"{passed_hard_occ.detail}; {inflight_detail}",
            )
        return passed_hard_occ
    if hard_occ_gates:
        gate = hard_occ_gates[0]
        if inflight_detail:
            return GateResult(gate.name, gate.passed, f"{gate.detail}; {inflight_detail}")
        return gate
    if (root / OPENVLA_OCCLUSION_BOTTLENECK_COMPLETE).exists():
        gate = learned_policy_summary_gate(
            root,
            OPENVLA_OCCLUSION_BOTTLENECK_COMPLETE,
            label="latest occlusion-bottleneck audit",
        )
        if inflight_detail:
            return GateResult(gate.name, gate.passed, f"{gate.detail}; {inflight_detail}")
        return gate
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
