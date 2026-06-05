"""Summarize whether the current evidence clears the internal AAAI-readiness gate."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

OPENVLA_NON_IDENTITY_PERTURBATIONS = {"blur", "brightness", "occlusion", "shift"}
OPENVLA_WEIGHTED_AVAILABLE = (
    "results/openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_step50500_lr5em7_identitylora_"
    "imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary_available.csv"
)
OPENVLA_LEGACY_COMPLETE = (
    "results/openvla_oft_perturb_eval_cleanmix_p4096_commonavail_step50500_lr5em7_identitylora_imageaug_"
    "officialtrainstats_prereg_fullgoal10x10_v1/summary.csv"
)


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

    return GateResult(
        "independent/pre-existing benchmark",
        not failures,
        "; ".join(failures) if failures else "at least one independent benchmark clears the promotion criteria",
    )


def learned_policy_gate(root: Path) -> GateResult:
    weighted_path = root / OPENVLA_WEIGHTED_AVAILABLE
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
        if random_episodes < 400:
            random_detail = f"{random_detail} available rows; random shift pending"
        official_gate_impossible = (
            bgr_episodes == 400
            and official_episodes == 400
            and (official_margin < 10 or official_rate_margin < 0.02)
        )
        gate_detail = ""
        if official_gate_impossible:
            gate_detail = "; official gate already impossible; pending random row is ledger completion only"
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
    return GateResult(
        "learned-policy OpenVLA/LIBERO",
        passed,
        (
            f"non-identity successes BGR {bgr_success}/{non_identity}, official {official_success}/{non_identity}, "
            f"random {random_success}/{non_identity}; identity BGR {bgr_identity}/{identity_eps}, "
            f"official {official_identity}/{identity_eps}, random {random_identity}/{identity_eps}; "
            f"margin_vs_best={margin_vs_best}"
        ),
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
    ]
    ready = all(gate.passed for gate in gates)
    for gate in gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"[{status}] {gate.name}: {gate.detail}")
    print(f"[decision] {'READY_FOR_90P_AAAI_CLAIM' if ready else 'NOT_READY_FOR_90P_AAAI_CLAIM'}")
    return 1 if args.require_ready and not ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
