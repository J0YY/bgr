#!/usr/bin/env python3
"""Analyze a recovery-replay screen against the internal promotion gates."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Comparison:
    treatment: str
    baseline: str
    metric: str
    treatment_mean: float
    baseline_mean: float
    delta: float
    wins: int
    losses: int
    ties: int


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def methods(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row["method"] for row in rows})


def seeds_for(rows: list[dict[str, str]], method: str) -> set[int]:
    return {int(row["seed"]) for row in rows if row["method"] == method}


def values_by_seed(rows: list[dict[str, str]], method: str, metric: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in rows:
        if row["method"] == method:
            values[int(row["seed"])] = float(row[metric])
    return values


def compare(rows: list[dict[str, str]], treatment: str, baseline: str, metric: str) -> Comparison | None:
    treatment_values = values_by_seed(rows, treatment, metric)
    baseline_values = values_by_seed(rows, baseline, metric)
    common = sorted(set(treatment_values) & set(baseline_values))
    if not common:
        return None
    treatment_mean = sum(treatment_values[seed] for seed in common) / len(common)
    baseline_mean = sum(baseline_values[seed] for seed in common) / len(common)
    wins = sum(treatment_values[seed] > baseline_values[seed] for seed in common)
    losses = sum(treatment_values[seed] < baseline_values[seed] for seed in common)
    ties = len(common) - wins - losses
    return Comparison(
        treatment=treatment,
        baseline=baseline,
        metric=metric,
        treatment_mean=treatment_mean,
        baseline_mean=baseline_mean,
        delta=treatment_mean - baseline_mean,
        wins=wins,
        losses=losses,
        ties=ties,
    )


def format_comparison(comparison: Comparison | None) -> str:
    if comparison is None:
        return "missing"
    return (
        f"{comparison.treatment_mean:.4f} vs {comparison.baseline_mean:.4f} "
        f"(delta {comparison.delta:+.4f}, W/L/T={comparison.wins}/{comparison.losses}/{comparison.ties})"
    )


def radius_failure(comparison: Comparison | None) -> str | None:
    if comparison is None:
        return None
    if (
        comparison.treatment_mean >= 0.99
        and comparison.baseline_mean >= 0.99
        and abs(comparison.delta) < 1e-12
    ):
        return "radius-ceiling-saturated"
    if (
        comparison.treatment_mean <= 0.01
        and comparison.baseline_mean <= 0.01
        and abs(comparison.delta) < 1e-12
    ):
        return "radius-floor-saturated"
    if comparison.delta < 0.0:
        return "radius-contradiction"
    return None


def analyze_treatment(
    rows: list[dict[str, str]],
    treatment: str,
    *,
    metric: str,
    radius_metric: str,
    uniform: str,
    required_baselines: list[str],
    ablation: str,
    min_uniform_delta: float,
    min_uniform_wins: int,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    seeds = seeds_for(rows, treatment)
    uniform_cmp = compare(rows, treatment, uniform, metric)
    required_cmps = [
        item for baseline in required_baselines if (item := compare(rows, treatment, baseline, metric)) is not None
    ]
    ablation_cmp = compare(rows, treatment, ablation, metric)
    radius_cmp = compare(rows, treatment, uniform, radius_metric)

    if len(seeds) < 4:
        failures.append(f"too-few-seeds:{len(seeds)}")
    if uniform_cmp is None or uniform_cmp.delta < min_uniform_delta or uniform_cmp.wins < min_uniform_wins:
        failures.append("uniform-gate")
    if len(required_cmps) != len(required_baselines) or any(item.delta <= 0.0 for item in required_cmps):
        failures.append("required-baseline")
    if ablation_cmp is None or ablation_cmp.delta <= 0.0:
        failures.append("state-priority-ablation")
    if (radius_reason := radius_failure(radius_cmp)) is not None:
        failures.append(radius_reason)

    print(f"\n## {treatment}")
    print(f"seeds: {len(seeds)}")
    print(f"{metric} vs {uniform}: {format_comparison(uniform_cmp)}")
    for baseline in required_baselines:
        print(f"{metric} vs {baseline}: {format_comparison(compare(rows, treatment, baseline, metric))}")
    print(f"{metric} vs {ablation}: {format_comparison(ablation_cmp)}")
    print(f"{radius_metric} vs {uniform}: {format_comparison(radius_cmp)}")
    print("decision:", "PASS" if not failures else "FAIL")
    if failures:
        print("failure_reasons:", ",".join(failures))
    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--treatments", default="bgr,bgr_coverage")
    parser.add_argument("--required-baselines", default="fixed,failure_only,td_loss")
    parser.add_argument("--uniform", default="uniform")
    parser.add_argument("--ablation", default="bgr_uniform_radius")
    parser.add_argument("--metric", default="final_rauc")
    parser.add_argument("--radius-metric", default="final_median_r80")
    parser.add_argument("--min-uniform-delta", type=float, default=0.01)
    parser.add_argument("--min-uniform-wins", type=int, default=3)
    args = parser.parse_args()

    rows = read_rows(args.summary)
    available = methods(rows)
    print(f"summary: {args.summary}")
    print("methods:", ",".join(available))

    treatments = [item.strip() for item in args.treatments.split(",") if item.strip()]
    required_baselines = [item.strip() for item in args.required_baselines.split(",") if item.strip()]
    decisions = [
        analyze_treatment(
            rows,
            treatment,
            metric=args.metric,
            radius_metric=args.radius_metric,
            uniform=args.uniform,
            required_baselines=required_baselines,
            ablation=args.ablation,
            min_uniform_delta=args.min_uniform_delta,
            min_uniform_wins=args.min_uniform_wins,
        )[0]
        for treatment in treatments
        if treatment in available
    ]
    print("\noverall:", "PASS" if any(decisions) else "FAIL")


if __name__ == "__main__":
    main()
