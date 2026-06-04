#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Comparison:
    baseline: str
    treatment_mean: float
    baseline_mean: float
    mean_delta: float
    wins: int
    losses: int
    ties: int


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def by_method_seed(rows: list[dict[str, str]]) -> dict[str, dict[int, dict[str, str]]]:
    grouped: dict[str, dict[int, dict[str, str]]] = {}
    for row in rows:
        method = row["method"]
        seed = int(float(row["seed"]))
        grouped.setdefault(method, {})[seed] = row
    return grouped


def compare(
    grouped: dict[str, dict[int, dict[str, str]]],
    *,
    treatment: str,
    baseline: str,
    metric: str,
) -> Comparison:
    treatment_rows = grouped.get(treatment, {})
    baseline_rows = grouped.get(baseline, {})
    seeds = sorted(set(treatment_rows) & set(baseline_rows))
    if not seeds:
        raise ValueError(f"no paired seeds for {treatment} vs {baseline}")
    treatment_values = np.array([float(treatment_rows[seed][metric]) for seed in seeds], dtype=float)
    baseline_values = np.array([float(baseline_rows[seed][metric]) for seed in seeds], dtype=float)
    diffs = treatment_values - baseline_values
    return Comparison(
        baseline=baseline,
        treatment_mean=float(np.mean(treatment_values)),
        baseline_mean=float(np.mean(baseline_values)),
        mean_delta=float(np.mean(diffs)),
        wins=int(np.sum(diffs > 0.0)),
        losses=int(np.sum(diffs < 0.0)),
        ties=int(np.sum(diffs == 0.0)),
    )


def format_comparison(comparison: Comparison) -> str:
    return (
        f"{comparison.baseline}: treatment={comparison.treatment_mean:.4f} "
        f"baseline={comparison.baseline_mean:.4f} delta={comparison.mean_delta:+.4f} "
        f"W/L/T={comparison.wins}/{comparison.losses}/{comparison.ties}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a candidate benchmark is promotable to the paper.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--treatment", default="bgr")
    parser.add_argument("--uniform", default="uniform")
    parser.add_argument("--baselines", default="failure_only,td_loss")
    parser.add_argument("--ablation", default="bgr_uniform_radius")
    parser.add_argument("--metric", default="final_rauc")
    parser.add_argument("--radius-metric", default="final_median_r80")
    parser.add_argument("--min-seeds", type=int, default=30)
    parser.add_argument("--min-wins", type=int, default=24)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--radius-saturation-threshold", type=float, default=0.99)
    parser.add_argument(
        "--allow-radius-saturation",
        action="store_true",
        help="Allow treatment and uniform to both have saturated critical-radius metrics.",
    )
    args = parser.parse_args()

    rows = read_rows(args.summary)
    grouped = by_method_seed(rows)
    treatment_rows = grouped.get(args.treatment, {})
    seeds = sorted(treatment_rows)
    failures: list[str] = []
    messages: list[str] = []

    if len(seeds) < args.min_seeds:
        failures.append(f"seed count {len(seeds)} < required {args.min_seeds}")
    else:
        messages.append(f"seed count {len(seeds)} >= {args.min_seeds}")

    uniform = compare(grouped, treatment=args.treatment, baseline=args.uniform, metric=args.metric)
    messages.append(format_comparison(uniform))
    if uniform.wins < args.min_wins:
        failures.append(f"{args.treatment} wins {uniform.wins}/{len(seeds)} vs {args.uniform}, below {args.min_wins}")
    if uniform.mean_delta < args.min_delta:
        failures.append(f"{args.treatment} mean delta {uniform.mean_delta:+.4f} vs {args.uniform}, below {args.min_delta:+.4f}")

    for baseline in [item.strip() for item in args.baselines.split(",") if item.strip()]:
        comparison = compare(grouped, treatment=args.treatment, baseline=baseline, metric=args.metric)
        messages.append(format_comparison(comparison))
        if comparison.mean_delta <= 0.0:
            failures.append(f"{args.treatment} does not beat {baseline} on {args.metric}")

    if args.ablation in grouped:
        ablation = compare(grouped, treatment=args.treatment, baseline=args.ablation, metric=args.metric)
        messages.append(format_comparison(ablation))
        if ablation.mean_delta <= 0.0:
            failures.append(f"{args.treatment} does not beat state-priority/radius ablation {args.ablation}")

    if args.radius_metric in rows[0]:
        radius_uniform = compare(grouped, treatment=args.treatment, baseline=args.uniform, metric=args.radius_metric)
        messages.append(f"{args.radius_metric} vs {args.uniform}: {format_comparison(radius_uniform)}")
        if radius_uniform.mean_delta < 0.0:
            failures.append(f"{args.radius_metric} contradicts {args.metric}: delta {radius_uniform.mean_delta:+.4f}")
        if (
            not args.allow_radius_saturation
            and radius_uniform.treatment_mean >= args.radius_saturation_threshold
            and radius_uniform.baseline_mean >= args.radius_saturation_threshold
            and abs(radius_uniform.mean_delta) < 1e-12
        ):
            failures.append(
                f"{args.radius_metric} is saturated for both {args.treatment} and {args.uniform} "
                f"(means {radius_uniform.treatment_mean:.4f}, {radius_uniform.baseline_mean:.4f})"
            )

    print(f"candidate={args.summary}")
    for message in messages:
        print(f"[check] {message}")
    if failures:
        print("[decision] DO NOT PROMOTE")
        for failure in failures:
            print(f"[fail] {failure}")
        raise SystemExit(1)
    print("[decision] PROMOTABLE FOR PAPER INTEGRATION")


if __name__ == "__main__":
    main()
