#!/usr/bin/env python3
"""Analyze original/replicated OpenML margin-replay suites.

The experiment runner writes per-seed rows only after a run completes. This
helper keeps the promotion readout deterministic: it compares BGR with uniform
and fixed-radius replay by dataset, target radius, and seed split, then reports
pooled paired differences across original and held-out runs.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METHODS = ("uniform", "fixed", "bgr")


@dataclass(frozen=True, slots=True)
class Comparison:
    treatment_mean: float
    baseline_mean: float
    delta: float
    wins: int
    losses: int
    ties: int
    n: int


def read_per_seed(path: Path, *, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"dataset", "target_radius", "method", "seed", "final_rauc"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    for row in rows:
        row["split"] = split
    return rows


def group_values(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[int, float]]:
    grouped: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], f"{float(row['target_radius']):.4f}", row["method"])
        grouped[key][int(float(row["seed"]))] = float(row["final_rauc"])
    return grouped


def compare(
    grouped: dict[tuple[str, str, str], dict[int, float]],
    *,
    dataset: str,
    target: str,
    baseline: str,
) -> Comparison:
    bgr_values = grouped.get((dataset, target, "bgr"), {})
    baseline_values = grouped.get((dataset, target, baseline), {})
    seeds = sorted(set(bgr_values) & set(baseline_values))
    if not seeds:
        raise ValueError(f"no paired seeds for {dataset} target {target} bgr vs {baseline}")
    treatment = np.asarray([bgr_values[seed] for seed in seeds], dtype=float)
    base = np.asarray([baseline_values[seed] for seed in seeds], dtype=float)
    diffs = treatment - base
    return Comparison(
        treatment_mean=float(np.mean(treatment)),
        baseline_mean=float(np.mean(base)),
        delta=float(np.mean(diffs)),
        wins=int(np.sum(diffs > 0.0)),
        losses=int(np.sum(diffs < 0.0)),
        ties=int(np.sum(diffs == 0.0)),
        n=len(seeds),
    )


def method_mean(
    grouped: dict[tuple[str, str, str], dict[int, float]],
    *,
    dataset: str,
    target: str,
    method: str,
) -> float:
    values = grouped.get((dataset, target, method), {})
    if not values:
        raise ValueError(f"missing {method} rows for {dataset} target {target}")
    return float(np.mean(list(values.values())))


def format_comp(comp: Comparison) -> str:
    return (
        f"{comp.treatment_mean:.4f} vs {comp.baseline_mean:.4f} "
        f"({comp.delta:+.4f}, W/L/T={comp.wins}/{comp.losses}/{comp.ties})"
    )


def summarize_split(rows: list[dict[str, str]], *, label: str) -> list[str]:
    grouped = group_values(rows)
    dataset_targets = sorted({(row["dataset"], f"{float(row['target_radius']):.4f}") for row in rows})
    lines = [f"[{label}]"]
    for dataset, target in dataset_targets:
        uniform = compare(grouped, dataset=dataset, target=target, baseline="uniform")
        fixed = compare(grouped, dataset=dataset, target=target, baseline="fixed")
        lines.append(
            f"{dataset} target={target}: vs uniform {format_comp(uniform)}; "
            f"vs fixed {format_comp(fixed)}"
        )
    method_macros = {
        method: float(np.mean([method_mean(grouped, dataset=dataset, target=target, method=method) for dataset, target in dataset_targets]))
        for method in METHODS
    }
    lines.append(
        "macro means: "
        + ", ".join(f"{method}={method_macros[method]:.4f}" for method in METHODS)
    )
    return lines


def summarize_pooled(original_rows: list[dict[str, str]], replication_rows: list[dict[str, str]]) -> list[str]:
    rows = original_rows + replication_rows
    grouped = group_values(rows)
    dataset_targets = sorted({(row["dataset"], f"{float(row['target_radius']):.4f}") for row in rows})
    lines = ["[pooled]"]
    bgr_beats_uniform = 0
    bgr_beats_fixed = 0
    promotable_like: list[str] = []
    for dataset, target in dataset_targets:
        uniform = compare(grouped, dataset=dataset, target=target, baseline="uniform")
        fixed = compare(grouped, dataset=dataset, target=target, baseline="fixed")
        if uniform.delta > 0.0:
            bgr_beats_uniform += 1
        if fixed.delta > 0.0:
            bgr_beats_fixed += 1
        if uniform.delta >= 0.03 and fixed.delta > 0.0 and uniform.wins >= int(0.7 * uniform.n):
            promotable_like.append(dataset)
        lines.append(
            f"{dataset} target={target}: vs uniform {format_comp(uniform)}; "
            f"vs fixed {format_comp(fixed)}"
        )
    method_macros = {
        method: float(np.mean([method_mean(grouped, dataset=dataset, target=target, method=method) for dataset, target in dataset_targets]))
        for method in METHODS
    }
    lines.append(
        "macro means: "
        + ", ".join(f"{method}={method_macros[method]:.4f}" for method in METHODS)
    )
    lines.append(
        f"dataset mean wins: BGR ahead on {bgr_beats_uniform}/{len(dataset_targets)} vs uniform "
        f"and {bgr_beats_fixed}/{len(dataset_targets)} vs fixed"
    )
    lines.append(
        "pooled promotable-like rows by simple screen: "
        + (", ".join(promotable_like) if promotable_like else "none")
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path, help="Original per_seed.csv")
    parser.add_argument("--replication", required=True, type=Path, help="Held-out replication per_seed.csv")
    args = parser.parse_args()

    original_rows = read_per_seed(args.original, split="original")
    replication_rows = read_per_seed(args.replication, split="replication")
    for line in summarize_split(original_rows, label="original"):
        print(line)
    for line in summarize_split(replication_rows, label="replication"):
        print(line)
    for line in summarize_pooled(original_rows, replication_rows):
        print(line)


if __name__ == "__main__":
    main()
