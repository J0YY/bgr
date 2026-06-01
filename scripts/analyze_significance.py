"""Run paired exact sign-flip tests for reported BGR comparisons."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class Comparison:
    benchmark: str
    summary_path: str
    treatment: str
    baseline: str
    metric: str
    direction: str = "higher"


COMPARISONS = [
    Comparison("Synthetic margin", "toy_fast_v3/summary.csv", "bgr", "uniform", "final_rauc"),
    Comparison("Synthetic margin", "toy_fast_v3/summary.csv", "bgr", "uniform", "rauc_aulc"),
    Comparison("Grid margin 15-seed", "grid_margin_pair_15seed_v1/summary.csv", "bgr", "uniform", "final_rauc"),
    Comparison("Grid margin 15-seed", "grid_margin_pair_15seed_v1/summary.csv", "bgr", "uniform", "rauc_aulc"),
    Comparison("Grid margin 15-seed", "grid_margin_pair_15seed_v1/summary.csv", "bgr", "uniform", "final_clean"),
    Comparison("Grid margin 15-seed", "grid_margin_pair_15seed_v1/summary.csv", "bgr", "uniform", "final_median_r80"),
    Comparison("Robot suffix 15-seed", "suffix_strategy_pair_15seed_v1/summary.csv", "bgr_broad", "uniform", "final_clean"),
    Comparison(
        "Robot suffix 15-seed",
        "suffix_strategy_pair_15seed_v1/summary.csv",
        "bgr_broad",
        "uniform",
        "final_transfer_rauc",
    ),
    Comparison("Robot suffix 15-seed", "suffix_strategy_pair_15seed_v1/summary.csv", "bgr_broad", "uniform", "rauc_aulc"),
    Comparison("Robot suffix 15-seed", "suffix_strategy_pair_15seed_v1/summary.csv", "bgr_broad", "uniform", "final_rauc"),
    Comparison("Estimator 15-seed", "estimator_pair_15seed_v1/summary.csv", "active", "uniform", "boundary_hit_rate"),
    Comparison("Estimator 15-seed", "estimator_pair_15seed_v1/summary.csv", "active", "uniform", "r80_mae", "lower"),
]


def read_summary(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    by_method: dict[str, dict[int, dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            seed = int(row["seed"])
            by_method.setdefault(method, {})[seed] = {
                key: float(value)
                for key, value in row.items()
                if key not in {"method", "seed"} and value != ""
            }
    return by_method


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    variance = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def sign_flip_pvalue(differences: list[float]) -> float:
    observed = abs(mean(differences))
    total = 0
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        flipped = [sign * diff for sign, diff in zip(signs, differences)]
        total += 1
        if abs(mean(flipped)) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def analyze(results_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for comparison in COMPARISONS:
        data = read_summary(results_dir / comparison.summary_path)
        treatment = data[comparison.treatment]
        baseline = data[comparison.baseline]
        seeds = sorted(set(treatment) & set(baseline))
        differences = [
            treatment[seed][comparison.metric] - baseline[seed][comparison.metric]
            for seed in seeds
        ]
        if comparison.direction == "lower":
            signed_effect = -mean(differences)
        else:
            signed_effect = mean(differences)
        rows.append(
            {
                "benchmark": comparison.benchmark,
                "metric": comparison.metric,
                "treatment": comparison.treatment,
                "baseline": comparison.baseline,
                "n": str(len(seeds)),
                "mean_treatment_minus_baseline": f"{mean(differences):.6f}",
                "paired_se": f"{standard_error(differences):.6f}",
                "two_sided_sign_flip_p": f"{sign_flip_pvalue(differences):.4f}",
                "direction": comparison.direction,
                "supports_treatment": str(signed_effect > 0.0).lower(),
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict[str, str]], path: Path) -> None:
    selected = [
        row
        for row in rows
        if (row["benchmark"], row["metric"])
        in {
            ("Synthetic margin", "final_rauc"),
            ("Grid margin 15-seed", "final_rauc"),
            ("Robot suffix 15-seed", "rauc_aulc"),
            ("Estimator 15-seed", "boundary_hit_rate"),
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{l l r r r}\\hline\n")
        handle.write("Benchmark & Metric & $n$ & Mean diff. & $p$ \\\\ \\hline\n")
        for row in selected:
            handle.write(
                f"{row['benchmark']} & {row['metric'].replace('_', ' ')} & "
                f"{row['n']} & {row['mean_treatment_minus_baseline']} & "
                f"{row['two_sided_sign_flip_p']} \\\\\n"
            )
        handle.write("\\hline\n\\end{tabular}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-csv", default="paper/figures/significance_tests.csv")
    parser.add_argument("--out-tex", default="paper/figures/significance_table.tex")
    args = parser.parse_args()

    rows = analyze(Path(args.results_dir))
    write_csv(rows, Path(args.out_csv))
    write_latex(rows, Path(args.out_tex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
