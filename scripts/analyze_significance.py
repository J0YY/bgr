"""Run paired exact sign tests for reported BGR comparisons."""

from __future__ import annotations

import argparse
import csv
import json
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
    Comparison("Synthetic margin 15-seed", "toy_15seed_v1/summary.csv", "bgr", "uniform", "final_rauc"),
    Comparison("Synthetic margin 15-seed", "toy_15seed_v1/summary.csv", "bgr", "uniform", "rauc_aulc"),
    Comparison("Synthetic margin 15-seed", "toy_15seed_v1/summary.csv", "bgr", "fixed", "final_rauc"),
    Comparison("Synthetic margin 15-seed", "toy_15seed_v1/summary.csv", "bgr", "failure_only", "final_rauc"),
    Comparison("Synthetic margin 15-seed", "toy_15seed_v1/summary.csv", "bgr", "plr_loss", "final_rauc"),
    Comparison("Synthetic margin 30-seed", "toy_30seed_v1/summary.csv", "bgr", "uniform", "final_rauc"),
    Comparison("Synthetic margin 30-seed", "toy_30seed_v1/summary.csv", "bgr", "uniform", "rauc_aulc"),
    Comparison("Synthetic margin 30-seed", "toy_30seed_v1/summary.csv", "bgr", "uniform", "final_clean"),
    Comparison("Synthetic margin 30-seed", "toy_30seed_v1/summary.csv", "bgr", "fixed", "final_rauc"),
    Comparison("Synthetic margin 30-seed", "toy_30seed_v1/summary.csv", "bgr", "failure_only", "final_rauc"),
    Comparison("Synthetic margin 30-seed", "toy_30seed_v1/summary.csv", "bgr", "plr_loss", "final_rauc"),
    Comparison("Grid margin 15-seed", "grid_margin_pair_15seed_v1/summary.csv", "bgr", "uniform", "final_rauc"),
    Comparison("Grid margin 15-seed", "grid_margin_pair_15seed_v1/summary.csv", "bgr", "uniform", "rauc_aulc"),
    Comparison("Grid margin 15-seed", "grid_margin_pair_15seed_v1/summary.csv", "bgr", "uniform", "final_clean"),
    Comparison("Grid margin 15-seed", "grid_margin_pair_15seed_v1/summary.csv", "bgr", "uniform", "final_median_r80"),
    Comparison("Grid margin full 15-seed", "grid_margin_full_15seed_v1/summary.csv", "bgr", "fixed", "final_rauc"),
    Comparison("Grid margin full 15-seed", "grid_margin_full_15seed_v1/summary.csv", "bgr", "failure_only", "final_rauc"),
    Comparison("Grid margin full 15-seed", "grid_margin_full_15seed_v1/summary.csv", "bgr", "plr_loss", "final_rauc"),
    Comparison("Grid margin full 15-seed", "grid_margin_full_15seed_v1/summary.csv", "bgr", "fixed", "rauc_aulc"),
    Comparison("Grid margin full 15-seed", "grid_margin_full_15seed_v1/summary.csv", "bgr", "failure_only", "rauc_aulc"),
    Comparison("Grid margin full 15-seed", "grid_margin_full_15seed_v1/summary.csv", "bgr", "plr_loss", "rauc_aulc"),
    Comparison("Grid margin full 30-seed", "grid_margin_full_30seed_v1/summary.csv", "bgr", "uniform", "final_rauc"),
    Comparison("Grid margin full 30-seed", "grid_margin_full_30seed_v1/summary.csv", "bgr", "uniform", "rauc_aulc"),
    Comparison("Grid margin full 30-seed", "grid_margin_full_30seed_v1/summary.csv", "bgr", "uniform", "final_clean"),
    Comparison("Grid margin full 30-seed", "grid_margin_full_30seed_v1/summary.csv", "bgr", "uniform", "final_median_r80"),
    Comparison("Grid margin full 30-seed", "grid_margin_full_30seed_v1/summary.csv", "bgr", "fixed", "final_rauc"),
    Comparison("Grid margin full 30-seed", "grid_margin_full_30seed_v1/summary.csv", "bgr", "failure_only", "final_rauc"),
    Comparison("Grid margin full 30-seed", "grid_margin_full_30seed_v1/summary.csv", "bgr", "plr_loss", "final_rauc"),
    Comparison("Grid margin full 30-seed", "grid_margin_full_30seed_v1/summary.csv", "bgr", "fixed", "rauc_aulc"),
    Comparison("Grid margin full 30-seed", "grid_margin_full_30seed_v1/summary.csv", "bgr", "failure_only", "rauc_aulc"),
    Comparison("Grid margin full 30-seed", "grid_margin_full_30seed_v1/summary.csv", "bgr", "plr_loss", "rauc_aulc"),
    Comparison("Grid margin replication 30-seed", "grid_margin_full_replication_30seed_v1/summary.csv", "bgr", "uniform", "final_rauc"),
    Comparison("Grid margin replication 30-seed", "grid_margin_full_replication_30seed_v1/summary.csv", "bgr", "uniform", "rauc_aulc"),
    Comparison("Grid margin replication 30-seed", "grid_margin_full_replication_30seed_v1/summary.csv", "bgr", "uniform", "final_clean"),
    Comparison("Grid margin ablation 15-seed", "grid_margin_ablation_15seed_v1/summary.csv", "bgr", "bgr_uniform_radius", "final_rauc"),
    Comparison("Grid margin ablation 15-seed", "grid_margin_ablation_15seed_v1/summary.csv", "bgr_uniform_radius", "uniform", "final_rauc"),
    Comparison("Grid margin ablation 15-seed", "grid_margin_ablation_15seed_v1/summary.csv", "bgr", "bgr_no_uncertainty", "final_rauc"),
    Comparison("Grid margin ablation 15-seed", "grid_margin_ablation_15seed_v1/summary.csv", "bgr", "bgr_no_sharpness", "final_rauc"),
    Comparison("Robot suffix full 15-seed", "suffix_full_15seed_v1/summary.csv", "bgr", "clean_ft", "final_rauc"),
    Comparison("Robot suffix full 15-seed", "suffix_full_15seed_v1/summary.csv", "bgr", "fixed", "final_rauc"),
    Comparison("Robot suffix full 15-seed", "suffix_full_15seed_v1/summary.csv", "bgr", "failure_only", "final_rauc"),
    Comparison("Robot suffix full 15-seed", "suffix_full_15seed_v1/summary.csv", "bgr", "loss_priority", "final_rauc"),
    Comparison("Robot suffix full 15-seed", "suffix_full_15seed_v1/summary.csv", "bgr", "uniform", "rauc_aulc"),
    Comparison("Robot suffix coverage 30-seed", "suffix_strategy_coverage_30seed_v1/summary.csv", "bgr_broad", "uniform", "final_clean"),
    Comparison(
        "Robot suffix coverage 30-seed",
        "suffix_strategy_coverage_30seed_v1/summary.csv",
        "bgr_broad",
        "uniform",
        "final_transfer_rauc",
    ),
    Comparison("Robot suffix coverage 30-seed", "suffix_strategy_coverage_30seed_v1/summary.csv", "bgr_broad", "uniform", "rauc_aulc"),
    Comparison("Robot suffix coverage 30-seed", "suffix_strategy_coverage_30seed_v1/summary.csv", "bgr_broad", "uniform", "final_rauc"),
    Comparison("Robot suffix coverage 30-seed", "suffix_strategy_coverage_30seed_v1/summary.csv", "bgr_broad", "uniform", "final_median_r80"),
    Comparison("Robot suffix replication 30-seed", "suffix_strategy_coverage_replication_30seed_v1/summary.csv", "bgr_broad", "uniform", "final_clean"),
    Comparison("Robot suffix replication 30-seed", "suffix_strategy_coverage_replication_30seed_v1/summary.csv", "bgr_broad", "uniform", "final_rauc"),
    Comparison(
        "Robot suffix replication 30-seed",
        "suffix_strategy_coverage_replication_30seed_v1/summary.csv",
        "bgr_broad",
        "uniform",
        "final_transfer_rauc",
    ),
    Comparison("Robot suffix replication 30-seed", "suffix_strategy_coverage_replication_30seed_v1/summary.csv", "bgr_broad", "uniform", "rauc_aulc"),
    Comparison(
        "Robot suffix coverage-full 30-seed",
        "suffix_coverage_full_30seed_v1/summary.csv",
        "bgr_broad",
        "clean_ft",
        "final_rauc",
    ),
    Comparison(
        "Robot suffix coverage-full 30-seed",
        "suffix_coverage_full_30seed_v1/summary.csv",
        "bgr_broad",
        "fixed",
        "final_rauc",
    ),
    Comparison(
        "Robot suffix coverage-full 30-seed",
        "suffix_coverage_full_30seed_v1/summary.csv",
        "bgr_broad",
        "failure_only",
        "final_rauc",
    ),
    Comparison(
        "Robot suffix coverage-full 30-seed",
        "suffix_coverage_full_30seed_v1/summary.csv",
        "bgr_broad",
        "loss_priority",
        "final_rauc",
    ),
    Comparison(
        "Robot suffix coverage-full 30-seed",
        "suffix_coverage_full_30seed_v1/summary.csv",
        "bgr_broad",
        "uniform",
        "final_rauc",
    ),
    Comparison(
        "Robot suffix coverage-full 30-seed",
        "suffix_coverage_full_30seed_v1/summary.csv",
        "bgr_broad",
        "uniform",
        "final_transfer_rauc",
    ),
    Comparison(
        "Robot suffix coverage-full 30-seed",
        "suffix_coverage_full_30seed_v1/summary.csv",
        "bgr_broad",
        "uniform",
        "rauc_aulc",
    ),
    Comparison(
        "Robot suffix coverage-full replication 30-seed",
        "suffix_coverage_full_replication_30seed_v1/summary.csv",
        "bgr_broad",
        "clean_ft",
        "final_rauc",
    ),
    Comparison(
        "Robot suffix coverage-full replication 30-seed",
        "suffix_coverage_full_replication_30seed_v1/summary.csv",
        "bgr_broad",
        "fixed",
        "final_rauc",
    ),
    Comparison(
        "Robot suffix coverage-full replication 30-seed",
        "suffix_coverage_full_replication_30seed_v1/summary.csv",
        "bgr_broad",
        "failure_only",
        "final_rauc",
    ),
    Comparison(
        "Robot suffix coverage-full replication 30-seed",
        "suffix_coverage_full_replication_30seed_v1/summary.csv",
        "bgr_broad",
        "loss_priority",
        "final_rauc",
    ),
    Comparison(
        "Robot suffix coverage-full replication 30-seed",
        "suffix_coverage_full_replication_30seed_v1/summary.csv",
        "bgr_broad",
        "uniform",
        "final_rauc",
    ),
    Comparison(
        "Robot suffix coverage-full replication 30-seed",
        "suffix_coverage_full_replication_30seed_v1/summary.csv",
        "bgr_broad",
        "uniform",
        "final_transfer_rauc",
    ),
    Comparison(
        "Robot suffix coverage-full replication 30-seed",
        "suffix_coverage_full_replication_30seed_v1/summary.csv",
        "bgr_broad",
        "uniform",
        "rauc_aulc",
    ),
    Comparison("Estimator 15-seed", "estimator_pair_15seed_v1/summary.csv", "active", "uniform", "boundary_hit_rate"),
    Comparison("Estimator 15-seed", "estimator_pair_15seed_v1/summary.csv", "active", "uniform", "r80_mae", "lower"),
    Comparison("Estimator 30-seed", "estimator_pair_30seed_v1/summary.csv", "active", "uniform", "boundary_hit_rate"),
    Comparison("Estimator 30-seed", "estimator_pair_30seed_v1/summary.csv", "active", "uniform", "r80_mae", "lower"),
    Comparison("Estimator 30-seed", "estimator_pair_30seed_v1/summary.csv", "active", "uniform", "rauc_mae", "lower"),
]
OPTIONAL_COMPARISONS: list[Comparison] = []

TARGET_SENSITIVITY_PATH = "grid_margin_target_sensitivity_15seed_v1/summary.csv"
TARGET_SENSITIVITY_BASELINE_PATH = "grid_margin_full_15seed_v1/summary.csv"
TARGET_SENSITIVITY_30_PATH = "grid_margin_target_sensitivity_30seed_v1/summary.csv"
TARGET_SENSITIVITY_30_BASELINE_PATH = "grid_margin_full_30seed_v1/summary.csv"
LEARNING_RATE_SENSITIVITY_PATH = "grid_margin_learning_rate_sensitivity_15seed_v1/summary.csv"
LEARNING_RATE_SENSITIVITY_30_PATH = "grid_margin_learning_rate_sensitivity_30seed_v1/summary.csv"
REGIME_SENSITIVITY_PATH = "grid_margin_regime_sensitivity_15seed_v1/summary.csv"
REGIME_SENSITIVITY_30_PATH = "grid_margin_regime_sensitivity_30seed_v1/summary.csv"
STRESS_SENSITIVITY_PATH = "grid_margin_stress_sensitivity_15seed_v1/summary.csv"
STRESS_SENSITIVITY_30_PATH = "grid_margin_stress_sensitivity_30seed_v1/summary.csv"
GRID_LEARNING_CURVE_PATH = "grid_margin_full_15seed_v1/results.json"


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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    variance = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def t_critical_975(n: int) -> float:
    """Two-sided 95% critical value for a paired mean with n observations."""
    by_df = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if n < 2:
        return 0.0
    return by_df.get(n - 1, 1.960)


def confidence_interval(differences: list[float]) -> tuple[float, float]:
    center = mean(differences)
    half_width = t_critical_975(len(differences)) * standard_error(differences)
    return center - half_width, center + half_width


def sign_test_pvalue(differences: list[float]) -> float:
    nonzero = [diff for diff in differences if diff != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    wins = sum(1 for diff in nonzero if diff > 0.0)
    losses = n - wins
    tail = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n))


def format_pvalue(value: float) -> str:
    return f"{value:.6g}"


def format_pvalue_latex(value: str) -> str:
    p_value = float(value)
    if p_value < 0.0001:
        return "$<0.0001$"
    return f"{p_value:.4f}"


def win_loss_tie_counts(differences: list[float], direction: str) -> tuple[int, int, int]:
    signed = [-diff if direction == "lower" else diff for diff in differences]
    wins = sum(1 for diff in signed if diff > 0.0)
    losses = sum(1 for diff in signed if diff < 0.0)
    ties = len(signed) - wins - losses
    return wins, losses, ties


def result_row(
    benchmark: str,
    condition: str,
    metric: str,
    treatment: str,
    baseline: str,
    differences: list[float],
    direction: str,
) -> dict[str, str]:
    mean_diff = mean(differences)
    ci_low, ci_high = confidence_interval(differences)
    signed_effect = -mean_diff if direction == "lower" else mean_diff
    wins, losses, ties = win_loss_tie_counts(differences, direction)
    return {
        "benchmark": benchmark,
        "condition": condition,
        "metric": metric,
        "treatment": treatment,
        "baseline": baseline,
        "n": str(len(differences)),
        "mean_treatment_minus_baseline": f"{mean_diff:.6f}",
        "paired_se": f"{standard_error(differences):.6f}",
        "paired_ci95_low": f"{ci_low:.6f}",
        "paired_ci95_high": f"{ci_high:.6f}",
        "paired_wins": str(wins),
        "paired_losses": str(losses),
        "paired_ties": str(ties),
        "two_sided_sign_test_p": format_pvalue(sign_test_pvalue(differences)),
        "direction": direction,
        "supports_treatment": str(signed_effect > 0.0).lower(),
    }


def analyze(results_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for comparison in COMPARISONS:
        rows.append(analyze_comparison(results_dir, comparison))
    for comparison in OPTIONAL_COMPARISONS:
        path = results_dir / comparison.summary_path
        if path.exists():
            rows.append(analyze_comparison(results_dir, comparison))
    rows.extend(analyze_target_sensitivity(results_dir))
    rows.extend(analyze_target_sensitivity_30(results_dir))
    rows.extend(analyze_learning_rate_sensitivity(results_dir))
    rows.extend(analyze_learning_rate_sensitivity_30(results_dir))
    rows.extend(analyze_regime_sensitivity(results_dir))
    rows.extend(analyze_regime_sensitivity_30(results_dir))
    rows.extend(analyze_stress_sensitivity(results_dir))
    rows.extend(analyze_stress_sensitivity_30(results_dir))
    rows.extend(analyze_grid_learning_curve(results_dir))
    return rows


def analyze_comparison(results_dir: Path, comparison: Comparison) -> dict[str, str]:
    data = read_summary(results_dir / comparison.summary_path)
    treatment = data[comparison.treatment]
    baseline = data[comparison.baseline]
    seeds = sorted(set(treatment) & set(baseline))
    differences = [
        treatment[seed][comparison.metric] - baseline[seed][comparison.metric]
        for seed in seeds
    ]
    return result_row(
        comparison.benchmark,
        "",
        comparison.metric,
        comparison.treatment,
        comparison.baseline,
        differences,
        comparison.direction,
    )


def analyze_target_sensitivity(results_dir: Path) -> list[dict[str, str]]:
    sensitivity_path = results_dir / TARGET_SENSITIVITY_PATH
    baseline_path = results_dir / TARGET_SENSITIVITY_BASELINE_PATH
    if not sensitivity_path.exists() or not baseline_path.exists():
        return []

    sensitivity_rows = read_rows(sensitivity_path)
    baseline = read_summary(baseline_path)["uniform"]
    out: list[dict[str, str]] = []
    target_margins = sorted({float(row["target_margin"]) for row in sensitivity_rows})
    for target_margin in target_margins:
        treatment = {
            int(row["seed"]): row
            for row in sensitivity_rows
            if float(row["target_margin"]) == target_margin
        }
        seeds = sorted(set(treatment) & set(baseline))
        for metric in ["final_rauc", "rauc_aulc"]:
            differences = [
                float(treatment[seed][metric]) - baseline[seed][metric]
                for seed in seeds
            ]
            out.append(
                result_row(
                    "Grid margin target sensitivity 15-seed",
                    f"target_margin={target_margin:.2f}",
                    metric,
                    f"bgr_target_{target_margin:.2f}",
                    "uniform",
                    differences,
                    "higher",
                )
            )
    return out


def analyze_target_sensitivity_30(results_dir: Path) -> list[dict[str, str]]:
    sensitivity_path = results_dir / TARGET_SENSITIVITY_30_PATH
    baseline_path = results_dir / TARGET_SENSITIVITY_30_BASELINE_PATH
    if not sensitivity_path.exists() or not baseline_path.exists():
        return []

    sensitivity_rows = read_rows(sensitivity_path)
    baseline = read_summary(baseline_path)["uniform"]
    out: list[dict[str, str]] = []
    target_margins = sorted({float(row["target_margin"]) for row in sensitivity_rows})
    for target_margin in target_margins:
        treatment = {
            int(row["seed"]): row
            for row in sensitivity_rows
            if float(row["target_margin"]) == target_margin
        }
        seeds = sorted(set(treatment) & set(baseline))
        for metric in ["final_rauc", "rauc_aulc", "final_clean"]:
            differences = [
                float(treatment[seed][metric]) - baseline[seed][metric]
                for seed in seeds
            ]
            out.append(
                result_row(
                    "Grid margin target sensitivity 30-seed",
                    f"target_margin={target_margin:.2f}",
                    metric,
                    f"bgr_target_{target_margin:.2f}",
                    "uniform",
                    differences,
                    "higher",
                )
            )
    return out


def analyze_learning_rate_sensitivity(results_dir: Path) -> list[dict[str, str]]:
    return analyze_learning_rate_sensitivity_path(
        results_dir / LEARNING_RATE_SENSITIVITY_PATH,
        "Grid margin learning-rate sensitivity 15-seed",
    )


def analyze_learning_rate_sensitivity_30(results_dir: Path) -> list[dict[str, str]]:
    return analyze_learning_rate_sensitivity_path(
        results_dir / LEARNING_RATE_SENSITIVITY_30_PATH,
        "Grid margin learning-rate sensitivity 30-seed",
    )


def analyze_learning_rate_sensitivity_path(path: Path, benchmark: str) -> list[dict[str, str]]:
    if not path.exists():
        return []

    rows = read_rows(path)
    out: list[dict[str, str]] = []
    learning_rates = sorted({float(row["learning_rate"]) for row in rows})
    for learning_rate in learning_rates:
        bgr = {
            int(row["seed"]): row
            for row in rows
            if float(row["learning_rate"]) == learning_rate and row["method"] == "bgr"
        }
        uniform = {
            int(row["seed"]): row
            for row in rows
            if float(row["learning_rate"]) == learning_rate and row["method"] == "uniform"
        }
        seeds = sorted(set(bgr) & set(uniform))
        for metric in ["final_clean", "final_rauc", "final_median_r80", "rauc_aulc"]:
            differences = [
                float(bgr[seed][metric]) - float(uniform[seed][metric])
                for seed in seeds
            ]
            out.append(
                result_row(
                    benchmark,
                    f"learning_rate={learning_rate:.3f}",
                    metric,
                    "bgr",
                    "uniform",
                    differences,
                    "higher",
                )
            )
    return out


def analyze_regime_sensitivity(results_dir: Path) -> list[dict[str, str]]:
    return analyze_regime_sensitivity_path(
        results_dir / REGIME_SENSITIVITY_PATH,
        "Grid margin regime sensitivity 15-seed",
    )


def analyze_regime_sensitivity_30(results_dir: Path) -> list[dict[str, str]]:
    return analyze_regime_sensitivity_path(
        results_dir / REGIME_SENSITIVITY_30_PATH,
        "Grid margin regime sensitivity 30-seed",
    )


def analyze_regime_sensitivity_path(path: Path, benchmark: str) -> list[dict[str, str]]:
    if not path.exists():
        return []

    rows = read_rows(path)
    out: list[dict[str, str]] = []
    regimes = sorted({row["regime"] for row in rows})
    for regime in regimes:
        bgr = {int(row["seed"]): row for row in rows if row["regime"] == regime and row["method"] == "bgr"}
        uniform = {int(row["seed"]): row for row in rows if row["regime"] == regime and row["method"] == "uniform"}
        seeds = sorted(set(bgr) & set(uniform))
        for metric in ["final_rauc", "rauc_aulc"]:
            differences = [float(bgr[seed][metric]) - float(uniform[seed][metric]) for seed in seeds]
            out.append(
                result_row(
                    benchmark,
                    f"regime={regime}",
                    metric,
                    "bgr",
                    "uniform",
                    differences,
                    "higher",
                )
            )
    return out


def analyze_stress_sensitivity(results_dir: Path) -> list[dict[str, str]]:
    return analyze_stress_sensitivity_path(
        results_dir / STRESS_SENSITIVITY_PATH,
        "Grid margin stress sensitivity 15-seed",
    )


def analyze_stress_sensitivity_30(results_dir: Path) -> list[dict[str, str]]:
    return analyze_stress_sensitivity_path(
        results_dir / STRESS_SENSITIVITY_30_PATH,
        "Grid margin stress sensitivity 30-seed",
    )


def analyze_stress_sensitivity_path(path: Path, benchmark: str) -> list[dict[str, str]]:
    if not path.exists():
        return []

    rows = read_rows(path)
    out: list[dict[str, str]] = []
    stress_cases = sorted({row["stress_case"] for row in rows})
    for stress_case in stress_cases:
        bgr = {int(row["seed"]): row for row in rows if row["stress_case"] == stress_case and row["method"] == "bgr"}
        uniform = {
            int(row["seed"]): row
            for row in rows
            if row["stress_case"] == stress_case and row["method"] == "uniform"
        }
        seeds = sorted(set(bgr) & set(uniform))
        for metric in ["final_rauc", "rauc_aulc"]:
            differences = [float(bgr[seed][metric]) - float(uniform[seed][metric]) for seed in seeds]
            out.append(
                result_row(
                    benchmark,
                    f"stress_case={stress_case}",
                    metric,
                    "bgr",
                    "uniform",
                    differences,
                    "higher",
                )
            )
    return out


def analyze_grid_learning_curve(results_dir: Path) -> list[dict[str, str]]:
    path = results_dir / GRID_LEARNING_CURVE_PATH
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    by_method: dict[str, dict[int, dict]] = {}
    for result in payload["results"]:
        by_method.setdefault(str(result["method"]), {})[int(result["seed"])] = result
    if "bgr" not in by_method or "uniform" not in by_method:
        return []

    seeds = sorted(set(by_method["bgr"]) & set(by_method["uniform"]))
    steps = [float(point["step"]) for point in by_method["bgr"][seeds[0]]["history"]]
    out: list[dict[str, str]] = []
    for step_idx, step in enumerate(steps):
        if step == 0:
            continue
        differences = [
            float(by_method["bgr"][seed]["history"][step_idx]["rauc"])
            - float(by_method["uniform"][seed]["history"][step_idx]["rauc"])
            for seed in seeds
        ]
        out.append(
            result_row(
                "Grid margin learning curve 15-seed",
                f"step={int(step)}",
                "rauc",
                "bgr",
                "uniform",
                differences,
                "higher",
            )
        )
    return out


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict[str, str]], path: Path) -> None:
    table_keys = [
        ("Synthetic margin 15-seed", "final_rauc", "uniform"),
        ("Grid margin full 30-seed", "final_rauc", "uniform"),
        ("Robot suffix coverage-full 30-seed", "final_rauc", "clean_ft"),
        ("Robot suffix coverage-full 30-seed", "final_rauc", "fixed"),
        ("Robot suffix coverage-full 30-seed", "final_rauc", "failure_only"),
        ("Robot suffix coverage-full 30-seed", "final_rauc", "loss_priority"),
        ("Robot suffix coverage-full 30-seed", "final_rauc", "uniform"),
        ("Robot suffix coverage-full 30-seed", "final_transfer_rauc", "uniform"),
        ("Robot suffix coverage-full 30-seed", "rauc_aulc", "uniform"),
        ("Estimator 15-seed", "boundary_hit_rate", "uniform"),
    ]
    available = {(row["benchmark"], row["metric"], row["baseline"]): row for row in rows}
    if ("Grid margin full 30-seed", "final_rauc", "uniform") not in available:
        table_keys = [
            key if key[0] != "Grid margin full 30-seed" else ("Grid margin 15-seed", key[1], key[2])
            for key in table_keys
        ]
    if not any(key[0] == "Robot suffix coverage-full 30-seed" and key in available for key in table_keys):
        table_keys = [
            key if key[0] != "Robot suffix coverage-full 30-seed" else ("Robot suffix coverage 30-seed", key[1], key[2])
            for key in table_keys
            if key[0] != "Robot suffix coverage-full 30-seed"
            or key in {("Robot suffix coverage-full 30-seed", "final_rauc", "uniform")}
        ]
    selected = [available[key] for key in table_keys if key in available]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{l l r r r r r}\\hline\n")
        handle.write("Comparison & Metric & $n$ & Mean diff. & 95\\% CI & W/L/T & $p$ \\\\ \\hline\n")
        for row in selected:
            ci = f"[{float(row['paired_ci95_low']):.4f}, {float(row['paired_ci95_high']):.4f}]"
            wins = f"{row['paired_wins']}/{row['paired_losses']}/{row['paired_ties']}"
            comparison = latex_comparison_label(row)
            handle.write(
                f"{comparison} & {latex_metric_label(row['metric'])} & "
                f"{row['n']} & {float(row['mean_treatment_minus_baseline']):.4f} & "
                f"{ci} & {wins} & "
                f"{format_pvalue_latex(row['two_sided_sign_test_p'])} \\\\\n"
            )
        handle.write("\\hline\n\\end{tabular}\n")


def latex_comparison_label(row: dict[str, str]) -> str:
    if row["benchmark"] == "Synthetic margin 15-seed":
        return "Synthetic vs uniform"
    if row["benchmark"] in {"Grid margin 15-seed", "Grid margin full 30-seed"}:
        return "Grid vs uniform"
    if row["benchmark"].startswith("Robot suffix"):
        baselines = {
            "clean_ft": "clean-only",
            "fixed": "fixed",
            "failure_only": "failure-only",
            "loss_priority": "loss-priority",
            "uniform": "uniform",
        }
        metric_prefixes = {
            "final_rauc": "Suffix RAUC",
            "final_transfer_rauc": "Suffix transfer",
            "rauc_aulc": "Suffix AULC",
        }
        prefix = metric_prefixes.get(row["metric"], "Suffix")
        return f"{prefix} vs {baselines.get(row['baseline'], row['baseline'])}"
    if row["benchmark"] == "Estimator 15-seed":
        return "Estimator vs uniform"
    return row["benchmark"].replace("_", "\\_")


def latex_metric_label(metric: str) -> str:
    labels = {
        "final_rauc": "final RAUC",
        "final_transfer_rauc": "transfer RAUC",
        "rauc_aulc": "RAUC AULC",
        "boundary_hit_rate": "boundary hit",
    }
    return labels.get(metric, metric.replace("_", " "))


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
