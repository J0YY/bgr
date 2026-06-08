"""Check headline manuscript claims against generated result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class Claim:
    label: str
    snippet: str
    source: str


@dataclass(frozen=True)
class SignificanceCheck:
    label: str
    benchmark: str
    condition: str
    metric: str
    treatment: str
    baseline: str
    supports_treatment: bool
    wins: int
    losses: int
    ties: int = 0
    max_p: float = 0.001


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_metric(rows: list[dict[str, str]], method: str, metric: str) -> float:
    values = [float(row[metric]) for row in rows if row.get("method") == method]
    if not values:
        raise ValueError(f"No rows for method={method!r}, metric={metric!r}")
    return mean(values)


def one_row(rows: list[dict[str, str]], column: str, value: str) -> dict[str, str]:
    matches = [row for row in rows if row.get(column) == value]
    if len(matches) != 1:
        raise ValueError(f"Expected one {column}={value!r} row, found {len(matches)}")
    return matches[0]


def fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def fmt_signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}"


def fmt_pvalue(value: float) -> str:
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def ratio(successes: int | str, episodes: int | str) -> str:
    return f"{int(successes)}/{int(episodes)}"


def mean_success_rate(
    rows: list[dict[str, str]],
    method: str,
    *,
    exclude_perturbations: set[str] | None = None,
) -> float:
    excluded = exclude_perturbations or set()
    values = [
        int(row["successes"]) / int(row["episodes"]) if "successes" in row and "episodes" in row else float(row["success_rate"])
        for row in rows
        if row.get("method") == method and row.get("perturbation", "") not in excluded
    ]
    if not values:
        raise ValueError(f"No success-rate rows for method={method!r}")
    return mean(values)


def pooled_success_rate(
    row_groups: list[list[dict[str, str]]],
    method: str,
    *,
    exclude_perturbations: set[str] | None = None,
) -> float:
    excluded = exclude_perturbations or set()
    total_successes = 0
    total_episodes = 0
    for rows in row_groups:
        for row in rows:
            if row.get("method") != method or row.get("perturbation", "") in excluded:
                continue
            total_successes += int(row["successes"])
            total_episodes += int(row["episodes"])
    if total_episodes == 0:
        raise ValueError(f"No pooled success-rate rows for method={method!r}")
    return total_successes / total_episodes


def paired_wins(rows: list[dict[str, str]], treatment: str, baseline: str, metric: str) -> tuple[int, int, int]:
    seeds = sorted({int(float(row["seed"])) for row in rows if row.get("method") in {treatment, baseline}})
    wins = losses = ties = 0
    for seed in seeds:
        seed_rows = [row for row in rows if int(float(row["seed"])) == seed]
        treatment_value = mean_metric(seed_rows, treatment, metric)
        baseline_value = mean_metric(seed_rows, baseline, metric)
        delta = treatment_value - baseline_value
        if abs(delta) < 1e-12:
            ties += 1
        elif delta > 0.0:
            wins += 1
        else:
            losses += 1
    return wins, losses, ties


def build_claims(results_dir: Path, figures_dir: Path) -> list[Claim]:
    claims: list[Claim] = []

    estimator = read_csv_rows(figures_dir / "estimator_stats.csv")
    active = one_row(estimator, "method", "Active BGR")
    uniform_est = one_row(estimator, "method", "Uniform")
    claims.append(
        Claim(
            "estimator boundary hit",
            f"{fmt(float(active['hit_rate_mean']), 3)} vs. {fmt(float(uniform_est['hit_rate_mean']), 3)}",
            "paper/figures/estimator_stats.csv",
        )
    )
    claims.append(
        Claim(
            "estimator r80 error",
            f"{fmt(float(active['r80_mae_mean']), 3)} vs. {fmt(float(uniform_est['r80_mae_mean']), 3)}",
            "paper/figures/estimator_stats.csv",
        )
    )

    toy = read_csv_rows(results_dir / "toy_30seed_v1" / "summary.csv")
    claims.extend(
        [
            Claim(
                "toy final RAUC",
                f"{fmt(mean_metric(toy, 'bgr', 'final_rauc'), 4)} vs. {fmt(mean_metric(toy, 'uniform', 'final_rauc'), 4)}",
                "results/toy_30seed_v1/summary.csv",
            ),
            Claim(
                "toy AULC",
                f"{fmt(mean_metric(toy, 'bgr', 'rauc_aulc'), 4)} vs. {fmt(mean_metric(toy, 'uniform', 'rauc_aulc'), 4)}",
                "results/toy_30seed_v1/summary.csv",
            ),
            Claim(
                "toy clean success",
                f"{fmt(mean_metric(toy, 'bgr', 'final_clean'), 4)} vs. {fmt(mean_metric(toy, 'uniform', 'final_clean'), 4)}",
                "results/toy_30seed_v1/summary.csv",
            ),
        ]
    )

    grid_summary = results_dir / "grid_margin_full_30seed_v1" / "summary.csv"
    if not grid_summary.exists():
        grid_summary = results_dir / "grid_margin_full_15seed_v1" / "summary.csv"
    grid = read_csv_rows(grid_summary)
    claims.extend(
        [
            Claim(
                "grid final RAUC",
                f"{fmt(mean_metric(grid, 'bgr', 'final_rauc'), 4)} vs. {fmt(mean_metric(grid, 'uniform', 'final_rauc'), 4)}",
                str(grid_summary.relative_to(results_dir.parent)),
            ),
            Claim(
                "grid AULC",
                f"{fmt(mean_metric(grid, 'bgr', 'rauc_aulc'), 4)} vs. {fmt(mean_metric(grid, 'uniform', 'rauc_aulc'), 4)}",
                str(grid_summary.relative_to(results_dir.parent)),
            ),
            Claim(
                "grid median r80",
                f"{fmt(mean_metric(grid, 'bgr', 'final_median_r80'), 4)} vs. {fmt(mean_metric(grid, 'uniform', 'final_median_r80'), 4)}",
                str(grid_summary.relative_to(results_dir.parent)),
            ),
            Claim(
                "grid clean success",
                f"{fmt(mean_metric(grid, 'bgr', 'final_clean'), 4)} vs. {fmt(mean_metric(grid, 'uniform', 'final_clean'), 4)}",
                str(grid_summary.relative_to(results_dir.parent)),
            ),
            Claim(
                "grid failure-only baseline",
                f"Failure-only replay ({fmt(mean_metric(grid, 'failure_only', 'final_rauc'), 4)})",
                str(grid_summary.relative_to(results_dir.parent)),
            ),
            Claim(
                "grid PLR baseline",
                f"({fmt(mean_metric(grid, 'plr_loss', 'final_rauc'), 4)})",
                str(grid_summary.relative_to(results_dir.parent)),
            ),
            Claim(
                "grid fixed-radius baseline",
                f"({fmt(mean_metric(grid, 'fixed', 'final_rauc'), 4)})",
                str(grid_summary.relative_to(results_dir.parent)),
            ),
        ]
    )
    openml_original_summary = results_dir / "openml_diabetes_margin_30seed_v1" / "summary.csv"
    openml_replication_summary = results_dir / "openml_diabetes_margin_replication_30seed_v1" / "summary.csv"
    openml_original_per_seed = results_dir / "openml_diabetes_margin_30seed_v1" / "per_seed.csv"
    openml_replication_per_seed = results_dir / "openml_diabetes_margin_replication_30seed_v1" / "per_seed.csv"
    openml_original = read_csv_rows(openml_original_summary)
    openml_replication = read_csv_rows(openml_replication_summary)
    openml_original_seeds = read_csv_rows(openml_original_per_seed)
    openml_replication_seeds = read_csv_rows(openml_replication_per_seed)
    pooled_openml = openml_original_seeds + openml_replication_seeds
    original_uniform_wlt = paired_wins(openml_original_seeds, "bgr", "uniform", "final_rauc")
    original_fixed_wlt = paired_wins(openml_original_seeds, "bgr", "fixed", "final_rauc")
    replication_uniform_wlt = paired_wins(openml_replication_seeds, "bgr", "uniform", "final_rauc")
    replication_fixed_wlt = paired_wins(openml_replication_seeds, "bgr", "fixed", "final_rauc")
    if not (
        original_uniform_wlt[0] >= 20
        and original_fixed_wlt[0] >= 18
        and replication_uniform_wlt[0] >= 20
        and replication_fixed_wlt[0] >= 18
    ):
        raise ValueError("Expected OpenML diabetes original and held-out comparisons to have paired support")
    claims.extend(
        [
            Claim(
                "OpenML diabetes original uniform",
                (
                    f"{fmt(mean_metric(openml_original, 'bgr', 'final_rauc_mean'), 4)} versus "
                    f"{fmt(mean_metric(openml_original, 'uniform', 'final_rauc_mean'), 4)}"
                ),
                "results/openml_diabetes_margin_30seed_v1/summary.csv",
            ),
            Claim(
                "OpenML diabetes original uniform WLT",
                (
                    f"gap {fmt_signed(mean_metric(openml_original, 'bgr', 'delta_vs_uniform'), 4)}; "
                    f"W/L/T={original_uniform_wlt[0]}/{original_uniform_wlt[1]}/{original_uniform_wlt[2]}"
                ),
                "results/openml_diabetes_margin_30seed_v1/per_seed.csv",
            ),
            Claim(
                "OpenML diabetes original fixed",
                (
                    f"{fmt(mean_metric(openml_original, 'fixed', 'final_rauc_mean'), 4)} fixed-radius "
                    f"(gap {fmt_signed(mean_metric(openml_original, 'bgr', 'final_rauc_mean') - mean_metric(openml_original, 'fixed', 'final_rauc_mean'), 4)}; "
                    f"W/L/T={original_fixed_wlt[0]}/{original_fixed_wlt[1]}/{original_fixed_wlt[2]})"
                ),
                "results/openml_diabetes_margin_30seed_v1/per_seed.csv",
            ),
            Claim(
                "OpenML diabetes replication uniform",
                (
                    f"{fmt(mean_metric(openml_replication, 'bgr', 'final_rauc_mean'), 4)} versus "
                    f"{fmt(mean_metric(openml_replication, 'uniform', 'final_rauc_mean'), 4)}"
                ),
                "results/openml_diabetes_margin_replication_30seed_v1/summary.csv",
            ),
            Claim(
                "OpenML diabetes replication uniform WLT",
                (
                    f"gap {fmt_signed(mean_metric(openml_replication, 'bgr', 'delta_vs_uniform'), 4)}; "
                    f"W/L/T={replication_uniform_wlt[0]}/{replication_uniform_wlt[1]}/{replication_uniform_wlt[2]}"
                ),
                "results/openml_diabetes_margin_replication_30seed_v1/per_seed.csv",
            ),
            Claim(
                "OpenML diabetes replication fixed",
                (
                    f"{fmt(mean_metric(openml_replication, 'fixed', 'final_rauc_mean'), 4)} fixed-radius "
                    f"(gap {fmt_signed(mean_metric(openml_replication, 'bgr', 'final_rauc_mean') - mean_metric(openml_replication, 'fixed', 'final_rauc_mean'), 4)}; "
                    f"W/L/T={replication_fixed_wlt[0]}/{replication_fixed_wlt[1]}/{replication_fixed_wlt[2]})"
                ),
                "results/openml_diabetes_margin_replication_30seed_v1/per_seed.csv",
            ),
            Claim(
                "OpenML diabetes pooled",
                (
                    f"{fmt(mean_metric(pooled_openml, 'bgr', 'final_rauc'), 4)} versus "
                    f"{fmt(mean_metric(pooled_openml, 'uniform', 'final_rauc'), 4)} uniform and "
                    f"{fmt(mean_metric(pooled_openml, 'fixed', 'final_rauc'), 4)} fixed-radius"
                ),
                "results/openml_diabetes_margin_*_30seed_v1/per_seed.csv",
            ),
        ]
    )
    openml_blood_original_summary = results_dir / "openml_numeric_external_fixed_target2_30seed_v1" / "summary.csv"
    openml_blood_replication_summary = results_dir / "openml_blood_transfusion_margin_replication_30seed_v1" / "summary.csv"
    openml_blood_original_per_seed = results_dir / "openml_numeric_external_fixed_target2_30seed_v1" / "per_seed.csv"
    openml_blood_replication_per_seed = results_dir / "openml_blood_transfusion_margin_replication_30seed_v1" / "per_seed.csv"
    blood_dataset = "blood-transfusion-service-center"
    openml_blood_original = [
        row for row in read_csv_rows(openml_blood_original_summary) if row.get("dataset") == blood_dataset
    ]
    openml_blood_replication = read_csv_rows(openml_blood_replication_summary)
    openml_blood_original_seeds = [
        row for row in read_csv_rows(openml_blood_original_per_seed) if row.get("dataset") == blood_dataset
    ]
    openml_blood_replication_seeds = read_csv_rows(openml_blood_replication_per_seed)
    pooled_blood = openml_blood_original_seeds + openml_blood_replication_seeds
    blood_original_uniform_wlt = paired_wins(openml_blood_original_seeds, "bgr", "uniform", "final_rauc")
    blood_original_fixed_wlt = paired_wins(openml_blood_original_seeds, "bgr", "fixed", "final_rauc")
    blood_replication_uniform_wlt = paired_wins(openml_blood_replication_seeds, "bgr", "uniform", "final_rauc")
    blood_replication_fixed_wlt = paired_wins(openml_blood_replication_seeds, "bgr", "fixed", "final_rauc")
    if not (
        blood_original_uniform_wlt[0] >= 20
        and blood_original_fixed_wlt[0] >= 18
        and blood_replication_uniform_wlt[0] >= 20
        and blood_replication_fixed_wlt[0] >= 18
    ):
        raise ValueError("Expected OpenML blood-transfusion original and held-out comparisons to have paired support")
    claims.extend(
        [
            Claim(
                "OpenML blood original uniform",
                (
                    f"{fmt(mean_metric(openml_blood_original, 'bgr', 'final_rauc_mean'), 4)} versus "
                    f"{fmt(mean_metric(openml_blood_original, 'uniform', 'final_rauc_mean'), 4)}"
                ),
                "results/openml_numeric_external_fixed_target2_30seed_v1/summary.csv",
            ),
            Claim(
                "OpenML blood original uniform WLT",
                (
                    f"gap {fmt_signed(mean_metric(openml_blood_original, 'bgr', 'delta_vs_uniform'), 4)}; "
                    f"W/L/T={blood_original_uniform_wlt[0]}/{blood_original_uniform_wlt[1]}/{blood_original_uniform_wlt[2]}"
                ),
                "results/openml_numeric_external_fixed_target2_30seed_v1/per_seed.csv",
            ),
            Claim(
                "OpenML blood original fixed",
                (
                    f"{fmt(mean_metric(openml_blood_original, 'fixed', 'final_rauc_mean'), 4)} fixed-radius "
                    f"(gap {fmt_signed(mean_metric(openml_blood_original, 'bgr', 'final_rauc_mean') - mean_metric(openml_blood_original, 'fixed', 'final_rauc_mean'), 4)}; "
                    f"W/L/T={blood_original_fixed_wlt[0]}/{blood_original_fixed_wlt[1]}/{blood_original_fixed_wlt[2]})"
                ),
                "results/openml_numeric_external_fixed_target2_30seed_v1/per_seed.csv",
            ),
            Claim(
                "OpenML blood replication uniform",
                (
                    f"{fmt(mean_metric(openml_blood_replication, 'bgr', 'final_rauc_mean'), 4)} versus "
                    f"{fmt(mean_metric(openml_blood_replication, 'uniform', 'final_rauc_mean'), 4)}"
                ),
                "results/openml_blood_transfusion_margin_replication_30seed_v1/summary.csv",
            ),
            Claim(
                "OpenML blood replication uniform WLT",
                (
                    f"gap {fmt_signed(mean_metric(openml_blood_replication, 'bgr', 'delta_vs_uniform'), 4)}; "
                    f"W/L/T={blood_replication_uniform_wlt[0]}/{blood_replication_uniform_wlt[1]}/{blood_replication_uniform_wlt[2]}"
                ),
                "results/openml_blood_transfusion_margin_replication_30seed_v1/per_seed.csv",
            ),
            Claim(
                "OpenML blood replication fixed",
                (
                    f"{fmt(mean_metric(openml_blood_replication, 'fixed', 'final_rauc_mean'), 4)} fixed-radius "
                    f"(gap {fmt_signed(mean_metric(openml_blood_replication, 'bgr', 'final_rauc_mean') - mean_metric(openml_blood_replication, 'fixed', 'final_rauc_mean'), 4)}; "
                    f"W/L/T={blood_replication_fixed_wlt[0]}/{blood_replication_fixed_wlt[1]}/{blood_replication_fixed_wlt[2]})"
                ),
                "results/openml_blood_transfusion_margin_replication_30seed_v1/per_seed.csv",
            ),
            Claim(
                "OpenML blood pooled",
                (
                    f"{fmt(mean_metric(pooled_blood, 'bgr', 'final_rauc'), 4)} versus "
                    f"{fmt(mean_metric(pooled_blood, 'uniform', 'final_rauc'), 4)} uniform and "
                    f"{fmt(mean_metric(pooled_blood, 'fixed', 'final_rauc'), 4)} fixed-radius"
                ),
                "results/openml_blood_transfusion_margin_*_30seed_v1/per_seed.csv",
            ),
        ]
    )
    witness = read_csv_rows(results_dir / "grid_margin_witness_sensitivity_30seed_v1" / "summary.csv")
    witness_exact = one_row(witness, "scenario", "exact")
    witness_sym10 = one_row(witness, "scenario", "symmetric_10")
    witness_sym20 = one_row(witness, "scenario", "symmetric_20")
    exact_valid = float(witness_exact["true_valid_accept_rate_mean"])
    sym10_valid = float(witness_sym10["true_valid_accept_rate_mean"])
    sym20_valid = float(witness_sym20["true_valid_accept_rate_mean"])
    sym10_recall = float(witness_sym10["true_boundary_recall_mean"])
    sym20_recall = float(witness_sym20["true_boundary_recall_mean"])
    if not (
        int(witness_exact["seeds"]) == 30
        and exact_valid == 1.0
        and sym10_recall < 0.95
        and sym20_recall < sym10_recall
        and sym20_valid > 0.99
    ):
        raise ValueError("Expected grid-margin witness diagnostic to show false-negative recall degradation")
    claims.append(
        Claim(
            "grid witness sensitivity diagnostic",
            (
                f"30-seed grid-margin witness diagnostic keeps exact-witness valid accepted samples at "
                f"{fmt(exact_valid, 4)}; symmetric 10\\%/20\\% witness noise preserves valid-accept rates at "
                f"{fmt(sym10_valid, 4)}/{fmt(sym20_valid, 4)} but lowers true-boundary recall to "
                f"{fmt(sym10_recall, 4)}/{fmt(sym20_recall, 4)}"
            ),
            "results/grid_margin_witness_sensitivity_30seed_v1/summary.csv",
        )
    )
    grid_replication = read_csv_rows(results_dir / "grid_margin_full_replication_30seed_v1" / "summary.csv")
    grid_replication_seeds = {
        int(float(row["seed"]))
        for row in grid_replication
        if row.get("method") in {"bgr", "uniform"}
    }
    if grid_replication_seeds != set(range(30, 60)):
        raise ValueError(f"Expected grid replication seeds 30-59, found {sorted(grid_replication_seeds)}")
    grid_replication_wins = sum(
        mean_metric(
            [row for row in grid_replication if int(float(row["seed"])) == seed],
            "bgr",
            "final_rauc",
        )
        > mean_metric(
            [row for row in grid_replication if int(float(row["seed"])) == seed],
            "uniform",
            "final_rauc",
        )
        for seed in grid_replication_seeds
    )
    if grid_replication_wins != 30:
        raise ValueError(f"Expected grid held-out replication to have 30/0 BGR RAUC wins, found {grid_replication_wins}/30")
    claims.append(
        Claim(
            "grid held-out replication RAUC",
            (
                f"A held-out seeds 30--59 replication gives "
                f"{fmt(mean_metric(grid_replication, 'bgr', 'final_rauc'), 4)} "
                f"vs. {fmt(mean_metric(grid_replication, 'uniform', 'final_rauc'), 4)} RAUC"
            ),
            "results/grid_margin_full_replication_30seed_v1/summary.csv",
        )
    )
    grid_pooled = grid + grid_replication
    if paired_wins(grid_pooled, "bgr", "uniform", "final_rauc") != (60, 0, 0):
        raise ValueError("Expected pooled grid original+held-out RAUC comparison to have 60/0 BGR wins")
    claims.append(
        Claim(
            "grid pooled 60-seed RAUC",
            (
                f"pooled original and held-out sweeps give "
                f"{fmt(mean_metric(grid_pooled, 'bgr', 'final_rauc'), 4)} "
                f"vs. {fmt(mean_metric(grid_pooled, 'uniform', 'final_rauc'), 4)} RAUC "
                f"with 60/0 paired wins"
            ),
            "results/grid_margin_full_30seed_v1/summary.csv and grid_margin_full_replication_30seed_v1/summary.csv",
        )
    )

    target = read_csv_rows(figures_dir / "grid_margin_target_sensitivity_stats.csv")
    target_raucs = [float(row["rauc_mean"]) for row in target]
    claims.append(
        Claim(
            "grid target sensitivity",
            (
                f"RAUC {fmt(min(target_raucs), 3)}--{fmt(max(target_raucs), 3)} "
                f"vs. uniform {fmt(mean_metric(grid, 'uniform', 'final_rauc'), 3)}"
            ),
            "paper/figures/grid_margin_target_sensitivity_stats.csv",
        )
    )

    regime = read_csv_rows(figures_dir / "grid_margin_regime_sensitivity_stats.csv")
    regime_bgr_raucs = [float(row["rauc_mean"]) for row in regime if row["method"] == "BGR"]
    regime_uniform_raucs = [float(row["rauc_mean"]) for row in regime if row["method"] == "Uniform"]
    if not regime_bgr_raucs or not regime_uniform_raucs:
        raise ValueError("Grid regime sensitivity stats are missing BGR or uniform rows")
    claims.append(
        Claim(
            "grid regime sensitivity",
            (
                f"obstacle regimes ({fmt(min(regime_bgr_raucs), 3)}--{fmt(max(regime_bgr_raucs), 3)} "
                f"vs. {fmt(mean(regime_uniform_raucs), 3)})"
            ),
            "paper/figures/grid_margin_regime_sensitivity_stats.csv",
        )
    )

    stress = read_csv_rows(figures_dir / "grid_margin_stress_sensitivity_stats.csv")
    stress_bgr_raucs = [float(row["rauc_mean"]) for row in stress if row["method"] == "BGR"]
    stress_uniform_raucs = [float(row["rauc_mean"]) for row in stress if row["method"] == "Uniform"]
    if not stress_bgr_raucs or not stress_uniform_raucs:
        raise ValueError("Grid stress sensitivity stats are missing BGR or uniform rows")
    claims.append(
        Claim(
            "grid stress sensitivity",
            (
                f"geometry stresses ({fmt(min(stress_bgr_raucs), 3)}--{fmt(max(stress_bgr_raucs), 3)} "
                f"vs. {fmt(min(stress_uniform_raucs), 3)}--{fmt(max(stress_uniform_raucs), 3)})"
            ),
            "paper/figures/grid_margin_stress_sensitivity_stats.csv",
        )
    )

    learning = read_csv_rows(figures_dir / "grid_margin_learning_curve_stats.csv")
    post_update_steps = [int(float(row["step"])) for row in learning if int(float(row["step"])) > 0]
    if not post_update_steps:
        raise ValueError("Grid learning-curve stats contain no post-update checkpoints")
    negative = [row for row in learning if int(float(row["step"])) > 0 and float(row["delta_mean"]) <= 0]
    if negative:
        steps = ", ".join(row["step"] for row in negative)
        raise ValueError(f"BGR is not ahead of uniform at grid learning-curve step(s): {steps}")
    claims.append(
        Claim(
            "grid learning-curve span",
            f"steps {min(post_update_steps)}--{max(post_update_steps)}",
            "paper/figures/grid_margin_learning_curve_stats.csv",
        )
    )

    lr_stats = read_csv_rows(figures_dir / "grid_margin_learning_rate_sensitivity_stats.csv")

    def lr_metric(learning_rate: float, method: str, metric: str) -> float:
        row = one_row(
            [item for item in lr_stats if fmt(float(item["learning_rate"]), 3) == fmt(learning_rate, 3)],
            "method",
            method,
        )
        return float(row[metric])

    claims.append(
        Claim(
            "grid learning-rate final RAUC scope",
            (
                f"{fmt(lr_metric(0.015, 'BGR', 'rauc_mean'), 3)} vs. {fmt(lr_metric(0.015, 'Uniform', 'rauc_mean'), 3)}; "
                f"{fmt(lr_metric(0.030, 'BGR', 'rauc_mean'), 3)} vs. {fmt(lr_metric(0.030, 'Uniform', 'rauc_mean'), 3)}"
            ),
            "paper/figures/grid_margin_learning_rate_sensitivity_stats.csv",
        )
    )
    claims.append(
        Claim(
            "grid high learning-rate final RAUC caveat",
            (
                f"high learning rate flips final RAUC to uniform "
                f"({fmt(lr_metric(0.060, 'BGR', 'rauc_mean'), 3)} vs. {fmt(lr_metric(0.060, 'Uniform', 'rauc_mean'), 3)})"
            ),
            "paper/figures/grid_margin_learning_rate_sensitivity_stats.csv",
        )
    )

    ablation = read_csv_rows(results_dir / "grid_margin_ablation_30seed_v1" / "summary.csv")
    ablation_replication = read_csv_rows(results_dir / "grid_margin_ablation_replication_30seed_v1" / "summary.csv")
    claims.extend(
        [
            Claim(
                "grid ablation RAUC drop",
                f"{fmt(mean_metric(ablation, 'bgr', 'final_rauc'), 3)} to {fmt(mean_metric(ablation, 'bgr_uniform_radius', 'final_rauc'), 3)}",
                "results/grid_margin_ablation_30seed_v1/summary.csv",
            ),
            Claim(
                "grid ablation AULC drop",
                f"{fmt(mean_metric(ablation, 'bgr', 'rauc_aulc'), 3)} to {fmt(mean_metric(ablation, 'bgr_uniform_radius', 'rauc_aulc'), 3)}",
                "results/grid_margin_ablation_30seed_v1/summary.csv",
            ),
            Claim(
                "grid ablation uniform RAUC",
                f"{fmt(mean_metric(ablation, 'uniform', 'final_rauc'), 3)} RAUC",
                "results/grid_margin_ablation_30seed_v1/summary.csv",
            ),
            Claim(
                "grid ablation held-out replication RAUC",
                (
                    f"held-out seeds 30--59 replication gives "
                    f"{fmt(mean_metric(ablation_replication, 'bgr', 'final_rauc'), 3)} vs. "
                    f"{fmt(mean_metric(ablation_replication, 'bgr_uniform_radius', 'final_rauc'), 3)} RAUC"
                ),
                "results/grid_margin_ablation_replication_30seed_v1/summary.csv",
            ),
            Claim(
                "grid ablation held-out replication AULC",
                (
                    f"{fmt(mean_metric(ablation_replication, 'bgr', 'rauc_aulc'), 3)} vs. "
                    f"{fmt(mean_metric(ablation_replication, 'bgr_uniform_radius', 'rauc_aulc'), 3)} AULC"
                ),
                "results/grid_margin_ablation_replication_30seed_v1/summary.csv",
            ),
            Claim(
                "grid ablation held-out uniform-radius caveat",
                (
                    f"uniform-radius remains below uniform replay "
                    f"({fmt(mean_metric(ablation_replication, 'bgr_uniform_radius', 'final_rauc'), 3)} vs. "
                    f"{fmt(mean_metric(ablation_replication, 'uniform', 'final_rauc'), 3)} RAUC)"
                ),
                "results/grid_margin_ablation_replication_30seed_v1/summary.csv",
            ),
        ]
    )

    suffix = read_csv_rows(results_dir / "suffix_strategy_coverage_30seed_v1" / "summary.csv")
    suffix_ablation = read_csv_rows(results_dir / "suffix_strategy_ablation_30seed_v1" / "summary.csv")
    claims.extend(
        [
            Claim(
                "suffix final object RAUC",
                f"{fmt(mean_metric(suffix, 'bgr_broad', 'final_rauc'), 4)} vs. {fmt(mean_metric(suffix, 'uniform', 'final_rauc'), 4)}",
                "results/suffix_strategy_coverage_30seed_v1/summary.csv",
            ),
            Claim(
                "suffix clean success",
                f"{fmt(mean_metric(suffix, 'bgr_broad', 'final_clean'), 4)} vs. {fmt(mean_metric(suffix, 'uniform', 'final_clean'), 4)}",
                "results/suffix_strategy_coverage_30seed_v1/summary.csv",
            ),
            Claim(
                "suffix transfer RAUC",
                f"{fmt(mean_metric(suffix, 'bgr_broad', 'final_transfer_rauc'), 4)} vs. {fmt(mean_metric(suffix, 'uniform', 'final_transfer_rauc'), 4)}",
                "results/suffix_strategy_coverage_30seed_v1/summary.csv",
            ),
            Claim(
                "suffix AULC",
                f"{fmt(mean_metric(suffix, 'bgr_broad', 'rauc_aulc'), 4)} vs. {fmt(mean_metric(suffix, 'uniform', 'rauc_aulc'), 4)}",
                "results/suffix_strategy_coverage_30seed_v1/summary.csv",
            ),
            Claim(
                "suffix median r80 caveat",
                f"{fmt(mean_metric(suffix, 'uniform', 'final_median_r80'), 4)} vs. {fmt(mean_metric(suffix, 'bgr_broad', 'final_median_r80'), 4)}",
                "results/suffix_strategy_coverage_30seed_v1/summary.csv",
            ),
            Claim(
                "suffix strategy ablation final object RAUC",
                (
                    f"strategy ablation gives {fmt(mean_metric(suffix_ablation, 'bgr_broad', 'final_rauc'), 4)} "
                    f"vs. {fmt(mean_metric(suffix_ablation, 'bgr_boundary', 'final_rauc'), 4)} "
                    f"and {fmt(mean_metric(suffix_ablation, 'bgr_hard', 'final_rauc'), 4)} final object RAUC"
                ),
                "results/suffix_strategy_ablation_30seed_v1/summary.csv",
            ),
            Claim(
                "suffix strategy hard transfer AULC caveat",
                (
                    f"hard-radius variant leads transfer RAUC and AULC "
                    f"({fmt(mean_metric(suffix_ablation, 'bgr_hard', 'final_transfer_rauc'), 4)}, "
                    f"{fmt(mean_metric(suffix_ablation, 'bgr_hard', 'rauc_aulc'), 4)})"
                ),
                "results/suffix_strategy_ablation_30seed_v1/summary.csv",
            ),
        ]
    )
    suffix_replication = read_csv_rows(results_dir / "suffix_strategy_coverage_replication_30seed_v1" / "summary.csv")
    replication_seeds = {
        int(float(row["seed"]))
        for row in suffix_replication
        if row.get("method") in {"bgr_broad", "uniform"}
    }
    if replication_seeds != set(range(30, 60)):
        raise ValueError(f"Expected suffix replication seeds 30-59, found {sorted(replication_seeds)}")
    replication_wins = sum(
        mean_metric(
            [row for row in suffix_replication if int(float(row["seed"])) == seed],
            "bgr_broad",
            "final_rauc",
        )
        > mean_metric(
            [row for row in suffix_replication if int(float(row["seed"])) == seed],
            "uniform",
            "final_rauc",
        )
        for seed in replication_seeds
    )
    if replication_wins != 30:
        raise ValueError(f"Expected suffix held-out replication to have 30/0 BGR RAUC wins, found {replication_wins}/30")
    claims.append(
        Claim(
            "suffix held-out replication RAUC",
            (
                f"held-out seeds 30--59 replication gives "
                f"{fmt(mean_metric(suffix_replication, 'bgr_broad', 'final_rauc'), 4)} "
                f"vs. {fmt(mean_metric(suffix_replication, 'uniform', 'final_rauc'), 4)} RAUC"
            ),
            "results/suffix_strategy_coverage_replication_30seed_v1/summary.csv",
        )
    )
    suffix_pooled = suffix + suffix_replication
    for metric in ["final_rauc", "final_clean", "final_transfer_rauc", "rauc_aulc"]:
        if paired_wins(suffix_pooled, "bgr_broad", "uniform", metric) != (60, 0, 0):
            raise ValueError(f"Expected pooled suffix {metric} comparison to have 60/0 BGR wins")
    if paired_wins(suffix_pooled, "bgr_broad", "uniform", "final_median_r80") != (1, 59, 0):
        raise ValueError("Expected pooled suffix median-r80 caveat to have 1/59 BGR/uniform wins")
    claims.append(
        Claim(
            "suffix pooled 60-seed object RAUC",
            (
                f"Pooling original and held-out suffix sweeps gives object RAUC "
                f"{fmt(mean_metric(suffix_pooled, 'bgr_broad', 'final_rauc'), 4)} "
                f"vs. {fmt(mean_metric(suffix_pooled, 'uniform', 'final_rauc'), 4)}"
            ),
            "results/suffix_strategy_coverage_30seed_v1/summary.csv and suffix_strategy_coverage_replication_30seed_v1/summary.csv",
        )
    )
    claims.append(
        Claim(
            "suffix pooled 60-seed clean transfer AULC",
            (
                f"clean {fmt(mean_metric(suffix_pooled, 'bgr_broad', 'final_clean'), 4)} "
                f"vs. {fmt(mean_metric(suffix_pooled, 'uniform', 'final_clean'), 4)}, "
                f"transfer RAUC {fmt(mean_metric(suffix_pooled, 'bgr_broad', 'final_transfer_rauc'), 4)} "
                f"vs. {fmt(mean_metric(suffix_pooled, 'uniform', 'final_transfer_rauc'), 4)}, "
                f"and AULC {fmt(mean_metric(suffix_pooled, 'bgr_broad', 'rauc_aulc'), 4)} "
                f"vs. {fmt(mean_metric(suffix_pooled, 'uniform', 'rauc_aulc'), 4)}"
            ),
            "results/suffix_strategy_coverage_30seed_v1/summary.csv and suffix_strategy_coverage_replication_30seed_v1/summary.csv",
        )
    )
    claims.append(
        Claim(
            "suffix pooled median r80 caveat",
            (
                f"uniform remains higher on median $r_{{80}}$ pooled "
                f"({fmt(mean_metric(suffix_pooled, 'uniform', 'final_median_r80'), 4)} "
                f"vs. {fmt(mean_metric(suffix_pooled, 'bgr_broad', 'final_median_r80'), 4)})"
            ),
            "results/suffix_strategy_coverage_30seed_v1/summary.csv and suffix_strategy_coverage_replication_30seed_v1/summary.csv",
        )
    )
    suffix_stress = read_csv_rows(results_dir / "suffix_stress_sensitivity_30seed_v1" / "summary.csv")
    stress_cases = sorted({row["stress_case"] for row in suffix_stress})
    for stress_case in stress_cases:
        case_rows = [row for row in suffix_stress if row["stress_case"] == stress_case]
        for metric in ["final_clean", "final_rauc", "final_transfer_rauc", "rauc_aulc"]:
            wins = paired_wins(case_rows, "bgr_broad", "uniform", metric)
            if wins != (30, 0, 0):
                raise ValueError(f"Expected suffix stress {stress_case} {metric} to have 30/0 BGR wins, found {wins}")
    bgr_stress_raucs = [
        mean_metric([row for row in suffix_stress if row["stress_case"] == stress_case], "bgr_broad", "final_rauc")
        for stress_case in stress_cases
    ]
    uniform_stress_raucs = [
        mean_metric([row for row in suffix_stress if row["stress_case"] == stress_case], "uniform", "final_rauc")
        for stress_case in stress_cases
    ]
    bgr_stress_transfer = [
        mean_metric([row for row in suffix_stress if row["stress_case"] == stress_case], "bgr_broad", "final_transfer_rauc")
        for stress_case in stress_cases
    ]
    uniform_stress_transfer = [
        mean_metric([row for row in suffix_stress if row["stress_case"] == stress_case], "uniform", "final_transfer_rauc")
        for stress_case in stress_cases
    ]
    claims.extend(
        [
            Claim(
                "suffix stress final object RAUC range",
                (
                    f"object RAUC {fmt(min(bgr_stress_raucs), 4)}--{fmt(max(bgr_stress_raucs), 4)} "
                    f"vs. uniform {fmt(min(uniform_stress_raucs), 4)}--{fmt(max(uniform_stress_raucs), 4)}"
                ),
                "results/suffix_stress_sensitivity_30seed_v1/summary.csv",
            ),
            Claim(
                "suffix stress transfer RAUC range",
                (
                    f"transfer RAUC {fmt(min(bgr_stress_transfer), 4)}--{fmt(max(bgr_stress_transfer), 4)} "
                    f"vs. {fmt(min(uniform_stress_transfer), 4)}--{fmt(max(uniform_stress_transfer), 4)}"
                ),
                "results/suffix_stress_sensitivity_30seed_v1/summary.csv",
            ),
        ]
    )

    frozenlake = read_csv_rows(results_dir / "frozenlake_recovery_focused_30seed_v1" / "summary.csv")
    frozenlake_rauc_wins = paired_wins(frozenlake, "bgr", "uniform", "final_rauc")
    frozenlake_clean_wins = paired_wins(frozenlake, "bgr", "uniform", "final_clean")
    if frozenlake_rauc_wins != (14, 16, 0) or frozenlake_clean_wins != (13, 17, 0):
        raise ValueError(
            "Expected FrozenLake diagnostic to remain a non-promoted BGR result "
            f"with RAUC signs {frozenlake_rauc_wins} and clean signs {frozenlake_clean_wins}"
        )
    if not (
        mean_metric(frozenlake, "uniform", "final_median_r80") > mean_metric(frozenlake, "bgr", "final_median_r80")
        and mean_metric(frozenlake, "uniform", "best_rauc") > mean_metric(frozenlake, "bgr", "best_rauc")
        and mean_metric(frozenlake, "failure_only", "final_rauc") > mean_metric(frozenlake, "bgr", "final_rauc")
        and mean_metric(frozenlake, "failure_only", "final_median_r80") > mean_metric(
            frozenlake, "bgr", "final_median_r80"
        )
        and mean_metric(frozenlake, "failure_only", "rauc_aulc") > mean_metric(frozenlake, "bgr", "rauc_aulc")
        and mean_metric(frozenlake, "failure_only", "best_rauc") > mean_metric(frozenlake, "bgr", "best_rauc")
    ):
        raise ValueError("Expected FrozenLake diagnostic to favor uniform/failure-only on limitation metrics")
    claims.append(
        Claim(
            "FrozenLake standard-environment limitation",
            (
                f"FrozenLake8x8 & BGR {fmt(mean_metric(frozenlake, 'bgr', 'final_rauc'), 4)}; "
                f"clean {fmt(mean_metric(frozenlake, 'bgr', 'final_clean'), 4)} & uniform "
                f"{fmt(mean_metric(frozenlake, 'uniform', 'final_rauc'), 4)}; signs "
                f"{frozenlake_rauc_wins[0]}/{frozenlake_rauc_wins[1]}, "
                f"{frozenlake_clean_wins[0]}/{frozenlake_clean_wins[1]} & not promoted"
            ),
            "results/frozenlake_recovery_focused_30seed_v1/summary.csv",
        )
    )

    minigrid = read_csv_rows(results_dir / "minigrid_fourrooms_recovery_probe_midband_4seed_v1" / "summary.csv")
    minigrid_coverage_rauc = mean_metric(minigrid, "bgr_coverage", "final_rauc")
    minigrid_uniform_rauc = mean_metric(minigrid, "uniform", "final_rauc")
    minigrid_failure_rauc = mean_metric(minigrid, "failure_only", "final_rauc")
    minigrid_fixed_rauc = mean_metric(minigrid, "fixed", "final_rauc")
    minigrid_coverage_r80 = mean_metric(minigrid, "bgr_coverage", "final_median_r80")
    minigrid_uniform_r80 = mean_metric(minigrid, "uniform", "final_median_r80")
    if not (
        minigrid_coverage_rauc < minigrid_uniform_rauc
        and minigrid_coverage_rauc < minigrid_failure_rauc
        and minigrid_coverage_rauc < minigrid_fixed_rauc
        and minigrid_coverage_r80 < minigrid_uniform_r80
    ):
        raise ValueError("Expected MiniGrid diagnostic to remain a non-promoted BGR-Coverage result")
    claims.append(
        Claim(
            "MiniGrid official-package limitation",
            f"FourRooms BGR-Coverage {fmt(minigrid_coverage_rauc, 4)}",
            "results/minigrid_fourrooms_recovery_probe_midband_4seed_v1/summary.csv",
        )
    )

    minigrid_mid25 = read_csv_rows(results_dir / "minigrid_fourrooms_recovery_probe_mid2_5_4seed_v1" / "summary.csv")
    minigrid_mid25_bgr_rauc = mean_metric(minigrid_mid25, "bgr", "final_rauc")
    minigrid_mid25_coverage_rauc = mean_metric(minigrid_mid25, "bgr_coverage", "final_rauc")
    minigrid_mid25_uniform_rauc = mean_metric(minigrid_mid25, "uniform", "final_rauc")
    minigrid_mid25_fixed_rauc = mean_metric(minigrid_mid25, "fixed", "final_rauc")
    minigrid_mid25_failure_rauc = mean_metric(minigrid_mid25, "failure_only", "final_rauc")
    minigrid_mid25_bgr_r80 = mean_metric(minigrid_mid25, "bgr", "final_median_r80")
    minigrid_mid25_uniform_r80 = mean_metric(minigrid_mid25, "uniform", "final_median_r80")
    minigrid_mid25_wins = paired_wins(minigrid_mid25, "bgr", "uniform", "final_rauc")
    if not (
        minigrid_mid25_bgr_rauc > minigrid_mid25_uniform_rauc
        and minigrid_mid25_wins == (2, 2, 0)
        and minigrid_mid25_bgr_rauc < minigrid_mid25_fixed_rauc
        and minigrid_mid25_bgr_rauc < minigrid_mid25_failure_rauc
        and minigrid_mid25_bgr_r80 < minigrid_mid25_uniform_r80
        and minigrid_mid25_coverage_rauc < minigrid_mid25_uniform_rauc
    ):
        raise ValueError("Expected MiniGrid mid2-5 diagnostic to remain non-promoted with stronger baselines and r80 contradiction")
    claims.append(
        Claim(
            "MiniGrid mid2-5 limitation",
            (
                f"FourRooms mid2--5/LavaGapS7 & BGR {fmt(minigrid_mid25_bgr_rauc, 4)}; "
                f"BGR-Cov. {fmt(minigrid_mid25_coverage_rauc, 4)}"
            ),
            "results/minigrid_fourrooms_recovery_probe_mid2_5_4seed_v1/summary.csv",
        )
    )

    minigrid_maxr10 = read_csv_rows(results_dir / "minigrid_fourrooms_recovery_probe_maxr10_4seed_v1" / "summary.csv")
    maxr10_coverage_rauc = mean_metric(minigrid_maxr10, "bgr_coverage", "final_rauc")
    maxr10_uniform_rauc = mean_metric(minigrid_maxr10, "uniform", "final_rauc")
    maxr10_coverage_r80 = mean_metric(minigrid_maxr10, "bgr_coverage", "final_median_r80")
    maxr10_uniform_r80 = mean_metric(minigrid_maxr10, "uniform", "final_median_r80")
    maxr10_wins = paired_wins(minigrid_maxr10, "bgr_coverage", "uniform", "final_rauc")
    if not (
        maxr10_coverage_rauc > maxr10_uniform_rauc
        and maxr10_coverage_rauc - maxr10_uniform_rauc < 0.01
        and maxr10_wins == (2, 2, 0)
        and maxr10_coverage_r80 == 1.0
        and maxr10_uniform_r80 == 1.0
    ):
        raise ValueError("Expected MiniGrid max-radius-10 diagnostic to remain saturated and non-promotable")
    claims.append(
        Claim(
            "MiniGrid max-radius-10 limitation",
            "FourRooms radius-10, HandReach-v3, and highway-fast-v0 follow-ups close simple rescue paths",
            "results/minigrid_fourrooms_recovery_probe_maxr10_4seed_v1/summary.csv",
        )
    )

    handreach = read_json(results_dir / "handreach_recovery_calibration_8seed_v1" / "summary.json")
    handreach_clean = float(handreach["clean_success"])
    handreach_decision = str(handreach["decision"])
    if handreach_clean != 0.0 or handreach_decision != "reject-calibration-low-clean-success":
        raise ValueError("Expected HandReach calibration to remain rejected for low clean success")
    claims.append(
        Claim(
            "HandReach-v3 calibration limitation",
            "FourRooms radius-10, HandReach-v3, and highway-fast-v0 follow-ups close simple rescue paths",
            "results/handreach_recovery_calibration_8seed_v1/summary.json",
        )
    )

    highway_lane = read_json(results_dir / "highway_lane_recovery_calibration_12seed_v1" / "summary.json")
    highway_lane_clean = float(highway_lane["clean_success"])
    highway_lane_min = float(highway_lane["min_recovery"])
    highway_lane_max = float(highway_lane["max_recovery"])
    highway_lane_r80 = float(highway_lane["r80"])
    highway_lane_radii = [float(value) for value in highway_lane["radii"]]
    highway_lane_decision = str(highway_lane["decision"])
    if not (
        highway_lane_clean < 0.80
        and highway_lane_decision == "reject-calibration-low-clean-success"
        and abs(highway_lane_r80 - highway_lane_radii[-1]) < 1e-12
    ):
        raise ValueError("Expected highway-fast-v0 lane calibration to remain rejected before method comparison")
    claims.append(
        Claim(
            "highway-fast-v0 lane calibration limitation",
            "FourRooms radius-10, HandReach-v3, and highway-fast-v0 follow-ups close simple rescue paths",
            "results/highway_lane_recovery_calibration_12seed_v1/summary.json",
        )
    )

    doorkey = read_csv_rows(results_dir / "minigrid_doorkey_recovery_probe_4seed_v1" / "summary.csv")
    doorkey_failure_rauc = mean_metric(doorkey, "failure_only", "final_rauc")
    doorkey_uniform_rauc = mean_metric(doorkey, "uniform", "final_rauc")
    doorkey_coverage_rauc = mean_metric(doorkey, "bgr_coverage", "final_rauc")
    doorkey_bgr_rauc = mean_metric(doorkey, "bgr", "final_rauc")
    doorkey_uniform_abs = mean_metric(doorkey, "uniform", "final_abs_r10")
    doorkey_coverage_abs = mean_metric(doorkey, "bgr_coverage", "final_abs_r10")
    doorkey_bgr_abs = mean_metric(doorkey, "bgr", "final_abs_r10")
    if not (
        doorkey_failure_rauc > doorkey_uniform_rauc
        and doorkey_uniform_rauc > doorkey_coverage_rauc
        and doorkey_coverage_rauc > doorkey_bgr_rauc
        and doorkey_coverage_abs < doorkey_uniform_abs
        and doorkey_bgr_abs < doorkey_uniform_abs
    ):
        raise ValueError("Expected MiniGrid-DoorKey diagnostic to remain a non-promoted negative result")
    claims.append(
        Claim(
            "MiniGrid-DoorKey official-package limitation",
            f"DoorKey-6x6 BGR-Coverage {fmt(doorkey_coverage_rauc, 4)}, BGR {fmt(doorkey_bgr_rauc, 4)}",
            "results/minigrid_doorkey_recovery_probe_4seed_v1/summary.csv",
        )
    )

    lavacrossing = read_csv_rows(results_dir / "minigrid_lavacrossing_recovery_probe_4seed_v1" / "summary.csv")
    lavacrossing_uniform_rauc = mean_metric(lavacrossing, "uniform", "final_rauc")
    lavacrossing_coverage_rauc = mean_metric(lavacrossing, "bgr_coverage", "final_rauc")
    lavacrossing_bgr_rauc = mean_metric(lavacrossing, "bgr", "final_rauc")
    lavacrossing_ablation_rauc = mean_metric(lavacrossing, "bgr_uniform_radius", "final_rauc")
    lavacrossing_uniform_abs = mean_metric(lavacrossing, "uniform", "final_abs_r10")
    lavacrossing_coverage_abs = mean_metric(lavacrossing, "bgr_coverage", "final_abs_r10")
    lavacrossing_bgr_abs = mean_metric(lavacrossing, "bgr", "final_abs_r10")
    if not (
        lavacrossing_uniform_rauc > lavacrossing_ablation_rauc
        and lavacrossing_ablation_rauc > lavacrossing_coverage_rauc
        and lavacrossing_coverage_rauc > lavacrossing_bgr_rauc
        and lavacrossing_coverage_abs < lavacrossing_uniform_abs
        and lavacrossing_bgr_abs < lavacrossing_uniform_abs
    ):
        raise ValueError("Expected MiniGrid-LavaCrossing diagnostic to remain a non-promoted negative result")
    claims.append(
        Claim(
            "MiniGrid-LavaCrossing official-package limitation",
            f"LavaCrossingS9N3 BGR-Coverage {fmt(lavacrossing_coverage_rauc, 4)}, BGR {fmt(lavacrossing_bgr_rauc, 4)}",
            "results/minigrid_lavacrossing_recovery_probe_4seed_v1/summary.csv",
        )
    )

    lavagap = read_csv_rows(results_dir / "minigrid_lavagap_s7_recovery_probe_4seed_v1" / "summary.csv")
    lavagap_uniform_rauc = mean_metric(lavagap, "uniform", "final_rauc")
    lavagap_coverage_rauc = mean_metric(lavagap, "bgr_coverage", "final_rauc")
    lavagap_bgr_rauc = mean_metric(lavagap, "bgr", "final_rauc")
    lavagap_ablation_rauc = mean_metric(lavagap, "bgr_uniform_radius", "final_rauc")
    if not (
        lavagap_ablation_rauc > lavagap_uniform_rauc
        and lavagap_uniform_rauc > lavagap_coverage_rauc
        and lavagap_coverage_rauc > lavagap_bgr_rauc
    ):
        raise ValueError("Expected MiniGrid-LavaGap diagnostic to remain a non-promoted negative result")
    claims.append(
        Claim(
            "MiniGrid-LavaGap official-package limitation",
            (
                f"FourRooms mid2--5/LavaGapS7 & BGR {fmt(minigrid_mid25_bgr_rauc, 4)}; "
                f"BGR-Cov. {fmt(minigrid_mid25_coverage_rauc, 4)}/{fmt(lavagap_coverage_rauc, 4)}; "
                f"BGR {fmt(lavagap_bgr_rauc, 4)}"
            ),
            "results/minigrid_lavagap_s7_recovery_probe_4seed_v1/summary.csv",
        )
    )

    pointmaze = read_csv_rows(results_dir / "pointmaze_umaze_recovery_probe_4seed_v1" / "summary.csv")
    pointmaze_shield = read_csv_rows(results_dir / "pointmaze_umaze_clean_shield_probe_4seed_v1" / "summary.csv")
    pointmaze_failure_rauc = mean_metric(pointmaze, "failure_only", "final_rauc")
    pointmaze_uniform_rauc = mean_metric(pointmaze, "uniform", "final_rauc")
    pointmaze_coverage_rauc = mean_metric(pointmaze, "bgr_coverage", "final_rauc")
    pointmaze_bgr_rauc = mean_metric(pointmaze, "bgr", "final_rauc")
    pointmaze_shield_rauc = mean_metric(pointmaze_shield, "bgr_clean_shield", "final_rauc")
    pointmaze_shield_abs = mean_metric(pointmaze_shield, "bgr_clean_shield", "final_abs_r20")
    pointmaze_uniform_abs = mean_metric(pointmaze_shield, "uniform", "final_abs_r20")
    pointmaze_shield_wins = paired_wins(pointmaze_shield, "bgr_clean_shield", "uniform", "final_rauc")
    pointmaze_shield_pairs = sum(pointmaze_shield_wins)
    if not (
        pointmaze_failure_rauc > pointmaze_uniform_rauc
        and pointmaze_uniform_rauc > pointmaze_coverage_rauc
        and pointmaze_coverage_rauc > pointmaze_bgr_rauc
        and pointmaze_shield_rauc > pointmaze_uniform_rauc
        and pointmaze_shield_rauc < pointmaze_failure_rauc
        and pointmaze_shield_wins == (2, 2, 0)
        and pointmaze_shield_abs < pointmaze_uniform_abs
    ):
        raise ValueError("Expected PointMaze diagnostics to remain non-promoted negative results")
    claims.append(
        Claim(
            "PointMaze official-package limitation",
            (
                f"PointMaze U-Maze BGR-Clean-Shield {fmt(pointmaze_shield_rauc, 4)}, "
                f"BGR {fmt(pointmaze_bgr_rauc, 4)}"
            ),
            "results/pointmaze_umaze_recovery_probe_4seed_v1/summary.csv and pointmaze_umaze_clean_shield_probe_4seed_v1/summary.csv",
        )
    )

    fetchreach_hard = read_csv_rows(results_dir / "fetchreach_goal_recovery_hard_probe_4seed_v1" / "summary.csv")
    fetchreach_hard_coverage_rauc = mean_metric(fetchreach_hard, "bgr_coverage", "final_rauc")
    fetchreach_hard_bgr_rauc = mean_metric(fetchreach_hard, "bgr", "final_rauc")
    fetchreach_hard_uniform_rauc = mean_metric(fetchreach_hard, "uniform", "final_rauc")
    fetchreach_hard_failure_rauc = mean_metric(fetchreach_hard, "failure_only", "final_rauc")
    fetchreach_hard_td_rauc = mean_metric(fetchreach_hard, "td_loss", "final_rauc")
    fetchreach_hard_ablation_rauc = mean_metric(fetchreach_hard, "bgr_uniform_radius", "final_rauc")
    if not (
        fetchreach_hard_td_rauc > fetchreach_hard_failure_rauc
        and fetchreach_hard_failure_rauc > fetchreach_hard_uniform_rauc
        and fetchreach_hard_uniform_rauc > fetchreach_hard_ablation_rauc
        and fetchreach_hard_ablation_rauc > fetchreach_hard_coverage_rauc
        and fetchreach_hard_coverage_rauc > fetchreach_hard_bgr_rauc
    ):
        raise ValueError("Expected hard-budget FetchReach diagnostic to remain a non-promoted negative result")
    claims.append(
        Claim(
            "FetchReach official-package limitation",
            (
                f"FetchReach BGR-Coverage {fmt(fetchreach_hard_coverage_rauc, 4)}, "
                f"BGR {fmt(fetchreach_hard_bgr_rauc, 4)}"
            ),
            "results/fetchreach_goal_recovery_hard_probe_4seed_v1/summary.csv",
        )
    )

    lunar = read_csv_rows(results_dir / "lunarlander_recovery_probe_4seed_v1" / "summary.csv")
    lunar_coverage_rauc = mean_metric(lunar, "bgr_coverage", "final_rauc")
    lunar_uniform_rauc = mean_metric(lunar, "uniform", "final_rauc")
    lunar_fixed_rauc = mean_metric(lunar, "fixed", "final_rauc")
    lunar_failure_rauc = mean_metric(lunar, "failure_only", "final_rauc")
    lunar_td_rauc = mean_metric(lunar, "td_loss", "final_rauc")
    lunar_ablation_rauc = mean_metric(lunar, "bgr_uniform_radius", "final_rauc")
    lunar_coverage_r80 = mean_metric(lunar, "bgr_coverage", "final_median_r80")
    lunar_uniform_r80 = mean_metric(lunar, "uniform", "final_median_r80")
    lunar_coverage_wins = paired_wins(lunar, "bgr_coverage", "uniform", "final_rauc")
    if not (
        lunar_coverage_rauc > lunar_uniform_rauc
        and lunar_coverage_rauc > max(lunar_fixed_rauc, lunar_failure_rauc, lunar_td_rauc, lunar_ablation_rauc)
        and lunar_coverage_wins == (2, 2, 0)
        and lunar_coverage_r80 < lunar_uniform_r80
    ):
        raise ValueError("Expected LunarLander-v3 diagnostic to remain a non-promoted near miss")
    claims.append(
        Claim(
            "LunarLander-v3 official-package limitation",
            (
                f"LunarLander is a near miss: BGR-Coverage has the best mean RAUC "
                f"({fmt(lunar_coverage_rauc, 4)} vs. {fmt(lunar_uniform_rauc, 4)} uniform) but wins only "
                f"{lunar_coverage_wins[0]}/4 seeds and has lower median $r_{{80}}$"
            ),
            "results/lunarlander_recovery_probe_4seed_v1/summary.csv",
        )
    )
    claims.append(
        Claim(
            "LunarLander-v3 scope-audit table row",
            (
                f"LunarLander-v3 & BGR-Coverage {fmt(lunar_coverage_rauc, 4)} & uniform "
                f"{fmt(lunar_uniform_rauc, 4)}; only {lunar_coverage_wins[0]}/4 wins; "
                f"r80 {fmt(lunar_coverage_r80, 4)} vs. {fmt(lunar_uniform_r80, 4)} & not promoted"
            ),
            "results/lunarlander_recovery_probe_4seed_v1/summary.csv",
        )
    )

    deepsea = read_csv_rows(results_dir / "bsuite_deepsea_recovery_probe_4seed_v1" / "summary.csv")
    deepsea_bgr_rauc = mean_metric(deepsea, "bgr", "final_rauc")
    deepsea_coverage_rauc = mean_metric(deepsea, "bgr_coverage", "final_rauc")
    deepsea_uniform_rauc = mean_metric(deepsea, "uniform", "final_rauc")
    deepsea_ablation_rauc = mean_metric(deepsea, "bgr_uniform_radius", "final_rauc")
    deepsea_bgr_r80 = mean_metric(deepsea, "bgr", "final_median_r80")
    deepsea_uniform_r80 = mean_metric(deepsea, "uniform", "final_median_r80")
    deepsea_wins = paired_wins(deepsea, "bgr", "uniform", "final_rauc")
    if not (
        deepsea_bgr_rauc > deepsea_uniform_rauc
        and deepsea_bgr_rauc < deepsea_ablation_rauc
        and deepsea_wins == (2, 1, 1)
        and deepsea_bgr_r80 < deepsea_uniform_r80
    ):
        raise ValueError("Expected bsuite DeepSea diagnostic to remain a non-promoted negative result")
    catch = read_csv_rows(results_dir / "bsuite_catch_recovery_probe_30seed_v1" / "summary.csv")
    catch_bgr_rauc = mean_metric(catch, "bgr", "final_rauc")
    catch_coverage_rauc = mean_metric(catch, "bgr_coverage", "final_rauc")
    catch_uniform_rauc = mean_metric(catch, "uniform", "final_rauc")
    catch_failure_rauc = mean_metric(catch, "failure_only", "final_rauc")
    catch_ablation_rauc = mean_metric(catch, "bgr_uniform_radius", "final_rauc")
    catch_bgr_r80 = mean_metric(catch, "bgr", "final_median_r80")
    catch_uniform_r80 = mean_metric(catch, "uniform", "final_median_r80")
    catch_wins = paired_wins(catch, "bgr", "uniform", "final_rauc")
    if not (
        catch_failure_rauc > catch_uniform_rauc > catch_bgr_rauc
        and catch_ablation_rauc > catch_bgr_rauc
        and catch_wins == (14, 16, 0)
        and catch_bgr_r80 < catch_uniform_r80
    ):
        raise ValueError("Expected bsuite Catch 30-seed diagnostic to remain a non-promoted negative result")

    cartpole = read_csv_rows(results_dir / "bsuite_cartpole_recovery_probe_4seed_v1" / "summary.csv")
    cartpole_bgr_rauc = mean_metric(cartpole, "bgr", "final_rauc")
    cartpole_coverage_rauc = mean_metric(cartpole, "bgr_coverage", "final_rauc")
    cartpole_uniform_rauc = mean_metric(cartpole, "uniform", "final_rauc")
    cartpole_td_rauc = mean_metric(cartpole, "td_loss", "final_rauc")
    cartpole_wins = paired_wins(cartpole, "bgr", "uniform", "final_rauc")
    if not (
        cartpole_td_rauc > cartpole_uniform_rauc > cartpole_bgr_rauc
        and cartpole_td_rauc > cartpole_coverage_rauc
        and cartpole_wins == (0, 4, 0)
    ):
        raise ValueError("Expected bsuite Cartpole diagnostic to remain a non-promoted negative result")
    claims.append(
        Claim(
            "bsuite compressed scope-audit table row",
            (
                f"bsuite/MinAtar & Catch BGR {fmt(catch_bgr_rauc, 4)}; "
                f"Cartpole BGR-Cov. {fmt(cartpole_coverage_rauc, 4)}; "
            ),
            "results/bsuite_cartpole_recovery_probe_4seed_v1/summary.csv",
        )
    )
    claims.append(
        Claim(
            "bsuite Catch compressed limitation",
            (
                f"bsuite Catch shows scale-up fragility ({fmt(catch_bgr_rauc, 4)} vs. "
                f"{fmt(catch_uniform_rauc, 4)} uniform)"
            ),
            "results/bsuite_catch_recovery_probe_30seed_v1/summary.csv",
        )
    )
    claims.append(
        Claim(
            "bsuite Cartpole compressed limitation",
            (
                "bsuite Cartpole trails TD-loss replay"
            ),
            "results/bsuite_cartpole_recovery_probe_4seed_v1/summary.csv",
        )
    )

    minatar = read_csv_rows(results_dir / "minatar_breakout_recovery_probe_4seed_v1" / "summary.csv")
    minatar_bgr_rauc = mean_metric(minatar, "bgr", "final_rauc")
    minatar_coverage_rauc = mean_metric(minatar, "bgr_coverage", "final_rauc")
    minatar_uniform_rauc = mean_metric(minatar, "uniform", "final_rauc")
    minatar_failure_rauc = mean_metric(minatar, "failure_only", "final_rauc")
    minatar_bgr_r80 = mean_metric(minatar, "bgr", "final_median_r80")
    minatar_coverage_r80 = mean_metric(minatar, "bgr_coverage", "final_median_r80")
    minatar_wins = paired_wins(minatar, "bgr", "uniform", "final_rauc")
    if not (
        abs(minatar_bgr_rauc - minatar_uniform_rauc) < 1e-12
        and abs(minatar_coverage_rauc - minatar_uniform_rauc) < 1e-12
        and abs(minatar_failure_rauc - minatar_uniform_rauc) < 1e-12
        and minatar_bgr_r80 >= 4.99
        and minatar_coverage_r80 >= 4.99
        and minatar_wins == (0, 0, 4)
    ):
        raise ValueError("Expected MinAtar Breakout diagnostic to remain a tied saturated negative result")
    claims.append(
        Claim(
            "MinAtar Breakout official-package limitation",
            (
                f"MinAtar Breakout ties uniform and failure-only at {fmt(minatar_bgr_rauc, 4)} RAUC "
                f"with saturated $r_{{80}}={fmt(minatar_bgr_r80, 4)}"
            ),
            "results/minatar_breakout_recovery_probe_4seed_v1/summary.csv",
        )
    )
    claims.append(
        Claim(
            "MinAtar Breakout compressed scope-audit table row",
            (
                f"Catch uniform {fmt(catch_uniform_rauc, 4)}; "
                f"Cartpole TD-loss {fmt(cartpole_td_rauc, 4)}"
            ),
            "results/minatar_breakout_recovery_probe_4seed_v1/summary.csv",
        )
    )

    asterix = read_csv_rows(results_dir / "minatar_asterix_recovery_probe_4seed_v1" / "summary.csv")
    asterix_bgr_rauc = mean_metric(asterix, "bgr", "final_rauc")
    asterix_coverage_rauc = mean_metric(asterix, "bgr_coverage", "final_rauc")
    asterix_uniform_rauc = mean_metric(asterix, "uniform", "final_rauc")
    asterix_failure_rauc = mean_metric(asterix, "failure_only", "final_rauc")
    asterix_coverage_wins = paired_wins(asterix, "bgr_coverage", "uniform", "final_rauc")
    if not (
        asterix_coverage_rauc > asterix_uniform_rauc
        and asterix_failure_rauc > asterix_coverage_rauc
        and asterix_coverage_wins == (1, 2, 1)
    ):
        raise ValueError("Expected MinAtar Asterix diagnostic to remain blocked by failure-only and paired signs")
    claims.append(
        Claim(
            "MinAtar Asterix official-package limitation",
            (
                f"MinAtar Asterix trails failure-only ({fmt(asterix_coverage_rauc, 4)} vs. "
                f"{fmt(asterix_failure_rauc, 4)} RAUC)"
            ),
            "results/minatar_asterix_recovery_probe_4seed_v1/summary.csv",
        )
    )
    claims.append(
        Claim(
            "MinAtar Asterix compressed scope-audit table row",
            (
                f"Asterix BGR-Cov. {fmt(asterix_coverage_rauc, 4)}"
            ),
            "results/minatar_asterix_recovery_probe_4seed_v1/summary.csv",
        )
    )
    claims.append(
        Claim(
            "MinAtar Asterix compressed blocker table row",
            (
                f"Asterix failure-only {fmt(asterix_failure_rauc, 4)}"
            ),
            "results/minatar_asterix_recovery_probe_4seed_v1/summary.csv",
        )
    )

    reacher = read_csv_rows(results_dir / "reacher_recovery_probe_12seed_v1" / "summary.csv")
    reacher_bgr_rauc = mean_metric(reacher, "bgr", "final_rauc")
    reacher_coverage_rauc = mean_metric(reacher, "bgr_coverage", "final_rauc")
    reacher_uniform_rauc = mean_metric(reacher, "uniform", "final_rauc")
    reacher_wins = paired_wins(reacher, "bgr", "uniform", "final_rauc")
    if not (
        reacher_uniform_rauc > reacher_bgr_rauc
        and reacher_uniform_rauc > reacher_coverage_rauc
        and reacher_wins == (4, 8, 0)
    ):
        raise ValueError("Expected Reacher-v5 diagnostic to remain a non-promoted negative result")
    claims.append(
        Claim(
            "Reacher-v5 official-package limitation",
            (
                f"Reacher-v5 BGR {fmt(reacher_bgr_rauc, 4)}, "
                f"BGR-Coverage {fmt(reacher_coverage_rauc, 4)}"
            ),
            "results/reacher_recovery_probe_12seed_v1/summary.csv",
        )
    )
    claims.append(
        Claim(
            "Reacher-v5 scope-audit table row",
            (
                f"Reacher uniform {fmt(reacher_uniform_rauc, 4)}, "
                f"BGR wins {reacher_wins[0]}/12 seeds"
            ),
            "results/reacher_recovery_probe_12seed_v1/summary.csv",
        )
    )

    inverted = read_csv_rows(results_dir / "inverted_pendulum_recovery_probe_4seed_v1" / "summary.csv")
    inverted_bgr_rauc = mean_metric(inverted, "bgr", "final_rauc")
    inverted_coverage_rauc = mean_metric(inverted, "bgr_coverage", "final_rauc")
    inverted_uniform_rauc = mean_metric(inverted, "uniform", "final_rauc")
    inverted_fixed_rauc = mean_metric(inverted, "fixed", "final_rauc")
    inverted_failure_rauc = mean_metric(inverted, "failure_only", "final_rauc")
    inverted_td_rauc = mean_metric(inverted, "td_loss", "final_rauc")
    inverted_ablation_rauc = mean_metric(inverted, "bgr_uniform_radius", "final_rauc")
    inverted_bgr_r80 = mean_metric(inverted, "bgr", "final_median_r80")
    inverted_wins = paired_wins(inverted, "bgr", "uniform", "final_rauc")
    if not (
        inverted_bgr_rauc
        == inverted_coverage_rauc
        == inverted_uniform_rauc
        == inverted_fixed_rauc
        == inverted_failure_rauc
        == inverted_td_rauc
        == inverted_ablation_rauc
        and inverted_bgr_r80 == mean_metric(inverted, "uniform", "final_median_r80")
        and inverted_wins == (0, 0, 4)
    ):
        raise ValueError("Expected InvertedPendulum-v5 diagnostic to remain a tied negative result")
    claims.append(
        Claim(
            "InvertedPendulum-v5 official-package limitation",
            (
                f"InvertedPendulum-v5 BGR {fmt(inverted_bgr_rauc, 4)}, "
                f"BGR-Cov. {fmt(inverted_coverage_rauc, 4)}"
            ),
            "results/inverted_pendulum_recovery_probe_4seed_v1/summary.csv",
        )
    )
    claims.append(
        Claim(
            "InvertedPendulum-v5 scope-audit table row",
            (
                f"Pendulum uniform/fixed/failure-only/TD-loss all {fmt(inverted_uniform_rauc, 4)}"
            ),
            "results/inverted_pendulum_recovery_probe_4seed_v1/summary.csv",
        )
    )

    inverted_double = read_csv_rows(results_dir / "inverted_double_pendulum_recovery_probe_4seed_v1" / "summary.csv")
    double_bgr_rauc = mean_metric(inverted_double, "bgr", "final_rauc")
    double_coverage_rauc = mean_metric(inverted_double, "bgr_coverage", "final_rauc")
    double_uniform_rauc = mean_metric(inverted_double, "uniform", "final_rauc")
    double_bgr_clean = mean_metric(inverted_double, "bgr", "final_clean")
    double_coverage_clean = mean_metric(inverted_double, "bgr_coverage", "final_clean")
    double_wins = paired_wins(inverted_double, "bgr", "uniform", "final_rauc")
    if not (
        double_bgr_rauc > double_uniform_rauc
        and double_bgr_clean < 0.75
        and double_coverage_clean == 0.0
        and double_wins == (1, 0, 3)
    ):
        raise ValueError("Expected InvertedDoublePendulum-v5 diagnostic to remain a collapsed negative result")
    claims.append(
        Claim(
            "InvertedDoublePendulum-v5 official-package limitation",
            (
                f"InvertedDoublePendulum-v5 gives a tiny BGR RAUC edge ({fmt(double_bgr_rauc, 4)} "
                f"vs. {fmt(double_uniform_rauc, 4)} uniform) but collapses clean success to "
                f"{fmt(double_bgr_clean, 4)} for BGR and {fmt(double_coverage_clean, 4)} for BGR-Coverage"
            ),
            "results/inverted_double_pendulum_recovery_probe_4seed_v1/summary.csv",
        )
    )
    claims.append(
        Claim(
            "InvertedDoublePendulum-v5 scope-audit table row",
            (
                f"InvertedDoublePendulum-v5 BGR {fmt(double_bgr_rauc, 4)}, "
                f"BGR-Cov. {fmt(double_coverage_rauc, 4)}"
            ),
            "results/inverted_double_pendulum_recovery_probe_4seed_v1/summary.csv",
        )
    )

    probe_rows = read_csv_rows(results_dir / "libero_probe_v2" / "summary.csv")
    valid_rows = sum(1 for row in probe_rows if float(row["valid_rate"]) == 1.0 and not row.get("error"))
    claims.append(
        Claim(
            "LIBERO valid radius rows",
            f"{valid_rows}/{len(probe_rows)} valid radius rows",
            "results/libero_probe_v2/summary.csv",
        )
    )

    openvla = read_csv_rows(figures_dir / "openvla_stats.csv")
    bgr_sel = one_row([row for row in openvla if row["audit"] == "Selection"], "name", "BGR-boundary")
    random_sel = one_row([row for row in openvla if row["audit"] == "Selection"], "name", "Random-balanced")
    claims.append(
        Claim(
            "OpenVLA selection boundary hit",
            f"{fmt(float(bgr_sel['metric_b']), 3)} vs. {fmt(float(random_sel['metric_b']), 3)}",
            "paper/figures/openvla_stats.csv",
        )
    )

    teacher_summary = read_json(results_dir / "openvla_teacher_replay_manifest_v1" / "summary.json")
    claims.append(
        Claim(
            "OpenVLA teacher rows",
            f"{int(teacher_summary['num_rows']):,} teacher-action rows",
            "results/openvla_teacher_replay_manifest_v1/summary.json",
        )
    )
    action_tfds = read_json(results_dir / "openvla_action_tfds_validation_v1" / "summary.json")
    expected_action_tfds = {
        "teacher_action_rows": 11776,
        "matched_tfds_transitions_per_method": 2048,
        "matched_tfds_trajectories_per_method": 32,
        "action_dim": 7,
        "state_dim": 8,
    }
    for key, value in expected_action_tfds.items():
        if int(action_tfds[key]) != value:
            raise ValueError(f"Expected OpenVLA action/TFDS validation {key}={value}")
    if action_tfds["loader_validation"]["action_shape"] != [64, 7]:
        raise ValueError("Expected OpenVLA loader action shape [64, 7]")
    if action_tfds["loader_validation"]["proprio_shape"] != [64, 8]:
        raise ValueError("Expected OpenVLA loader proprio shape [64, 8]")
    if action_tfds["loader_validation"]["stats_action_mean_shape"] != [7]:
        raise ValueError("Expected OpenVLA action-stat shape [7]")
    if action_tfds["loader_validation"]["stats_proprio_mean_shape"] != [8]:
        raise ValueError("Expected OpenVLA proprio-stat shape [8]")
    if action_tfds["lora_smoke"]["max_steps"] != 10 or not all(
        action_tfds["lora_smoke"]["checkpoint_written"].values()
    ):
        raise ValueError("Expected matched OpenVLA 10-step LoRA checkpoint smokes")
    if action_tfds["closed_loop_finetuning_gain_claim"] is not False:
        raise ValueError("OpenVLA action/TFDS validation artifact must not claim a fine-tuning gain")
    claims.append(
        Claim(
            "OpenVLA action/TFDS validation",
            "action-label/TFDS plumbing validates 2,048-transition matched BGR/random exports with 7D actions and 8D state",
            "results/openvla_action_tfds_validation_v1/summary.json",
        )
    )

    sanity = read_csv_rows(results_dir / "openvla_oft_sanity_eval_sanity_v1" / "summary.csv")
    official = one_row(sanity, "method", "oft-goal")

    final_clean = read_csv_rows(
        results_dir
        / "openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1"
        / "summary.csv"
    )
    final_bgr_clean = one_row(final_clean, "method", "bgr")
    final_random_clean = one_row(final_clean, "method", "random")
    if ratio(final_bgr_clean["successes"], final_bgr_clean["episodes"]) != ratio(
        final_random_clean["successes"], final_random_clean["episodes"]
    ):
        raise ValueError("Expected p1024 BGR/random clean success ratios to match")

    final_perturb = read_csv_rows(
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1"
        / "summary.csv"
    )
    offset_perturb = read_csv_rows(
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
        / "summary.csv"
    )
    p1024_pooled_bgr = pooled_success_rate([final_perturb, offset_perturb], "bgr", exclude_perturbations={"identity"})
    p1024_pooled_random = pooled_success_rate(
        [final_perturb, offset_perturb], "random", exclude_perturbations={"identity"}
    )
    p1024_pooled_official = pooled_success_rate(
        [final_perturb, offset_perturb], "official", exclude_perturbations={"identity"}
    )
    p2048_clean = read_csv_rows(
        results_dir
        / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
        / "summary.csv"
    )
    p2048_bgr_clean = one_row(p2048_clean, "method", "bgr")
    p2048_random_clean = one_row(p2048_clean, "method", "random")
    if ratio(p2048_bgr_clean["successes"], p2048_bgr_clean["episodes"]) != ratio(
        p2048_random_clean["successes"], p2048_random_clean["episodes"]
    ):
        raise ValueError("Expected p2048 BGR/random clean success ratios to match")

    p2048_perturb = read_csv_rows(
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
        / "summary.csv"
    )
    p2048_offset_perturb = read_csv_rows(
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
        / "summary.csv"
    )
    p2048_bgr_visual = mean_success_rate(p2048_perturb, "bgr", exclude_perturbations={"identity"})
    p2048_random_visual = mean_success_rate(p2048_perturb, "random", exclude_perturbations={"identity"})
    p2048_official_visual = mean_success_rate(p2048_perturb, "official", exclude_perturbations={"identity"})
    p2048_offset_bgr = mean_success_rate(p2048_offset_perturb, "bgr", exclude_perturbations={"identity"})
    p2048_offset_random = mean_success_rate(p2048_offset_perturb, "random", exclude_perturbations={"identity"})
    p2048_offset_official = mean_success_rate(p2048_offset_perturb, "official", exclude_perturbations={"identity"})
    p2048_pooled_bgr = pooled_success_rate(
        [p2048_perturb, p2048_offset_perturb], "bgr", exclude_perturbations={"identity"}
    )
    p2048_pooled_random = pooled_success_rate(
        [p2048_perturb, p2048_offset_perturb], "random", exclude_perturbations={"identity"}
    )
    p2048_pooled_official = pooled_success_rate(
        [p2048_perturb, p2048_offset_perturb], "official", exclude_perturbations={"identity"}
    )
    p2048_fullgoal_clean = read_csv_rows(
        results_dir
        / "openvla_oft_clean_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1"
        / "summary.csv"
    )
    fullgoal_clean_ratios = {
        method: ratio(
            one_row(p2048_fullgoal_clean, "method", method)["successes"],
            one_row(p2048_fullgoal_clean, "method", method)["episodes"],
        )
        for method in ["bgr", "random", "official"]
    }
    if len(set(fullgoal_clean_ratios.values())) != 1:
        raise ValueError("Expected p2048 full-goal clean identity audit to tie BGR, random, and official")
    claims.append(
        Claim(
            "OpenVLA p2048 full-goal clean audit",
            f"Identity is {fullgoal_clean_ratios['bgr']} for BGR, matched random, and official in the full-goal audit",
            "results/openvla_oft_clean_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1/summary.csv",
        )
    )

    p2048_fullgoal_perturb = read_csv_rows(
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1"
        / "summary.csv"
    )

    def perturbed_total(rows: list[dict[str, str]], method: str) -> tuple[int, int]:
        rows = [
            row
            for row in rows
            if row["method"] == method and row["perturbation"] != "identity"
        ]
        return (
            sum(int(row["successes"]) for row in rows),
            sum(int(row["episodes"]) for row in rows),
        )

    bgr_successes, bgr_episodes = perturbed_total(p2048_fullgoal_perturb, "bgr")
    official_successes, official_episodes = perturbed_total(p2048_fullgoal_perturb, "official")
    random_successes, random_episodes = perturbed_total(p2048_fullgoal_perturb, "random")
    if not (
        bgr_episodes == official_episodes == random_episodes
        and bgr_successes == official_successes
        and random_successes == bgr_successes + 1
    ):
        raise ValueError("Expected p2048 full-goal perturbation audit to tie official and trail random by one episode")
    claims.append(
        Claim(
            "OpenVLA p2048 full-goal visual perturbation audit",
            (
                f"visual perturbation gives BGR {bgr_successes}/{bgr_episodes}, "
                f"official {official_successes}/{official_episodes}, and random {random_successes}/{random_episodes}"
            ),
            "results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1/summary.csv",
        )
    )
    p2048_imageaug_300_perturb = read_csv_rows(
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1"
        / "summary.csv"
    )
    imageaug_bgr_successes, imageaug_bgr_episodes = perturbed_total(p2048_imageaug_300_perturb, "bgr")
    imageaug_official_successes, imageaug_official_episodes = perturbed_total(
        p2048_imageaug_300_perturb, "official"
    )
    imageaug_random_successes, imageaug_random_episodes = perturbed_total(p2048_imageaug_300_perturb, "random")
    imageaug_identity = {
        row["method"]: int(row["successes"])
        for row in p2048_imageaug_300_perturb
        if row["perturbation"] == "identity"
    }
    if not (
        imageaug_bgr_episodes == imageaug_official_episodes == imageaug_random_episodes == 400
        and imageaug_bgr_successes == imageaug_random_successes == 368
        and imageaug_official_successes == 367
        and imageaug_identity == {"bgr": 98, "official": 99, "random": 100}
    ):
        raise ValueError("Expected p2048 300-step image-augmentation audit to tie random, edge official by one perturbed episode, and trail identity")
    claims.append(
        Claim(
            "OpenVLA p2048 300-step image-augmentation audit",
            (
                f"Image augmentation gives BGR/random {imageaug_bgr_successes}/{imageaug_bgr_episodes} "
                f"vs. official {imageaug_official_successes}/{imageaug_official_episodes} but worse BGR identity"
            ),
            "results/openvla_oft_perturb_eval_cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1/summary.csv",
        )
    )
    p2048_imageaug_1000_low_lr_perturb = read_csv_rows(
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p2048_step51000_lr1em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1"
        / "summary.csv"
    )
    low_lr_bgr_successes, low_lr_bgr_episodes = perturbed_total(p2048_imageaug_1000_low_lr_perturb, "bgr")
    low_lr_official_successes, low_lr_official_episodes = perturbed_total(
        p2048_imageaug_1000_low_lr_perturb, "official"
    )
    low_lr_random_successes, low_lr_random_episodes = perturbed_total(
        p2048_imageaug_1000_low_lr_perturb, "random"
    )
    low_lr_identity = {
        row["method"]: int(row["successes"])
        for row in p2048_imageaug_1000_low_lr_perturb
        if row["perturbation"] == "identity"
    }
    if not (
        low_lr_bgr_episodes == low_lr_official_episodes == low_lr_random_episodes == 400
        and low_lr_bgr_successes == 366
        and low_lr_official_successes == 367
        and low_lr_random_successes == 370
        and low_lr_identity == {"bgr": 98, "official": 99, "random": 99}
    ):
        raise ValueError("Expected p2048 1,000-step low-LR image-augmentation audit to trail both controls")
    claims.append(
        Claim(
            "OpenVLA p2048 1,000-step low-LR image-augmentation audit",
            (
                f"low-LR continuation gives BGR {low_lr_bgr_successes}/{low_lr_bgr_episodes} "
                f"vs. official {low_lr_official_successes}/{low_lr_official_episodes} "
                f"and random {low_lr_random_successes}/{low_lr_random_episodes}"
            ),
            "results/openvla_oft_perturb_eval_cleanmix_p2048_step51000_lr1em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1/summary.csv",
        )
    )
    p2048_proximal_anchor_perturb = read_csv_rows(
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_proxanchor_l2_1em0_ddpgradfix_v1_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1"
        / "summary.csv"
    )
    proximal_bgr_successes, proximal_bgr_episodes = perturbed_total(p2048_proximal_anchor_perturb, "bgr")
    proximal_official_successes, proximal_official_episodes = perturbed_total(
        p2048_proximal_anchor_perturb, "official"
    )
    proximal_random_successes, proximal_random_episodes = perturbed_total(
        p2048_proximal_anchor_perturb, "random"
    )
    proximal_identity = {
        row["method"]: int(row["successes"])
        for row in p2048_proximal_anchor_perturb
        if row["perturbation"] == "identity"
    }
    if not (
        proximal_bgr_episodes == proximal_official_episodes == proximal_random_episodes == 400
        and proximal_bgr_successes == 368
        and proximal_official_successes == 367
        and proximal_random_successes == 368
        and proximal_identity == {"bgr": 98, "official": 99, "random": 98}
    ):
        raise ValueError("Expected proximal-anchor OpenVLA audit to tie random and edge official by one episode")
    claims.append(
        Claim(
            "OpenVLA proximal-anchor audit",
            (
                f"proximal anchor ties random at {proximal_bgr_successes}/{proximal_bgr_episodes} "
                f"with BGR identity {proximal_identity['bgr']}/100 vs. official {proximal_identity['official']}/100"
            ),
            "results/openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_proxanchor_l2_1em0_ddpgradfix_v1_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv",
        )
    )
    p2048_perturb_only_anchor_perturb = read_csv_rows(
        results_dir
        / "openvla_oft_perturb_eval_p2048unique_perturbonly_anchor_prereg_perturbonly_proxanchor_l2_5em0_step50300_lr2em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1"
        / "summary.csv"
    )
    perturb_only_bgr_successes, perturb_only_bgr_episodes = perturbed_total(
        p2048_perturb_only_anchor_perturb, "bgr"
    )
    perturb_only_official_successes, perturb_only_official_episodes = perturbed_total(
        p2048_perturb_only_anchor_perturb, "official"
    )
    perturb_only_random_successes, perturb_only_random_episodes = perturbed_total(
        p2048_perturb_only_anchor_perturb, "random"
    )
    perturb_only_identity = {
        row["method"]: int(row["successes"])
        for row in p2048_perturb_only_anchor_perturb
        if row["perturbation"] == "identity"
    }
    if not (
        perturb_only_bgr_episodes == perturb_only_official_episodes == perturb_only_random_episodes == 400
        and perturb_only_bgr_successes == 371
        and perturb_only_official_successes == 367
        and perturb_only_random_successes == 372
        and perturb_only_identity == {"bgr": 99, "official": 99, "random": 99}
    ):
        raise ValueError("Expected perturb-only anchored OpenVLA audit to trail matched random")
    claims.append(
        Claim(
            "OpenVLA perturb-only anchored audit",
            (
                f"Perturb-only anchoring preserves identity at {perturb_only_identity['bgr']}/100 "
                f"and raises BGR to {perturb_only_bgr_successes}/{perturb_only_bgr_episodes}, "
                f"but official is {perturb_only_official_successes}/{perturb_only_official_episodes} "
                f"and random is {perturb_only_random_successes}/{perturb_only_random_episodes}"
            ),
            "results/openvla_oft_perturb_eval_p2048unique_perturbonly_anchor_prereg_perturbonly_proxanchor_l2_5em0_step50300_lr2em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv",
        )
    )
    if not (p1024_pooled_bgr > p1024_pooled_random and p2048_pooled_bgr >= p2048_pooled_random):
        raise ValueError("Expected corrected OpenVLA diagnostics to show a p1024 edge and non-worse p2048 pooled comparison")
    if not (p1024_pooled_bgr < p1024_pooled_official and p2048_pooled_bgr < p2048_pooled_official):
        raise ValueError("Expected corrected OpenVLA diagnostics not to beat the official checkpoint")
    claims.extend(
        [
            Claim(
                "OpenVLA corrected diagnostic limitation",
                "no stable official improvement",
                "p1024/p2048 clean and perturbation summary CSVs",
            ),
            Claim(
                "OpenVLA no official-checkpoint improvement",
                "no stable official improvement",
                "p1024/p2048 perturbation summary CSVs",
            ),
        ]
    )

    return claims


def missing_claims(paper_text: str, claims: list[Claim]) -> list[Claim]:
    normalized_paper = " ".join(paper_text.split())
    return [claim for claim in claims if " ".join(claim.snippet.split()) not in normalized_paper]


def forbidden_terms(paper_text: str) -> list[str]:
    return [
        term
        for term in [
            "Bifurcation",
            "bifurcation",
            r"\input{figures/openvla_table.tex}",
            r"\label{tab:openvla}",
            r"Table~\ref{tab:openvla}",
        ]
        if term in paper_text
    ]


def effect_size_framing_issues(paper_text: str) -> list[str]:
    required_snippets = [
        (
            "significance table prioritizes effect sizes",
            "Mean differences and 95\\% confidence intervals are the primary quantities",
        ),
        (
            "sign tests framed as directional consistency",
            "a 30/0 sign pattern can certify direction while the absolute gain remains small",
        ),
        (
            "synthetic result framed as modest",
            "BGR gives a modest final recovery-AUC gain over uniform replay",
        ),
        (
            "synthetic absolute mean difference reported",
            "mean difference 0.0084",
        ),
        (
            "grid practical effect size named",
            "The endpoint gap is about 0.038 RAUC",
        ),
        (
            "suffix result framed as small",
            "raises final object RAUC over uniform by a small but consistent amount",
        ),
        (
            "suffix metric tension retained",
            "uniform remains higher on median $r_{80}$",
        ),
    ]
    return [label for label, snippet in required_snippets if snippet not in paper_text]


def unverified_result_claims(paper_text: str, results_dir: Path) -> list[str]:
    p1024_summary_paths = [
        results_dir
        / "openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1"
        / "summary.csv",
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1"
        / "summary.csv",
    ]
    p1024_offset_summary_paths = [
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
        / "summary.csv",
    ]
    p2048_summary_paths = [
        results_dir
        / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
        / "summary.csv",
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
        / "summary.csv",
    ]
    p2048_offset_summary_paths = [
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
        / "summary.csv",
    ]
    p2048_fullgoal_summary_paths = [
        results_dir
        / "openvla_oft_clean_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1"
        / "summary.csv",
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1"
        / "summary.csv",
    ]
    p2048_imageaug_300_summary_paths = [
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1"
        / "summary.csv",
    ]
    p2048_imageaug_1000_low_lr_summary_paths = [
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p2048_step51000_lr1em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1"
        / "summary.csv",
    ]
    p2048_weighted_perturb_paths = [
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1"
        / "summary.csv",
    ]
    p2048_proximal_anchor_paths = [
        results_dir
        / "openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_proxanchor_l2_1em0_ddpgradfix_v1_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1"
        / "summary.csv",
    ]
    p2048_perturb_only_anchor_paths = [
        results_dir
        / "openvla_oft_perturb_eval_p2048unique_perturbonly_anchor_prereg_perturbonly_proxanchor_l2_5em0_step50300_lr2em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1"
        / "summary.csv",
    ]
    action_tfds_summary_paths = [
        results_dir / "openvla_action_tfds_validation_v1" / "summary.json",
    ]
    guarded_results = {
        "cleanmix_p1024": p1024_summary_paths,
        "p1024 clean-mix": p1024_summary_paths,
        "latest p1024 diagnostic": p1024_summary_paths,
        "offset-3 follow-up": p1024_offset_summary_paths,
        "0.8550": p1024_summary_paths + p1024_offset_summary_paths,
        "p2048 scale-up": p2048_summary_paths,
        "p2048 ties": p2048_summary_paths,
        "p2048 offset": p2048_offset_summary_paths,
        "Pooling p2048": p2048_summary_paths + p2048_offset_summary_paths,
        "full-goal identity audit": p2048_fullgoal_summary_paths,
        "10-task visual perturbation audit": p2048_fullgoal_summary_paths,
        "367/400": p2048_fullgoal_summary_paths + p2048_weighted_perturb_paths + p2048_proximal_anchor_paths,
        "300-step image-augmentation continuation": p2048_imageaug_300_summary_paths,
        "368/400": p2048_imageaug_300_summary_paths + p2048_proximal_anchor_paths,
        "1,000-step low-learning-rate continuation": p2048_imageaug_1000_low_lr_summary_paths,
        "366/400": p2048_imageaug_1000_low_lr_summary_paths,
        "370/400": p2048_imageaug_1000_low_lr_summary_paths + p2048_weighted_perturb_paths,
        "weighted perturbation curriculum": p2048_weighted_perturb_paths,
        "proximal-anchor objective": p2048_proximal_anchor_paths,
        "perturb-only anchored objective": p2048_perturb_only_anchor_paths,
        "371/400": p2048_perturb_only_anchor_paths,
        "372/400": p2048_perturb_only_anchor_paths,
        "action-label/TFDS plumbing validates": action_tfds_summary_paths,
        "2,048-transition matched BGR/random exports": action_tfds_summary_paths,
    }
    missing: list[str] = []
    if "273/300" in paper_text:
        missing.append("273/300: stale partial weighted OpenVLA audit; use the complete 370/400 matched-random total")
    for token, required_paths in guarded_results.items():
        if token not in paper_text:
            continue
        absent = [path for path in required_paths if not path.exists()]
        if absent:
            missing.append(f"{token}: missing local summaries {', '.join(str(path) for path in absent)}")
    return missing


def forbidden_paper_negative_claims(paper_text: str) -> list[str]:
    forbidden_tokens = ["p4096", "4096-step", "4096 scale", "common-availability"]
    return [token for token in forbidden_tokens if token in paper_text]


def find_significance_row(
    rows: list[dict[str, str]],
    *,
    benchmark: str,
    condition: str,
    metric: str,
    treatment: str,
    baseline: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["benchmark"] == benchmark
        and row["condition"] == condition
        and row["metric"] == metric
        and row["treatment"] == treatment
        and row["baseline"] == baseline
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one significance row for "
            f"{benchmark}/{condition}/{metric}/{treatment}/{baseline}, found {len(matches)}"
        )
    return matches[0]


def build_significance_checks() -> list[SignificanceCheck]:
    checks = [
        SignificanceCheck("synthetic final RAUC sign test", "Synthetic margin 30-seed", "", "final_rauc", "bgr", "uniform", True, 29, 1),
        SignificanceCheck("synthetic 30-seed AULC confirmation", "Synthetic margin 30-seed", "", "rauc_aulc", "bgr", "uniform", True, 30, 0),
        SignificanceCheck("synthetic 30-seed clean confirmation", "Synthetic margin 30-seed", "", "final_clean", "bgr", "uniform", True, 30, 0),
        SignificanceCheck("synthetic 30-seed fixed RAUC confirmation", "Synthetic margin 30-seed", "", "final_rauc", "bgr", "fixed", True, 30, 0),
        SignificanceCheck("synthetic 30-seed failure RAUC confirmation", "Synthetic margin 30-seed", "", "final_rauc", "bgr", "failure_only", True, 30, 0),
        SignificanceCheck("synthetic 30-seed PLR RAUC confirmation", "Synthetic margin 30-seed", "", "final_rauc", "bgr", "plr_loss", True, 30, 0),
        SignificanceCheck("grid 15-seed diagnostic final RAUC sign test", "Grid margin 15-seed", "", "final_rauc", "bgr", "uniform", True, 15, 0),
        SignificanceCheck("grid 15-seed diagnostic fixed baseline sign test", "Grid margin full 15-seed", "", "final_rauc", "bgr", "fixed", True, 15, 0),
        SignificanceCheck("grid 15-seed diagnostic failure baseline sign test", "Grid margin full 15-seed", "", "final_rauc", "bgr", "failure_only", True, 15, 0),
        SignificanceCheck("grid 15-seed diagnostic PLR baseline sign test", "Grid margin full 15-seed", "", "final_rauc", "bgr", "plr_loss", True, 15, 0),
        SignificanceCheck("grid 30-seed final RAUC sign test", "Grid margin full 30-seed", "", "final_rauc", "bgr", "uniform", True, 30, 0),
        SignificanceCheck("grid 30-seed AULC sign test", "Grid margin full 30-seed", "", "rauc_aulc", "bgr", "uniform", True, 30, 0),
        SignificanceCheck("grid 30-seed clean sign test", "Grid margin full 30-seed", "", "final_clean", "bgr", "uniform", True, 30, 0),
        SignificanceCheck("grid 30-seed median r80 sign test", "Grid margin full 30-seed", "", "final_median_r80", "bgr", "uniform", True, 30, 0),
        SignificanceCheck("grid 30-seed fixed RAUC sign test", "Grid margin full 30-seed", "", "final_rauc", "bgr", "fixed", True, 30, 0),
        SignificanceCheck("grid 30-seed failure RAUC sign test", "Grid margin full 30-seed", "", "final_rauc", "bgr", "failure_only", True, 30, 0),
        SignificanceCheck("grid 30-seed PLR RAUC sign test", "Grid margin full 30-seed", "", "final_rauc", "bgr", "plr_loss", True, 30, 0),
        SignificanceCheck("grid 30-seed fixed AULC sign test", "Grid margin full 30-seed", "", "rauc_aulc", "bgr", "fixed", True, 30, 0),
        SignificanceCheck("grid 30-seed failure AULC sign test", "Grid margin full 30-seed", "", "rauc_aulc", "bgr", "failure_only", True, 30, 0),
        SignificanceCheck("grid 30-seed PLR AULC sign test", "Grid margin full 30-seed", "", "rauc_aulc", "bgr", "plr_loss", True, 30, 0),
        SignificanceCheck("grid held-out replication final RAUC sign test", "Grid margin replication 30-seed", "", "final_rauc", "bgr", "uniform", True, 30, 0),
        SignificanceCheck("grid held-out replication AULC sign test", "Grid margin replication 30-seed", "", "rauc_aulc", "bgr", "uniform", True, 30, 0),
        SignificanceCheck("grid held-out replication clean sign test", "Grid margin replication 30-seed", "", "final_clean", "bgr", "uniform", True, 30, 0),
        SignificanceCheck("grid ablation boundary-radius sign test", "Grid margin ablation 30-seed", "", "final_rauc", "bgr", "bgr_uniform_radius", True, 30, 0),
        SignificanceCheck("grid ablation boundary-radius AULC sign test", "Grid margin ablation 30-seed", "", "rauc_aulc", "bgr", "bgr_uniform_radius", True, 30, 0),
        SignificanceCheck("grid ablation uniform-radius caveat", "Grid margin ablation 30-seed", "", "final_rauc", "bgr_uniform_radius", "uniform", True, 30, 0),
        SignificanceCheck("grid ablation uniform-radius AULC caveat", "Grid margin ablation 30-seed", "", "rauc_aulc", "bgr_uniform_radius", "uniform", True, 30, 0),
        SignificanceCheck(
            "grid ablation held-out boundary-radius sign test",
            "Grid margin ablation replication 30-seed",
            "",
            "final_rauc",
            "bgr",
            "bgr_uniform_radius",
            True,
            30,
            0,
        ),
        SignificanceCheck(
            "grid ablation held-out boundary-radius AULC sign test",
            "Grid margin ablation replication 30-seed",
            "",
            "rauc_aulc",
            "bgr",
            "bgr_uniform_radius",
            True,
            30,
            0,
        ),
        SignificanceCheck(
            "grid ablation held-out uniform-radius caveat",
            "Grid margin ablation replication 30-seed",
            "",
            "final_rauc",
            "bgr_uniform_radius",
            "uniform",
            True,
            30,
            0,
        ),
        SignificanceCheck(
            "grid ablation held-out uniform-radius AULC caveat",
            "Grid margin ablation replication 30-seed",
            "",
            "rauc_aulc",
            "bgr_uniform_radius",
            "uniform",
            True,
            30,
            0,
        ),
        SignificanceCheck("suffix clean sign test", "Robot suffix coverage 30-seed", "", "final_clean", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("suffix object RAUC sign test", "Robot suffix coverage 30-seed", "", "final_rauc", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("suffix transfer sign test", "Robot suffix coverage 30-seed", "", "final_transfer_rauc", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("suffix AULC sign test", "Robot suffix coverage 30-seed", "", "rauc_aulc", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("suffix median r80 caveat", "Robot suffix coverage 30-seed", "", "final_median_r80", "bgr_broad", "uniform", False, 1, 29),
        SignificanceCheck("suffix strategy ablation coverage final RAUC", "Robot suffix strategy ablation 30-seed", "", "final_rauc", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("suffix strategy ablation boundary final RAUC caveat", "Robot suffix strategy ablation 30-seed", "", "final_rauc", "bgr_boundary", "uniform", True, 30, 0),
        SignificanceCheck("suffix strategy ablation broad vs boundary", "Robot suffix strategy ablation 30-seed", "", "final_rauc", "bgr_broad", "bgr_boundary", True, 30, 0),
        SignificanceCheck("suffix strategy ablation broad vs hard final RAUC", "Robot suffix strategy ablation 30-seed", "", "final_rauc", "bgr_broad", "bgr_hard", True, 30, 0),
        SignificanceCheck("suffix strategy ablation hard transfer", "Robot suffix strategy ablation 30-seed", "", "final_transfer_rauc", "bgr_hard", "uniform", True, 30, 0),
        SignificanceCheck("suffix strategy ablation hard AULC", "Robot suffix strategy ablation 30-seed", "", "rauc_aulc", "bgr_hard", "uniform", True, 30, 0),
        SignificanceCheck("suffix held-out replication clean sign test", "Robot suffix replication 30-seed", "", "final_clean", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("suffix held-out replication object RAUC sign test", "Robot suffix replication 30-seed", "", "final_rauc", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("suffix held-out replication transfer sign test", "Robot suffix replication 30-seed", "", "final_transfer_rauc", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("suffix held-out replication AULC sign test", "Robot suffix replication 30-seed", "", "rauc_aulc", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("suffix full 30-seed RAUC vs clean-only sign test", "Robot suffix coverage-full 30-seed", "", "final_rauc", "bgr_broad", "clean_ft", True, 30, 0),
        SignificanceCheck("suffix full 30-seed RAUC vs fixed sign test", "Robot suffix coverage-full 30-seed", "", "final_rauc", "bgr_broad", "fixed", True, 30, 0),
        SignificanceCheck("suffix full 30-seed RAUC vs failure-only sign test", "Robot suffix coverage-full 30-seed", "", "final_rauc", "bgr_broad", "failure_only", True, 30, 0),
        SignificanceCheck("suffix full 30-seed RAUC vs loss-priority sign test", "Robot suffix coverage-full 30-seed", "", "final_rauc", "bgr_broad", "loss_priority", True, 30, 0),
        SignificanceCheck("suffix full 30-seed RAUC vs uniform sign test", "Robot suffix coverage-full 30-seed", "", "final_rauc", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("suffix full 30-seed transfer sign test", "Robot suffix coverage-full 30-seed", "", "final_transfer_rauc", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("suffix full 30-seed AULC sign test", "Robot suffix coverage-full 30-seed", "", "rauc_aulc", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("suffix full held-out 30-seed RAUC vs clean-only sign test", "Robot suffix coverage-full replication 30-seed", "", "final_rauc", "bgr_broad", "clean_ft", True, 30, 0),
        SignificanceCheck("suffix full held-out 30-seed RAUC vs fixed sign test", "Robot suffix coverage-full replication 30-seed", "", "final_rauc", "bgr_broad", "fixed", True, 30, 0),
        SignificanceCheck("suffix full held-out 30-seed RAUC vs failure-only sign test", "Robot suffix coverage-full replication 30-seed", "", "final_rauc", "bgr_broad", "failure_only", True, 30, 0),
        SignificanceCheck("suffix full held-out 30-seed RAUC vs loss-priority sign test", "Robot suffix coverage-full replication 30-seed", "", "final_rauc", "bgr_broad", "loss_priority", True, 30, 0),
        SignificanceCheck("suffix full held-out 30-seed RAUC vs uniform sign test", "Robot suffix coverage-full replication 30-seed", "", "final_rauc", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("suffix full held-out 30-seed transfer sign test", "Robot suffix coverage-full replication 30-seed", "", "final_transfer_rauc", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("suffix full held-out 30-seed AULC sign test", "Robot suffix coverage-full replication 30-seed", "", "rauc_aulc", "bgr_broad", "uniform", True, 30, 0),
        SignificanceCheck("estimator boundary hit sign test", "Estimator 15-seed", "", "boundary_hit_rate", "active", "uniform", True, 15, 0),
        SignificanceCheck("estimator 30-seed boundary-hit confirmation", "Estimator 30-seed", "", "boundary_hit_rate", "active", "uniform", True, 30, 0),
        SignificanceCheck("estimator 30-seed r80 MAE confirmation", "Estimator 30-seed", "", "r80_mae", "active", "uniform", True, 30, 0),
        SignificanceCheck("estimator 30-seed RAUC MAE confirmation", "Estimator 30-seed", "", "rauc_mae", "active", "uniform", True, 24, 6, 0, 0.002),
    ]
    for target_margin in [0.26, 0.32, 0.38, 0.46, 0.54]:
        for metric in ["final_rauc", "rauc_aulc"]:
            checks.append(
                SignificanceCheck(
                    f"grid target {target_margin:.2f} {metric} sign test",
                    "Grid margin target sensitivity 15-seed",
                    f"target_margin={target_margin:.2f}",
                    metric,
                    f"bgr_target_{target_margin:.2f}",
                    "uniform",
                    True,
                    15,
                    0,
                )
            )
        for metric in ["final_rauc", "rauc_aulc", "final_clean"]:
            checks.append(
                SignificanceCheck(
                    f"grid target 30-seed {target_margin:.2f} {metric} confirmation",
                    "Grid margin target sensitivity 30-seed",
                    f"target_margin={target_margin:.2f}",
                    metric,
                    f"bgr_target_{target_margin:.2f}",
                    "uniform",
                    True,
                    30,
                    0,
                )
            )
    for regime in ["high_obstacle", "low_obstacle", "nominal"]:
        for metric in ["final_rauc", "rauc_aulc"]:
            checks.append(
                SignificanceCheck(
                    f"grid regime {regime} {metric} sign test",
                    "Grid margin regime sensitivity 15-seed",
                    f"regime={regime}",
                    metric,
                    "bgr",
                    "uniform",
                    True,
                    15,
                    0,
                )
            )
            checks.append(
                SignificanceCheck(
                    f"grid regime 30-seed {regime} {metric} confirmation",
                    "Grid margin regime sensitivity 30-seed",
                    f"regime={regime}",
                    metric,
                    "bgr",
                    "uniform",
                    True,
                    30,
                    0,
                )
            )
    for learning_rate, supports_final_rauc in [(0.015, True), (0.030, True), (0.060, False)]:
        checks.append(
            SignificanceCheck(
                f"grid learning-rate {learning_rate:.3f} final_rauc sign test",
                "Grid margin learning-rate sensitivity 15-seed",
                f"learning_rate={learning_rate:.3f}",
                "final_rauc",
                "bgr",
                "uniform",
                supports_final_rauc,
                15 if supports_final_rauc else 1,
                0 if supports_final_rauc else 14,
            )
        )
        checks.append(
            SignificanceCheck(
                f"grid learning-rate {learning_rate:.3f} AULC sign test",
                "Grid margin learning-rate sensitivity 15-seed",
                f"learning_rate={learning_rate:.3f}",
                "rauc_aulc",
                "bgr",
                "uniform",
                True,
                15,
                0,
            )
        )
        checks.append(
            SignificanceCheck(
                f"grid learning-rate 30-seed {learning_rate:.3f} final_rauc confirmation",
                "Grid margin learning-rate sensitivity 30-seed",
                f"learning_rate={learning_rate:.3f}",
                "final_rauc",
                "bgr",
                "uniform",
                supports_final_rauc,
                30 if supports_final_rauc else 1,
                0 if supports_final_rauc else 29,
            )
        )
        checks.append(
            SignificanceCheck(
                f"grid learning-rate 30-seed {learning_rate:.3f} AULC confirmation",
                "Grid margin learning-rate sensitivity 30-seed",
                f"learning_rate={learning_rate:.3f}",
                "rauc_aulc",
                "bgr",
                "uniform",
                True,
                30,
                0,
            )
        )
    for stress_case in ["diffuse_boundary", "low_feasibility", "sharp_low_margin"]:
        for metric in ["final_rauc", "rauc_aulc"]:
            checks.append(
                SignificanceCheck(
                    f"grid stress {stress_case} {metric} sign test",
                    "Grid margin stress sensitivity 15-seed",
                    f"stress_case={stress_case}",
                    metric,
                    "bgr",
                    "uniform",
                    True,
                    15,
                    0,
                )
            )
            checks.append(
                SignificanceCheck(
                    f"grid stress 30-seed {stress_case} {metric} confirmation",
                    "Grid margin stress sensitivity 30-seed",
                    f"stress_case={stress_case}",
                    metric,
                    "bgr",
                    "uniform",
                    True,
                    30,
                    0,
                )
            )
    suffix_stress_median_wins = {
        "diffuse_boundary": 0,
        "high_clutter": 1,
        "low_teacher": 0,
        "tight_feasible": 3,
    }
    for stress_case in ["diffuse_boundary", "high_clutter", "low_teacher", "tight_feasible"]:
        for metric in ["final_clean", "final_rauc", "final_transfer_rauc", "rauc_aulc"]:
            checks.append(
                SignificanceCheck(
                    f"suffix stress 30-seed {stress_case} {metric} confirmation",
                    "Robot suffix stress sensitivity 30-seed",
                    f"stress_case={stress_case}",
                    metric,
                    "bgr_broad",
                    "uniform",
                    True,
                    30,
                    0,
                )
            )
        checks.append(
            SignificanceCheck(
                f"suffix stress 30-seed {stress_case} median-r80 caveat",
                "Robot suffix stress sensitivity 30-seed",
                f"stress_case={stress_case}",
                "final_median_r80",
                "bgr_broad",
                "uniform",
                False,
                suffix_stress_median_wins[stress_case],
                30 - suffix_stress_median_wins[stress_case],
            )
        )
    for step in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]:
        checks.append(
            SignificanceCheck(
                f"grid learning curve step {step} sign test",
                "Grid margin learning curve 15-seed",
                f"step={step}",
                "rauc",
                "bgr",
                "uniform",
                True,
                15,
                0,
            )
        )
    return checks


def validate_significance_checks(figures_dir: Path) -> list[str]:
    rows = read_csv_rows(figures_dir / "significance_tests.csv")
    messages: list[str] = []
    for check in build_significance_checks():
        row = find_significance_row(
            rows,
            benchmark=check.benchmark,
            condition=check.condition,
            metric=check.metric,
            treatment=check.treatment,
            baseline=check.baseline,
        )
        supports = row["supports_treatment"] == "true"
        wins = int(row["paired_wins"])
        losses = int(row["paired_losses"])
        ties = int(row["paired_ties"])
        p_value = float(row["two_sided_sign_test_p"])
        if supports != check.supports_treatment:
            raise ValueError(f"{check.label}: expected supports_treatment={check.supports_treatment}, got {supports}")
        if (wins, losses, ties) != (check.wins, check.losses, check.ties):
            raise ValueError(f"{check.label}: expected W/L/T {check.wins}/{check.losses}/{check.ties}, got {wins}/{losses}/{ties}")
        if p_value > check.max_p:
            raise ValueError(f"{check.label}: expected p <= {check.max_p}, got {p_value}")
        messages.append(f"{check.label}: W/L/T {wins}/{losses}/{ties}, p={fmt_pvalue(p_value)}")
    messages.extend(validate_grid_margin_full_30_significance(rows))
    messages.extend(validate_suffix_coverage_full_significance(rows))
    return messages


def validate_grid_margin_full_30_significance(rows: list[dict[str, str]]) -> list[str]:
    benchmark = "Grid margin full 30-seed"
    grid_rows = [row for row in rows if row["benchmark"] == benchmark]
    expected = {
        ("final_rauc", "uniform"),
        ("rauc_aulc", "uniform"),
        ("final_clean", "uniform"),
        ("final_median_r80", "uniform"),
        ("final_rauc", "fixed"),
        ("final_rauc", "failure_only"),
        ("final_rauc", "plr_loss"),
        ("rauc_aulc", "fixed"),
        ("rauc_aulc", "failure_only"),
        ("rauc_aulc", "plr_loss"),
    }
    found = {(row["metric"], row["baseline"]) for row in grid_rows}
    missing = expected - found
    extra = found - expected
    if missing or extra:
        raise ValueError(
            "Grid margin full 30-seed significance rows mismatch: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    bad_n = [
        f"{row['metric']}/{row['baseline']} n={row['n']}"
        for row in grid_rows
        if int(row["n"]) != 30
    ]
    if bad_n:
        raise ValueError(f"Grid margin full 30-seed significance rows must use 30 seeds: {', '.join(bad_n)}")
    unsupported = [
        f"{row['metric']}/{row['baseline']} supports={row['supports_treatment']} W/L/T={row['paired_wins']}/{row['paired_losses']}/{row['paired_ties']}"
        for row in grid_rows
        if row["supports_treatment"] != "true"
        or row["paired_wins"] != "30"
        or row["paired_losses"] != "0"
        or row["paired_ties"] != "0"
    ]
    if unsupported:
        raise ValueError(
            "Grid margin full 30-seed significance rows must support BGR: "
            + ", ".join(unsupported)
        )
    return ["Grid margin full 30-seed significance rows use complete positive 30-seed comparisons"]


def validate_suffix_coverage_full_significance(rows: list[dict[str, str]]) -> list[str]:
    expected = {
        ("final_rauc", "clean_ft"),
        ("final_rauc", "fixed"),
        ("final_rauc", "failure_only"),
        ("final_rauc", "loss_priority"),
        ("final_rauc", "uniform"),
        ("final_transfer_rauc", "uniform"),
        ("rauc_aulc", "uniform"),
    }
    benchmarks = [
        "Robot suffix coverage-full 30-seed",
        "Robot suffix coverage-full replication 30-seed",
    ]
    for benchmark in benchmarks:
        suffix_rows = [row for row in rows if row["benchmark"] == benchmark]
        found = {(row["metric"], row["baseline"]) for row in suffix_rows}
        missing = expected - found
        extra = found - expected
        if missing or extra:
            raise ValueError(
                f"{benchmark} significance rows mismatch: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        bad_n = [
            f"{row['metric']}/{row['baseline']} n={row['n']}"
            for row in suffix_rows
            if int(row["n"]) != 30
        ]
        if bad_n:
            raise ValueError(f"{benchmark} significance rows must use 30 seeds: {', '.join(bad_n)}")
        unsupported = [
            f"{row['metric']}/{row['baseline']} supports={row['supports_treatment']} W/L/T={row['paired_wins']}/{row['paired_losses']}/{row['paired_ties']}"
            for row in suffix_rows
            if row["supports_treatment"] != "true"
            or row["paired_wins"] != "30"
            or row["paired_losses"] != "0"
            or row["paired_ties"] != "0"
        ]
        if unsupported:
            raise ValueError(
                f"{benchmark} significance rows must support BGR-Coverage: "
                + ", ".join(unsupported)
            )
    return ["Robot suffix coverage-full significance rows use complete positive original and held-out 30-seed comparisons"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=Path("paper/main.tex"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    paper_text = args.paper.read_text(encoding="utf-8")
    claims = build_claims(args.results_dir, args.figures_dir)
    missing = missing_claims(paper_text, claims)
    if missing:
        print("Paper claim check failed:")
        for claim in missing:
            print(f"- {claim.label}: missing {claim.snippet!r} from {claim.source}")
        return 1
    forbidden = forbidden_terms(paper_text)
    if forbidden:
        print("Paper framing check failed:")
        for term in forbidden:
            print(f"- forbidden term remains in paper: {term!r}")
        return 1
    unverified = unverified_result_claims(paper_text, args.results_dir)
    if unverified:
        print("Paper unverified-result check failed:")
        for item in unverified:
            print(f"- {item}")
        return 1
    paper_negative = forbidden_paper_negative_claims(paper_text)
    if paper_negative:
        print("Paper paper-negative diagnostic check failed:")
        for token in paper_negative:
            print(f"- paper-negative OpenVLA diagnostic appears in paper: {token!r}")
        return 1
    effect_framing = effect_size_framing_issues(paper_text)
    if effect_framing:
        print("Paper effect-size framing check failed:")
        for item in effect_framing:
            print(f"- missing or weakened framing: {item}")
        return 1

    try:
        significance_messages = validate_significance_checks(args.figures_dir)
    except ValueError as exc:
        print("Paper significance check failed:")
        print(f"- {exc}")
        return 1

    for claim in claims:
        print(f"[ok] {claim.label}: {claim.snippet} ({claim.source})")
    for message in significance_messages:
        print(f"[ok] {message} (paper/figures/significance_tests.csv)")
    print("[ok] effect-size-first framing retained in paper text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
