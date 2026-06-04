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
                f"RAUC {fmt(mean_metric(grid, 'failure_only', 'final_rauc'), 4)}",
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
                f"held-out seeds 30--59 BGR-vs-uniform replication gives "
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
                f"Pooling original and held-out grid sweeps gives "
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
                f"BGR RAUC {fmt(min(regime_bgr_raucs), 3)}--{fmt(max(regime_bgr_raucs), 3)} "
                f"vs. uniform {fmt(mean(regime_uniform_raucs), 3)}"
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
                f"stress RAUC {fmt(min(stress_bgr_raucs), 3)}--{fmt(max(stress_bgr_raucs), 3)} "
                f"vs. uniform {fmt(min(stress_uniform_raucs), 3)}--{fmt(max(stress_uniform_raucs), 3)}"
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
                f"uniform remains higher on median $r_{{80}}$ "
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
    claims.append(
        Claim(
            "OpenVLA official checkpoint success",
            ratio(official["successes"], official["episodes"]),
            "results/openvla_oft_sanity_eval_sanity_v1/summary.csv",
        )
    )

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
    claims.append(
        Claim(
            "OpenVLA final clean BGR vs random",
            f"BGR and matched random both score {ratio(final_bgr_clean['successes'], final_bgr_clean['episodes'])} clean episodes",
            "results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv",
        )
    )

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
    claims.append(
        Claim(
            "OpenVLA original mean visual perturbation BGR vs random",
            (
                f"{fmt(mean_success_rate(final_perturb, 'bgr', exclude_perturbations={'identity'}), 4)} "
                f"vs. {fmt(mean_success_rate(final_perturb, 'random', exclude_perturbations={'identity'}), 4)}"
            ),
            "results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv",
        )
    )
    claims.append(
        Claim(
            "OpenVLA offset mean visual perturbation BGR vs random",
            (
                f"{fmt(mean_success_rate(offset_perturb, 'bgr', exclude_perturbations={'identity'}), 4)} "
                f"vs. {fmt(mean_success_rate(offset_perturb, 'random', exclude_perturbations={'identity'}), 4)}"
            ),
            "results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/summary.csv",
        )
    )
    claims.append(
        Claim(
            "OpenVLA offset official mean visual perturbation",
            f"official reaches {fmt(mean_success_rate(offset_perturb, 'official', exclude_perturbations={'identity'}), 4)}",
            "results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/summary.csv",
        )
    )
    p1024_pooled_bgr = pooled_success_rate([final_perturb, offset_perturb], "bgr", exclude_perturbations={"identity"})
    p1024_pooled_random = pooled_success_rate(
        [final_perturb, offset_perturb], "random", exclude_perturbations={"identity"}
    )
    p1024_pooled_official = pooled_success_rate(
        [final_perturb, offset_perturb], "official", exclude_perturbations={"identity"}
    )
    claims.append(
        Claim(
            "OpenVLA pooled mean visual perturbation BGR vs random",
            (
                f"{fmt(p1024_pooled_bgr, 4)} "
                f"vs. {fmt(p1024_pooled_random, 4)} "
                f"for random"
            ),
            "results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv and offset3 summary.csv",
        )
    )
    claims.append(
        Claim(
            "OpenVLA pooled official mean visual perturbation",
            f"trailing the unadapted official checkpoint at {fmt(p1024_pooled_official, 4)}",
            "results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv and offset3 summary.csv",
        )
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
    claims.append(
        Claim(
            "OpenVLA p2048 clean BGR vs random",
            f"BGR and random tie clean ({ratio(p2048_bgr_clean['successes'], p2048_bgr_clean['episodes'])} each)",
            "results/openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv",
        )
    )

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
    claims.append(
        Claim(
            "OpenVLA original p2048 mean visual perturbation BGR vs random vs official",
            (
                f"original perturbations give {fmt(p2048_bgr_visual, 4)} "
                f"vs. {fmt(p2048_random_visual, 4)} "
                f"random, tying official at {fmt(p2048_official_visual, 4)}"
            ),
            "results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv",
        )
    )
    p2048_offset_bgr = mean_success_rate(p2048_offset_perturb, "bgr", exclude_perturbations={"identity"})
    p2048_offset_random = mean_success_rate(p2048_offset_perturb, "random", exclude_perturbations={"identity"})
    p2048_offset_official = mean_success_rate(p2048_offset_perturb, "official", exclude_perturbations={"identity"})
    claims.append(
        Claim(
            "OpenVLA p2048 offset mean visual perturbation BGR vs random vs official",
            (
                f"Offset-3 gives {fmt(p2048_offset_bgr, 4)} "
                f"vs. {fmt(p2048_offset_random, 4)} random; "
                f"official reaches {fmt(p2048_offset_official, 4)}"
            ),
            "results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/summary.csv",
        )
    )
    p2048_pooled_bgr = pooled_success_rate(
        [p2048_perturb, p2048_offset_perturb], "bgr", exclude_perturbations={"identity"}
    )
    p2048_pooled_random = pooled_success_rate(
        [p2048_perturb, p2048_offset_perturb], "random", exclude_perturbations={"identity"}
    )
    p2048_pooled_official = pooled_success_rate(
        [p2048_perturb, p2048_offset_perturb], "official", exclude_perturbations={"identity"}
    )
    claims.append(
        Claim(
            "OpenVLA p2048 pooled mean visual perturbation BGR vs random vs official",
            (
                f"Pooling p2048 gives BGR {fmt(p2048_pooled_bgr, 4)} "
                f"vs. {fmt(p2048_pooled_random, 4)} random, "
                f"trailing official at {fmt(p2048_pooled_official, 4)}"
            ),
            "results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv and offset3 summary.csv",
        )
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
            (
                f"full-goal identity audit gives {fullgoal_clean_ratios['bgr']} clean successes "
                "for BGR, matched random, and the official checkpoint"
            ),
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
                f"BGR {bgr_successes}/{bgr_episodes} perturbed successes, tying official "
                f"and trailing matched random by one episode ({random_successes}/{random_episodes})"
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
                f"300-step image-augmentation continuation gives BGR and matched random "
                f"{imageaug_bgr_successes}/{imageaug_bgr_episodes} perturbed successes each, "
                f"only one episode above official ({imageaug_official_successes}/{imageaug_official_episodes}), "
                "while BGR trails both on identity"
            ),
            "results/openvla_oft_perturb_eval_cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1/summary.csv",
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
    return [claim for claim in claims if claim.snippet not in paper_text]


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
        "367/400": p2048_fullgoal_summary_paths,
        "300-step image-augmentation continuation": p2048_imageaug_300_summary_paths,
        "368/400": p2048_imageaug_300_summary_paths,
        "action-label/TFDS plumbing validates": action_tfds_summary_paths,
        "2,048-transition matched BGR/random exports": action_tfds_summary_paths,
    }
    missing: list[str] = []
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
