#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


BENCHMARKS = {
    "toy_30seed_v1": {
        "label": "Synthetic",
        "primary": ["bgr", "uniform", "fixed", "failure_only", "plr_loss"],
        "display": {
            "bgr": "BGR",
            "uniform": "Uniform",
            "fixed": "Fixed",
            "failure_only": "Failure",
            "plr_loss": "Loss-priority",
        },
    },
    "grid_margin_full_30seed_v1": {
        "label": "GridMargin",
        "primary": ["bgr", "uniform"],
        "display": {
            "bgr": "BGR",
            "uniform": "Uniform",
        },
    },
    "suffix_strategy_coverage_30seed_v1": {
        "label": "RobotSuffix",
        "primary": ["bgr_broad", "uniform"],
        "display": {
            "bgr_broad": "BGR-Coverage",
            "uniform": "Uniform",
        },
    },
    "openml_diabetes_margin_30seed_v1": {
        "label": "OpenML diabetes",
        "file": "per_seed.csv",
        "primary": ["bgr", "uniform", "fixed"],
        "metrics": [("final_rauc", "RAUC")],
        "display": {
            "bgr": "BGR",
            "uniform": "Uniform",
            "fixed": "Fixed-radius",
        },
    },
    "openml_numeric_external_fixed_target2_30seed_v1": {
        "label": "OpenML blood",
        "file": "per_seed.csv",
        "filter_dataset": "blood-transfusion-service-center",
        "primary": ["bgr", "uniform", "fixed"],
        "metrics": [("final_rauc", "RAUC")],
        "display": {
            "bgr": "BGR",
            "uniform": "Uniform",
            "fixed": "Fixed-radius",
        },
    },
}

METRICS = [
    ("final_clean", "Clean"),
    ("final_rauc", "RAUC"),
    ("final_median_r80", "MedianR80"),
    ("rauc_aulc", "AULC"),
]

ESTIMATOR_RUN = "estimator_pair_30seed_v1"
GRID_FULL_RUN = "grid_margin_full_30seed_v1"
ABLATION_RUN = "grid_margin_ablation_30seed_v1"
TARGET_SENSITIVITY_RUN = "grid_margin_target_sensitivity_30seed_v1"
LEARNING_RATE_SENSITIVITY_RUN = "grid_margin_learning_rate_sensitivity_30seed_v1"
REGIME_SENSITIVITY_RUN = "grid_margin_regime_sensitivity_30seed_v1"
STRESS_SENSITIVITY_RUN = "grid_margin_stress_sensitivity_30seed_v1"
SUFFIX_STRESS_SENSITIVITY_RUN = "suffix_stress_sensitivity_30seed_v1"
GRID_LEARNING_CURVE_RUN = "grid_margin_full_15seed_v1"
OPENVLA_RECOVERY_RUN = "libero_openvla_recovery_v1"
OPENVLA_SELECTION_RUN = "libero_openvla_boundary_selection_balanced_v1"
OPENVLA_FULL_CLEAN_RUN = "openvla_oft_clean_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1"
OPENVLA_FULL_PERTURB_RUN = "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1"
OPENVLA_IMAGEAUG_300_RUN = "openvla_oft_perturb_eval_cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1"
OPENVLA_LOWLR_1000_RUN = "openvla_oft_perturb_eval_cleanmix_p2048_step51000_lr1em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1"
OPENVLA_WEIGHTED_PERTURB_RUN = (
    "openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_step50500_lr5em7_identitylora_"
    "imageaug_officialtrainstats_fullgoal10x10_perturb_v1"
)
OPENVLA_PROXIMAL_PERTURB_RUN = (
    "openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_proxanchor_l2_1em0_ddpgradfix_v1_"
    "step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1"
)
OPENVLA_PERTURB_ONLY_ANCHOR_RUN = (
    "openvla_oft_perturb_eval_p2048unique_perturbonly_anchor_prereg_perturbonly_proxanchor_l2_5em0_"
    "step50300_lr2em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="paper/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str | float | int]] = []
    effect_rows: list[dict[str, str | float]] = []
    loaded: dict[str, list[dict[str, str]]] = {}

    for run_name, spec in BENCHMARKS.items():
        path = results_dir / run_name / str(spec.get("file", "summary.csv"))
        rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
        if "filter_dataset" in spec:
            rows = [row for row in rows if row.get("dataset") == spec["filter_dataset"]]
        loaded[run_name] = rows
        metrics = spec.get("metrics", METRICS)
        for method in spec["primary"]:
            method_rows = [row for row in rows if row["method"] == method]
            for metric, metric_label in metrics:
                vals = [float(row[metric]) for row in method_rows]
                summary_rows.append(
                    {
                        "benchmark": spec["label"],
                        "method": spec["display"].get(method, method),
                        "metric": metric_label,
                        "n": len(vals),
                        "mean": mean(vals),
                        "sem": sem(vals),
                    }
                )

        treatment = "bgr" if "bgr" in spec["primary"] else spec["primary"][0]
        for baseline in [m for m in spec["primary"] if m != treatment]:
            for metric, metric_label in metrics:
                bgr_vals = [float(row[metric]) for row in rows if row["method"] == treatment]
                base_vals = [float(row[metric]) for row in rows if row["method"] == baseline]
                diffs = [a - b for a, b in zip(bgr_vals, base_vals, strict=True)]
                effect_rows.append(
                    {
                        "benchmark": spec["label"],
                        "treatment": spec["display"].get(treatment, treatment),
                        "baseline": spec["display"].get(baseline, baseline),
                        "metric": metric_label,
                        "mean_delta": mean(diffs),
                        "sem_delta": sem(diffs),
                    }
                )

    write_csv(out_dir / "summary_stats.csv", summary_rows)
    write_csv(out_dir / "bgr_deltas.csv", effect_rows)
    write_latex_table(out_dir / "summary_table.tex", summary_rows)
    estimator_rows = load_estimator(results_dir / ESTIMATOR_RUN / "summary.csv")
    if estimator_rows:
        write_csv(out_dir / "estimator_stats.csv", estimator_rows)
        write_estimator_table(out_dir / "estimator_table.tex", estimator_rows)
    grid_full_rows = load_grid_full(results_dir / GRID_FULL_RUN / "summary.csv")
    if grid_full_rows:
        write_csv(out_dir / "grid_margin_full_stats.csv", grid_full_rows)
        write_grid_full_table(out_dir / "grid_margin_full_table.tex", grid_full_rows)
    ablation_rows = load_ablation(results_dir / ABLATION_RUN / "summary.csv")
    if ablation_rows:
        write_csv(out_dir / "grid_margin_ablation_stats.csv", ablation_rows)
        write_ablation_table(out_dir / "grid_margin_ablation_table.tex", ablation_rows)
    target_sensitivity_rows = load_target_sensitivity(results_dir / TARGET_SENSITIVITY_RUN / "summary.csv")
    if target_sensitivity_rows:
        write_csv(out_dir / "grid_margin_target_sensitivity_stats.csv", target_sensitivity_rows)
        write_target_sensitivity_table(out_dir / "grid_margin_target_sensitivity_table.tex", target_sensitivity_rows)
    learning_rate_sensitivity_rows = load_learning_rate_sensitivity(
        results_dir / LEARNING_RATE_SENSITIVITY_RUN / "summary.csv"
    )
    if learning_rate_sensitivity_rows:
        write_csv(out_dir / "grid_margin_learning_rate_sensitivity_stats.csv", learning_rate_sensitivity_rows)
        write_learning_rate_sensitivity_table(
            out_dir / "grid_margin_learning_rate_sensitivity_table.tex",
            learning_rate_sensitivity_rows,
        )
    regime_sensitivity_rows = load_regime_sensitivity(results_dir / REGIME_SENSITIVITY_RUN / "summary.csv")
    if regime_sensitivity_rows:
        write_csv(out_dir / "grid_margin_regime_sensitivity_stats.csv", regime_sensitivity_rows)
        write_regime_sensitivity_table(
            out_dir / "grid_margin_regime_sensitivity_table.tex",
            regime_sensitivity_rows,
        )
    stress_sensitivity_rows = load_stress_sensitivity(results_dir / STRESS_SENSITIVITY_RUN / "summary.csv")
    if stress_sensitivity_rows:
        write_csv(out_dir / "grid_margin_stress_sensitivity_stats.csv", stress_sensitivity_rows)
        write_stress_sensitivity_table(
            out_dir / "grid_margin_stress_sensitivity_table.tex",
            stress_sensitivity_rows,
        )
    suffix_stress_sensitivity_rows = load_stress_sensitivity(
        results_dir / SUFFIX_STRESS_SENSITIVITY_RUN / "summary.csv",
        treatment_method="bgr_broad",
        treatment_label="BGR-Coverage",
        display={
            "low_teacher": "Low teacher",
            "high_clutter": "High clutter",
            "tight_feasible": "Tight feasible",
            "diffuse_boundary": "Diffuse boundary",
        },
    )
    if suffix_stress_sensitivity_rows:
        write_csv(out_dir / "suffix_stress_sensitivity_stats.csv", suffix_stress_sensitivity_rows)
        write_stress_sensitivity_table(
            out_dir / "suffix_stress_sensitivity_table.tex",
            suffix_stress_sensitivity_rows,
        )
    learning_curve_rows = load_grid_learning_curve(results_dir / GRID_LEARNING_CURVE_RUN / "results.json")
    if learning_curve_rows:
        write_csv(out_dir / "grid_margin_learning_curve_stats.csv", learning_curve_rows)
        write_grid_learning_curve_table(out_dir / "grid_margin_learning_curve_table.tex", learning_curve_rows)
    openvla_rows = load_openvla(
        results_dir / OPENVLA_RECOVERY_RUN / "summary.csv",
        results_dir / OPENVLA_SELECTION_RUN / "aggregate.csv",
    )
    if openvla_rows:
        write_csv(out_dir / "openvla_stats.csv", openvla_rows)
        write_openvla_table(out_dir / "openvla_table.tex", openvla_rows)
    openvla_adaptation_rows = load_openvla_adaptation(results_dir)
    if openvla_adaptation_rows:
        write_openvla_adaptation_table(out_dir / "openvla_adaptation_table.tex", openvla_adaptation_rows)
    try:
        make_boundary_intuition_figure(out_dir, results_dir)
        make_figures(out_dir, summary_rows)
        if estimator_rows:
            make_estimator_figure(out_dir, estimator_rows)
        if learning_curve_rows:
            make_grid_learning_curve_figure(out_dir, learning_curve_rows)
    except Exception as exc:  # pragma: no cover - optional plotting path.
        print(f"[warn] skipped figure generation: {exc}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path: Path, rows: list[dict]) -> None:
    selected = [row for row in rows if row["metric"] in {"Clean", "RAUC", "MedianR80", "AULC"}]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{llcccc}\n")
        handle.write("\\hline\n")
        handle.write("Benchmark & Method & Clean & RAUC & Median $r_{80}$ & AULC \\\\\n")
        handle.write("\\hline\n")
        by_key: dict[tuple[str, str], dict[str, str]] = {}
        for row in selected:
            by_key.setdefault((str(row["benchmark"]), str(row["method"])), {})[str(row["metric"])] = fmt(row)
        for benchmark in ["Synthetic", "GridMargin", "OpenML diabetes", "OpenML blood", "RobotSuffix"]:
            for (bench, method), vals in by_key.items():
                if bench != benchmark:
                    continue
                handle.write(
                    f"{bench} & {method} & {vals.get('Clean','--')} & {vals.get('RAUC','--')} & "
                    f"{vals.get('MedianR80','--')} & {vals.get('AULC','--')} \\\\\n"
                )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")


def load_estimator(path: Path) -> list[dict[str, str | float | int]]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    out: list[dict[str, str | float | int]] = []
    display = {"active": "Active BGR", "coarse": "Coarse", "uniform": "Uniform"}
    for method in ["active", "coarse", "uniform"]:
        items = [row for row in rows if row["method"] == method]
        if not items:
            continue
        out.append(
            {
                "method": display[method],
                "n": len(items),
                "probes_per_state": int(float(items[0]["probes_per_state"])),
                "r80_mae_mean": mean([float(row["r80_mae"]) for row in items]),
                "r80_mae_sem": sem([float(row["r80_mae"]) for row in items]),
                "rauc_mae_mean": mean([float(row["rauc_mae"]) for row in items]),
                "rauc_mae_sem": sem([float(row["rauc_mae"]) for row in items]),
                "hit_rate_mean": mean([float(row["boundary_hit_rate"]) for row in items]),
                "hit_rate_sem": sem([float(row["boundary_hit_rate"]) for row in items]),
            }
        )
    return out


def write_estimator_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lccc}\n")
        handle.write("\\hline\n")
        handle.write("Estimator & $r_{80}$ MAE & RAUC MAE & Hit rate \\\\\n")
        handle.write("\\hline\n")
        for row in rows:
            handle.write(
                f"{row['method']} & "
                f"{float(row['r80_mae_mean']):.3f}$\\pm${float(row['r80_mae_sem']):.3f} & "
                f"{float(row['rauc_mae_mean']):.3f}$\\pm${float(row['rauc_mae_sem']):.3f} & "
                f"{float(row['hit_rate_mean']):.3f}$\\pm${float(row['hit_rate_sem']):.3f} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")


def load_grid_full(path: Path) -> list[dict[str, str | float | int]]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    display = {
        "bgr": "BGR",
        "uniform": "Uniform",
        "failure_only": "Failure-only",
        "plr_loss": "Loss-priority",
        "fixed": "Fixed radius",
    }
    out: list[dict[str, str | float | int]] = []
    for method in ["bgr", "uniform", "failure_only", "plr_loss", "fixed"]:
        items = [row for row in rows if row["method"] == method]
        if not items:
            continue
        out.append(
            {
                "method": display[method],
                "n": len(items),
                "clean_mean": mean([float(row["final_clean"]) for row in items]),
                "clean_sem": sem([float(row["final_clean"]) for row in items]),
                "rauc_mean": mean([float(row["final_rauc"]) for row in items]),
                "rauc_sem": sem([float(row["final_rauc"]) for row in items]),
                "r80_mean": mean([float(row["final_median_r80"]) for row in items]),
                "r80_sem": sem([float(row["final_median_r80"]) for row in items]),
                "aulc_mean": mean([float(row["rauc_aulc"]) for row in items]),
                "aulc_sem": sem([float(row["rauc_aulc"]) for row in items]),
            }
        )
    return out


def write_grid_full_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lcccc}\n")
        handle.write("\\hline\n")
        handle.write("Method & Clean & RAUC & Median $r_{80}$ & AULC \\\\\n")
        handle.write("\\hline\n")
        for row in rows:
            handle.write(
                f"{row['method']} & "
                f"{float(row['clean_mean']):.3f}$\\pm${float(row['clean_sem']):.3f} & "
                f"{float(row['rauc_mean']):.3f}$\\pm${float(row['rauc_sem']):.3f} & "
                f"{float(row['r80_mean']):.3f}$\\pm${float(row['r80_sem']):.3f} & "
                f"{float(row['aulc_mean']):.3f}$\\pm${float(row['aulc_sem']):.3f} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")


def load_ablation(path: Path) -> list[dict[str, str | float | int]]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    display = {
        "bgr": "BGR",
        "bgr_no_uncertainty": "No uncertainty",
        "bgr_no_sharpness": "No sharpness",
        "bgr_uniform_radius": "BGR state + uniform radius",
        "uniform": "Uniform state + uniform radius",
    }
    out: list[dict[str, str | float | int]] = []
    for method in ["bgr", "bgr_no_uncertainty", "bgr_no_sharpness", "bgr_uniform_radius", "uniform"]:
        items = [row for row in rows if row["method"] == method]
        if not items:
            continue
        out.append(
            {
                "method": display[method],
                "n": len(items),
                "clean_mean": mean([float(row["final_clean"]) for row in items]),
                "clean_sem": sem([float(row["final_clean"]) for row in items]),
                "rauc_mean": mean([float(row["final_rauc"]) for row in items]),
                "rauc_sem": sem([float(row["final_rauc"]) for row in items]),
                "r80_mean": mean([float(row["final_median_r80"]) for row in items]),
                "r80_sem": sem([float(row["final_median_r80"]) for row in items]),
                "aulc_mean": mean([float(row["rauc_aulc"]) for row in items]),
                "aulc_sem": sem([float(row["rauc_aulc"]) for row in items]),
            }
        )
    return out


def write_ablation_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lcccc}\n")
        handle.write("\\hline\n")
        handle.write("Method & Clean & RAUC & Median $r_{80}$ & AULC \\\\\n")
        handle.write("\\hline\n")
        for row in rows:
            handle.write(
                f"{row['method']} & "
                f"{float(row['clean_mean']):.3f}$\\pm${float(row['clean_sem']):.3f} & "
                f"{float(row['rauc_mean']):.3f}$\\pm${float(row['rauc_sem']):.3f} & "
                f"{float(row['r80_mean']):.3f}$\\pm${float(row['r80_sem']):.3f} & "
                f"{float(row['aulc_mean']):.3f}$\\pm${float(row['aulc_sem']):.3f} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")


def load_target_sensitivity(path: Path) -> list[dict[str, str | float | int]]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    out: list[dict[str, str | float | int]] = []
    targets = sorted({float(row["target_margin"]) for row in rows})
    for target_margin in targets:
        items = [row for row in rows if float(row["target_margin"]) == target_margin]
        out.append(
            {
                "target_margin": target_margin,
                "n": len(items),
                "clean_mean": mean([float(row["final_clean"]) for row in items]),
                "clean_sem": sem([float(row["final_clean"]) for row in items]),
                "rauc_mean": mean([float(row["final_rauc"]) for row in items]),
                "rauc_sem": sem([float(row["final_rauc"]) for row in items]),
                "r80_mean": mean([float(row["final_median_r80"]) for row in items]),
                "r80_sem": sem([float(row["final_median_r80"]) for row in items]),
                "aulc_mean": mean([float(row["rauc_aulc"]) for row in items]),
                "aulc_sem": sem([float(row["rauc_aulc"]) for row in items]),
            }
        )
    return out


def write_target_sensitivity_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{ccccc}\n")
        handle.write("\\hline\n")
        handle.write("Target $r_{80}$ & Clean & RAUC & Median $r_{80}$ & AULC \\\\\n")
        handle.write("\\hline\n")
        for row in rows:
            handle.write(
                f"{float(row['target_margin']):.2f} & "
                f"{float(row['clean_mean']):.3f}$\\pm${float(row['clean_sem']):.3f} & "
                f"{float(row['rauc_mean']):.3f}$\\pm${float(row['rauc_sem']):.3f} & "
                f"{float(row['r80_mean']):.3f}$\\pm${float(row['r80_sem']):.3f} & "
                f"{float(row['aulc_mean']):.3f}$\\pm${float(row['aulc_sem']):.3f} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")


def load_learning_rate_sensitivity(path: Path) -> list[dict[str, str | float | int]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    display = {"bgr": "BGR", "uniform": "Uniform"}
    out: list[dict[str, str | float | int]] = []
    for learning_rate in sorted({float(row["learning_rate"]) for row in rows}):
        for method in ["bgr", "uniform"]:
            items = [
                row
                for row in rows
                if float(row["learning_rate"]) == learning_rate and row["method"] == method
            ]
            if not items:
                continue
            out.append(
                {
                    "learning_rate": learning_rate,
                    "method": display[method],
                    "n": len(items),
                    "clean_mean": mean([float(row["final_clean"]) for row in items]),
                    "clean_sem": sem([float(row["final_clean"]) for row in items]),
                    "rauc_mean": mean([float(row["final_rauc"]) for row in items]),
                    "rauc_sem": sem([float(row["final_rauc"]) for row in items]),
                    "r80_mean": mean([float(row["final_median_r80"]) for row in items]),
                    "r80_sem": sem([float(row["final_median_r80"]) for row in items]),
                    "aulc_mean": mean([float(row["rauc_aulc"]) for row in items]),
                    "aulc_sem": sem([float(row["rauc_aulc"]) for row in items]),
                }
            )
    return out


def write_learning_rate_sensitivity_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{clcccc}\n")
        handle.write("\\hline\n")
        handle.write("LR & Method & Clean & RAUC & Median $r_{80}$ & AULC \\\\\n")
        handle.write("\\hline\n")
        for row in rows:
            handle.write(
                f"{float(row['learning_rate']):.3f} & {row['method']} & "
                f"{float(row['clean_mean']):.3f}$\\pm${float(row['clean_sem']):.3f} & "
                f"{float(row['rauc_mean']):.3f}$\\pm${float(row['rauc_sem']):.3f} & "
                f"{float(row['r80_mean']):.3f}$\\pm${float(row['r80_sem']):.3f} & "
                f"{float(row['aulc_mean']):.3f}$\\pm${float(row['aulc_sem']):.3f} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")


def load_regime_sensitivity(path: Path) -> list[dict[str, str | float | int]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    display = {
        "low_obstacle": "Low obstacle",
        "nominal": "Nominal",
        "high_obstacle": "High obstacle",
        "bgr": "BGR",
        "uniform": "Uniform",
    }
    out: list[dict[str, str | float | int]] = []
    regimes = sorted({row["regime"] for row in rows})
    for regime in regimes:
        for method in ["bgr", "uniform"]:
            items = [row for row in rows if row["regime"] == regime and row["method"] == method]
            if not items:
                continue
            out.append(
                {
                    "regime": regime,
                    "regime_label": display.get(regime, regime.replace("_", " ").title()),
                    "method": display[method],
                    "obstacle_prob": float(items[0]["obstacle_prob"]),
                    "grid_size": int(float(items[0]["grid_size"])),
                    "max_offset": int(float(items[0]["max_offset"])),
                    "n": len(items),
                    "clean_mean": mean([float(row["final_clean"]) for row in items]),
                    "clean_sem": sem([float(row["final_clean"]) for row in items]),
                    "rauc_mean": mean([float(row["final_rauc"]) for row in items]),
                    "rauc_sem": sem([float(row["final_rauc"]) for row in items]),
                    "r80_mean": mean([float(row["final_median_r80"]) for row in items]),
                    "r80_sem": sem([float(row["final_median_r80"]) for row in items]),
                    "aulc_mean": mean([float(row["rauc_aulc"]) for row in items]),
                    "aulc_sem": sem([float(row["rauc_aulc"]) for row in items]),
                }
            )
    return out


def write_regime_sensitivity_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{llcccc}\n")
        handle.write("\\hline\n")
        handle.write("Regime & Method & Clean & RAUC & Median $r_{80}$ & AULC \\\\\n")
        handle.write("\\hline\n")
        for row in rows:
            handle.write(
                f"{row['regime_label']} & {row['method']} & "
                f"{float(row['clean_mean']):.3f}$\\pm${float(row['clean_sem']):.3f} & "
                f"{float(row['rauc_mean']):.3f}$\\pm${float(row['rauc_sem']):.3f} & "
                f"{float(row['r80_mean']):.3f}$\\pm${float(row['r80_sem']):.3f} & "
                f"{float(row['aulc_mean']):.3f}$\\pm${float(row['aulc_sem']):.3f} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")


def load_stress_sensitivity(
    path: Path,
    treatment_method: str = "bgr",
    treatment_label: str = "BGR",
    display: dict[str, str] | None = None,
) -> list[dict[str, str | float | int]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    display_map = {
        "sharp_low_margin": "Sharp low-margin",
        "diffuse_boundary": "Diffuse boundary",
        "low_feasibility": "Low feasibility",
        treatment_method: treatment_label,
        "uniform": "Uniform",
    }
    if display:
        display_map.update(display)
    out: list[dict[str, str | float | int]] = []
    stress_cases = sorted({row["stress_case"] for row in rows})
    for stress_case in stress_cases:
        for method in [treatment_method, "uniform"]:
            items = [row for row in rows if row["stress_case"] == stress_case and row["method"] == method]
            if not items:
                continue
            out.append(
                {
                    "stress_case": stress_case,
                    "stress_label": display_map.get(stress_case, stress_case.replace("_", " ").title()),
                    "method": display_map[method],
                    "n": len(items),
                    "clean_mean": mean([float(row["final_clean"]) for row in items]),
                    "clean_sem": sem([float(row["final_clean"]) for row in items]),
                    "rauc_mean": mean([float(row["final_rauc"]) for row in items]),
                    "rauc_sem": sem([float(row["final_rauc"]) for row in items]),
                    "r80_mean": mean([float(row["final_median_r80"]) for row in items]),
                    "r80_sem": sem([float(row["final_median_r80"]) for row in items]),
                    "transfer_rauc_mean": mean([float(row.get("final_transfer_rauc", "nan")) for row in items])
                    if "final_transfer_rauc" in items[0]
                    else "",
                    "transfer_rauc_sem": sem([float(row.get("final_transfer_rauc", "nan")) for row in items])
                    if "final_transfer_rauc" in items[0]
                    else "",
                    "aulc_mean": mean([float(row["rauc_aulc"]) for row in items]),
                    "aulc_sem": sem([float(row["rauc_aulc"]) for row in items]),
                }
            )
    return out


def write_stress_sensitivity_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{llcccc}\n")
        handle.write("\\hline\n")
        handle.write("Stress case & Method & Clean & RAUC & Median $r_{80}$ & AULC \\\\\n")
        handle.write("\\hline\n")
        for row in rows:
            handle.write(
                f"{row['stress_label']} & {row['method']} & "
                f"{float(row['clean_mean']):.3f}$\\pm${float(row['clean_sem']):.3f} & "
                f"{float(row['rauc_mean']):.3f}$\\pm${float(row['rauc_sem']):.3f} & "
                f"{float(row['r80_mean']):.3f}$\\pm${float(row['r80_sem']):.3f} & "
                f"{float(row['aulc_mean']):.3f}$\\pm${float(row['aulc_sem']):.3f} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")


def load_grid_learning_curve(path: Path) -> list[dict[str, str | float | int]]:
    if not path.exists():
        return []
    import json

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    by_method: dict[str, dict[int, dict]] = {}
    for result in payload["results"]:
        by_method.setdefault(str(result["method"]), {})[int(result["seed"])] = result
    if "bgr" not in by_method or "uniform" not in by_method:
        return []
    seeds = sorted(set(by_method["bgr"]) & set(by_method["uniform"]))
    steps = [float(point["step"]) for point in by_method["bgr"][seeds[0]]["history"]]
    out: list[dict[str, str | float | int]] = []
    for step_idx, step in enumerate(steps):
        bgr_vals = [float(by_method["bgr"][seed]["history"][step_idx]["rauc"]) for seed in seeds]
        uniform_vals = [float(by_method["uniform"][seed]["history"][step_idx]["rauc"]) for seed in seeds]
        diffs = [bgr - uniform for bgr, uniform in zip(bgr_vals, uniform_vals, strict=True)]
        out.append(
            {
                "step": int(step),
                "n": len(seeds),
                "bgr_rauc_mean": mean(bgr_vals),
                "bgr_rauc_sem": sem(bgr_vals),
                "uniform_rauc_mean": mean(uniform_vals),
                "uniform_rauc_sem": sem(uniform_vals),
                "delta_mean": mean(diffs),
                "delta_sem": sem(diffs),
            }
        )
    return out


def write_grid_learning_curve_table(path: Path, rows: list[dict]) -> None:
    selected_steps = {30, 60, 120, 300}
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{rrrr}\n")
        handle.write("\\hline\n")
        handle.write("Step & BGR RAUC & Uniform RAUC & $\\Delta$ \\\\\n")
        handle.write("\\hline\n")
        for row in rows:
            if int(row["step"]) not in selected_steps:
                continue
            handle.write(
                f"{int(row['step'])} & "
                f"{float(row['bgr_rauc_mean']):.3f}$\\pm${float(row['bgr_rauc_sem']):.3f} & "
                f"{float(row['uniform_rauc_mean']):.3f}$\\pm${float(row['uniform_rauc_sem']):.3f} & "
                f"{float(row['delta_mean']):.3f}$\\pm${float(row['delta_sem']):.3f} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")


def load_openvla(recovery_path: Path, selection_path: Path) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    if recovery_path.exists():
        recovery_rows = list(csv.DictReader(recovery_path.open("r", encoding="utf-8")))
        for family in ["occlusion", "blur", "shift", "brightness"]:
            items = [row for row in recovery_rows if row["family"] == family]
            if not items:
                continue
            item = items[0]
            rows.append(
                {
                    "audit": "Recovery",
                    "name": family.title(),
                    "metric_a": float(item["clean_mean"]),
                    "metric_b": float(item["rauc_mean"]),
                    "metric_c": float(item["r80_mean"]),
                    "n": int(float(item["num_states"])),
                }
            )
    if selection_path.exists():
        selection_rows = list(csv.DictReader(selection_path.open("r", encoding="utf-8")))
        display = {
            "proposal_guided": "Proposal-guided",
            "bgr_boundary": "BGR-boundary",
            "random_balanced": "Random-balanced",
        }
        method_order = ["bgr_boundary", "proposal_guided", "random_balanced"]
        methods = [method for method in method_order if any(row["method"] == method for row in selection_rows)]
        methods.extend(
            method
            for method in sorted({row["method"] for row in selection_rows})
            if method not in methods
        )
        for method in methods:
            items = [row for row in selection_rows if row["method"] == method]
            if not items:
                continue
            item = items[0]
            rows.append(
                {
                    "audit": "Selection",
                    "name": display.get(method, method.replace("_", "-").title()),
                    "metric_a": float(item["mean_observed_cf_rate"]),
                    "metric_b": float(item["boundary_hit_rate"]),
                    "metric_c": float(item["mean_abs_distance_to_half"]),
                    "n": int(float(item["runs"])),
                }
            )
    return rows


def write_openvla_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{llcccc}\n")
        handle.write("\\hline\n")
        handle.write("Audit & Family/Method & Clean/CF & RAUC/Hit & $r_{80}$/Dist & $n$ \\\\\n")
        handle.write("\\hline\n")
        for row in rows:
            handle.write(
                f"{row['audit']} & {row['name']} & "
                f"{float(row['metric_a']):.3f} & {float(row['metric_b']):.3f} & "
                f"{float(row['metric_c']):.3f} & {int(row['n'])} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")


def load_openvla_adaptation(results_dir: Path) -> list[dict[str, str | int]]:
    runs = [
        (
            "100-step non-identity perturbations",
            results_dir / OPENVLA_FULL_PERTURB_RUN / "summary.csv",
            {"blur", "brightness", "occlusion", "shift"},
        ),
        (
            "300-step image-aug non-identity",
            results_dir / OPENVLA_IMAGEAUG_300_RUN / "summary.csv",
            {"blur", "brightness", "occlusion", "shift"},
        ),
        (
            "1000-step low-LR non-identity",
            results_dir / OPENVLA_LOWLR_1000_RUN / "summary.csv",
            {"blur", "brightness", "occlusion", "shift"},
        ),
        (
            "Weighted perturbation non-id.",
            results_dir / OPENVLA_WEIGHTED_PERTURB_RUN / "summary.csv",
            {"blur", "brightness", "occlusion", "shift"},
        ),
        (
            "Proximal anchor non-id.",
            results_dir / OPENVLA_PROXIMAL_PERTURB_RUN / "summary.csv",
            {"blur", "brightness", "occlusion", "shift"},
        ),
        (
            "Perturb-only anchor non-id.",
            results_dir / OPENVLA_PERTURB_ONLY_ANCHOR_RUN / "summary.csv",
            {"blur", "brightness", "occlusion", "shift"},
        ),
    ]
    rows: list[dict[str, str | int]] = []
    for label, path, perturbations in runs:
        if not path.exists():
            return []
        summary_rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
        item: dict[str, str | int] = {"audit": label}
        for method in ["bgr", "official", "random"]:
            selected = [
                row
                for row in summary_rows
                if row["method"] == method and row["perturbation"] in perturbations
            ]
            if not selected:
                return []
            successes = sum(int(float(row["successes"])) for row in selected)
            episodes = sum(int(float(row["episodes"])) for row in selected)
            item[method] = f"{successes}/{episodes}"
        rows.append(item)
    return rows


def write_openvla_adaptation_table(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lccc}\n")
        handle.write("\\hline\n")
        handle.write("Audit & BGR & Official & Random \\\\\n")
        handle.write("\\hline\n")
        for row in rows:
            handle.write(f"{row['audit']} & {row['bgr']} & {row['official']} & {row['random']} \\\\\n")
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")


def fmt(row: dict) -> str:
    return f"{float(row['mean']):.3f}$\\pm${float(row['sem']):.3f}"


def make_figures(out_dir: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib_pdf_fonts(plt)
    for metric in ["RAUC", "AULC", "Clean"]:
        fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.4), sharey=False)
        for ax, benchmark in zip(axes, ["Synthetic", "GridMargin", "RobotSuffix"], strict=True):
            subset = [row for row in rows if row["benchmark"] == benchmark and row["metric"] == metric]
            labels = [short_label(str(row["method"])) for row in subset]
            means = [float(row["mean"]) for row in subset]
            errors = [float(row["sem"]) for row in subset]
            colors = ["#1f77b4" if "BGR" in str(row["method"]) else "#b8b8b8" for row in subset]
            ax.bar(range(len(subset)), means, yerr=errors, color=colors, edgecolor="#333333", linewidth=0.5, capsize=2)
            ax.set_title(benchmark, fontsize=9)
            ax.set_xticks(range(len(subset)))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.set_ylim(bottom=0)
        axes[0].set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric.lower()}_bars.pdf")
        fig.savefig(out_dir / f"{metric.lower()}_bars.png", dpi=200)
        plt.close(fig)


def make_boundary_intuition_figure(out_dir: Path, results_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    configure_matplotlib_pdf_fonts(plt)
    sigma = np.linspace(0.0, 1.0, 240)
    clean = 0.94
    radius = 0.42
    recovery = clean / (1.0 + np.exp((sigma - radius) / 0.055))
    alpha = 0.8
    threshold = alpha * clean

    trace = grid_margin_boundary_trace(results_dir)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.05),
        gridspec_kw={"width_ratios": [1.0, 1.22, 0.95]},
    )
    ax = axes[0]
    ax.plot(sigma, recovery, color="#1f77b4", linewidth=2.0, label="Recovery curve")
    ax.axhline(threshold, color="#666666", linestyle="--", linewidth=1.0, label=r"$0.8 R(0)$")
    ax.axvline(radius, color="#d62728", linestyle=":", linewidth=1.4, label=r"$r_{80}$")
    ax.fill_between(sigma, recovery, color="#1f77b4", alpha=0.16)
    ax.set_xlabel("Perturbation radius")
    ax.set_ylabel("Success probability")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.0)
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=6.1, frameon=False)

    ax = axes[1]
    empirical_sigma = trace["sigma"]
    curve_specs = [
        ("bgr", "BGR", "#1f77b4", 2.0, "-"),
        ("uniform", "Uniform", "#666666", 1.5, "-"),
        ("failure_only", "Failure-only", "#d62728", 1.25, "--"),
        ("fixed", "Fixed radius", "#9467bd", 1.25, "-."),
        ("plr_loss", "PLR-loss", "#2ca02c", 1.25, ":"),
    ]
    for method, label, color, linewidth, linestyle in curve_specs:
        ax.plot(
            empirical_sigma,
            trace[f"{method}_curve"],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
        )
    ax.axvline(trace["bgr_median_r80"], color="#1f77b4", linestyle=":", linewidth=1.1)
    ax.axvline(trace["uniform_median_r80"], color="#666666", linestyle=":", linewidth=1.0)
    hist, edges = np.histogram(trace["bgr_sampled_sigmas"], bins=np.linspace(0.0, 1.0, 21))
    heights = hist / max(1, int(hist.max())) * 0.18
    ax.bar(
        edges[:-1],
        heights,
        width=np.diff(edges),
        align="edge",
        color="#1f77b4",
        alpha=0.20,
        edgecolor="none",
        label="BGR train radii",
    )
    inset = ax.inset_axes([0.56, 0.12, 0.38, 0.28])
    inset.boxplot(
        [trace["uniform_r80"], trace["bgr_r80"]],
        vert=False,
        widths=0.55,
        patch_artist=True,
        tick_labels=["U", "B"],
        medianprops={"color": "#222222", "linewidth": 0.8},
        boxprops={"facecolor": "#eeeeee", "edgecolor": "#666666", "linewidth": 0.6},
        whiskerprops={"color": "#666666", "linewidth": 0.6},
        capprops={"color": "#666666", "linewidth": 0.6},
        flierprops={"marker": ".", "markersize": 1.8, "markeredgecolor": "#666666"},
    )
    inset.set_xlim(0.0, 1.0)
    inset.set_title(r"$r_{80}$ dist.", fontsize=5.6, pad=1.0)
    inset.tick_params(axis="both", labelsize=5.3, length=1.5, pad=1.0)
    inset.grid(axis="x", alpha=0.18, linewidth=0.4)
    ax.set_xlabel("Perturbation radius")
    ax.set_ylabel("Success probability")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.0)
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=5.2, frameon=False, ncol=1)

    cross_metric = metric_cross_checks(results_dir)
    ax = axes[2]
    labels = [row["label"] for row in cross_metric]
    y = np.arange(len(labels))
    height = 0.32
    rauc_delta = [row["rauc_delta"] for row in cross_metric]
    r80_delta = [row["r80_delta"] for row in cross_metric]
    ax.barh(y - height / 2, rauc_delta, height=height, color="#1f77b4", alpha=0.78, label=r"$\Delta$RAUC")
    ax.barh(y + height / 2, r80_delta, height=height, color="#ff7f0e", alpha=0.78, label=r"$\Delta r_{80}$")
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("BGR - Uniform")
    ax.set_xlim(-0.012, 0.043)
    ax.grid(axis="x", alpha=0.22, linewidth=0.5)
    ax.legend(loc="lower right", fontsize=5.7, frameon=False)

    fig.tight_layout(w_pad=0.8)
    fig.savefig(out_dir / "boundary_intuition.pdf")
    fig.savefig(out_dir / "boundary_intuition.png", dpi=200)
    plt.close(fig)

    cross_metric_rows = []
    for row in cross_metric:
        key = row["key"]
        cross_metric_rows.extend(
            [
                {"metric": f"{key}_rauc_delta", "value": row["rauc_delta"]},
                {"metric": f"{key}_median_r80_delta", "value": row["r80_delta"]},
            ]
        )

    write_csv(
        out_dir / "boundary_intuition_stats.csv",
        [
            {"metric": "seed", "value": trace["seed"]},
            *[
                {"metric": f"{method}_final_rauc", "value": trace[f"{method}_rauc"]}
                for method, *_unused in curve_specs
            ],
            *[
                {"metric": f"{method}_median_r80", "value": trace[f"{method}_median_r80"]}
                for method, *_unused in curve_specs
            ],
            {"metric": "bgr_sample_radius_q25", "value": trace["bgr_sample_radius_q25"]},
            {"metric": "bgr_sample_radius_median", "value": trace["bgr_sample_radius_median"]},
            {"metric": "bgr_sample_radius_q75", "value": trace["bgr_sample_radius_q75"]},
            {"metric": "uniform_r80_q25", "value": trace["uniform_r80_q25"]},
            {"metric": "uniform_r80_q75", "value": trace["uniform_r80_q75"]},
            {"metric": "bgr_r80_q25", "value": trace["bgr_r80_q25"]},
            {"metric": "bgr_r80_q75", "value": trace["bgr_r80_q75"]},
            *cross_metric_rows,
        ],
    )


def metric_cross_checks(results_dir: Path) -> list[dict[str, float | str]]:
    checks = [
        (
            "grid_margin",
            "Grid",
            results_dir / "grid_margin_full_30seed_v1" / "summary.csv",
            "bgr",
            "uniform",
        ),
        (
            "robot_suffix",
            "Suffix",
            results_dir / "suffix_strategy_coverage_30seed_v1" / "summary.csv",
            "bgr_broad",
            "uniform",
        ),
    ]
    rows: list[dict[str, float | str]] = []
    for key, label, path, treatment, baseline in checks:
        table = list(csv.DictReader(path.open("r", encoding="utf-8")))
        treatment_rows = [row for row in table if row["method"] == treatment]
        baseline_rows = [row for row in table if row["method"] == baseline]
        if len(treatment_rows) != len(baseline_rows):
            raise ValueError(f"cannot pair metric cross-check rows for {path}")
        rows.append(
            {
                "key": key,
                "label": label,
                "rauc_delta": mean(
                    [
                        float(t_row["final_rauc"]) - float(b_row["final_rauc"])
                        for t_row, b_row in zip(treatment_rows, baseline_rows, strict=True)
                    ]
                ),
                "r80_delta": mean(
                    [
                        float(t_row["final_median_r80"]) - float(b_row["final_median_r80"])
                        for t_row, b_row in zip(treatment_rows, baseline_rows, strict=True)
                    ]
                ),
            }
        )
    return rows


def grid_margin_boundary_trace(results_dir: Path) -> dict:
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from bgr.metrics import critical_radius, recovery_auc

    config = json.loads((results_dir / "grid_margin_full_30seed_v1" / "results.json").read_text(encoding="utf-8"))[
        "config"
    ]
    seed = 0
    sigma_grid = np.linspace(0.0, 1.0, 101)
    alpha = float(config["experiment"].get("alpha", 0.8))
    methods = ["bgr", "uniform", "failure_only", "fixed", "plr_loss"]
    traces = {method: _run_grid_margin_trace(config, method, seed, sigma_grid) for method in methods}
    method_stats = {}
    for method, method_trace in traces.items():
        radii = [critical_radius(sigma_grid, curve, alpha=alpha) for curve in method_trace["state_curves"]]
        raucs = [recovery_auc(sigma_grid, curve, sigma_max=1.0) for curve in method_trace["state_curves"]]
        method_stats[f"{method}_curve"] = np.mean(np.vstack(method_trace["state_curves"]), axis=0)
        method_stats[f"{method}_rauc"] = float(np.mean(raucs))
        method_stats[f"{method}_median_r80"] = float(np.median(radii))
        method_stats[f"{method}_r80"] = np.asarray(radii, dtype=float)
    sampled = np.asarray(traces["bgr"]["sampled_sigmas"], dtype=float)
    return {
        "seed": seed,
        "sigma": sigma_grid,
        **method_stats,
        "bgr_sampled_sigmas": sampled,
        "bgr_sample_radius_q25": float(np.quantile(sampled, 0.25)),
        "bgr_sample_radius_median": float(np.quantile(sampled, 0.50)),
        "bgr_sample_radius_q75": float(np.quantile(sampled, 0.75)),
        "uniform_r80_q25": float(np.quantile(method_stats["uniform_r80"], 0.25)),
        "uniform_r80_q75": float(np.quantile(method_stats["uniform_r80"], 0.75)),
        "bgr_r80_q25": float(np.quantile(method_stats["bgr_r80"], 0.25)),
        "bgr_r80_q75": float(np.quantile(method_stats["bgr_r80"], 0.75)),
    }


def _run_grid_margin_trace(config: dict, method: str, seed: int, sigma_grid) -> dict:
    import numpy as np

    from bgr.envs.grid_recovery import GridMarginRecoveryBenchmark
    from bgr.experiments.grid_margin import (
        _init_records,
        _refresh_records,
        _sample_training_pair,
        _uses_bgr_records,
    )
    from bgr.priorities import BGRPriorityScorer

    exp = config["experiment"]
    bgr_cfg = config.get("bgr", {})
    rng = np.random.default_rng(seed + 30_000)
    bench = GridMarginRecoveryBenchmark(
        num_tasks=int(exp["num_tasks"]),
        grid_size=int(exp["grid_size"]),
        obstacle_prob=float(exp["obstacle_prob"]),
        replay_states_per_task=int(exp["replay_states_per_task"]),
        max_offset=int(exp["max_offset"]),
        learning_rate=float(exp["learning_rate"]),
        seed=seed,
        feasible_radius_floor=float(exp.get("feasible_radius_floor", 0.35)),
        feasible_radius_base=float(exp.get("feasible_radius_base", 0.45)),
        feasible_radius_path_scale=float(exp.get("feasible_radius_path_scale", 0.45)),
        feasible_radius_noise=float(exp.get("feasible_radius_noise", 0.03)),
        initial_margin_base=float(exp.get("initial_margin_base", 0.12)),
        initial_margin_path_scale=float(exp.get("initial_margin_path_scale", 0.22)),
        initial_margin_noise=float(exp.get("initial_margin_noise", 0.05)),
        initial_margin_min=float(exp.get("initial_margin_min", 0.05)),
        initial_margin_max=float(exp.get("initial_margin_max", 0.45)),
        clean_success_min=float(exp.get("clean_success_min", 0.80)),
        clean_success_max=float(exp.get("clean_success_max", 0.96)),
        temperature_min=float(exp.get("temperature_min", 0.045)),
        temperature_max=float(exp.get("temperature_max", 0.10)),
        boundary_width=float(exp.get("boundary_width", 0.15)),
    )
    alpha = float(exp.get("alpha", 0.8))
    records = _init_records(bench, bgr_cfg, alpha, rng)
    scorer = BGRPriorityScorer(target_radius=float(exp.get("target_margin", 0.38)))
    sampled_sigmas: list[float] = []
    for step in range(int(exp["iterations"]) + 1):
        if step % int(exp["eval_every"]) == 0 and _uses_bgr_records(method):
            _refresh_records(bench, records, bgr_cfg, alpha, rng, step)
        if step == int(exp["iterations"]):
            break
        for _ in range(int(exp["train_batch_size"])):
            replay_idx, sigma = _sample_training_pair(method, bench, records, scorer, config, rng, step)
            if method == "bgr":
                sampled_sigmas.append(sigma)
            bench.train_step(replay_idx, sigma, rng)
            if _uses_bgr_records(method):
                success = bench.rollout(replay_idx, sigma, rng)
                records[replay_idx].add_observation(sigma, success)
                records[replay_idx].replay_count += 1
    state_curves = [
        np.array([bench.success_prob(replay_idx, float(sigma)) for sigma in sigma_grid], dtype=float)
        for replay_idx in range(len(bench.states))
    ]
    return {"state_curves": state_curves, "sampled_sigmas": sampled_sigmas}


def make_estimator_figure(out_dir: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib_pdf_fonts(plt)
    labels = [str(row["method"]).replace("Active ", "") for row in rows]
    means = [float(row["r80_mae_mean"]) for row in rows]
    errors = [float(row["r80_mae_sem"]) for row in rows]
    colors = ["#1f77b4" if "Active" in str(row["method"]) else "#b8b8b8" for row in rows]
    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    ax.bar(range(len(rows)), means, yerr=errors, color=colors, edgecolor="#333333", linewidth=0.5, capsize=2)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("$r_{80}$ MAE")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "estimator_r80_mae.pdf")
    fig.savefig(out_dir / "estimator_r80_mae.png", dpi=200)
    plt.close(fig)


def make_grid_learning_curve_figure(out_dir: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib_pdf_fonts(plt)
    steps = [int(row["step"]) for row in rows]
    bgr = [float(row["bgr_rauc_mean"]) for row in rows]
    bgr_sem = [float(row["bgr_rauc_sem"]) for row in rows]
    uniform = [float(row["uniform_rauc_mean"]) for row in rows]
    uniform_sem = [float(row["uniform_rauc_sem"]) for row in rows]
    delta = [float(row["delta_mean"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.35), gridspec_kw={"width_ratios": [2.0, 1.0]})
    ax = axes[0]
    ax.plot(steps, bgr, color="#1f77b4", linewidth=1.9, label="BGR")
    ax.fill_between(
        steps,
        [mean - err for mean, err in zip(bgr, bgr_sem, strict=True)],
        [mean + err for mean, err in zip(bgr, bgr_sem, strict=True)],
        color="#1f77b4",
        alpha=0.16,
        linewidth=0,
    )
    ax.plot(steps, uniform, color="#666666", linewidth=1.7, label="Uniform")
    ax.fill_between(
        steps,
        [mean - err for mean, err in zip(uniform, uniform_sem, strict=True)],
        [mean + err for mean, err in zip(uniform, uniform_sem, strict=True)],
        color="#888888",
        alpha=0.18,
        linewidth=0,
    )
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Grid-margin RAUC")
    ax.set_xlim(min(steps), max(steps))
    ax.set_ylim(bottom=0.20)
    ax.grid(alpha=0.24, linewidth=0.5)
    ax.legend(loc="lower right", fontsize=7.2, frameon=False)

    ax = axes[1]
    ax.plot(steps, delta, color="#1f77b4", linewidth=1.8)
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.fill_between(steps, delta, 0.0, color="#1f77b4", alpha=0.18)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("BGR - Uniform")
    ax.set_xlim(min(steps), max(steps))
    ax.grid(alpha=0.24, linewidth=0.5)

    fig.tight_layout(w_pad=1.1)
    fig.savefig(out_dir / "grid_margin_learning_curve.pdf")
    fig.savefig(out_dir / "grid_margin_learning_curve.png", dpi=200)
    plt.close(fig)


def short_label(label: str) -> str:
    return {
        "Loss-priority": "Loss",
        "Failure": "Fail",
        "Uniform": "Unif",
        "BGR-Suffix": "BGR",
        "Clean FT": "Clean",
    }.get(label, label)


def configure_matplotlib_pdf_fonts(plt) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def sem(vals: list[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    mu = mean(vals)
    var = sum((val - mu) ** 2 for val in vals) / (len(vals) - 1)
    return math.sqrt(var / len(vals))


if __name__ == "__main__":
    main()
