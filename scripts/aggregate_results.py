#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


BENCHMARKS = {
    "toy_fast_v3": {
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
    "grid_margin_full_v1": {
        "label": "GridMargin",
        "primary": ["bgr", "uniform", "failure_only", "plr_loss", "fixed"],
        "display": {
            "bgr": "BGR",
            "uniform": "Uniform",
            "failure_only": "Failure",
            "plr_loss": "Loss-priority",
            "fixed": "Fixed",
        },
    },
    "suffix_strategy_v1": {
        "label": "RobotSuffix",
        "primary": ["bgr_broad", "uniform", "bgr", "bgr_hard", "bgr_boundary"],
        "display": {
            "bgr_broad": "BGR-Broad",
            "uniform": "Uniform",
            "bgr": "BGR",
            "bgr_hard": "BGR-Hard",
            "bgr_boundary": "BGR-Boundary",
        },
    },
}

METRICS = [
    ("final_clean", "Clean"),
    ("final_rauc", "RAUC"),
    ("final_median_r80", "MedianR80"),
    ("rauc_aulc", "AULC"),
]

ESTIMATOR_RUN = "estimator_full_v1"
ABLATION_RUN = "grid_margin_ablation_v1"
OPENVLA_RECOVERY_RUN = "libero_openvla_recovery_v1"
OPENVLA_SELECTION_RUN = "libero_openvla_boundary_selection_v1"


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
        path = results_dir / run_name / "summary.csv"
        rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
        loaded[run_name] = rows
        for method in spec["primary"]:
            method_rows = [row for row in rows if row["method"] == method]
            for metric, metric_label in METRICS:
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

        for baseline in [m for m in spec["primary"] if m != "bgr"]:
            for metric, metric_label in METRICS:
                bgr_vals = [float(row[metric]) for row in rows if row["method"] == "bgr"]
                base_vals = [float(row[metric]) for row in rows if row["method"] == baseline]
                diffs = [a - b for a, b in zip(bgr_vals, base_vals, strict=True)]
                effect_rows.append(
                    {
                        "benchmark": spec["label"],
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
    ablation_rows = load_ablation(results_dir / ABLATION_RUN / "summary.csv")
    if ablation_rows:
        write_csv(out_dir / "grid_margin_ablation_stats.csv", ablation_rows)
        write_ablation_table(out_dir / "grid_margin_ablation_table.tex", ablation_rows)
    openvla_rows = load_openvla(
        results_dir / OPENVLA_RECOVERY_RUN / "summary.csv",
        results_dir / OPENVLA_SELECTION_RUN / "aggregate.csv",
    )
    if openvla_rows:
        write_csv(out_dir / "openvla_stats.csv", openvla_rows)
        write_openvla_table(out_dir / "openvla_table.tex", openvla_rows)
    try:
        make_figures(out_dir, summary_rows)
        if estimator_rows:
            make_estimator_figure(out_dir, estimator_rows)
    except Exception as exc:  # pragma: no cover - optional plotting path.
        print(f"[warn] skipped figure generation: {exc}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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
        for benchmark in ["Synthetic", "GridMargin", "RobotSuffix"]:
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


def load_ablation(path: Path) -> list[dict[str, str | float | int]]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    display = {
        "bgr": "BGR",
        "bgr_no_uncertainty": "No uncertainty",
        "bgr_no_sharpness": "No sharpness",
        "bgr_uniform_radius": "Uniform radius",
        "uniform": "Uniform replay",
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
        display = {"proposal_guided": "Proposal-guided", "random_balanced": "Random-balanced"}
        for method in ["proposal_guided", "random_balanced"]:
            items = [row for row in selection_rows if row["method"] == method]
            if not items:
                continue
            item = items[0]
            rows.append(
                {
                    "audit": "Selection",
                    "name": display[method],
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


def fmt(row: dict) -> str:
    return f"{float(row['mean']):.3f}$\\pm${float(row['sem']):.3f}"


def make_figures(out_dir: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

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


def make_estimator_figure(out_dir: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

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


def short_label(label: str) -> str:
    return {
        "Loss-priority": "Loss",
        "Failure": "Fail",
        "Uniform": "Unif",
        "BGR-Suffix": "BGR",
        "Clean FT": "Clean",
    }.get(label, label)


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
