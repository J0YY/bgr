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
    "suffix_full_v1": {
        "label": "RobotSuffix",
        "primary": ["bgr", "uniform", "clean_ft", "failure_only", "loss_priority", "fixed"],
        "display": {
            "bgr": "BGR-Suffix",
            "uniform": "Uniform",
            "clean_ft": "Clean FT",
            "failure_only": "Failure",
            "loss_priority": "Loss-priority",
            "fixed": "Fixed",
        },
    },
}

METRICS = [
    ("final_clean", "Clean"),
    ("final_rauc", "RAUC"),
    ("final_median_r80", "MedianR80"),
    ("rauc_aulc", "AULC"),
]


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
    try:
        make_figures(out_dir, summary_rows)
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
