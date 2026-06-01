#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize OpenVLA perturbation-selection artifacts as boundary-discovery diagnostics."
    )
    parser.add_argument("--proposal-dir", action="append", default=[], help="Directory with proposal_guided_summary/top_k.")
    parser.add_argument("--random-dir", action="append", default=[], help="Directory with random_balanced_summary/top_k.")
    parser.add_argument("--proposal-method-name", default="proposal_guided")
    parser.add_argument("--random-method-name", default="random_balanced")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lower", type=float, default=0.25)
    parser.add_argument("--upper", type=float, default=0.75)
    args = parser.parse_args()

    rows = []
    for path in args.proposal_dir:
        rows.append(
            _summarize_dir(
                Path(path),
                method=str(args.proposal_method_name),
                lower=args.lower,
                upper=args.upper,
                summary_name="proposal_guided_summary.json",
            )
        )
    for path in args.random_dir:
        rows.append(
            _summarize_dir(
                Path(path),
                method=str(args.random_method_name),
                lower=args.lower,
                upper=args.upper,
                summary_name="random_balanced_summary.json",
            )
        )
    if not rows:
        raise SystemExit("No input directories provided.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = _aggregate(rows)
    _write_csv(out_dir / "summary.csv", rows)
    _write_csv(out_dir / "aggregate.csv", aggregate)
    (out_dir / "results.json").write_text(
        json.dumps({"rows": rows, "aggregate": aggregate, "boundary_band": [args.lower, args.upper]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


def _summarize_dir(path: Path, method: str, lower: float, upper: float, summary_name: str | None = None) -> dict[str, Any]:
    candidates = json.loads((path / "top_k_candidates.json").read_text(encoding="utf-8"))
    if summary_name is None:
        summary_name = "random_balanced_summary.json" if method == "random_balanced" else "proposal_guided_summary.json"
    summary = json.loads((path / summary_name).read_text(encoding="utf-8"))
    rates = [float(row.get("observed_cf_rate", 0.0)) for row in candidates]
    predictions = [float(row.get("predicted_cf_rate", np.nan)) for row in candidates if "predicted_cf_rate" in row]
    families = sorted({str(row.get("perturbation_type", "")) for row in candidates})
    return {
        "method": method,
        "run": path.name,
        "n_candidates": len(candidates),
        "n_counterfactual_certificates": int(summary.get("n_counterfactual_certificates", 0)),
        "mean_observed_cf_rate": _safe_mean(rates),
        "boundary_hit_rate": _safe_mean([lower <= rate <= upper for rate in rates]),
        "mean_abs_distance_to_half": _safe_mean([abs(rate - 0.5) for rate in rates]),
        "mean_predicted_cf_rate": _safe_mean(predictions) if predictions else "",
        "families": ";".join(families),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in sorted({str(row["method"]) for row in rows}):
        items = [row for row in rows if row["method"] == method]
        output.append(
            {
                "method": method,
                "runs": len(items),
                "mean_observed_cf_rate": _field_mean(items, "mean_observed_cf_rate"),
                "mean_observed_cf_rate_sem": _field_sem(items, "mean_observed_cf_rate"),
                "boundary_hit_rate": _field_mean(items, "boundary_hit_rate"),
                "boundary_hit_rate_sem": _field_sem(items, "boundary_hit_rate"),
                "mean_abs_distance_to_half": _field_mean(items, "mean_abs_distance_to_half"),
                "mean_abs_distance_to_half_sem": _field_sem(items, "mean_abs_distance_to_half"),
                "certificates_mean": _field_mean(items, "n_counterfactual_certificates"),
                "certificates_sem": _field_sem(items, "n_counterfactual_certificates"),
            }
        )
    return output


def _safe_mean(values: list[float | bool]) -> float:
    if not values:
        return 0.0
    return float(mean(float(value) for value in values))


def _field_mean(rows: list[dict[str, Any]], key: str) -> float:
    return _safe_mean([float(row[key]) for row in rows])


def _field_sem(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
