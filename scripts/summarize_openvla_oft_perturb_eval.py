#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts.summarize_openvla_oft_eval import _summarize_method


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize nested OpenVLA-OFT perturbation eval logs.")
    parser.add_argument(
        "--logs-root",
        required=True,
        help="Root containing METHOD/PERTURBATION/*.txt eval logs.",
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    args = parser.parse_args()

    rows = summarize_perturbation_logs(Path(args.logs_root))
    if not rows:
        raise SystemExit(f"No perturbation eval logs found under {args.logs_root}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "summary.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "aggregate": aggregate_by_method(rows),
                "note": "Parsed from nested OpenVLA-OFT perturbation eval local logs.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def summarize_perturbation_logs(logs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not logs_root.exists():
        raise FileNotFoundError(logs_root)
    for method_dir in sorted(path for path in logs_root.iterdir() if path.is_dir()):
        perturbation_dirs = sorted(
            (path for path in method_dir.iterdir() if path.is_dir()),
            key=lambda path: (_perturbation_order(path.name), path.name),
        )
        for perturbation_dir in perturbation_dirs:
            try:
                row = _summarize_method(method_dir.name, perturbation_dir)
            except (FileNotFoundError, ValueError):
                continue
            row["perturbation"] = perturbation_dir.name
            rows.append(_ordered_row(row))
    return rows


def aggregate_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows})
    aggregates = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        perturbed_rows = [row for row in method_rows if row["perturbation"] not in {"identity", "none"}]
        aggregates.append(
            {
                "method": method,
                "identity_success_rate": _success_rate_for(method_rows, "identity"),
                "mean_perturbed_success_rate": _mean_success_rate(perturbed_rows),
                "num_perturbations": len(method_rows),
                "num_perturbed": len(perturbed_rows),
            }
        )
    return aggregates


def _success_rate_for(rows: list[dict[str, Any]], perturbation: str) -> float | None:
    for row in rows:
        if row["perturbation"] == perturbation:
            return _row_success_rate(row)
    return None


def _mean_success_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    total_episodes = sum(int(row.get("episodes", 0)) for row in rows)
    total_successes = sum(int(row.get("successes", 0)) for row in rows)
    if total_episodes > 0:
        return total_successes / total_episodes
    return sum(float(row["success_rate"]) for row in rows) / len(rows)


def _row_success_rate(row: dict[str, Any]) -> float:
    episodes = int(row.get("episodes", 0))
    if episodes > 0 and "successes" in row:
        return int(row["successes"]) / episodes
    return float(row["success_rate"])


def _perturbation_order(name: str) -> int:
    order = {
        "identity": 0,
        "blur": 1,
        "brightness": 2,
        "occlusion": 3,
        "shift": 4,
    }
    return order.get(name, len(order))


def _ordered_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": row["method"],
        "perturbation": row["perturbation"],
        "episodes": row["episodes"],
        "successes": row["successes"],
        "success_rate": row["success_rate"],
        "num_tasks": row["num_tasks"],
        "task_success_rates": row["task_success_rates"],
        "log": row["log"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
