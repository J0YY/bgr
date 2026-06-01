#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bgr.libero_probe import probe_task, row_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe LIBERO resettable state perturbations for BGR-Suffix.")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--init-state-ids", default="0,1,2")
    parser.add_argument("--radii", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--trials-per-radius", type=int, default=4)
    parser.add_argument("--max-radius-meters", type=float, default=0.08)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/libero_probe")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = _parse_ints(args.task_ids)
    init_state_ids = _parse_ints(args.init_state_ids)
    radii = _parse_floats(args.radii)

    rows = []
    for task_id in task_ids:
        print(f"[probe] suite={args.suite} task_id={task_id}", flush=True)
        rows.extend(
            probe_task(
                suite_name=args.suite,
                task_id=task_id,
                init_state_ids=init_state_ids,
                radii=radii,
                trials_per_radius=args.trials_per_radius,
                max_radius_meters=args.max_radius_meters,
                settle_steps=args.settle_steps,
                image_size=args.image_size,
                seed=args.seed + task_id,
            )
        )

    dict_rows = [row_to_dict(row) for row in rows]
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dict_rows)
    with open(out_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "rows": dict_rows, "summary": _summary(dict_rows)}, handle, indent=2)
    print(_format_summary(dict_rows))


def _parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _summary(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        return {"rows": 0}
    return {
        "rows": len(rows),
        "mean_valid_rate": sum(float(row["valid_rate"]) for row in rows) / len(rows),
        "mean_success_rate": sum(float(row["success_rate"]) for row in rows) / len(rows),
        "mean_free_object_joints": sum(float(row["num_free_object_joints"]) for row in rows) / len(rows),
        "rows_with_errors": sum(1 for row in rows if row["error"]),
    }


def _format_summary(rows: list[dict]) -> str:
    summary = _summary(rows)
    lines = ["metric,value"]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"{key},{value:.4f}")
        else:
            lines.append(f"{key},{value}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
