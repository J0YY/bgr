#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export validated OpenVLA boundary candidates into a BGR fine-tuning manifest."
    )
    parser.add_argument("--bgr-dir", action="append", default=[], help="Directory with BGR/proposal top_k_candidates.json.")
    parser.add_argument("--random-dir", action="append", default=[], help="Directory with random top_k_candidates.json.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lower", type=float, default=0.25)
    parser.add_argument("--upper", type=float, default=0.75)
    parser.add_argument("--openvla-oft-root", default="/work/anonymous/external_validation/openvla_oft_smoke_746787/openvla-oft")
    parser.add_argument("--rlds-root", default="/work/anonymous/bgr/openvla_rlds")
    parser.add_argument("--run-root", default="/work/anonymous/bgr/openvla_oft_runs")
    parser.add_argument("--base-checkpoint", default="openvla/openvla-7b-finetuned-libero-object")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.bgr_dir:
        rows.extend(_load_candidates(Path(path), "bgr_boundary", float(args.lower), float(args.upper)))
    for path in args.random_dir:
        rows.extend(_load_candidates(Path(path), "random_balanced", float(args.lower), float(args.upper)))
    if not rows:
        raise SystemExit("No candidate directories provided.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (str(row["method"]), str(row["run"]), int(row["candidate_index"])))
    _write_jsonl(out_dir / "manifest.jsonl", rows)
    _write_csv(out_dir / "manifest.csv", rows)
    summary = {
        "num_candidates": len(rows),
        "boundary_band": [float(args.lower), float(args.upper)],
        "methods": _summarize(rows),
        "note": "Manifest identifies validated perturbation candidates. OpenVLA-OFT still requires converting replay states to an RLDS training dataset before LoRA fine-tuning.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_plan(out_dir / "openvla_oft_finetune_plan.sh", args)
    return 0


def _load_candidates(path: Path, method: str, lower: float, upper: float) -> list[dict[str, Any]]:
    payload = json.loads((path / "top_k_candidates.json").read_text(encoding="utf-8"))
    rows = []
    for idx, item in enumerate(payload):
        observed = float(item.get("observed_cf_rate", 0.0))
        in_band = lower <= observed <= upper
        rows.append(
            {
                "method": method,
                "run": path.name,
                "source_artifact": str(path),
                "candidate_index": idx,
                "name": str(item.get("name", "")),
                "spec": str(item.get("spec", "")),
                "perturbation_type": str(item.get("perturbation_type", "")),
                "perturbation_params": json.dumps(item.get("perturbation_params", {}), sort_keys=True),
                "observed_cf_rate": observed,
                "predicted_cf_rate": _optional_float(item.get("predicted_cf_rate")),
                "observed_cf_failures": int(item.get("observed_cf_failures", 0)),
                "observed_episodes": int(item.get("observed_episodes", 0)),
                "observed_success_rate": float(item.get("observed_success_rate", 0.0)),
                "boundary_abs_distance": abs(observed - 0.5),
                "boundary_score": max(0.0, 1.0 - abs(observed - 0.5) / 0.5),
                "in_boundary_band": in_band,
                "training_role": "boundary_candidate" if in_band else "coverage_candidate",
            }
        )
    return rows


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return number


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summary = {}
    for method, items in sorted(grouped.items()):
        summary[method] = {
            "candidates": len(items),
            "boundary_candidates": sum(bool(row["in_boundary_band"]) for row in items),
            "mean_observed_cf_rate": mean(float(row["observed_cf_rate"]) for row in items),
            "mean_boundary_score": mean(float(row["boundary_score"]) for row in items),
            "families": sorted({str(row["perturbation_type"]) for row in items}),
        }
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_plan(path: Path, args: argparse.Namespace) -> None:
    openvla_root = Path(args.openvla_oft_root)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Convert manifest.jsonl into two RLDS datasets before launching these commands:",
        f"#   {args.rlds_root}/bgr_boundary_libero_object",
        f"#   {args.rlds_root}/random_balanced_libero_object",
        "# The manifest contains validated perturbation specs and source artifact pointers,",
        "# but the OpenVLA-OFT trainer consumes RLDS episodes with images, actions, and language.",
        "",
        f"OPENVLA_OFT_ROOT={openvla_root}",
        f"RLDS_ROOT={args.rlds_root}",
        f"RUN_ROOT={args.run_root}",
        f"BASE_CHECKPOINT={args.base_checkpoint}",
        "",
        "python ${OPENVLA_OFT_ROOT}/vla-scripts/finetune.py \\",
        "  --vla_path ${BASE_CHECKPOINT} \\",
        "  --data_root_dir ${RLDS_ROOT} \\",
        "  --dataset_name bgr_boundary_libero_object \\",
        "  --run_root_dir ${RUN_ROOT} \\",
        "  --batch_size 1 \\",
        "  --grad_accumulation_steps 8 \\",
        "  --max_steps 1000 \\",
        "  --save_freq 500 \\",
        "  --lora_rank 16 \\",
        "  --shuffle_buffer_size 2048 \\",
        "  --run_id_note bgr_boundary",
        "",
        "python ${OPENVLA_OFT_ROOT}/vla-scripts/finetune.py \\",
        "  --vla_path ${BASE_CHECKPOINT} \\",
        "  --data_root_dir ${RLDS_ROOT} \\",
        "  --dataset_name random_balanced_libero_object \\",
        "  --run_root_dir ${RUN_ROOT} \\",
        "  --batch_size 1 \\",
        "  --grad_accumulation_steps 8 \\",
        "  --max_steps 1000 \\",
        "  --save_freq 500 \\",
        "  --lora_rank 16 \\",
        "  --shuffle_buffer_size 2048 \\",
        "  --run_id_note random_balanced",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


if __name__ == "__main__":
    raise SystemExit(main())
