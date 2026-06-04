#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from bgr.experiments.frozenlake_recovery import run_method, serialize_result
from scripts.run_toy_experiment import _load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/frozenlake_recovery_30seed.yaml")
    parser.add_argument("--out", default="runs/frozenlake_recovery")
    parser.add_argument("--methods", help="Comma-separated method override for small probes.")
    parser.add_argument("--seeds", help="Comma-separated seed override for small probes.")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    if args.methods:
        config["experiment"]["methods"] = [item.strip() for item in args.methods.split(",") if item.strip()]
    if args.seeds:
        config["experiment"]["seeds"] = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | int]] = []
    results = []
    for method in config["experiment"]["methods"]:
        for seed in config["experiment"]["seeds"]:
            start = time.perf_counter()
            print(f"[run] method={method} seed={seed}", flush=True)
            result = run_method(config, method, int(seed))
            elapsed = time.perf_counter() - start
            print(
                f"[done] method={method} seed={seed} "
                f"rauc={result.final_rauc:.4f} aulc={result.rauc_aulc:.4f} elapsed={elapsed:.2f}s",
                flush=True,
            )
            results.append(serialize_result(result))
            rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "final_clean": result.final_clean,
                    "final_rauc": result.final_rauc,
                    "final_median_r80": result.final_median_r80,
                    "rauc_aulc": result.rauc_aulc,
                    "best_rauc": result.best_rauc,
                }
            )

    with open(out_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump({"config": config, "results": results}, handle, indent=2)
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(_summary(rows))


def _summary(rows: list[dict[str, float | str | int]]) -> str:
    by_method: dict[str, list[dict[str, float | str | int]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    lines = ["method,final_clean_mean,final_rauc_mean,final_median_r80_mean,rauc_aulc_mean,best_rauc_mean"]
    for method, items in sorted(by_method.items()):
        lines.append(
            ",".join(
                [
                    method,
                    f"{_mean(items, 'final_clean'):.4f}",
                    f"{_mean(items, 'final_rauc'):.4f}",
                    f"{_mean(items, 'final_median_r80'):.4f}",
                    f"{_mean(items, 'rauc_aulc'):.4f}",
                    f"{_mean(items, 'best_rauc'):.4f}",
                ]
            )
        )
    return "\n".join(lines)


def _mean(items: list[dict[str, float | str | int]], key: str) -> float:
    return sum(float(item[key]) for item in items) / len(items)


if __name__ == "__main__":
    main()
