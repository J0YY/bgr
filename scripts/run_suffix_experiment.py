#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.run_toy_experiment import _load_config
from bgr.experiments.suffix import run_method, serialize_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/suffix_bgr.yaml")
    parser.add_argument("--out", default="runs/suffix")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | int]] = []
    results = []
    for method in config["experiment"]["methods"]:
        for seed in config["experiment"]["seeds"]:
            print(f"[run] method={method} seed={seed}", flush=True)
            result = run_method(config, method, int(seed))
            results.append(serialize_result(result))
            rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "final_clean": result.final_clean,
                    "final_rauc": result.final_rauc,
                    "final_median_r80": result.final_median_r80,
                    "final_transfer_rauc": result.final_transfer_rauc,
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
    lines = ["method,final_clean_mean,final_rauc_mean,final_median_r80_mean,final_transfer_rauc_mean,rauc_aulc_mean,best_rauc_mean"]
    for method, items in sorted(by_method.items()):
        lines.append(
            ",".join(
                [
                    method,
                    f"{_mean(items, 'final_clean'):.4f}",
                    f"{_mean(items, 'final_rauc'):.4f}",
                    f"{_mean(items, 'final_median_r80'):.4f}",
                    f"{_mean(items, 'final_transfer_rauc'):.4f}",
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
