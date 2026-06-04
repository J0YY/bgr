#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from bgr.experiments.grid_margin import run_method, serialize_result
from scripts.run_toy_experiment import _load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grid_margin_target_sensitivity_15seed.yaml")
    parser.add_argument("--out", default="runs/grid_margin_target_sensitivity")
    args = parser.parse_args()

    sweep_config = _load_config(Path(args.config))
    base_config = _load_config(Path(sweep_config["experiment"]["base_config"]))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | int]] = []
    results = []
    method = str(sweep_config["experiment"].get("method", "bgr"))
    for target_margin in sweep_config["experiment"]["target_margins"]:
        for seed in sweep_config["experiment"]["seeds"]:
            run_config = config_for_target(base_config, float(target_margin), int(seed), method)
            print(f"[run] method={method} target_margin={float(target_margin):.3f} seed={int(seed)}", flush=True)
            result = run_method(run_config, method, int(seed))
            result_payload = serialize_result(result)
            result_payload["target_margin"] = float(target_margin)
            results.append(result_payload)
            rows.append(
                {
                    "method": method,
                    "target_margin": float(target_margin),
                    "seed": int(seed),
                    "final_clean": result.final_clean,
                    "final_rauc": result.final_rauc,
                    "final_median_r80": result.final_median_r80,
                    "rauc_aulc": result.rauc_aulc,
                    "best_rauc": result.best_rauc,
                }
            )

    with (out_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": sweep_config, "base_config": base_config, "results": results}, handle, indent=2)
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(_summary(rows))


def config_for_target(base_config: dict, target_margin: float, seed: int, method: str) -> dict:
    config = copy.deepcopy(base_config)
    config["experiment"]["target_margin"] = target_margin
    config["experiment"]["seeds"] = [seed]
    config["experiment"]["methods"] = [method]
    return config


def _summary(rows: list[dict[str, float | str | int]]) -> str:
    by_target: dict[float, list[dict[str, float | str | int]]] = {}
    for row in rows:
        by_target.setdefault(float(row["target_margin"]), []).append(row)
    lines = ["target_margin,n,final_clean_mean,final_rauc_mean,final_median_r80_mean,rauc_aulc_mean,best_rauc_mean"]
    for target_margin, items in sorted(by_target.items()):
        lines.append(
            ",".join(
                [
                    f"{target_margin:.3f}",
                    str(len(items)),
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
