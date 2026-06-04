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
    parser.add_argument("--config", default="configs/grid_margin_learning_rate_sensitivity_15seed.yaml")
    parser.add_argument("--out", default="runs/grid_margin_learning_rate_sensitivity")
    args = parser.parse_args()

    sweep_config = _load_config(Path(args.config))
    base_config = _load_config(Path(sweep_config["experiment"]["base_config"]))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [str(method) for method in sweep_config["experiment"].get("methods", ["uniform", "bgr"])]
    rows: list[dict[str, float | str | int]] = []
    results = []
    for learning_rate in sweep_config["experiment"]["learning_rates"]:
        for method in methods:
            for seed in sweep_config["experiment"]["seeds"]:
                run_config = config_for_learning_rate(base_config, float(learning_rate), int(seed), method)
                print(f"[run] learning_rate={float(learning_rate):.4f} method={method} seed={int(seed)}", flush=True)
                result = run_method(run_config, method, int(seed))
                result_payload = serialize_result(result)
                result_payload["learning_rate"] = float(learning_rate)
                results.append(result_payload)
                rows.append(
                    {
                        "learning_rate": float(learning_rate),
                        "method": method,
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


def config_for_learning_rate(base_config: dict, learning_rate: float, seed: int, method: str) -> dict:
    config = copy.deepcopy(base_config)
    config["experiment"]["name"] = f"{base_config['experiment'].get('name', 'grid_margin')}_lr{learning_rate:g}"
    config["experiment"]["learning_rate"] = learning_rate
    config["experiment"]["seeds"] = [seed]
    config["experiment"]["methods"] = [method]
    return config


def _summary(rows: list[dict[str, float | str | int]]) -> str:
    by_key: dict[tuple[float, str], list[dict[str, float | str | int]]] = {}
    for row in rows:
        by_key.setdefault((float(row["learning_rate"]), str(row["method"])), []).append(row)
    lines = ["learning_rate,method,n,final_clean_mean,final_rauc_mean,final_median_r80_mean,rauc_aulc_mean,best_rauc_mean"]
    for (learning_rate, method), items in sorted(by_key.items()):
        lines.append(
            ",".join(
                [
                    f"{learning_rate:.4f}",
                    method,
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
