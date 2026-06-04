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
    parser.add_argument("--config", default="configs/grid_margin_regime_sensitivity_15seed.yaml")
    parser.add_argument("--out", default="runs/grid_margin_regime_sensitivity")
    args = parser.parse_args()

    sweep_config = _load_config(Path(args.config))
    base_config = _load_config(Path(sweep_config["experiment"]["base_config"]))
    regimes = parse_regimes(sweep_config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | int]] = []
    results = []
    methods = [str(method) for method in sweep_config["experiment"].get("methods", ["uniform", "bgr"])]
    for regime in regimes:
        for method in methods:
            for seed in sweep_config["experiment"]["seeds"]:
                run_config = config_for_regime(base_config, regime, int(seed), method)
                print(f"[run] regime={regime.name} method={method} seed={int(seed)}", flush=True)
                result = run_method(run_config, method, int(seed))
                result_payload = serialize_result(result)
                result_payload.update(regime.to_row())
                results.append(result_payload)
                rows.append(
                    {
                        "regime": regime.name,
                        "method": method,
                        "obstacle_prob": regime.obstacle_prob,
                        "grid_size": regime.grid_size,
                        "max_offset": regime.max_offset,
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


class Regime:
    def __init__(self, name: str, obstacle_prob: float, grid_size: int, max_offset: int) -> None:
        self.name = name
        self.obstacle_prob = obstacle_prob
        self.grid_size = grid_size
        self.max_offset = max_offset

    def to_row(self) -> dict[str, str | float | int]:
        return {
            "regime": self.name,
            "obstacle_prob": self.obstacle_prob,
            "grid_size": self.grid_size,
            "max_offset": self.max_offset,
        }


def parse_regimes(config: dict) -> list[Regime]:
    exp = config["experiment"]
    names = [str(name) for name in exp["regime_names"]]
    obstacle_probs = [float(value) for value in exp["obstacle_probs"]]
    grid_sizes = [int(value) for value in exp["grid_sizes"]]
    max_offsets = [int(value) for value in exp["max_offsets"]]
    lengths = {len(names), len(obstacle_probs), len(grid_sizes), len(max_offsets)}
    if len(lengths) != 1:
        raise ValueError("regime_names, obstacle_probs, grid_sizes, and max_offsets must have matching lengths")
    return [
        Regime(name, obstacle_prob, grid_size, max_offset)
        for name, obstacle_prob, grid_size, max_offset in zip(
            names,
            obstacle_probs,
            grid_sizes,
            max_offsets,
            strict=True,
        )
    ]


def config_for_regime(base_config: dict, regime: Regime, seed: int, method: str) -> dict:
    config = copy.deepcopy(base_config)
    config["experiment"]["name"] = f"{base_config['experiment'].get('name', 'grid_margin')}_{regime.name}"
    config["experiment"]["obstacle_prob"] = regime.obstacle_prob
    config["experiment"]["grid_size"] = regime.grid_size
    config["experiment"]["max_offset"] = regime.max_offset
    config["experiment"]["seeds"] = [seed]
    config["experiment"]["methods"] = [method]
    return config


def _summary(rows: list[dict[str, float | str | int]]) -> str:
    by_key: dict[tuple[str, str], list[dict[str, float | str | int]]] = {}
    for row in rows:
        by_key.setdefault((str(row["regime"]), str(row["method"])), []).append(row)
    lines = ["regime,method,n,final_clean_mean,final_rauc_mean,final_median_r80_mean,rauc_aulc_mean,best_rauc_mean"]
    for (regime, method), items in sorted(by_key.items()):
        lines.append(
            ",".join(
                [
                    regime,
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
