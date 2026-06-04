#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bgr.experiments.suffix import run_method, serialize_result
from scripts.run_toy_experiment import _load_config


@dataclass(frozen=True)
class SuffixStressCase:
    name: str
    overrides: dict[str, Any]

    def to_row(self) -> dict[str, str | float | int]:
        return {"stress_case": self.name, **self.overrides}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/suffix_stress_sensitivity_15seed.yaml")
    parser.add_argument("--out", default="runs/suffix_stress_sensitivity")
    args = parser.parse_args()

    sweep_config = _load_config(Path(args.config))
    base_config = _load_config(Path(sweep_config["experiment"]["base_config"]))
    cases = parse_stress_cases(sweep_config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | int]] = []
    results = []
    methods = [str(method) for method in sweep_config["experiment"].get("methods", ["uniform", "bgr_broad"])]
    for case in cases:
        for method in methods:
            for seed in sweep_config["experiment"]["seeds"]:
                run_config = config_for_stress_case(base_config, case, int(seed), method)
                print(f"[run] stress_case={case.name} method={method} seed={int(seed)}", flush=True)
                result = run_method(run_config, method, int(seed))
                payload = serialize_result(result)
                payload.update(case.to_row())
                results.append(payload)
                rows.append(
                    {
                        "stress_case": case.name,
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

    with (out_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": sweep_config, "base_config": base_config, "results": results}, handle, indent=2)
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(_summary(rows))


def parse_stress_cases(config: dict) -> list[SuffixStressCase]:
    raw_cases = config["experiment"].get("stress_cases", [])
    if not raw_cases:
        raise ValueError("experiment.stress_cases must contain at least one case")
    cases: list[SuffixStressCase] = []
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("each suffix stress case must be a mapping")
        if "name" not in raw:
            raise ValueError("each suffix stress case requires a name")
        overrides = {str(key): value for key, value in raw.items() if key != "name"}
        if not overrides:
            raise ValueError(f"suffix stress case {raw['name']!r} has no overrides")
        cases.append(SuffixStressCase(str(raw["name"]), overrides))
    return cases


def config_for_stress_case(base_config: dict, case: SuffixStressCase, seed: int, method: str) -> dict:
    config = copy.deepcopy(base_config)
    config["experiment"]["name"] = f"{base_config['experiment'].get('name', 'suffix')}_{case.name}"
    config["experiment"].update(case.overrides)
    config["experiment"]["seeds"] = [seed]
    config["experiment"]["methods"] = [method]
    return config


def _summary(rows: list[dict[str, float | str | int]]) -> str:
    by_key: dict[tuple[str, str], list[dict[str, float | str | int]]] = {}
    for row in rows:
        by_key.setdefault((str(row["stress_case"]), str(row["method"])), []).append(row)
    lines = [
        "stress_case,method,n,final_clean_mean,final_rauc_mean,final_median_r80_mean,"
        "final_transfer_rauc_mean,rauc_aulc_mean,best_rauc_mean"
    ]
    for (stress_case, method), items in sorted(by_key.items()):
        lines.append(
            ",".join(
                [
                    stress_case,
                    method,
                    str(len(items)),
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
