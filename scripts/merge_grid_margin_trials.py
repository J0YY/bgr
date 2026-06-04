#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.run_grid_margin_experiment import _summary
from scripts.run_toy_experiment import _load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    out_dir = Path(args.out)
    expected = {
        (str(method), int(seed))
        for method in config["experiment"]["methods"]
        for seed in config["experiment"]["seeds"]
    }
    results_by_key = _read_trials(out_dir / "trials")
    missing = sorted(expected - set(results_by_key), key=lambda item: (item[0], item[1]))
    extra = sorted(set(results_by_key) - expected, key=lambda item: (item[0], item[1]))
    if missing:
        raise ValueError(f"missing grid-margin trial result(s): {missing[:10]}")
    if extra:
        raise ValueError(f"unexpected grid-margin trial result(s): {extra[:10]}")

    results = [
        results_by_key[(str(method), int(seed))]
        for method in config["experiment"]["methods"]
        for seed in config["experiment"]["seeds"]
    ]
    rows = [
        {
            "method": result["method"],
            "seed": int(result["seed"]),
            "final_clean": result["final_clean"],
            "final_rauc": result["final_rauc"],
            "final_median_r80": result["final_median_r80"],
            "rauc_aulc": result["rauc_aulc"],
            "best_rauc": result["best_rauc"],
        }
        for result in results
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": config, "results": results}, handle, indent=2)
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(_summary(rows))


def _read_trials(trials_dir: Path) -> dict[tuple[str, int], dict]:
    results: dict[tuple[str, int], dict] = {}
    for path in sorted(trials_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        result = dict(payload["result"])
        key = (str(result["method"]), int(result["seed"]))
        if key in results:
            raise ValueError(f"duplicate grid-margin trial result: {key}")
        results[key] = result
    return results


if __name__ == "__main__":
    main()
