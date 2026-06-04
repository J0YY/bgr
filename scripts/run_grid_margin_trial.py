#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bgr.experiments.grid_margin import run_method, serialize_result
from scripts.run_toy_experiment import _load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    methods = [str(method) for method in config["experiment"]["methods"]]
    seeds = [int(seed) for seed in config["experiment"]["seeds"]]
    if args.method not in methods:
        raise ValueError(f"method={args.method!r} is not configured")
    if args.seed not in seeds:
        raise ValueError(f"seed={args.seed} is not configured")

    print(f"[run] method={args.method} seed={args.seed}", flush=True)
    result = run_method(config, args.method, args.seed)
    payload = {
        "method": args.method,
        "seed": args.seed,
        "result": serialize_result(result),
    }

    trials_dir = Path(args.out) / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    path = trials_dir / f"{args.method}_seed{args.seed}.json"
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


if __name__ == "__main__":
    main()
