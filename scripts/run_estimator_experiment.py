#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

from bgr.experiments.estimator import run_method, serialize_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/estimator_bgr.yaml")
    parser.add_argument("--out", default="runs/estimator")
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
            rows.append(serialize_result(result))

    with open(out_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump({"config": config, "results": results}, handle, indent=2)

    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(_summary(rows))
    try:
        _plot(results, out_dir)
    except Exception as exc:  # pragma: no cover - optional plotting path.
        print(f"[warn] skipped plotting: {exc}")


def _summary(rows: list[dict[str, float | str | int]]) -> str:
    by_method: dict[str, list[dict[str, float | str | int]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    lines = ["method,r80_mae_mean,r80_bias_mean,rauc_mae_mean,boundary_hit_rate_mean,mean_uncertainty_mean"]
    for method, items in sorted(by_method.items()):
        lines.append(
            ",".join(
                [
                    method,
                    f"{_mean(items, 'r80_mae'):.4f}",
                    f"{_mean(items, 'r80_bias'):.4f}",
                    f"{_mean(items, 'rauc_mae'):.4f}",
                    f"{_mean(items, 'boundary_hit_rate'):.4f}",
                    f"{_mean(items, 'mean_uncertainty'):.4f}",
                ]
            )
        )
    return "\n".join(lines)


def _load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict:
    root: dict[str, dict] = {}
    section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            if not line.endswith(":"):
                raise ValueError(f"expected section line, got: {raw_line}")
            section = line[:-1]
            root[section] = {}
            continue
        if section is None:
            raise ValueError(f"key outside section: {raw_line}")
        key, value = line.strip().split(":", 1)
        root[section][key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str):
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("["):
        try:
            return ast.literal_eval(value)
        except ValueError:
            inner = value.strip()[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _mean(items: list[dict[str, float | str | int]], key: str) -> float:
    return sum(float(item[key]) for item in items) / len(items)


def _plot(results: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    methods = sorted({str(result["method"]) for result in results})
    means = [_mean([r for r in results if r["method"] == method], "r80_mae") for method in methods]
    plt.figure(figsize=(4.6, 3.0))
    plt.bar(methods, means, color=["#b8b8b8" if method != "active" else "#1f77b4" for method in methods])
    plt.ylabel("mean absolute $r_{80}$ error")
    plt.tight_layout()
    plt.savefig(out_dir / "r80_mae.png", dpi=200)


if __name__ == "__main__":
    main()
