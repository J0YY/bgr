#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

from bgr.experiments.toy import run_method, serialize_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/toy_bgr.yaml")
    parser.add_argument("--out", default="runs/toy")
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
    try:
        _plot(results, out_dir)
    except Exception as exc:  # pragma: no cover - plotting is optional on clusters.
        print(f"[warn] skipped plotting: {exc}")


def _summary(rows: list[dict[str, float | str | int]]) -> str:
    by_method: dict[str, list[dict[str, float | str | int]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    lines = ["method,final_clean_mean,final_rauc_mean,final_median_r80_mean,best_rauc_mean"]
    for method, items in sorted(by_method.items()):
        lines.append(
            ",".join(
                [
                    method,
                    f"{_mean(items, 'final_clean'):.4f}",
                    f"{_mean(items, 'final_rauc'):.4f}",
                    f"{_mean(items, 'final_median_r80'):.4f}",
                    f"{_mean(items, 'best_rauc'):.4f}",
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
    """Small YAML subset parser for the checked-in experiment config."""

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

    by_method: dict[str, list[dict]] = {}
    for result in results:
        by_method.setdefault(result["method"], []).append(result)

    plt.figure(figsize=(6.5, 4.0))
    for method, items in sorted(by_method.items()):
        steps = [point["step"] for point in items[0]["history"]]
        means = []
        for idx in range(len(steps)):
            means.append(sum(item["history"][idx]["rauc"] for item in items) / len(items))
        plt.plot(steps, means, label=method)
    plt.xlabel("training iteration")
    plt.ylabel("held-out recovery AUC")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "rauc_learning_curve.png", dpi=200)


if __name__ == "__main__":
    main()
