#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine rendered OpenVLA-OFT example directories while preserving relative asset paths."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source in LABEL=DIR form, where DIR contains examples.jsonl and rendered assets.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sources = [_parse_source(value) for value in args.source]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = combine_sources(sources, out_dir)
    (out_dir / "examples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = _summary(sources, rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


def combine_sources(sources: list[tuple[str, Path]], out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for label, source_dir in sources:
        if label in seen_labels:
            raise ValueError(f"duplicate source label: {label}")
        seen_labels.add(label)
        source_rows = _load_examples(source_dir / "examples.jsonl")
        for row in source_rows:
            rows.append(_copy_row_assets(row, source_dir=source_dir, out_dir=out_dir, label=label))
    return rows


def _copy_row_assets(row: dict[str, Any], *, source_dir: Path, out_dir: Path, label: str) -> dict[str, Any]:
    updated = dict(row)
    updated["mix_source"] = label
    for key in ("image", "wrist_image", "array"):
        relative = Path(str(row[key]))
        source_path = source_dir / relative
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        target_relative = Path(label) / relative
        target_path = out_dir / target_relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        updated[key] = target_relative.as_posix()
    return updated


def _parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must be in LABEL=DIR form")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("source label must not be empty")
    return label, Path(path)


def _load_examples(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _summary(sources: list[tuple[str, Path]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter(str(row.get("mix_source", "")) for row in rows)
    by_perturbation = Counter(str(row.get("perturbation_type", "")) for row in rows)
    return {
        "sources": {label: str(path) for label, path in sources},
        "examples": len(rows),
        "examples_by_source": dict(sorted(by_source.items())),
        "perturbation_types": dict(sorted(by_perturbation.items())),
    }


if __name__ == "__main__":
    raise SystemExit(main())
