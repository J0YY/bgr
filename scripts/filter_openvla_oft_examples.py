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
        description="Filter rendered OpenVLA-OFT examples and copy selected assets."
    )
    parser.add_argument("--source", required=True, help="Directory containing examples.jsonl and rendered assets.")
    parser.add_argument("--out", required=True, help="Output directory for filtered examples and copied assets.")
    parser.add_argument(
        "--cap",
        action="append",
        default=[],
        help="Maximum rows to keep for a perturbation family, in FAMILY=N form. Can be repeated.",
    )
    parser.add_argument(
        "--include-family",
        action="append",
        default=[],
        help="Perturbation family to keep. Can be repeated. When omitted, all families are eligible.",
    )
    args = parser.parse_args()

    caps = _parse_caps(args.cap)
    source_dir = Path(args.source)
    out_dir = Path(args.out)
    include_families = set(args.include_family)
    rows = filter_examples(source_dir, out_dir, caps, include_families=include_families)
    (out_dir / "examples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = _summary(source_dir, rows, caps, include_families)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


def filter_examples(
    source_dir: Path,
    out_dir: Path,
    caps: dict[str, int],
    *,
    include_families: set[str] | None = None,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_examples(source_dir / "examples.jsonl")
    kept: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        family = str(row.get("perturbation_type", ""))
        if include_families and family not in include_families:
            continue
        cap = caps.get(family)
        if cap is not None and counts[family] >= cap:
            continue
        kept.append(_copy_row_assets(row, source_dir=source_dir, out_dir=out_dir))
        counts[family] += 1
    return kept


def _copy_row_assets(row: dict[str, Any], *, source_dir: Path, out_dir: Path) -> dict[str, Any]:
    updated = dict(row)
    for key in ("image", "wrist_image", "array"):
        relative = Path(str(row[key]))
        source_path = source_dir / relative
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        target_path = out_dir / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    return updated


def _parse_caps(values: list[str]) -> dict[str, int]:
    caps: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("--cap must be in FAMILY=N form")
        family, raw_cap = value.split("=", 1)
        family = family.strip()
        if not family:
            raise argparse.ArgumentTypeError("cap family must not be empty")
        cap = int(raw_cap)
        if cap < 0:
            raise argparse.ArgumentTypeError("cap must be non-negative")
        caps[family] = cap
    return caps


def _load_examples(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _summary(
    source_dir: Path,
    rows: list[dict[str, Any]],
    caps: dict[str, int],
    include_families: set[str],
) -> dict[str, Any]:
    return {
        "source": str(source_dir),
        "examples": len(rows),
        "caps": dict(sorted(caps.items())),
        "include_families": sorted(include_families),
        "perturbation_types": dict(sorted(Counter(str(row.get("perturbation_type", "")) for row in rows).items())),
    }


if __name__ == "__main__":
    raise SystemExit(main())
