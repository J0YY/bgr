#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FINAL_RE = re.compile(r"Overall success rate:\s*([0-9.]+)")
EPISODES_RE = re.compile(r"Total episodes:\s*(\d+)")
SUCCESSES_RE = re.compile(r"Total successes:\s*(\d+)")
TASK_RE = re.compile(r"Current task success rate:\s*([0-9.]+)")
PARTIAL_EPISODES_RE = re.compile(r"episodes completed so far:\s*(\d+)")
PARTIAL_SUCCESSES_RE = re.compile(r"# successes:\s*(\d+)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize OpenVLA-OFT LIBERO eval logs.")
    parser.add_argument("--method-log-dir", action="append", default=[], help="METHOD=DIR containing eval .txt logs.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="If no complete eval log is available, summarize the latest partial episode counters.",
    )
    args = parser.parse_args()

    rows = []
    for item in args.method_log_dir:
        if "=" not in item:
            raise SystemExit(f"--method-log-dir must be METHOD=DIR, got {item!r}")
        method, directory = item.split("=", 1)
        rows.append(_summarize_method(method, Path(directory), allow_partial=args.allow_partial))
    if not rows:
        raise SystemExit("At least one --method-log-dir is required.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "summary.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps({"methods": rows, "note": "Parsed from OpenVLA-OFT LIBERO eval local logs."}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return 0


def _summarize_method(method: str, directory: Path, *, allow_partial: bool = False) -> dict[str, Any]:
    logs = sorted(directory.glob("*.txt"), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"No eval .txt logs found in {directory}")
    errors = []
    for log in reversed(logs):
        text = log.read_text(encoding="utf-8", errors="replace")
        try:
            episodes = _last_int(EPISODES_RE, text)
            successes = _last_int(SUCCESSES_RE, text)
            success_rate = _last_float(FINAL_RE, text)
        except ValueError as exc:
            if allow_partial:
                try:
                    return _summarize_partial_log(method, log, text)
                except ValueError as partial_exc:
                    errors.append(f"{log}: {exc}; partial: {partial_exc}")
                    continue
            errors.append(f"{log}: {exc}")
            continue
        task_rates = [float(match.group(1)) for match in TASK_RE.finditer(text)]
        return {
            "method": method,
            "log": str(log),
            "episodes": episodes,
            "successes": successes,
            "success_rate": success_rate,
            "task_success_rates": json.dumps(task_rates),
            "num_tasks": len(task_rates),
        }
    raise ValueError(f"No complete eval logs found in {directory}: {'; '.join(errors)}")


def _summarize_partial_log(method: str, log: Path, text: str) -> dict[str, Any]:
    episodes = _last_int(PARTIAL_EPISODES_RE, text)
    successes = _last_int(PARTIAL_SUCCESSES_RE, text)
    if episodes <= 0:
        raise ValueError("partial log has zero completed episodes")
    return {
        "method": method,
        "log": str(log),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "task_success_rates": "[]",
        "num_tasks": 0,
    }


def _last_int(pattern: re.Pattern[str], text: str) -> int:
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"Missing pattern {pattern.pattern!r}")
    return int(matches[-1].group(1))


def _last_float(pattern: re.Pattern[str], text: str) -> float:
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"Missing pattern {pattern.pattern!r}")
    return float(matches[-1].group(1))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
