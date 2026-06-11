#!/usr/bin/env python3
"""Check whether an OpenVLA route-selection scout justifies a full gate run."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_METHODS = ("bgr", "official", "random")


@dataclass(frozen=True)
class MethodTotal:
    successes: int
    episodes: int

    @property
    def rate(self) -> float:
        return self.successes / self.episodes


@dataclass(frozen=True)
class ScoutDecision:
    complete: bool
    promote: bool
    detail: str


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_scout(
    rows: list[dict[str, str]],
    *,
    perturbation: str,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    min_episodes: int = 100,
    min_episode_margin: int = 5,
    min_rate_margin: float = 0.05,
) -> ScoutDecision:
    totals: dict[str, MethodTotal] = {}
    missing: list[str] = []
    incomplete: list[str] = []
    for method in methods:
        selected = [
            row
            for row in rows
            if row.get("method") == method and row.get("perturbation") == perturbation
        ]
        if not selected:
            missing.append(f"{method}/{perturbation}")
            continue
        total = MethodTotal(
            successes=sum(int(float(row["successes"])) for row in selected),
            episodes=sum(int(float(row["episodes"])) for row in selected),
        )
        totals[method] = total
        if total.episodes < min_episodes:
            incomplete.append(f"{method}/{perturbation} {total.successes}/{total.episodes}")

    if missing or incomplete:
        parts = []
        if missing:
            parts.append(f"missing {', '.join(missing)}")
        if incomplete:
            parts.append(f"incomplete {', '.join(incomplete)}")
        return ScoutDecision(False, False, "; ".join(parts))

    bgr = totals["bgr"]
    comparators = [method for method in methods if method != "bgr"]
    best_comparator = max(
        comparators,
        key=lambda method: (totals[method].rate, totals[method].successes),
    )
    best = totals[best_comparator]
    episode_margin = bgr.successes - best.successes
    rate_margin = bgr.rate - best.rate
    promote = episode_margin >= min_episode_margin and rate_margin >= min_rate_margin
    totals_detail = ", ".join(
        f"{method} {totals[method].successes}/{totals[method].episodes}"
        for method in methods
    )
    detail = (
        f"{perturbation} successes {totals_detail}; "
        f"best_comparator={best_comparator}; episode_margin={episode_margin}; "
        f"rate_margin={rate_margin:+.4f}; promotion_thresholds="
        f"+{min_episode_margin} episodes and +{min_rate_margin:.4f} rate"
    )
    return ScoutDecision(True, promote, detail)


def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Scout summary.csv or summary_available.csv.")
    parser.add_argument("--perturbation", default="occlusion_shift")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--min-episodes", type=int, default=100)
    parser.add_argument("--min-episode-margin", type=int, default=5)
    parser.add_argument("--min-rate-margin", type=float, default=0.05)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-promote", action="store_true")
    args = parser.parse_args()

    decision = evaluate_scout(
        read_rows(args.summary),
        perturbation=args.perturbation,
        methods=_parse_csv_tuple(args.methods),
        min_episodes=args.min_episodes,
        min_episode_margin=args.min_episode_margin,
        min_rate_margin=args.min_rate_margin,
    )
    if not decision.complete:
        status = "INCOMPLETE"
    else:
        status = "PROMOTE_FULL_GATE" if decision.promote else "CLOSE_NEGATIVE"
    print(f"[{status}] OpenVLA route scout: {decision.detail}")
    if args.require_promote and not decision.promote:
        return 1
    if args.require_complete and not decision.complete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
