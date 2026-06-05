#!/usr/bin/env python
"""Check the preregistered OpenVLA-OFT perturbation promotion gate."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_METHODS = ("bgr", "official", "random")
DEFAULT_PERTURBATIONS = ("blur", "brightness", "occlusion", "shift")


@dataclass(frozen=True)
class MethodTotal:
    successes: int
    episodes: int

    @property
    def rate(self) -> float:
        return self.successes / self.episodes


@dataclass(frozen=True)
class GateDecision:
    complete: bool
    passed: bool
    detail: str


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_gate(
    perturb_rows: list[dict[str, str]],
    *,
    clean_rows: list[dict[str, str]] | None = None,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    non_identity_perturbations: tuple[str, ...] = DEFAULT_PERTURBATIONS,
    min_episode_margin: int = 10,
    min_rate_margin: float = 0.02,
    max_identity_deficit: int = 1,
) -> GateDecision:
    missing = _missing_rows(perturb_rows, methods, non_identity_perturbations)
    if missing:
        return GateDecision(False, False, f"incomplete perturbation summary; missing {', '.join(missing)}")

    totals = {
        method: _total(
            perturb_rows,
            method,
            perturbations=set(non_identity_perturbations),
        )
        for method in methods
    }
    bgr = totals["bgr"]
    comparators = [method for method in methods if method != "bgr"]
    best_comparator = max(comparators, key=lambda method: totals[method].successes)
    best = totals[best_comparator]

    identity_totals = _identity_totals(perturb_rows, clean_rows, methods)
    missing_identity = [method for method in methods if method not in identity_totals]
    if missing_identity:
        return GateDecision(False, False, f"incomplete identity summary; missing {', '.join(missing_identity)}")
    bgr_identity = identity_totals["bgr"]
    best_identity = max(identity_totals[method].successes for method in comparators)

    episode_margin = bgr.successes - best.successes
    rate_margin = bgr.rate - best.rate
    identity_deficit = best_identity - bgr_identity.successes
    passed = (
        episode_margin >= min_episode_margin
        and rate_margin >= min_rate_margin
        and identity_deficit <= max_identity_deficit
    )
    return GateDecision(
        True,
        passed,
        (
            f"non-identity successes BGR {bgr.successes}/{bgr.episodes}, "
            f"official {totals['official'].successes}/{totals['official'].episodes}, "
            f"random {totals['random'].successes}/{totals['random'].episodes}; "
            f"best_comparator={best_comparator}; episode_margin={episode_margin}; "
            f"rate_margin={rate_margin:.4f}; identity_deficit={identity_deficit}"
        ),
    )


def _missing_rows(
    rows: list[dict[str, str]],
    methods: tuple[str, ...],
    perturbations: tuple[str, ...],
) -> list[str]:
    present = {(row.get("method"), row.get("perturbation")) for row in rows}
    return [
        f"{method}/{perturbation}"
        for method in methods
        for perturbation in perturbations
        if (method, perturbation) not in present
    ]


def _total(rows: list[dict[str, str]], method: str, *, perturbations: set[str]) -> MethodTotal:
    selected = [row for row in rows if row.get("method") == method and row.get("perturbation") in perturbations]
    if not selected:
        raise ValueError(f"no rows for {method=} {perturbations=}")
    return MethodTotal(
        successes=sum(int(float(row["successes"])) for row in selected),
        episodes=sum(int(float(row["episodes"])) for row in selected),
    )


def _identity_totals(
    perturb_rows: list[dict[str, str]],
    clean_rows: list[dict[str, str]] | None,
    methods: tuple[str, ...],
) -> dict[str, MethodTotal]:
    totals: dict[str, MethodTotal] = {}
    if clean_rows is not None:
        for method in methods:
            selected = [row for row in clean_rows if row.get("method") == method]
            if selected:
                totals[method] = MethodTotal(
                    successes=sum(int(float(row["successes"])) for row in selected),
                    episodes=sum(int(float(row["episodes"])) for row in selected),
                )
    for method in methods:
        if method in totals:
            continue
        selected = [
            row
            for row in perturb_rows
            if row.get("method") == method and row.get("perturbation") == "identity"
        ]
        if selected:
            totals[method] = MethodTotal(
                successes=sum(int(float(row["successes"])) for row in selected),
                episodes=sum(int(float(row["episodes"])) for row in selected),
            )
    return totals


def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perturb-summary", type=Path, required=True, help="Perturbation summary.csv.")
    parser.add_argument(
        "--clean-summary",
        type=Path,
        help="Optional clean identity summary.csv. Falls back to perturbation identity rows when omitted.",
    )
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--non-identity-perturbations", default=",".join(DEFAULT_PERTURBATIONS))
    parser.add_argument("--min-episode-margin", type=int, default=10)
    parser.add_argument("--min-rate-margin", type=float, default=0.02)
    parser.add_argument("--max-identity-deficit", type=int, default=1)
    parser.add_argument("--require-complete", action="store_true", help="Exit nonzero if required rows are missing.")
    parser.add_argument("--require-pass", action="store_true", help="Exit nonzero unless the gate passes.")
    args = parser.parse_args()

    clean_rows = read_rows(args.clean_summary) if args.clean_summary else None
    decision = evaluate_gate(
        read_rows(args.perturb_summary),
        clean_rows=clean_rows,
        methods=_parse_csv_tuple(args.methods),
        non_identity_perturbations=_parse_csv_tuple(args.non_identity_perturbations),
        min_episode_margin=args.min_episode_margin,
        min_rate_margin=args.min_rate_margin,
        max_identity_deficit=args.max_identity_deficit,
    )
    if not decision.complete:
        status = "INCOMPLETE"
    else:
        status = "PASS" if decision.passed else "FAIL"
    print(f"[{status}] OpenVLA perturb promotion gate: {decision.detail}")
    if args.require_pass and not decision.passed:
        return 1
    if args.require_complete and not decision.complete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
