#!/usr/bin/env python3
"""Stress-test the grid-margin feasibility witness.

This diagnostic does not train or compare methods. It samples the boundary
radii BGR would prefer in the controlled grid-margin benchmark, corrupts the
binary feasibility witness with fixed false-positive/false-negative rates, and
measures how many accepted samples remain valid boundary samples under the
exact witness.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bgr.envs.grid_recovery import GridMarginRecoveryBenchmark
from bgr.experiments.grid_margin import evaluate
from bgr.metrics import critical_radius
from bgr.samplers import sample_boundary_radius
from scripts.run_toy_experiment import _load_config


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    false_positive_rate: float
    false_negative_rate: float


@dataclass(frozen=True, slots=True)
class SensitivityRow:
    scenario: str
    seed: int
    attempted_samples: int
    accepted_samples: int
    accepted_fraction: float
    true_valid_accept_rate: float
    invalid_accept_rate: float
    boundary_hit_accept_rate: float
    true_boundary_recall: float
    mean_abs_boundary_error: float
    clean: float
    rauc: float
    median_r80: float


DEFAULT_SCENARIOS = (
    Scenario("exact", 0.0, 0.0),
    Scenario("false_negative_10", 0.0, 0.10),
    Scenario("false_positive_10", 0.10, 0.0),
    Scenario("symmetric_10", 0.10, 0.10),
    Scenario("symmetric_20", 0.20, 0.20),
)


def parse_scenarios(value: str) -> list[Scenario]:
    if not value:
        return list(DEFAULT_SCENARIOS)
    scenarios: list[Scenario] = []
    for item in value.split(","):
        fields = [field.strip() for field in item.split(":")]
        if len(fields) != 3:
            raise ValueError("scenario entries must be name:false_positive:false_negative")
        name, fp_text, fn_text = fields
        fp = float(fp_text)
        fn = float(fn_text)
        if not 0.0 <= fp <= 1.0 or not 0.0 <= fn <= 1.0:
            raise ValueError("false-positive and false-negative rates must be in [0, 1]")
        scenarios.append(Scenario(name, fp, fn))
    return scenarios


def seeds_from_config(config: dict, override: str | None) -> list[int]:
    if override:
        seeds = [int(item.strip()) for item in override.split(",") if item.strip()]
    else:
        seeds = [int(seed) for seed in config["experiment"]["seeds"]]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def build_benchmark(config: dict, seed: int) -> GridMarginRecoveryBenchmark:
    exp = config["experiment"]
    return GridMarginRecoveryBenchmark(
        num_tasks=int(exp["num_tasks"]),
        grid_size=int(exp["grid_size"]),
        obstacle_prob=float(exp["obstacle_prob"]),
        replay_states_per_task=int(exp["replay_states_per_task"]),
        max_offset=int(exp["max_offset"]),
        learning_rate=float(exp["learning_rate"]),
        seed=seed,
        feasible_radius_floor=float(exp.get("feasible_radius_floor", 0.35)),
        feasible_radius_base=float(exp.get("feasible_radius_base", 0.45)),
        feasible_radius_path_scale=float(exp.get("feasible_radius_path_scale", 0.45)),
        feasible_radius_noise=float(exp.get("feasible_radius_noise", 0.03)),
        initial_margin_base=float(exp.get("initial_margin_base", 0.12)),
        initial_margin_path_scale=float(exp.get("initial_margin_path_scale", 0.22)),
        initial_margin_noise=float(exp.get("initial_margin_noise", 0.05)),
        initial_margin_min=float(exp.get("initial_margin_min", 0.05)),
        initial_margin_max=float(exp.get("initial_margin_max", 0.45)),
        clean_success_min=float(exp.get("clean_success_min", 0.80)),
        clean_success_max=float(exp.get("clean_success_max", 0.96)),
        temperature_min=float(exp.get("temperature_min", 0.045)),
        temperature_max=float(exp.get("temperature_max", 0.10)),
        boundary_width=float(exp.get("boundary_width", 0.15)),
    )


def exact_radii(bench: GridMarginRecoveryBenchmark, eval_grid: np.ndarray, alpha: float) -> np.ndarray:
    radii = []
    for replay_idx in range(len(bench.states)):
        curve = np.array([bench.success_prob(replay_idx, float(sigma)) for sigma in eval_grid], dtype=float)
        radii.append(critical_radius(eval_grid, curve, alpha=alpha))
    return np.array(radii, dtype=float)


def corrupt_witness(true_feasible: bool, scenario: Scenario, rng: np.random.Generator) -> bool:
    if true_feasible:
        return bool(rng.random() >= scenario.false_negative_rate)
    return bool(rng.random() < scenario.false_positive_rate)


def run_scenario(
    config: dict,
    scenario: Scenario,
    seed: int,
    *,
    samples_per_state: int,
    boundary_band: float,
    feasibility_threshold: float,
) -> SensitivityRow:
    exp = config["experiment"]
    bgr_cfg = config.get("bgr", {})
    alpha = float(exp.get("alpha", 0.8))
    eval_grid = np.linspace(0.0, 1.0, int(exp.get("eval_grid_size", 9)))
    bench = build_benchmark(config, seed)
    metrics = evaluate(bench, eval_grid, alpha)
    radii = exact_radii(bench, eval_grid, alpha)
    rng = np.random.default_rng(80_000 + 997 * seed + stable_scenario_offset(scenario.name))

    attempted = accepted = 0
    accepted_valid = 0
    accepted_boundary = 0
    true_boundary_attempted = 0
    true_boundary_accepted = 0
    errors: list[float] = []
    for replay_idx, radius in enumerate(radii):
        for _ in range(samples_per_state):
            sigma = sample_boundary_radius(
                rng,
                float(radius),
                1.0,
                radius_noise=float(bgr_cfg.get("radius_noise", 0.07)),
            )
            true_feasible = bench.feasibility(replay_idx, sigma) >= feasibility_threshold
            boundary_hit = abs(sigma - float(radius)) <= boundary_band
            if true_feasible and boundary_hit:
                true_boundary_attempted += 1
            attempted += 1
            observed_feasible = corrupt_witness(true_feasible, scenario, rng)
            if not observed_feasible:
                continue
            accepted += 1
            accepted_valid += int(true_feasible)
            accepted_boundary += int(boundary_hit)
            true_boundary_accepted += int(true_feasible and boundary_hit)
            errors.append(abs(sigma - float(radius)))

    return SensitivityRow(
        scenario=scenario.name,
        seed=seed,
        attempted_samples=attempted,
        accepted_samples=accepted,
        accepted_fraction=accepted / max(1, attempted),
        true_valid_accept_rate=accepted_valid / max(1, accepted),
        invalid_accept_rate=1.0 - accepted_valid / max(1, accepted),
        boundary_hit_accept_rate=accepted_boundary / max(1, accepted),
        true_boundary_recall=true_boundary_accepted / max(1, true_boundary_attempted),
        mean_abs_boundary_error=float(np.mean(errors)) if errors else 0.0,
        clean=float(metrics["clean"]),
        rauc=float(metrics["rauc"]),
        median_r80=float(metrics["median_r80"]),
    )


def stable_scenario_offset(name: str) -> int:
    value = 0
    for char in name:
        value = (value * 131 + ord(char)) % 10_000
    return value


def summarize(rows: list[SensitivityRow]) -> list[dict[str, str]]:
    by_scenario: dict[str, list[SensitivityRow]] = {}
    for row in rows:
        by_scenario.setdefault(row.scenario, []).append(row)
    summary_rows: list[dict[str, str]] = []
    for scenario, items in sorted(by_scenario.items()):
        exact_valid = mean_metric(by_scenario.get("exact", items), "true_valid_accept_rate")
        exact_recall = mean_metric(by_scenario.get("exact", items), "true_boundary_recall")
        summary_rows.append(
            {
                "scenario": scenario,
                "seeds": str(len(items)),
                "accepted_fraction_mean": f"{mean_metric(items, 'accepted_fraction'):.6f}",
                "true_valid_accept_rate_mean": f"{mean_metric(items, 'true_valid_accept_rate'):.6f}",
                "invalid_accept_rate_mean": f"{mean_metric(items, 'invalid_accept_rate'):.6f}",
                "boundary_hit_accept_rate_mean": f"{mean_metric(items, 'boundary_hit_accept_rate'):.6f}",
                "true_boundary_recall_mean": f"{mean_metric(items, 'true_boundary_recall'):.6f}",
                "mean_abs_boundary_error_mean": f"{mean_metric(items, 'mean_abs_boundary_error'):.6f}",
                "valid_rate_drop_vs_exact": f"{mean_metric(items, 'true_valid_accept_rate') - exact_valid:.6f}",
                "recall_drop_vs_exact": f"{mean_metric(items, 'true_boundary_recall') - exact_recall:.6f}",
                "clean_mean": f"{mean_metric(items, 'clean'):.6f}",
                "rauc_mean": f"{mean_metric(items, 'rauc'):.6f}",
                "median_r80_mean": f"{mean_metric(items, 'median_r80'):.6f}",
            }
        )
    return summary_rows


def mean_metric(rows: list[SensitivityRow], field: str) -> float:
    return float(np.mean([float(getattr(row, field)) for row in rows]))


def write_outputs(out_dir: Path, rows: list[SensitivityRow], summary_rows: list[dict[str, str]], config: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "sample_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grid_margin_full_30seed.yaml")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--samples-per-state", type=int, default=24)
    parser.add_argument("--boundary-band", type=float, default=0.08)
    parser.add_argument("--feasibility-threshold", type=float, default=0.55)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    seeds = seeds_from_config(config, args.seeds)
    scenarios = parse_scenarios(args.scenarios)
    rows: list[SensitivityRow] = []
    for scenario in scenarios:
        for seed in seeds:
            rows.append(
                run_scenario(
                    config,
                    scenario,
                    seed,
                    samples_per_state=args.samples_per_state,
                    boundary_band=args.boundary_band,
                    feasibility_threshold=args.feasibility_threshold,
                )
            )
    summary_rows = summarize(rows)
    write_outputs(args.out, rows, summary_rows, config)
    for row in summary_rows:
        print(
            f"{row['scenario']}: valid_accept={row['true_valid_accept_rate_mean']} "
            f"boundary_recall={row['true_boundary_recall_mean']} "
            f"invalid_accept={row['invalid_accept_rate_mean']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
