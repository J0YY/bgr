#!/usr/bin/env python3
"""Calibrate package-backed highway-env lane recovery curves.

This is a pre-comparison diagnostic for a different external benchmark route.
It uses highway-env's own highway-fast-v0 dynamics, kinematic observation,
discrete controller actions, and collision termination. It does not compare BGR
methods.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bgr.metrics import critical_radius, recovery_auc


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    seed: int
    sigma: float
    success: int
    crashed: int
    on_road: int
    steps: int


def parse_radii(value: str) -> np.ndarray:
    radii = np.array([float(item.strip()) for item in value.split(",") if item.strip()], dtype=float)
    if radii.ndim != 1 or len(radii) < 2:
        raise ValueError("--radii must contain at least two comma-separated values")
    if not np.all(np.diff(radii) > 0.0):
        raise ValueError("--radii must be strictly increasing")
    return radii


def package_versions() -> dict[str, str]:
    try:
        import gymnasium
        import highway_env
        import pygame
        from importlib.metadata import version
    except ImportError as exc:
        raise SystemExit(
            "Highway lane calibration requires highway-env in an isolated "
            "environment, for example /tmp/bgr_highway311_venv."
        ) from exc
    return {
        "gymnasium": gymnasium.__version__,
        "highway-env": version("highway-env"),
        "numpy": np.__version__,
        "pygame": pygame.version.ver,
    }


def reset_perturbed(env, *, seed: int, sigma: float, lateral_sign: float, heading_gain: float):
    obs, _info = env.reset(seed=seed)
    vehicle = env.unwrapped.vehicle
    vehicle.position = np.array(vehicle.position, dtype=float)
    vehicle.position[1] += float(lateral_sign) * float(sigma)
    vehicle.heading = float(vehicle.heading + float(lateral_sign) * float(heading_gain) * float(sigma))
    return obs


def controller_action(_obs: np.ndarray, _step: int) -> int:
    # Fixed non-learning policy: hold lane and speed. This calibrates whether
    # the environment exposes clean-but-perturbable recovery states.
    return 1


def rollout(env, *, seed: int, sigma: float, horizon: int, heading_gain: float) -> CalibrationRow:
    lateral_sign = -1.0 if seed % 2 else 1.0
    obs = reset_perturbed(env, seed=seed, sigma=sigma, lateral_sign=lateral_sign, heading_gain=heading_gain)
    crashed = False
    on_road = True
    step = 0
    for step in range(1, horizon + 1):
        obs, _reward, terminated, truncated, info = env.step(controller_action(obs, step))
        vehicle = env.unwrapped.vehicle
        crashed = crashed or bool(info.get("crashed", getattr(vehicle, "crashed", False)))
        on_road = bool(getattr(vehicle, "on_road", True))
        if terminated or truncated:
            break
    success = (not crashed) and on_road and step >= horizon
    return CalibrationRow(
        seed=seed,
        sigma=float(sigma),
        success=int(success),
        crashed=int(crashed),
        on_road=int(on_road),
        steps=int(step),
    )


def summarize(rows: list[CalibrationRow], radii: np.ndarray, *, alpha: float) -> dict[str, float | int | str | list[float]]:
    curve = []
    crashes = []
    on_road = []
    for sigma in radii:
        radius_rows = [row for row in rows if abs(row.sigma - float(sigma)) < 1e-12]
        if not radius_rows:
            raise ValueError(f"missing rows for sigma={sigma}")
        curve.append(float(np.mean([row.success for row in radius_rows])))
        crashes.append(float(np.mean([row.crashed for row in radius_rows])))
        on_road.append(float(np.mean([row.on_road for row in radius_rows])))
    curve_array = np.array(curve, dtype=float)
    clean_success = float(curve_array[0])
    min_recovery = float(np.min(curve_array))
    max_recovery = float(np.max(curve_array))
    r80 = float(critical_radius(radii, curve_array, alpha=alpha))
    recovery_range = max_recovery - min_recovery
    if clean_success < 0.80:
        decision = "reject-calibration-low-clean-success"
    elif recovery_range < 0.20 - 1e-12:
        decision = "reject-calibration-flat-recovery"
    elif r80 >= float(radii[-1]) - 1e-12:
        decision = "reject-calibration-saturated-radius"
    else:
        decision = "usable-calibration"
    return {
        "env_id": "highway-fast-v0",
        "seeds": len({row.seed for row in rows}),
        "radii": [float(sigma) for sigma in radii],
        "clean_success": clean_success,
        "min_recovery": min_recovery,
        "max_recovery": max_recovery,
        "mean_crash_rate": float(np.mean(crashes)),
        "mean_on_road_rate": float(np.mean(on_road)),
        "rauc": float(recovery_auc(radii, curve_array, sigma_max=float(radii[-1]))),
        "r80": r80,
        "decision": decision,
    }


def calibrate(args: argparse.Namespace) -> tuple[list[CalibrationRow], dict[str, float | int | str | list[float]]]:
    try:
        import gymnasium as gym
        import highway_env  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Highway lane calibration requires highway-env in an isolated "
            "environment, for example /tmp/bgr_highway311_venv."
        ) from exc

    radii = parse_radii(args.radii)
    env = gym.make(
        args.env_id,
        render_mode=None,
        config={
            "duration": args.horizon,
            "vehicles_count": args.vehicles_count,
            "lanes_count": args.lanes_count,
            "simulation_frequency": args.simulation_frequency,
            "policy_frequency": args.policy_frequency,
        },
    )
    rows: list[CalibrationRow] = []
    try:
        for seed in range(args.seeds):
            for sigma in radii:
                rows.append(rollout(env, seed=seed, sigma=float(sigma), horizon=args.horizon, heading_gain=args.heading_gain))
    finally:
        env.close()
    summary = summarize(rows, radii, alpha=args.alpha)
    summary.update(
        {
            "env_id": args.env_id,
            "horizon": args.horizon,
            "vehicles_count": args.vehicles_count,
            "lanes_count": args.lanes_count,
            "simulation_frequency": args.simulation_frequency,
            "policy_frequency": args.policy_frequency,
            "heading_gain": args.heading_gain,
            "policy": "idle-lane-keep",
        }
    )
    return rows, summary


def write_outputs(out_dir: Path, rows: list[CalibrationRow], summary: dict[str, float | int | str | list[float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "package_versions.json").open("w", encoding="utf-8") as handle:
        json.dump(package_versions(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (out_dir / "recovery_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--env-id", default="highway-fast-v0")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--radii", default="0,1,2,3,4,5,6")
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--vehicles-count", type=int, default=15)
    parser.add_argument("--lanes-count", type=int, default=3)
    parser.add_argument("--simulation-frequency", type=int, default=15)
    parser.add_argument("--policy-frequency", type=int, default=5)
    parser.add_argument("--heading-gain", type=float, default=0.12)
    parser.add_argument("--alpha", type=float, default=0.80)
    args = parser.parse_args()

    rows, summary = calibrate(args)
    write_outputs(args.out, rows, summary)
    print(
        "Highway lane calibration: "
        f"clean={summary['clean_success']:.4f} "
        f"range={summary['min_recovery']:.4f}--{summary['max_recovery']:.4f} "
        f"r80={summary['r80']:.4f} "
        f"crash={summary['mean_crash_rate']:.4f} "
        f"decision={summary['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
