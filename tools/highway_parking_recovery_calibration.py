#!/usr/bin/env python3
"""Calibrate package-backed highway-env parking recovery curves.

This is a pre-comparison diagnostic for a different external benchmark route.
It uses highway-env's own parking-v0 dynamics, goal observation, reward, and
success predicate. It does not compare BGR methods.
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
        import scipy
        from importlib.metadata import version
    except ImportError as exc:
        raise SystemExit(
            "Highway parking calibration requires highway-env in an isolated "
            "environment, for example /tmp/bgr_highway311_venv."
        ) from exc
    return {
        "gymnasium": gymnasium.__version__,
        "highway-env": version("highway-env"),
        "numpy": np.__version__,
        "pygame": pygame.version.ver,
        "scipy": scipy.__version__,
    }


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def reset_perturbed(env, *, seed: int, sigma: float, angle: float):
    env.reset(seed=seed)
    unwrapped = env.unwrapped
    vehicle = unwrapped.controlled_vehicles[0]
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    vehicle.position = np.array(vehicle.position, dtype=float) + float(sigma) * direction
    vehicle.heading = float(vehicle.heading + 0.25 * float(sigma) * np.sin(angle + 0.7))
    vehicle.speed = 0.0
    return unwrapped.observation_type_parking.observe()


def controller_action(obs: dict[str, np.ndarray]) -> np.ndarray:
    achieved = np.array(obs["achieved_goal"], dtype=float)
    desired = np.array(obs["desired_goal"], dtype=float)
    position = achieved[:2] * 100.0
    goal = desired[:2] * 100.0
    heading = float(np.arctan2(achieved[5], achieved[4]))
    goal_heading = float(np.arctan2(desired[5], desired[4]))
    vector = goal - position
    distance = float(np.linalg.norm(vector))

    if distance > 0.4:
        target_angle = float(np.arctan2(vector[1], vector[0]))
    else:
        target_angle = goal_heading
    front_error = wrap_angle(target_angle - heading)
    back_error = wrap_angle(target_angle - (heading + np.pi))
    reverse = abs(back_error) < abs(front_error) and distance > 1.5
    heading_error = back_error if reverse else front_error
    steering = float(np.clip(2.5 * heading_error, -1.0, 1.0))
    acceleration = float(np.clip(0.18 * distance, -0.6, 0.8))
    if reverse:
        acceleration = -min(0.7, max(0.25, 0.12 * distance))
    if distance < 2.5:
        speed = float(np.linalg.norm(achieved[2:4]) * 5.0)
        acceleration = float(np.clip(-0.35 * speed, -0.5, 0.5))
        steering = float(np.clip(2.5 * wrap_angle(goal_heading - heading), -1.0, 1.0))
    return np.array([acceleration, steering], dtype=np.float32)


def rollout(env, *, seed: int, sigma: float, angle: float, horizon: int) -> CalibrationRow:
    obs = reset_perturbed(env, seed=seed, sigma=sigma, angle=angle)
    success = False
    crashed = False
    step = 0
    for step in range(1, horizon + 1):
        obs, _reward, terminated, truncated, info = env.step(controller_action(obs))
        success = bool(info.get("is_success"))
        crashed = bool(info.get("crashed"))
        if terminated or truncated:
            break
    return CalibrationRow(
        seed=seed,
        sigma=float(sigma),
        success=int(success),
        crashed=int(crashed),
        steps=int(step),
    )


def summarize(rows: list[CalibrationRow], radii: np.ndarray, *, alpha: float) -> dict[str, float | int | str]:
    curve = []
    crashes = []
    for sigma in radii:
        radius_rows = [row for row in rows if abs(row.sigma - float(sigma)) < 1e-12]
        if not radius_rows:
            raise ValueError(f"missing rows for sigma={sigma}")
        curve.append(float(np.mean([row.success for row in radius_rows])))
        crashes.append(float(np.mean([row.crashed for row in radius_rows])))
    curve_array = np.array(curve, dtype=float)
    crash_array = np.array(crashes, dtype=float)
    clean_success = float(curve_array[0])
    min_recovery = float(np.min(curve_array))
    max_recovery = float(np.max(curve_array))
    if clean_success < 0.80:
        decision = "reject-calibration-low-clean-success"
    elif max_recovery - min_recovery < 0.20:
        decision = "reject-calibration-flat-recovery"
    else:
        decision = "usable-calibration"
    return {
        "env_id": "parking-v0",
        "seeds": len({row.seed for row in rows}),
        "radii": [float(sigma) for sigma in radii],
        "clean_success": clean_success,
        "min_recovery": min_recovery,
        "max_recovery": max_recovery,
        "mean_crash_rate": float(np.mean(crash_array)),
        "rauc": float(recovery_auc(radii, curve_array, sigma_max=float(radii[-1]))),
        "r80": float(critical_radius(radii, curve_array, alpha=alpha)),
        "decision": decision,
    }


def calibrate(args: argparse.Namespace) -> tuple[list[CalibrationRow], dict[str, float | int | str]]:
    try:
        import gymnasium as gym
        import highway_env  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Highway parking calibration requires highway-env in an isolated "
            "environment, for example /tmp/bgr_highway311_venv."
        ) from exc

    radii = parse_radii(args.radii)
    env = gym.make(
        args.env_id,
        render_mode=None,
        config={
            "duration": args.horizon,
            "vehicles_count": 0,
            "add_walls": True,
        },
    )
    rows: list[CalibrationRow] = []
    try:
        for seed in range(args.seeds):
            angle = (args.angle_stride * seed) % (2 * np.pi)
            for sigma in radii:
                rows.append(rollout(env, seed=seed, sigma=float(sigma), angle=angle, horizon=args.horizon))
    finally:
        env.close()
    summary = summarize(rows, radii, alpha=args.alpha)
    summary["env_id"] = args.env_id
    summary["horizon"] = args.horizon
    summary["angle_stride"] = args.angle_stride
    return rows, summary


def write_outputs(out_dir: Path, rows: list[CalibrationRow], summary: dict[str, float | int | str]) -> None:
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
    parser.add_argument("--env-id", default="parking-v0")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--radii", default="0,1,2,3,4,5,6,8,10")
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--angle-stride", type=float, default=1.618)
    parser.add_argument("--alpha", type=float, default=0.80)
    args = parser.parse_args()

    rows, summary = calibrate(args)
    write_outputs(args.out, rows, summary)
    print(
        "Highway parking calibration: "
        f"clean={summary['clean_success']:.4f} "
        f"range={summary['min_recovery']:.4f}--{summary['max_recovery']:.4f} "
        f"crash={summary['mean_crash_rate']:.4f} "
        f"decision={summary['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
