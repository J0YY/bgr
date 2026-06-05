#!/usr/bin/env python3
"""Calibrate package-backed FetchReach goal-recovery curves.

This is a pre-comparison diagnostic for an external Gymnasium-Robotics route.
It does not compare BGR methods. It checks whether FetchReach-v4 can provide
non-saturated recovery curves under exact reset seeds and fixed goal
perturbations before spending effort on a full replay-training screen.
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
    replay_index: int
    sigma: float
    trials: int
    successes: int
    recovery: float


def package_versions() -> dict[str, str]:
    try:
        import gymnasium
        import gymnasium_robotics
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "FetchReach calibration requires gymnasium-robotics/gymnasium/mujoco "
            "in an isolated environment such as /tmp/bgr_pointmaze_venv."
        ) from exc
    return {
        "gymnasium": gymnasium.__version__,
        "gymnasium_robotics": gymnasium_robotics.__version__,
        "mujoco": mujoco.__version__,
    }


def parse_radii(value: str) -> np.ndarray:
    radii = np.array([float(item.strip()) for item in value.split(",") if item.strip()], dtype=float)
    if radii.ndim != 1 or len(radii) < 2:
        raise ValueError("--radii must contain at least two comma-separated values")
    if not np.all(np.diff(radii) > 0.0):
        raise ValueError("--radii must be strictly increasing")
    return radii


def adverse_direction(base_goal: np.ndarray, center: np.ndarray, rng: np.random.Generator, jitter: float) -> np.ndarray:
    direction = np.array(base_goal, dtype=float) - np.array(center, dtype=float)
    if float(np.linalg.norm(direction)) < 1e-9:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
    noise = rng.normal(0.0, 1.0, size=3)
    noise[2] *= 0.55
    direction = direction + float(jitter) * noise
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return direction / norm


def clipped_goal(base_goal: np.ndarray, direction: np.ndarray, sigma: float, center: np.ndarray, target_range: float) -> np.ndarray:
    low = center - target_range
    high = center + target_range
    return np.clip(base_goal + sigma * direction, low, high)


def rollout_success(env, *, goal: np.ndarray, seed: int, horizon: int, gain: float) -> bool:
    obs, _info = env.reset(seed=seed)
    unwrapped = env.unwrapped
    unwrapped.goal = np.array(goal, dtype=float)
    obs = unwrapped._get_obs()
    for _step in range(horizon):
        delta = np.array(obs["desired_goal"], dtype=float) - np.array(obs["achieved_goal"], dtype=float)
        action = np.zeros(4, dtype=float)
        action[:3] = np.clip(gain * delta, -1.0, 1.0)
        obs, _reward, terminated, truncated, _info = env.step(action)
        if bool(terminated):
            return True
        if bool(truncated):
            return False
    distance = float(np.linalg.norm(np.array(obs["achieved_goal"], dtype=float) - np.array(obs["desired_goal"], dtype=float)))
    return distance <= float(unwrapped.distance_threshold)


def calibrate(args: argparse.Namespace) -> tuple[list[CalibrationRow], dict[str, float | int | str]]:
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    env = gym.make(args.env_id)
    radii = parse_radii(args.radii)
    rows: list[CalibrationRow] = []
    try:
        for seed in range(args.seeds):
            base_rng = np.random.default_rng(args.seed_offset + seed)
            for replay_index in range(args.replay_states):
                obs, _info = env.reset(seed=args.seed_offset + seed * 1000 + replay_index)
                unwrapped = env.unwrapped
                center = np.array(unwrapped.initial_gripper_xpos, dtype=float)
                target_range = float(unwrapped.target_range)
                base_goal = np.array(obs["desired_goal"], dtype=float)
                for sigma in radii:
                    successes = 0
                    for trial in range(args.trials):
                        direction = adverse_direction(base_goal, center, base_rng, args.direction_jitter)
                        goal = clipped_goal(base_goal, direction, float(sigma), center, target_range)
                        trial_seed = args.seed_offset + seed * 100_000 + replay_index * 1000 + int(round(float(sigma) * 10_000)) + trial
                        if rollout_success(env, goal=goal, seed=trial_seed, horizon=args.horizon, gain=args.controller_gain):
                            successes += 1
                    rows.append(
                        CalibrationRow(
                            seed=seed,
                            replay_index=replay_index,
                            sigma=float(sigma),
                            trials=args.trials,
                            successes=successes,
                            recovery=successes / args.trials,
                        )
                    )
    finally:
        env.close()

    by_radius: list[float] = []
    for sigma in radii:
        values = [row.recovery for row in rows if abs(row.sigma - float(sigma)) < 1e-12]
        by_radius.append(float(np.mean(values)))
    curve = np.array(by_radius, dtype=float)
    summary = {
        "env_id": args.env_id,
        "seeds": args.seeds,
        "replay_states": args.replay_states,
        "trials": args.trials,
        "horizon": args.horizon,
        "controller_gain": args.controller_gain,
        "direction_jitter": args.direction_jitter,
        "clean_success": float(curve[0]),
        "rauc": float(recovery_auc(radii, curve, sigma_max=float(radii[-1]))),
        "r80": float(critical_radius(radii, curve, alpha=args.alpha)),
        "min_recovery": float(np.min(curve)),
        "max_recovery": float(np.max(curve)),
    }
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
    parser.add_argument("--env-id", default="FetchReach-v4")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--replay-states", type=int, default=4)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--radii", default="0.00,0.03,0.06,0.09,0.12,0.15")
    parser.add_argument("--horizon", type=int, default=18)
    parser.add_argument("--controller-gain", type=float, default=4.0)
    parser.add_argument("--direction-jitter", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.80)
    parser.add_argument("--seed-offset", type=int, default=91_000)
    args = parser.parse_args()

    rows, summary = calibrate(args)
    write_outputs(args.out, rows, summary)
    print(
        "FetchReach calibration: "
        f"clean={summary['clean_success']:.4f} rauc={summary['rauc']:.4f} "
        f"r80={summary['r80']:.4f} min={summary['min_recovery']:.4f} max={summary['max_recovery']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
