#!/usr/bin/env python3
"""Calibrate package-backed Fetch object-goal recovery curves.

This is a pre-comparison diagnostic for harder Gymnasium-Robotics Fetch tasks.
It uses exact reset seeds, perturbs the object goal, and evaluates a fixed
scripted controller. It does not compare BGR methods.
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
class FetchObjectReplayState:
    seed: int
    replay_index: int
    base_goal: list[float]
    center: list[float]
    target_range: float
    direction: list[float]


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    seed: int
    replay_index: int
    sigma: float
    trial: int
    success: int
    final_distance: float
    steps: int


def package_versions() -> dict[str, str]:
    try:
        import gymnasium
        import gymnasium_robotics
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "Fetch object calibration requires gymnasium-robotics/gymnasium/mujoco "
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
    noise[2] *= 0.25
    direction = direction + float(jitter) * noise
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return direction / norm


def clipped_goal(base_goal: np.ndarray, direction: np.ndarray, sigma: float, center: np.ndarray, target_range: float) -> np.ndarray:
    low = center - target_range
    high = center + target_range
    goal = np.clip(base_goal + float(sigma) * direction, low, high)
    goal[2] = base_goal[2]
    return goal


def push_action(obs: dict[str, np.ndarray], *, threshold: float, gain: float) -> np.ndarray:
    raw = np.array(obs["observation"], dtype=float)
    gripper = raw[0:3]
    obj = raw[3:6]
    goal = np.array(obs["desired_goal"], dtype=float)
    distance = float(np.linalg.norm(np.array(obs["achieved_goal"], dtype=float) - goal))
    if distance <= threshold:
        return np.zeros(4, dtype=float)

    direction_xy = goal[:2] - obj[:2]
    norm = float(np.linalg.norm(direction_xy))
    if norm < 1e-8:
        unit_xy = np.array([1.0, 0.0], dtype=float)
    else:
        unit_xy = direction_xy / norm

    behind = obj[:2] - 0.035 * unit_xy
    if float(np.linalg.norm(gripper[:2] - behind)) > 0.018:
        target = np.array([behind[0], behind[1], obj[2] + 0.002], dtype=float)
    else:
        target = np.array([goal[0] + 0.04 * unit_xy[0], goal[1] + 0.04 * unit_xy[1], obj[2] + 0.002], dtype=float)

    action = np.zeros(4, dtype=float)
    action[:3] = np.clip(float(gain) * (target - gripper), -1.0, 1.0)
    return action


def rollout_success(env, *, base_seed: int, goal: np.ndarray, horizon: int, gain: float) -> tuple[bool, float, int]:
    obs, _info = env.reset(seed=base_seed)
    unwrapped = env.unwrapped
    unwrapped.goal = np.array(goal, dtype=float)
    obs = unwrapped._get_obs()
    threshold = float(unwrapped.distance_threshold)
    for step in range(horizon + 1):
        distance = float(np.linalg.norm(np.array(obs["achieved_goal"], dtype=float) - np.array(obs["desired_goal"], dtype=float)))
        if distance <= threshold:
            return True, distance, step
        if step == horizon:
            return False, distance, step
        obs, _reward, _terminated, truncated, _info = env.step(push_action(obs, threshold=threshold, gain=gain))
        if bool(truncated):
            distance = float(
                np.linalg.norm(np.array(obs["achieved_goal"], dtype=float) - np.array(obs["desired_goal"], dtype=float))
            )
            return distance <= threshold, distance, step + 1
    raise AssertionError("unreachable")


def make_replay_states(env, args: argparse.Namespace) -> list[FetchObjectReplayState]:
    states: list[FetchObjectReplayState] = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(args.seed_offset + seed)
        for replay_index in range(args.replay_states):
            base_seed = args.seed_offset + seed * 1000 + replay_index
            obs, _info = env.reset(seed=base_seed)
            unwrapped = env.unwrapped
            center = np.array(unwrapped.initial_gripper_xpos, dtype=float)
            base_goal = np.array(obs["desired_goal"], dtype=float)
            direction = adverse_direction(base_goal, center, rng, args.direction_jitter)
            states.append(
                FetchObjectReplayState(
                    seed=seed,
                    replay_index=replay_index,
                    base_goal=base_goal.tolist(),
                    center=center.tolist(),
                    target_range=float(unwrapped.target_range),
                    direction=direction.tolist(),
                )
            )
    return states


def calibrate(args: argparse.Namespace) -> tuple[list[CalibrationRow], dict[str, float | int | str]]:
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    radii = parse_radii(args.radii)
    env = gym.make(args.env_id)
    rows: list[CalibrationRow] = []
    try:
        states = make_replay_states(env, args)
        for state in states:
            base_goal = np.array(state.base_goal, dtype=float)
            center = np.array(state.center, dtype=float)
            direction = np.array(state.direction, dtype=float)
            for sigma in radii:
                goal = clipped_goal(base_goal, direction, float(sigma), center, float(state.target_range))
                for trial in range(args.trials):
                    success, distance, steps = rollout_success(
                        env,
                        base_seed=state.seed * 1000 + args.seed_offset + state.replay_index,
                        goal=goal,
                        horizon=args.horizon,
                        gain=args.controller_gain,
                    )
                    rows.append(
                        CalibrationRow(
                            seed=state.seed,
                            replay_index=state.replay_index,
                            sigma=float(sigma),
                            trial=trial,
                            success=int(success),
                            final_distance=float(distance),
                            steps=int(steps),
                        )
                    )
    finally:
        env.close()

    by_radius: list[float] = []
    for sigma in radii:
        values = [float(row.success) for row in rows if abs(row.sigma - float(sigma)) < 1e-12]
        by_radius.append(float(np.mean(values)))
    curve = np.array(by_radius, dtype=float)
    summary = {
        "env_id": args.env_id,
        "seeds": args.seeds,
        "replay_states": args.replay_states,
        "trials": args.trials,
        "radii": ",".join(f"{float(radius):.4f}" for radius in radii),
        "horizon": args.horizon,
        "controller": "scripted_push",
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
    parser.add_argument("--env-id", default="FetchPush-v4")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--replay-states", type=int, default=4)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--radii", default="0.00,0.02,0.04,0.06,0.08,0.12")
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--controller-gain", type=float, default=6.0)
    parser.add_argument("--direction-jitter", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.80)
    parser.add_argument("--seed-offset", type=int, default=121_000)
    args = parser.parse_args()

    rows, summary = calibrate(args)
    write_outputs(args.out, rows, summary)
    print(
        "Fetch object calibration: "
        f"env={summary['env_id']} clean={summary['clean_success']:.4f} "
        f"rauc={summary['rauc']:.4f} r80={summary['r80']:.4f} "
        f"min={summary['min_recovery']:.4f} max={summary['max_recovery']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
