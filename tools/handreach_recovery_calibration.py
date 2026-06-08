#!/usr/bin/env python3
"""Calibrate package-backed Gymnasium-Robotics HandReach recovery curves.

This is a pre-comparison diagnostic for a ShadowHand route. It uses
Gymnasium-Robotics HandReach-v3 goals and dynamics, perturbs the hand joint
state, and evaluates a fixed random-shooting controller. It does not compare
BGR methods.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from bgr.metrics import critical_radius, recovery_auc

ACTION_DIMS_BY_FINGER = {
    0: [2, 3, 4],
    1: [5, 6, 7],
    2: [8, 9, 10],
    3: [11, 12, 13, 14],
    4: [15, 16, 17, 18, 19],
}
PERTURBED_QPOS = np.array(
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    dtype=int,
)


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    seed: int
    sigma: float
    success: int
    best_distance: float
    initial_distance: float
    active_fingers: str
    active_action_dims: str


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
        import gymnasium_robotics
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "HandReach calibration requires gymnasium-robotics/gymnasium/mujoco "
            "in an isolated environment such as /tmp/bgr_pointmaze_venv."
        ) from exc
    return {
        "gymnasium": gymnasium.__version__,
        "gymnasium_robotics": gymnasium_robotics.__version__,
        "gymnasium-robotics-package": version("gymnasium-robotics"),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }


def observation_distance(obs: dict[str, np.ndarray]) -> float:
    return float(np.linalg.norm(np.asarray(obs["achieved_goal"], dtype=float) - np.asarray(obs["desired_goal"], dtype=float)))


def action_from_qpos(env) -> np.ndarray:
    unwrapped = env.unwrapped
    ctrl_range = np.asarray(unwrapped.model.actuator_ctrlrange, dtype=float)
    center = (ctrl_range[:, 0] + ctrl_range[:, 1]) / 2.0
    half_range = (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0
    qpos = np.asarray(unwrapped.data.qpos, dtype=float)
    action = np.zeros(unwrapped.model.nu, dtype=float)
    for actuator_idx in range(unwrapped.model.nu):
        joint_idx = int(unwrapped.model.actuator_trnid[actuator_idx, 0])
        qpos_idx = int(unwrapped.model.jnt_qposadr[joint_idx])
        action[actuator_idx] = (qpos[qpos_idx] - center[actuator_idx]) / half_range[actuator_idx]
    return np.clip(action, -1.0, 1.0)


def active_fingers_and_dims(obs: dict[str, np.ndarray], threshold: float) -> tuple[list[int], list[int]]:
    per_finger = np.linalg.norm(
        (np.asarray(obs["desired_goal"], dtype=float) - np.asarray(obs["achieved_goal"], dtype=float)).reshape(5, 3),
        axis=1,
    )
    active_fingers = [idx for idx, distance in enumerate(per_finger) if float(distance) > float(threshold)]
    dims = [0, 1]
    for finger_idx in active_fingers:
        dims.extend(ACTION_DIMS_BY_FINGER[finger_idx])
    if any(finger_idx != 4 for finger_idx in active_fingers):
        dims.extend(ACTION_DIMS_BY_FINGER[4])
    return active_fingers, sorted(set(dims))


def save_state(env) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    unwrapped = env.unwrapped
    return (
        np.asarray(unwrapped.data.qpos, dtype=float).copy(),
        np.asarray(unwrapped.data.qvel, dtype=float).copy(),
        np.asarray(unwrapped.goal, dtype=float).copy(),
        np.asarray(unwrapped.data.ctrl, dtype=float).copy(),
        float(unwrapped.data.time),
    )


def restore_state(env, state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]) -> None:
    qpos, qvel, goal, ctrl, time = state
    unwrapped = env.unwrapped
    unwrapped.goal = goal.copy()
    unwrapped.data.qpos[:] = qpos
    unwrapped.data.qvel[:] = qvel
    unwrapped.data.ctrl[:] = ctrl
    unwrapped.data.time = time
    unwrapped._mujoco.mj_forward(unwrapped.model, unwrapped.data)


def perturb_hand_state(env, sigma: float, rng: np.random.Generator) -> None:
    if float(sigma) <= 0.0:
        return
    unwrapped = env.unwrapped
    direction = rng.normal(size=len(PERTURBED_QPOS))
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction = np.ones_like(direction)
        norm = float(np.linalg.norm(direction))
    qpos = np.asarray(unwrapped.data.qpos, dtype=float).copy()
    qvel = np.asarray(unwrapped.data.qvel, dtype=float).copy()
    qpos[PERTURBED_QPOS] += float(sigma) * direction / norm
    for qpos_idx in PERTURBED_QPOS:
        qpos[qpos_idx] = np.clip(qpos[qpos_idx], unwrapped.model.jnt_range[qpos_idx, 0], unwrapped.model.jnt_range[qpos_idx, 1])
    qvel[:] = 0.0
    unwrapped.data.qpos[:] = qpos
    unwrapped.data.qvel[:] = qvel
    unwrapped._mujoco.mj_forward(unwrapped.model, unwrapped.data)


def evaluate_constant_action(env, state, action: np.ndarray, horizon: int) -> float:
    restore_state(env, state)
    best_distance = observation_distance(env.unwrapped._get_obs())
    for _step in range(horizon):
        obs, _reward, _terminated, truncated, _info = env.step(action.astype(np.float32))
        best_distance = min(best_distance, observation_distance(obs))
        if truncated:
            break
    return float(best_distance)


def cem_constant_action(env, rng: np.random.Generator, args: argparse.Namespace) -> tuple[bool, float, float, list[int], list[int]]:
    state = save_state(env)
    obs = env.unwrapped._get_obs()
    initial_distance = observation_distance(obs)
    active_fingers, active_dims = active_fingers_and_dims(obs, threshold=args.active_finger_threshold)
    mean = action_from_qpos(env)
    std = np.ones_like(mean) * float(args.initial_action_std)
    frozen = mean.copy()
    best_action = mean.copy()
    best_distance = evaluate_constant_action(env, state, best_action, args.horizon)

    for _iter_idx in range(args.cem_iters):
        scored: list[tuple[float, np.ndarray]] = []
        for _sample_idx in range(args.population):
            action = frozen.copy()
            action[active_dims] = np.clip(rng.normal(mean[active_dims], std[active_dims]), -1.0, 1.0)
            scored.append((evaluate_constant_action(env, state, action, args.horizon), action))
        scored.sort(key=lambda item: item[0])
        elites = np.array([action for _score, action in scored[: args.elite]], dtype=float)
        if scored[0][0] < best_distance:
            best_distance = float(scored[0][0])
            best_action = scored[0][1].copy()
        mean[active_dims] = np.mean(elites[:, active_dims], axis=0)
        std[active_dims] = np.maximum(float(args.min_action_std), np.std(elites[:, active_dims], axis=0))

    best_distance = evaluate_constant_action(env, state, best_action, args.horizon)
    success = best_distance <= float(env.unwrapped.distance_threshold)
    return bool(success), float(best_distance), float(initial_distance), active_fingers, active_dims


def summarize(rows: list[CalibrationRow], radii: np.ndarray, *, alpha: float) -> dict[str, float | int | str | list[float]]:
    curve = []
    for sigma in radii:
        radius_rows = [row for row in rows if abs(row.sigma - float(sigma)) < 1e-12]
        if not radius_rows:
            raise ValueError(f"missing rows for sigma={sigma}")
        curve.append(float(np.mean([row.success for row in radius_rows])))
    curve_array = np.array(curve, dtype=float)
    clean_success = float(curve_array[0])
    min_recovery = float(np.min(curve_array))
    max_recovery = float(np.max(curve_array))
    recovery_range = max_recovery - min_recovery
    if clean_success < 0.80:
        decision = "reject-calibration-low-clean-success"
    elif recovery_range < 0.20:
        decision = "reject-calibration-flat-recovery"
    else:
        decision = "usable-calibration"
    return {
        "env_id": "HandReach-v3",
        "seeds": len({row.seed for row in rows}),
        "radii": [float(sigma) for sigma in radii],
        "clean_success": clean_success,
        "min_recovery": min_recovery,
        "max_recovery": max_recovery,
        "recovery_range": float(recovery_range),
        "rauc": float(recovery_auc(radii, curve_array, sigma_max=float(radii[-1]))),
        "r80": float(critical_radius(radii, curve_array, alpha=alpha)),
        "decision": decision,
    }


def calibrate(args: argparse.Namespace) -> tuple[list[CalibrationRow], dict[str, float | int | str | list[float]]]:
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    radii = parse_radii(args.radii)
    rows: list[CalibrationRow] = []
    env = gym.make(args.env_id)
    try:
        for seed in range(args.seeds):
            for sigma in radii:
                rng = np.random.default_rng(args.seed_offset + seed * 10_000 + int(round(float(sigma) * 1_000)))
                env.reset(seed=args.seed_offset + seed)
                perturb_hand_state(env, float(sigma), rng)
                success, best_distance, initial_distance, active_fingers, active_dims = cem_constant_action(env, rng, args)
                rows.append(
                    CalibrationRow(
                        seed=seed,
                        sigma=float(sigma),
                        success=int(success),
                        best_distance=float(best_distance),
                        initial_distance=float(initial_distance),
                        active_fingers=",".join(str(idx) for idx in active_fingers),
                        active_action_dims=",".join(str(idx) for idx in active_dims),
                    )
                )
    finally:
        env.close()
    summary = summarize(rows, radii, alpha=args.alpha)
    summary["env_id"] = args.env_id
    summary["horizon"] = args.horizon
    summary["population"] = args.population
    summary["elite"] = args.elite
    summary["cem_iters"] = args.cem_iters
    summary["initial_action_std"] = args.initial_action_std
    summary["min_action_std"] = args.min_action_std
    summary["active_finger_threshold"] = args.active_finger_threshold
    summary["seed_offset"] = args.seed_offset
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
    parser.add_argument("--env-id", default="HandReach-v3")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--radii", default="0.00,0.05,0.10,0.15,0.20")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elite", type=int, default=8)
    parser.add_argument("--cem-iters", type=int, default=4)
    parser.add_argument("--initial-action-std", type=float, default=0.90)
    parser.add_argument("--min-action-std", type=float, default=0.08)
    parser.add_argument("--active-finger-threshold", type=float, default=0.015)
    parser.add_argument("--alpha", type=float, default=0.80)
    parser.add_argument("--seed-offset", type=int, default=171_000)
    args = parser.parse_args()
    if args.elite <= 0 or args.elite > args.population:
        raise ValueError("--elite must be in [1, --population]")

    rows, summary = calibrate(args)
    write_outputs(args.out, rows, summary)
    print(
        "HandReach calibration: "
        f"env={summary['env_id']} clean={summary['clean_success']:.4f} "
        f"rauc={summary['rauc']:.4f} r80={summary['r80']:.4f} "
        f"min={summary['min_recovery']:.4f} max={summary['max_recovery']:.4f} "
        f"decision={summary['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
