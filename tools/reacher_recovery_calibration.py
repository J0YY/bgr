#!/usr/bin/env python3
"""Calibrate package-backed Gymnasium MuJoCo Reacher recovery curves.

This is a pre-comparison diagnostic for an independent benchmark route. It uses
Gymnasium's own Reacher-v5 MuJoCo dynamics and target sampling, then perturbs
the two arm joint angles before running a fixed inverse-kinematics controller.
It does not compare BGR methods.
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

LINK_1 = 0.10
LINK_2 = 0.11


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    seed: int
    sigma: float
    success: int
    steps: int
    best_distance: float
    final_distance: float


def parse_radii(value: str) -> np.ndarray:
    radii = np.array([float(item.strip()) for item in value.split(",") if item.strip()], dtype=float)
    if radii.ndim != 1 or len(radii) < 2:
        raise ValueError("--radii must contain at least two comma-separated values")
    if not np.all(np.diff(radii) > 0.0):
        raise ValueError("--radii must be strictly increasing")
    return radii


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def ik_solutions(target_xy: np.ndarray) -> list[np.ndarray]:
    x, y = float(target_xy[0]), float(target_xy[1])
    radius_sq = x * x + y * y
    cos_q2 = np.clip((radius_sq - LINK_1**2 - LINK_2**2) / (2 * LINK_1 * LINK_2), -1.0, 1.0)
    sin_q2_abs = float(np.sqrt(max(0.0, 1.0 - cos_q2 * cos_q2)))
    solutions: list[np.ndarray] = []
    for sin_q2 in (sin_q2_abs, -sin_q2_abs):
        q2 = float(np.arctan2(sin_q2, cos_q2))
        q1 = float(np.arctan2(y, x) - np.arctan2(LINK_2 * sin_q2, LINK_1 + LINK_2 * cos_q2))
        solutions.append(np.array([q1, q2], dtype=float))
    return solutions


def target_joint_angles(target_xy: np.ndarray, current_angles: np.ndarray) -> np.ndarray:
    return min(
        ik_solutions(target_xy),
        key=lambda candidate: float(np.linalg.norm(wrap_angle(candidate - current_angles))),
    )


def package_versions() -> dict[str, str]:
    try:
        import gymnasium
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "Reacher recovery calibration requires Gymnasium with MuJoCo in an isolated environment, "
            "for example /tmp/bgr_pointmaze_venv."
        ) from exc
    return {
        "gymnasium": gymnasium.__version__,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "gymnasium-package": version("gymnasium"),
    }


def fingertip_distance(env) -> float:
    unwrapped = env.unwrapped
    fingertip_xy = unwrapped.data.xpos[3, :2]
    target_xy = unwrapped.data.xpos[4, :2]
    return float(np.linalg.norm(fingertip_xy - target_xy))


def reset_perturbed(env, *, seed: int, sigma: float, angle: float) -> None:
    env.reset(seed=seed)
    unwrapped = env.unwrapped
    qpos = np.array(unwrapped.data.qpos.copy(), dtype=float)
    qvel = np.array(unwrapped.data.qvel.copy(), dtype=float)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    qpos[:2] = qpos[:2] + float(sigma) * direction
    qvel[:2] = 0.0
    unwrapped.set_state(qpos, qvel)


def controller_action(env, *, kp: float, kd: float, torque_limit: float) -> np.ndarray:
    unwrapped = env.unwrapped
    qpos = np.array(unwrapped.data.qpos.copy(), dtype=float)
    qvel = np.array(unwrapped.data.qvel.copy(), dtype=float)
    current_angles = qpos[:2]
    target_xy = qpos[2:4]
    target_angles = target_joint_angles(target_xy, current_angles)
    angle_error = wrap_angle(target_angles - current_angles)
    torque = kp * angle_error - kd * qvel[:2]
    return np.clip(torque, -torque_limit, torque_limit).astype(np.float32)


def rollout(
    env,
    *,
    seed: int,
    sigma: float,
    angle: float,
    horizon: int,
    success_threshold: float,
    kp: float,
    kd: float,
    torque_limit: float,
) -> CalibrationRow:
    reset_perturbed(env, seed=seed, sigma=sigma, angle=angle)
    best_distance = fingertip_distance(env)
    final_distance = best_distance
    success = best_distance <= success_threshold
    step = 0
    for step in range(1, horizon + 1):
        _obs, _reward, terminated, truncated, _info = env.step(
            controller_action(env, kp=kp, kd=kd, torque_limit=torque_limit)
        )
        final_distance = fingertip_distance(env)
        best_distance = min(best_distance, final_distance)
        success = success or best_distance <= success_threshold
        if terminated or truncated:
            break
    return CalibrationRow(
        seed=seed,
        sigma=float(sigma),
        success=int(success),
        steps=int(step),
        best_distance=float(best_distance),
        final_distance=float(final_distance),
    )


def summarize(rows: list[CalibrationRow], radii: np.ndarray, *, alpha: float) -> dict[str, float | int | str | list[float]]:
    curve = []
    best_distances = []
    for sigma in radii:
        radius_rows = [row for row in rows if abs(row.sigma - float(sigma)) < 1e-12]
        if not radius_rows:
            raise ValueError(f"missing rows for sigma={sigma}")
        curve.append(float(np.mean([row.success for row in radius_rows])))
        best_distances.append(float(np.mean([row.best_distance for row in radius_rows])))
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
        "env_id": "Reacher-v5",
        "seeds": len({row.seed for row in rows}),
        "radii": [float(sigma) for sigma in radii],
        "clean_success": clean_success,
        "min_recovery": min_recovery,
        "max_recovery": max_recovery,
        "recovery_range": float(recovery_range),
        "mean_best_distance": float(np.mean(best_distances)),
        "rauc": float(recovery_auc(radii, curve_array, sigma_max=float(radii[-1]))),
        "r80": float(critical_radius(radii, curve_array, alpha=alpha)),
        "decision": decision,
    }


def calibrate(args: argparse.Namespace) -> tuple[list[CalibrationRow], dict[str, float | int | str | list[float]]]:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit(
            "Reacher recovery calibration requires Gymnasium with MuJoCo in an isolated environment, "
            "for example /tmp/bgr_pointmaze_venv."
        ) from exc

    radii = parse_radii(args.radii)
    env = gym.make(args.env_id)
    rows: list[CalibrationRow] = []
    try:
        for seed in range(args.seeds):
            angle = (args.angle_stride * seed) % (2 * np.pi)
            for sigma in radii:
                rows.append(
                    rollout(
                        env,
                        seed=seed,
                        sigma=float(sigma),
                        angle=angle,
                        horizon=args.horizon,
                        success_threshold=args.success_threshold,
                        kp=args.kp,
                        kd=args.kd,
                        torque_limit=args.torque_limit,
                    )
                )
    finally:
        env.close()
    summary = summarize(rows, radii, alpha=args.alpha)
    summary["env_id"] = args.env_id
    summary["horizon"] = args.horizon
    summary["success_threshold"] = args.success_threshold
    summary["kp"] = args.kp
    summary["kd"] = args.kd
    summary["torque_limit"] = args.torque_limit
    summary["angle_stride"] = args.angle_stride
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
    parser.add_argument("--env-id", default="Reacher-v5")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--radii", default="0,0.25,0.5,0.75,1,1.5,2,2.5,3,3.5,4")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--success-threshold", type=float, default=0.025)
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--kd", type=float, default=0.2)
    parser.add_argument("--torque-limit", type=float, default=0.4)
    parser.add_argument("--angle-stride", type=float, default=1.618)
    parser.add_argument("--alpha", type=float, default=0.80)
    args = parser.parse_args()

    rows, summary = calibrate(args)
    write_outputs(args.out, rows, summary)
    print(
        "Reacher recovery calibration: "
        f"clean={summary['clean_success']:.4f} "
        f"range={summary['min_recovery']:.4f}--{summary['max_recovery']:.4f} "
        f"r80={summary['r80']:.4f} "
        f"decision={summary['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
