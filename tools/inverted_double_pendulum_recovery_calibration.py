#!/usr/bin/env python3
"""Calibrate Gymnasium MuJoCo InvertedDoublePendulum recovery curves.

This is a pre-comparison diagnostic for an independent benchmark route. It uses
Gymnasium's InvertedDoublePendulum-v5 dynamics, exact simulator state resets,
and two-pole angular perturbations before running a fixed finite-difference LQR
controller. It does not compare BGR methods.
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


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    seed: int
    sigma: float
    angle: float
    success: int
    steps: int
    max_abs_pole_angle: float
    final_abs_pole_angle: float


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
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "InvertedDoublePendulum recovery calibration requires Gymnasium with MuJoCo in an isolated environment, "
            "for example /tmp/bgr_pointmaze_venv."
        ) from exc
    return {
        "gymnasium": gymnasium.__version__,
        "gymnasium-package": version("gymnasium"),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }


def state_vector(env) -> np.ndarray:
    data = env.unwrapped.data
    return np.concatenate([np.array(data.qpos.copy(), dtype=float), np.array(data.qvel.copy(), dtype=float)])


def set_state_vector(env, x: np.ndarray) -> None:
    env.unwrapped.set_state(np.asarray(x[:3], dtype=float), np.asarray(x[3:], dtype=float))


def linearize(env, *, eps: float) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=123)
    x0 = np.zeros(6, dtype=float)
    u0 = np.zeros(1, dtype=float)
    set_state_vector(env, x0)

    def step_from(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        set_state_vector(env, x)
        env.step(np.array(u, dtype=np.float32))
        return state_vector(env)

    a_mat = np.zeros((6, 6), dtype=float)
    b_mat = np.zeros((6, 1), dtype=float)
    for idx in range(6):
        dx = np.zeros(6, dtype=float)
        dx[idx] = eps
        a_mat[:, idx] = (step_from(x0 + dx, u0) - step_from(x0 - dx, u0)) / (2.0 * eps)
    du = np.array([eps], dtype=float)
    b_mat[:, 0] = (step_from(x0, u0 + du) - step_from(x0, u0 - du)) / (2.0 * eps)
    return a_mat, b_mat


def solve_dare(a_mat: np.ndarray, b_mat: np.ndarray, q_mat: np.ndarray, r_mat: np.ndarray) -> np.ndarray:
    p_mat = q_mat.copy()
    for _ in range(5000):
        gain = np.linalg.solve(r_mat + b_mat.T @ p_mat @ b_mat, b_mat.T @ p_mat @ a_mat)
        next_p = q_mat + a_mat.T @ p_mat @ (a_mat - b_mat @ gain)
        if np.max(np.abs(next_p - p_mat)) < 1e-10:
            p_mat = next_p
            break
        p_mat = next_p
    return np.linalg.solve(r_mat + b_mat.T @ p_mat @ b_mat, b_mat.T @ p_mat @ a_mat)


def lqr_gain(env, args: argparse.Namespace) -> np.ndarray:
    a_mat, b_mat = linearize(env, eps=args.linearization_eps)
    q_mat = np.diag(
        [
            args.q_cart,
            args.q_pole1,
            args.q_pole2,
            args.q_cart_vel,
            args.q_pole1_vel,
            args.q_pole2_vel,
        ]
    )
    r_mat = np.array([[args.r_action]], dtype=float)
    return solve_dare(a_mat, b_mat, q_mat, r_mat)


def perturb_angle(seed_idx: int, trial_idx: int) -> float:
    return float((0.61803398875 * (seed_idx + 1) + 1.713 * trial_idx) % (2.0 * np.pi))


def reset_perturbed(env, *, seed: int, sigma: float, angle: float) -> None:
    env.reset(seed=int(seed))
    qpos = np.array(env.unwrapped.data.qpos.copy(), dtype=float)
    qvel = np.array(env.unwrapped.data.qvel.copy(), dtype=float)
    qpos[1] = qpos[1] + float(sigma) * float(np.cos(angle))
    qpos[2] = qpos[2] + float(sigma) * float(np.sin(angle))
    qvel[:] = 0.0
    env.unwrapped.set_state(qpos, qvel)


def controller_action(env, gain: np.ndarray) -> np.ndarray:
    force = float(np.squeeze(-gain @ state_vector(env).reshape(-1, 1)))
    force = float(np.clip(force, env.action_space.low[0], env.action_space.high[0]))
    return np.array([force], dtype=np.float32)


def rollout(
    env,
    *,
    seed: int,
    sigma: float,
    angle: float,
    horizon: int,
    gain: np.ndarray,
) -> CalibrationRow:
    reset_perturbed(env, seed=seed, sigma=sigma, angle=angle)
    terminated = False
    step = 0
    max_abs_angle = float(np.max(np.abs(env.unwrapped.data.qpos[1:3])))
    for step in range(1, horizon + 1):
        _obs, _reward, terminated, truncated, _info = env.step(controller_action(env, gain))
        max_abs_angle = max(max_abs_angle, float(np.max(np.abs(env.unwrapped.data.qpos[1:3]))))
        if terminated or truncated:
            break
    return CalibrationRow(
        seed=int(seed),
        sigma=float(sigma),
        angle=float(angle),
        success=int((not terminated) and step >= horizon),
        steps=int(step),
        max_abs_pole_angle=max_abs_angle,
        final_abs_pole_angle=float(np.max(np.abs(env.unwrapped.data.qpos[1:3]))),
    )


def summarize(rows: list[CalibrationRow], radii: np.ndarray, *, alpha: float) -> dict[str, float | int | str | list[float]]:
    curve = []
    mean_max_abs_angles = []
    for sigma in radii:
        radius_rows = [row for row in rows if abs(row.sigma - float(sigma)) < 1e-12]
        if not radius_rows:
            raise ValueError(f"missing rows for sigma={sigma}")
        curve.append(float(np.mean([row.success for row in radius_rows])))
        mean_max_abs_angles.append(float(np.mean([row.max_abs_pole_angle for row in radius_rows])))
    curve_array = np.array(curve, dtype=float)
    clean_success = float(curve_array[0])
    min_recovery = float(np.min(curve_array))
    max_recovery = float(np.max(curve_array))
    recovery_range = max_recovery - min_recovery
    r80 = float(critical_radius(radii, curve_array, alpha=alpha))
    if clean_success < 0.80:
        decision = "reject-calibration-low-clean-success"
    elif recovery_range < 0.20:
        decision = "reject-calibration-flat-recovery"
    elif r80 >= float(radii[-1]):
        decision = "reject-calibration-radius-saturated"
    else:
        decision = "usable-calibration"
    return {
        "env_id": "InvertedDoublePendulum-v5",
        "rows": len(rows),
        "radii": [float(sigma) for sigma in radii],
        "clean_success": clean_success,
        "min_recovery": min_recovery,
        "max_recovery": max_recovery,
        "recovery_range": float(recovery_range),
        "mean_max_abs_pole_angle": float(np.mean(mean_max_abs_angles)),
        "rauc": float(recovery_auc(radii, curve_array, sigma_max=float(radii[-1]))),
        "r80": r80,
        "decision": decision,
        "mean_curve": [float(value) for value in curve_array],
    }


def calibrate(args: argparse.Namespace) -> tuple[list[CalibrationRow], dict[str, float | int | str | list[float]]]:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit(
            "InvertedDoublePendulum recovery calibration requires Gymnasium with MuJoCo in an isolated environment, "
            "for example /tmp/bgr_pointmaze_venv."
        ) from exc

    radii = parse_radii(args.radii)
    env = gym.make(args.env_id)
    rows: list[CalibrationRow] = []
    try:
        gain = lqr_gain(env, args)
        for seed_idx in range(args.seeds):
            for sigma in radii:
                for trial_idx in range(args.trials):
                    rows.append(
                        rollout(
                            env,
                            seed=args.seed_offset + seed_idx,
                            sigma=float(sigma),
                            angle=perturb_angle(seed_idx, trial_idx),
                            horizon=args.horizon,
                            gain=gain,
                        )
                    )
    finally:
        env.close()
    summary = summarize(rows, radii, alpha=args.alpha)
    summary.update(
        {
            "env_id": args.env_id,
            "seeds": args.seeds,
            "trials": args.trials,
            "horizon": args.horizon,
            "seed_offset": args.seed_offset,
            "linearization_eps": args.linearization_eps,
            "q_cart": args.q_cart,
            "q_pole1": args.q_pole1,
            "q_pole2": args.q_pole2,
            "q_cart_vel": args.q_cart_vel,
            "q_pole1_vel": args.q_pole1_vel,
            "q_pole2_vel": args.q_pole2_vel,
            "r_action": args.r_action,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--env-id", default="InvertedDoublePendulum-v5")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--radii", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=0.80)
    parser.add_argument("--seed-offset", type=int, default=420_000)
    parser.add_argument("--linearization-eps", type=float, default=1e-5)
    parser.add_argument("--q-cart", type=float, default=1.0)
    parser.add_argument("--q-pole1", type=float, default=40.0)
    parser.add_argument("--q-pole2", type=float, default=48.0)
    parser.add_argument("--q-cart-vel", type=float, default=0.2)
    parser.add_argument("--q-pole1-vel", type=float, default=1.5)
    parser.add_argument("--q-pole2-vel", type=float, default=1.5)
    parser.add_argument("--r-action", type=float, default=0.02)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows, summary = calibrate(args)
    write_outputs(args.out, rows, summary)
    print(
        "InvertedDoublePendulum recovery calibration: "
        f"clean={summary['clean_success']:.4f} "
        f"range={summary['min_recovery']:.4f}--{summary['max_recovery']:.4f} "
        f"r80={summary['r80']:.4f} "
        f"decision={summary['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
