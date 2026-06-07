#!/usr/bin/env python3
"""Calibrate package-backed Gymnasium MuJoCo InvertedPendulum recovery curves.

This is a pre-comparison diagnostic for an independent benchmark route. It uses
Gymnasium's own InvertedPendulum-v5 MuJoCo dynamics, exact simulator state
resets, and pole-angle perturbations before running a fixed PD controller. It
does not compare BGR methods.
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
    direction: int
    success: int
    steps: int
    max_abs_angle: float
    final_abs_angle: float


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
            "InvertedPendulum recovery calibration requires Gymnasium with MuJoCo in an isolated environment, "
            "for example /tmp/bgr_pointmaze_venv."
        ) from exc
    return {
        "gymnasium": gymnasium.__version__,
        "gymnasium-package": version("gymnasium"),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }


def reset_perturbed(env, *, seed: int, sigma: float, direction: int) -> None:
    env.reset(seed=seed)
    unwrapped = env.unwrapped
    qpos = np.array(unwrapped.data.qpos.copy(), dtype=float)
    qvel = np.array(unwrapped.data.qvel.copy(), dtype=float)
    qpos[1] = qpos[1] + int(direction) * float(sigma)
    qvel[:] = 0.0
    unwrapped.set_state(qpos, qvel)


def controller_action(env, *, kp: float, kd: float, cart_kp: float, cart_kd: float) -> np.ndarray:
    unwrapped = env.unwrapped
    qpos = np.array(unwrapped.data.qpos.copy(), dtype=float)
    qvel = np.array(unwrapped.data.qvel.copy(), dtype=float)
    force = kp * qpos[1] + kd * qvel[1] + cart_kp * qpos[0] + cart_kd * qvel[0]
    return np.array([np.clip(force, env.action_space.low[0], env.action_space.high[0])], dtype=np.float32)


def rollout(
    env,
    *,
    seed: int,
    sigma: float,
    direction: int,
    horizon: int,
    kp: float,
    kd: float,
    cart_kp: float,
    cart_kd: float,
) -> CalibrationRow:
    reset_perturbed(env, seed=seed, sigma=sigma, direction=direction)
    terminated = False
    step = 0
    max_abs_angle = abs(float(env.unwrapped.data.qpos[1]))
    for step in range(1, horizon + 1):
        _obs, _reward, terminated, truncated, _info = env.step(
            controller_action(env, kp=kp, kd=kd, cart_kp=cart_kp, cart_kd=cart_kd)
        )
        max_abs_angle = max(max_abs_angle, abs(float(env.unwrapped.data.qpos[1])))
        if terminated or truncated:
            break
    return CalibrationRow(
        seed=seed,
        sigma=float(sigma),
        direction=int(direction),
        success=int((not terminated) and step >= horizon),
        steps=int(step),
        max_abs_angle=float(max_abs_angle),
        final_abs_angle=float(abs(env.unwrapped.data.qpos[1])),
    )


def summarize(rows: list[CalibrationRow], radii: np.ndarray, *, alpha: float) -> dict[str, float | int | str | list[float]]:
    curve = []
    mean_max_abs_angles = []
    for sigma in radii:
        radius_rows = [row for row in rows if abs(row.sigma - float(sigma)) < 1e-12]
        if not radius_rows:
            raise ValueError(f"missing rows for sigma={sigma}")
        curve.append(float(np.mean([row.success for row in radius_rows])))
        mean_max_abs_angles.append(float(np.mean([row.max_abs_angle for row in radius_rows])))
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
        "env_id": "InvertedPendulum-v5",
        "seeds": len({row.seed for row in rows}),
        "radii": [float(sigma) for sigma in radii],
        "clean_success": clean_success,
        "min_recovery": min_recovery,
        "max_recovery": max_recovery,
        "recovery_range": float(recovery_range),
        "mean_max_abs_angle": float(np.mean(mean_max_abs_angles)),
        "rauc": float(recovery_auc(radii, curve_array, sigma_max=float(radii[-1]))),
        "r80": r80,
        "decision": decision,
    }


def calibrate(args: argparse.Namespace) -> tuple[list[CalibrationRow], dict[str, float | int | str | list[float]]]:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit(
            "InvertedPendulum recovery calibration requires Gymnasium with MuJoCo in an isolated environment, "
            "for example /tmp/bgr_pointmaze_venv."
        ) from exc

    radii = parse_radii(args.radii)
    env = gym.make(args.env_id)
    rows: list[CalibrationRow] = []
    try:
        for seed in range(args.seeds):
            direction = -1 if seed % 2 else 1
            for sigma in radii:
                rows.append(
                    rollout(
                        env,
                        seed=args.seed_offset + seed,
                        sigma=float(sigma),
                        direction=direction,
                        horizon=args.horizon,
                        kp=args.kp,
                        kd=args.kd,
                        cart_kp=args.cart_kp,
                        cart_kd=args.cart_kd,
                    )
                )
    finally:
        env.close()
    summary = summarize(rows, radii, alpha=args.alpha)
    summary["env_id"] = args.env_id
    summary["horizon"] = args.horizon
    summary["kp"] = args.kp
    summary["kd"] = args.kd
    summary["cart_kp"] = args.cart_kp
    summary["cart_kd"] = args.cart_kd
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
    parser.add_argument("--env-id", default="InvertedPendulum-v5")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--radii", default="0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.20,0.25,0.30")
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--kp", type=float, default=10.0)
    parser.add_argument("--kd", type=float, default=1.0)
    parser.add_argument("--cart-kp", type=float, default=0.4)
    parser.add_argument("--cart-kd", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.80)
    parser.add_argument("--seed-offset", type=int, default=310_000)
    args = parser.parse_args()

    rows, summary = calibrate(args)
    write_outputs(args.out, rows, summary)
    print(
        "InvertedPendulum recovery calibration: "
        f"clean={summary['clean_success']:.4f} "
        f"range={summary['min_recovery']:.4f}--{summary['max_recovery']:.4f} "
        f"r80={summary['r80']:.4f} "
        f"decision={summary['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
