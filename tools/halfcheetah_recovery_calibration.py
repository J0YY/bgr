#!/usr/bin/env python3
"""Pre-method recovery calibration for Gymnasium MuJoCo HalfCheetah-v5."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from bgr.metrics import critical_radius, recovery_auc


@dataclass(frozen=True, slots=True)
class HalfCheetahCheckpoint:
    seed: int
    qpos: list[float]
    qvel: list[float]
    phase: int


@dataclass(frozen=True, slots=True)
class HalfCheetahCalibrationRow:
    seed: int
    radius: float
    trial: int
    success: int
    progress: float
    start_x: float
    end_x: float
    terminated: int


def parse_radii(value: str) -> np.ndarray:
    radii = np.array([float(item.strip()) for item in value.split(",") if item.strip()], dtype=float)
    if radii.ndim != 1 or len(radii) < 2:
        raise ValueError("--radii must contain at least two comma-separated values")
    if not np.all(np.diff(radii) > 0.0):
        raise ValueError("--radii must be strictly increasing")
    return radii


def package_versions(env_id: str) -> dict[str, str]:
    import gymnasium
    import mujoco

    versions = {
        "env_id": env_id,
        "gymnasium": gymnasium.__version__,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }
    for package in ("gymnasium", "mujoco"):
        try:
            versions[f"{package}-package"] = version(package)
        except PackageNotFoundError:
            pass
    return versions


def sinusoidal_action(step: int, *, amplitude: float, period: float, action_dim: int) -> np.ndarray:
    phase = 2.0 * math.pi * float(step) / float(period)
    offsets = np.arange(action_dim, dtype=float)
    return np.clip(float(amplitude) * np.sin(phase + offsets), -1.0, 1.0)


def checkpoint(env, *, seed: int, burn_in: int, amplitude: float, period: float) -> HalfCheetahCheckpoint:
    env.reset(seed=int(seed))
    action_dim = int(np.prod(env.action_space.shape))
    phase = 0
    for phase in range(int(burn_in)):
        _obs, _reward, terminated, truncated, _info = env.step(
            sinusoidal_action(phase, amplitude=amplitude, period=period, action_dim=action_dim)
        )
        if terminated or truncated:
            break
    return HalfCheetahCheckpoint(
        seed=int(seed),
        qpos=[float(x) for x in env.unwrapped.data.qpos.copy()],
        qvel=[float(x) for x in env.unwrapped.data.qvel.copy()],
        phase=int(phase) + 1,
    )


def normalized_direction(seed: int, radius_index: int, trial: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(991_000 + 10_003 * int(seed) + 997 * int(radius_index) + int(trial))
    direction = rng.normal(0.0, 1.0, size=int(dim))
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction[0] = 1.0
        return direction
    return direction / norm


def perturb_state(
    item: HalfCheetahCheckpoint,
    *,
    radius: float,
    radius_index: int,
    trial: int,
    position_scale: float,
    velocity_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.array(item.qpos, dtype=float)
    qvel = np.array(item.qvel, dtype=float)
    direction = normalized_direction(item.seed, radius_index, trial, len(qpos) + len(qvel))
    qpos_direction = direction[: len(qpos)]
    qvel_direction = direction[len(qpos) :]
    perturbed_qpos = qpos.copy()
    perturbed_qvel = qvel.copy()
    perturbed_qpos[1:] += float(radius) * float(position_scale) * qpos_direction[1:]
    perturbed_qvel += float(radius) * float(velocity_scale) * qvel_direction
    return perturbed_qpos, perturbed_qvel


def rollout(
    env,
    item: HalfCheetahCheckpoint,
    *,
    radius: float,
    radius_index: int,
    trial: int,
    args: argparse.Namespace,
) -> HalfCheetahCalibrationRow:
    qpos, qvel = perturb_state(
        item,
        radius=radius,
        radius_index=radius_index,
        trial=trial,
        position_scale=args.position_scale,
        velocity_scale=args.velocity_scale,
    )
    env.unwrapped.set_state(qpos, qvel)
    action_dim = int(np.prod(env.action_space.shape))
    start_x = float(env.unwrapped.data.qpos[0])
    terminated_any = False
    for step in range(int(args.horizon)):
        _obs, _reward, terminated, truncated, _info = env.step(
            sinusoidal_action(
                item.phase + step,
                amplitude=args.controller_amplitude,
                period=args.controller_period,
                action_dim=action_dim,
            )
        )
        terminated_any = terminated_any or bool(terminated or truncated)
        if terminated or truncated:
            break
    end_x = float(env.unwrapped.data.qpos[0])
    progress = end_x - start_x
    success = (not terminated_any) and progress >= float(args.min_progress)
    return HalfCheetahCalibrationRow(
        seed=int(item.seed),
        radius=float(radius),
        trial=int(trial),
        success=int(success),
        progress=float(progress),
        start_x=start_x,
        end_x=end_x,
        terminated=int(terminated_any),
    )


def summarize(rows: list[HalfCheetahCalibrationRow], radii: np.ndarray, *, alpha: float) -> dict[str, object]:
    curve = []
    progress_curve = []
    termination_curve = []
    for radius in radii:
        radius_rows = [row for row in rows if abs(row.radius - float(radius)) < 1e-12]
        if not radius_rows:
            raise ValueError(f"missing rows for radius {radius}")
        curve.append(float(np.mean([row.success for row in radius_rows])))
        progress_curve.append(float(np.mean([row.progress for row in radius_rows])))
        termination_curve.append(float(np.mean([row.terminated for row in radius_rows])))
    curve_array = np.array(curve, dtype=float)
    clean_success = float(curve_array[0])
    min_recovery = float(np.min(curve_array))
    max_recovery = float(np.max(curve_array))
    r80 = float(critical_radius(radii, curve_array, alpha=alpha))
    rauc = float(recovery_auc(radii, curve_array))
    if clean_success < 0.80:
        decision = "reject-calibration-low-clean-success"
    elif max_recovery - min_recovery < 0.20:
        decision = "reject-calibration-flat-recovery"
    elif r80 >= float(radii[-1]):
        decision = "reject-calibration-radius-saturated"
    elif r80 <= float(radii[0]) and min_recovery <= 0.01:
        decision = "reject-calibration-radius-floor"
    else:
        decision = "usable-calibration"
    return {
        "env_id": "HalfCheetah-v5",
        "rows": len(rows),
        "radii": [float(radius) for radius in radii],
        "curve": curve,
        "progress_curve": progress_curve,
        "termination_curve": termination_curve,
        "clean_success": clean_success,
        "min_recovery": min_recovery,
        "max_recovery": max_recovery,
        "r80": r80,
        "rauc": rauc,
        "decision": decision,
    }


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    radii = parse_radii(args.radii)
    rows: list[HalfCheetahCalibrationRow] = []
    for seed in range(int(args.seeds)):
        env = gym.make(args.env_id)
        item = checkpoint(
            env,
            seed=seed,
            burn_in=args.burn_in,
            amplitude=args.controller_amplitude,
            period=args.controller_period,
        )
        for radius_index, radius in enumerate(radii):
            for trial in range(int(args.trials)):
                rows.append(
                    rollout(
                        env,
                        item,
                        radius=float(radius),
                        radius_index=radius_index,
                        trial=trial,
                        args=args,
                    )
                )
        env.close()

    summary = summarize(rows, radii, alpha=args.alpha)
    summary.update(
        {
            "seeds": int(args.seeds),
            "trials": int(args.trials),
            "burn_in": int(args.burn_in),
            "horizon": int(args.horizon),
            "min_progress": float(args.min_progress),
            "position_scale": float(args.position_scale),
            "velocity_scale": float(args.velocity_scale),
            "controller_amplitude": float(args.controller_amplitude),
            "controller_period": float(args.controller_period),
        }
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "package_versions.json").write_text(
        json.dumps(package_versions(args.env_id), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out / "recovery_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--env-id", default="HalfCheetah-v5")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--radii", default="0,0.25,0.5,0.8,1.1,1.5,2.0,2.5")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--burn-in", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--min-progress", type=float, default=1.0)
    parser.add_argument("--position-scale", type=float, default=0.35)
    parser.add_argument("--velocity-scale", type=float, default=1.00)
    parser.add_argument("--controller-amplitude", type=float, default=0.80)
    parser.add_argument("--controller-period", type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=0.80)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
