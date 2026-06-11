#!/usr/bin/env python3
"""Calibrate Gymnasium Box2D LunarLander recovery curves.

This is a pre-comparison diagnostic for an independent benchmark route. It uses
Gymnasium's package-owned LunarLander-v3 dynamics and heuristic controller,
then perturbs exact Box2D body state at a fixed descent checkpoint. It does not
compare BGR methods.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from bgr.metrics import critical_radius, recovery_auc


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    seed: int
    trial: int
    sigma: float
    success: int
    crashed: int
    steps: int
    total_reward: float
    checkpoint_x: float
    checkpoint_y: float


def parse_radii(value: str) -> np.ndarray:
    radii = np.array([float(item.strip()) for item in value.split(",") if item.strip()], dtype=float)
    if radii.ndim != 1 or len(radii) < 2:
        raise ValueError("--radii must contain at least two comma-separated values")
    if not np.all(np.diff(radii) > 0.0):
        raise ValueError("--radii must be strictly increasing")
    return radii


def package_versions(env_id: str = "LunarLander-v3", *, continuous: bool = False) -> dict[str, str | bool]:
    try:
        import gymnasium
    except ImportError as exc:
        raise SystemExit(
            "LunarLander recovery calibration requires Gymnasium with Box2D in an isolated environment, "
            "for example /tmp/bgr_lunar_venv."
        ) from exc
    try:
        pygame_version = version("pygame-ce")
    except PackageNotFoundError:
        pygame_version = version("pygame")
    return {
        "env_id": env_id,
        "continuous": bool(continuous),
        "gymnasium": gymnasium.__version__,
        "gymnasium-package": version("gymnasium"),
        "box2d": version("box2d"),
        "numpy": np.__version__,
        "pygame-ce": pygame_version,
        "swig": version("swig"),
    }


def step_with_heuristic(env, obs: np.ndarray) -> tuple[np.ndarray, float, bool, bool]:
    from gymnasium.envs.box2d.lunar_lander import heuristic

    next_obs, reward, terminated, truncated, _info = env.step(heuristic(env, obs))
    return np.asarray(next_obs, dtype=float), float(reward), bool(terminated), bool(truncated)


def body_state(body) -> dict[str, float]:
    return {
        "x": float(body.position.x),
        "y": float(body.position.y),
        "vx": float(body.linearVelocity.x),
        "vy": float(body.linearVelocity.y),
        "angle": float(body.angle),
        "angular_velocity": float(body.angularVelocity),
    }


def set_body_state(body, state: dict[str, float]) -> None:
    body.position = (state["x"], state["y"])
    body.linearVelocity = (state["vx"], state["vy"])
    body.angle = state["angle"]
    body.angularVelocity = state["angular_velocity"]
    body.awake = True


def capture_state(env) -> tuple[dict[str, float], list[dict[str, float]], np.ndarray]:
    unwrapped = env.unwrapped
    lander_state = body_state(unwrapped.lander)
    leg_states = [body_state(leg) for leg in unwrapped.legs]
    obs = obs_from_state(env, lander_state)
    return lander_state, leg_states, obs


def obs_from_state(env, lander_state: dict[str, float]) -> np.ndarray:
    from gymnasium.envs.box2d.lunar_lander import FPS, LEG_DOWN, SCALE, VIEWPORT_H, VIEWPORT_W

    unwrapped = env.unwrapped
    world_x_scale = VIEWPORT_W / SCALE / 2.0
    world_y_scale = VIEWPORT_H / SCALE / 2.0
    return np.array(
        [
            (lander_state["x"] - world_x_scale) / world_x_scale,
            (lander_state["y"] - (unwrapped.helipad_y + LEG_DOWN / SCALE)) / world_y_scale,
            lander_state["vx"] * world_x_scale / FPS,
            lander_state["vy"] * world_y_scale / FPS,
            lander_state["angle"],
            20.0 * lander_state["angular_velocity"] / FPS,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def perturb_direction(seed: int, trial: int) -> np.ndarray:
    rng = np.random.default_rng(seed * 10_003 + trial * 997 + 81_337)
    direction = rng.normal(0.0, 1.0, size=6)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction[0] = 1.0
        norm = 1.0
    return direction / norm


def restore_perturbed(
    env,
    *,
    lander_state: dict[str, float],
    leg_states: list[dict[str, float]],
    sigma: float,
    direction: np.ndarray,
) -> np.ndarray:
    from gymnasium.envs.box2d.lunar_lander import FPS, SCALE, VIEWPORT_H, VIEWPORT_W

    obs_scales = np.array([0.42, 0.35, 0.85, 0.85, 0.32, 0.75], dtype=float)
    delta_obs = float(sigma) * obs_scales * direction
    world_x_scale = VIEWPORT_W / SCALE / 2.0
    world_y_scale = VIEWPORT_H / SCALE / 2.0
    next_lander = dict(lander_state)
    next_lander["x"] += float(delta_obs[0] * world_x_scale)
    next_lander["y"] += float(delta_obs[1] * world_y_scale)
    next_lander["vx"] += float(delta_obs[2] * FPS / world_x_scale)
    next_lander["vy"] += float(delta_obs[3] * FPS / world_y_scale)
    next_lander["angle"] += float(delta_obs[4])
    next_lander["angular_velocity"] += float(delta_obs[5] * FPS / 20.0)

    dx = next_lander["x"] - lander_state["x"]
    dy = next_lander["y"] - lander_state["y"]
    dvx = next_lander["vx"] - lander_state["vx"]
    dvy = next_lander["vy"] - lander_state["vy"]
    dangle = next_lander["angle"] - lander_state["angle"]
    dangular_velocity = next_lander["angular_velocity"] - lander_state["angular_velocity"]

    set_body_state(env.unwrapped.lander, next_lander)
    for leg, state in zip(env.unwrapped.legs, leg_states, strict=True):
        next_leg = dict(state)
        next_leg["x"] += dx
        next_leg["y"] += dy
        next_leg["vx"] += dvx
        next_leg["vy"] += dvy
        next_leg["angle"] += dangle
        next_leg["angular_velocity"] += dangular_velocity
        set_body_state(leg, next_leg)
        leg.ground_contact = False
    env.unwrapped.game_over = False
    env.unwrapped.prev_shaping = None
    return obs_from_state(env, next_lander)


def checkpoint(env, *, seed: int, burn_in: int) -> tuple[dict[str, float], list[dict[str, float]], np.ndarray]:
    obs, _info = env.reset(seed=int(seed))
    obs = np.asarray(obs, dtype=float)
    for _ in range(int(burn_in)):
        obs, _reward, terminated, truncated = step_with_heuristic(env, obs)
        if terminated or truncated:
            break
    return capture_state(env)


def rollout(
    env,
    *,
    seed: int,
    trial: int,
    sigma: float,
    burn_in: int,
    horizon: int,
) -> CalibrationRow:
    lander_state, leg_states, checkpoint_obs = checkpoint(env, seed=seed, burn_in=burn_in)
    obs = restore_perturbed(
        env,
        lander_state=lander_state,
        leg_states=leg_states,
        sigma=sigma,
        direction=perturb_direction(seed, trial),
    )
    total_reward = 0.0
    success = False
    crashed = False
    step = 0
    for step in range(1, int(horizon) + 1):
        obs, reward, terminated, truncated = step_with_heuristic(env, obs)
        total_reward += reward
        if terminated or truncated:
            success = bool(reward >= 100.0 and not env.unwrapped.game_over)
            crashed = bool(env.unwrapped.game_over or abs(float(obs[0])) >= 1.0)
            break
    return CalibrationRow(
        seed=int(seed),
        trial=int(trial),
        sigma=float(sigma),
        success=int(success),
        crashed=int(crashed),
        steps=int(step),
        total_reward=float(total_reward),
        checkpoint_x=float(checkpoint_obs[0]),
        checkpoint_y=float(checkpoint_obs[1]),
    )


def summarize(
    rows: list[CalibrationRow],
    radii: np.ndarray,
    *,
    alpha: float,
) -> dict[str, float | int | str | list[float]]:
    curve = []
    crashes = []
    for sigma in radii:
        radius_rows = [row for row in rows if abs(row.sigma - float(sigma)) < 1e-12]
        if not radius_rows:
            raise ValueError(f"missing rows for sigma={sigma}")
        curve.append(float(np.mean([row.success for row in radius_rows])))
        crashes.append(float(np.mean([row.crashed for row in radius_rows])))
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
        "env_id": "LunarLander-v3",
        "rows": len(rows),
        "radii": [float(sigma) for sigma in radii],
        "clean_success": clean_success,
        "min_recovery": min_recovery,
        "max_recovery": max_recovery,
        "recovery_range": float(recovery_range),
        "mean_crash_rate": float(np.mean(crashes)),
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
            "LunarLander recovery calibration requires Gymnasium with Box2D in an isolated environment, "
            "for example /tmp/bgr_lunar_venv."
        ) from exc

    radii = parse_radii(args.radii)
    env = gym.make(args.env_id, continuous=bool(args.continuous), enable_wind=False)
    rows: list[CalibrationRow] = []
    try:
        for seed_idx in range(args.seeds):
            seed = args.seed_offset + seed_idx
            for sigma in radii:
                for trial in range(args.trials):
                    rows.append(
                        rollout(
                            env,
                            seed=seed,
                            trial=trial,
                            sigma=float(sigma),
                            burn_in=args.burn_in,
                            horizon=args.horizon,
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
            "seed_offset": args.seed_offset,
            "burn_in": args.burn_in,
            "horizon": args.horizon,
            "alpha": args.alpha,
        }
    )
    return rows, summary


def write_outputs(
    out_dir: Path,
    rows: list[CalibrationRow],
    summary: dict[str, float | int | str | bool | list[float]],
    *,
    env_id: str,
    continuous: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "package_versions.json").open("w", encoding="utf-8") as handle:
        json.dump(package_versions(env_id, continuous=continuous), handle, indent=2, sort_keys=True)
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
    parser.add_argument("--env-id", default="LunarLander-v3")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--radii", default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--burn-in", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=260)
    parser.add_argument("--alpha", type=float, default=0.80)
    args = parser.parse_args()

    rows, summary = calibrate(args)
    write_outputs(args.out, rows, summary, env_id=args.env_id, continuous=bool(args.continuous))
    print(
        "LunarLander calibration: "
        f"clean={summary['clean_success']:.4f} "
        f"range={summary['min_recovery']:.4f}--{summary['max_recovery']:.4f} "
        f"rauc={summary['rauc']:.4f} "
        f"r80={summary['r80']:.4f} "
        f"decision={summary['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
