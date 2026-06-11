#!/usr/bin/env python3
"""Pre-method recovery calibration for Gymnasium BipedalWalker-v3.

This is a calibration only. It checks whether Gymnasium's package-owned
BipedalWalker dynamics plus the package heuristic expose a resettable recovery
interface before any BGR replay comparison is implemented.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class BodyState:
    position: tuple[float, float]
    angle: float
    linear_velocity: tuple[float, float]
    angular_velocity: float
    ground_contact: bool | None = None


@dataclass(frozen=True, slots=True)
class HeuristicState:
    state: int
    moving_leg: int
    supporting_leg: int
    supporting_knee_angle: float
    action: list[float]


@dataclass(frozen=True, slots=True)
class WalkerCheckpoint:
    seed: int
    hull: BodyState
    legs: list[BodyState]
    heuristic: HeuristicState


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def body_state(body, *, ground_contact: bool | None = None) -> BodyState:
    return BodyState(
        position=(float(body.position.x), float(body.position.y)),
        angle=float(body.angle),
        linear_velocity=(float(body.linearVelocity.x), float(body.linearVelocity.y)),
        angular_velocity=float(body.angularVelocity),
        ground_contact=ground_contact,
    )


def capture_checkpoint(env, heuristic, seed: int) -> WalkerCheckpoint:
    return WalkerCheckpoint(
        seed=int(seed),
        hull=body_state(env.unwrapped.hull),
        legs=[
            body_state(leg, ground_contact=bool(getattr(leg, "ground_contact", False)))
            for leg in env.unwrapped.legs
        ],
        heuristic=HeuristicState(
            state=int(heuristic.state),
            moving_leg=int(heuristic.moving_leg),
            supporting_leg=int(heuristic.supporting_leg),
            supporting_knee_angle=float(heuristic.supporting_knee_angle),
            action=[float(x) for x in heuristic.a],
        ),
    )


def restore_body(body, state: BodyState) -> None:
    body.position = state.position
    body.angle = state.angle
    body.linearVelocity = state.linear_velocity
    body.angularVelocity = state.angular_velocity
    if state.ground_contact is not None and hasattr(body, "ground_contact"):
        body.ground_contact = bool(state.ground_contact)


def restore_checkpoint(env, checkpoint: WalkerCheckpoint, sigma: float, direction: np.ndarray):
    from gymnasium.envs.box2d.bipedal_walker import BipedalWalkerHeuristics

    obs, _info = env.reset(seed=checkpoint.seed)
    del obs
    restore_body(env.unwrapped.hull, checkpoint.hull)
    for leg, state in zip(env.unwrapped.legs, checkpoint.legs, strict=True):
        restore_body(leg, state)

    # Perturb hull pose and velocity in a fixed normalized coordinate system.
    scaled = float(sigma) * direction
    hull = env.unwrapped.hull
    hull.position = (
        float(hull.position.x + 0.80 * scaled[0]),
        float(hull.position.y + 0.30 * scaled[1]),
    )
    hull.angle = float(hull.angle + 0.50 * scaled[2])
    hull.linearVelocity = (
        float(hull.linearVelocity.x + 1.00 * scaled[3]),
        float(hull.linearVelocity.y + 0.60 * scaled[4]),
    )
    hull.angularVelocity = float(hull.angularVelocity + 0.75 * scaled[5])

    heuristic = BipedalWalkerHeuristics()
    heuristic.state = checkpoint.heuristic.state
    heuristic.moving_leg = checkpoint.heuristic.moving_leg
    heuristic.supporting_leg = checkpoint.heuristic.supporting_leg
    heuristic.supporting_knee_angle = checkpoint.heuristic.supporting_knee_angle
    heuristic.a = np.array(checkpoint.heuristic.action, dtype=float)
    return heuristic


def current_obs(env) -> np.ndarray:
    from gymnasium.envs.box2d.bipedal_walker import (
        FPS,
        LIDAR_RANGE,
        SCALE,
        SPEED_HIP,
        SPEED_KNEE,
        VIEWPORT_H,
        VIEWPORT_W,
    )

    walker = env.unwrapped
    pos = walker.hull.position
    vel = walker.hull.linearVelocity
    for i in range(10):
        walker.lidar[i].fraction = 1.0
        walker.lidar[i].p1 = pos
        walker.lidar[i].p2 = (
            pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
            pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
        )
        walker.world.RayCast(walker.lidar[i], walker.lidar[i].p1, walker.lidar[i].p2)
    state = [
        walker.hull.angle,
        2.0 * walker.hull.angularVelocity / FPS,
        0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,
        0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
        walker.joints[0].angle,
        walker.joints[0].speed / SPEED_HIP,
        walker.joints[1].angle + 1.0,
        walker.joints[1].speed / SPEED_KNEE,
        1.0 if walker.legs[1].ground_contact else 0.0,
        walker.joints[2].angle,
        walker.joints[2].speed / SPEED_HIP,
        walker.joints[3].angle + 1.0,
        walker.joints[3].speed / SPEED_KNEE,
        1.0 if walker.legs[3].ground_contact else 0.0,
    ]
    state += [lidar.fraction for lidar in walker.lidar]
    return np.array(state, dtype=float)


def direction_for(seed: int, radius_idx: int, trial: int) -> np.ndarray:
    rng = np.random.default_rng(917_000 + 10_000 * int(seed) + 101 * int(radius_idx) + int(trial))
    direction = rng.normal(0.0, 1.0, size=6)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction[0] = 1.0
        return direction
    return direction / norm


def rollout(env, heuristic, obs: np.ndarray, *, horizon: int, min_progress: float) -> tuple[bool, float, bool]:
    start_x = float(env.unwrapped.hull.position.x)
    fell = False
    for _ in range(int(horizon)):
        obs, _reward, terminated, truncated, _info = env.step(heuristic.step_heuristic(obs))
        if terminated or truncated:
            fell = bool(terminated)
            break
    progress = float(env.unwrapped.hull.position.x - start_x)
    return (not fell and progress >= float(min_progress)), progress, fell


def evaluate_radius(env, checkpoint: WalkerCheckpoint, radius: float, radius_idx: int, args: argparse.Namespace) -> dict[str, float]:
    successes = []
    progresses = []
    falls = []
    for trial in range(int(args.trials)):
        direction = direction_for(checkpoint.seed, radius_idx, trial)
        heuristic = restore_checkpoint(env, checkpoint, radius, direction)
        obs = current_obs(env)
        success, progress, fell = rollout(
            env,
            heuristic,
            obs,
            horizon=args.horizon,
            min_progress=args.min_progress,
        )
        successes.append(float(success))
        progresses.append(float(progress))
        falls.append(float(fell))
    return {
        "success": float(np.mean(successes)),
        "progress": float(np.mean(progresses)),
        "fall_rate": float(np.mean(falls)),
    }


def critical_radius_from_curve(radii: list[float], successes: list[float], threshold: float) -> float:
    accepted = [radius for radius, success in zip(radii, successes, strict=True) if success >= threshold]
    if not accepted:
        return float(min(radii))
    return float(max(accepted))


def auc_from_curve(radii: list[float], successes: list[float]) -> float:
    max_radius = float(max(radii))
    if max_radius <= 0.0:
        return float(np.mean(successes))
    order = np.argsort(radii)
    xs = np.array([radii[int(i)] for i in order], dtype=float) / max_radius
    ys = np.array([successes[int(i)] for i in order], dtype=float)
    return float(np.trapezoid(ys, xs))


def package_versions() -> dict[str, str]:
    import Box2D
    import gymnasium

    return {
        "gymnasium": gymnasium.__version__,
        "Box2D": getattr(Box2D, "__version__", "unknown"),
        "numpy": np.__version__,
    }


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    from gymnasium.envs.box2d.bipedal_walker import BipedalWalkerHeuristics

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    radii = parse_floats(args.radii)
    rows: list[dict[str, float | int]] = []
    clean_successes = []
    r80_values = []
    auc_values = []

    for seed in range(int(args.seeds)):
        env = gym.make(args.env_id)
        obs, _info = env.reset(seed=seed)
        heuristic = BipedalWalkerHeuristics()
        for _ in range(int(args.burn_in)):
            obs, _reward, terminated, truncated, _info = env.step(heuristic.step_heuristic(obs))
            if terminated or truncated:
                break
        checkpoint = capture_checkpoint(env, heuristic, seed)
        seed_successes = []
        for radius_idx, radius in enumerate(radii):
            metrics = evaluate_radius(env, checkpoint, radius, radius_idx, args)
            seed_successes.append(metrics["success"])
            rows.append(
                {
                    "seed": seed,
                    "radius": float(radius),
                    "success": metrics["success"],
                    "progress": metrics["progress"],
                    "fall_rate": metrics["fall_rate"],
                }
            )
        clean_successes.append(seed_successes[0])
        r80_values.append(critical_radius_from_curve(radii, seed_successes, args.success_threshold))
        auc_values.append(auc_from_curve(radii, seed_successes))
        env.close()

    clean = float(np.mean(clean_successes))
    min_recovery = float(min(row["success"] for row in rows))
    max_recovery = float(max(row["success"] for row in rows))
    summary = {
        "env_id": args.env_id,
        "seeds": int(args.seeds),
        "burn_in": int(args.burn_in),
        "horizon": int(args.horizon),
        "trials": int(args.trials),
        "radii": radii,
        "clean_success": clean,
        "recovery_min": min_recovery,
        "recovery_max": max_recovery,
        "mean_rauc": float(np.mean(auc_values)),
        "median_r80": float(np.median(r80_values)),
        "decision": (
            "usable-calibration"
            if clean >= args.min_clean_success and max_recovery > min_recovery and float(np.median(r80_values)) < max(radii)
            else "reject-calibration"
        ),
    }

    with (out / "recovery_rows.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seed", "radius", "success", "progress", "fall_rate"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "package_versions.json").write_text(json.dumps(package_versions(), indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--env-id", default="BipedalWalker-v3")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--burn-in", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--radii", default="0.00,0.20,0.40,0.70,1.00,1.40")
    parser.add_argument("--min-progress", type=float, default=2.0)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--min-clean-success", type=float, default=0.8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
