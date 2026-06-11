#!/usr/bin/env python3
"""Calibrate package-backed MinAtar Seaquest recovery curves.

This is a pre-comparison diagnostic for a new independent benchmark route. It
uses MinAtar's own Seaquest dynamics, a fixed safety-aware controller, and
integer leftward submarine-column perturbations. It does not compare BGR
methods.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from bgr.metrics import critical_radius, recovery_auc


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    seed: int
    sigma: float
    success: int
    terminal: int
    steps: int
    reward: int
    checkpoint_x: int
    checkpoint_y: int
    perturbed_x: int
    perturbed_y: int
    oxygen: int
    diver_count: int


def parse_radii(value: str) -> np.ndarray:
    radii = np.array([float(item.strip()) for item in value.split(",") if item.strip()], dtype=float)
    if radii.ndim != 1 or len(radii) < 2:
        raise ValueError("--radii must contain at least two comma-separated values")
    if not np.all(np.diff(radii) > 0.0):
        raise ValueError("--radii must be strictly increasing")
    if not np.all(np.isclose(radii, np.round(radii))):
        raise ValueError("--radii must be integer submarine-column offsets")
    return radii


def package_versions() -> dict[str, str]:
    try:
        import minatar  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "MinAtar Seaquest calibration requires MinAtar in an isolated "
            "environment, for example /tmp/bgr_minatar_venv."
        ) from exc
    return {
        "MinAtar": version("MinAtar"),
        "numpy": np.__version__,
    }


def imminent_danger(game: Any, x: int | None = None, y: int | None = None) -> bool:
    sub_x = int(game.sub_x if x is None else x)
    sub_y = int(game.sub_y if y is None else y)
    for obj in list(game.e_fish) + list(game.e_subs):
        obj_x, obj_y, left_to_right = int(obj[0]), int(obj[1]), bool(obj[2])
        if [obj_x, obj_y] == [sub_x, sub_y]:
            return True
        if int(obj[3]) == 0:
            next_x = obj_x + (1 if left_to_right else -1)
            if [next_x, obj_y] == [sub_x, sub_y]:
                return True
    for bullet in game.e_bullets:
        bullet_x, bullet_y, left_to_right = int(bullet[0]), int(bullet[1]), bool(bullet[2])
        if [bullet_x, bullet_y] == [sub_x, sub_y]:
            return True
        next_x = bullet_x + (1 if left_to_right else -1)
        if [next_x, bullet_y] == [sub_x, sub_y]:
            return True
    return False


def nearest_target(game: Any) -> list[int] | None:
    same_row = [obj for obj in list(game.e_fish) + list(game.e_subs) if int(obj[1]) == int(game.sub_y)]
    if same_row:
        return min(same_row, key=lambda obj: abs(int(obj[0]) - int(game.sub_x)))
    objects = list(game.e_fish) + list(game.e_subs) + list(game.divers)
    if not objects:
        return None
    return min(objects, key=lambda obj: abs(int(obj[0]) - int(game.sub_x)) + abs(int(obj[1]) - int(game.sub_y)))


def controller_action(env: Any) -> int:
    """A fixed safety-aware Seaquest controller using MinAtar action ids."""
    game = env.env
    sub_x = int(game.sub_x)
    sub_y = int(game.sub_y)

    if int(game.oxygen) < 40 or int(game.diver_count) >= 1:
        if sub_y > 0 and not imminent_danger(game, sub_x, sub_y - 1):
            return 2

    if imminent_danger(game, sub_x, sub_y):
        escape_moves = [
            (2, sub_x, max(0, sub_y - 1)),
            (4, sub_x, min(8, sub_y + 1)),
            (1, max(0, sub_x - 1), sub_y),
            (3, min(9, sub_x + 1), sub_y),
        ]
        for action, next_x, next_y in escape_moves:
            if not imminent_danger(game, next_x, next_y):
                return action

    same_row = [obj for obj in list(game.e_fish) + list(game.e_subs) if int(obj[1]) == sub_y]
    if same_row and int(game.shot_timer) == 0:
        target = min(same_row, key=lambda obj: abs(int(obj[0]) - sub_x))
        target_x = int(target[0])
        if (target_x > sub_x and bool(game.sub_or)) or (target_x < sub_x and not bool(game.sub_or)):
            return 5
        return 3 if target_x > sub_x else 1

    target = nearest_target(game)
    if target is None:
        return 4 if sub_y < 4 and int(game.oxygen) > 80 else 0

    target_x = int(target[0])
    target_y = int(target[1])
    if sub_y != target_y and int(game.oxygen) > 50:
        next_y = sub_y + (1 if target_y > sub_y else -1)
        if not imminent_danger(game, sub_x, next_y):
            return 4 if target_y > sub_y else 2
    if sub_x != target_x:
        next_x = sub_x + (1 if target_x > sub_x else -1)
        if not imminent_danger(game, next_x, sub_y):
            return 3 if target_x > sub_x else 1
    return 0


def seed_and_reset(env: Any, seed: int) -> None:
    env.seed(int(seed))
    env.reset()
    env.last_action = 0


def checkpoint_state(env: Any, *, seed: int, burn_in: int) -> Any:
    seed_and_reset(env, seed)
    for _step in range(int(burn_in)):
        _reward, terminal = env.act(controller_action(env))
        if terminal:
            break
    return copy.deepcopy(env.env)


def perturb_submarine(env: Any, *, sigma: float, mode: str) -> tuple[int, int]:
    game = env.env
    if mode == "left":
        game.sub_x = int(np.clip(int(game.sub_x) - int(round(float(sigma))), 0, 9))
    elif mode == "right":
        game.sub_x = int(np.clip(int(game.sub_x) + int(round(float(sigma))), 0, 9))
    elif mode == "down":
        game.sub_y = int(np.clip(int(game.sub_y) + int(round(float(sigma))), 0, 8))
    else:
        raise ValueError(f"unknown perturbation mode: {mode}")
    game.terminal = False
    return int(game.sub_x), int(game.sub_y)


def rollout(env: Any, *, seed: int, sigma: float, burn_in: int, horizon: int, perturbation: str) -> CalibrationRow:
    state = checkpoint_state(env, seed=seed, burn_in=burn_in)
    checkpoint_x = int(state.sub_x)
    checkpoint_y = int(state.sub_y)
    env.env = copy.deepcopy(state)
    perturbed_x, perturbed_y = perturb_submarine(env, sigma=sigma, mode=perturbation)
    reward = 0
    step = 0
    terminal = False
    for step in range(1, int(horizon) + 1):
        step_reward, terminal = env.act(controller_action(env))
        reward += int(step_reward)
        if terminal:
            break
    success = not bool(env.env.terminal)
    return CalibrationRow(
        seed=int(seed),
        sigma=float(sigma),
        success=int(success),
        terminal=int(bool(env.env.terminal)),
        steps=int(step),
        reward=int(reward),
        checkpoint_x=checkpoint_x,
        checkpoint_y=checkpoint_y,
        perturbed_x=perturbed_x,
        perturbed_y=perturbed_y,
        oxygen=int(env.env.oxygen),
        diver_count=int(env.env.diver_count),
    )


def summarize(rows: list[CalibrationRow], radii: np.ndarray, *, alpha: float) -> dict[str, float | int | str | list[float]]:
    curve = []
    terminal_rates = []
    rewards = []
    for sigma in radii:
        radius_rows = [row for row in rows if abs(row.sigma - float(sigma)) < 1e-12]
        if not radius_rows:
            raise ValueError(f"missing rows for sigma={sigma}")
        curve.append(float(np.mean([row.success for row in radius_rows])))
        terminal_rates.append(float(np.mean([row.terminal for row in radius_rows])))
        rewards.append(float(np.mean([row.reward for row in radius_rows])))
    curve_array = np.array(curve, dtype=float)
    clean_success = float(curve_array[0])
    min_recovery = float(np.min(curve_array))
    max_recovery = float(np.max(curve_array))
    recovery_range = max_recovery - min_recovery
    r80 = float(critical_radius(radii, curve_array, alpha=alpha))
    if clean_success < 0.80:
        decision = "reject-calibration-low-clean-success"
    elif recovery_range < 0.20 - 1e-12:
        decision = "reject-calibration-flat-recovery"
    elif r80 >= float(radii[-1]) - 1e-12:
        decision = "reject-calibration-saturated-radius"
    else:
        decision = "usable-calibration"
    return {
        "env_id": "minatar-seaquest",
        "seeds": len({row.seed for row in rows}),
        "radii": [float(sigma) for sigma in radii],
        "clean_success": clean_success,
        "min_recovery": min_recovery,
        "max_recovery": max_recovery,
        "recovery_range": float(recovery_range),
        "mean_terminal_rate": float(np.mean(terminal_rates)),
        "mean_reward": float(np.mean(rewards)),
        "rauc": float(recovery_auc(radii, curve_array, sigma_max=float(radii[-1]))),
        "r80": r80,
        "decision": decision,
        "mean_curve": [float(value) for value in curve_array],
    }


def calibrate(args: argparse.Namespace) -> tuple[list[CalibrationRow], dict[str, float | int | str | list[float]]]:
    try:
        from minatar import Environment
    except ImportError as exc:
        raise SystemExit(
            "MinAtar Seaquest calibration requires MinAtar in an isolated "
            "environment, for example /tmp/bgr_minatar_venv."
        ) from exc
    radii = parse_radii(args.radii)
    env = Environment(
        "seaquest",
        sticky_action_prob=args.sticky_action_prob,
        difficulty_ramping=args.difficulty_ramping,
    )
    rows: list[CalibrationRow] = []
    try:
        for seed in range(int(args.seeds)):
            for sigma in radii:
                rows.append(
                    rollout(
                        env,
                        seed=seed,
                        sigma=float(sigma),
                        burn_in=args.burn_in,
                        horizon=args.horizon,
                        perturbation=args.perturbation,
                    )
                )
    finally:
        env.close_display()
    return rows, summarize(rows, radii, alpha=args.alpha)


def write_rows(path: Path, rows: list[CalibrationRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/minatar_seaquest_recovery_calibration_30seed_v1")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--radii", default="0,1,2,3,4,5")
    parser.add_argument("--burn-in", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--perturbation", choices=["left", "right", "down"], default="left")
    parser.add_argument("--sticky-action-prob", type=float, default=0.0)
    parser.add_argument("--difficulty-ramping", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows, summary = calibrate(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "rows.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "package_versions.json").write_text(
        json.dumps(package_versions(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
