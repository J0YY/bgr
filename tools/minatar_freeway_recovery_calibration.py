#!/usr/bin/env python3
"""Calibrate package-backed MinAtar Freeway recovery curves.

This is a pre-comparison diagnostic for a new independent benchmark route. It
uses MinAtar's own Freeway dynamics, a fixed safe-up controller, and integer
downward player-row perturbations at a mid-episode checkpoint. It does not
compare BGR methods.
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
    success: int
    terminal: int
    steps: int
    reward: int
    checkpoint_pos: int
    perturbed_pos: int


def parse_radii(value: str) -> np.ndarray:
    radii = np.array([float(item.strip()) for item in value.split(",") if item.strip()], dtype=float)
    if radii.ndim != 1 or len(radii) < 2:
        raise ValueError("--radii must contain at least two comma-separated values")
    if not np.all(np.diff(radii) > 0.0):
        raise ValueError("--radii must be strictly increasing")
    if not np.all(np.isclose(radii, np.round(radii))):
        raise ValueError("--radii must be integer player-row offsets")
    return radii


def package_versions() -> dict[str, str]:
    try:
        import minatar  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "MinAtar Freeway calibration requires MinAtar in an isolated "
            "environment, for example /tmp/bgr_minatar_venv."
        ) from exc
    return {
        "MinAtar": version("MinAtar"),
        "numpy": np.__version__,
    }


def row_danger(game, row: int) -> bool:
    """Whether the chicken would collide on this row before/after car motion."""
    for car_x, car_y, timer, speed in game.cars:
        if int(car_y) != int(row):
            continue
        if int(car_x) == 4:
            return True
        if int(timer) == 0:
            next_x = int(car_x) + (1 if int(speed) > 0 else -1)
            if next_x < 0:
                next_x = 9
            elif next_x > 9:
                next_x = 0
            if next_x == 4:
                return True
    return False


def controller_action(env) -> int:
    """A fixed safety-aware Freeway controller using MinAtar action ids."""
    game = env.env
    if int(game.move_timer) != 0:
        return 0
    up_row = max(0, int(game.pos) - 1)
    down_row = min(9, int(game.pos) + 1)
    if not row_danger(game, up_row):
        return 2
    if not row_danger(game, int(game.pos)):
        return 0
    if not row_danger(game, down_row):
        return 4
    return 0


def seed_and_reset(env, seed: int) -> None:
    env.seed(int(seed))
    env.reset()
    env.last_action = 0


def checkpoint(env, *, seed: int, burn_in: int) -> int:
    seed_and_reset(env, seed)
    for _step in range(int(burn_in)):
        _reward, terminal = env.act(controller_action(env))
        if terminal:
            break
    return int(env.env.pos)


def perturb_player(env, *, sigma: float) -> int:
    game = env.env
    game.pos = int(np.clip(int(game.pos) + int(round(float(sigma))), 0, 9))
    game.terminal = False
    return int(game.pos)


def rollout(env, *, seed: int, sigma: float, burn_in: int, horizon: int) -> CalibrationRow:
    checkpoint_pos = checkpoint(env, seed=seed, burn_in=burn_in)
    perturbed_pos = perturb_player(env, sigma=sigma)
    reward = 0
    step = 0
    terminal = False
    for step in range(1, int(horizon) + 1):
        step_reward, terminal = env.act(controller_action(env))
        reward += int(step_reward)
        if terminal:
            break
    success = (not bool(env.env.terminal)) and reward >= 1
    return CalibrationRow(
        seed=int(seed),
        sigma=float(sigma),
        success=int(success),
        terminal=int(bool(env.env.terminal)),
        steps=int(step),
        reward=int(reward),
        checkpoint_pos=checkpoint_pos,
        perturbed_pos=perturbed_pos,
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
        "env_id": "minatar-freeway",
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
            "MinAtar Freeway calibration requires MinAtar in an isolated "
            "environment, for example /tmp/bgr_minatar_venv."
        ) from exc

    radii = parse_radii(args.radii)
    env = Environment(args.game, sticky_action_prob=args.sticky_action_prob, difficulty_ramping=args.difficulty_ramping)
    rows: list[CalibrationRow] = []
    try:
        for seed in range(args.seeds):
            for sigma in radii:
                rows.append(rollout(env, seed=seed, sigma=float(sigma), burn_in=args.burn_in, horizon=args.horizon))
    finally:
        env.close_display()
    summary = summarize(rows, radii, alpha=args.alpha)
    summary.update(
        {
            "game": args.game,
            "burn_in": args.burn_in,
            "horizon": args.horizon,
            "sticky_action_prob": args.sticky_action_prob,
            "difficulty_ramping": bool(args.difficulty_ramping),
            "policy": "safe-up-controller",
            "perturbation": "downward-player-row-displacement",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--game", default="freeway")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--radii", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--sticky-action-prob", type=float, default=0.0)
    parser.add_argument("--difficulty-ramping", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.80)
    args = parser.parse_args()

    rows, summary = calibrate(args)
    write_outputs(args.out, rows, summary)
    print(
        "MinAtar Freeway calibration: "
        f"clean={summary['clean_success']:.4f} "
        f"range={summary['min_recovery']:.4f}--{summary['max_recovery']:.4f} "
        f"rauc={summary['rauc']:.4f} "
        f"r80={summary['r80']:.4f} "
        f"decision={summary['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
