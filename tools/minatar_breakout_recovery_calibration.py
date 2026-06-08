#!/usr/bin/env python3
"""Calibrate package-backed MinAtar Breakout recovery curves.

This is a pre-comparison diagnostic for a different external benchmark route.
It uses MinAtar's own Breakout dynamics and a fixed paddle-tracking controller,
then perturbs the paddle position at a checkpoint. It does not compare BGR
methods.
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
    checkpoint_ball_x: int
    checkpoint_ball_y: int
    checkpoint_paddle_x: int
    perturbed_paddle_x: int


def parse_radii(value: str) -> np.ndarray:
    radii = np.array([float(item.strip()) for item in value.split(",") if item.strip()], dtype=float)
    if radii.ndim != 1 or len(radii) < 2:
        raise ValueError("--radii must contain at least two comma-separated values")
    if not np.all(np.diff(radii) > 0.0):
        raise ValueError("--radii must be strictly increasing")
    if not np.all(np.isclose(radii, np.round(radii))):
        raise ValueError("--radii must be integer paddle-cell offsets")
    return radii


def package_versions() -> dict[str, str]:
    try:
        import minatar  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "MinAtar Breakout calibration requires MinAtar in an isolated "
            "environment, for example /tmp/bgr_minatar_venv."
        ) from exc
    return {
        "MinAtar": version("MinAtar"),
        "numpy": np.__version__,
    }


def controller_action(env) -> int:
    game = env.env
    if int(game.ball_x) < int(game.pos):
        return 1
    if int(game.ball_x) > int(game.pos):
        return 3
    return 0


def seed_and_reset(env, seed: int) -> None:
    env.seed(int(seed))
    env.reset()


def checkpoint(env, *, seed: int, burn_in: int) -> tuple[int, int, int]:
    seed_and_reset(env, seed)
    for _step in range(int(burn_in)):
        _reward, terminal = env.act(controller_action(env))
        if terminal:
            break
    game = env.env
    return int(game.ball_x), int(game.ball_y), int(game.pos)


def perturb_paddle(env, *, seed: int, sigma: float) -> int:
    game = env.env
    sign = -1 if int(seed) % 2 else 1
    perturbed = int(np.clip(int(game.pos) + sign * int(round(float(sigma))), 0, 9))
    game.pos = perturbed
    game.terminal = False
    return perturbed


def rollout(env, *, seed: int, sigma: float, burn_in: int, horizon: int) -> CalibrationRow:
    ball_x, ball_y, paddle_x = checkpoint(env, seed=seed, burn_in=burn_in)
    perturbed_paddle_x = perturb_paddle(env, seed=seed, sigma=sigma)
    reward = 0
    step = 0
    terminal = False
    for step in range(1, int(horizon) + 1):
        step_reward, terminal = env.act(controller_action(env))
        reward += int(step_reward)
        if terminal:
            break
    success = (not bool(env.env.terminal)) and step >= int(horizon)
    return CalibrationRow(
        seed=int(seed),
        sigma=float(sigma),
        success=int(success),
        terminal=int(bool(env.env.terminal)),
        steps=int(step),
        reward=int(reward),
        checkpoint_ball_x=ball_x,
        checkpoint_ball_y=ball_y,
        checkpoint_paddle_x=paddle_x,
        perturbed_paddle_x=perturbed_paddle_x,
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
    r80 = float(critical_radius(radii, curve_array, alpha=alpha))
    recovery_range = max_recovery - min_recovery
    if clean_success < 0.80:
        decision = "reject-calibration-low-clean-success"
    elif recovery_range < 0.20 - 1e-12:
        decision = "reject-calibration-flat-recovery"
    elif r80 >= float(radii[-1]) - 1e-12:
        decision = "reject-calibration-saturated-radius"
    else:
        decision = "usable-calibration"
    return {
        "env_id": "minatar-breakout",
        "seeds": len({row.seed for row in rows}),
        "radii": [float(sigma) for sigma in radii],
        "clean_success": clean_success,
        "min_recovery": min_recovery,
        "max_recovery": max_recovery,
        "mean_terminal_rate": float(np.mean(terminal_rates)),
        "mean_reward": float(np.mean(rewards)),
        "rauc": float(recovery_auc(radii, curve_array, sigma_max=float(radii[-1]))),
        "r80": r80,
        "decision": decision,
    }


def calibrate(args: argparse.Namespace) -> tuple[list[CalibrationRow], dict[str, float | int | str | list[float]]]:
    try:
        from minatar import Environment
    except ImportError as exc:
        raise SystemExit(
            "MinAtar Breakout calibration requires MinAtar in an isolated "
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
            "policy": "track-ball-with-paddle",
            "perturbation": "signed-paddle-cell-offset",
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
    parser.add_argument("--game", default="breakout")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--radii", default="0,1,2,3,4,5")
    parser.add_argument("--burn-in", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--sticky-action-prob", type=float, default=0.0)
    parser.add_argument("--difficulty-ramping", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.80)
    args = parser.parse_args()

    rows, summary = calibrate(args)
    write_outputs(args.out, rows, summary)
    print(
        "MinAtar Breakout calibration: "
        f"clean={summary['clean_success']:.4f} "
        f"range={summary['min_recovery']:.4f}--{summary['max_recovery']:.4f} "
        f"r80={summary['r80']:.4f} "
        f"terminal={summary['mean_terminal_rate']:.4f} "
        f"decision={summary['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
