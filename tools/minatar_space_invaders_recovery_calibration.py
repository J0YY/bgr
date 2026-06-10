#!/usr/bin/env python3
"""Calibrate package-backed MinAtar Space Invaders recovery curves.

This is a pre-comparison diagnostic for a new independent benchmark route. It
uses MinAtar's own Space Invaders dynamics, a fixed align-and-fire controller,
and integer player-column perturbations. It does not compare BGR methods.
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
    checkpoint_pos: int
    perturbed_pos: int


def parse_radii(value: str) -> np.ndarray:
    radii = np.array([float(item.strip()) for item in value.split(",") if item.strip()], dtype=float)
    if radii.ndim != 1 or len(radii) < 2:
        raise ValueError("--radii must contain at least two comma-separated values")
    if not np.all(np.diff(radii) > 0.0):
        raise ValueError("--radii must be strictly increasing")
    if not np.all(np.isclose(radii, np.round(radii))):
        raise ValueError("--radii must be integer player-column offsets")
    return radii


def package_versions() -> dict[str, str]:
    try:
        import minatar  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "MinAtar Space Invaders calibration requires MinAtar in an isolated "
            "environment, for example /tmp/bgr_minatar_venv."
        ) from exc
    return {
        "MinAtar": version("MinAtar"),
        "numpy": np.__version__,
    }


def nearest_alien_col(game: Any, col: int | None = None) -> int | None:
    alien_cols = np.where(np.asarray(game.alien_map).any(axis=0))[0]
    if alien_cols.size == 0:
        return None
    ref = int(game.pos) if col is None else int(col)
    return int(alien_cols[np.argmin(np.abs(alien_cols - ref))])


def column_has_alien(game: Any, col: int) -> bool:
    return bool(np.asarray(game.alien_map)[:, int(col)].any())


def bullet_danger(game: Any, col: int | None = None) -> bool:
    player_col = int(game.pos if col is None else col)
    bullets = np.asarray(game.e_bullet_map)
    for delta in (-1, 0, 1):
        candidate = player_col + delta
        if 0 <= candidate < 10 and bullets[7:10, candidate].any():
            return True
    return False


def controller_action(env: Any) -> int:
    """A fixed bullet-aware align-and-fire Space Invaders controller."""
    game = env.env
    pos = int(game.pos)
    if bullet_danger(game, pos):
        left_safe = pos > 0 and not bullet_danger(game, pos - 1)
        right_safe = pos < 9 and not bullet_danger(game, pos + 1)
        if left_safe and not right_safe:
            return 1
        if right_safe and not left_safe:
            return 3
        if left_safe and right_safe:
            target = nearest_alien_col(game, pos)
            return 3 if target is not None and target > pos else 1
    if int(game.shot_timer) == 0 and column_has_alien(game, pos):
        return 5
    target = nearest_alien_col(game, pos)
    if target is None:
        return 0
    if target < pos:
        return 1
    if target > pos:
        return 3
    return 5 if int(game.shot_timer) == 0 else 0


def seed_and_reset(env: Any, seed: int) -> None:
    env.seed(int(seed))
    env.reset()
    env.last_action = 0


def checkpoint(env: Any, *, seed: int, burn_in: int) -> int:
    seed_and_reset(env, seed)
    for _step in range(int(burn_in)):
        _reward, terminal = env.act(controller_action(env))
        if terminal:
            break
    return int(env.env.pos)


def checkpoint_state(env: Any, *, seed: int, burn_in: int) -> Any:
    checkpoint(env, seed=seed, burn_in=burn_in)
    return copy.deepcopy(env.env)


def restore_state(env: Any, state: Any) -> None:
    env.env = copy.deepcopy(state)


def perturb_player(env: Any, *, sigma: float, mode: str) -> int:
    game = env.env
    pos = int(game.pos)
    if mode == "right":
        perturbed = pos + int(round(float(sigma)))
    elif mode == "away":
        target = nearest_alien_col(game, pos)
        direction = 1 if target is None or pos <= target else -1
        perturbed = pos + direction * int(round(float(sigma)))
    else:
        raise ValueError(f"unknown perturbation mode: {mode}")
    game.pos = int(np.clip(perturbed, 0, 9))
    game.terminal = False
    return int(game.pos)


def rollout(env: Any, *, seed: int, sigma: float, burn_in: int, horizon: int, reward_threshold: int, perturbation: str) -> CalibrationRow:
    state = checkpoint_state(env, seed=seed, burn_in=burn_in)
    checkpoint_pos = int(state.pos)
    restore_state(env, state)
    perturbed_pos = perturb_player(env, sigma=sigma, mode=perturbation)
    reward = 0
    step = 0
    terminal = False
    for step in range(1, int(horizon) + 1):
        step_reward, terminal = env.act(controller_action(env))
        reward += int(step_reward)
        if terminal:
            break
    success = (not bool(env.env.terminal)) and reward >= int(reward_threshold)
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
        "env_id": "minatar-space-invaders",
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
            "MinAtar Space Invaders calibration requires MinAtar in an isolated "
            "environment, for example /tmp/bgr_minatar_venv."
        ) from exc

    radii = parse_radii(args.radii)
    env = Environment(args.game, sticky_action_prob=args.sticky_action_prob, difficulty_ramping=args.difficulty_ramping)
    rows: list[CalibrationRow] = []
    try:
        for seed in range(args.seeds):
            for sigma in radii:
                rows.append(
                    rollout(
                        env,
                        seed=seed,
                        sigma=float(sigma),
                        burn_in=args.burn_in,
                        horizon=args.horizon,
                        reward_threshold=args.reward_threshold,
                        perturbation=args.perturbation,
                    )
                )
    finally:
        env.close_display()
    summary = summarize(rows, radii, alpha=args.alpha)
    summary.update(
        {
            "game": args.game,
            "burn_in": args.burn_in,
            "horizon": args.horizon,
            "reward_threshold": args.reward_threshold,
            "sticky_action_prob": args.sticky_action_prob,
            "difficulty_ramping": bool(args.difficulty_ramping),
            "policy": "bullet-aware-align-and-fire-controller",
            "perturbation": f"{args.perturbation}-player-column-displacement",
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
    parser.add_argument("--game", default="space_invaders")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--radii", default="0,1,2,3,4,5,6")
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--reward-threshold", type=int, default=2)
    parser.add_argument("--perturbation", choices=["right", "away"], default="right")
    parser.add_argument("--sticky-action-prob", type=float, default=0.0)
    parser.add_argument("--difficulty-ramping", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.80)
    args = parser.parse_args()

    rows, summary = calibrate(args)
    write_outputs(args.out, rows, summary)
    print(
        "MinAtar Space Invaders calibration: "
        f"clean={summary['clean_success']:.4f} "
        f"range={summary['min_recovery']:.4f}--{summary['max_recovery']:.4f} "
        f"rauc={summary['rauc']:.4f} "
        f"r80={summary['r80']:.4f} "
        f"decision={summary['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
