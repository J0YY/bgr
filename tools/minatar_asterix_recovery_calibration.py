#!/usr/bin/env python3
"""Calibrate package-backed MinAtar Asterix recovery curves.

This is a pre-comparison diagnostic for a new independent benchmark route. It
uses MinAtar's own Asterix dynamics and a fixed gold-seeking/avoidance
controller, then perturbs the player position at a mid-episode checkpoint. It
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
    success: int
    terminal: int
    steps: int
    reward: int
    checkpoint_player_x: int
    checkpoint_player_y: int
    perturbed_player_x: int
    perturbed_player_y: int


def parse_radii(value: str) -> np.ndarray:
    radii = np.array([float(item.strip()) for item in value.split(",") if item.strip()], dtype=float)
    if radii.ndim != 1 or len(radii) < 2:
        raise ValueError("--radii must contain at least two comma-separated values")
    if not np.all(np.diff(radii) > 0.0):
        raise ValueError("--radii must be strictly increasing")
    if not np.all(np.isclose(radii, np.round(radii))):
        raise ValueError("--radii must be integer player-cell offsets")
    return radii


def package_versions() -> dict[str, str]:
    try:
        import minatar  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "MinAtar Asterix calibration requires MinAtar in an isolated "
            "environment, for example /tmp/bgr_minatar_venv."
        ) from exc
    return {
        "MinAtar": version("MinAtar"),
        "numpy": np.__version__,
    }


def enemy_threatens(game, x: int, y: int) -> bool:
    for entity in game.entities:
        if entity is None:
            continue
        entity_x, entity_y, left_to_right, is_gold = entity
        if is_gold or int(entity_y) != int(y):
            continue
        next_x = int(entity_x) + (1 if left_to_right else -1)
        if int(entity_x) == int(x) or next_x == int(x):
            return True
    return False


def safe_cell(game, x: int, y: int) -> bool:
    return 0 <= int(x) <= 9 and 1 <= int(y) <= 8 and not enemy_threatens(game, int(x), int(y))


def controller_action(env) -> int:
    game = env.env
    moves = [
        (0, 0, 0),
        (-1, 0, 1),
        (0, -1, 2),
        (1, 0, 3),
        (0, 1, 4),
    ]
    if not safe_cell(game, game.player_x, game.player_y):
        for dx, dy, action in [(-1, 0, 1), (1, 0, 3), (0, -1, 2), (0, 1, 4), (0, 0, 0)]:
            if safe_cell(game, game.player_x + dx, game.player_y + dy):
                return action

    gold_entities = [entity for entity in game.entities if entity is not None and entity[3]]
    if gold_entities:
        gold_entities.sort(key=lambda entity: abs(entity[0] - game.player_x) + abs(entity[1] - game.player_y))
        target_x, target_y, _direction, _is_gold = gold_entities[0]
        preferences: list[tuple[int, int, int]] = []
        if int(target_y) < int(game.player_y):
            preferences.append((0, -1, 2))
        elif int(target_y) > int(game.player_y):
            preferences.append((0, 1, 4))
        if int(target_x) < int(game.player_x):
            preferences.append((-1, 0, 1))
        elif int(target_x) > int(game.player_x):
            preferences.append((1, 0, 3))
        preferences.append((0, 0, 0))
        for dx, dy, action in preferences:
            if safe_cell(game, game.player_x + dx, game.player_y + dy):
                return action

    for dx, dy, action in moves:
        if safe_cell(game, game.player_x + dx, game.player_y + dy):
            return action
    return 0


def seed_and_reset(env, seed: int) -> None:
    env.seed(int(seed))
    env.reset()


def checkpoint(env, *, seed: int, burn_in: int) -> tuple[int, int]:
    seed_and_reset(env, seed)
    for _step in range(int(burn_in)):
        _reward, terminal = env.act(controller_action(env))
        if terminal:
            break
    game = env.env
    return int(game.player_x), int(game.player_y)


def perturb_player(env, *, seed: int, sigma: float) -> tuple[int, int]:
    game = env.env
    rng = np.random.default_rng(seed * 10_003 + int(round(float(sigma))) * 997 + 19_891)
    direction = rng.normal(0.0, 1.0, size=2)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction[0] = 1.0
        norm = 1.0
    direction /= norm
    game.player_x = int(np.clip(int(game.player_x) + round(float(sigma) * float(direction[0])), 0, 9))
    game.player_y = int(np.clip(int(game.player_y) + round(float(sigma) * float(direction[1])), 1, 8))
    game.terminal = False
    return int(game.player_x), int(game.player_y)


def rollout(env, *, seed: int, sigma: float, burn_in: int, horizon: int) -> CalibrationRow:
    player_x, player_y = checkpoint(env, seed=seed, burn_in=burn_in)
    perturbed_x, perturbed_y = perturb_player(env, seed=seed, sigma=sigma)
    reward = 0
    step = 0
    terminal = False
    for step in range(1, int(horizon) + 1):
        step_reward, terminal = env.act(controller_action(env))
        reward += int(step_reward)
        if terminal:
            break
    success = (not bool(env.env.terminal)) and step >= int(horizon) and reward >= 1
    return CalibrationRow(
        seed=int(seed),
        sigma=float(sigma),
        success=int(success),
        terminal=int(bool(env.env.terminal)),
        steps=int(step),
        reward=int(reward),
        checkpoint_player_x=player_x,
        checkpoint_player_y=player_y,
        perturbed_player_x=perturbed_x,
        perturbed_player_y=perturbed_y,
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
        "env_id": "minatar-asterix",
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
            "MinAtar Asterix calibration requires MinAtar in an isolated "
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
            "policy": "gold-seeking-with-one-step-enemy-avoidance",
            "perturbation": "seed-fixed-random-player-cell-displacement",
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
    parser.add_argument("--game", default="asterix")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--radii", default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--burn-in", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--sticky-action-prob", type=float, default=0.0)
    parser.add_argument("--difficulty-ramping", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.80)
    args = parser.parse_args()

    rows, summary = calibrate(args)
    write_outputs(args.out, rows, summary)
    print(
        "MinAtar Asterix calibration: "
        f"clean={summary['clean_success']:.4f} "
        f"range={summary['min_recovery']:.4f}--{summary['max_recovery']:.4f} "
        f"rauc={summary['rauc']:.4f} "
        f"r80={summary['r80']:.4f} "
        f"decision={summary['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
