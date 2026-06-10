#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bgr.curve_estimators import IsotonicCurveEstimator
from bgr.metrics import critical_radius, recovery_auc
from bgr.priorities import BGRPriorityScorer
from bgr.records import LevelRecord
from bgr.samplers import mixed_priority_probs, sample_boundary_radius


MOVE_ACTIONS = (0, 1, 2)
DIR_TO_VEC = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}
DIRS = tuple(DIR_TO_VEC)
NEIGHBORS = tuple(DIR_TO_VEC.values())


@dataclass(frozen=True, slots=True)
class ReplayState:
    x: int
    y: int
    direction: int


@dataclass(frozen=True, slots=True)
class ProbeResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    final_abs_r10: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


def package_versions() -> dict[str, str]:
    try:
        import gymnasium as gym
        import minigrid
    except ImportError as exc:
        raise SystemExit(
            "MiniGrid DynamicObstacles probe requires minigrid/gymnasium in an isolated environment."
        ) from exc
    return {"gymnasium": gym.__version__, "minigrid": minigrid.__version__}


class DynamicObstaclesProbe:
    def __init__(
        self,
        *,
        env_id: str,
        seed: int,
        replay_state_count: int,
        max_radius: int,
        rollout_horizon: int,
        learning_rate: float,
        epsilon: float,
        q_init_blend: float,
        q_init_noise: float,
    ) -> None:
        import gymnasium as gym

        self.env_id = env_id
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed + 557_000)
        self.env = gym.make(env_id)
        self.env.reset(seed=seed)
        self.unwrapped = self.env.unwrapped
        self.width = int(self.unwrapped.width)
        self.height = int(self.unwrapped.height)
        self.base_grid = self.unwrapped.grid.copy()
        self.max_radius = int(max_radius)
        self.rollout_horizon = int(rollout_horizon)
        self.learning_rate = float(learning_rate)
        self.epsilon = float(epsilon)
        self.q_init_blend = float(q_init_blend)
        self.q_init_noise = float(q_init_noise)
        self.q_table: dict[tuple[Any, ...], np.ndarray] = {}
        self.free_cells = self._free_cells()
        self.goal_cell = self._goal_cell()
        self.states = self._select_replay_states(int(replay_state_count))
        self.perturbation_cache = self._build_perturbation_cache()

    def _free_cells(self) -> tuple[tuple[int, int], ...]:
        cells: list[tuple[int, int]] = []
        for x in range(self.width):
            for y in range(self.height):
                obj = self.base_grid.get(x, y)
                if obj is None or obj.can_overlap():
                    cells.append((x, y))
        return tuple(cells)

    def _goal_cell(self) -> tuple[int, int]:
        for x, y in self.free_cells:
            obj = self.base_grid.get(x, y)
            if obj is not None and obj.type == "goal":
                return (x, y)
        raise ValueError(f"{self.env_id} has no goal cell")

    def _obstacle_cells(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for x in range(self.width):
            for y in range(self.height):
                obj = self.unwrapped.grid.get(x, y)
                if obj is not None and obj.type == "ball":
                    cells.add((x, y))
        return cells

    def _base_obstacle_cells(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for x in range(self.width):
            for y in range(self.height):
                obj = self.base_grid.get(x, y)
                if obj is not None and obj.type == "ball":
                    cells.add((x, y))
        return cells

    def _select_replay_states(self, count: int) -> list[ReplayState]:
        path = self._shortest_path((1, 1), self.goal_cell, self._base_obstacle_cells()) or []
        path_cells = set(path)
        candidates: list[tuple[int, int, ReplayState]] = []
        obstacle_cells = self._base_obstacle_cells()
        for x, y in self.free_cells:
            if (x, y) == self.goal_cell or (x, y) in obstacle_cells:
                continue
            dist_to_path = min((abs(x - px) + abs(y - py) for px, py in path_cells), default=0)
            dist_to_goal = abs(x - self.goal_cell[0]) + abs(y - self.goal_cell[1])
            if dist_to_path <= 2:
                for direction in DIRS:
                    candidates.append((dist_to_goal, dist_to_path, ReplayState(x, y, direction)))
        candidates.sort(key=lambda item: (item[0], item[1], item[2].direction, item[2].x, item[2].y))
        if count >= len(candidates):
            return [item[2] for item in candidates]
        indexes = np.linspace(0, len(candidates) - 1, count, dtype=int)
        return [candidates[int(index)][2] for index in indexes]

    def _build_perturbation_cache(self) -> tuple[tuple[ReplayState, ...], ...]:
        cache: list[tuple[ReplayState, ...]] = []
        base_obstacles = self._base_obstacle_cells()
        free = [cell for cell in self.free_cells if cell != self.goal_cell and cell not in base_obstacles]
        for replay in self.states:
            states: list[ReplayState] = []
            for x, y in free:
                if abs(x - replay.x) + abs(y - replay.y) <= self.max_radius:
                    states.append(ReplayState(x, y, replay.direction))
            cache.append(tuple(states or [replay]))
        return tuple(cache)

    def perturbation_states(self, replay_idx: int, sigma: float) -> tuple[ReplayState, ...]:
        target_radius = int(round(float(np.clip(sigma, 0.0, 1.0)) * self.max_radius))
        replay = self.states[replay_idx]
        selected = [
            state
            for state in self.perturbation_cache[replay_idx]
            if abs(state.x - replay.x) + abs(state.y - replay.y) == target_radius
        ]
        return tuple(selected or [replay])

    def restore_state(self, replay: ReplayState, rollout_seed: int) -> None:
        self.env.reset(seed=int(rollout_seed))
        self.unwrapped.grid = self.base_grid.copy()
        obstacles = []
        for x in range(self.width):
            for y in range(self.height):
                obj = self.unwrapped.grid.get(x, y)
                if obj is not None and obj.type == "ball":
                    obj.cur_pos = (x, y)
                    obstacles.append(obj)
        self.unwrapped.obstacles = obstacles
        self.unwrapped.agent_pos = (int(replay.x), int(replay.y))
        self.unwrapped.agent_dir = int(replay.direction)
        self.unwrapped.step_count = 0
        self.unwrapped.carrying = None

    def state_key(self) -> tuple[Any, ...]:
        obstacles = tuple(sorted(self._obstacle_cells()))
        x, y = self.unwrapped.agent_pos
        return (int(x), int(y), int(self.unwrapped.agent_dir), obstacles)

    def q_values(self) -> np.ndarray:
        key = self.state_key()
        if key not in self.q_table:
            q = self.rng.normal(0.0, self.q_init_noise, size=len(MOVE_ACTIONS))
            expert = self.expert_action()
            q[expert] += self.q_init_blend
            self.q_table[key] = q
        return self.q_table[key]

    def expert_action(self) -> int:
        x, y = self.unwrapped.agent_pos
        path = self._shortest_path((int(x), int(y)), self.goal_cell, self._obstacle_cells())
        if len(path) < 2:
            return 0
        nx, ny = path[1]
        dx, dy = nx - int(x), ny - int(y)
        target_dir = next(direction for direction, vec in DIR_TO_VEC.items() if vec == (dx, dy))
        current = int(self.unwrapped.agent_dir)
        if target_dir == current:
            return 2
        if (target_dir - current) % 4 == 1:
            return 1
        return 0

    def _shortest_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        obstacles: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        frontier = [start]
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while frontier:
            x, y = frontier.pop(0)
            if (x, y) == goal:
                path = [(x, y)]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])  # type: ignore[arg-type]
                return list(reversed(path))
            for dx, dy in NEIGHBORS:
                nx, ny = x + dx, y + dy
                if (nx, ny) in parent or (nx, ny) in obstacles:
                    continue
                obj = self.unwrapped.grid.get(nx, ny) if hasattr(self, "unwrapped") else self.base_grid.get(nx, ny)
                if obj is not None and obj.type not in {"goal"} and not obj.can_overlap():
                    continue
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                parent[(nx, ny)] = (x, y)
                frontier.append((nx, ny))
        return []

    def rollout(self, replay_idx: int, sigma: float, rng: np.random.Generator, *, train: bool = False) -> bool:
        replay = rng.choice(self.perturbation_states(replay_idx, sigma))
        rollout_seed = int(rng.integers(0, 2**31 - 1))
        self.restore_state(replay, rollout_seed)
        for _ in range(self.rollout_horizon):
            q = self.q_values()
            if train:
                expert = self.expert_action()
                target = np.zeros_like(q)
                target[expert] = 1.0
                q += self.learning_rate * (target - q)
            if (not train) and rng.random() < self.epsilon:
                action_idx = int(rng.integers(len(MOVE_ACTIONS)))
            else:
                action_idx = int(np.argmax(q))
            _obs, reward, terminated, truncated, _info = self.env.step(int(MOVE_ACTIONS[action_idx]))
            if terminated or truncated:
                return bool(reward > 0)
        return False

    def success_prob(self, replay_idx: int, sigma: float, rng: np.random.Generator, trials: int) -> float:
        return float(np.mean([self.rollout(replay_idx, sigma, rng, train=False) for _ in range(trials)]))

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> None:
        self.rollout(replay_idx, sigma, rng, train=True)

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        replay = rng.choice(self.perturbation_states(replay_idx, sigma))
        self.restore_state(replay, int(rng.integers(0, 2**31 - 1)))
        q = self.q_values()
        return float(1.0 - q[self.expert_action()])


def run_method(args: argparse.Namespace, method: str, seed: int) -> ProbeResult:
    rng = np.random.default_rng(seed + 559_000)
    bench = DynamicObstaclesProbe(
        env_id=args.env_id,
        seed=seed,
        replay_state_count=args.replay_states,
        max_radius=args.max_radius,
        rollout_horizon=args.rollout_horizon,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        q_init_blend=args.q_init_blend,
        q_init_noise=args.q_init_noise,
    )
    records = init_records(bench, rng, args)
    scorer = BGRPriorityScorer(
        clean_threshold=0.0,
        feasibility_threshold=0.0,
        target_radius=args.target_radius,
        radius_bandwidth=args.radius_bandwidth,
    )
    history: list[dict[str, float]] = []
    for step in range(args.iterations + 1):
        if step % args.eval_every == 0:
            metrics = evaluate(bench, rng, args)
            metrics["step"] = float(step)
            history.append(metrics)
            if method.startswith("bgr"):
                refresh_records(bench, records, rng, args, step)
        if step == args.iterations:
            break
        for _ in range(args.train_batch_size):
            replay_idx, sigma = sample_training_pair(method, bench, records, scorer, rng, args, step)
            if method == "bgr_clean_shield" and bench.success_prob(replay_idx, 0.0, rng, args.eval_trials) < args.clean_shield_threshold:
                sigma = 0.0
            bench.train_step(replay_idx, sigma, rng)
            if (
                method == "bgr_clean_shield"
                and sigma > 0.0
                and rng.random() < args.clean_shield_anchor_mix
            ):
                bench.train_step(replay_idx, 0.0, rng)
            if method.startswith("bgr"):
                success = bench.rollout(replay_idx, sigma, rng, train=False)
                records[replay_idx].add_observation(sigma, success)
                records[replay_idx].replay_count += 1
    xs = np.array([row["step"] for row in history], dtype=float)
    ys = np.array([row["rauc"] for row in history], dtype=float)
    final = history[-1]
    return ProbeResult(
        method=method,
        seed=seed,
        final_clean=final["clean"],
        final_rauc=final["rauc"],
        final_median_r80=final["median_r80"],
        final_abs_r10=final["abs_r10"],
        rauc_aulc=float(np.trapezoid(ys, xs) / (xs[-1] - xs[0])) if xs[-1] > xs[0] else final["rauc"],
        best_rauc=max(row["rauc"] for row in history),
        history=history,
    )


def init_records(bench: DynamicObstaclesProbe, rng: np.random.Generator, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, replay in enumerate(bench.states):
        record = LevelRecord(
            id=f"minigrid_dynamic_obstacles_{replay.x}_{replay.y}_{replay.direction}",
            domain="minigrid_dynamic_obstacles_recovery",
            task_id=bench.env_id,
            clean_success_hat=bench.success_prob(replay_idx, 0.0, rng, args.eval_trials),
            feasibility_hat=1.0,
            perturbation_family="official_minigrid_dynamic_obstacle_position_restart",
            sigma_grid=args.initial_probes,
        )
        estimator = IsotonicCurveEstimator(1.0, args.alpha)
        for sigma in args.initial_probes:
            for _ in range(args.min_trials):
                success = bench.rollout(replay_idx, sigma, rng, train=False)
                record.add_observation(sigma, success)
                estimator.update_bernoulli(sigma, success)
        write_estimate(record, estimator.fit())
        records.append(record)
    return records


def refresh_records(
    bench: DynamicObstaclesProbe,
    records: list[LevelRecord],
    rng: np.random.Generator,
    args: argparse.Namespace,
    step: int,
) -> None:
    scores = np.array([1.0 + record.uncertainty_hat + 0.002 * (step - record.last_evaluated_step) for record in records])
    probs = scores / np.sum(scores)
    for replay_idx in rng.choice(len(records), size=min(args.refresh_per_eval, len(records)), replace=False, p=probs):
        record = records[int(replay_idx)]
        estimator = IsotonicCurveEstimator(1.0, args.alpha)
        for sigma, trials in record.trials.items():
            estimator.update(sigma, record.successes.get(sigma, 0), trials)
        sigma = estimator.next_probe(rng, jitter=args.refresh_jitter)
        for _ in range(args.refresh_trials):
            success = bench.rollout(int(replay_idx), sigma, rng, train=False)
            record.add_observation(sigma, success)
            estimator.update_bernoulli(sigma, success)
        write_estimate(record, estimator.fit())
        record.clean_success_hat = bench.success_prob(int(replay_idx), 0.0, rng, args.eval_trials)
        record.last_evaluated_step = step


def write_estimate(record: LevelRecord, estimate: Any) -> None:
    record.r_alpha_hat = estimate.r_alpha
    record.sharpness_hat = estimate.sharpness
    record.uncertainty_hat = estimate.r_uncertainty
    record.recovery_curve_hat = estimate.recovery.tolist()


def evaluate(bench: DynamicObstaclesProbe, rng: np.random.Generator, args: argparse.Namespace) -> dict[str, float]:
    grid = np.linspace(0.0, 1.0, args.eval_grid_size)
    clean: list[float] = []
    raucs: list[float] = []
    radii: list[float] = []
    abs_r10: list[float] = []
    for replay_idx in range(len(bench.states)):
        curve = np.array([bench.success_prob(replay_idx, sigma, rng, args.eval_trials) for sigma in grid], dtype=float)
        clean.append(float(curve[0]))
        raucs.append(recovery_auc(grid, curve, sigma_max=1.0))
        radii.append(critical_radius(grid, curve, alpha=args.alpha))
        abs_r10.append(critical_radius(grid, curve, alpha=args.absolute_radius_alpha, relative_to_clean=False))
    return {
        "clean": float(np.mean(clean)),
        "rauc": float(np.mean(raucs)),
        "median_r80": float(np.median(radii)),
        "abs_r10": float(np.median(abs_r10)),
    }


def sample_training_pair(
    method: str,
    bench: DynamicObstaclesProbe,
    records: list[LevelRecord],
    scorer: BGRPriorityScorer,
    rng: np.random.Generator,
    args: argparse.Namespace,
    step: int,
) -> tuple[int, float]:
    if method == "uniform":
        return int(rng.integers(len(records))), float(rng.uniform(0.0, 1.0))
    if method == "fixed":
        return int(rng.integers(len(records))), args.fixed_radius
    if method == "failure_only":
        candidates = rng.choice(len(records), size=min(args.baseline_candidates, len(records)), replace=False)
        sigmas = rng.uniform(0.0, 1.0, size=len(candidates))
        scores = [1.0 - bench.success_prob(int(candidate), float(sigma), rng, args.eval_trials) for candidate, sigma in zip(candidates, sigmas, strict=True)]
        idx = int(np.argmax(scores))
        return int(candidates[idx]), float(sigmas[idx])
    if method == "td_loss":
        candidates = rng.choice(len(records), size=min(args.baseline_candidates, len(records)), replace=False)
        sigmas = rng.uniform(0.0, 1.0, size=len(candidates))
        scores = [bench.loss_proxy(int(candidate), float(sigma), rng) for candidate, sigma in zip(candidates, sigmas, strict=True)]
        idx = int(np.argmax(scores))
        return int(candidates[idx]), float(sigmas[idx])
    if method.startswith("bgr"):
        priorities = np.array([scorer.score(record, step) for record in records], dtype=float)
        probs = mixed_priority_probs(priorities, temperature=args.priority_temperature, uniform_mix=args.uniform_mix)
        replay_idx = int(rng.choice(len(records), p=probs))
        if method == "bgr_uniform_radius":
            sigma = float(rng.uniform(0.0, 1.0))
        elif method == "bgr_coverage" and rng.random() < args.radius_uniform_mix:
            sigma = float(rng.uniform(0.0, 1.0))
        else:
            sigma = sample_boundary_radius(rng, records[replay_idx].r_alpha_hat, 1.0, radius_noise=args.radius_noise)
        return replay_idx, sigma
    raise ValueError(f"unknown method: {method}")


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an official MiniGrid DynamicObstacles recovery replay scout.")
    parser.add_argument("--out", default="runs/minigrid_dynamic_obstacles_recovery_probe")
    parser.add_argument("--env-id", default="MiniGrid-Dynamic-Obstacles-6x6-v0")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--replay-states", type=int, default=24)
    parser.add_argument("--max-radius", type=int, default=3)
    parser.add_argument("--rollout-horizon", type=int, default=80)
    parser.add_argument("--eval-grid-size", type=int, default=7)
    parser.add_argument("--eval-trials", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--q-init-blend", type=float, default=0.02)
    parser.add_argument("--q-init-noise", type=float, default=0.06)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--absolute-radius-alpha", type=float, default=0.10)
    parser.add_argument("--target-radius", type=float, default=0.50)
    parser.add_argument("--radius-bandwidth", type=float, default=0.35)
    parser.add_argument("--fixed-radius", type=float, default=0.50)
    parser.add_argument("--baseline-candidates", type=int, default=10)
    parser.add_argument("--initial-probes", type=float, nargs="+", default=[0.0, 0.33, 0.67, 1.0])
    parser.add_argument("--min-trials", type=int, default=1)
    parser.add_argument("--refresh-per-eval", type=int, default=8)
    parser.add_argument("--refresh-trials", type=int, default=1)
    parser.add_argument("--refresh-jitter", type=float, default=0.08)
    parser.add_argument("--radius-noise", type=float, default=0.08)
    parser.add_argument("--radius-uniform-mix", type=float, default=0.25)
    parser.add_argument("--priority-temperature", type=float, default=0.8)
    parser.add_argument("--uniform-mix", type=float, default=0.10)
    parser.add_argument("--clean-shield-threshold", type=float, default=0.65)
    parser.add_argument("--clean-shield-anchor-mix", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "package_versions.json").write_text(json.dumps(package_versions(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    methods = parse_csv_strings(args.methods)
    seeds = parse_csv_ints(args.seeds)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for method in methods:
            start = time.time()
            result = run_method(args, method, seed)
            row = asdict(result)
            row.pop("history")
            row["runtime_sec"] = time.time() - start
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "method",
            "seed",
            "final_clean",
            "final_rauc",
            "final_median_r80",
            "final_abs_r10",
            "rauc_aulc",
            "best_rauc",
            "runtime_sec",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
