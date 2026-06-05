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


ACTION_VECTORS = np.array(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [0.0, 0.0],
    ],
    dtype=float,
)
ACTION_VECTORS[:8] /= np.linalg.norm(ACTION_VECTORS[:8], axis=1, keepdims=True)


@dataclass(frozen=True, slots=True)
class PointMazeReplayState:
    row: int
    col: int
    x: float
    y: float
    graph_distance_to_goal: int


@dataclass(frozen=True, slots=True)
class PointMazeProbeResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    final_abs_r20: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


def package_versions() -> dict[str, str]:
    try:
        import gymnasium as gym
        import gymnasium_robotics
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "PointMaze probe requires gymnasium-robotics/gymnasium/mujoco. "
            "Install them in an isolated environment before running this internal diagnostic."
        ) from exc
    return {
        "gymnasium": gym.__version__,
        "gymnasium_robotics": gymnasium_robotics.__version__,
        "mujoco": mujoco.__version__,
    }


class OfficialPointMazeProbe:
    def __init__(
        self,
        *,
        seed: int,
        env_id: str,
        replay_state_count: int,
        learning_rate: float,
        discount: float,
        epsilon: float,
        q_init_blend: float,
        q_init_noise: float,
        max_steps: int,
        action_scale: float,
        perturb_cells: int,
        replay_distance_min: int,
        replay_distance_max: int,
        position_noise: float,
        success_radius: float,
    ) -> None:
        import gymnasium as gym
        import gymnasium_robotics  # noqa: F401

        self.rng = np.random.default_rng(seed + 421_000)
        self.env_id = str(env_id)
        self.env = gym.make(self.env_id, continuing_task=False, reset_target=False)
        obs, _info = self.env.reset(seed=seed)
        self.unwrapped = self.env.unwrapped
        self.maze = self.unwrapped.maze
        self.goal = np.array(obs["desired_goal"], dtype=float)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.epsilon = float(epsilon)
        self.max_steps = int(max_steps)
        self.action_scale = float(action_scale)
        self.perturb_cells = int(perturb_cells)
        self.replay_distance_min = int(replay_distance_min)
        self.replay_distance_max = int(replay_distance_max)
        self.position_noise = float(position_noise)
        self.success_radius = float(success_radius)
        self.free_cells = self._free_cells()
        self.cell_to_idx = {cell: idx for idx, cell in enumerate(self.free_cells)}
        self.idx_to_cell = {idx: cell for cell, idx in self.cell_to_idx.items()}
        self.cell_xy = {cell: np.array(self.maze.cell_rowcol_to_xy(cell), dtype=float) for cell in self.free_cells}
        self.goal_cell = min(self.free_cells, key=lambda cell: float(np.linalg.norm(self.cell_xy[cell] - self.goal)))
        self.goal_distances = self._graph_distances(self.goal_cell)
        self.q = self._heuristic_q(float(q_init_blend))
        self.q += self.rng.normal(0.0, float(q_init_noise), size=self.q.shape)
        self.states = self._select_replay_states(int(replay_state_count))
        if not self.states:
            raise ValueError("PointMaze probe selected no replay states")

    def _free_cells(self) -> tuple[tuple[int, int], ...]:
        cells: list[tuple[int, int]] = []
        for row, values in enumerate(self.maze.maze_map):
            for col, value in enumerate(values):
                if value == 0:
                    cells.append((int(row), int(col)))
        return tuple(cells)

    def _neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        row, col = cell
        candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        return [item for item in candidates if item in self.cell_to_idx]

    def _graph_distances(self, start: tuple[int, int]) -> dict[tuple[int, int], int]:
        distances = {start: 0}
        queue = [start]
        for cell in queue:
            for neighbor in self._neighbors(cell):
                if neighbor not in distances:
                    distances[neighbor] = distances[cell] + 1
                    queue.append(neighbor)
        return distances

    def _heuristic_q(self, blend: float) -> np.ndarray:
        q = np.zeros((len(self.free_cells), len(ACTION_VECTORS)), dtype=float)
        for idx, cell in enumerate(self.free_cells):
            pos = self.cell_xy[cell]
            current_dist = float(np.linalg.norm(pos - self.goal))
            for action_idx, action in enumerate(ACTION_VECTORS):
                next_pos = pos + self.action_scale * 0.35 * action
                next_dist = float(np.linalg.norm(next_pos - self.goal))
                q[idx, action_idx] = current_dist - next_dist
            best_neighbor = min([cell, *self._neighbors(cell)], key=lambda item: self.goal_distances.get(item, 999))
            direction = self.cell_xy[best_neighbor] - pos
            if float(np.linalg.norm(direction)) > 1e-9:
                direction = direction / np.linalg.norm(direction)
                for action_idx, action in enumerate(ACTION_VECTORS):
                    q[idx, action_idx] += float(np.dot(direction, action))
        return float(blend) * q

    def _select_replay_states(self, count: int) -> list[PointMazeReplayState]:
        reachable = [
            cell
            for cell in self.free_cells
            if self.replay_distance_min <= self.goal_distances.get(cell, 999) <= self.replay_distance_max
        ]
        if not reachable:
            reachable = [
                cell
                for cell in self.free_cells
                if 1 <= self.goal_distances.get(cell, 999) <= max(2, self.perturb_cells + 4)
            ]
        reachable.sort(key=lambda cell: (self.goal_distances[cell], cell[0], cell[1]))
        if count < len(reachable):
            indexes = np.linspace(0, len(reachable) - 1, count, dtype=int)
            reachable = [reachable[int(index)] for index in indexes]
        return [
            PointMazeReplayState(
                row=cell[0],
                col=cell[1],
                x=float(self.cell_xy[cell][0]),
                y=float(self.cell_xy[cell][1]),
                graph_distance_to_goal=int(self.goal_distances[cell]),
            )
            for cell in reachable
        ]

    def _cell_for_position(self, position: np.ndarray) -> tuple[int, int]:
        return min(self.free_cells, key=lambda cell: float(np.linalg.norm(self.cell_xy[cell] - position)))

    def _index_for_position(self, position: np.ndarray) -> int:
        return self.cell_to_idx[self._cell_for_position(position)]

    def _set_state(self, position: np.ndarray, velocity: np.ndarray | None = None) -> None:
        vel = np.zeros(2, dtype=float) if velocity is None else np.array(velocity, dtype=float)
        self.unwrapped.goal = self.goal.copy()
        self.unwrapped.point_env.set_state(np.array(position, dtype=float), vel)

    def _obs(self) -> np.ndarray:
        return np.array(self.unwrapped.point_env._get_obs()[0], dtype=float)

    def _success(self, position: np.ndarray) -> bool:
        return bool(np.linalg.norm(position - self.goal) <= self.success_radius)

    def clean_success(self, replay_idx: int) -> float:
        return self.success_prob(replay_idx, 0.0)

    def success_prob(self, replay_idx: int, sigma: float) -> float:
        starts = self.evaluation_starts(replay_idx, sigma)
        return float(np.mean([self.rollout_position(position, train=False, epsilon=False) for position in starts]))

    def evaluation_starts(self, replay_idx: int, sigma: float) -> list[np.ndarray]:
        replay = self.states[replay_idx]
        sigma = float(np.clip(sigma, 0.0, 1.0))
        if sigma == 0.0:
            return [np.array([replay.x, replay.y], dtype=float)]
        radius = max(1, int(round(sigma * self.perturb_cells)))
        starts = self._adverse_cells((replay.row, replay.col), radius)
        offsets = [
            np.array([0.0, 0.0]),
            np.array([self.position_noise, 0.0]),
            np.array([-self.position_noise, 0.0]),
            np.array([0.0, self.position_noise]),
            np.array([0.0, -self.position_noise]),
        ]
        return [self._clip_position(self.cell_xy[cell] + sigma * offset) for cell in starts for offset in offsets]

    def _adverse_cells(self, cell: tuple[int, int], radius: int) -> list[tuple[int, int]]:
        distances = self._graph_distances(cell)
        candidates = [
            item
            for item, dist in distances.items()
            if dist <= radius and self.goal_distances.get(item, 999) >= self.goal_distances.get(cell, 999)
        ]
        if not candidates:
            return [cell]
        max_dist = max(distances[item] for item in candidates)
        frontier = [item for item in candidates if distances[item] == max_dist]
        frontier.sort(key=lambda item: (-self.goal_distances.get(item, 999), item[0], item[1]))
        return frontier[:3]

    def sample_start(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
        replay = self.states[replay_idx]
        sigma = float(np.clip(sigma, 0.0, 1.0))
        if sigma == 0.0:
            base = np.array([replay.x, replay.y], dtype=float)
        else:
            radius = max(1, int(round(sigma * self.perturb_cells)))
            candidates = self._adverse_cells((replay.row, replay.col), radius)
            cell = candidates[int(rng.integers(len(candidates)))]
            base = self.cell_xy[cell]
        noise = rng.normal(0.0, self.position_noise * max(0.15, sigma), size=2)
        return self._clip_position(base + noise)

    def _clip_position(self, position: np.ndarray) -> np.ndarray:
        cell = self._cell_for_position(np.array(position, dtype=float))
        center = self.cell_xy[cell]
        return center + np.clip(np.array(position, dtype=float) - center, -0.38, 0.38)

    def rollout(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> bool:
        return self.rollout_position(self.sample_start(replay_idx, sigma, rng), train=False, epsilon=True, rng=rng)

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> None:
        self.rollout_position(self.sample_start(replay_idx, sigma, rng), train=True, epsilon=True, rng=rng)

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        position = self.sample_start(replay_idx, sigma, rng)
        state_idx = self._index_for_position(position)
        action_idx = self._action(position, rng, epsilon=True)
        before = float(self.q[state_idx, action_idx])
        next_position, reward, done = self._one_step(position, action_idx)
        target = reward if done else reward + self.discount * float(np.max(self.q[self._index_for_position(next_position)]))
        return abs(target - before)

    def rollout_position(
        self,
        position: np.ndarray,
        *,
        train: bool,
        epsilon: bool,
        rng: np.random.Generator | None = None,
    ) -> bool:
        self._set_state(position)
        obs = self._obs()
        for _ in range(self.max_steps):
            position_now = obs[:2]
            state_idx = self._index_for_position(position_now)
            action_idx = self._action(position_now, rng, epsilon=epsilon)
            action = (self.action_scale * ACTION_VECTORS[action_idx]).astype(np.float32)
            next_obs, _reward, _terminated, _truncated, info = self.env.step(action)
            obs = np.array(next_obs["observation"], dtype=float)
            success = bool(info.get("success", False)) or self._success(obs[:2])
            reward = 1.0 if success else -0.01
            if train:
                target = reward if success else reward + self.discount * float(np.max(self.q[self._index_for_position(obs[:2])]))
                self.q[state_idx, action_idx] += self.learning_rate * (target - float(self.q[state_idx, action_idx]))
            if success:
                return True
        return False

    def _one_step(self, position: np.ndarray, action_idx: int) -> tuple[np.ndarray, float, bool]:
        self._set_state(position)
        action = (self.action_scale * ACTION_VECTORS[action_idx]).astype(np.float32)
        obs, _reward, _terminated, _truncated, info = self.env.step(action)
        next_position = np.array(obs["observation"][:2], dtype=float)
        success = bool(info.get("success", False)) or self._success(next_position)
        return next_position, 1.0 if success else -0.01, success

    def _action(self, position: np.ndarray, rng: np.random.Generator | None, *, epsilon: bool) -> int:
        if epsilon and rng is not None and rng.random() < self.epsilon:
            return int(rng.integers(len(ACTION_VECTORS)))
        return int(np.argmax(self.q[self._index_for_position(position)]))


def run_method(args: argparse.Namespace, method: str, seed: int) -> PointMazeProbeResult:
    rng = np.random.default_rng(seed + 423_000)
    bench = OfficialPointMazeProbe(
        seed=seed,
        env_id=args.env_id,
        replay_state_count=args.replay_states,
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        q_init_blend=args.q_init_blend,
        q_init_noise=args.q_init_noise,
        max_steps=args.max_steps,
        action_scale=args.action_scale,
        perturb_cells=args.perturb_cells,
        replay_distance_min=args.replay_distance_min,
        replay_distance_max=args.replay_distance_max,
        position_noise=args.position_noise,
        success_radius=args.success_radius,
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
            metrics = evaluate(bench, args)
            metrics["step"] = float(step)
            history.append(metrics)
            if method.startswith("bgr"):
                refresh_records(bench, records, rng, args, step)
        if step == args.iterations:
            break
        for _ in range(args.train_batch_size):
            replay_idx, sigma = sample_training_pair(method, bench, records, scorer, rng, args, step)
            bench.train_step(replay_idx, sigma, rng)
            if method.startswith("bgr"):
                records[replay_idx].add_observation(sigma, bench.rollout(replay_idx, sigma, rng))
                records[replay_idx].replay_count += 1

    xs = np.array([row["step"] for row in history], dtype=float)
    ys = np.array([row["rauc"] for row in history], dtype=float)
    final = history[-1]
    return PointMazeProbeResult(
        method=method,
        seed=seed,
        final_clean=final["clean"],
        final_rauc=final["rauc"],
        final_median_r80=final["median_r80"],
        final_abs_r20=final["abs_r20"],
        rauc_aulc=float(np.trapezoid(ys, xs) / (xs[-1] - xs[0])) if xs[-1] > xs[0] else final["rauc"],
        best_rauc=max(row["rauc"] for row in history),
        history=history,
    )


def init_records(bench: OfficialPointMazeProbe, rng: np.random.Generator, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, replay in enumerate(bench.states):
        record = LevelRecord(
            id=f"pointmaze_{replay.row}_{replay.col}",
            domain="official_pointmaze_recovery",
            task_id=bench.env_id,
            clean_success_hat=bench.clean_success(replay_idx),
            feasibility_hat=1.0,
            perturbation_family="official_pointmaze_graph_distance_restart",
            sigma_grid=args.initial_probes,
        )
        estimator = IsotonicCurveEstimator(1.0, args.alpha)
        for sigma in args.initial_probes:
            for _ in range(args.min_trials):
                success = bench.rollout(replay_idx, sigma, rng)
                record.add_observation(sigma, success)
                estimator.update_bernoulli(sigma, success)
        write_estimate(record, estimator.fit())
        records.append(record)
    return records


def refresh_records(
    bench: OfficialPointMazeProbe,
    records: list[LevelRecord],
    rng: np.random.Generator,
    args: argparse.Namespace,
    step: int,
) -> None:
    scores = np.array(
        [1.0 + record.uncertainty_hat + 0.002 * (step - record.last_evaluated_step) for record in records],
        dtype=float,
    )
    probs = scores / np.sum(scores)
    count = min(args.refresh_per_eval, len(records))
    for replay_idx in rng.choice(len(records), size=count, replace=False, p=probs):
        record = records[int(replay_idx)]
        estimator = IsotonicCurveEstimator(1.0, args.alpha)
        for sigma, trials in record.trials.items():
            estimator.update(sigma, record.successes.get(sigma, 0), trials)
        sigma = estimator.next_probe(rng, jitter=args.refresh_jitter)
        for _ in range(args.refresh_trials):
            success = bench.rollout(int(replay_idx), sigma, rng)
            record.add_observation(sigma, success)
            estimator.update_bernoulli(sigma, success)
        write_estimate(record, estimator.fit())
        record.clean_success_hat = bench.clean_success(int(replay_idx))
        record.feasibility_hat = 1.0
        record.last_evaluated_step = step


def write_estimate(record: LevelRecord, estimate: Any) -> None:
    record.r_alpha_hat = estimate.r_alpha
    record.sharpness_hat = estimate.sharpness
    record.uncertainty_hat = estimate.r_uncertainty
    record.recovery_curve_hat = estimate.recovery.tolist()


def evaluate(bench: OfficialPointMazeProbe, args: argparse.Namespace) -> dict[str, float]:
    grid = np.linspace(0.0, 1.0, args.eval_grid_size)
    clean: list[float] = []
    raucs: list[float] = []
    radii: list[float] = []
    abs_r20: list[float] = []
    for replay_idx in range(len(bench.states)):
        curve = np.array([bench.success_prob(replay_idx, sigma) for sigma in grid], dtype=float)
        clean.append(float(curve[0]))
        raucs.append(recovery_auc(grid, curve, sigma_max=1.0))
        radii.append(critical_radius(grid, curve, alpha=args.alpha))
        abs_r20.append(critical_radius(grid, curve, alpha=args.absolute_radius_alpha, relative_to_clean=False))
    return {
        "clean": float(np.mean(clean)),
        "rauc": float(np.mean(raucs)),
        "median_r80": float(np.median(radii)),
        "abs_r20": float(np.median(abs_r20)),
    }


def sample_training_pair(
    method: str,
    bench: OfficialPointMazeProbe,
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
        scores = [1.0 - bench.success_prob(int(candidate), float(sigma)) for candidate, sigma in zip(candidates, sigmas, strict=True)]
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
    parser = argparse.ArgumentParser(description="Run an internal official PointMaze recovery replay diagnostic.")
    parser.add_argument("--out", default="runs/pointmaze_recovery_probe")
    parser.add_argument("--env-id", default="PointMaze_Medium-v3")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=10)
    parser.add_argument("--replay-states", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--eval-grid-size", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--discount", type=float, default=0.97)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--q-init-blend", type=float, default=0.45)
    parser.add_argument("--q-init-noise", type=float, default=0.10)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--perturb-cells", type=int, default=5)
    parser.add_argument("--replay-distance-min", type=int, default=1)
    parser.add_argument("--replay-distance-max", type=int, default=9)
    parser.add_argument("--position-noise", type=float, default=0.12)
    parser.add_argument("--success-radius", type=float, default=0.45)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--absolute-radius-alpha", type=float, default=0.20)
    parser.add_argument("--target-radius", type=float, default=0.45)
    parser.add_argument("--radius-bandwidth", type=float, default=0.35)
    parser.add_argument("--fixed-radius", type=float, default=0.65)
    parser.add_argument("--baseline-candidates", type=int, default=12)
    parser.add_argument("--initial-probes", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--min-trials", type=int, default=1)
    parser.add_argument("--refresh-per-eval", type=int, default=12)
    parser.add_argument("--refresh-trials", type=int, default=1)
    parser.add_argument("--refresh-jitter", type=float, default=0.08)
    parser.add_argument("--radius-noise", type=float, default=0.08)
    parser.add_argument("--radius-uniform-mix", type=float, default=0.25)
    parser.add_argument("--priority-temperature", type=float, default=0.8)
    parser.add_argument("--uniform-mix", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    methods = parse_csv_strings(args.methods)
    seeds = parse_csv_ints(args.seeds)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    versions = package_versions()
    rows: list[dict[str, float | int | str]] = []
    results: list[dict] = []
    for method in methods:
        for seed in seeds:
            start = time.perf_counter()
            result = run_method(args, method, seed)
            elapsed = time.perf_counter() - start
            print(
                f"[done] method={method} seed={seed} rauc={result.final_rauc:.4f} "
                f"clean={result.final_clean:.4f} aulc={result.rauc_aulc:.4f} elapsed={elapsed:.2f}s",
                flush=True,
            )
            results.append(asdict(result))
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "final_clean": result.final_clean,
                    "final_rauc": result.final_rauc,
                    "final_median_r80": result.final_median_r80,
                    "final_abs_r20": result.final_abs_r20,
                    "rauc_aulc": result.rauc_aulc,
                    "best_rauc": result.best_rauc,
                }
            )

    (out_dir / "package_versions.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps({"args": vars(args), "package_versions": versions, "results": results}, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"package_versions={versions}")
    print(summary(rows))


def summary(rows: list[dict[str, float | int | str]]) -> str:
    by_method: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    lines = ["method,final_clean_mean,final_rauc_mean,final_median_r80_mean,final_abs_r20_mean,rauc_aulc_mean,best_rauc_mean"]
    for method, method_rows in sorted(by_method.items()):
        lines.append(
            ",".join(
                [
                    method,
                    f"{np.mean([float(row['final_clean']) for row in method_rows]):.4f}",
                    f"{np.mean([float(row['final_rauc']) for row in method_rows]):.4f}",
                    f"{np.mean([float(row['final_median_r80']) for row in method_rows]):.4f}",
                    f"{np.mean([float(row['final_abs_r20']) for row in method_rows]):.4f}",
                    f"{np.mean([float(row['rauc_aulc']) for row in method_rows]):.4f}",
                    f"{np.mean([float(row['best_rauc']) for row in method_rows]):.4f}",
                ]
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
