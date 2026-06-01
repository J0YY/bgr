from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib

import numpy as np


ACTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True, slots=True)
class GridTask:
    task_id: int
    walls: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
    path: tuple[tuple[int, int], ...]
    distances: np.ndarray
    task_hash: str


@dataclass(frozen=True, slots=True)
class GridReplayState:
    id: int
    task_id: int
    position: tuple[int, int]


@dataclass(slots=True)
class GridMarginState:
    replay: GridReplayState
    feasible_radius: float
    margin: float
    clean_success: float
    temperature: float
    loss_bias: float


class TabularRecoveryPolicy:
    """Small policy used to test replay curricula in a procedural grid domain."""

    def __init__(self, learning_rate: float = 1.0, init_logit: float = 0.0) -> None:
        self.learning_rate = float(learning_rate)
        self.init_logit = float(init_logit)
        self.logits: dict[tuple[str, tuple[int, int]], np.ndarray] = {}

    def action_probs(self, task: GridTask, position: tuple[int, int]) -> np.ndarray:
        logits = self.logits.get((task.task_hash, position))
        if logits is None:
            logits = np.full(len(ACTIONS), self.init_logit, dtype=float)
        masked = logits.copy()
        for action_idx, (dr, dc) in enumerate(ACTIONS):
            nr, nc = position[0] + dr, position[1] + dc
            if not _in_bounds(task.walls, (nr, nc)) or task.walls[nr, nc]:
                masked[action_idx] = -20.0
        shifted = masked - np.max(masked)
        probs = np.exp(shifted)
        total = np.sum(probs)
        if total <= 0:
            return np.ones(len(ACTIONS), dtype=float) / len(ACTIONS)
        return probs / total

    def sample_action(self, task: GridTask, position: tuple[int, int], rng: np.random.Generator) -> int:
        return int(rng.choice(len(ACTIONS), p=self.action_probs(task, position)))

    def train_oracle_step(self, task: GridTask, position: tuple[int, int]) -> float:
        oracle = oracle_action(task, position)
        if oracle is None:
            return 0.0
        key = (task.task_hash, position)
        logits = self.logits.setdefault(key, np.zeros(len(ACTIONS), dtype=float))
        probs = self.action_probs(task, position)
        before = float(probs[oracle])
        logits[oracle] += self.learning_rate
        for action_idx in range(len(ACTIONS)):
            if action_idx != oracle:
                logits[action_idx] -= 0.2 * self.learning_rate / (len(ACTIONS) - 1)
        after = float(self.action_probs(task, position)[oracle])
        return after - before

    def loss_proxy(self, task: GridTask, position: tuple[int, int]) -> float:
        oracle = oracle_action(task, position)
        if oracle is None:
            return 0.0
        return float(1.0 - self.action_probs(task, position)[oracle])


class GridRecoveryBenchmark:
    """Procedural replayable-state benchmark with exact feasibility witness."""

    def __init__(
        self,
        num_tasks: int,
        grid_size: int,
        obstacle_prob: float,
        replay_states_per_task: int,
        max_offset: int,
        horizon: int,
        learning_rate: float,
        seed: int,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.max_offset = int(max_offset)
        self.horizon = int(horizon)
        self.tasks = self._make_tasks(num_tasks, grid_size, obstacle_prob)
        self.replay_states = self._make_replay_states(replay_states_per_task)
        self.policy = TabularRecoveryPolicy(learning_rate=learning_rate)

    def perturb_position(
        self,
        replay: GridReplayState,
        sigma: float,
        rng: np.random.Generator,
    ) -> tuple[int, int] | None:
        task = self.tasks[replay.task_id]
        radius = max(0, int(round(np.clip(sigma, 0.0, 1.0) * self.max_offset)))
        candidates = []
        for row in range(task.walls.shape[0]):
            for col in range(task.walls.shape[1]):
                pos = (row, col)
                if task.walls[pos]:
                    continue
                dist = abs(row - replay.position[0]) + abs(col - replay.position[1])
                if dist <= radius and np.isfinite(task.distances[pos]):
                    candidates.append(pos)
        if not candidates:
            return None
        if radius == 0:
            return replay.position
        weights = np.array(
            [1.0 + abs(pos[0] - replay.position[0]) + abs(pos[1] - replay.position[1]) for pos in candidates],
            dtype=float,
        )
        weights /= np.sum(weights)
        return candidates[int(rng.choice(len(candidates), p=weights))]

    def feasibility(self, replay_idx: int, sigma: float, trials: int = 16) -> float:
        replay = self.replay_states[replay_idx]
        rng = np.random.default_rng(50_000 + 997 * replay_idx + int(1000 * sigma))
        feasible = 0
        for _ in range(trials):
            pos = self.perturb_position(replay, sigma, rng)
            if pos is not None and self.tasks[replay.task_id].distances[pos] <= self.horizon:
                feasible += 1
        return feasible / max(1, trials)

    def rollout(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> bool:
        replay = self.replay_states[replay_idx]
        task = self.tasks[replay.task_id]
        pos = self.perturb_position(replay, sigma, rng)
        if pos is None:
            return False
        for _ in range(self.horizon):
            if pos == task.goal:
                return True
            action_idx = self.policy.sample_action(task, pos, rng)
            dr, dc = ACTIONS[action_idx]
            nxt = (pos[0] + dr, pos[1] + dc)
            if _in_bounds(task.walls, nxt) and not task.walls[nxt]:
                pos = nxt
        return pos == task.goal

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        replay = self.replay_states[replay_idx]
        task = self.tasks[replay.task_id]
        pos = self.perturb_position(replay, sigma, rng)
        if pos is None:
            return 0.0
        gain = 0.0
        for _ in range(min(4, self.horizon)):
            if pos == task.goal:
                break
            gain += self.policy.train_oracle_step(task, pos)
            action_idx = oracle_action(task, pos)
            if action_idx is None:
                break
            dr, dc = ACTIONS[action_idx]
            pos = (pos[0] + dr, pos[1] + dc)
        return gain

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        replay = self.replay_states[replay_idx]
        pos = self.perturb_position(replay, sigma, rng)
        if pos is None:
            return 0.0
        return self.policy.loss_proxy(self.tasks[replay.task_id], pos)

    def _make_tasks(self, num_tasks: int, grid_size: int, obstacle_prob: float) -> list[GridTask]:
        tasks: list[GridTask] = []
        attempts = 0
        while len(tasks) < num_tasks and attempts < 5000:
            attempts += 1
            walls = self.rng.random((grid_size, grid_size)) < obstacle_prob
            start = (0, 0)
            goal = (grid_size - 1, grid_size - 1)
            walls[start] = False
            walls[goal] = False
            distances = shortest_path_distances(walls, goal)
            if not np.isfinite(distances[start]) or distances[start] < grid_size:
                continue
            path = shortest_path(walls, start, goal, distances)
            if len(path) < 6:
                continue
            task_hash = _task_hash(walls, goal)
            tasks.append(GridTask(len(tasks), walls, start, goal, tuple(path), distances, task_hash))
        if len(tasks) < num_tasks:
            raise RuntimeError(f"only generated {len(tasks)} valid tasks after {attempts} attempts")
        return tasks

    def _make_replay_states(self, per_task: int) -> list[GridReplayState]:
        states: list[GridReplayState] = []
        for task in self.tasks:
            path_indices = np.linspace(1, len(task.path) - 2, per_task, dtype=int)
            for path_idx in path_indices:
                states.append(GridReplayState(len(states), task.task_id, task.path[int(path_idx)]))
        return states


class GridMarginRecoveryBenchmark:
    """Grid-backed recovery-margin benchmark with exact perturbation validity.

    The grid supplies procedural replay states and feasibility constraints. Each
    replay state also has a policy-specific recovery margin that expands most
    efficiently when training perturbations are sampled near the current margin.
    """

    def __init__(
        self,
        num_tasks: int,
        grid_size: int,
        obstacle_prob: float,
        replay_states_per_task: int,
        max_offset: int,
        learning_rate: float,
        seed: int,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.max_offset = int(max_offset)
        self.learning_rate = float(learning_rate)
        base = GridRecoveryBenchmark(
            num_tasks=num_tasks,
            grid_size=grid_size,
            obstacle_prob=obstacle_prob,
            replay_states_per_task=replay_states_per_task,
            max_offset=max_offset,
            horizon=max(2 * grid_size, 16),
            learning_rate=1.0,
            seed=seed,
        )
        self.tasks = base.tasks
        self.replay_states = base.replay_states
        self.states = self._make_margin_states()

    def success_prob(self, replay_idx: int, sigma: float) -> float:
        state = self.states[replay_idx]
        feasible = self.feasibility(replay_idx, sigma)
        transition = 1.0 / (1.0 + np.exp((sigma - state.margin) / state.temperature))
        floor = 0.02 * feasible
        return float(np.clip(floor + (state.clean_success - floor) * transition * feasible, 0.0, 1.0))

    def rollout(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> bool:
        return bool(rng.random() < self.success_prob(replay_idx, sigma))

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        state = self.states[replay_idx]
        feasible = self.feasibility(replay_idx, sigma)
        boundary_signal = np.exp(-((sigma - state.margin) / 0.15) ** 2)
        clean_anchor = 0.25 * np.exp(-(sigma / 0.12) ** 2)
        saturation = max(0.0, 1.0 - state.margin / max(state.feasible_radius, 1e-6))
        gain = self.learning_rate * feasible * (boundary_signal + clean_anchor) * saturation
        state.margin = float(np.clip(state.margin + gain, 0.0, state.feasible_radius))
        state.clean_success = float(np.clip(state.clean_success + 0.02 * clean_anchor, 0.0, 0.995))
        if sigma > state.margin + 0.2:
            state.clean_success = float(np.clip(state.clean_success - 0.001 * (sigma - state.margin), 0.70, 0.995))
        return float(gain)

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        state = self.states[replay_idx]
        return float((1.0 - self.success_prob(replay_idx, sigma)) + state.loss_bias)

    def feasibility(self, replay_idx: int, sigma: float) -> float:
        state = self.states[replay_idx]
        if sigma <= state.feasible_radius:
            return 1.0
        return float(np.exp(-10.0 * (sigma - state.feasible_radius)))

    def _make_margin_states(self) -> list[GridMarginState]:
        states: list[GridMarginState] = []
        for replay in self.replay_states:
            task = self.tasks[replay.task_id]
            local_room = self._local_reachable_radius(task, replay.position)
            path_fraction = float(task.distances[replay.position] / max(1.0, task.distances[task.start]))
            feasible_radius = float(
                np.clip(
                    min(local_room / max(1, self.max_offset), 0.45 + 0.45 * path_fraction + self.rng.normal(0.0, 0.03)),
                    0.35,
                    0.95,
                )
            )
            initial_margin = float(np.clip(0.12 + 0.22 * (1.0 - path_fraction) + self.rng.normal(0.0, 0.05), 0.05, 0.45))
            clean = float(self.rng.uniform(0.80, 0.96))
            temp = float(self.rng.uniform(0.045, 0.10))
            loss_bias = float(self.rng.uniform(0.0, 0.15))
            states.append(GridMarginState(replay, feasible_radius, initial_margin, clean, temp, loss_bias))
        return states

    def _local_reachable_radius(self, task: GridTask, position: tuple[int, int]) -> int:
        best = 0
        for row in range(task.walls.shape[0]):
            for col in range(task.walls.shape[1]):
                pos = (row, col)
                if task.walls[pos] or not np.isfinite(task.distances[pos]):
                    continue
                dist = abs(row - position[0]) + abs(col - position[1])
                if dist <= self.max_offset:
                    best = max(best, dist)
        return best


def shortest_path_distances(walls: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    distances = np.full(walls.shape, np.inf, dtype=float)
    distances[goal] = 0.0
    queue: deque[tuple[int, int]] = deque([goal])
    while queue:
        row, col = queue.popleft()
        for dr, dc in ACTIONS:
            nxt = (row + dr, col + dc)
            if not _in_bounds(walls, nxt) or walls[nxt]:
                continue
            if np.isfinite(distances[nxt]):
                continue
            distances[nxt] = distances[row, col] + 1.0
            queue.append(nxt)
    return distances


def shortest_path(
    walls: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    distances: np.ndarray,
) -> list[tuple[int, int]]:
    path = [start]
    pos = start
    while pos != goal:
        action = oracle_action_from_distances(walls, distances, pos)
        if action is None:
            break
        dr, dc = ACTIONS[action]
        pos = (pos[0] + dr, pos[1] + dc)
        path.append(pos)
    return path


def oracle_action(task: GridTask, position: tuple[int, int]) -> int | None:
    return oracle_action_from_distances(task.walls, task.distances, position)


def oracle_action_from_distances(walls: np.ndarray, distances: np.ndarray, position: tuple[int, int]) -> int | None:
    best_action = None
    best_distance = distances[position]
    for action_idx, (dr, dc) in enumerate(ACTIONS):
        nxt = (position[0] + dr, position[1] + dc)
        if not _in_bounds(walls, nxt) or walls[nxt]:
            continue
        if distances[nxt] < best_distance:
            best_distance = distances[nxt]
            best_action = action_idx
    return best_action


def _in_bounds(walls: np.ndarray, position: tuple[int, int]) -> bool:
    return 0 <= position[0] < walls.shape[0] and 0 <= position[1] < walls.shape[1]


def _task_hash(walls: np.ndarray, goal: tuple[int, int]) -> str:
    digest = hashlib.sha1()
    digest.update(walls.astype(np.uint8).tobytes())
    digest.update(bytes(goal))
    return digest.hexdigest()[:12]
