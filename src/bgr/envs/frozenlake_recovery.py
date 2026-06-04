from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


FROZENLAKE_8X8_MAP: tuple[str, ...] = (
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
)


@dataclass(frozen=True, slots=True)
class FrozenLakeReplayState:
    state: int
    row: int
    col: int
    optimal_success: float


class FrozenLakeRecoveryBenchmark:
    """Resettable recovery benchmark on the canonical Gym FrozenLake 8x8 map."""

    def __init__(
        self,
        *,
        replay_state_count: int = 36,
        max_radius: int = 6,
        learning_rate: float = 0.35,
        discount: float = 0.97,
        epsilon: float = 0.08,
        q_init_blend: float = 0.45,
        q_init_noise: float = 0.03,
        seed: int = 0,
    ) -> None:
        self.desc = np.array([list(row) for row in FROZENLAKE_8X8_MAP])
        self.nrow, self.ncol = self.desc.shape
        self.num_states = int(self.nrow * self.ncol)
        self.num_actions = 4
        self.max_radius = int(max_radius)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed + 97_000)

        self.goal_state = self._to_state(self.nrow - 1, self.ncol - 1)
        self.safe_states = [
            self._to_state(row, col)
            for row in range(self.nrow)
            for col in range(self.ncol)
            if self.desc[row, col] in {"S", "F"}
        ]
        self.optimal_q = self._value_iteration()
        self.q = float(q_init_blend) * self.optimal_q
        self.q += self.rng.normal(0.0, float(q_init_noise), size=self.q.shape)
        self.q[self._terminal_mask(), :] = 0.0
        self.states = self._select_replay_states(int(replay_state_count))

    def clean_success(self, replay_idx: int) -> float:
        return self.success_prob(replay_idx, 0.0)

    def success_prob(self, replay_idx: int, sigma: float) -> float:
        starts = self.perturbation_states(replay_idx, sigma)
        values = self._policy_success_values(self._greedy_policy())
        return float(np.mean([values[state] for state in starts]))

    def rollout(self, replay_idx: int, sigma: float, rng: np.random.Generator, max_steps: int = 96) -> bool:
        state = int(rng.choice(self.perturbation_states(replay_idx, sigma)))
        for _ in range(max_steps):
            tile = self._tile(state)
            if tile == "G":
                return True
            if tile == "H":
                return False
            action = self._epsilon_greedy_action(state, rng)
            state, reward, done = self._sample_transition(state, action, rng)
            if done:
                return reward > 0.0
        return False

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator, max_steps: int = 96) -> None:
        state = int(rng.choice(self.perturbation_states(replay_idx, sigma)))
        for _ in range(max_steps):
            tile = self._tile(state)
            if tile in {"G", "H"}:
                return
            action = self._epsilon_greedy_action(state, rng)
            next_state, reward, done = self._sample_transition(state, action, rng)
            target = reward if done else reward + self.discount * float(np.max(self.q[next_state]))
            self.q[state, action] += self.learning_rate * (target - self.q[state, action])
            state = next_state
            if done:
                return

    def feasibility(self, replay_idx: int, sigma: float) -> float:
        return 1.0 if self.perturbation_states(replay_idx, sigma) else 0.0

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        state = int(rng.choice(self.perturbation_states(replay_idx, sigma)))
        action = self._epsilon_greedy_action(state, rng)
        next_state, reward, done = self._sample_transition(state, action, rng)
        target = reward if done else reward + self.discount * float(np.max(self.q[next_state]))
        return abs(target - float(self.q[state, action]))

    def perturbation_states(self, replay_idx: int, sigma: float) -> list[int]:
        replay = self.states[replay_idx]
        radius = int(round(float(np.clip(sigma, 0.0, 1.0)) * self.max_radius))
        candidates: list[int] = []
        for state in self.safe_states:
            row, col = self._to_pos(state)
            dist = abs(row - replay.row) + abs(col - replay.col)
            if dist == radius:
                candidates.append(state)
        if candidates:
            return candidates
        if radius > 0:
            for state in self.safe_states:
                row, col = self._to_pos(state)
                dist = abs(row - replay.row) + abs(col - replay.col)
                if dist <= radius:
                    candidates.append(state)
        return candidates or [replay.state]

    def _select_replay_states(self, count: int) -> list[FrozenLakeReplayState]:
        values = self._optimal_success_values()
        candidates: list[FrozenLakeReplayState] = []
        for state in self.safe_states:
            row, col = self._to_pos(state)
            if values[state] >= 0.08:
                candidates.append(FrozenLakeReplayState(state=state, row=row, col=col, optimal_success=float(values[state])))
        candidates.sort(key=lambda item: (item.optimal_success, item.row, item.col), reverse=True)
        if count >= len(candidates):
            return candidates
        idxs = np.linspace(0, len(candidates) - 1, count, dtype=int)
        return [candidates[int(idx)] for idx in idxs]

    def _value_iteration(self, iterations: int = 600) -> np.ndarray:
        q = np.zeros((self.num_states, self.num_actions), dtype=float)
        for _ in range(iterations):
            next_q = q.copy()
            for state in range(self.num_states):
                if self._tile(state) in {"G", "H"}:
                    next_q[state, :] = 0.0
                    continue
                for action in range(self.num_actions):
                    total = 0.0
                    for prob, next_state, reward, done in self._transitions(state, action):
                        total += prob * (reward if done else reward + self.discount * float(np.max(q[next_state])))
                    next_q[state, action] = total
            if float(np.max(np.abs(next_q - q))) < 1e-10:
                q = next_q
                break
            q = next_q
        return q

    def _optimal_success_values(self) -> np.ndarray:
        policy = np.argmax(self.optimal_q, axis=1)
        return self._policy_success_values(policy)

    def _policy_success_values(self, policy: np.ndarray) -> np.ndarray:
        safe = [state for state in range(self.num_states) if self._tile(state) not in {"G", "H"}]
        index = {state: idx for idx, state in enumerate(safe)}
        a = np.eye(len(safe), dtype=float)
        b = np.zeros(len(safe), dtype=float)
        for state in safe:
            row = index[state]
            for prob, next_state, reward, done in self._transitions(state, int(policy[state])):
                if done:
                    b[row] += prob * reward
                else:
                    a[row, index[next_state]] -= prob
        solution = np.linalg.solve(a, b)
        values = np.zeros(self.num_states, dtype=float)
        values[self.goal_state] = 1.0
        for state, idx in index.items():
            values[state] = solution[idx]
        return values

    def _greedy_policy(self) -> np.ndarray:
        return np.argmax(self.q, axis=1)

    def _epsilon_greedy_action(self, state: int, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(self.num_actions))
        return int(np.argmax(self.q[state]))

    def _sample_transition(self, state: int, action: int, rng: np.random.Generator) -> tuple[int, float, bool]:
        transitions = self._transitions(state, action)
        idx = int(rng.choice(len(transitions), p=[item[0] for item in transitions]))
        _, next_state, reward, done = transitions[idx]
        return next_state, reward, done

    @cached_property
    def _transition_cache(self) -> dict[tuple[int, int], tuple[tuple[float, int, float, bool], ...]]:
        return {
            (state, action): tuple(self._build_transitions(state, action))
            for state in range(self.num_states)
            for action in range(self.num_actions)
        }

    def _transitions(self, state: int, action: int) -> tuple[tuple[float, int, float, bool], ...]:
        return self._transition_cache[(state, action)]

    def _build_transitions(self, state: int, action: int) -> list[tuple[float, int, float, bool]]:
        if self._tile(state) in {"G", "H"}:
            return [(1.0, state, 0.0, True)]
        actual_actions = ((action - 1) % 4, action, (action + 1) % 4)
        merged: dict[int, float] = {}
        for actual in actual_actions:
            next_state = self._move(state, actual)
            merged[next_state] = merged.get(next_state, 0.0) + 1.0 / 3.0
        return [
            (prob, next_state, 1.0 if self._tile(next_state) == "G" else 0.0, self._tile(next_state) in {"G", "H"})
            for next_state, prob in sorted(merged.items())
        ]

    def _move(self, state: int, action: int) -> int:
        row, col = self._to_pos(state)
        if action == 0:
            col = max(col - 1, 0)
        elif action == 1:
            row = min(row + 1, self.nrow - 1)
        elif action == 2:
            col = min(col + 1, self.ncol - 1)
        elif action == 3:
            row = max(row - 1, 0)
        else:
            raise ValueError(f"unknown action: {action}")
        return self._to_state(row, col)

    def _terminal_mask(self) -> np.ndarray:
        return np.array([self._tile(state) in {"G", "H"} for state in range(self.num_states)], dtype=bool)

    def _tile(self, state: int) -> str:
        row, col = self._to_pos(state)
        return str(self.desc[row, col])

    def _to_state(self, row: int, col: int) -> int:
        return int(row * self.ncol + col)

    def _to_pos(self, state: int) -> tuple[int, int]:
        return int(state // self.ncol), int(state % self.ncol)
