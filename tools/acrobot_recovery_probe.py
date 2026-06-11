#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bgr.curve_estimators import IsotonicCurveEstimator
from bgr.metrics import critical_radius, recovery_auc
from bgr.priorities import BGRPriorityScorer
from bgr.records import LevelRecord
from bgr.samplers import mixed_priority_probs, sample_boundary_radius


LINK_LENGTH_1 = 1.0
LINK_LENGTH_2 = 1.0
LINK_MASS_1 = 1.0
LINK_MASS_2 = 1.0
LINK_COM_POS_1 = 0.5
LINK_COM_POS_2 = 0.5
LINK_MOI = 1.0
MAX_VEL_1 = 4.0 * np.pi
MAX_VEL_2 = 9.0 * np.pi
DT = 0.2
AVAIL_TORQUE = (-1.0, 0.0, 1.0)
DOWN_ANCHOR = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)


@dataclass(frozen=True, slots=True)
class AcrobotReplayState:
    state: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class AcrobotProbeResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


def wrap(value: float, low: float, high: float) -> float:
    width = high - low
    return float(((value - low) % width) + low)


def bound(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def terminal(state: np.ndarray) -> bool:
    theta1, theta2 = float(state[0]), float(state[1])
    return bool(-np.cos(theta1) - np.cos(theta1 + theta2) > 1.0)


def acrobot_dsdt(state: np.ndarray, torque: float) -> np.ndarray:
    theta1, theta2, dtheta1, dtheta2 = [float(item) for item in state]
    m1 = LINK_MASS_1
    m2 = LINK_MASS_2
    l1 = LINK_LENGTH_1
    lc1 = LINK_COM_POS_1
    lc2 = LINK_COM_POS_2
    i1 = LINK_MOI
    i2 = LINK_MOI
    g = 9.8

    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2.0 * l1 * lc2 * np.cos(theta2)) + i1 + i2
    d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(theta2)) + i2
    phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2**2 * np.sin(theta2)
        - 2.0 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2.0)
        + phi2
    )
    ddtheta2 = (
        torque
        + d2 / d1 * phi1
        - m2 * l1 * lc2 * dtheta1**2 * np.sin(theta2)
        - phi2
    ) / (m2 * lc2**2 + i2 - d2**2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2], dtype=float)


def rk4_step(state: np.ndarray, torque: float, dt: float = DT) -> np.ndarray:
    y0 = np.array(state, dtype=float)
    k1 = acrobot_dsdt(y0, torque)
    k2 = acrobot_dsdt(y0 + dt * k1 / 2.0, torque)
    k3 = acrobot_dsdt(y0 + dt * k2 / 2.0, torque)
    k4 = acrobot_dsdt(y0 + dt * k3, torque)
    return y0 + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def step_state(state: np.ndarray, action: int) -> tuple[np.ndarray, bool]:
    next_state = rk4_step(np.array(state, dtype=float), AVAIL_TORQUE[action])
    next_state[0] = wrap(float(next_state[0]), -np.pi, np.pi)
    next_state[1] = wrap(float(next_state[1]), -np.pi, np.pi)
    next_state[2] = bound(float(next_state[2]), -MAX_VEL_1, MAX_VEL_1)
    next_state[3] = bound(float(next_state[3]), -MAX_VEL_2, MAX_VEL_2)
    return next_state, terminal(next_state)


class AcrobotRecoveryProbe:
    def __init__(
        self,
        *,
        seed: int,
        env_id: str,
        dynamics_backend: str,
        replay_state_count: int,
        theta1_bins: int,
        theta2_bins: int,
        vel1_bins: int,
        vel2_bins: int,
        learning_rate: float,
        discount: float,
        epsilon: float,
        q_init_blend: float,
        q_init_noise: float,
        max_steps: int,
        value_iterations: int,
        transition_cache: np.ndarray | None = None,
        optimal_q: np.ndarray | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed + 127_000)
        self.env_id = str(env_id)
        self.dynamics_backend = str(dynamics_backend)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.epsilon = float(epsilon)
        self.max_steps = int(max_steps)
        self.env = None
        self.unwrapped = None
        if self.dynamics_backend == "gymnasium":
            import gymnasium as gym

            self.env = gym.make(self.env_id)
            self.env.reset(seed=seed)
            self.unwrapped = self.env.unwrapped
        elif self.dynamics_backend != "internal":
            raise ValueError(f"unknown Acrobot dynamics backend: {self.dynamics_backend}")
        self.theta1_grid = np.linspace(-np.pi, np.pi, int(theta1_bins))
        self.theta2_grid = np.linspace(-np.pi, np.pi, int(theta2_bins))
        self.vel1_grid = np.linspace(-MAX_VEL_1, MAX_VEL_1, int(vel1_bins))
        self.vel2_grid = np.linspace(-MAX_VEL_2, MAX_VEL_2, int(vel2_bins))
        self.transition_cache = self._build_transition_cache() if transition_cache is None else transition_cache
        self.optimal_q = self._value_iteration(iterations=int(value_iterations)) if optimal_q is None else optimal_q
        self.q = float(q_init_blend) * self.optimal_q
        self.q += self.rng.normal(0.0, float(q_init_noise), size=self.q.shape)
        self.states = self._select_replay_states(int(replay_state_count))

    def _build_transition_cache(self) -> np.ndarray:
        shape = (
            len(self.theta1_grid),
            len(self.theta2_grid),
            len(self.vel1_grid),
            len(self.vel2_grid),
            3,
            5,
        )
        cache = np.zeros(shape, dtype=float)
        for i, theta1 in enumerate(self.theta1_grid):
            for j, theta2 in enumerate(self.theta2_grid):
                for k, vel1 in enumerate(self.vel1_grid):
                    for l, vel2 in enumerate(self.vel2_grid):
                        state = np.array([theta1, theta2, vel1, vel2], dtype=float)
                        for action in range(3):
                            next_state, done = self._step_state(state, action)
                            next_idx = self._index(next_state)
                            cache[i, j, k, l, action, :4] = np.array(next_idx, dtype=float)
                            cache[i, j, k, l, action, 4] = 1.0 if done else 0.0
        return cache

    def _step_state(self, state: np.ndarray, action: int) -> tuple[np.ndarray, bool]:
        if self.dynamics_backend == "internal":
            return step_state(state, action)
        if self.unwrapped is None:
            raise RuntimeError("gymnasium dynamics selected without an environment")
        if hasattr(self.env, "_elapsed_steps"):
            self.env._elapsed_steps = 0  # type: ignore[attr-defined]
        self.unwrapped.state = np.array(state, dtype=np.float32)
        _obs, _reward, terminated, truncated, _info = self.env.step(int(action))
        next_state = self._clip_state(np.array(self.unwrapped.state, dtype=float))
        return next_state, bool(terminated or truncated)

    def _value_iteration(self, iterations: int) -> np.ndarray:
        q = np.zeros(self.transition_cache.shape[:-1], dtype=float)
        for _ in range(iterations):
            next_q = q.copy()
            max_delta = 0.0
            for index in np.ndindex(q.shape[:-1]):
                for action in range(3):
                    cached = self.transition_cache[index + (action,)]
                    done = bool(cached[4] > 0.5)
                    if terminal(np.array([self.theta1_grid[index[0]], self.theta2_grid[index[1]], self.vel1_grid[index[2]], self.vel2_grid[index[3]]])):
                        target = 0.0
                    else:
                        next_idx = tuple(int(item) for item in cached[:4])
                        target = 0.0 if done else -1.0 + self.discount * float(np.max(q[next_idx]))
                    max_delta = max(max_delta, abs(target - next_q[index + (action,)]))
                    next_q[index + (action,)] = target
            q = next_q
            if max_delta < 1e-5:
                break
        return q

    def _select_replay_states(self, count: int) -> list[AcrobotReplayState]:
        theta1s = np.linspace(0.85, 2.55, 10)
        theta2s = np.linspace(-1.35, 1.35, 9)
        vel1s = np.linspace(-2.0, 2.0, 5)
        vel2s = np.linspace(-4.0, 4.0, 5)
        candidates: list[tuple[float, AcrobotReplayState]] = []
        for theta1 in theta1s:
            for theta2 in theta2s:
                for vel1 in vel1s:
                    for vel2 in vel2s:
                        state = np.array([theta1, theta2, vel1, vel2], dtype=float)
                        if terminal(state):
                            continue
                        height = -np.cos(theta1) - np.cos(theta1 + theta2)
                        candidates.append((float(height), AcrobotReplayState(tuple(float(x) for x in state))))
        candidates.sort(key=lambda item: (item[0], abs(item[1].state[2]) + abs(item[1].state[3])))
        if count >= len(candidates):
            selected = candidates
        else:
            indexes = np.linspace(max(0, len(candidates) // 3), len(candidates) - 1, count, dtype=int)
            selected = [candidates[int(index)] for index in indexes]
        return [item[1] for item in selected]

    def _index(self, state: np.ndarray) -> tuple[int, int, int, int]:
        return (
            int(np.argmin(np.abs(self.theta1_grid - float(state[0])))),
            int(np.argmin(np.abs(self.theta2_grid - float(state[1])))),
            int(np.argmin(np.abs(self.vel1_grid - float(state[2])))),
            int(np.argmin(np.abs(self.vel2_grid - float(state[3])))),
        )

    def clean_success(self, replay_idx: int) -> float:
        return self.success_prob(replay_idx, 0.0)

    def success_prob(self, replay_idx: int, sigma: float) -> float:
        starts = self.evaluation_starts(replay_idx, sigma)
        return float(np.mean([self.rollout_state(start, train=False, epsilon=False) for start in starts]))

    def evaluation_starts(self, replay_idx: int, sigma: float) -> list[np.ndarray]:
        replay = np.array(self.states[replay_idx].state, dtype=float)
        sigma = float(np.clip(sigma, 0.0, 1.0))
        if sigma == 0.0:
            return [replay]
        center = self._adverse_center(replay, sigma)
        scales = np.array([0.55, 0.55, 1.4, 2.8], dtype=float)
        directions = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0, -1.0]),
        ]
        return [self._clip_state(center + sigma * scales * direction) for direction in directions]

    def sample_start(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
        replay = np.array(self.states[replay_idx].state, dtype=float)
        sigma = float(np.clip(sigma, 0.0, 1.0))
        if sigma == 0.0:
            return replay
        center = self._adverse_center(replay, sigma)
        scales = np.array([0.45, 0.45, 1.2, 2.4], dtype=float)
        noise = rng.normal(0.0, 1.0, size=4)
        norm = float(np.linalg.norm(noise))
        direction = noise / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])
        radius = sigma * np.sqrt(float(rng.uniform(0.0, 1.0)))
        return self._clip_state(center + radius * scales * direction)

    def _adverse_center(self, replay: np.ndarray, sigma: float) -> np.ndarray:
        return self._clip_state((1.0 - sigma) * replay + sigma * DOWN_ANCHOR)

    def _clip_state(self, state: np.ndarray) -> np.ndarray:
        clipped = np.array(state, dtype=float)
        clipped[0] = wrap(float(clipped[0]), -np.pi, np.pi)
        clipped[1] = wrap(float(clipped[1]), -np.pi, np.pi)
        clipped[2] = bound(float(clipped[2]), -MAX_VEL_1, MAX_VEL_1)
        clipped[3] = bound(float(clipped[3]), -MAX_VEL_2, MAX_VEL_2)
        return clipped

    def rollout(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> bool:
        return self.rollout_state(self.sample_start(replay_idx, sigma, rng), train=False, epsilon=True, rng=rng)

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> None:
        self.rollout_state(self.sample_start(replay_idx, sigma, rng), train=True, epsilon=True, rng=rng)

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        state = self.sample_start(replay_idx, sigma, rng)
        state_idx = self._index(state)
        action = self._action(state, rng, epsilon=True)
        next_state, done = self._step_state(state, action)
        target = 0.0 if done else -1.0 + self.discount * float(np.max(self.q[self._index(next_state)]))
        return abs(target - float(self.q[state_idx + (action,)]))

    def rollout_state(
        self,
        state: np.ndarray,
        *,
        train: bool,
        epsilon: bool,
        rng: np.random.Generator | None = None,
    ) -> bool:
        state = np.array(state, dtype=float)
        if terminal(state):
            return True
        for _ in range(self.max_steps):
            action = self._action(state, rng, epsilon=epsilon)
            state_idx = self._index(state)
            next_state, done = self._step_state(state, action)
            if train:
                target = 0.0 if done else -1.0 + self.discount * float(np.max(self.q[self._index(next_state)]))
                self.q[state_idx + (action,)] += self.learning_rate * (target - float(self.q[state_idx + (action,)]))
            state = next_state
            if done:
                return True
        return False

    def _action(self, state: np.ndarray, rng: np.random.Generator | None, *, epsilon: bool) -> int:
        if epsilon and rng is not None and rng.random() < self.epsilon:
            return int(rng.integers(3))
        return int(np.argmax(self.q[self._index(state)]))


def run_method(
    args: argparse.Namespace,
    method: str,
    seed: int,
    *,
    transition_cache: np.ndarray,
    optimal_q: np.ndarray,
) -> AcrobotProbeResult:
    rng = np.random.default_rng(seed + 131_000)
    bench = AcrobotRecoveryProbe(
        seed=seed,
        env_id=args.env_id,
        dynamics_backend=args.dynamics_backend,
        replay_state_count=args.replay_states,
        theta1_bins=args.theta1_bins,
        theta2_bins=args.theta2_bins,
        vel1_bins=args.vel1_bins,
        vel2_bins=args.vel2_bins,
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        q_init_blend=args.q_init_blend,
        q_init_noise=args.q_init_noise,
        max_steps=args.max_steps,
        value_iterations=args.value_iterations,
        transition_cache=transition_cache,
        optimal_q=optimal_q,
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
    return AcrobotProbeResult(
        method=method,
        seed=seed,
        final_clean=final["clean"],
        final_rauc=final["rauc"],
        final_median_r80=final["median_r80"],
        rauc_aulc=float(np.trapezoid(ys, xs) / (xs[-1] - xs[0])) if xs[-1] > xs[0] else final["rauc"],
        best_rauc=max(row["rauc"] for row in history),
        history=history,
    )


def init_records(bench: AcrobotRecoveryProbe, rng: np.random.Generator, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, _replay in enumerate(bench.states):
        record = LevelRecord(
            id=f"acrobot_{replay_idx}",
            domain="acrobot_v1_recovery",
            task_id=args.env_id,
            clean_success_hat=bench.clean_success(replay_idx),
            feasibility_hat=1.0,
            perturbation_family="adverse_state_restart",
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
    bench: AcrobotRecoveryProbe,
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


def write_estimate(record: LevelRecord, estimate) -> None:
    record.r_alpha_hat = estimate.r_alpha
    record.sharpness_hat = estimate.sharpness
    record.uncertainty_hat = estimate.r_uncertainty
    record.recovery_curve_hat = estimate.recovery.tolist()


def evaluate(bench: AcrobotRecoveryProbe, args: argparse.Namespace) -> dict[str, float]:
    grid = np.linspace(0.0, 1.0, args.eval_grid_size)
    clean: list[float] = []
    raucs: list[float] = []
    radii: list[float] = []
    for replay_idx in range(len(bench.states)):
        curve = np.array([bench.success_prob(replay_idx, sigma) for sigma in grid], dtype=float)
        clean.append(float(curve[0]))
        raucs.append(recovery_auc(grid, curve, sigma_max=1.0))
        radii.append(critical_radius(grid, curve, alpha=args.alpha))
    return {
        "clean": float(np.mean(clean)),
        "rauc": float(np.mean(raucs)),
        "median_r80": float(np.median(radii)),
    }


def sample_training_pair(
    method: str,
    bench: AcrobotRecoveryProbe,
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
    parser = argparse.ArgumentParser(description="Run an internal Acrobot-v1 recovery replay diagnostic.")
    parser.add_argument("--out", default="runs/acrobot_recovery_probe")
    parser.add_argument("--env-id", default="Acrobot-v1")
    parser.add_argument("--dynamics-backend", choices=["internal", "gymnasium"], default="internal")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=6)
    parser.add_argument("--replay-states", type=int, default=24)
    parser.add_argument("--theta1-bins", type=int, default=9)
    parser.add_argument("--theta2-bins", type=int, default=9)
    parser.add_argument("--vel1-bins", type=int, default=5)
    parser.add_argument("--vel2-bins", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--value-iterations", type=int, default=20)
    parser.add_argument("--eval-grid-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--q-init-blend", type=float, default=0.18)
    parser.add_argument("--q-init-noise", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--target-radius", type=float, default=0.45)
    parser.add_argument("--radius-bandwidth", type=float, default=0.35)
    parser.add_argument("--fixed-radius", type=float, default=0.65)
    parser.add_argument("--baseline-candidates", type=int, default=4)
    parser.add_argument("--initial-probes", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--min-trials", type=int, default=1)
    parser.add_argument("--refresh-per-eval", type=int, default=8)
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
    asset_start = time.perf_counter()
    asset_probe = AcrobotRecoveryProbe(
        seed=0,
        env_id=args.env_id,
        dynamics_backend=args.dynamics_backend,
        replay_state_count=1,
        theta1_bins=args.theta1_bins,
        theta2_bins=args.theta2_bins,
        vel1_bins=args.vel1_bins,
        vel2_bins=args.vel2_bins,
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        q_init_blend=1.0,
        q_init_noise=0.0,
        max_steps=args.max_steps,
        value_iterations=args.value_iterations,
    )
    if args.dynamics_backend == "gymnasium":
        import gymnasium as gym

        (out_dir / "package_versions.json").write_text(
            json.dumps({"gymnasium": gym.__version__, "env_id": args.env_id}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"[setup] value_assets_elapsed={time.perf_counter() - asset_start:.2f}s", flush=True)

    rows: list[dict[str, float | int | str]] = []
    results: list[dict] = []
    for method in methods:
        for seed in seeds:
            start = time.perf_counter()
            result = run_method(
                args,
                method,
                seed,
                transition_cache=asset_probe.transition_cache,
                optimal_q=asset_probe.optimal_q,
            )
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
                    "rauc_aulc": result.rauc_aulc,
                    "best_rauc": result.best_rauc,
                }
            )

    (out_dir / "results.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(summary(rows))


def summary(rows: list[dict[str, float | int | str]]) -> str:
    by_method: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    lines = ["method,final_clean_mean,final_rauc_mean,final_median_r80_mean,rauc_aulc_mean,best_rauc_mean"]
    for method, method_rows in sorted(by_method.items()):
        lines.append(
            ",".join(
                [
                    method,
                    f"{np.mean([float(row['final_clean']) for row in method_rows]):.4f}",
                    f"{np.mean([float(row['final_rauc']) for row in method_rows]):.4f}",
                    f"{np.mean([float(row['final_median_r80']) for row in method_rows]):.4f}",
                    f"{np.mean([float(row['rauc_aulc']) for row in method_rows]):.4f}",
                    f"{np.mean([float(row['best_rauc']) for row in method_rows]):.4f}",
                ]
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
