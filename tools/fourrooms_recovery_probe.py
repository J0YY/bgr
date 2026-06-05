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


SIZE = 11
DOORS = {(5, 1), (5, 8), (1, 5), (8, 5)}
GOAL = (10, 10)
ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True, slots=True)
class FourRoomsReplayState:
    row: int
    col: int


@dataclass(frozen=True, slots=True)
class FourRoomsProbeResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


def is_wall(row: int, col: int) -> bool:
    if (row, col) in DOORS:
        return False
    return row == 5 or col == 5


def free_cells() -> list[tuple[int, int]]:
    return [(row, col) for row in range(SIZE) for col in range(SIZE) if not is_wall(row, col)]


FREE_CELLS = tuple(free_cells())
CELL_TO_STATE = {cell: idx for idx, cell in enumerate(FREE_CELLS)}
STATE_TO_CELL = {idx: cell for cell, idx in CELL_TO_STATE.items()}


def encode(row: int, col: int) -> int:
    return CELL_TO_STATE[(row, col)]


def decode(state: int) -> tuple[int, int]:
    return STATE_TO_CELL[int(state)]


def fourrooms_step(state: int, action: int) -> tuple[int, float, bool]:
    row, col = decode(state)
    dr, dc = ACTIONS[action]
    next_row = int(np.clip(row + dr, 0, SIZE - 1))
    next_col = int(np.clip(col + dc, 0, SIZE - 1))
    if is_wall(next_row, next_col):
        next_row, next_col = row, col
    done = (next_row, next_col) == GOAL
    return encode(next_row, next_col), 0.0 if done else -1.0, done


def shortest_path_distance(start: tuple[int, int], goal: tuple[int, int]) -> int:
    frontier = [start]
    distances = {start: 0}
    while frontier:
        row, col = frontier.pop(0)
        if (row, col) == goal:
            return distances[(row, col)]
        for dr, dc in ACTIONS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < SIZE and 0 <= nc < SIZE and not is_wall(nr, nc) and (nr, nc) not in distances:
                distances[(nr, nc)] = distances[(row, col)] + 1
                frontier.append((nr, nc))
    raise ValueError(f"unreachable cell: {start} -> {goal}")


def value_iteration(discount: float, iterations: int = 900) -> np.ndarray:
    q = np.zeros((len(FREE_CELLS), 4), dtype=float)
    goal_state = encode(*GOAL)
    for _ in range(iterations):
        next_q = q.copy()
        max_delta = 0.0
        for state in range(len(FREE_CELLS)):
            if state == goal_state:
                next_q[state, :] = 0.0
                continue
            for action in range(4):
                next_state, reward, done = fourrooms_step(state, action)
                target = reward if done else reward + discount * float(np.max(q[next_state]))
                max_delta = max(max_delta, abs(target - next_q[state, action]))
                next_q[state, action] = target
        q = next_q
        if max_delta < 1e-10:
            break
    return q


class FourRoomsRecoveryProbe:
    def __init__(
        self,
        *,
        optimal_q: np.ndarray,
        seed: int,
        replay_state_count: int,
        max_radius: int,
        learning_rate: float,
        discount: float,
        epsilon: float,
        q_init_blend: float,
        q_init_noise: float,
        max_steps: int,
    ) -> None:
        self.rng = np.random.default_rng(seed + 223_000)
        self.max_radius = int(max_radius)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.epsilon = float(epsilon)
        self.max_steps = int(max_steps)
        self.q = float(q_init_blend) * optimal_q
        self.q += self.rng.normal(0.0, float(q_init_noise), size=self.q.shape)
        self.states = self._select_replay_states(int(replay_state_count))
        self.perturbation_cache = self._build_perturbation_cache()

    def _select_replay_states(self, count: int) -> list[FourRoomsReplayState]:
        door_neighbors: set[tuple[int, int]] = set()
        for door_row, door_col in DOORS:
            for dr, dc in ACTIONS:
                row, col = door_row + dr, door_col + dc
                if 0 <= row < SIZE and 0 <= col < SIZE and not is_wall(row, col):
                    door_neighbors.add((row, col))
        candidates = [
            (shortest_path_distance(cell, GOAL), FourRoomsReplayState(*cell))
            for cell in FREE_CELLS
            if cell != GOAL and (cell in door_neighbors or min(abs(cell[0] - r) + abs(cell[1] - c) for r, c in DOORS) <= 3)
        ]
        candidates.sort(key=lambda item: (item[0], item[1].row, item[1].col))
        if count >= len(candidates):
            return [item[1] for item in candidates]
        indexes = np.linspace(0, len(candidates) - 1, count, dtype=int)
        return [candidates[int(index)][1] for index in indexes]

    def _build_perturbation_cache(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        cache: list[tuple[tuple[int, ...], ...]] = []
        for replay in self.states:
            radii: list[tuple[int, ...]] = []
            for radius in range(self.max_radius + 1):
                states = tuple(
                    encode(row, col)
                    for row, col in FREE_CELLS
                    if (row, col) != GOAL and abs(row - replay.row) + abs(col - replay.col) == radius
                )
                radii.append(states or (encode(replay.row, replay.col),))
            cache.append(tuple(radii))
        return tuple(cache)

    def perturbation_states(self, replay_idx: int, sigma: float) -> tuple[int, ...]:
        radius = int(round(float(np.clip(sigma, 0.0, 1.0)) * self.max_radius))
        return self.perturbation_cache[replay_idx][radius]

    def current_success_values(self) -> np.ndarray:
        policy = np.argmax(self.q, axis=1)
        values = np.zeros(len(FREE_CELLS), dtype=float)
        for state in range(len(FREE_CELLS)):
            values[state] = float(self._rollout_state(state, policy=policy, train=False, epsilon=False))
        return values

    def success_prob_from_values(self, replay_idx: int, sigma: float, values: np.ndarray) -> float:
        starts = self.perturbation_states(replay_idx, sigma)
        return float(np.mean([values[state] for state in starts]))

    def success_prob(self, replay_idx: int, sigma: float) -> float:
        return self.success_prob_from_values(replay_idx, sigma, self.current_success_values())

    def rollout(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> bool:
        state = int(rng.choice(self.perturbation_states(replay_idx, sigma)))
        return self._rollout_state(state, policy=None, train=False, epsilon=True, rng=rng)

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> None:
        state = int(rng.choice(self.perturbation_states(replay_idx, sigma)))
        self._rollout_state(state, policy=None, train=True, epsilon=True, rng=rng)

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        state = int(rng.choice(self.perturbation_states(replay_idx, sigma)))
        action = self._action(state, rng)
        next_state, reward, done = fourrooms_step(state, action)
        target = reward if done else reward + self.discount * float(np.max(self.q[next_state]))
        return abs(target - float(self.q[state, action]))

    def _rollout_state(
        self,
        state: int,
        *,
        policy: np.ndarray | None,
        train: bool,
        epsilon: bool,
        rng: np.random.Generator | None = None,
    ) -> bool:
        for _ in range(self.max_steps):
            action = int(policy[state]) if policy is not None else self._action(state, rng, epsilon=epsilon)
            next_state, reward, done = fourrooms_step(state, action)
            if train:
                target = reward if done else reward + self.discount * float(np.max(self.q[next_state]))
                self.q[state, action] += self.learning_rate * (target - float(self.q[state, action]))
            state = next_state
            if done:
                return True
        return False

    def _action(self, state: int, rng: np.random.Generator | None, *, epsilon: bool = True) -> int:
        if epsilon and rng is not None and rng.random() < self.epsilon:
            return int(rng.integers(4))
        return int(np.argmax(self.q[state]))


def run_method(args: argparse.Namespace, optimal_q: np.ndarray, method: str, seed: int) -> FourRoomsProbeResult:
    rng = np.random.default_rng(seed + 229_000)
    bench = FourRoomsRecoveryProbe(
        optimal_q=optimal_q,
        seed=seed,
        replay_state_count=args.replay_states,
        max_radius=args.max_radius,
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        q_init_blend=args.q_init_blend,
        q_init_noise=args.q_init_noise,
        max_steps=args.max_steps,
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
        values = bench.current_success_values()
        for _ in range(args.train_batch_size):
            replay_idx, sigma = sample_training_pair(method, bench, records, scorer, rng, args, step, values)
            bench.train_step(replay_idx, sigma, rng)
            if method.startswith("bgr"):
                records[replay_idx].add_observation(sigma, bench.rollout(replay_idx, sigma, rng))
                records[replay_idx].replay_count += 1

    xs = np.array([row["step"] for row in history], dtype=float)
    ys = np.array([row["rauc"] for row in history], dtype=float)
    final = history[-1]
    return FourRoomsProbeResult(
        method=method,
        seed=seed,
        final_clean=final["clean"],
        final_rauc=final["rauc"],
        final_median_r80=final["median_r80"],
        rauc_aulc=float(np.trapezoid(ys, xs) / (xs[-1] - xs[0])) if xs[-1] > xs[0] else final["rauc"],
        best_rauc=max(row["rauc"] for row in history),
        history=history,
    )


def init_records(bench: FourRoomsRecoveryProbe, rng: np.random.Generator, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, replay in enumerate(bench.states):
        record = LevelRecord(
            id=f"fourrooms_{replay.row}_{replay.col}",
            domain="fourrooms_recovery",
            task_id="FourRooms-11x11",
            clean_success_hat=bench.success_prob(replay_idx, 0.0),
            feasibility_hat=1.0,
            perturbation_family="fourrooms_manhattan_restart",
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
    bench: FourRoomsRecoveryProbe,
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
        record.clean_success_hat = bench.success_prob(int(replay_idx), 0.0)
        record.feasibility_hat = 1.0
        record.last_evaluated_step = step


def write_estimate(record: LevelRecord, estimate) -> None:
    record.r_alpha_hat = estimate.r_alpha
    record.sharpness_hat = estimate.sharpness
    record.uncertainty_hat = estimate.r_uncertainty
    record.recovery_curve_hat = estimate.recovery.tolist()


def evaluate(bench: FourRoomsRecoveryProbe, args: argparse.Namespace) -> dict[str, float]:
    grid = np.linspace(0.0, 1.0, args.eval_grid_size)
    values = bench.current_success_values()
    clean: list[float] = []
    raucs: list[float] = []
    radii: list[float] = []
    for replay_idx in range(len(bench.states)):
        curve = np.array([bench.success_prob_from_values(replay_idx, sigma, values) for sigma in grid], dtype=float)
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
    bench: FourRoomsRecoveryProbe,
    records: list[LevelRecord],
    scorer: BGRPriorityScorer,
    rng: np.random.Generator,
    args: argparse.Namespace,
    step: int,
    values: np.ndarray,
) -> tuple[int, float]:
    if method == "uniform":
        return int(rng.integers(len(records))), float(rng.uniform(0.0, 1.0))
    if method == "fixed":
        return int(rng.integers(len(records))), args.fixed_radius
    if method == "failure_only":
        candidates = rng.choice(len(records), size=min(args.baseline_candidates, len(records)), replace=False)
        sigmas = rng.uniform(0.0, 1.0, size=len(candidates))
        scores = [
            1.0 - bench.success_prob_from_values(int(candidate), float(sigma), values)
            for candidate, sigma in zip(candidates, sigmas, strict=True)
        ]
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
    parser = argparse.ArgumentParser(description="Run an internal FourRooms recovery replay diagnostic.")
    parser.add_argument("--out", default="runs/fourrooms_recovery_probe")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--train-batch-size", type=int, default=12)
    parser.add_argument("--replay-states", type=int, default=28)
    parser.add_argument("--max-radius", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=42)
    parser.add_argument("--eval-grid-size", type=int, default=9)
    parser.add_argument("--learning-rate", type=float, default=0.32)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--q-init-blend", type=float, default=0.14)
    parser.add_argument("--q-init-noise", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--target-radius", type=float, default=0.45)
    parser.add_argument("--radius-bandwidth", type=float, default=0.35)
    parser.add_argument("--fixed-radius", type=float, default=0.65)
    parser.add_argument("--baseline-candidates", type=int, default=12)
    parser.add_argument("--initial-probes", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--min-trials", type=int, default=1)
    parser.add_argument("--refresh-per-eval", type=int, default=14)
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
    optimal_q = value_iteration(args.discount)

    rows: list[dict[str, float | int | str]] = []
    results: list[dict] = []
    for method in methods:
        for seed in seeds:
            start = time.perf_counter()
            result = run_method(args, optimal_q, method, seed)
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
                    f"{mean(method_rows, 'final_clean'):.4f}",
                    f"{mean(method_rows, 'final_rauc'):.4f}",
                    f"{mean(method_rows, 'final_median_r80'):.4f}",
                    f"{mean(method_rows, 'rauc_aulc'):.4f}",
                    f"{mean(method_rows, 'best_rauc'):.4f}",
                ]
            )
        )
    return "\n".join(lines)


def mean(rows: list[dict[str, float | int | str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


if __name__ == "__main__":
    main()
