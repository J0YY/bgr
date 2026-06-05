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
DIRS = (0, 1, 2, 3)
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True, slots=True)
class MiniGridReplayState:
    x: int
    y: int
    direction: int


@dataclass(frozen=True, slots=True)
class MiniGridProbeResult:
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
            "MiniGrid probe requires the external minigrid/gymnasium packages. "
            "Install them in an isolated environment before running this internal diagnostic."
        ) from exc
    return {"gymnasium": gym.__version__, "minigrid": minigrid.__version__}


class OfficialMiniGridFourRoomsProbe:
    def __init__(
        self,
        *,
        seed: int,
        replay_state_count: int,
        max_radius: int,
        learning_rate: float,
        discount: float,
        epsilon: float,
        q_init_blend: float,
        q_init_noise: float,
        rollout_horizon: int,
    ) -> None:
        import gymnasium as gym

        self.rng = np.random.default_rng(seed + 337_000)
        self.env_id = "MiniGrid-FourRooms-v0"
        self.env = gym.make(self.env_id)
        self.env.reset(seed=seed)
        self.unwrapped = self.env.unwrapped
        self.grid = self.unwrapped.grid.copy()
        self.width = int(self.unwrapped.width)
        self.height = int(self.unwrapped.height)
        self.max_radius = int(max_radius)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.epsilon = float(epsilon)
        self.rollout_horizon = int(rollout_horizon)
        self.free_cells = self._free_cells()
        self.goal_cells = self._goal_cells()
        self.state_to_idx = {
            (x, y, direction): idx
            for idx, (x, y, direction) in enumerate((x, y, d) for x, y in self.free_cells if (x, y) not in self.goal_cells for d in DIRS)
        }
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}
        self.transitions = self._build_transitions()
        optimal_q = self._value_iteration()
        self.q = float(q_init_blend) * optimal_q
        self.q += self.rng.normal(0.0, float(q_init_noise), size=self.q.shape)
        self.states = self._select_replay_states(int(replay_state_count))
        self.perturbation_cache = self._build_perturbation_cache()

    def _free_cells(self) -> tuple[tuple[int, int], ...]:
        cells: list[tuple[int, int]] = []
        for x in range(self.width):
            for y in range(self.height):
                obj = self.grid.get(x, y)
                if obj is None or obj.can_overlap():
                    cells.append((x, y))
        return tuple(cells)

    def _goal_cells(self) -> set[tuple[int, int]]:
        goals: set[tuple[int, int]] = set()
        for x, y in self.free_cells:
            obj = self.grid.get(x, y)
            if obj is not None and obj.type == "goal":
                goals.add((x, y))
        return goals

    def _set_state(self, state_idx: int) -> None:
        x, y, direction = self.idx_to_state[int(state_idx)]
        self.unwrapped.grid = self.grid.copy()
        self.unwrapped.agent_pos = (int(x), int(y))
        self.unwrapped.agent_dir = int(direction)
        self.unwrapped.step_count = 0
        self.unwrapped.carrying = None

    def _step_from(self, state_idx: int, action_idx: int) -> tuple[int, float, bool]:
        self._set_state(state_idx)
        _obs, reward, terminated, truncated, _info = self.env.step(int(MOVE_ACTIONS[action_idx]))
        done = bool(terminated or truncated)
        if bool(terminated):
            return state_idx, 1.0, True
        if bool(truncated):
            return state_idx, 0.0, True
        x, y = self.unwrapped.agent_pos
        direction = int(self.unwrapped.agent_dir)
        next_idx = self.state_to_idx[(int(x), int(y), direction)]
        return next_idx, float(reward), done

    def _build_transitions(self) -> list[list[tuple[int, float, bool]]]:
        return [[self._step_from(state_idx, action_idx) for action_idx in range(len(MOVE_ACTIONS))] for state_idx in range(len(self.state_to_idx))]

    def _value_iteration(self, iterations: int = 900) -> np.ndarray:
        q = np.zeros((len(self.state_to_idx), len(MOVE_ACTIONS)), dtype=float)
        for _ in range(iterations):
            next_q = q.copy()
            max_delta = 0.0
            for state_idx, row in enumerate(self.transitions):
                for action_idx, (next_idx, reward, done) in enumerate(row):
                    target = reward if done else reward + self.discount * float(np.max(q[next_idx]))
                    max_delta = max(max_delta, abs(target - next_q[state_idx, action_idx]))
                    next_q[state_idx, action_idx] = target
            q = next_q
            if max_delta < 1e-10:
                break
        return q

    def _select_replay_states(self, count: int) -> list[MiniGridReplayState]:
        bottlenecks = self._bottleneck_cells()
        distances = self._distances_to_goal()
        candidates: list[tuple[int, int, int, MiniGridReplayState]] = []
        for x, y in self.free_cells:
            if (x, y) in self.goal_cells:
                continue
            nearest_bottleneck = min(abs(x - bx) + abs(y - by) for bx, by in bottlenecks)
            if nearest_bottleneck <= 2:
                for direction in DIRS:
                    dist = distances.get((x, y), self.width * self.height)
                    candidates.append((dist, nearest_bottleneck, direction, MiniGridReplayState(x, y, direction)))
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].x, item[3].y))
        if count >= len(candidates):
            return [item[3] for item in candidates]
        indexes = np.linspace(0, len(candidates) - 1, count, dtype=int)
        return [candidates[int(index)][3] for index in indexes]

    def _bottleneck_cells(self) -> tuple[tuple[int, int], ...]:
        cells: list[tuple[int, int]] = []
        for x, y in self.free_cells:
            if (x, y) in self.goal_cells:
                continue
            wall_neighbors = 0
            for dx, dy in NEIGHBORS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    obj = self.grid.get(nx, ny)
                    wall_neighbors += int(obj is not None and obj.type == "wall")
            if wall_neighbors >= 2:
                cells.append((x, y))
        return tuple(cells)

    def _distances_to_goal(self) -> dict[tuple[int, int], int]:
        frontier = list(self.goal_cells)
        distances = {cell: 0 for cell in frontier}
        while frontier:
            x, y = frontier.pop(0)
            for dx, dy in NEIGHBORS:
                nx, ny = x + dx, y + dy
                if (nx, ny) in distances or (nx, ny) not in self.free_cells:
                    continue
                distances[(nx, ny)] = distances[(x, y)] + 1
                frontier.append((nx, ny))
        return distances

    def _build_perturbation_cache(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        cache: list[tuple[tuple[int, ...], ...]] = []
        free_non_goal = [(x, y) for x, y in self.free_cells if (x, y) not in self.goal_cells]
        for replay in self.states:
            radii: list[tuple[int, ...]] = []
            for radius in range(self.max_radius + 1):
                states = tuple(
                    self.state_to_idx[(x, y, replay.direction)]
                    for x, y in free_non_goal
                    if abs(x - replay.x) + abs(y - replay.y) == radius
                )
                radii.append(states or (self.state_to_idx[(replay.x, replay.y, replay.direction)],))
            cache.append(tuple(radii))
        return tuple(cache)

    def perturbation_states(self, replay_idx: int, sigma: float) -> tuple[int, ...]:
        radius = int(round(float(np.clip(sigma, 0.0, 1.0)) * self.max_radius))
        return self.perturbation_cache[replay_idx][radius]

    def current_success_values(self) -> np.ndarray:
        policy = np.argmax(self.q, axis=1)
        return np.array([float(self._rollout_state(state_idx, policy=policy, train=False, epsilon=False)) for state_idx in range(len(self.state_to_idx))], dtype=float)

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
        next_state, reward, done = self.transitions[state][action]
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
        for _ in range(self.rollout_horizon):
            action = int(policy[state]) if policy is not None else self._action(state, rng, epsilon=epsilon)
            next_state, reward, done = self.transitions[state][action]
            if train:
                target = reward if done else reward + self.discount * float(np.max(self.q[next_state]))
                self.q[state, action] += self.learning_rate * (target - float(self.q[state, action]))
            state = next_state
            if done and reward > 0.0:
                return True
            if done:
                return False
        return False

    def _action(self, state: int, rng: np.random.Generator | None, *, epsilon: bool = True) -> int:
        if epsilon and rng is not None and rng.random() < self.epsilon:
            return int(rng.integers(len(MOVE_ACTIONS)))
        return int(np.argmax(self.q[state]))


def run_method(args: argparse.Namespace, method: str, seed: int) -> MiniGridProbeResult:
    rng = np.random.default_rng(seed + 339_000)
    bench = OfficialMiniGridFourRoomsProbe(
        seed=seed,
        replay_state_count=args.replay_states,
        max_radius=args.max_radius,
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        q_init_blend=args.q_init_blend,
        q_init_noise=args.q_init_noise,
        rollout_horizon=args.rollout_horizon,
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
    return MiniGridProbeResult(
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


def init_records(bench: OfficialMiniGridFourRoomsProbe, rng: np.random.Generator, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, replay in enumerate(bench.states):
        record = LevelRecord(
            id=f"minigrid_fourrooms_{replay.x}_{replay.y}_{replay.direction}",
            domain="minigrid_fourrooms_recovery",
            task_id=bench.env_id,
            clean_success_hat=bench.success_prob(replay_idx, 0.0),
            feasibility_hat=1.0,
            perturbation_family="official_minigrid_position_restart",
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
    bench: OfficialMiniGridFourRoomsProbe,
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


def write_estimate(record: LevelRecord, estimate: Any) -> None:
    record.r_alpha_hat = estimate.r_alpha
    record.sharpness_hat = estimate.sharpness
    record.uncertainty_hat = estimate.r_uncertainty
    record.recovery_curve_hat = estimate.recovery.tolist()


def evaluate(bench: OfficialMiniGridFourRoomsProbe, args: argparse.Namespace) -> dict[str, float]:
    grid = np.linspace(0.0, 1.0, args.eval_grid_size)
    values = bench.current_success_values()
    clean: list[float] = []
    raucs: list[float] = []
    radii: list[float] = []
    abs_r10: list[float] = []
    for replay_idx in range(len(bench.states)):
        curve = np.array([bench.success_prob_from_values(replay_idx, sigma, values) for sigma in grid], dtype=float)
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
    bench: OfficialMiniGridFourRoomsProbe,
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
    parser = argparse.ArgumentParser(description="Run an internal official MiniGrid-FourRooms recovery replay diagnostic.")
    parser.add_argument("--out", default="runs/minigrid_fourrooms_recovery_probe")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--replay-states", type=int, default=32)
    parser.add_argument("--max-radius", type=int, default=5)
    parser.add_argument("--rollout-horizon", type=int, default=50)
    parser.add_argument("--eval-grid-size", type=int, default=9)
    parser.add_argument("--learning-rate", type=float, default=0.30)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--q-init-blend", type=float, default=0.03)
    parser.add_argument("--q-init-noise", type=float, default=0.06)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--absolute-radius-alpha", type=float, default=0.10)
    parser.add_argument("--target-radius", type=float, default=0.45)
    parser.add_argument("--radius-bandwidth", type=float, default=0.35)
    parser.add_argument("--fixed-radius", type=float, default=0.65)
    parser.add_argument("--baseline-candidates", type=int, default=12)
    parser.add_argument("--initial-probes", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--min-trials", type=int, default=1)
    parser.add_argument("--refresh-per-eval", type=int, default=16)
    parser.add_argument("--refresh-trials", type=int, default=1)
    parser.add_argument("--refresh-jitter", type=float, default=0.08)
    parser.add_argument("--radius-noise", type=float, default=0.08)
    parser.add_argument("--radius-uniform-mix", type=float, default=0.25)
    parser.add_argument("--priority-temperature", type=float, default=0.8)
    parser.add_argument("--uniform-mix", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    versions = package_versions()
    methods = parse_csv_strings(args.methods)
    seeds = parse_csv_ints(args.seeds)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
                    "final_abs_r10": result.final_abs_r10,
                    "rauc_aulc": result.rauc_aulc,
                    "best_rauc": result.best_rauc,
                }
            )

    metadata = {"args": vars(args), "package_versions": versions, "results": results}
    (out_dir / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "package_versions.json").write_text(json.dumps(versions, indent=2, sort_keys=True), encoding="utf-8")
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
    lines = ["method,final_clean_mean,final_rauc_mean,final_median_r80_mean,final_abs_r10_mean,rauc_aulc_mean,best_rauc_mean"]
    for method, method_rows in sorted(by_method.items()):
        lines.append(
            ",".join(
                [
                    method,
                    f"{mean(method_rows, 'final_clean'):.4f}",
                    f"{mean(method_rows, 'final_rauc'):.4f}",
                    f"{mean(method_rows, 'final_median_r80'):.4f}",
                    f"{mean(method_rows, 'final_abs_r10'):.4f}",
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
