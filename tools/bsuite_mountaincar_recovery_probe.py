#!/usr/bin/env python3
"""Run a fixed bsuite MountainCar recovery replay screen.

This optional external-package screen uses bsuite's package-owned MountainCar
task constants and one-step dynamics through exact private restart fields. The
perturbation family moves replay states away from right-moving progress and
back toward the low-energy valley anchor, creating a recovery curve over how
much momentum/position recovery is required before reaching the goal.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from bgr.curve_estimators import IsotonicCurveEstimator
from bgr.metrics import critical_radius, recovery_auc
from bgr.priorities import BGRPriorityScorer
from bgr.records import LevelRecord
from bgr.samplers import mixed_priority_probs, sample_boundary_radius


POSITION_MIN = -1.2
POSITION_MAX = 0.6
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07
GOAL_POSITION = 0.5
ACTION_FORCES = (-1.0, 0.0, 1.0)
LOW_ENERGY_ANCHOR = (-0.52, 0.0)


@dataclass(frozen=True, slots=True)
class MountainCarReplayState:
    position: float
    velocity: float


@dataclass(frozen=True, slots=True)
class MountainCarProbeResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


def package_versions() -> dict[str, str]:
    try:
        import bsuite
        import dm_env
    except ImportError as exc:
        raise SystemExit(
            "bsuite MountainCar recovery probe requires bsuite in an isolated environment, "
            "for example /tmp/bgr_bsuite_venv."
        ) from exc
    return {
        "bsuite": getattr(bsuite, "__version__", version("bsuite")),
        "bsuite-package": version("bsuite"),
        "dm-env": version("dm-env"),
        "numpy": np.__version__,
    }


def step_state(position: float, velocity: float, action: int) -> tuple[float, float, bool]:
    velocity += 0.001 * ACTION_FORCES[action] - 0.0025 * np.cos(3.0 * position)
    velocity = float(np.clip(velocity, VELOCITY_MIN, VELOCITY_MAX))
    position += velocity
    position = float(np.clip(position, POSITION_MIN, POSITION_MAX))
    if position <= POSITION_MIN and velocity < 0.0:
        velocity = 0.0
    return position, velocity, position >= GOAL_POSITION


class MountainCarRecoveryProbe:
    def __init__(
        self,
        *,
        seed: int,
        replay_state_count: int,
        position_bins: int,
        velocity_bins: int,
        learning_rate: float,
        discount: float,
        epsilon: float,
        q_init_blend: float,
        q_init_noise: float,
        max_steps: int,
        perturb_position_scale: float,
        perturb_velocity_scale: float,
        value_iterations: int,
        perturbation_mode: str,
    ) -> None:
        try:
            from bsuite.environments.mountain_car import MountainCar
        except ImportError as exc:
            raise SystemExit(
                "bsuite MountainCar recovery probe requires bsuite in an isolated environment, "
                "for example /tmp/bgr_bsuite_venv."
            ) from exc

        self.rng = np.random.default_rng(seed + 83_000)
        self.env = MountainCar(max_steps=max_steps, seed=seed + 41_000)
        self.position_bins = int(position_bins)
        self.velocity_bins = int(velocity_bins)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.epsilon = float(epsilon)
        self.max_steps = int(max_steps)
        self.perturb_position_scale = float(perturb_position_scale)
        self.perturb_velocity_scale = float(perturb_velocity_scale)
        self.perturbation_mode = str(perturbation_mode)
        self.position_grid = np.linspace(POSITION_MIN, POSITION_MAX, self.position_bins)
        self.velocity_grid = np.linspace(VELOCITY_MIN, VELOCITY_MAX, self.velocity_bins)
        optimal_q = self._value_iteration(iterations=int(value_iterations))
        heuristic_q = self._energy_pumping_q()
        self.q = float(q_init_blend) * heuristic_q + max(0.0, 1.0 - float(q_init_blend)) * 0.05 * optimal_q
        self.q += self.rng.normal(0.0, float(q_init_noise), size=self.q.shape)
        self.states = self._select_replay_states(int(replay_state_count))

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def step_state(self, position: float, velocity: float, action: int) -> tuple[float, float, bool]:
        self.env._position = float(position)
        self.env._velocity = float(velocity)
        self.env._timestep = 0
        self.env._step(int(action))
        return (
            float(self.env._position),
            float(self.env._velocity),
            bool(float(self.env._position) >= float(self.env._goal_pos)),
        )

    def _select_replay_states(self, count: int) -> list[MountainCarReplayState]:
        positions = np.linspace(-0.55, 0.30, 12)
        velocities = np.linspace(0.005, 0.060, 9)
        candidates = [
            MountainCarReplayState(float(position), float(velocity))
            for position in positions
            for velocity in velocities
        ]
        candidates.sort(key=lambda item: (item.position, item.velocity))
        if count >= len(candidates):
            return candidates
        indexes = np.linspace(0, len(candidates) - 1, count, dtype=int)
        return [candidates[int(index)] for index in indexes]

    def _value_iteration(self, iterations: int = 900) -> np.ndarray:
        q = np.zeros((self.position_bins, self.velocity_bins, 3), dtype=float)
        for _ in range(iterations):
            next_q = q.copy()
            max_delta = 0.0
            for pos_idx, position in enumerate(self.position_grid):
                for vel_idx, velocity in enumerate(self.velocity_grid):
                    if position >= GOAL_POSITION:
                        next_q[pos_idx, vel_idx, :] = 0.0
                        continue
                    for action in range(3):
                        next_position, next_velocity, done = self.step_state(float(position), float(velocity), action)
                        target = 0.0 if done else -1.0 + self.discount * float(np.max(q[self._index(next_position, next_velocity)]))
                        max_delta = max(max_delta, abs(target - next_q[pos_idx, vel_idx, action]))
                        next_q[pos_idx, vel_idx, action] = target
            q = next_q
            if max_delta < 1e-5:
                break
        return q

    def _energy_pumping_q(self) -> np.ndarray:
        q = np.zeros((self.position_bins, self.velocity_bins, 3), dtype=float)
        for pos_idx, _position in enumerate(self.position_grid):
            for vel_idx, velocity in enumerate(self.velocity_grid):
                preferred = 2 if velocity >= 0.0 else 0
                q[pos_idx, vel_idx, :] = -0.05
                q[pos_idx, vel_idx, preferred] = 0.05
        return q

    def _index(self, position: float, velocity: float) -> tuple[int, int]:
        pos_idx = int(np.argmin(np.abs(self.position_grid - position)))
        vel_idx = int(np.argmin(np.abs(self.velocity_grid - velocity)))
        return pos_idx, vel_idx

    def clean_success(self, replay_idx: int) -> float:
        return self.success_prob(replay_idx, 0.0)

    def success_prob(self, replay_idx: int, sigma: float) -> float:
        starts = self.evaluation_starts(replay_idx, sigma)
        return float(np.mean([self.rollout_state(position, velocity, train=False, epsilon=False) for position, velocity in starts]))

    def evaluation_starts(self, replay_idx: int, sigma: float) -> list[tuple[float, float]]:
        replay = self.states[replay_idx]
        sigma = float(np.clip(sigma, 0.0, 1.0))
        if sigma == 0.0:
            return [(replay.position, replay.velocity)]
        if self.perturbation_mode == "adverse":
            center_position, center_velocity = self._adverse_center(replay, sigma)
            offsets = [(0.0, 0.0), (-0.15, 0.0), (0.15, 0.0), (0.0, -0.15), (0.0, 0.15)]
            return [
                self._clip_state(
                    center_position + dx * sigma * self.perturb_position_scale,
                    center_velocity + dv * sigma * self.perturb_velocity_scale,
                )
                for dx, dv in offsets
            ]
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        starts: list[tuple[float, float]] = []
        for angle in angles:
            position = np.clip(
                replay.position + sigma * self.perturb_position_scale * np.cos(angle),
                POSITION_MIN,
                GOAL_POSITION - 1e-6,
            )
            velocity = np.clip(
                replay.velocity + sigma * self.perturb_velocity_scale * np.sin(angle),
                VELOCITY_MIN,
                VELOCITY_MAX,
            )
            starts.append((float(position), float(velocity)))
        return starts

    def sample_start(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> tuple[float, float]:
        replay = self.states[replay_idx]
        sigma = float(np.clip(sigma, 0.0, 1.0))
        if sigma == 0.0:
            return replay.position, replay.velocity
        if self.perturbation_mode == "adverse":
            center_position, center_velocity = self._adverse_center(replay, sigma)
            position = center_position + rng.normal(0.0, 0.10 * sigma * self.perturb_position_scale)
            velocity = center_velocity + rng.normal(0.0, 0.10 * sigma * self.perturb_velocity_scale)
            return self._clip_state(position, velocity)
        radius = sigma * np.sqrt(float(rng.uniform(0.0, 1.0)))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        position = np.clip(
            replay.position + radius * self.perturb_position_scale * np.cos(angle),
            POSITION_MIN,
            GOAL_POSITION - 1e-6,
        )
        velocity = np.clip(
            replay.velocity + radius * self.perturb_velocity_scale * np.sin(angle),
            VELOCITY_MIN,
            VELOCITY_MAX,
        )
        return float(position), float(velocity)

    def _adverse_center(self, replay: MountainCarReplayState, sigma: float) -> tuple[float, float]:
        anchor_position, anchor_velocity = LOW_ENERGY_ANCHOR
        position = (1.0 - sigma) * replay.position + sigma * anchor_position
        velocity = (1.0 - sigma) * replay.velocity + sigma * anchor_velocity
        return self._clip_state(position, velocity)

    def _clip_state(self, position: float, velocity: float) -> tuple[float, float]:
        return (
            float(np.clip(position, POSITION_MIN, GOAL_POSITION - 1e-6)),
            float(np.clip(velocity, VELOCITY_MIN, VELOCITY_MAX)),
        )

    def rollout(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> bool:
        position, velocity = self.sample_start(replay_idx, sigma, rng)
        return self.rollout_state(position, velocity, train=False, epsilon=True, rng=rng)

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> None:
        position, velocity = self.sample_start(replay_idx, sigma, rng)
        self.rollout_state(position, velocity, train=True, epsilon=True, rng=rng)

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        position, velocity = self.sample_start(replay_idx, sigma, rng)
        pos_idx, vel_idx = self._index(position, velocity)
        action = self._action(position, velocity, rng, epsilon=True)
        next_position, next_velocity, done = self.step_state(position, velocity, action)
        target = 0.0 if done else -1.0 + self.discount * float(np.max(self.q[self._index(next_position, next_velocity)]))
        return abs(target - float(self.q[pos_idx, vel_idx, action]))

    def rollout_state(
        self,
        position: float,
        velocity: float,
        *,
        train: bool,
        epsilon: bool,
        rng: np.random.Generator | None = None,
    ) -> bool:
        for _ in range(self.max_steps):
            action = self._action(position, velocity, rng, epsilon=epsilon)
            pos_idx, vel_idx = self._index(position, velocity)
            next_position, next_velocity, done = self.step_state(position, velocity, action)
            if train:
                target = 0.0 if done else -1.0 + self.discount * float(np.max(self.q[self._index(next_position, next_velocity)]))
                self.q[pos_idx, vel_idx, action] += self.learning_rate * (target - float(self.q[pos_idx, vel_idx, action]))
            position, velocity = next_position, next_velocity
            if done:
                return True
        return False

    def _action(
        self,
        position: float,
        velocity: float,
        rng: np.random.Generator | None,
        *,
        epsilon: bool,
    ) -> int:
        if epsilon and rng is not None and rng.random() < self.epsilon:
            return int(rng.integers(3))
        return int(np.argmax(self.q[self._index(position, velocity)]))


def run_method(args: argparse.Namespace, method: str, seed: int) -> MountainCarProbeResult:
    rng = np.random.default_rng(seed + 89_000)
    bench = MountainCarRecoveryProbe(
        seed=seed,
        replay_state_count=args.replay_states,
        position_bins=args.position_bins,
        velocity_bins=args.velocity_bins,
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        q_init_blend=args.q_init_blend,
        q_init_noise=args.q_init_noise,
        max_steps=args.max_steps,
        perturb_position_scale=args.perturb_position_scale,
        perturb_velocity_scale=args.perturb_velocity_scale,
        value_iterations=args.value_iterations,
        perturbation_mode=args.perturbation_mode,
    )
    try:
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
        return MountainCarProbeResult(
            method=method,
            seed=seed,
            final_clean=final["clean"],
            final_rauc=final["rauc"],
            final_median_r80=final["median_r80"],
            rauc_aulc=float(np.trapezoid(ys, xs) / (xs[-1] - xs[0])) if xs[-1] > xs[0] else final["rauc"],
            best_rauc=max(row["rauc"] for row in history),
            history=history,
        )
    finally:
        bench.close()


def init_records(bench: MountainCarRecoveryProbe, rng: np.random.Generator, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, replay in enumerate(bench.states):
        record = LevelRecord(
            id=f"mountaincar_{replay_idx}",
            domain="mountaincar_v0_recovery",
            task_id="MountainCar-v0",
            clean_success_hat=bench.clean_success(replay_idx),
            feasibility_hat=1.0,
            perturbation_family="position_velocity_ellipsoid_restart",
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
    bench: MountainCarRecoveryProbe,
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


def evaluate(bench: MountainCarRecoveryProbe, args: argparse.Namespace) -> dict[str, float]:
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
    bench: MountainCarRecoveryProbe,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/bsuite_mountaincar_recovery_probe")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=6)
    parser.add_argument("--replay-states", type=int, default=48)
    parser.add_argument("--position-bins", type=int, default=37)
    parser.add_argument("--velocity-bins", type=int, default=37)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--value-iterations", type=int, default=50)
    parser.add_argument("--eval-grid-size", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.12)
    parser.add_argument("--q-init-blend", type=float, default=1.0)
    parser.add_argument("--q-init-noise", type=float, default=0.02)
    parser.add_argument("--perturb-position-scale", type=float, default=0.25)
    parser.add_argument("--perturb-velocity-scale", type=float, default=0.040)
    parser.add_argument("--perturbation-mode", choices=["adverse", "ellipsoid"], default="adverse")
    parser.add_argument("--alpha", type=float, default=0.8)
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
                    "rauc_aulc": result.rauc_aulc,
                    "best_rauc": result.best_rauc,
                }
            )

    (out_dir / "results.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "package_versions.json").write_text(json.dumps(package_versions(), indent=2, sort_keys=True), encoding="utf-8")
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
