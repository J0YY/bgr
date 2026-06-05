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


MAX_SPEED = 8.0
MAX_TORQUE = 2.0
DT = 0.05
G = 10.0
M = 1.0
L = 1.0
DOWN_ANCHOR = np.array([1.6, 0.0], dtype=float)
FEATURE_SCALE = np.array([1.0, 1.0, 0.25, 1.0], dtype=float)
EXPERT_WEIGHTS = np.array([-12.0, 0.0, -2.4, 0.0], dtype=float)


@dataclass(frozen=True, slots=True)
class PendulumReplayState:
    state: tuple[float, float]


@dataclass(frozen=True, slots=True)
class PendulumProbeResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


def angle_normalize(theta: float) -> float:
    return float(((theta + np.pi) % (2.0 * np.pi)) - np.pi)


def clip_state(state: np.ndarray) -> np.ndarray:
    clipped = np.array(state, dtype=float)
    clipped[0] = angle_normalize(float(clipped[0]))
    clipped[1] = float(np.clip(clipped[1], -MAX_SPEED, MAX_SPEED))
    return clipped


def pendulum_step(state: np.ndarray, torque: float) -> np.ndarray:
    theta, theta_dot = float(state[0]), float(state[1])
    torque = float(np.clip(torque, -MAX_TORQUE, MAX_TORQUE))
    new_theta_dot = theta_dot + (3.0 * G / (2.0 * L) * np.sin(theta) + 3.0 / (M * L**2) * torque) * DT
    new_theta_dot = float(np.clip(new_theta_dot, -MAX_SPEED, MAX_SPEED))
    new_theta = angle_normalize(theta + new_theta_dot * DT)
    return np.array([new_theta, new_theta_dot], dtype=float)


def features(state: np.ndarray) -> np.ndarray:
    theta, theta_dot = float(state[0]), float(state[1])
    return FEATURE_SCALE * np.array([np.sin(theta), 1.0 - np.cos(theta), theta_dot, 1.0], dtype=float)


def expert_action(state: np.ndarray) -> float:
    return float(np.clip(np.dot(EXPERT_WEIGHTS, features(state)), -MAX_TORQUE, MAX_TORQUE))


class PendulumRecoveryProbe:
    def __init__(
        self,
        *,
        seed: int,
        replay_state_count: int,
        learning_rate: float,
        policy_init_blend: float,
        policy_init_noise: float,
        max_steps: int,
        success_angle: float,
        success_velocity: float,
        perturb_angle_scale: float,
        perturb_velocity_scale: float,
    ) -> None:
        self.rng = np.random.default_rng(seed + 149_000)
        self.learning_rate = float(learning_rate)
        self.max_steps = int(max_steps)
        self.success_angle = float(success_angle)
        self.success_velocity = float(success_velocity)
        self.perturb_angle_scale = float(perturb_angle_scale)
        self.perturb_velocity_scale = float(perturb_velocity_scale)
        self.weights = float(policy_init_blend) * EXPERT_WEIGHTS
        self.weights += self.rng.normal(0.0, float(policy_init_noise), size=self.weights.shape)
        self.states = self._select_replay_states(int(replay_state_count))

    def _select_replay_states(self, count: int) -> list[PendulumReplayState]:
        angles = np.linspace(-0.32, 0.32, 13)
        velocities = np.linspace(-0.95, 0.95, 13)
        candidates: list[tuple[float, PendulumReplayState]] = []
        for theta in angles:
            for theta_dot in velocities:
                difficulty = abs(theta) + 0.16 * abs(theta_dot)
                if difficulty < 0.04:
                    continue
                replay = PendulumReplayState((float(theta), float(theta_dot)))
                candidates.append((difficulty, replay))
        candidates.sort(key=lambda item: (item[0], abs(item[1].state[0]), abs(item[1].state[1])))
        if count >= len(candidates):
            selected = candidates
        else:
            indexes = np.linspace(0, len(candidates) - 1, count, dtype=int)
            selected = [candidates[int(index)] for index in indexes]
        return [item[1] for item in selected]

    def policy_action(self, state: np.ndarray) -> float:
        return float(np.clip(np.dot(self.weights, features(state)), -MAX_TORQUE, MAX_TORQUE))

    def clean_success(self, replay_idx: int) -> float:
        return self.success_prob(replay_idx, 0.0)

    def success_prob(self, replay_idx: int, sigma: float) -> float:
        starts = self.evaluation_starts(replay_idx, sigma)
        return float(np.mean([self.rollout_state(start, train=False) for start in starts]))

    def evaluation_starts(self, replay_idx: int, sigma: float) -> list[np.ndarray]:
        replay = np.array(self.states[replay_idx].state, dtype=float)
        sigma = float(np.clip(sigma, 0.0, 1.0))
        if sigma == 0.0:
            return [replay]
        center = self._adverse_center(replay, sigma)
        directions = [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
            np.array([0.7, 0.7]),
            np.array([-0.7, -0.7]),
        ]
        scale = np.array([self.perturb_angle_scale, self.perturb_velocity_scale], dtype=float)
        return [clip_state(center + sigma * scale * direction) for direction in directions]

    def sample_start(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
        replay = np.array(self.states[replay_idx].state, dtype=float)
        sigma = float(np.clip(sigma, 0.0, 1.0))
        if sigma == 0.0:
            return replay
        center = self._adverse_center(replay, sigma)
        direction = rng.normal(0.0, 1.0, size=2)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            direction = np.array([1.0, 0.0])
        else:
            direction /= norm
        radius = sigma * np.sqrt(float(rng.uniform(0.0, 1.0)))
        scale = np.array([self.perturb_angle_scale, self.perturb_velocity_scale], dtype=float)
        return clip_state(center + radius * scale * direction)

    def _adverse_center(self, replay: np.ndarray, sigma: float) -> np.ndarray:
        direction = np.array([angle_normalize(float(DOWN_ANCHOR[0] - replay[0])), DOWN_ANCHOR[1] - replay[1]])
        return clip_state(replay + sigma * direction)

    def rollout(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> bool:
        return self.rollout_state(self.sample_start(replay_idx, sigma, rng), train=False)

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> None:
        self.rollout_state(self.sample_start(replay_idx, sigma, rng), train=True)

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        state = self.sample_start(replay_idx, sigma, rng)
        return abs(expert_action(state) - self.policy_action(state))

    def rollout_state(self, state: np.ndarray, *, train: bool) -> bool:
        state = clip_state(state)
        stable_steps = 0
        for _ in range(self.max_steps):
            if train:
                self._teacher_update(state)
            torque = self.policy_action(state)
            state = pendulum_step(state, torque)
            if abs(float(state[0])) <= self.success_angle and abs(float(state[1])) <= self.success_velocity:
                stable_steps += 1
            else:
                stable_steps = 0
        return stable_steps >= max(4, self.max_steps // 10)

    def _teacher_update(self, state: np.ndarray) -> None:
        phi = features(state)
        prediction = float(np.dot(self.weights, phi))
        target = expert_action(state)
        error = prediction - target
        if abs(error) > 1e-6:
            self.weights -= self.learning_rate * error * phi


def run_method(args: argparse.Namespace, method: str, seed: int) -> PendulumProbeResult:
    rng = np.random.default_rng(seed + 151_000)
    bench = PendulumRecoveryProbe(
        seed=seed,
        replay_state_count=args.replay_states,
        learning_rate=args.learning_rate,
        policy_init_blend=args.policy_init_blend,
        policy_init_noise=args.policy_init_noise,
        max_steps=args.max_steps,
        success_angle=args.success_angle,
        success_velocity=args.success_velocity,
        perturb_angle_scale=args.perturb_angle_scale,
        perturb_velocity_scale=args.perturb_velocity_scale,
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
    return PendulumProbeResult(
        method=method,
        seed=seed,
        final_clean=final["clean"],
        final_rauc=final["rauc"],
        final_median_r80=final["median_r80"],
        rauc_aulc=float(np.trapezoid(ys, xs) / (xs[-1] - xs[0])) if xs[-1] > xs[0] else final["rauc"],
        best_rauc=max(row["rauc"] for row in history),
        history=history,
    )


def init_records(bench: PendulumRecoveryProbe, rng: np.random.Generator, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, _replay in enumerate(bench.states):
        record = LevelRecord(
            id=f"pendulum_{replay_idx}",
            domain="pendulum_v1_recovery",
            task_id="Pendulum-v1",
            clean_success_hat=bench.clean_success(replay_idx),
            feasibility_hat=1.0,
            perturbation_family="adverse_angle_velocity_restart",
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
    bench: PendulumRecoveryProbe,
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


def evaluate(bench: PendulumRecoveryProbe, args: argparse.Namespace) -> dict[str, float]:
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
    bench: PendulumRecoveryProbe,
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
    parser = argparse.ArgumentParser(description="Run an internal Pendulum-v1 recovery replay diagnostic.")
    parser.add_argument("--out", default="runs/pendulum_recovery_probe")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--replay-states", type=int, default=48)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--eval-grid-size", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--policy-init-blend", type=float, default=0.85)
    parser.add_argument("--policy-init-noise", type=float, default=0.20)
    parser.add_argument("--success-angle", type=float, default=0.60)
    parser.add_argument("--success-velocity", type=float, default=2.25)
    parser.add_argument("--perturb-angle-scale", type=float, default=0.35)
    parser.add_argument("--perturb-velocity-scale", type=float, default=0.90)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--target-radius", type=float, default=0.45)
    parser.add_argument("--radius-bandwidth", type=float, default=0.35)
    parser.add_argument("--fixed-radius", type=float, default=0.65)
    parser.add_argument("--baseline-candidates", type=int, default=8)
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
