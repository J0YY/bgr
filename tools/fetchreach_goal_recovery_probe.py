#!/usr/bin/env python3
"""Run a preregistered FetchReach-v4 goal-recovery replay screen."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bgr.curve_estimators import IsotonicCurveEstimator
from bgr.metrics import critical_radius, recovery_auc
from bgr.priorities import BGRPriorityScorer
from bgr.records import LevelRecord
from bgr.samplers import mixed_priority_probs, sample_boundary_radius
from fetchreach_goal_recovery_calibration import adverse_direction, clipped_goal, package_versions


@dataclass(frozen=True, slots=True)
class FetchReplayState:
    base_seed: int
    base_goal: list[float]
    center: list[float]
    target_range: float
    direction: list[float]


@dataclass(frozen=True, slots=True)
class FetchProbeResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


class LinearGoalController:
    def __init__(self, rng: np.random.Generator, *, init_gain: float, init_noise: float, learning_rate: float) -> None:
        self.weights = np.eye(3, dtype=float) * float(init_gain)
        self.weights += rng.normal(0.0, float(init_noise), size=(3, 3))
        self.learning_rate = float(learning_rate)

    def action(self, delta: np.ndarray) -> np.ndarray:
        action = np.zeros(4, dtype=float)
        action[:3] = np.clip(self.weights @ delta, -1.0, 1.0)
        return action

    def update(self, delta: np.ndarray, target: np.ndarray) -> float:
        pred = self.weights @ delta
        error = pred - target
        self.weights -= self.learning_rate * np.outer(error, delta)
        return float(np.mean(error * error))


class FetchReachProbe:
    def __init__(self, args: argparse.Namespace, seed: int) -> None:
        import gymnasium as gym
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)
        self.env = gym.make(args.env_id)
        self.args = args
        self.rng = np.random.default_rng(args.seed_offset + seed)
        self.policy = LinearGoalController(
            self.rng,
            init_gain=args.init_gain,
            init_noise=args.init_noise,
            learning_rate=args.learning_rate,
        )
        self.states = self._init_states(seed)

    def close(self) -> None:
        self.env.close()

    def _init_states(self, seed: int) -> list[FetchReplayState]:
        states: list[FetchReplayState] = []
        for replay_idx in range(self.args.replay_states):
            base_seed = self.args.seed_offset + seed * 1000 + replay_idx
            obs, _info = self.env.reset(seed=base_seed)
            unwrapped = self.env.unwrapped
            center = np.array(unwrapped.initial_gripper_xpos, dtype=float)
            base_goal = np.array(obs["desired_goal"], dtype=float)
            direction = adverse_direction(base_goal, center, self.rng, 0.0)
            states.append(
                FetchReplayState(
                    base_seed=base_seed,
                    base_goal=base_goal.tolist(),
                    center=center.tolist(),
                    target_range=float(unwrapped.target_range),
                    direction=direction.tolist(),
                )
            )
        return states

    def goal(self, replay_idx: int, sigma: float) -> np.ndarray:
        state = self.states[replay_idx]
        return clipped_goal(
            np.array(state.base_goal, dtype=float),
            np.array(state.direction, dtype=float),
            float(sigma),
            np.array(state.center, dtype=float),
            float(state.target_range),
        )

    def rollout(self, replay_idx: int, sigma: float, trial_seed: int, *, train: bool) -> tuple[bool, float]:
        obs, _info = self.env.reset(seed=trial_seed)
        unwrapped = self.env.unwrapped
        unwrapped.goal = self.goal(replay_idx, sigma)
        obs = unwrapped._get_obs()
        losses: list[float] = []
        for _step in range(self.args.horizon):
            delta = np.array(obs["desired_goal"], dtype=float) - np.array(obs["achieved_goal"], dtype=float)
            if train:
                target = np.clip(self.args.teacher_gain * delta, -1.0, 1.0)
                losses.append(self.policy.update(delta, target))
            obs, _reward, terminated, truncated, _info = self.env.step(self.policy.action(delta))
            if bool(terminated):
                return True, float(np.mean(losses)) if losses else 0.0
            if bool(truncated):
                return False, float(np.mean(losses)) if losses else 0.0
        distance = float(np.linalg.norm(np.array(obs["achieved_goal"], dtype=float) - np.array(obs["desired_goal"], dtype=float)))
        return distance <= float(unwrapped.distance_threshold), float(np.mean(losses)) if losses else 0.0

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> tuple[bool, float]:
        trial_seed = int(rng.integers(1_000_000_000))
        return self.rollout(replay_idx, sigma, trial_seed, train=True)

    def success_prob(self, replay_idx: int, sigma: float, rng: np.random.Generator, trials: int) -> float:
        successes = 0
        for _ in range(trials):
            trial_seed = int(rng.integers(1_000_000_000))
            success, _loss = self.rollout(replay_idx, sigma, trial_seed, train=False)
            successes += int(success)
        return successes / trials

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        _success, loss = self.rollout(replay_idx, sigma, int(rng.integers(1_000_000_000)), train=True)
        return loss


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_estimate(record: LevelRecord, estimate: Any) -> None:
    record.r_alpha_hat = float(estimate.r_alpha)
    record.sharpness_hat = float(estimate.sharpness)
    record.uncertainty_hat = float(estimate.r_uncertainty)
    record.recovery_curve_hat = estimate.recovery.tolist()


def init_records(bench: FetchReachProbe, rng: np.random.Generator, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, _state in enumerate(bench.states):
        record = LevelRecord(
            id=f"fetchreach_{replay_idx}",
            domain="official_fetchreach_goal_recovery",
            task_id=args.env_id,
            clean_success_hat=bench.success_prob(replay_idx, 0.0, rng, args.record_trials),
            feasibility_hat=1.0,
            perturbation_family="adverse_goal_offset",
            sigma_grid=args.initial_probes,
        )
        estimator = IsotonicCurveEstimator(args.max_radius, args.alpha)
        for sigma in args.initial_probes:
            for _ in range(args.record_trials):
                success, _loss = bench.rollout(replay_idx, sigma, int(rng.integers(1_000_000_000)), train=False)
                record.add_observation(sigma, success)
                estimator.update_bernoulli(sigma, success)
        write_estimate(record, estimator.fit())
        records.append(record)
    return records


def refresh_records(
    bench: FetchReachProbe,
    records: list[LevelRecord],
    rng: np.random.Generator,
    args: argparse.Namespace,
    step: int,
) -> None:
    count = min(args.refresh_per_eval, len(records))
    scores = np.array([1.0 + record.uncertainty_hat for record in records], dtype=float)
    probs = scores / np.sum(scores)
    for replay_idx in rng.choice(len(records), size=count, replace=False, p=probs):
        record = records[int(replay_idx)]
        estimator = IsotonicCurveEstimator(args.max_radius, args.alpha)
        for sigma, trials in record.trials.items():
            estimator.update(float(sigma), record.successes.get(sigma, 0), trials)
        sigma = estimator.next_probe(rng, jitter=args.refresh_jitter)
        sigma = float(np.clip(sigma, 0.0, args.max_radius))
        for _ in range(args.refresh_trials):
            success, _loss = bench.rollout(int(replay_idx), sigma, int(rng.integers(1_000_000_000)), train=False)
            record.add_observation(sigma, success)
            estimator.update_bernoulli(sigma, success)
        write_estimate(record, estimator.fit())
        record.clean_success_hat = bench.success_prob(int(replay_idx), 0.0, rng, args.record_trials)
        record.last_evaluated_step = step


def evaluate(bench: FetchReachProbe, rng: np.random.Generator, args: argparse.Namespace) -> dict[str, float]:
    clean: list[float] = []
    raucs: list[float] = []
    radii: list[float] = []
    for replay_idx in range(len(bench.states)):
        curve = np.array(
            [bench.success_prob(replay_idx, sigma, rng, args.eval_trials) for sigma in args.eval_radii],
            dtype=float,
        )
        clean.append(float(curve[0]))
        raucs.append(recovery_auc(np.array(args.eval_radii, dtype=float), curve, sigma_max=args.max_radius))
        radii.append(critical_radius(np.array(args.eval_radii, dtype=float), curve, alpha=args.alpha))
    return {
        "clean": float(np.mean(clean)),
        "rauc": float(np.mean(raucs)),
        "median_r80": float(np.median(radii)),
    }


def sample_pair(
    method: str,
    bench: FetchReachProbe,
    records: list[LevelRecord],
    scorer: BGRPriorityScorer,
    rng: np.random.Generator,
    args: argparse.Namespace,
    step: int,
) -> tuple[int, float]:
    if method == "uniform":
        return int(rng.integers(len(records))), float(rng.uniform(0.0, args.max_radius))
    if method == "fixed":
        return int(rng.integers(len(records))), args.fixed_radius
    if method == "failure_only":
        candidates = rng.choice(len(records), size=min(args.baseline_candidates, len(records)), replace=False)
        sigmas = rng.uniform(0.0, args.max_radius, size=len(candidates))
        scores = [1.0 - bench.success_prob(int(idx), float(sigma), rng, args.quick_trials) for idx, sigma in zip(candidates, sigmas, strict=True)]
        best = int(np.argmax(scores))
        return int(candidates[best]), float(sigmas[best])
    if method == "td_loss":
        candidates = rng.choice(len(records), size=min(args.baseline_candidates, len(records)), replace=False)
        sigmas = rng.uniform(0.0, args.max_radius, size=len(candidates))
        scores = [bench.loss_proxy(int(idx), float(sigma), rng) for idx, sigma in zip(candidates, sigmas, strict=True)]
        best = int(np.argmax(scores))
        return int(candidates[best]), float(sigmas[best])
    if method.startswith("bgr"):
        priorities = np.array([scorer.score(record, step) for record in records], dtype=float)
        probs = mixed_priority_probs(priorities, temperature=args.priority_temperature, uniform_mix=args.uniform_mix)
        replay_idx = int(rng.choice(len(records), p=probs))
        if method == "bgr_uniform_radius":
            sigma = float(rng.uniform(0.0, args.max_radius))
        elif method == "bgr_coverage" and rng.random() < args.radius_uniform_mix:
            sigma = float(rng.uniform(0.0, args.max_radius))
        else:
            sigma = sample_boundary_radius(rng, records[replay_idx].r_alpha_hat, args.max_radius, radius_noise=args.radius_noise)
        return replay_idx, float(np.clip(sigma, 0.0, args.max_radius))
    raise ValueError(f"unknown method: {method}")


def run_method(args: argparse.Namespace, method: str, seed: int) -> FetchProbeResult:
    rng = np.random.default_rng(args.seed_offset + 50_000 + seed)
    bench = FetchReachProbe(args, seed)
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
                metrics = evaluate(bench, rng, args)
                metrics["step"] = float(step)
                history.append(metrics)
                if method.startswith("bgr"):
                    refresh_records(bench, records, rng, args, step)
            if step == args.iterations:
                break
            for _ in range(args.train_batch_size):
                replay_idx, sigma = sample_pair(method, bench, records, scorer, rng, args, step)
                success, _loss = bench.train_step(replay_idx, sigma, rng)
                if method.startswith("bgr"):
                    records[replay_idx].add_observation(sigma, success)
                    records[replay_idx].replay_count += 1
        xs = np.array([row["step"] for row in history], dtype=float)
        ys = np.array([row["rauc"] for row in history], dtype=float)
        final = history[-1]
        return FetchProbeResult(
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


def write_outputs(out_dir: Path, rows: list[dict[str, float | int | str]]) -> None:
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["method", "seed", "final_clean", "final_rauc", "final_median_r80", "rauc_aulc", "best_rauc"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "package_versions.json").open("w", encoding="utf-8") as handle:
        json.dump(package_versions(), handle, indent=2, sort_keys=True)
        handle.write("\n")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--env-id", default="FetchReach-v4")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--replay-states", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--init-gain", type=float, default=2.0)
    parser.add_argument("--init-noise", type=float, default=0.04)
    parser.add_argument("--teacher-gain", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=0.20)
    parser.add_argument("--max-radius", type=float, default=0.15)
    parser.add_argument("--fixed-radius", type=float, default=0.06)
    parser.add_argument("--eval-radii", default="0.00,0.03,0.06,0.09,0.12,0.15")
    parser.add_argument("--initial-probes", default="0.00,0.06,0.12,0.15")
    parser.add_argument("--eval-trials", type=int, default=4)
    parser.add_argument("--record-trials", type=int, default=2)
    parser.add_argument("--quick-trials", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.80)
    parser.add_argument("--target-radius", type=float, default=0.045)
    parser.add_argument("--radius-bandwidth", type=float, default=0.040)
    parser.add_argument("--radius-noise", type=float, default=0.18)
    parser.add_argument("--radius-uniform-mix", type=float, default=0.30)
    parser.add_argument("--uniform-mix", type=float, default=0.10)
    parser.add_argument("--priority-temperature", type=float, default=1.0)
    parser.add_argument("--baseline-candidates", type=int, default=4)
    parser.add_argument("--refresh-per-eval", type=int, default=2)
    parser.add_argument("--refresh-trials", type=int, default=2)
    parser.add_argument("--refresh-jitter", type=float, default=0.02)
    parser.add_argument("--seed-offset", type=int, default=113_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.eval_radii = parse_floats(args.eval_radii)
    args.initial_probes = parse_floats(args.initial_probes)
    args.seeds = parse_ints(args.seeds)
    methods = parse_strings(args.methods)
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for method in methods:
        for seed in args.seeds:
            result = run_method(args, method, seed)
            row = asdict(result)
            row.pop("history")
            rows.append(row)
            print(f"[done] method={method} seed={seed} rauc={result.final_rauc:.4f} clean={result.final_clean:.4f}")
            with (args.out / "results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
            write_outputs(args.out, rows)
    print(summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
