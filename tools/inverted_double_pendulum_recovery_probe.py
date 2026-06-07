#!/usr/bin/env python3
"""Run a fixed all-method Gymnasium MuJoCo InvertedDoublePendulum comparison.

This is an independent-package screen built on Gymnasium's
InvertedDoublePendulum-v5 dynamics. The learner is a small linear controller
trained by imitation of a stronger finite-difference LQR teacher from replayed
perturbed simulator states. The script compares replay sampling methods only
after the pre-method calibration has shown clean, non-flat recovery curves.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from bgr.curve_estimators import IsotonicCurveEstimator
from bgr.metrics import critical_radius, recovery_auc
from bgr.priorities import BGRPriorityScorer
from bgr.records import LevelRecord
from bgr.samplers import mixed_priority_probs, sample_boundary_radius
from tools.inverted_double_pendulum_recovery_calibration import (
    lqr_gain,
    parse_radii,
    perturb_angle,
    reset_perturbed,
    state_vector,
)

MAX_ERROR = 20.0
MAX_UPDATE_NORM = 0.10
MAX_WEIGHT_NORM = 80.0
MAX_BIAS_NORM = 20.0


@dataclass(frozen=True, slots=True)
class InvertedDoublePendulumProbeResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


class LinearLQRPolicy:
    def __init__(
        self,
        *,
        teacher_gain: np.ndarray,
        init_scale: float,
        action_low: float,
        action_high: float,
        learning_rate: float,
    ) -> None:
        self.weights = -float(init_scale) * np.asarray(teacher_gain, dtype=float).reshape(6)
        self.bias = 0.0
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.learning_rate = float(learning_rate)

    def features(self, env) -> np.ndarray:
        return state_vector(env)

    def predict_unclipped(self, features: np.ndarray) -> float:
        return float(self.weights @ features + self.bias)

    def action_from_features(self, features: np.ndarray) -> np.ndarray:
        force = np.clip(self.predict_unclipped(features), self.action_low, self.action_high)
        return np.array([force], dtype=np.float32)

    def action(self, env) -> np.ndarray:
        return self.action_from_features(self.features(env))

    def update(self, features: np.ndarray, teacher: float) -> float:
        if not np.all(np.isfinite(features)) or not np.isfinite(teacher):
            return 0.0
        prediction = self.predict_unclipped(features)
        if not np.isfinite(prediction):
            prediction = 0.0
        error = float(np.clip(teacher - prediction, -MAX_ERROR, MAX_ERROR))
        weight_update = self.learning_rate * error * features
        update_norm = float(np.linalg.norm(weight_update))
        if update_norm > MAX_UPDATE_NORM:
            weight_update *= MAX_UPDATE_NORM / update_norm
        self.weights += weight_update
        self.bias += self.learning_rate * error
        weight_norm = float(np.linalg.norm(self.weights))
        if weight_norm > MAX_WEIGHT_NORM:
            self.weights *= MAX_WEIGHT_NORM / weight_norm
        self.bias = float(np.clip(self.bias, -MAX_BIAS_NORM, MAX_BIAS_NORM))
        return float(error * error)


class InvertedDoublePendulumRecoveryProbe:
    def __init__(self, args: argparse.Namespace, seed: int) -> None:
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise SystemExit(
                "InvertedDoublePendulum comparison requires Gymnasium with MuJoCo in an isolated environment, "
                "for example /tmp/bgr_pointmaze_venv."
            ) from exc

        self.args = args
        self.seed = int(seed)
        self.env = gym.make(args.env_id)
        self.teacher_gain = lqr_gain(self.env, args)
        self.policy = LinearLQRPolicy(
            teacher_gain=self.teacher_gain,
            init_scale=args.init_gain_scale,
            action_low=float(self.env.action_space.low[0]),
            action_high=float(self.env.action_space.high[0]),
            learning_rate=args.learning_rate,
        )
        self.replay_seeds = [args.replay_seed_offset + 1000 * seed + idx for idx in range(args.replay_states)]

    def close(self) -> None:
        self.env.close()

    def trial_angle(self, replay_idx: int, trial_seed: int) -> float:
        return perturb_angle(replay_idx, trial_seed % 997)

    def teacher_force(self, features: np.ndarray) -> float:
        force = float(np.squeeze(-self.teacher_gain @ features.reshape(-1, 1)))
        return float(np.clip(force, self.env.action_space.low[0], self.env.action_space.high[0]))

    def rollout(self, replay_idx: int, sigma: float, trial_seed: int, *, update: bool) -> tuple[bool, float]:
        reset_perturbed(
            self.env,
            seed=self.replay_seeds[replay_idx],
            sigma=float(sigma),
            angle=self.trial_angle(replay_idx, trial_seed),
        )
        losses: list[float] = []
        terminated = False
        step = 0
        for step in range(1, self.args.horizon + 1):
            features = self.policy.features(self.env)
            teacher = self.teacher_force(features)
            if update:
                losses.append(self.policy.update(features, teacher))
            else:
                error = teacher - self.policy.predict_unclipped(features)
                losses.append(float(error * error))
            _obs, _reward, terminated, truncated, _info = self.env.step(self.policy.action_from_features(features))
            if terminated or truncated:
                break
        return bool((not terminated) and step >= self.args.horizon), float(np.mean(losses)) if losses else 0.0

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> tuple[bool, float]:
        return self.rollout(replay_idx, sigma, int(rng.integers(1_000_000_000)), update=True)

    def success_prob(self, replay_idx: int, sigma: float, rng: np.random.Generator, trials: int) -> float:
        successes = [
            self.rollout(replay_idx, sigma, int(rng.integers(1_000_000_000)), update=False)[0]
            for _ in range(trials)
        ]
        return float(np.mean(successes))

    def clean_success(self, replay_idx: int, rng: np.random.Generator) -> float:
        return self.success_prob(replay_idx, 0.0, rng, self.args.quick_trials)

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        success, loss = self.rollout(replay_idx, sigma, int(rng.integers(1_000_000_000)), update=False)
        return float(loss + (2.0 if not success else 0.0))


def parse_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def parse_strings(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one method")
    return parsed


def package_versions() -> dict[str, str]:
    try:
        import gymnasium
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "InvertedDoublePendulum comparison requires Gymnasium with MuJoCo in an isolated environment, "
            "for example /tmp/bgr_pointmaze_venv."
        ) from exc
    return {
        "gymnasium": gymnasium.__version__,
        "gymnasium-package": version("gymnasium"),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }


def write_estimate(record: LevelRecord, estimate: Any) -> None:
    record.r_alpha_hat = estimate.r_alpha
    record.sharpness_hat = estimate.sharpness
    record.uncertainty_hat = estimate.r_uncertainty
    record.recovery_curve_hat = estimate.recovery.tolist()


def init_records(
    bench: InvertedDoublePendulumRecoveryProbe,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, replay_seed in enumerate(bench.replay_seeds):
        record = LevelRecord(
            id=f"inverted_double_pendulum_seed_{replay_seed}",
            domain="official_inverted_double_pendulum_recovery",
            task_id=bench.args.env_id,
            clean_success_hat=bench.clean_success(replay_idx, rng),
            feasibility_hat=1.0,
            perturbation_family="inverted_double_pendulum_two_pole_angle_restart",
            sigma_grid=[float(sigma) for sigma in args.initial_probes],
        )
        estimator = IsotonicCurveEstimator(args.max_radius, args.alpha)
        for sigma in args.initial_probes:
            for _ in range(args.record_trials):
                success = bench.rollout(replay_idx, float(sigma), int(rng.integers(1_000_000_000)), update=False)[0]
                record.add_observation(float(sigma), success)
                estimator.update_bernoulli(float(sigma), success)
        write_estimate(record, estimator.fit())
        records.append(record)
    return records


def refresh_records(
    bench: InvertedDoublePendulumRecoveryProbe,
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
        estimator = IsotonicCurveEstimator(args.max_radius, args.alpha)
        for sigma, trials in record.trials.items():
            estimator.update(sigma, record.successes.get(sigma, 0), trials)
        sigma = estimator.next_probe(rng, jitter=args.refresh_jitter)
        for _ in range(args.refresh_trials):
            success = bench.rollout(int(replay_idx), sigma, int(rng.integers(1_000_000_000)), update=False)[0]
            record.add_observation(sigma, success)
            estimator.update_bernoulli(sigma, success)
        write_estimate(record, estimator.fit())
        record.clean_success_hat = bench.clean_success(int(replay_idx), rng)
        record.feasibility_hat = 1.0
        record.last_evaluated_step = step


def evaluate(bench: InvertedDoublePendulumRecoveryProbe, rng: np.random.Generator, args: argparse.Namespace) -> dict[str, float]:
    clean: list[float] = []
    raucs: list[float] = []
    radii: list[float] = []
    for replay_idx in range(len(bench.replay_seeds)):
        curve = np.array(
            [bench.success_prob(replay_idx, float(sigma), rng, args.eval_trials) for sigma in args.eval_radii],
            dtype=float,
        )
        clean.append(float(curve[0]))
        raucs.append(recovery_auc(args.eval_radii, curve, sigma_max=args.max_radius))
        radii.append(critical_radius(args.eval_radii, curve, alpha=args.alpha))
    return {
        "clean": float(np.mean(clean)),
        "rauc": float(np.mean(raucs)),
        "median_r80": float(np.median(radii)),
    }


def sample_pair(
    method: str,
    bench: InvertedDoublePendulumRecoveryProbe,
    records: list[LevelRecord],
    scorer: BGRPriorityScorer,
    rng: np.random.Generator,
    args: argparse.Namespace,
    step: int,
) -> tuple[int, float]:
    if method == "uniform":
        return int(rng.integers(len(records))), float(rng.uniform(0.0, args.max_radius))
    if method == "fixed":
        return int(rng.integers(len(records))), float(args.fixed_radius)
    if method == "failure_only":
        candidates = rng.choice(len(records), size=min(args.baseline_candidates, len(records)), replace=False)
        sigmas = rng.uniform(0.0, args.max_radius, size=len(candidates))
        scores = [
            1.0 - bench.success_prob(int(idx), float(sigma), rng, args.quick_trials)
            for idx, sigma in zip(candidates, sigmas, strict=True)
        ]
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
            sigma = sample_boundary_radius(
                rng,
                records[replay_idx].r_alpha_hat,
                args.max_radius,
                radius_noise=args.radius_noise,
            )
        return replay_idx, float(np.clip(sigma, 0.0, args.max_radius))
    raise ValueError(f"unknown method: {method}")


def run_method(args: argparse.Namespace, method: str, seed: int) -> InvertedDoublePendulumProbeResult:
    rng = np.random.default_rng(args.seed_offset + 70_000 + seed)
    bench = InvertedDoublePendulumRecoveryProbe(args, seed)
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
        return InvertedDoublePendulumProbeResult(
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


def aggregate_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    by_method: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    output: list[dict[str, str]] = []
    for method, method_rows in sorted(by_method.items()):
        output.append(
            {
                "method": method,
                "seeds": str(len(method_rows)),
                "final_clean_mean": f"{np.mean([float(row['final_clean']) for row in method_rows]):.6f}",
                "final_rauc_mean": f"{np.mean([float(row['final_rauc']) for row in method_rows]):.6f}",
                "final_median_r80_mean": f"{np.mean([float(row['final_median_r80']) for row in method_rows]):.6f}",
                "rauc_aulc_mean": f"{np.mean([float(row['rauc_aulc']) for row in method_rows]):.6f}",
                "best_rauc_mean": f"{np.mean([float(row['best_rauc']) for row in method_rows]):.6f}",
            }
        )
    return output


def write_outputs(out_dir: Path, results: list[InvertedDoublePendulumProbeResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "method": result.method,
            "seed": result.seed,
            "final_clean": result.final_clean,
            "final_rauc": result.final_rauc,
            "final_median_r80": result.final_median_r80,
            "rauc_aulc": result.rauc_aulc,
            "best_rauc": result.best_rauc,
        }
        for result in results
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["method", "seed", "final_clean", "final_rauc", "final_median_r80", "rauc_aulc", "best_rauc"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "method",
            "seeds",
            "final_clean_mean",
            "final_rauc_mean",
            "final_median_r80_mean",
            "rauc_aulc_mean",
            "best_rauc_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggregate_rows(rows))
    with (out_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["method", "seed", "step", "clean", "rauc", "median_r80"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in results:
            for row in result.history:
                writer.writerow({"method": result.method, "seed": result.seed, **row})
    with (out_dir / "package_versions.json").open("w", encoding="utf-8") as handle:
        json.dump(package_versions(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def summary_text(results: list[InvertedDoublePendulumProbeResult]) -> str:
    rows = [
        {
            "method": result.method,
            "final_clean": result.final_clean,
            "final_rauc": result.final_rauc,
            "final_median_r80": result.final_median_r80,
            "rauc_aulc": result.rauc_aulc,
            "best_rauc": result.best_rauc,
        }
        for result in results
    ]
    lines = ["method,final_clean_mean,final_rauc_mean,final_median_r80_mean,rauc_aulc_mean,best_rauc_mean"]
    for row in aggregate_rows(rows):
        lines.append(
            ",".join(
                [
                    row["method"],
                    f"{float(row['final_clean_mean']):.4f}",
                    f"{float(row['final_rauc_mean']):.4f}",
                    f"{float(row['final_median_r80_mean']):.4f}",
                    f"{float(row['rauc_aulc_mean']):.4f}",
                    f"{float(row['best_rauc_mean']):.4f}",
                ]
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--env-id", default="InvertedDoublePendulum-v5")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--replay-states", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--init-gain-scale", type=float, default=0.70)
    parser.add_argument("--max-radius", type=float, default=0.90)
    parser.add_argument("--fixed-radius", type=float, default=0.30)
    parser.add_argument("--eval-radii", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--initial-probes", default="0,0.2,0.3,0.5,0.9")
    parser.add_argument("--eval-trials", type=int, default=2)
    parser.add_argument("--record-trials", type=int, default=1)
    parser.add_argument("--quick-trials", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.80)
    parser.add_argument("--target-radius", type=float, default=0.30)
    parser.add_argument("--radius-bandwidth", type=float, default=0.12)
    parser.add_argument("--radius-noise", type=float, default=0.05)
    parser.add_argument("--radius-uniform-mix", type=float, default=0.25)
    parser.add_argument("--uniform-mix", type=float, default=0.10)
    parser.add_argument("--priority-temperature", type=float, default=0.9)
    parser.add_argument("--baseline-candidates", type=int, default=8)
    parser.add_argument("--refresh-per-eval", type=int, default=6)
    parser.add_argument("--refresh-trials", type=int, default=1)
    parser.add_argument("--refresh-jitter", type=float, default=0.04)
    parser.add_argument("--replay-seed-offset", type=int, default=430_000)
    parser.add_argument("--seed-offset", type=int, default=431_000)
    parser.add_argument("--linearization-eps", type=float, default=1e-5)
    parser.add_argument("--q-cart", type=float, default=1.0)
    parser.add_argument("--q-pole1", type=float, default=40.0)
    parser.add_argument("--q-pole2", type=float, default=48.0)
    parser.add_argument("--q-cart-vel", type=float, default=0.2)
    parser.add_argument("--q-pole1-vel", type=float, default=1.5)
    parser.add_argument("--q-pole2-vel", type=float, default=1.5)
    parser.add_argument("--r-action", type=float, default=0.02)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.seeds = parse_ints(args.seeds)
    args.methods = parse_strings(args.methods)
    args.eval_radii = parse_radii(args.eval_radii)
    args.initial_probes = parse_radii(args.initial_probes)
    if float(args.eval_radii[-1]) != float(args.max_radius):
        raise ValueError("--eval-radii must end at --max-radius")
    if float(args.initial_probes[-1]) > float(args.max_radius):
        raise ValueError("--initial-probes must not exceed --max-radius")

    results: list[InvertedDoublePendulumProbeResult] = []
    for method in args.methods:
        for seed in args.seeds:
            results.append(run_method(args, method, seed))
    write_outputs(args.out, results)
    print(summary_text(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
