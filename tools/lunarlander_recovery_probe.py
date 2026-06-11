#!/usr/bin/env python3
"""Run a fixed Gymnasium LunarLander-v3 recovery replay screen.

This is an internal independent-benchmark screen opened only after the
pre-method LunarLander calibration cleared. It uses Gymnasium's package-owned
Box2D dynamics and heuristic controller. The learner is a fixed linear
imitation policy; methods differ only in replay state/radius selection.
"""

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
from tools.lunarlander_recovery_calibration import checkpoint
from tools.lunarlander_recovery_calibration import package_versions
from tools.lunarlander_recovery_calibration import parse_radii
from tools.lunarlander_recovery_calibration import perturb_direction
from tools.lunarlander_recovery_calibration import restore_perturbed


@dataclass(frozen=True, slots=True)
class LanderCheckpoint:
    seed: int
    lander_state: dict[str, float]
    leg_states: list[dict[str, float]]
    checkpoint_obs: list[float]


@dataclass(frozen=True, slots=True)
class LanderProbeResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


class LinearImitationPolicy:
    def __init__(self, *, rng: np.random.Generator, learning_rate: float, init_noise: float) -> None:
        self.rng = rng
        self.learning_rate = float(learning_rate)
        self.weights = rng.normal(0.0, float(init_noise), size=(4, 9))

    @staticmethod
    def features(obs: np.ndarray) -> np.ndarray:
        clipped = np.array(obs, dtype=float)
        clipped[:6] = np.clip(clipped[:6], -2.5, 2.5)
        return np.concatenate([clipped, np.ones(1, dtype=float)])

    def action(self, obs: np.ndarray) -> int:
        logits = self.weights @ self.features(obs)
        return int(np.argmax(logits))

    def loss(self, obs: np.ndarray, target: int) -> float:
        logits = self.weights @ self.features(obs)
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        probs /= float(np.sum(probs))
        return float(-np.log(max(probs[int(target)], 1e-12)))

    def update(self, obs: np.ndarray, target: int) -> None:
        x = self.features(obs)
        logits = self.weights @ x
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        probs /= float(np.sum(probs))
        grad = probs[:, None] * x[None, :]
        grad[int(target), :] -= x
        self.weights -= self.learning_rate * grad


class ContinuousLinearImitationPolicy:
    def __init__(self, *, rng: np.random.Generator, learning_rate: float, init_noise: float) -> None:
        self.rng = rng
        self.learning_rate = float(learning_rate)
        self.weights = rng.normal(0.0, float(init_noise), size=(2, 9))

    @staticmethod
    def features(obs: np.ndarray) -> np.ndarray:
        clipped = np.array(obs, dtype=float)
        clipped[:6] = np.clip(clipped[:6], -2.5, 2.5)
        return np.concatenate([clipped, np.ones(1, dtype=float)])

    def action(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(self.weights @ self.features(obs), -1.0, 1.0)

    def loss(self, obs: np.ndarray, target: np.ndarray) -> float:
        error = self.action(obs) - np.asarray(target, dtype=float)
        return float(np.mean(error * error))

    def update(self, obs: np.ndarray, target: np.ndarray) -> None:
        x = self.features(obs)
        error = self.action(obs) - np.asarray(target, dtype=float)
        self.weights -= self.learning_rate * np.clip(np.outer(error, x), -5.0, 5.0)


class LunarLanderRecoveryProbe:
    def __init__(
        self,
        *,
        seed: int,
        args: argparse.Namespace,
        replay_checkpoints: list[LanderCheckpoint],
        eval_checkpoints: list[LanderCheckpoint],
    ) -> None:
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise SystemExit(
                "LunarLander recovery probe requires Gymnasium with Box2D in an isolated environment, "
                "for example /tmp/bgr_lunar_venv."
            ) from exc

        self.rng = np.random.default_rng(seed + 211_000)
        self.env = gym.make(args.env_id, continuous=bool(args.continuous), enable_wind=False)
        self.args = args
        self.replay_checkpoints = replay_checkpoints
        self.eval_checkpoints = eval_checkpoints
        self.radii = parse_radii(args.radii)
        policy_cls = ContinuousLinearImitationPolicy if args.continuous else LinearImitationPolicy
        self.policy = policy_cls(rng=self.rng, learning_rate=args.learning_rate, init_noise=args.policy_init_noise)
        self._pretrain(seed)

    def close(self) -> None:
        self.env.close()

    def _teacher_action(self, obs: np.ndarray) -> int | np.ndarray:
        from gymnasium.envs.box2d.lunar_lander import heuristic

        action = heuristic(self.env, obs)
        if self.args.continuous:
            return np.asarray(action, dtype=float)
        return int(action)

    def _pretrain(self, seed: int) -> None:
        for step in range(self.args.policy_init_steps):
            checkpoint_idx = int(self.rng.integers(len(self.replay_checkpoints)))
            sigma = float(self.rng.uniform(0.0, 1.0))
            obs = self.start_obs(
                self.replay_checkpoints[checkpoint_idx],
                sigma=sigma,
                direction=perturb_direction(seed + self.args.init_seed_offset + checkpoint_idx, step),
            )
            self.policy.update(obs, self._teacher_action(obs))

    def start_obs(self, item: LanderCheckpoint, *, sigma: float, direction: np.ndarray) -> np.ndarray:
        self.env.reset(seed=item.seed)
        return restore_perturbed(
            self.env,
            lander_state=item.lander_state,
            leg_states=item.leg_states,
            sigma=sigma,
            direction=direction,
        )

    def rollout_policy(self, obs: np.ndarray, *, train: bool) -> bool:
        obs = np.asarray(obs, dtype=float)
        for _ in range(self.args.horizon):
            if train:
                self.policy.update(obs, self._teacher_action(obs))
            action = self.policy.action(obs)
            obs, reward, terminated, truncated, _info = self.env.step(action)
            obs = np.asarray(obs, dtype=float)
            if terminated or truncated:
                return bool(reward >= 100.0 and not self.env.unwrapped.game_over)
        return False

    def success_prob(self, replay_idx: int, sigma: float) -> float:
        item = self.replay_checkpoints[replay_idx]
        outcomes = []
        for trial in range(self.args.record_trials):
            obs = self.start_obs(
                item,
                sigma=sigma,
                direction=perturb_direction(item.seed, trial),
            )
            outcomes.append(self.rollout_policy(obs, train=False))
        return float(np.mean(outcomes))

    def rollout(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> bool:
        item = self.replay_checkpoints[replay_idx]
        obs = self.start_obs(
            item,
            sigma=sigma,
            direction=normalize_direction(rng.normal(0.0, 1.0, size=6)),
        )
        return self.rollout_policy(obs, train=False)

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> None:
        item = self.replay_checkpoints[replay_idx]
        obs = self.start_obs(
            item,
            sigma=sigma,
            direction=normalize_direction(rng.normal(0.0, 1.0, size=6)),
        )
        self.rollout_policy(obs, train=True)

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        item = self.replay_checkpoints[replay_idx]
        obs = self.start_obs(
            item,
            sigma=sigma,
            direction=normalize_direction(rng.normal(0.0, 1.0, size=6)),
        )
        return self.policy.loss(obs, self._teacher_action(obs))


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.zeros(6, dtype=float)
        direction[0] = 1.0
        return direction
    return direction / norm


def build_checkpoints(args: argparse.Namespace, *, count: int, seed_offset: int) -> list[LanderCheckpoint]:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit(
            "LunarLander recovery probe requires Gymnasium with Box2D in an isolated environment, "
            "for example /tmp/bgr_lunar_venv."
        ) from exc

    env = gym.make(args.env_id, continuous=bool(args.continuous), enable_wind=False)
    checkpoints: list[LanderCheckpoint] = []
    try:
        for idx in range(count):
            seed = seed_offset + idx
            lander_state, leg_states, obs = checkpoint(env, seed=seed, burn_in=args.burn_in)
            checkpoints.append(
                LanderCheckpoint(
                    seed=seed,
                    lander_state=lander_state,
                    leg_states=leg_states,
                    checkpoint_obs=[float(value) for value in obs],
                )
            )
    finally:
        env.close()
    return checkpoints


def init_records(bench: LunarLanderRecoveryProbe, rng: np.random.Generator, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, item in enumerate(bench.replay_checkpoints):
        record = LevelRecord(
            id=f"lunarlander_{replay_idx}",
            domain="lunarlander_v3_recovery",
            task_id=args.env_id,
            clean_success_hat=bench.success_prob(replay_idx, 0.0),
            feasibility_hat=1.0,
            perturbation_family="box2d_descent_restart",
            sigma_grid=list(bench.radii),
        )
        estimator = IsotonicCurveEstimator(float(bench.radii[-1]), args.alpha)
        for sigma in args.initial_probes:
            for _ in range(args.min_trials):
                success = bench.rollout(replay_idx, sigma, rng)
                record.add_observation(sigma, success)
                estimator.update_bernoulli(sigma, success)
        write_estimate(record, estimator.fit())
        records.append(record)
    return records


def refresh_records(
    bench: LunarLanderRecoveryProbe,
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
        estimator = IsotonicCurveEstimator(float(bench.radii[-1]), args.alpha)
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


def evaluate(bench: LunarLanderRecoveryProbe, args: argparse.Namespace) -> dict[str, float]:
    radii = bench.radii
    curve: list[float] = []
    per_checkpoint_r80: list[float] = []
    for sigma in radii:
        outcomes = []
        for item in bench.eval_checkpoints:
            for trial in range(args.eval_trials):
                obs = bench.start_obs(
                    item,
                    sigma=float(sigma),
                    direction=perturb_direction(item.seed, trial),
                )
                outcomes.append(bench.rollout_policy(obs, train=False))
        curve.append(float(np.mean(outcomes)))
    for item in bench.eval_checkpoints:
        item_curve = []
        for sigma in radii:
            outcomes = []
            for trial in range(args.eval_trials):
                obs = bench.start_obs(
                    item,
                    sigma=float(sigma),
                    direction=perturb_direction(item.seed, trial),
                )
                outcomes.append(bench.rollout_policy(obs, train=False))
            item_curve.append(float(np.mean(outcomes)))
        per_checkpoint_r80.append(critical_radius(radii, np.array(item_curve, dtype=float), alpha=args.alpha))
    curve_array = np.array(curve, dtype=float)
    return {
        "clean": float(curve_array[0]),
        "rauc": float(recovery_auc(radii, curve_array, sigma_max=float(radii[-1]))),
        "median_r80": float(np.median(per_checkpoint_r80)),
    }


def sample_training_pair(
    method: str,
    bench: LunarLanderRecoveryProbe,
    records: list[LevelRecord],
    scorer: BGRPriorityScorer,
    rng: np.random.Generator,
    args: argparse.Namespace,
    step: int,
) -> tuple[int, float]:
    sigma_max = float(bench.radii[-1])
    if method == "uniform":
        return int(rng.integers(len(records))), float(rng.uniform(0.0, sigma_max))
    if method == "fixed":
        return int(rng.integers(len(records))), args.fixed_radius
    if method == "failure_only":
        candidates = rng.choice(len(records), size=min(args.baseline_candidates, len(records)), replace=False)
        sigmas = rng.uniform(0.0, sigma_max, size=len(candidates))
        scores = [1.0 - bench.success_prob(int(candidate), float(sigma)) for candidate, sigma in zip(candidates, sigmas, strict=True)]
        idx = int(np.argmax(scores))
        return int(candidates[idx]), float(sigmas[idx])
    if method == "td_loss":
        candidates = rng.choice(len(records), size=min(args.baseline_candidates, len(records)), replace=False)
        sigmas = rng.uniform(0.0, sigma_max, size=len(candidates))
        scores = [bench.loss_proxy(int(candidate), float(sigma), rng) for candidate, sigma in zip(candidates, sigmas, strict=True)]
        idx = int(np.argmax(scores))
        return int(candidates[idx]), float(sigmas[idx])
    if method.startswith("bgr"):
        priorities = np.array([scorer.score(record, step) for record in records], dtype=float)
        probs = mixed_priority_probs(priorities, temperature=args.priority_temperature, uniform_mix=args.uniform_mix)
        replay_idx = int(rng.choice(len(records), p=probs))
        if method == "bgr_uniform_radius":
            sigma = float(rng.uniform(0.0, sigma_max))
        elif method == "bgr_coverage" and rng.random() < args.radius_uniform_mix:
            sigma = float(rng.uniform(0.0, sigma_max))
        else:
            sigma = sample_boundary_radius(rng, records[replay_idx].r_alpha_hat, sigma_max, radius_noise=args.radius_noise)
        return replay_idx, sigma
    raise ValueError(f"unknown method: {method}")


def run_method(
    args: argparse.Namespace,
    method: str,
    seed: int,
    replay_checkpoints: list[LanderCheckpoint],
    eval_checkpoints: list[LanderCheckpoint],
) -> LanderProbeResult:
    rng = np.random.default_rng(seed + 213_000)
    bench = LunarLanderRecoveryProbe(
        seed=seed,
        args=args,
        replay_checkpoints=replay_checkpoints,
        eval_checkpoints=eval_checkpoints,
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
    finally:
        bench.close()

    xs = np.array([row["step"] for row in history], dtype=float)
    ys = np.array([row["rauc"] for row in history], dtype=float)
    final = history[-1]
    return LanderProbeResult(
        method=method,
        seed=seed,
        final_clean=final["clean"],
        final_rauc=final["rauc"],
        final_median_r80=final["median_r80"],
        rauc_aulc=float(np.trapezoid(ys, xs) / (xs[-1] - xs[0])) if xs[-1] > xs[0] else final["rauc"],
        best_rauc=max(row["rauc"] for row in history),
        history=history,
    )


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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
    parser.add_argument("--out", default="results/lunarlander_recovery_probe_4seed_v1")
    parser.add_argument("--env-id", default="LunarLander-v3")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--radii", default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--replay-states", type=int, default=24)
    parser.add_argument("--eval-states", type=int, default=12)
    parser.add_argument("--eval-trials", type=int, default=3)
    parser.add_argument("--record-trials", type=int, default=2)
    parser.add_argument("--burn-in", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=260)
    parser.add_argument("--replay-seed-offset", type=int, default=200)
    parser.add_argument("--eval-seed-offset", type=int, default=0)
    parser.add_argument("--init-seed-offset", type=int, default=10_000)
    parser.add_argument("--policy-init-steps", type=int, default=256)
    parser.add_argument("--policy-init-noise", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--target-radius", type=float, default=0.55)
    parser.add_argument("--radius-bandwidth", type=float, default=0.35)
    parser.add_argument("--fixed-radius", type=float, default=0.6)
    parser.add_argument("--baseline-candidates", type=int, default=8)
    parser.add_argument("--initial-probes", type=float, nargs="+", default=[0.0, 0.4, 0.8, 1.0])
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

    replay_checkpoints = build_checkpoints(args, count=args.replay_states, seed_offset=args.replay_seed_offset)
    eval_checkpoints = build_checkpoints(args, count=args.eval_states, seed_offset=args.eval_seed_offset)

    rows: list[dict[str, float | int | str]] = []
    history_rows: list[dict[str, float | int | str]] = []
    results: list[dict] = []
    for method in methods:
        for seed in seeds:
            start = time.perf_counter()
            result = run_method(args, method, seed, replay_checkpoints, eval_checkpoints)
            elapsed = time.perf_counter() - start
            print(
                f"[done] method={method} seed={seed} rauc={result.final_rauc:.4f} "
                f"clean={result.final_clean:.4f} r80={result.final_median_r80:.4f} elapsed={elapsed:.2f}s",
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
            for item in result.history:
                history_rows.append({"method": method, "seed": seed, **item})

    (out_dir / "results.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2), encoding="utf-8")
    (out_dir / "package_versions.json").write_text(
        json.dumps(package_versions(args.env_id, continuous=bool(args.continuous)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history_rows)
    print(summary(rows))


if __name__ == "__main__":
    main()
