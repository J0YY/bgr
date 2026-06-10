#!/usr/bin/env python3
"""Run a FetchPush-v4 object-state recovery replay screen.

This screen uses Gymnasium-Robotics' package-owned FetchPush-v4 dynamics and a
seed-fixed object-state reset perturbation. Replay methods choose which
object-state/radius pairs to train on; the learned policy is a small linear
imitator of the fixed scripted push teacher. This is a method comparison, not a
calibration.
"""

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
from fetch_object_goal_recovery_calibration import (
    adverse_direction,
    clipped_object,
    controller_action,
    package_versions,
    set_object_position,
)


@dataclass(frozen=True, slots=True)
class FetchPushReplayState:
    base_seed: int
    base_goal: list[float]
    base_object: list[float]
    center: list[float]
    obj_range: float
    direction: list[float]


@dataclass(frozen=True, slots=True)
class FetchPushProbeResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


class LinearPushPolicy:
    def __init__(self, rng: np.random.Generator, *, feature_dim: int, init_noise: float, learning_rate: float) -> None:
        self.weights = rng.normal(0.0, float(init_noise), size=(4, feature_dim))
        self.learning_rate = float(learning_rate)

    def action(self, features: np.ndarray) -> np.ndarray:
        return np.clip(self.weights @ features, -1.0, 1.0)

    def update(self, features: np.ndarray, target: np.ndarray) -> float:
        pred = self.weights @ features
        error = pred - target
        self.weights -= self.learning_rate * np.outer(error, features)
        return float(np.mean(error * error))


class MLPPushPolicy:
    def __init__(
        self,
        rng: np.random.Generator,
        *,
        feature_dim: int,
        hidden_dim: int,
        init_noise: float,
        learning_rate: float,
    ) -> None:
        scale = float(init_noise)
        self.w1 = rng.normal(0.0, scale, size=(int(hidden_dim), feature_dim))
        self.b1 = np.zeros(int(hidden_dim), dtype=float)
        self.w2 = rng.normal(0.0, scale, size=(4, int(hidden_dim)))
        self.b2 = np.zeros(4, dtype=float)
        self.learning_rate = float(learning_rate)

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(self.w1 @ features + self.b1)
        action = np.tanh(self.w2 @ hidden + self.b2)
        return hidden, action

    def action(self, features: np.ndarray) -> np.ndarray:
        _hidden, action = self._forward(features)
        return np.clip(action, -1.0, 1.0)

    def update(self, features: np.ndarray, target: np.ndarray) -> float:
        hidden, action = self._forward(features)
        error = action - target
        grad_out = (2.0 / len(error)) * error * (1.0 - action * action)
        grad_w2 = np.outer(grad_out, hidden)
        grad_b2 = grad_out
        grad_hidden = (self.w2.T @ grad_out) * (1.0 - hidden * hidden)
        grad_w1 = np.outer(grad_hidden, features)
        grad_b1 = grad_hidden
        clip = 5.0
        self.w2 -= self.learning_rate * np.clip(grad_w2, -clip, clip)
        self.b2 -= self.learning_rate * np.clip(grad_b2, -clip, clip)
        self.w1 -= self.learning_rate * np.clip(grad_w1, -clip, clip)
        self.b1 -= self.learning_rate * np.clip(grad_b1, -clip, clip)
        return float(np.mean(error * error))


class KNNPushPolicy:
    def __init__(self, *, max_memory: int, neighbors: int) -> None:
        self.max_memory = int(max_memory)
        self.neighbors = int(neighbors)
        self.features: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []

    def action(self, features: np.ndarray) -> np.ndarray:
        if not self.features:
            return np.zeros(4, dtype=float)
        matrix = np.vstack(self.features)
        dists = np.sum((matrix - features) ** 2, axis=1)
        count = min(self.neighbors, len(self.features))
        idx = np.argpartition(dists, count - 1)[:count]
        weights = 1.0 / (dists[idx] + 1e-6)
        actions = np.vstack([self.actions[int(i)] for i in idx])
        return np.clip(np.average(actions, axis=0, weights=weights), -1.0, 1.0)

    def update(self, features: np.ndarray, target: np.ndarray) -> float:
        pred = self.action(features)
        self.features.append(np.array(features, dtype=float))
        self.actions.append(np.array(target, dtype=float))
        overflow = len(self.features) - self.max_memory
        if overflow > 0:
            del self.features[:overflow]
            del self.actions[:overflow]
        error = pred - target
        return float(np.mean(error * error))


class TrajectoryLibraryPolicy:
    def __init__(self, *, max_trajectories: int) -> None:
        self.max_trajectories = int(max_trajectories)
        self.trajectories: list[dict[str, Any]] = []
        self.active_actions: list[np.ndarray] | None = None

    def start_episode(self, replay_idx: int, sigma: float) -> None:
        if not self.trajectories:
            self.active_actions = None
            return
        best = min(
            self.trajectories,
            key=lambda item: (
                0.0 if int(item["replay_idx"]) == int(replay_idx) else 1.0,
                abs(float(item["sigma"]) - float(sigma)),
            ),
        )
        self.active_actions = [np.array(action, dtype=float) for action in best["actions"]]

    def action(self, _features: np.ndarray, step: int = 0) -> np.ndarray:
        if self.active_actions is None or int(step) >= len(self.active_actions):
            return np.zeros(4, dtype=float)
        return np.clip(self.active_actions[int(step)], -1.0, 1.0)

    def add_trajectory(self, replay_idx: int, sigma: float, actions: list[np.ndarray]) -> None:
        if not actions:
            return
        self.trajectories.append(
            {
                "replay_idx": int(replay_idx),
                "sigma": float(sigma),
                "actions": [np.array(action, dtype=float) for action in actions],
            }
        )
        overflow = len(self.trajectories) - self.max_trajectories
        if overflow > 0:
            del self.trajectories[:overflow]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def object_direction(base_object: np.ndarray, base_goal: np.ndarray, rng: np.random.Generator, jitter: float) -> np.ndarray:
    direction = adverse_direction(base_object, base_goal, rng, jitter)
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return direction / norm


def obs_features(obs: dict[str, np.ndarray]) -> np.ndarray:
    raw = np.array(obs["observation"], dtype=float)
    gripper = raw[0:3]
    obj = raw[3:6]
    goal = np.array(obs["desired_goal"], dtype=float)
    achieved = np.array(obs["achieved_goal"], dtype=float)
    obj_to_goal = goal - obj
    gripper_to_obj = obj - gripper
    gripper_to_goal = goal - gripper
    return np.concatenate(
        [
            np.ones(1, dtype=float),
            gripper,
            obj,
            goal,
            achieved,
            obj_to_goal,
            gripper_to_obj,
            gripper_to_goal,
            np.array([np.linalg.norm(obj_to_goal), np.linalg.norm(gripper_to_obj)], dtype=float),
        ]
    )


class FetchPushObjectProbe:
    def __init__(self, args: argparse.Namespace, seed: int) -> None:
        import gymnasium as gym
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)
        self.env = gym.make(args.env_id, max_episode_steps=args.horizon)
        self.args = args
        self.rng = np.random.default_rng(args.seed_offset + seed)
        feature_dim = len(obs_features(self.env.reset(seed=args.seed_offset + seed)[0]))
        if args.policy == "linear":
            self.policy = LinearPushPolicy(
                self.rng,
                feature_dim=feature_dim,
                init_noise=args.init_noise,
                learning_rate=args.learning_rate,
            )
        elif args.policy == "mlp":
            self.policy = MLPPushPolicy(
                self.rng,
                feature_dim=feature_dim,
                hidden_dim=args.hidden_dim,
                init_noise=args.init_noise,
                learning_rate=args.learning_rate,
            )
        elif args.policy == "knn":
            self.policy = KNNPushPolicy(max_memory=args.max_memory, neighbors=args.neighbors)
        elif args.policy == "trajectory":
            self.policy = TrajectoryLibraryPolicy(max_trajectories=args.max_trajectories)
        else:
            raise ValueError(f"unknown policy: {args.policy}")
        self.states = self._init_states(seed)

    def close(self) -> None:
        self.env.close()

    def _init_states(self, seed: int) -> list[FetchPushReplayState]:
        states: list[FetchPushReplayState] = []
        for replay_idx in range(self.args.replay_states):
            base_seed = self.args.seed_offset + seed * 1000 + replay_idx
            obs, _info = self.env.reset(seed=base_seed)
            unwrapped = self.env.unwrapped
            base_goal = np.array(obs["desired_goal"], dtype=float)
            base_object = np.array(obs["achieved_goal"], dtype=float)
            states.append(
                FetchPushReplayState(
                    base_seed=base_seed,
                    base_goal=base_goal.tolist(),
                    base_object=base_object.tolist(),
                    center=np.array(unwrapped.initial_gripper_xpos, dtype=float).tolist(),
                    obj_range=float(unwrapped.obj_range),
                    direction=object_direction(base_object, base_goal, self.rng, self.args.direction_jitter).tolist(),
                )
            )
        return states

    def object_position(self, replay_idx: int, sigma: float) -> np.ndarray:
        state = self.states[replay_idx]
        return clipped_object(
            np.array(state.base_object, dtype=float),
            np.array(state.direction, dtype=float),
            float(sigma),
            np.array(state.center, dtype=float),
            float(state.obj_range),
        )

    def reset_perturbed(self, replay_idx: int, sigma: float, trial_seed: int) -> dict[str, np.ndarray]:
        state = self.states[replay_idx]
        obs, _info = self.env.reset(seed=trial_seed)
        unwrapped = self.env.unwrapped
        unwrapped.goal = np.array(state.base_goal, dtype=float)
        set_object_position(unwrapped, self.object_position(replay_idx, sigma))
        return unwrapped._get_obs()

    def rollout(self, replay_idx: int, sigma: float, trial_seed: int, *, train: bool) -> tuple[bool, float]:
        obs = self.reset_perturbed(replay_idx, sigma, trial_seed)
        threshold = float(self.env.unwrapped.distance_threshold)
        if hasattr(self.policy, "start_episode"):
            self.policy.start_episode(replay_idx, sigma)
        losses: list[float] = []
        trajectory_actions: list[np.ndarray] = []
        for step in range(self.args.horizon + 1):
            distance = float(
                np.linalg.norm(np.array(obs["achieved_goal"], dtype=float) - np.array(obs["desired_goal"], dtype=float))
            )
            if distance <= threshold:
                if train and hasattr(self.policy, "add_trajectory"):
                    self.policy.add_trajectory(replay_idx, sigma, trajectory_actions)
                return True, float(np.mean(losses)) if losses else 0.0
            if step == self.args.horizon:
                if train and hasattr(self.policy, "add_trajectory"):
                    self.policy.add_trajectory(replay_idx, sigma, trajectory_actions)
                return False, float(np.mean(losses)) if losses else 0.0
            features = obs_features(obs)
            try:
                action = self.policy.action(features, step)
            except TypeError:
                action = self.policy.action(features)
            if train:
                target = controller_action(
                    self.args.teacher_controller,
                    obs,
                    threshold=threshold,
                    gain=self.args.teacher_gain,
                )
                trajectory_actions.append(np.array(target, dtype=float))
                if hasattr(self.policy, "update"):
                    losses.append(self.policy.update(features, target))
                if self.args.teacher_force_train:
                    action = target
            obs, _reward, _terminated, truncated, _info = self.env.step(action)
            if bool(truncated):
                if train and hasattr(self.policy, "add_trajectory"):
                    self.policy.add_trajectory(replay_idx, sigma, trajectory_actions)
                distance = float(
                    np.linalg.norm(np.array(obs["achieved_goal"], dtype=float) - np.array(obs["desired_goal"], dtype=float))
                )
                return distance <= threshold, float(np.mean(losses)) if losses else 0.0
        raise AssertionError("unreachable")

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> tuple[bool, float]:
        return self.rollout(replay_idx, sigma, int(rng.integers(1_000_000_000)), train=True)

    def success_prob(self, replay_idx: int, sigma: float, rng: np.random.Generator, trials: int) -> float:
        successes = 0
        for _ in range(trials):
            success, _loss = self.rollout(replay_idx, sigma, int(rng.integers(1_000_000_000)), train=False)
            successes += int(success)
        return successes / trials

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        _success, loss = self.rollout(replay_idx, sigma, int(rng.integers(1_000_000_000)), train=True)
        return loss


def write_estimate(record: LevelRecord, estimate: Any) -> None:
    record.r_alpha_hat = float(estimate.r_alpha)
    record.sharpness_hat = float(estimate.sharpness)
    record.uncertainty_hat = float(estimate.r_uncertainty)
    record.recovery_curve_hat = estimate.recovery.tolist()


def init_records(bench: FetchPushObjectProbe, rng: np.random.Generator, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, _state in enumerate(bench.states):
        record = LevelRecord(
            id=f"fetchpush_object_{replay_idx}",
            domain="official_fetchpush_object_state_recovery",
            task_id=args.env_id,
            clean_success_hat=bench.success_prob(replay_idx, 0.0, rng, args.record_trials),
            feasibility_hat=1.0,
            perturbation_family="adverse_object_state_offset",
            sigma_grid=args.initial_probes,
        )
        estimator = IsotonicCurveEstimator(args.max_radius, args.alpha)
        for sigma in args.initial_probes:
            for _ in range(args.record_trials):
                trial_seed = int(rng.integers(1_000_000_000))
                success, _loss = bench.rollout(replay_idx, sigma, trial_seed, train=False)
                record.add_observation(sigma, success)
                estimator.update_bernoulli(sigma, success)
                if args.warmstart_policy:
                    bench.rollout(replay_idx, sigma, trial_seed, train=True)
        write_estimate(record, estimator.fit())
        records.append(record)
    return records


def refresh_records(
    bench: FetchPushObjectProbe,
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
        sigma = float(np.clip(estimator.next_probe(rng, jitter=args.refresh_jitter), 0.0, args.max_radius))
        for _ in range(args.refresh_trials):
            success, _loss = bench.rollout(int(replay_idx), sigma, int(rng.integers(1_000_000_000)), train=False)
            record.add_observation(sigma, success)
            estimator.update_bernoulli(sigma, success)
        write_estimate(record, estimator.fit())
        record.clean_success_hat = bench.success_prob(int(replay_idx), 0.0, rng, args.record_trials)
        record.last_evaluated_step = step


def evaluate(bench: FetchPushObjectProbe, rng: np.random.Generator, args: argparse.Namespace) -> dict[str, float]:
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
    bench: FetchPushObjectProbe,
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


def run_method(args: argparse.Namespace, method: str, seed: int) -> FetchPushProbeResult:
    rng = np.random.default_rng(args.seed_offset + 50_000 + seed)
    bench = FetchPushObjectProbe(args, seed)
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
        return FetchPushProbeResult(
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
    parser.add_argument("--env-id", default="FetchPush-v4")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--replay-states", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--teacher-controller", choices=["scripted_push", "scripted_push_far", "scripted_push_sweep"], default="scripted_push_sweep")
    parser.add_argument("--teacher-gain", type=float, default=8.0)
    parser.add_argument("--teacher-force-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmstart-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--policy", choices=["linear", "mlp", "knn", "trajectory"], default="linear")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-memory", type=int, default=20000)
    parser.add_argument("--max-trajectories", type=int, default=512)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--init-noise", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-radius", type=float, default=0.20)
    parser.add_argument("--fixed-radius", type=float, default=0.02)
    parser.add_argument("--eval-radii", default="0.00,0.02,0.04,0.06,0.08,0.12,0.16,0.20")
    parser.add_argument("--initial-probes", default="0.00,0.02,0.08,0.20")
    parser.add_argument("--eval-trials", type=int, default=2)
    parser.add_argument("--record-trials", type=int, default=2)
    parser.add_argument("--quick-trials", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.80)
    parser.add_argument("--target-radius", type=float, default=0.014)
    parser.add_argument("--radius-bandwidth", type=float, default=0.025)
    parser.add_argument("--radius-noise", type=float, default=0.18)
    parser.add_argument("--radius-uniform-mix", type=float, default=0.30)
    parser.add_argument("--uniform-mix", type=float, default=0.10)
    parser.add_argument("--priority-temperature", type=float, default=1.0)
    parser.add_argument("--baseline-candidates", type=int, default=4)
    parser.add_argument("--refresh-per-eval", type=int, default=2)
    parser.add_argument("--refresh-trials", type=int, default=2)
    parser.add_argument("--refresh-jitter", type=float, default=0.02)
    parser.add_argument("--direction-jitter", type=float, default=0.10)
    parser.add_argument("--seed-offset", type=int, default=121_000)
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
