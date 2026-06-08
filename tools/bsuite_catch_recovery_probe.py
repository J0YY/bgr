#!/usr/bin/env python3
"""Run a fixed bsuite Catch recovery replay screen.

This optional external-package screen uses bsuite's Catch task definition with
exact restart states. The perturbation radius moves the paddle away from the
falling ball while preserving task feasibility, so larger radii require more
recovery control before the ball reaches the bottom row.
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


ACTIONS = (-1, 0, 1)


@dataclass(frozen=True, slots=True)
class CatchReplayState:
    ball_x: int
    ball_y: int
    paddle_x: int


@dataclass(frozen=True, slots=True)
class CatchProbeResult:
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
            "bsuite Catch recovery probe requires bsuite in an isolated environment, "
            "for example /tmp/bgr_bsuite_venv."
        ) from exc
    return {
        "bsuite": getattr(bsuite, "__version__", version("bsuite")),
        "bsuite-package": version("bsuite"),
        "dm-env": version("dm-env"),
        "numpy": np.__version__,
    }


class LinearSoftmaxPolicy:
    def __init__(self, *, rng: np.random.Generator, learning_rate: float, init_noise: float, feature_dim: int) -> None:
        self.learning_rate = float(learning_rate)
        self.weights = rng.normal(0.0, float(init_noise), size=(3, feature_dim))

    def action(self, features: np.ndarray) -> int:
        return int(np.argmax(self.weights @ features))

    def loss(self, features: np.ndarray, target: int) -> float:
        probs = self._probs(features)
        return float(-np.log(max(probs[int(target)], 1e-12)))

    def update(self, features: np.ndarray, target: int) -> None:
        probs = self._probs(features)
        grad = probs[:, None] * features[None, :]
        grad[int(target), :] -= features
        self.weights -= self.learning_rate * grad

    def _probs(self, features: np.ndarray) -> np.ndarray:
        logits = self.weights @ features
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        probs /= float(np.sum(probs))
        return probs


class CatchRecoveryProbe:
    def __init__(self, *, seed: int, args: argparse.Namespace) -> None:
        try:
            from bsuite.environments.catch import Catch
        except ImportError as exc:
            raise SystemExit(
                "bsuite Catch recovery probe requires bsuite in an isolated environment, "
                "for example /tmp/bgr_bsuite_venv."
            ) from exc

        self.rng = np.random.default_rng(seed + 331_000)
        self.args = args
        self.env = Catch(rows=args.rows, columns=args.columns, seed=seed + 23_000)
        self.rows = int(args.rows)
        self.columns = int(args.columns)
        self.states = self._select_replay_states(args.replay_states)
        self.policy = LinearSoftmaxPolicy(
            rng=self.rng,
            learning_rate=args.learning_rate,
            init_noise=args.policy_init_noise,
            feature_dim=self.feature_dim,
        )
        self._pretrain()

    @property
    def feature_dim(self) -> int:
        return 8

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def _select_replay_states(self, count: int) -> list[CatchReplayState]:
        min_y = max(1, int(self.args.min_ball_y))
        max_y = min(self.rows - self.args.max_radius - 2, int(self.args.max_ball_y))
        candidates = [
            CatchReplayState(ball_x=ball_x, ball_y=ball_y, paddle_x=ball_x)
            for ball_y in range(min_y, max_y + 1)
            for ball_x in range(self.columns)
        ]
        if not candidates:
            raise ValueError("no feasible bsuite Catch replay states selected")
        candidates.sort(key=lambda item: (item.ball_y, item.ball_x))
        if count >= len(candidates):
            return candidates
        indexes = np.linspace(0, len(candidates) - 1, int(count), dtype=int)
        return [candidates[int(index)] for index in indexes]

    def _pretrain(self) -> None:
        for _ in range(int(self.args.policy_init_steps)):
            replay_idx = int(self.rng.integers(len(self.states)))
            sigma = float(self.rng.uniform(0.0, 1.0))
            state = self.sample_start(replay_idx, sigma, self.rng)
            features = self.features(*state)
            self.policy.update(features, self.teacher_action(*state))

    def perturbation_states(self, replay_idx: int, sigma: float) -> list[tuple[int, int, int]]:
        replay = self.states[replay_idx]
        radius = int(round(float(np.clip(sigma, 0.0, 1.0)) * self.args.max_radius))
        if radius == 0:
            return [(replay.ball_x, replay.ball_y, replay.paddle_x)]
        states: list[tuple[int, int, int]] = []
        for sign in (-1, 1):
            paddle_x = replay.paddle_x + sign * radius
            if 0 <= paddle_x < self.columns:
                states.append((replay.ball_x, replay.ball_y, paddle_x))
        return states or [(replay.ball_x, replay.ball_y, replay.paddle_x)]

    def sample_start(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> tuple[int, int, int]:
        states = self.perturbation_states(replay_idx, sigma)
        return states[int(rng.integers(len(states)))]

    def success_prob(self, replay_idx: int, sigma: float) -> float:
        states = self.perturbation_states(replay_idx, sigma)
        return float(np.mean([self.rollout_state(*state, train=False) for state in states]))

    def rollout(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> bool:
        return self.rollout_state(*self.sample_start(replay_idx, sigma, rng), train=False)

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> None:
        self.rollout_state(*self.sample_start(replay_idx, sigma, rng), train=True)

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        state = self.sample_start(replay_idx, sigma, rng)
        features = self.features(*state)
        return self.policy.loss(features, self.teacher_action(*state))

    def rollout_state(self, ball_x: int, ball_y: int, paddle_x: int, *, train: bool) -> bool:
        for _ in range(self.rows):
            features = self.features(ball_x, ball_y, paddle_x)
            if train:
                self.policy.update(features, self.teacher_action(ball_x, ball_y, paddle_x))
            action = self.policy.action(features)
            paddle_x = int(np.clip(paddle_x + ACTIONS[action], 0, self.columns - 1))
            ball_y += 1
            if ball_y == self.rows - 1:
                return paddle_x == ball_x
        return False

    def teacher_action(self, ball_x: int, _ball_y: int, paddle_x: int) -> int:
        if paddle_x > ball_x:
            return 0
        if paddle_x < ball_x:
            return 2
        return 1

    def features(self, ball_x: int, ball_y: int, paddle_x: int) -> np.ndarray:
        width = max(1, self.columns - 1)
        height = max(1, self.rows - 1)
        distance = ball_x - paddle_x
        remaining = self.rows - 1 - ball_y
        return np.array(
            [
                ball_x / width,
                ball_y / height,
                paddle_x / width,
                distance / width,
                abs(distance) / width,
                remaining / height,
                float(abs(distance) <= remaining),
                1.0,
            ],
            dtype=float,
        )


def run_method(args: argparse.Namespace, method: str, seed: int) -> CatchProbeResult:
    rng = np.random.default_rng(seed + 337_000)
    bench = CatchRecoveryProbe(seed=seed, args=args)
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
        return CatchProbeResult(
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


def init_records(bench: CatchRecoveryProbe, rng: np.random.Generator, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, replay in enumerate(bench.states):
        record = LevelRecord(
            id=f"bsuite_catch_{replay.ball_x}_{replay.ball_y}",
            domain="bsuite_catch_recovery",
            task_id=f"Catch-{bench.rows}x{bench.columns}",
            clean_success_hat=bench.success_prob(replay_idx, 0.0),
            feasibility_hat=1.0,
            perturbation_family="paddle_column_restart",
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
    bench: CatchRecoveryProbe,
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


def evaluate(bench: CatchRecoveryProbe, args: argparse.Namespace) -> dict[str, float]:
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
    bench: CatchRecoveryProbe,
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
    parser.add_argument("--out", default="runs/bsuite_catch_recovery_probe")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--columns", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=10)
    parser.add_argument("--replay-states", type=int, default=36)
    parser.add_argument("--min-ball-y", type=int, default=2)
    parser.add_argument("--max-ball-y", type=int, default=6)
    parser.add_argument("--max-radius", type=int, default=4)
    parser.add_argument("--eval-grid-size", type=int, default=9)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    parser.add_argument("--policy-init-noise", type=float, default=0.10)
    parser.add_argument("--policy-init-steps", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--target-radius", type=float, default=0.45)
    parser.add_argument("--radius-bandwidth", type=float, default=0.35)
    parser.add_argument("--fixed-radius", type=float, default=0.50)
    parser.add_argument("--baseline-candidates", type=int, default=12)
    parser.add_argument("--initial-probes", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--min-trials", type=int, default=1)
    parser.add_argument("--refresh-per-eval", type=int, default=12)
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
