#!/usr/bin/env python3
"""Run a fixed bsuite DeepSea recovery replay screen.

This is an optional external-package pre-promotion screen. It uses bsuite's
DeepSea task definition and randomized action mapping, with exact restart
states along the sparse-reward chain. Perturbation radius moves a restart state
leftward from the solvable diagonal, so larger radii put the learner farther
from the narrow rewarding path.
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


@dataclass(frozen=True, slots=True)
class DeepSeaReplayState:
    row: int
    column: int


@dataclass(frozen=True, slots=True)
class DeepSeaProbeResult:
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
            "bsuite DeepSea recovery probe requires bsuite in an isolated environment, "
            "for example /tmp/bgr_bsuite_venv."
        ) from exc
    return {
        "bsuite": getattr(bsuite, "__version__", version("bsuite")),
        "bsuite-package": version("bsuite"),
        "dm-env": version("dm-env"),
        "numpy": np.__version__,
    }


class DeepSeaRecoveryProbe:
    def __init__(
        self,
        *,
        seed: int,
        args: argparse.Namespace,
    ) -> None:
        try:
            from bsuite.environments.deep_sea import DeepSea
        except ImportError as exc:
            raise SystemExit(
                "bsuite DeepSea recovery probe requires bsuite in an isolated environment, "
                "for example /tmp/bgr_bsuite_venv."
            ) from exc

        self.rng = np.random.default_rng(seed + 307_000)
        self.args = args
        self.env = DeepSea(
            size=args.size,
            deterministic=True,
            unscaled_move_cost=args.unscaled_move_cost,
            randomize_actions=args.randomize_actions,
            seed=seed + 11_000,
            mapping_seed=seed + 17_000,
        )
        self.mapping = np.asarray(self.env._action_mapping, dtype=int)
        self.size = int(args.size)
        self.q = self.rng.normal(0.0, float(args.q_init_noise), size=(self.size, self.size, 2))
        self.states = self._select_replay_states(args.replay_states)

    def _select_replay_states(self, count: int) -> list[DeepSeaReplayState]:
        lo = max(1, int(self.args.min_replay_row))
        hi = min(self.size - 2, int(self.args.max_replay_row if self.args.max_replay_row >= 0 else self.size - 2))
        rows = np.arange(lo, hi + 1, dtype=int)
        if rows.size == 0:
            raise ValueError("no DeepSea replay rows selected")
        if count < rows.size:
            rows = rows[np.linspace(0, rows.size - 1, int(count), dtype=int)]
        return [DeepSeaReplayState(row=int(row), column=int(row)) for row in rows]

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def perturbation_state(self, replay_idx: int, sigma: float) -> tuple[int, int]:
        replay = self.states[replay_idx]
        radius = int(round(float(np.clip(sigma, 0.0, 1.0)) * self.args.max_radius))
        column = max(0, replay.column - radius)
        return replay.row, column

    def success_prob(self, replay_idx: int, sigma: float) -> float:
        row, column = self.perturbation_state(replay_idx, sigma)
        return float(self.rollout_state(row, column, train=False, epsilon=False))

    def rollout(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> bool:
        row, column = self.perturbation_state(replay_idx, sigma)
        return self.rollout_state(row, column, train=False, epsilon=True, rng=rng)

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> None:
        row, column = self.perturbation_state(replay_idx, sigma)
        self.rollout_state(row, column, train=True, epsilon=True, rng=rng)

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        row, column = self.perturbation_state(replay_idx, sigma)
        action = self._action(row, column, rng, epsilon=True)
        next_row, next_column, reward, done = self.transition(row, column, action)
        target = reward if done else reward + self.args.discount * float(np.max(self.q[next_row, next_column]))
        return abs(target - float(self.q[row, column, action]))

    def rollout_state(
        self,
        row: int,
        column: int,
        *,
        train: bool,
        epsilon: bool,
        rng: np.random.Generator | None = None,
    ) -> bool:
        total_reward = 0.0
        row = int(row)
        column = int(column)
        while row < self.size:
            action = self._action(row, column, rng, epsilon=epsilon)
            next_row, next_column, reward, done = self.transition(row, column, action)
            total_reward += reward
            if train:
                target = reward if done else reward + self.args.discount * float(np.max(self.q[next_row, next_column]))
                self.q[row, column, action] += self.args.learning_rate * (target - float(self.q[row, column, action]))
            row, column = next_row, next_column
            if done:
                break
        return bool(total_reward >= self.args.success_return_threshold)

    def transition(self, row: int, column: int, action: int) -> tuple[int, int, float, bool]:
        action_right = int(action) == int(self.mapping[row, column])
        reward = 0.0
        if column == self.size - 1 and action_right:
            reward += 1.0
        if action_right:
            column = min(column + 1, self.size - 1)
            reward -= float(self.args.unscaled_move_cost) / self.size
        else:
            column = max(column - 1, 0)
        row += 1
        return row, column, reward, row == self.size

    def _action(
        self,
        row: int,
        column: int,
        rng: np.random.Generator | None,
        *,
        epsilon: bool,
    ) -> int:
        if epsilon and rng is not None and rng.random() < self.args.epsilon:
            return int(rng.integers(2))
        return int(np.argmax(self.q[row, column]))


def run_method(args: argparse.Namespace, method: str, seed: int) -> DeepSeaProbeResult:
    rng = np.random.default_rng(seed + 313_000)
    bench = DeepSeaRecoveryProbe(seed=seed, args=args)
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
        return DeepSeaProbeResult(
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


def init_records(bench: DeepSeaRecoveryProbe, rng: np.random.Generator, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, replay in enumerate(bench.states):
        record = LevelRecord(
            id=f"bsuite_deepsea_{replay.row}_{replay.column}",
            domain="bsuite_deepsea_recovery",
            task_id=f"DeepSea-size{bench.size}",
            clean_success_hat=bench.success_prob(replay_idx, 0.0),
            feasibility_hat=1.0,
            perturbation_family="left_column_restart",
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
    bench: DeepSeaRecoveryProbe,
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


def evaluate(bench: DeepSeaRecoveryProbe, args: argparse.Namespace) -> dict[str, float]:
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
    bench: DeepSeaRecoveryProbe,
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
    parser.add_argument("--out", default="runs/bsuite_deepsea_recovery_probe")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--size", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--eval-every", type=int, default=60)
    parser.add_argument("--train-batch-size", type=int, default=24)
    parser.add_argument("--replay-states", type=int, default=10)
    parser.add_argument("--min-replay-row", type=int, default=3)
    parser.add_argument("--max-replay-row", type=int, default=-1)
    parser.add_argument("--max-radius", type=int, default=4)
    parser.add_argument("--eval-grid-size", type=int, default=9)
    parser.add_argument("--learning-rate", type=float, default=0.50)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.12)
    parser.add_argument("--q-init-noise", type=float, default=0.01)
    parser.add_argument("--unscaled-move-cost", type=float, default=0.01)
    parser.add_argument("--randomize-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--success-return-threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--target-radius", type=float, default=0.45)
    parser.add_argument("--radius-bandwidth", type=float, default=0.35)
    parser.add_argument("--fixed-radius", type=float, default=0.50)
    parser.add_argument("--baseline-candidates", type=int, default=10)
    parser.add_argument("--initial-probes", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--min-trials", type=int, default=1)
    parser.add_argument("--refresh-per-eval", type=int, default=10)
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
