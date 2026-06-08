#!/usr/bin/env python3
"""Run a fixed MinAtar Asterix recovery replay screen.

This external-package screen uses MinAtar's Asterix dynamics and exact
seed/checkpoint reconstruction. It is authorized by the fixed 12-seed Asterix
calibration; replay states are the calibration-clean checkpoint seeds fixed
before method comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from bgr.curve_estimators import IsotonicCurveEstimator
from bgr.metrics import critical_radius, recovery_auc
from bgr.priorities import BGRPriorityScorer
from bgr.records import LevelRecord
from bgr.samplers import mixed_priority_probs, sample_boundary_radius
from tools.minatar_asterix_recovery_calibration import (
    checkpoint,
    controller_action,
    enemy_threatens,
    perturb_player,
    safe_cell,
)


ACTIONS = (0, 1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class AsterixProbeResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


class LinearSoftmaxPolicy:
    def __init__(self, *, rng: np.random.Generator, learning_rate: float, init_noise: float, feature_dim: int) -> None:
        self.learning_rate = float(learning_rate)
        self.weights = rng.normal(0.0, float(init_noise), size=(len(ACTIONS), int(feature_dim)))

    def action(self, features: np.ndarray) -> int:
        return ACTIONS[int(np.argmax(self.weights @ features))]

    def loss(self, features: np.ndarray, target_action: int) -> float:
        probs = self._probs(features)
        return float(-np.log(max(probs[action_index(target_action)], 1e-12)))

    def update(self, features: np.ndarray, target_action: int) -> None:
        target = action_index(target_action)
        probs = self._probs(features)
        grad = probs[:, None] * features[None, :]
        grad[target, :] -= features
        self.weights -= self.learning_rate * grad

    def _probs(self, features: np.ndarray) -> np.ndarray:
        logits = self.weights @ features
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        probs /= float(np.sum(probs))
        return probs


def action_index(action: int) -> int:
    return ACTIONS.index(int(action))


def package_versions() -> dict[str, str]:
    try:
        import minatar  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "MinAtar Asterix recovery probe requires MinAtar in an isolated "
            "environment, for example /tmp/bgr_minatar_venv."
        ) from exc
    return {
        "MinAtar": version("MinAtar"),
        "numpy": np.__version__,
    }


class AsterixRecoveryProbe:
    def __init__(self, *, seed: int, args: argparse.Namespace) -> None:
        try:
            from minatar import Environment
        except ImportError as exc:
            raise SystemExit(
                "MinAtar Asterix recovery probe requires MinAtar in an isolated "
                "environment, for example /tmp/bgr_minatar_venv."
            ) from exc

        self.args = args
        self.rng = np.random.default_rng(seed + 881_000)
        self.env = Environment(
            args.game,
            sticky_action_prob=args.sticky_action_prob,
            difficulty_ramping=args.difficulty_ramping,
        )
        self.replay_seeds = parse_ints(args.replay_seeds)
        if not self.replay_seeds:
            raise ValueError("--replay-seeds must not be empty")
        self.policy = LinearSoftmaxPolicy(
            rng=self.rng,
            learning_rate=args.learning_rate,
            init_noise=args.policy_init_noise,
            feature_dim=self.feature_dim,
        )
        self._pretrain()

    @property
    def feature_dim(self) -> int:
        return 22

    def close(self) -> None:
        self.env.close_display()

    def restore_start(self, replay_idx: int, sigma: float) -> None:
        replay_seed = int(self.replay_seeds[int(replay_idx)])
        checkpoint(self.env, seed=replay_seed, burn_in=self.args.burn_in)
        perturb_player(self.env, seed=replay_seed, sigma=float(sigma))
        self.env.last_action = 0

    def _pretrain(self) -> None:
        integer_radii = np.arange(0, int(round(float(self.args.max_radius))) + 1, dtype=float)
        for _ in range(int(self.args.policy_init_steps)):
            replay_idx = int(self.rng.integers(len(self.replay_seeds)))
            sigma = float(self.rng.choice(integer_radii))
            self.rollout(replay_idx, sigma, train=True)

    def success_prob(self, replay_idx: int, sigma: float) -> float:
        return float(self.rollout(replay_idx, sigma, train=False))

    def train_step(self, replay_idx: int, sigma: float) -> None:
        self.rollout(replay_idx, sigma, train=True)

    def loss_proxy(self, replay_idx: int, sigma: float) -> float:
        self.restore_start(replay_idx, sigma)
        return self.policy.loss(self.features(), controller_action(self.env))

    def rollout(self, replay_idx: int, sigma: float, *, train: bool) -> bool:
        self.restore_start(replay_idx, sigma)
        terminal = False
        reward = 0
        step = 0
        for step in range(1, int(self.args.horizon) + 1):
            features = self.features()
            target = controller_action(self.env)
            if train:
                self.policy.update(features, target)
            step_reward, terminal = self.env.act(self.policy.action(features))
            reward += int(step_reward)
            if terminal:
                break
        return (not bool(self.env.env.terminal)) and step >= int(self.args.horizon) and reward >= 1

    def features(self) -> np.ndarray:
        return asterix_features(self.env)


def nearest_entity(game: Any, *, want_gold: bool) -> tuple[int, int, bool, bool] | None:
    entities = [
        (int(entity[0]), int(entity[1]), bool(entity[2]), bool(entity[3]))
        for entity in game.entities
        if entity is not None and bool(entity[3]) is want_gold
    ]
    if not entities:
        return None
    entities.sort(key=lambda entity: abs(entity[0] - int(game.player_x)) + abs(entity[1] - int(game.player_y)))
    return entities[0]


def asterix_features(env: Any) -> np.ndarray:
    game = env.env
    gold = nearest_entity(game, want_gold=True)
    enemy = nearest_entity(game, want_gold=False)
    gold_dx, gold_dy = (0, 0) if gold is None else (gold[0] - int(game.player_x), gold[1] - int(game.player_y))
    enemy_dx, enemy_dy = (0, 0) if enemy is None else (enemy[0] - int(game.player_x), enemy[1] - int(game.player_y))
    safe_moves = [
        float(safe_cell(game, int(game.player_x) + dx, int(game.player_y) + dy))
        for dx, dy in [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)]
    ]
    return np.array(
        [
            int(game.player_x) / 9.0,
            int(game.player_y) / 8.0,
            gold_dx / 9.0,
            gold_dy / 8.0,
            abs(gold_dx) / 9.0,
            abs(gold_dy) / 8.0,
            float(gold is not None),
            enemy_dx / 9.0,
            enemy_dy / 8.0,
            abs(enemy_dx) / 9.0,
            abs(enemy_dy) / 8.0,
            float(enemy is not None),
            float(enemy_threatens(game, int(game.player_x), int(game.player_y))),
            int(game.spawn_timer) / 10.0,
            int(game.move_timer) / 5.0,
            float(sum(entity is not None for entity in game.entities)) / 8.0,
            *safe_moves,
            1.0,
        ],
        dtype=float,
    )


def run_method(args: argparse.Namespace, method: str, seed: int) -> AsterixProbeResult:
    rng = np.random.default_rng(seed + 883_000)
    bench = AsterixRecoveryProbe(seed=seed, args=args)
    try:
        records = init_records(bench, args)
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
                bench.train_step(replay_idx, sigma)
                if method.startswith("bgr"):
                    records[replay_idx].add_observation(sigma, bench.success_prob(replay_idx, sigma))
                    records[replay_idx].replay_count += 1

        xs = np.array([row["step"] for row in history], dtype=float)
        ys = np.array([row["rauc"] for row in history], dtype=float)
        final = history[-1]
        return AsterixProbeResult(
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


def init_records(bench: AsterixRecoveryProbe, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, replay_seed in enumerate(bench.replay_seeds):
        record = LevelRecord(
            id=f"minatar_asterix_seed_{replay_seed}",
            domain="minatar_asterix_recovery",
            task_id="MinAtar-Asterix",
            clean_success_hat=bench.success_prob(replay_idx, 0.0),
            feasibility_hat=1.0,
            perturbation_family="seed_fixed_player_cell_displacement",
            sigma_grid=args.initial_probes,
        )
        estimator = IsotonicCurveEstimator(args.max_radius, args.alpha)
        for sigma in args.initial_probes:
            for _ in range(args.min_trials):
                success = bench.success_prob(replay_idx, sigma)
                record.add_observation(sigma, success)
                estimator.update_bernoulli(sigma, success)
        write_estimate(record, estimator.fit())
        records.append(record)
    return records


def refresh_records(
    bench: AsterixRecoveryProbe,
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
            success = bench.success_prob(int(replay_idx), sigma)
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


def evaluate(bench: AsterixRecoveryProbe, args: argparse.Namespace) -> dict[str, float]:
    grid = np.linspace(0.0, args.max_radius, args.eval_grid_size)
    clean: list[float] = []
    raucs: list[float] = []
    radii: list[float] = []
    for replay_idx in range(len(bench.replay_seeds)):
        curve = np.array([bench.success_prob(replay_idx, sigma) for sigma in grid], dtype=float)
        clean.append(float(curve[0]))
        raucs.append(recovery_auc(grid, curve, sigma_max=args.max_radius))
        radii.append(critical_radius(grid, curve, alpha=args.alpha))
    return {
        "clean": float(np.mean(clean)),
        "rauc": float(np.mean(raucs)),
        "median_r80": float(np.median(radii)),
    }


def sample_training_pair(
    method: str,
    bench: AsterixRecoveryProbe,
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
        scores = [1.0 - bench.success_prob(int(candidate), float(sigma)) for candidate, sigma in zip(candidates, sigmas, strict=True)]
        idx = int(np.argmax(scores))
        return int(candidates[idx]), float(sigmas[idx])
    if method == "td_loss":
        candidates = rng.choice(len(records), size=min(args.baseline_candidates, len(records)), replace=False)
        sigmas = rng.uniform(0.0, args.max_radius, size=len(candidates))
        scores = [bench.loss_proxy(int(candidate), float(sigma)) for candidate, sigma in zip(candidates, sigmas, strict=True)]
        idx = int(np.argmax(scores))
        return int(candidates[idx]), float(sigmas[idx])
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
        return replay_idx, sigma
    raise ValueError(f"unknown method: {method}")


def parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one string")
    return values


def aggregate_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    methods = sorted({str(row["method"]) for row in rows})
    aggregate: list[dict[str, str]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        aggregate.append(
            {
                "method": method,
                "seeds": str(len(method_rows)),
                "final_clean_mean": f"{mean(method_rows, 'final_clean'):.6f}",
                "final_rauc_mean": f"{mean(method_rows, 'final_rauc'):.6f}",
                "final_median_r80_mean": f"{mean(method_rows, 'final_median_r80'):.6f}",
                "rauc_aulc_mean": f"{mean(method_rows, 'rauc_aulc'):.6f}",
                "best_rauc_mean": f"{mean(method_rows, 'best_rauc'):.6f}",
            }
        )
    return aggregate


def mean(rows: list[dict[str, float | int | str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/minatar_asterix_recovery_probe")
    parser.add_argument("--game", default="asterix")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--replay-seeds", default="0,1,3,4,5,6,7,8,9,11")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=10)
    parser.add_argument("--burn-in", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--max-radius", type=float, default=8.0)
    parser.add_argument("--eval-grid-size", type=int, default=9)
    parser.add_argument("--learning-rate", type=float, default=0.50)
    parser.add_argument("--policy-init-noise", type=float, default=0.01)
    parser.add_argument("--policy-init-steps", type=int, default=1000)
    parser.add_argument("--sticky-action-prob", type=float, default=0.0)
    parser.add_argument("--difficulty-ramping", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--target-radius", type=float, default=5.333333333333333)
    parser.add_argument("--radius-bandwidth", type=float, default=2.0)
    parser.add_argument("--fixed-radius", type=float, default=5.0)
    parser.add_argument("--baseline-candidates", type=int, default=8)
    parser.add_argument("--initial-probes", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0, 6.0, 8.0])
    parser.add_argument("--min-trials", type=int, default=1)
    parser.add_argument("--refresh-per-eval", type=int, default=8)
    parser.add_argument("--refresh-trials", type=int, default=1)
    parser.add_argument("--refresh-jitter", type=float, default=0.35)
    parser.add_argument("--radius-noise", type=float, default=0.70)
    parser.add_argument("--radius-uniform-mix", type=float, default=0.25)
    parser.add_argument("--priority-temperature", type=float, default=0.8)
    parser.add_argument("--uniform-mix", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    methods = parse_strings(args.methods)
    seeds = parse_ints(args.seeds)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    history_rows: list[dict[str, float | int | str]] = []
    results: list[dict[str, Any]] = []
    for method in methods:
        for seed in seeds:
            start = time.perf_counter()
            result = run_method(args, method, seed)
            elapsed = time.perf_counter() - start
            print(
                f"[done] method={method} seed={seed} rauc={result.final_rauc:.4f} "
                f"clean={result.final_clean:.4f} r80={result.final_median_r80:.4f} "
                f"aulc={result.rauc_aulc:.4f} elapsed={elapsed:.2f}s",
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
                history_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "step": item["step"],
                        "clean": item["clean"],
                        "rauc": item["rauc"],
                        "median_r80": item["median_r80"],
                    }
                )

    (out_dir / "results.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2), encoding="utf-8")
    write_csv(out_dir / "summary.csv", rows)
    write_csv(out_dir / "history.csv", history_rows)
    write_csv(out_dir / "aggregate.csv", aggregate_rows(rows))
    (out_dir / "package_versions.json").write_text(json.dumps(package_versions(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary(rows))


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary(rows: list[dict[str, float | int | str]]) -> str:
    lines = ["method,final_clean_mean,final_rauc_mean,final_median_r80_mean,rauc_aulc_mean,best_rauc_mean"]
    for row in aggregate_rows(rows):
        lines.append(
            ",".join(
                [
                    row["method"],
                    row["final_clean_mean"][:8],
                    row["final_rauc_mean"][:8],
                    row["final_median_r80_mean"][:8],
                    row["rauc_aulc_mean"][:8],
                    row["best_rauc_mean"][:8],
                ]
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
