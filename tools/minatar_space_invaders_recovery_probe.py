#!/usr/bin/env python3
"""Run a fixed MinAtar Space Invaders recovery replay screen.

This external-package screen is authorized by the fixed 20-seed Space Invaders
calibration. It uses MinAtar's package-owned dynamics, exact seed/checkpoint
reconstruction, a fixed align-and-fire teacher, and rightward player-column
perturbations.
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
from tools.minatar_space_invaders_recovery_calibration import (
    bullet_danger,
    checkpoint,
    column_has_alien,
    controller_action,
    nearest_alien_col,
    perturb_player,
)


ACTIONS = (0, 1, 3, 5)


@dataclass(frozen=True, slots=True)
class SpaceInvadersProbeResult:
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
            "MinAtar Space Invaders recovery probe requires MinAtar in an isolated "
            "environment, for example /tmp/bgr_minatar_venv."
        ) from exc
    return {
        "MinAtar": version("MinAtar"),
        "numpy": np.__version__,
    }


class SpaceInvadersRecoveryProbe:
    def __init__(self, *, seed: int, args: argparse.Namespace) -> None:
        try:
            from minatar import Environment
        except ImportError as exc:
            raise SystemExit(
                "MinAtar Space Invaders recovery probe requires MinAtar in an isolated "
                "environment, for example /tmp/bgr_minatar_venv."
            ) from exc

        self.args = args
        self.rng = np.random.default_rng(seed + 917_000)
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
        return len(space_invaders_features(self.env))

    def close(self) -> None:
        self.env.close_display()

    def restore_start(self, replay_idx: int, sigma: float) -> None:
        replay_seed = int(self.replay_seeds[int(replay_idx)])
        checkpoint(self.env, seed=replay_seed, burn_in=self.args.burn_in)
        perturb_player(self.env, sigma=float(sigma), mode=self.args.perturbation)
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
        reward = 0
        for _step in range(1, int(self.args.horizon) + 1):
            features = self.features()
            target = controller_action(self.env)
            if train:
                self.policy.update(features, target)
            step_reward, terminal = self.env.act(self.policy.action(features))
            reward += int(step_reward)
            if terminal:
                break
        return (not bool(self.env.env.terminal)) and reward >= int(self.args.reward_threshold)

    def features(self) -> np.ndarray:
        return space_invaders_features(self.env)


def space_invaders_features(env: Any) -> np.ndarray:
    game = env.env
    pos = int(game.pos)
    alien_map = np.asarray(game.alien_map)
    enemy_bullets = np.asarray(game.e_bullet_map)
    friendly_bullets = np.asarray(game.f_bullet_map)
    target = nearest_alien_col(game, pos)
    target_dx = 0 if target is None else int(target) - pos
    col_counts = alien_map.sum(axis=0).astype(float)
    local_cols = [max(0, pos - 1), pos, min(9, pos + 1)]
    local_aliens = [float(col_counts[col] / 4.0) for col in local_cols]
    local_enemy_bullets = [float(enemy_bullets[:, col].any()) for col in local_cols]
    lower_enemy_bullets = [float(enemy_bullets[7:10, col].any()) for col in local_cols]
    local_friendly_bullets = [float(friendly_bullets[:, col].any()) for col in local_cols]
    nearest_rows = np.where(alien_map[:, pos])[0]
    nearest_row = 0 if nearest_rows.size == 0 else int(nearest_rows.max())
    return np.array(
        [
            pos / 9.0,
            target_dx / 9.0,
            float(target_dx < 0),
            float(target_dx == 0),
            float(target_dx > 0),
            int(game.shot_timer) / 10.0,
            int(game.alien_move_timer) / 10.0,
            int(game.alien_shot_timer) / 10.0,
            float(int(game.alien_dir) > 0),
            float(column_has_alien(game, pos)),
            float(bullet_danger(game, pos)),
            float(bullet_danger(game, pos - 1)) if pos > 0 else 1.0,
            float(bullet_danger(game, pos + 1)) if pos < 9 else 1.0,
            float(pos == 0),
            float(pos == 9),
            float(np.count_nonzero(alien_map) / 24.0),
            float(np.count_nonzero(enemy_bullets) / 10.0),
            float(np.count_nonzero(friendly_bullets) / 10.0),
            nearest_row / 9.0,
            *local_aliens,
            *local_enemy_bullets,
            *lower_enemy_bullets,
            *local_friendly_bullets,
            1.0,
        ],
        dtype=float,
    )


def run_method(args: argparse.Namespace, method: str, seed: int) -> SpaceInvadersProbeResult:
    rng = np.random.default_rng(seed + 918_000)
    bench = SpaceInvadersRecoveryProbe(seed=seed, args=args)
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
        return SpaceInvadersProbeResult(
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


def init_records(bench: SpaceInvadersRecoveryProbe, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, replay_seed in enumerate(bench.replay_seeds):
        record = LevelRecord(
            id=f"minatar_space_invaders_seed_{replay_seed}",
            domain="minatar_space_invaders_recovery",
            task_id="MinAtar-SpaceInvaders",
            clean_success_hat=bench.success_prob(replay_idx, 0.0),
            feasibility_hat=1.0,
            perturbation_family=f"{args.perturbation}_player_column_displacement",
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
    bench: SpaceInvadersRecoveryProbe,
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


def evaluate(bench: SpaceInvadersRecoveryProbe, args: argparse.Namespace) -> dict[str, float]:
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
    bench: SpaceInvadersRecoveryProbe,
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
    parser.add_argument("--out", default="runs/minatar_space_invaders_recovery_probe")
    parser.add_argument("--game", default="space_invaders")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--replay-seeds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=10)
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--reward-threshold", type=int, default=2)
    parser.add_argument("--perturbation", choices=["right", "away"], default="right")
    parser.add_argument("--max-radius", type=float, default=6.0)
    parser.add_argument("--eval-grid-size", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--policy-init-noise", type=float, default=0.01)
    parser.add_argument("--policy-init-steps", type=int, default=250)
    parser.add_argument("--sticky-action-prob", type=float, default=0.0)
    parser.add_argument("--difficulty-ramping", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--target-radius", type=float, default=2.2)
    parser.add_argument("--radius-bandwidth", type=float, default=1.2)
    parser.add_argument("--fixed-radius", type=float, default=2.0)
    parser.add_argument("--baseline-candidates", type=int, default=8)
    parser.add_argument("--initial-probes", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0])
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
                    row["final_clean_mean"],
                    row["final_rauc_mean"],
                    row["final_median_r80_mean"],
                    row["rauc_aulc_mean"],
                    row["best_rauc_mean"],
                ]
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
