#!/usr/bin/env python3
"""Run a fixed MinAtar Breakout recovery replay screen.

This external-package screen uses MinAtar's Breakout dynamics with exact
internal-state restores. The perturbation radius moves the paddle away from the
ball after a fixed checkpoint, so larger radii require more recovery control
before the ball reaches the bottom row.
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


ACTIONS = (0, 1, 3)


@dataclass(frozen=True, slots=True)
class BreakoutSnapshot:
    ball_x: int
    ball_y: int
    ball_dir: int
    pos: int
    brick_map: np.ndarray
    strike: bool
    last_x: int
    last_y: int
    terminal: bool


@dataclass(frozen=True, slots=True)
class BreakoutProbeResult:
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


def package_versions() -> dict[str, str]:
    try:
        import minatar  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "MinAtar Breakout recovery probe requires MinAtar in an isolated "
            "environment, for example /tmp/bgr_minatar_venv."
        ) from exc
    return {
        "MinAtar": version("MinAtar"),
        "numpy": np.__version__,
    }


class BreakoutRecoveryProbe:
    def __init__(self, *, seed: int, args: argparse.Namespace) -> None:
        try:
            from minatar import Environment
        except ImportError as exc:
            raise SystemExit(
                "MinAtar Breakout recovery probe requires MinAtar in an isolated "
                "environment, for example /tmp/bgr_minatar_venv."
            ) from exc

        self.args = args
        self.rng = np.random.default_rng(seed + 777_000)
        self.env = Environment(
            args.game,
            sticky_action_prob=args.sticky_action_prob,
            difficulty_ramping=args.difficulty_ramping,
        )
        self.states = self._select_replay_states(args.replay_states, seed=seed)
        self.policy = LinearSoftmaxPolicy(
            rng=self.rng,
            learning_rate=args.learning_rate,
            init_noise=args.policy_init_noise,
            feature_dim=self.feature_dim,
        )
        self._pretrain()

    @property
    def feature_dim(self) -> int:
        return 10

    def close(self) -> None:
        self.env.close_display()

    def _select_replay_states(self, count: int, *, seed: int) -> list[BreakoutSnapshot]:
        states: list[BreakoutSnapshot] = []
        seen: set[tuple[int, int, int, int, int]] = set()
        offset = int(seed) * 1_000
        candidate_seed = 0
        while len(states) < int(count) and candidate_seed < int(count) * 25:
            self.seed_and_reset(offset + candidate_seed)
            burn_in = int(self.args.burn_in) + (candidate_seed % max(1, int(self.args.burn_in_jitter) + 1))
            for _step in range(burn_in):
                _reward, terminal = self.env.act(controller_action(self.env))
                if terminal:
                    break
            snapshot = self.capture()
            key = (
                snapshot.ball_x,
                snapshot.ball_y,
                snapshot.ball_dir,
                snapshot.pos,
                int(np.count_nonzero(snapshot.brick_map)),
            )
            if not snapshot.terminal and key not in seen:
                states.append(snapshot)
                seen.add(key)
            candidate_seed += 1
        if not states:
            raise ValueError("no feasible MinAtar Breakout replay states selected")
        return states

    def seed_and_reset(self, seed: int) -> None:
        self.env.seed(int(seed))
        self.env.reset()
        self.env.last_action = 0

    def capture(self) -> BreakoutSnapshot:
        game = self.env.env
        return BreakoutSnapshot(
            ball_x=int(game.ball_x),
            ball_y=int(game.ball_y),
            ball_dir=int(game.ball_dir),
            pos=int(game.pos),
            brick_map=np.array(game.brick_map, copy=True),
            strike=bool(game.strike),
            last_x=int(game.last_x),
            last_y=int(game.last_y),
            terminal=bool(game.terminal),
        )

    def restore(self, snapshot: BreakoutSnapshot, *, sigma: float, sign: int) -> int:
        game = self.env.env
        game.ball_x = int(snapshot.ball_x)
        game.ball_y = int(snapshot.ball_y)
        game.ball_dir = int(snapshot.ball_dir)
        game.pos = int(np.clip(snapshot.pos + int(sign) * int(round(float(sigma))), 0, 9))
        game.brick_map = np.array(snapshot.brick_map, copy=True)
        game.strike = bool(snapshot.strike)
        game.last_x = int(snapshot.last_x)
        game.last_y = int(snapshot.last_y)
        game.terminal = False
        self.env.last_action = 0
        return int(game.pos)

    def _pretrain(self) -> None:
        for _ in range(int(self.args.policy_init_steps)):
            replay_idx = int(self.rng.integers(len(self.states)))
            sigma = float(self.rng.uniform(0.0, self.args.max_radius))
            sign = -1 if self.rng.random() < 0.5 else 1
            self.train_start(replay_idx, sigma, sign)

    def perturbation_signs(self, replay_idx: int, sigma: float) -> list[int]:
        snapshot = self.states[replay_idx]
        radius = int(round(float(np.clip(sigma, 0.0, self.args.max_radius))))
        if radius == 0:
            return [1]
        signs = []
        for sign in (-1, 1):
            pos = snapshot.pos + sign * radius
            if 0 <= pos <= 9:
                signs.append(sign)
        return signs or [1]

    def success_prob(self, replay_idx: int, sigma: float) -> float:
        signs = self.perturbation_signs(replay_idx, sigma)
        return float(np.mean([self.rollout(replay_idx, sigma, sign=sign, train=False) for sign in signs]))

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> None:
        signs = self.perturbation_signs(replay_idx, sigma)
        sign = int(signs[int(rng.integers(len(signs)))])
        self.train_start(replay_idx, sigma, sign)

    def train_start(self, replay_idx: int, sigma: float, sign: int) -> None:
        self.rollout(replay_idx, sigma, sign=sign, train=True)

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        signs = self.perturbation_signs(replay_idx, sigma)
        sign = int(signs[int(rng.integers(len(signs)))])
        self.restore(self.states[replay_idx], sigma=sigma, sign=sign)
        features = self.features()
        return self.policy.loss(features, teacher_action_index(self.env))

    def rollout(self, replay_idx: int, sigma: float, *, sign: int, train: bool) -> bool:
        self.restore(self.states[replay_idx], sigma=sigma, sign=sign)
        terminal = False
        step = 0
        for step in range(1, int(self.args.horizon) + 1):
            features = self.features()
            target = teacher_action_index(self.env)
            if train:
                self.policy.update(features, target)
            action = ACTIONS[self.policy.action(features)]
            _reward, terminal = self.env.act(action)
            if terminal:
                break
        return (not bool(self.env.env.terminal)) and step >= int(self.args.horizon)

    def features(self) -> np.ndarray:
        game = self.env.env
        dx = int(game.ball_x) - int(game.pos)
        dy_bottom = 9 - int(game.ball_y)
        brick_frac = float(np.count_nonzero(game.brick_map)) / 30.0
        return np.array(
            [
                int(game.ball_x) / 9.0,
                int(game.ball_y) / 9.0,
                int(game.pos) / 9.0,
                dx / 9.0,
                abs(dx) / 9.0,
                dy_bottom / 9.0,
                float(abs(dx) <= dy_bottom),
                int(game.ball_dir) / 3.0,
                brick_frac,
                1.0,
            ],
            dtype=float,
        )


def controller_action(env) -> int:
    game = env.env
    if int(game.ball_x) < int(game.pos):
        return 1
    if int(game.ball_x) > int(game.pos):
        return 3
    return 0


def teacher_action_index(env) -> int:
    action = controller_action(env)
    return ACTIONS.index(action)


def run_method(args: argparse.Namespace, method: str, seed: int) -> BreakoutProbeResult:
    rng = np.random.default_rng(seed + 779_000)
    bench = BreakoutRecoveryProbe(seed=seed, args=args)
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
                    records[replay_idx].add_observation(sigma, bench.success_prob(replay_idx, sigma))
                    records[replay_idx].replay_count += 1

        xs = np.array([row["step"] for row in history], dtype=float)
        ys = np.array([row["rauc"] for row in history], dtype=float)
        final = history[-1]
        return BreakoutProbeResult(
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


def init_records(bench: BreakoutRecoveryProbe, rng: np.random.Generator, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, snapshot in enumerate(bench.states):
        record = LevelRecord(
            id=f"minatar_breakout_{replay_idx}_{snapshot.ball_x}_{snapshot.ball_y}_{snapshot.ball_dir}",
            domain="minatar_breakout_recovery",
            task_id="MinAtar-Breakout",
            clean_success_hat=bench.success_prob(replay_idx, 0.0),
            feasibility_hat=1.0,
            perturbation_family="paddle_cell_offset",
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
    bench: BreakoutRecoveryProbe,
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


def evaluate(bench: BreakoutRecoveryProbe, args: argparse.Namespace) -> dict[str, float]:
    grid = np.linspace(0.0, args.max_radius, args.eval_grid_size)
    clean: list[float] = []
    raucs: list[float] = []
    radii: list[float] = []
    for replay_idx in range(len(bench.states)):
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
    bench: BreakoutRecoveryProbe,
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
        scores = [bench.loss_proxy(int(candidate), float(sigma), rng) for candidate, sigma in zip(candidates, sigmas, strict=True)]
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
    parser.add_argument("--out", default="runs/minatar_breakout_recovery_probe")
    parser.add_argument("--game", default="breakout")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=10)
    parser.add_argument("--replay-states", type=int, default=12)
    parser.add_argument("--burn-in", type=int, default=4)
    parser.add_argument("--burn-in-jitter", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--max-radius", type=float, default=5.0)
    parser.add_argument("--eval-grid-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    parser.add_argument("--policy-init-noise", type=float, default=0.10)
    parser.add_argument("--policy-init-steps", type=int, default=12)
    parser.add_argument("--sticky-action-prob", type=float, default=0.0)
    parser.add_argument("--difficulty-ramping", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--target-radius", type=float, default=1.0)
    parser.add_argument("--radius-bandwidth", type=float, default=1.0)
    parser.add_argument("--fixed-radius", type=float, default=1.0)
    parser.add_argument("--baseline-candidates", type=int, default=8)
    parser.add_argument("--initial-probes", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0, 5.0])
    parser.add_argument("--min-trials", type=int, default=1)
    parser.add_argument("--refresh-per-eval", type=int, default=8)
    parser.add_argument("--refresh-trials", type=int, default=1)
    parser.add_argument("--refresh-jitter", type=float, default=0.35)
    parser.add_argument("--radius-noise", type=float, default=0.35)
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
