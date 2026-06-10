#!/usr/bin/env python3
"""Run a Gymnasium Blackjack-v1 recovery replay scout.

This is an independent standard-environment scout with an exact tabular reset
interface. Replay states are public Blackjack observations. Perturbation radius
makes the public state harder by lowering the player sum and increasing the
dealer upcard toward ten. Methods differ only in replay state/radius selection.
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


ACTION_STICK = 0
ACTION_HIT = 1
PLAYER_SUMS = tuple(range(4, 22))
DEALER_UPCARDS = tuple(range(1, 11))
USABLE_FLAGS = (0, 1)
CARD_VALUES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10], dtype=int)


@dataclass(frozen=True, slots=True)
class BlackjackState:
    player_sum: int
    dealer_upcard: int
    usable_ace: int


@dataclass(frozen=True, slots=True)
class BlackjackProbeResult:
    method: str
    seed: int
    final_clean: float
    final_rauc: float
    final_median_r80: float
    rauc_aulc: float
    best_rauc: float
    history: list[dict[str, float]]


def draw_card(rng: np.random.Generator) -> int:
    return int(rng.choice(CARD_VALUES))


def add_card(player_sum: int, usable_ace: int, card: int) -> tuple[int, int]:
    total = int(player_sum)
    usable = int(usable_ace)
    if card == 1:
        if total + 11 <= 21:
            total += 11
            usable = 1
        else:
            total += 1
    else:
        total += int(card)
    if total > 21 and usable:
        total -= 10
        usable = 0
    return total, usable


def dealer_score(upcard: int, rng: np.random.Generator) -> int:
    total, usable = add_card(0, 0, int(upcard))
    total, usable = add_card(total, usable, draw_card(rng))
    while total < 17:
        total, usable = add_card(total, usable, draw_card(rng))
    return 0 if total > 21 else total


def package_versions() -> dict[str, str]:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit(
            "Blackjack recovery probe requires Gymnasium in an isolated environment, "
            "for example /tmp/bgr_lunar_venv."
        ) from exc
    return {"gymnasium": gym.__version__, "gymnasium-package": version("gymnasium"), "numpy": np.__version__}


def state_key(state: BlackjackState) -> tuple[int, int, int]:
    return (int(state.player_sum), int(state.dealer_upcard), int(state.usable_ace))


class BlackjackRecoveryProbe:
    def __init__(
        self,
        *,
        seed: int,
        replay_state_count: int,
        max_radius: int,
        learning_rate: float,
        discount: float,
        epsilon: float,
        q_init_noise: float,
        eval_trials: int,
        train_horizon: int,
    ) -> None:
        self.rng = np.random.default_rng(seed + 827_000)
        self.max_radius = int(max_radius)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.epsilon = float(epsilon)
        self.eval_trials = int(eval_trials)
        self.train_horizon = int(train_horizon)
        self.q = self.rng.normal(0.0, float(q_init_noise), size=(22, 11, 2, 2))
        self._warm_start_heuristic()
        self.states = self._select_replay_states(int(replay_state_count))

    def _warm_start_heuristic(self) -> None:
        for player_sum in PLAYER_SUMS:
            for dealer in DEALER_UPCARDS:
                for usable in USABLE_FLAGS:
                    stick_bias = 0.08 if player_sum >= 18 else -0.08
                    if usable and player_sum >= 18:
                        stick_bias += 0.04
                    if dealer >= 9 and player_sum < 20:
                        stick_bias -= 0.04
                    self.q[player_sum, dealer, usable, ACTION_STICK] += stick_bias
                    self.q[player_sum, dealer, usable, ACTION_HIT] -= stick_bias

    def _select_replay_states(self, count: int) -> list[BlackjackState]:
        candidates: list[tuple[int, int, BlackjackState]] = []
        for player_sum in range(17, 22):
            for dealer in DEALER_UPCARDS:
                for usable in USABLE_FLAGS:
                    if usable and player_sum < 12:
                        continue
                    difficulty = abs(player_sum - 19) + max(0, dealer - 6)
                    candidates.append((difficulty, dealer, BlackjackState(player_sum, dealer, usable)))
        candidates.sort(key=lambda item: (item[0], item[1], item[2].player_sum, item[2].usable_ace))
        if count >= len(candidates):
            return [item[2] for item in candidates]
        indexes = np.linspace(0, len(candidates) - 1, count, dtype=int)
        return [candidates[int(index)][2] for index in indexes]

    def perturbation_state(self, replay_idx: int, sigma: float) -> BlackjackState:
        base = self.states[replay_idx]
        radius = int(round(float(np.clip(sigma, 0.0, 1.0)) * self.max_radius))
        player_sum = max(12, base.player_sum - radius)
        dealer = min(10, base.dealer_upcard + max(0, radius - 1))
        usable = int(base.usable_ace and player_sum >= 12)
        return BlackjackState(player_sum, dealer, usable)

    def success_prob(self, replay_idx: int, sigma: float, rng: np.random.Generator | None = None) -> float:
        local_rng = rng if rng is not None else self.rng
        state = self.perturbation_state(replay_idx, sigma)
        wins = [self.rollout_state(state, local_rng, train=False, epsilon=False) > 0.0 for _ in range(self.eval_trials)]
        return float(np.mean(wins))

    def rollout(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> bool:
        state = self.perturbation_state(replay_idx, sigma)
        return self.rollout_state(state, rng, train=False, epsilon=True) > 0.0

    def train_step(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> None:
        state = self.perturbation_state(replay_idx, sigma)
        self.rollout_state(state, rng, train=True, epsilon=True)

    def loss_proxy(self, replay_idx: int, sigma: float, rng: np.random.Generator) -> float:
        state = self.perturbation_state(replay_idx, sigma)
        key = state_key(state)
        action = self.action(state, rng, epsilon=True)
        reward, next_state, done = self.transition(state, action, rng)
        target = reward if done else reward + self.discount * float(np.max(self.q[state_key(next_state)]))
        return abs(target - float(self.q[key][action]))

    def rollout_state(self, start: BlackjackState, rng: np.random.Generator, *, train: bool, epsilon: bool) -> float:
        state = start
        trajectory: list[tuple[BlackjackState, int, float, BlackjackState | None, bool]] = []
        reward = 0.0
        done = False
        for _ in range(self.train_horizon):
            action = self.action(state, rng, epsilon=epsilon)
            reward, next_state, done = self.transition(state, action, rng)
            trajectory.append((state, action, reward, next_state, done))
            if done:
                break
            state = next_state
        if train:
            for item_state, action, item_reward, next_state, item_done in reversed(trajectory):
                key = state_key(item_state)
                target = item_reward if item_done else item_reward + self.discount * float(np.max(self.q[state_key(next_state)]))
                self.q[key][action] += self.learning_rate * (target - float(self.q[key][action]))
        return float(reward if done else 0.0)

    def action(self, state: BlackjackState, rng: np.random.Generator, *, epsilon: bool) -> int:
        if epsilon and rng.random() < self.epsilon:
            return int(rng.integers(2))
        return int(np.argmax(self.q[state_key(state)]))

    def transition(
        self, state: BlackjackState, action: int, rng: np.random.Generator
    ) -> tuple[float, BlackjackState | None, bool]:
        if action == ACTION_STICK:
            dealer = dealer_score(state.dealer_upcard, rng)
            if dealer == 0 or state.player_sum > dealer:
                return 1.0, None, True
            if state.player_sum == dealer:
                return 0.0, None, True
            return -1.0, None, True
        player_sum, usable = add_card(state.player_sum, state.usable_ace, draw_card(rng))
        if player_sum > 21:
            return -1.0, None, True
        return 0.0, BlackjackState(player_sum, state.dealer_upcard, usable), False


def run_method(args: argparse.Namespace, method: str, seed: int) -> BlackjackProbeResult:
    rng = np.random.default_rng(seed + 829_000)
    bench = BlackjackRecoveryProbe(
        seed=seed,
        replay_state_count=args.replay_states,
        max_radius=args.max_radius,
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        q_init_noise=args.q_init_noise,
        eval_trials=args.eval_trials,
        train_horizon=args.train_horizon,
    )
    records = init_records(bench, rng, args)
    scorer = BGRPriorityScorer(
        clean_threshold=0.0,
        feasibility_threshold=1.0,
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
            replay_idx, sigma = sample_training_pair(method, bench, records, scorer, rng, args, step)
            bench.train_step(replay_idx, sigma, rng)
            if method.startswith("bgr"):
                records[replay_idx].add_observation(sigma, bench.rollout(replay_idx, sigma, rng))
                records[replay_idx].replay_count += 1

    xs = np.array([row["step"] for row in history], dtype=float)
    ys = np.array([row["rauc"] for row in history], dtype=float)
    final = history[-1]
    return BlackjackProbeResult(
        method=method,
        seed=seed,
        final_clean=final["clean"],
        final_rauc=final["rauc"],
        final_median_r80=final["median_r80"],
        rauc_aulc=float(np.trapezoid(ys, xs) / (xs[-1] - xs[0])) if xs[-1] > xs[0] else final["rauc"],
        best_rauc=max(row["rauc"] for row in history),
        history=history,
    )


def init_records(bench: BlackjackRecoveryProbe, rng: np.random.Generator, args: argparse.Namespace) -> list[LevelRecord]:
    records: list[LevelRecord] = []
    for replay_idx, replay in enumerate(bench.states):
        record = LevelRecord(
            id=f"blackjack_{replay.player_sum}_{replay.dealer_upcard}_{replay.usable_ace}",
            domain="gym_blackjack_v1_recovery",
            task_id="Blackjack-v1",
            clean_success_hat=bench.success_prob(replay_idx, 0.0, rng),
            feasibility_hat=1.0,
            perturbation_family="adverse_player_sum_dealer_upcard",
            sigma_grid=list(args.initial_probes),
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
    bench: BlackjackRecoveryProbe,
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
        record.clean_success_hat = bench.success_prob(int(replay_idx), 0.0, rng)
        record.last_evaluated_step = step


def write_estimate(record: LevelRecord, estimate) -> None:
    record.r_alpha_hat = estimate.r_alpha
    record.sharpness_hat = estimate.sharpness
    record.uncertainty_hat = estimate.r_uncertainty
    record.recovery_curve_hat = estimate.recovery.tolist()


def evaluate(bench: BlackjackRecoveryProbe, rng: np.random.Generator, args: argparse.Namespace) -> dict[str, float]:
    grid = np.linspace(0.0, 1.0, args.eval_grid_size)
    clean: list[float] = []
    raucs: list[float] = []
    radii: list[float] = []
    for replay_idx in range(len(bench.states)):
        curve = np.array([bench.success_prob(replay_idx, sigma, rng) for sigma in grid], dtype=float)
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
    bench: BlackjackRecoveryProbe,
    records: list[LevelRecord],
    scorer: BGRPriorityScorer,
    rng: np.random.Generator,
    args: argparse.Namespace,
    step: int,
) -> tuple[int, float]:
    if method == "uniform":
        return int(rng.integers(len(records))), float(rng.uniform(0.0, 1.0))
    if method == "fixed":
        return int(rng.integers(len(records))), float(args.fixed_radius)
    if method == "failure_only":
        candidates = rng.choice(len(records), size=min(args.baseline_candidates, len(records)), replace=False)
        sigmas = rng.uniform(0.0, 1.0, size=len(candidates))
        scores = [1.0 - bench.success_prob(int(candidate), float(sigma), rng) for candidate, sigma in zip(candidates, sigmas, strict=True)]
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
    parser.add_argument("--out", default="results/blackjack_recovery_probe_4seed_v1")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--methods", default="uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr")
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--eval-every", type=int, default=60)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--replay-states", type=int, default=48)
    parser.add_argument("--max-radius", type=int, default=7)
    parser.add_argument("--train-horizon", type=int, default=8)
    parser.add_argument("--eval-grid-size", type=int, default=8)
    parser.add_argument("--eval-trials", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--q-init-noise", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--target-radius", type=float, default=0.45)
    parser.add_argument("--radius-bandwidth", type=float, default=0.35)
    parser.add_argument("--fixed-radius", type=float, default=0.55)
    parser.add_argument("--baseline-candidates", type=int, default=12)
    parser.add_argument("--initial-probes", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--min-trials", type=int, default=2)
    parser.add_argument("--refresh-per-eval", type=int, default=18)
    parser.add_argument("--refresh-trials", type=int, default=2)
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
    history_rows: list[dict[str, float | int | str]] = []
    results: list[dict] = []
    for method in methods:
        for seed in seeds:
            start = time.perf_counter()
            result = run_method(args, method, seed)
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
    (out_dir / "package_versions.json").write_text(json.dumps(package_versions(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history_rows)
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
