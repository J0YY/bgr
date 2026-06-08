#!/usr/bin/env python
"""Scout BGR-style replay on small pre-existing OpenML tabular datasets.

This is an internal route-selection diagnostic, not paper evidence. It checks
whether a non-authored supervised dataset shows a large enough fixed-protocol
signal to justify a preregistered 30-seed follow-up. The default gate is
+0.03 mean RAUC over uniform and at least 3/4 paired wins with no losses.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


METHODS = ("uniform", "fixed", "bgr")
DEFAULT_DATASETS = ("ionosphere", "sonar", "diabetes", "spambase")
DEFAULT_TARGETS = (0.5, 1.0, 1.5, 2.0)
OPENML_DATASETS = {
    "ionosphere": {"name": "ionosphere", "version": 1},
    "sonar": {"name": "sonar", "version": 1},
    "diabetes": {"name": "diabetes", "version": 1},
    "spambase": {"name": "spambase", "version": 1},
}


@dataclass(frozen=True)
class TrialResult:
    dataset: str
    target_radius: float
    method: str
    seed: int
    final_rauc: float


def parse_csv(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def parse_float_csv(raw: str) -> list[float]:
    return [float(value) for value in parse_csv(raw)]


def perturb(x: np.ndarray, radius: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(size=x.shape)
    return x + noise / (float(np.linalg.norm(noise)) + 1e-12) * radius


def rauc_from_values(radii: np.ndarray, values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    return float(np.trapezoid(vals, radii) / (radii[-1] - radii[0]))


def load_openml_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:  # pragma: no cover - optional scout dependency.
        raise RuntimeError(
            "scikit-learn is required for this optional internal scout; "
            "install it in a temporary environment instead of adding it to "
            "submission runtime dependencies."
        ) from exc
    spec = OPENML_DATASETS[dataset]
    data = fetch_openml(
        name=spec["name"],
        version=spec["version"],
        as_frame=False,
        parser="auto",
    )
    x_all = np.asarray(data.data, dtype=float)
    x_all = SimpleImputer(strategy="median").fit_transform(x_all)
    y_all = LabelEncoder().fit_transform(np.asarray(data.target))
    return x_all, y_all


def evaluate_rauc(model, x_eval: np.ndarray, y_eval: np.ndarray, radii: np.ndarray, rng: np.random.Generator) -> float:
    values = []
    for radius in radii:
        values.append(
            float(
                np.mean(
                    [
                        model.predict([perturb(x, float(radius), rng)])[0] == label
                        for x, label in zip(x_eval, y_eval)
                    ]
                )
            )
        )
    return rauc_from_values(radii, values)


def run_trial(
    *,
    dataset: str,
    seed: int,
    method: str,
    target_radius: float,
    steps: int,
    batch_size: int,
    candidate_count: int,
    max_radius: float,
    eval_examples: int,
) -> TrialResult:
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    x_all, y_all = load_openml_dataset(dataset)
    classes = np.unique(y_all)
    x_train, x_eval, y_train, y_eval = train_test_split(
        x_all,
        y_all,
        test_size=0.35,
        random_state=seed,
        stratify=y_all,
    )
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_eval = scaler.transform(x_eval)

    rng = np.random.default_rng(1000 * len(dataset) + seed)
    initial_indices: list[int] = []
    per_class = 8 if len(classes) == 2 else 4
    for label in classes:
        label_indices = np.where(y_train == label)[0]
        initial_indices.extend(rng.choice(label_indices, min(per_class, len(label_indices)), replace=False))

    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        learning_rate="optimal",
        max_iter=1,
        tol=None,
        random_state=seed,
        warm_start=True,
    )
    initial = np.asarray(initial_indices, dtype=int)
    model.partial_fit(x_train[initial], y_train[initial], classes=classes)

    boundary_grid = np.linspace(0.0, max_radius, 7)
    for _ in range(steps):
        add_x: list[np.ndarray] = []
        add_y: list[int] = []
        if method == "uniform":
            for _ in range(batch_size):
                index = int(rng.integers(len(x_train)))
                radius = float(rng.uniform(0.0, max_radius))
                add_x.append(perturb(x_train[index], radius, rng))
                add_y.append(int(y_train[index]))
        elif method == "fixed":
            for _ in range(batch_size):
                index = int(rng.integers(len(x_train)))
                add_x.append(perturb(x_train[index], target_radius, rng))
                add_y.append(int(y_train[index]))
        elif method == "bgr":
            candidate_indices = rng.choice(len(x_train), min(candidate_count, len(x_train)), replace=False)
            scored: list[tuple[float, int, float]] = []
            for index in candidate_indices:
                successes = [
                    model.predict([perturb(x_train[index], float(radius), rng)])[0] == y_train[index]
                    for radius in boundary_grid
                ]
                critical_radius = 0.0
                for radius, success in zip(boundary_grid, successes):
                    if success:
                        critical_radius = float(radius)
                transition_penalty = abs(float(np.mean(successes)) - 0.5)
                score = -abs(critical_radius - target_radius) - 0.15 * transition_penalty
                scored.append((score, int(index), critical_radius))
            scored.sort(reverse=True)
            for _, index, critical_radius in scored[:batch_size]:
                radius = float(np.clip(rng.normal(critical_radius, 0.25), 0.0, max_radius))
                add_x.append(perturb(x_train[index], radius, rng))
                add_y.append(int(y_train[index]))
        else:
            raise ValueError(f"unknown method: {method}")
        model.partial_fit(np.asarray(add_x), np.asarray(add_y))

    eval_radii = np.linspace(0.0, max_radius, 9)
    return TrialResult(
        dataset=dataset,
        target_radius=target_radius,
        method=method,
        seed=seed,
        final_rauc=evaluate_rauc(model, x_eval[:eval_examples], y_eval[:eval_examples], eval_radii, rng),
    )


def paired_counts(left: list[float], right: list[float]) -> tuple[int, int, int]:
    wins = losses = ties = 0
    for a, b in zip(left, right):
        if a > b:
            wins += 1
        elif a < b:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def write_outputs(results: list[TrialResult], out_dir: Path, datasets: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "target_radius", "method", "seed", "final_rauc"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "dataset": row.dataset,
                    "target_radius": f"{row.target_radius:.4f}",
                    "method": row.method,
                    "seed": row.seed,
                    "final_rauc": f"{row.final_rauc:.6f}",
                }
            )

    by_key: dict[tuple[str, float, str], list[TrialResult]] = {}
    for row in results:
        by_key.setdefault((row.dataset, row.target_radius, row.method), []).append(row)

    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "target_radius",
            "method",
            "n",
            "final_rauc_mean",
            "delta_vs_uniform",
            "wins_vs_uniform",
            "losses_vs_uniform",
            "ties_vs_uniform",
            "decision",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        targets = sorted({row.target_radius for row in results})
        for dataset in sorted(datasets):
            for target_radius in targets:
                uniform = sorted(by_key[(dataset, target_radius, "uniform")], key=lambda row: row.seed)
                uniform_values = [row.final_rauc for row in uniform]
                for method in METHODS:
                    rows = sorted(by_key[(dataset, target_radius, method)], key=lambda row: row.seed)
                    values = [row.final_rauc for row in rows]
                    wins, losses, ties = paired_counts(values, uniform_values)
                    delta = float(np.mean(values) - np.mean(uniform_values))
                    decision = "reject-scout"
                    if method == "bgr" and delta >= 0.03 and wins >= 3 and losses == 0:
                        decision = "candidate-for-preregistration"
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "target_radius": f"{target_radius:.4f}",
                            "method": method,
                            "n": len(rows),
                            "final_rauc_mean": f"{float(np.mean(values)):.6f}",
                            "delta_vs_uniform": f"{delta:.6f}",
                            "wins_vs_uniform": wins,
                            "losses_vs_uniform": losses,
                            "ties_vs_uniform": ties,
                            "decision": decision,
                        }
                    )

    versions = {
        "datasets": {dataset: OPENML_DATASETS[dataset] for dataset in datasets},
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    try:
        import sklearn

        versions["scikit_learn"] = sklearn.__version__
    except ImportError:
        versions["scikit_learn"] = "unavailable"
    (out_dir / "package_versions.json").write_text(json.dumps(versions, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/openml_margin_scout_v0"))
    parser.add_argument("--datasets", type=parse_csv, default=list(DEFAULT_DATASETS))
    parser.add_argument("--targets", type=parse_float_csv, default=list(DEFAULT_TARGETS))
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-count", type=int, default=128)
    parser.add_argument("--max-radius", type=float, default=2.0)
    parser.add_argument("--eval-examples", type=int, default=250)
    args = parser.parse_args()

    unknown = sorted(set(args.datasets) - set(OPENML_DATASETS))
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(unknown)}")

    results = []
    for dataset in args.datasets:
        for target_radius in args.targets:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                for method in METHODS:
                    result = run_trial(
                        dataset=dataset,
                        seed=seed,
                        method=method,
                        target_radius=float(target_radius),
                        steps=args.steps,
                        batch_size=args.batch_size,
                        candidate_count=args.candidate_count,
                        max_radius=args.max_radius,
                        eval_examples=args.eval_examples,
                    )
                    results.append(result)
                    print(
                        f"[run] dataset={dataset} target={target_radius:.4f} "
                        f"seed={seed} method={method} final_rauc={result.final_rauc:.6f}",
                        flush=True,
                    )
    write_outputs(results, args.out, list(args.datasets))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
