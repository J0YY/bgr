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
    "banknote-authentication": {"name": "banknote-authentication", "version": 1},
    "blood-transfusion-service-center": {"name": "blood-transfusion-service-center", "version": 1},
    "climate-model-simulation-crashes": {"name": "climate-model-simulation-crashes", "version": 1},
    "heart-statlog": {"name": "heart-statlog", "version": 1},
    "qsar-biodeg": {"name": "qsar-biodeg", "version": 1},
    "mammography": {"name": "mammography", "version": 1},
    "breast-w": {"name": "breast-w", "version": 1},
    "haberman": {"name": "haberman", "version": 1},
    "MagicTelescope": {"name": "MagicTelescope", "version": 1},
    "eeg-eye-state": {"name": "eeg-eye-state", "version": 1},
    "ozone-level-8hr": {"name": "ozone-level-8hr", "version": 1},
    "Bioresponse": {"name": "Bioresponse", "version": 1},
    "steel-plates-fault": {"name": "steel-plates-fault", "version": 1},
    "optdigits": {"name": "optdigits", "version": 1},
    "pendigits": {"name": "pendigits", "version": 1},
    "satimage": {"name": "satimage", "version": 1},
    "segment": {"name": "segment", "version": 1},
    "letter": {"name": "letter", "version": 1},
    "vehicle": {"name": "vehicle", "version": 1},
    "texture": {"name": "texture", "version": 1},
    "mfeat-fourier": {"name": "mfeat-fourier", "version": 1},
    "mfeat-karhunen": {"name": "mfeat-karhunen", "version": 1},
    "mfeat-pixel": {"name": "mfeat-pixel", "version": 1},
    "kc1": {"name": "kc1", "version": 1},
    "kc2": {"name": "kc2", "version": 1},
    "pc2": {"name": "pc2", "version": 1},
    "pc3": {"name": "pc3", "version": 1},
    "pc4": {"name": "pc4", "version": 1},
    "mc1": {"name": "mc1", "version": 1},
    "jm1": {"name": "jm1", "version": 1},
    "hill-valley": {"name": "hill-valley", "version": 1},
    "madelon": {"name": "madelon", "version": 1},
    "gina_agnostic": {"name": "gina_agnostic", "version": 1},
    "electricity": {"name": "electricity", "version": 1},
    "mozilla4": {"name": "mozilla4", "version": 1},
    "pc1": {"name": "pc1", "version": 1},
    "phoneme": {"name": "phoneme", "version": 1},
    "wdbc": {"name": "wdbc", "version": 1},
    "credit-g": {"name": "credit-g", "version": 1},
    "kr-vs-kp": {"name": "kr-vs-kp", "version": 1},
    "tic-tac-toe": {"name": "tic-tac-toe", "version": 1},
    "mushroom": {"name": "mushroom", "version": 1},
    "bank-marketing": {"name": "bank-marketing", "version": 1},
    "adult": {"name": "adult", "version": 1},
    "PhishingWebsites": {"name": "PhishingWebsites", "version": 1},
    "credit-approval": {"name": "credit-approval", "version": 1},
}
EXTERNAL_VALIDATION_DATASETS = (
    "banknote-authentication",
    "blood-transfusion-service-center",
    "climate-model-simulation-crashes",
    "kc1",
    "mozilla4",
    "pc1",
    "phoneme",
    "wdbc",
)
BROAD_NUMERIC_DATASETS = (
    "heart-statlog",
    "qsar-biodeg",
    "mammography",
    "breast-w",
    "haberman",
    "MagicTelescope",
    "eeg-eye-state",
    "ozone-level-8hr",
    "Bioresponse",
    "steel-plates-fault",
)
MULTICLASS_NUMERIC_DATASETS = (
    "optdigits",
    "pendigits",
    "satimage",
    "segment",
    "letter",
    "vehicle",
    "texture",
    "mfeat-fourier",
    "mfeat-karhunen",
    "mfeat-pixel",
)
SECONDARY_NUMERIC_DATASETS = (
    "kc2",
    "pc2",
    "pc3",
    "pc4",
    "mc1",
    "jm1",
    "hill-valley",
    "madelon",
    "gina_agnostic",
    "electricity",
)
MIXED_BINARY_DATASETS = (
    "credit-g",
    "kr-vs-kp",
    "tic-tac-toe",
    "mushroom",
    "bank-marketing",
    "adult",
    "PhishingWebsites",
    "credit-approval",
)


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


def read_existing_results(path: Path) -> list[TrialResult]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            TrialResult(
                dataset=row["dataset"],
                target_radius=float(row["target_radius"]),
                method=row["method"],
                seed=int(row["seed"]),
                final_rauc=float(row["final_rauc"]),
            )
            for row in csv.DictReader(handle)
        ]


def perturb(x: np.ndarray, radius: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(size=x.shape)
    return x + noise / (float(np.linalg.norm(noise)) + 1e-12) * radius


def rauc_from_values(radii: np.ndarray, values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    return float(np.trapezoid(vals, radii) / (radii[-1] - radii[0]))


def load_openml_dataset(
    dataset: str,
    cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if cache is not None and dataset in cache:
        return cache[dataset]
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
    if cache is not None:
        cache[dataset] = (x_all, y_all)
    return x_all, y_all


def load_openml_mixed_dataset(
    dataset: str,
    cache: dict[str, tuple[object, np.ndarray]] | None = None,
) -> tuple[object, np.ndarray]:
    if cache is not None and dataset in cache:
        return cache[dataset]
    try:
        from sklearn.datasets import fetch_openml
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
        as_frame=True,
        parser="auto",
    )
    x_all = data.data
    y_all = LabelEncoder().fit_transform(np.asarray(data.target))
    if cache is not None:
        cache[dataset] = (x_all, y_all)
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
    preprocessing: str,
    dataset_cache: dict[str, tuple[object, np.ndarray]] | None = None,
) -> TrialResult:
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import train_test_split

    if preprocessing == "numeric":
        from sklearn.preprocessing import StandardScaler

        x_all, y_all = load_openml_dataset(dataset, dataset_cache)  # type: ignore[arg-type]
    elif preprocessing == "mixed":
        x_all, y_all = load_openml_mixed_dataset(dataset, dataset_cache)
    else:
        raise ValueError(f"unknown preprocessing: {preprocessing}")
    classes = np.unique(y_all)
    x_train, x_eval, y_train, y_eval = train_test_split(
        x_all,
        y_all,
        test_size=0.35,
        random_state=seed,
        stratify=y_all,
    )
    if preprocessing == "numeric":
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_eval = scaler.transform(x_eval)
    else:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import StandardScaler

        numeric_columns = list(x_train.select_dtypes(include=["number", "bool"]).columns)
        categorical_columns = [column for column in x_train.columns if column not in numeric_columns]
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                    numeric_columns,
                ),
                (
                    "cat",
                    make_pipeline(
                        SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    categorical_columns,
                ),
            ],
            sparse_threshold=0.0,
        )
        x_train = np.asarray(preprocessor.fit_transform(x_train), dtype=float)
        x_eval = np.asarray(preprocessor.transform(x_eval), dtype=float)

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
    results = sorted(results, key=lambda row: (row.dataset, row.target_radius, row.seed, row.method))
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
                uniform_rows = by_key.get((dataset, target_radius, "uniform"), [])
                if not uniform_rows:
                    continue
                uniform_by_seed = {row.seed: row.final_rauc for row in uniform_rows}
                for method in METHODS:
                    rows = by_key.get((dataset, target_radius, method), [])
                    if not rows:
                        continue
                    values_by_seed = {row.seed: row.final_rauc for row in rows}
                    paired_seeds = sorted(set(values_by_seed) & set(uniform_by_seed))
                    if not paired_seeds:
                        continue
                    values = [values_by_seed[seed] for seed in paired_seeds]
                    uniform_values = [uniform_by_seed[seed] for seed in paired_seeds]
                    wins, losses, ties = paired_counts(values, uniform_values)
                    delta = float(np.mean(values) - np.mean(uniform_values))
                    decision = "reject-scout"
                    if method == "bgr" and len(values) >= 4 and delta >= 0.03 and wins >= 3 and losses == 0:
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
    parser.add_argument(
        "--external-validation-suite",
        action="store_true",
        help="Use the fixed numeric OpenML external-validation suite instead of DEFAULT_DATASETS.",
    )
    parser.add_argument(
        "--broad-numeric-suite",
        action="store_true",
        help="Use a fixed broader suite of binary numeric OpenML datasets that pass the existing numeric pipeline.",
    )
    parser.add_argument(
        "--multiclass-numeric-suite",
        action="store_true",
        help="Use a fixed broader suite of multiclass numeric OpenML datasets that pass the existing numeric pipeline.",
    )
    parser.add_argument(
        "--secondary-numeric-suite",
        action="store_true",
        help="Use a second fixed suite of binary numeric OpenML datasets that pass the existing numeric pipeline.",
    )
    parser.add_argument(
        "--mixed-binary-suite",
        action="store_true",
        help="Use a fixed suite of binary OpenML datasets with mixed numeric/categorical preprocessing.",
    )
    parser.add_argument(
        "--preprocessing",
        choices=("numeric", "mixed"),
        default="numeric",
        help="Feature preprocessing pipeline. Fixed suite flags set this automatically when needed.",
    )
    parser.add_argument("--targets", type=parse_float_csv, default=list(DEFAULT_TARGETS))
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-count", type=int, default=128)
    parser.add_argument("--max-radius", type=float, default=2.0)
    parser.add_argument("--eval-examples", type=int, default=250)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Write partial outputs after this many newly completed trial rows; 0 disables checkpointing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load an existing per_seed.csv from --out and skip completed dataset/target/seed/method rows.",
    )
    args = parser.parse_args()
    selected_suites = [
        args.external_validation_suite,
        args.broad_numeric_suite,
        args.multiclass_numeric_suite,
        args.secondary_numeric_suite,
        args.mixed_binary_suite,
    ]
    if sum(bool(selected) for selected in selected_suites) > 1:
        raise ValueError("choose at most one fixed suite")
    if args.external_validation_suite:
        args.datasets = list(EXTERNAL_VALIDATION_DATASETS)
    if args.broad_numeric_suite:
        args.datasets = list(BROAD_NUMERIC_DATASETS)
    if args.multiclass_numeric_suite:
        args.datasets = list(MULTICLASS_NUMERIC_DATASETS)
    if args.secondary_numeric_suite:
        args.datasets = list(SECONDARY_NUMERIC_DATASETS)
    if args.mixed_binary_suite:
        args.datasets = list(MIXED_BINARY_DATASETS)
        args.preprocessing = "mixed"

    unknown = sorted(set(args.datasets) - set(OPENML_DATASETS))
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(unknown)}")

    results = read_existing_results(args.out / "per_seed.csv") if args.resume else []
    completed = {
        (row.dataset, row.target_radius, row.seed, row.method)
        for row in results
    }
    if results:
        print(f"[resume] loaded {len(results)} existing trial rows from {args.out / 'per_seed.csv'}", flush=True)
    dataset_cache: dict[str, tuple[object, np.ndarray]] = {}
    completed_since_checkpoint = 0
    for dataset in args.datasets:
        for target_radius in args.targets:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                for method in METHODS:
                    key = (dataset, float(target_radius), seed, method)
                    if key in completed:
                        print(
                            f"[skip] dataset={dataset} target={target_radius:.4f} "
                            f"seed={seed} method={method}",
                            flush=True,
                        )
                        continue
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
                        preprocessing=args.preprocessing,
                        dataset_cache=dataset_cache,
                    )
                    results.append(result)
                    completed.add(key)
                    completed_since_checkpoint += 1
                    print(
                        f"[run] dataset={dataset} target={target_radius:.4f} "
                        f"seed={seed} method={method} final_rauc={result.final_rauc:.6f}",
                        flush=True,
                    )
                    if args.checkpoint_every > 0 and completed_since_checkpoint >= args.checkpoint_every:
                        write_outputs(results, args.out, list(args.datasets))
                        print(f"[checkpoint] wrote {len(results)} trial rows to {args.out}", flush=True)
                        completed_since_checkpoint = 0
    write_outputs(results, args.out, list(args.datasets))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
