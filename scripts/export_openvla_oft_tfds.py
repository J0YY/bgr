#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pack_openvla_oft_examples import _load_examples, _load_record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export rendered OpenVLA-OFT examples as a small RLDS-style TFDS dataset."
    )
    parser.add_argument("--examples", required=True, help="Path to examples.jsonl from render_openvla_teacher_examples.py.")
    parser.add_argument("--out", required=True, help="TFDS data_dir output root.")
    parser.add_argument("--dataset-name", default="bgr_libero_oft_smoke")
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()

    examples_path = Path(args.examples)
    root = examples_path.parent
    records = [_load_record(root, row) for row in _load_examples(examples_path)]
    episodes = list(_episode_records(records))

    import tensorflow_datasets as tfds

    builder = tfds.dataset_builders.AdhocBuilder(
        name=args.dataset_name,
        version=args.version,
        data_dir=args.out,
        features=_feature_spec(tfds),
        split_datasets={"train": episodes},
        description=(
            "BGR OpenVLA-OFT smoke dataset with LIBERO-style RGB, wrist RGB, "
            "state, action, and language fields."
        ),
        disable_shuffling=True,
    )
    builder.download_and_prepare()
    builder_dir = Path(args.out) / args.dataset_name / args.version
    summary = _summary(args.dataset_name, args.version, builder_dir, episodes)
    (builder_dir / "bgr_export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


def _episode_records(records: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = record["metadata"]
        key = f"{metadata['task_name']}::{metadata['candidate_name']}"
        grouped[key].append(record)

    episodes = []
    for episode_idx, (group_key, items) in enumerate(sorted(grouped.items())):
        steps = []
        for step_idx, item in enumerate(items):
            steps.append(_step_record(item, step_idx=step_idx, num_steps=len(items)))
        metadata = dict(items[0]["metadata"])
        metadata["num_steps"] = len(items)
        metadata["group_key"] = group_key
        episodes.append(
            (
                f"episode_{episode_idx:05d}",
                {
                    "steps": steps,
                    "episode_metadata": {
                        "file_path": metadata["array"],
                        "task_name": metadata["task_name"],
                        "candidate_name": metadata["candidate_name"],
                        "perturbation_type": metadata["perturbation_type"],
                    },
                },
            )
        )
    return episodes


def _step_record(record: dict[str, Any], *, step_idx: int, num_steps: int) -> dict[str, Any]:
    is_first = step_idx == 0
    is_last = step_idx == num_steps - 1
    return {
        "observation": {
            "image": record["image"],
            "wrist_image": record["wrist_image"],
            "state": record["state"].astype(np.float32),
        },
        "action": record["action"].astype(np.float32),
        "language_instruction": record["language"],
        "reward": np.float32(1.0 if is_last else 0.0),
        "discount": np.float32(1.0),
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_last,
    }


def _feature_spec(tfds: Any) -> Any:
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            "image": tfds.features.Image(shape=(224, 224, 3), dtype=np.uint8),
                            "wrist_image": tfds.features.Image(shape=(224, 224, 3), dtype=np.uint8),
                            "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                        }
                    ),
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "language_instruction": tfds.features.Text(),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                }
            ),
            "episode_metadata": tfds.features.FeaturesDict(
                {
                    "file_path": tfds.features.Text(),
                    "task_name": tfds.features.Text(),
                    "candidate_name": tfds.features.Text(),
                    "perturbation_type": tfds.features.Text(),
                }
            ),
        }
    )


def _summary(dataset_name: str, version: str, builder_dir: Path, episodes: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    step_count = sum(len(example["steps"]) for _key, example in episodes)
    perturbations = sorted(
        {example["episode_metadata"]["perturbation_type"] for _key, example in episodes}
    )
    return {
        "dataset_name": dataset_name,
        "version": version,
        "builder_dir": str(builder_dir),
        "episodes": len(episodes),
        "steps": step_count,
        "perturbation_types": perturbations,
        "load_hint": f"tfds.builder_from_directory({str(builder_dir)!r})",
    }


if __name__ == "__main__":
    raise SystemExit(main())
