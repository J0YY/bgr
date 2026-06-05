#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_IMAGE_SHAPE = (224, 224, 3)
EXPECTED_STATE_DIM = 8
EXPECTED_ACTION_DIM = 7


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and optionally pack rendered OpenVLA-OFT examples into a LIBERO-style HDF5 smoke dataset."
    )
    parser.add_argument("--examples", required=True, help="Path to examples.jsonl from render_openvla_teacher_examples.py.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--write-hdf5", action="store_true")
    args = parser.parse_args()

    examples_path = Path(args.examples)
    root = examples_path.parent
    rows = _load_examples(examples_path)
    records = [_load_record(root, row) for row in rows]
    summary = _summary(records)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(record["metadata"], sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    if bool(args.write_hdf5):
        _write_hdf5(out_dir / "libero_oft_smoke.hdf5", records)
    return 0


def _load_examples(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise SystemExit("No examples found.")
    return rows


def _load_record(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    array_path = root / str(row["array"])
    payload = np.load(array_path)
    image = np.asarray(payload["image"])
    wrist_image = np.asarray(payload["wrist_image"])
    state = np.asarray(payload["state"], dtype=np.float32).reshape(-1)
    action = np.asarray(payload["action"], dtype=np.float32).reshape(-1)
    language = str(payload["language_instruction"])
    errors = []
    if image.shape != EXPECTED_IMAGE_SHAPE:
        errors.append(f"image shape {image.shape} != {EXPECTED_IMAGE_SHAPE}")
    if wrist_image.shape != EXPECTED_IMAGE_SHAPE:
        errors.append(f"wrist image shape {wrist_image.shape} != {EXPECTED_IMAGE_SHAPE}")
    if state.shape != (EXPECTED_STATE_DIM,):
        errors.append(f"state shape {state.shape} != ({EXPECTED_STATE_DIM},)")
    if action.shape != (EXPECTED_ACTION_DIM,):
        errors.append(f"action shape {action.shape} != ({EXPECTED_ACTION_DIM},)")
    if image.dtype != np.uint8:
        errors.append(f"image dtype {image.dtype} != uint8")
    if wrist_image.dtype != np.uint8:
        errors.append(f"wrist image dtype {wrist_image.dtype} != uint8")
    if not language:
        errors.append("empty language instruction")
    if errors:
        raise ValueError(f"{array_path}: {'; '.join(errors)}")
    metadata = {
        "array": str(row["array"]),
        "suite": str(row["suite"]),
        "task_idx": int(row["task_idx"]),
        "task_name": str(row["task_name"]),
        "episode_idx": int(row.get("episode_idx", 0)),
        "init_state_idx": int(row.get("init_state_idx", 0)),
        "step_idx": int(row.get("step_idx", 0)),
        "candidate_name": str(row["candidate_name"]),
        "perturbation_type": str(row["perturbation_type"]),
        "instruction": str(row["instruction"]),
        "method": str(row.get("method", "")),
        "run": str(row.get("run", "")),
        "mix_source": str(row.get("mix_source", "")),
        "episode_uid": str(row.get("episode_uid", "")),
    }
    return {
        "image": image,
        "wrist_image": wrist_image,
        "state": state,
        "action": action,
        "language": language,
        "metadata": metadata,
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({record["metadata"]["perturbation_type"] for record in records})
    tasks = sorted({record["metadata"]["task_name"] for record in records})
    return {
        "examples": len(records),
        "families": families,
        "tasks": tasks,
        "image_shape": list(EXPECTED_IMAGE_SHAPE),
        "state_dim": EXPECTED_STATE_DIM,
        "action_dim": EXPECTED_ACTION_DIM,
        "hdf5_layout": {
            "actions": [EXPECTED_ACTION_DIM],
            "obs/agentview_rgb": list(EXPECTED_IMAGE_SHAPE),
            "obs/eye_in_hand_rgb": list(EXPECTED_IMAGE_SHAPE),
            "obs/ee_states": [6],
            "obs/gripper_states": [2],
        },
    }


def _write_hdf5(path: Path, records: list[dict[str, Any]]) -> None:
    import h5py

    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[_record_group_key(record)].append(record)
        for demo_idx, (_key, items) in enumerate(sorted(grouped.items())):
            items = sorted(items, key=lambda item: int(item["metadata"].get("step_idx", 0)))
            demo = data.create_group(f"demo_{demo_idx}")
            obs = demo.create_group("obs")
            actions = np.stack([item["action"] for item in items], axis=0).astype(np.float32)
            states = np.stack([item["state"] for item in items], axis=0).astype(np.float32)
            obs.create_dataset("agentview_rgb", data=np.stack([item["image"] for item in items], axis=0))
            obs.create_dataset("eye_in_hand_rgb", data=np.stack([item["wrist_image"] for item in items], axis=0))
            obs.create_dataset("ee_states", data=states[:, :6])
            obs.create_dataset("gripper_states", data=states[:, -2:])
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("states", data=states)
            demo.create_dataset("robot_states", data=states)
            rewards = np.zeros((len(items),), dtype=np.uint8)
            dones = np.zeros((len(items),), dtype=np.uint8)
            rewards[-1] = 1
            dones[-1] = 1
            demo.create_dataset("rewards", data=rewards)
            demo.create_dataset("dones", data=dones)
            demo.attrs["language_instruction"] = items[0]["language"]
            demo.attrs["task_name"] = items[0]["metadata"]["task_name"]
            demo.attrs["candidate_name"] = items[0]["metadata"]["candidate_name"]
            demo.attrs["episode_idx"] = int(items[0]["metadata"].get("episode_idx", 0))
            demo.attrs["init_state_idx"] = int(items[0]["metadata"].get("init_state_idx", 0))


def _record_group_key(record: dict[str, Any]) -> str:
    metadata = record["metadata"]
    return "::".join(
        [
            str(metadata.get("suite", "")),
            str(metadata.get("task_idx", "")),
            str(metadata.get("task_name", "")),
            str(metadata.get("episode_idx", 0)),
            str(metadata.get("init_state_idx", 0)),
            str(metadata.get("candidate_name", "")),
            str(metadata.get("mix_source", "")),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
