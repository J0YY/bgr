#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


EPISODE_KEYS = ("suite", "task_idx", "task_name", "episode_idx", "init_state_idx")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export step-level teacher actions for replaying OpenVLA boundary candidates. "
            "The output is an intermediate manifest for rendering/RLDS conversion, not RLDS itself."
        )
    )
    parser.add_argument("--bgr-dir", action="append", default=[], help="Directory with top_k_candidates.json and validate/observation_steps.jsonl.")
    parser.add_argument("--random-dir", action="append", default=[], help="Random-balanced artifact directory with the same layout.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lower", type=float, default=0.25)
    parser.add_argument("--upper", type=float, default=0.75)
    parser.add_argument("--max-steps-per-episode", type=int, default=64)
    parser.add_argument("--boundary-only", action="store_true", help="Export only candidates inside the observed boundary band.")
    parser.add_argument(
        "--include-native-anchors",
        action="store_true",
        help="Also export successful native OpenVLA replay episodes as identity clean-anchor candidates.",
    )
    parser.add_argument(
        "--native-anchors-only",
        action="store_true",
        help="Export only successful native OpenVLA replay episodes as identity clean-anchor candidates.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.bgr_dir:
        artifact = Path(path)
        if not bool(args.native_anchors_only):
            rows.extend(
                _export_dir(
                    artifact,
                    method="bgr_boundary",
                    lower=float(args.lower),
                    upper=float(args.upper),
                    max_steps_per_episode=int(args.max_steps_per_episode),
                    boundary_only=bool(args.boundary_only),
                )
            )
        if bool(args.include_native_anchors) or bool(args.native_anchors_only):
            rows.extend(_export_native_anchors(artifact, method="bgr_boundary", max_steps_per_episode=int(args.max_steps_per_episode)))
    for path in args.random_dir:
        artifact = Path(path)
        if not bool(args.native_anchors_only):
            rows.extend(
                _export_dir(
                    artifact,
                    method="random_balanced",
                    lower=float(args.lower),
                    upper=float(args.upper),
                    max_steps_per_episode=int(args.max_steps_per_episode),
                    boundary_only=bool(args.boundary_only),
                )
            )
        if bool(args.include_native_anchors) or bool(args.native_anchors_only):
            rows.extend(_export_native_anchors(artifact, method="random_balanced", max_steps_per_episode=int(args.max_steps_per_episode)))
    if not rows:
        raise SystemExit("No replay rows exported.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (str(row["method"]), str(row["run"]), str(row["candidate_name"]), str(row["episode_uid"]), int(row["step_idx"])))
    _write_jsonl(out_dir / "teacher_replay_manifest.jsonl", rows)
    _write_csv(out_dir / "teacher_replay_manifest.csv", rows)
    summary = _summarize(rows, lower=float(args.lower), upper=float(args.upper), max_steps_per_episode=int(args.max_steps_per_episode))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_readme(out_dir / "README.md")
    return 0


def _export_dir(
    path: Path,
    method: str,
    lower: float,
    upper: float,
    max_steps_per_episode: int,
    boundary_only: bool,
) -> list[dict[str, Any]]:
    candidates = json.loads((path / "top_k_candidates.json").read_text(encoding="utf-8"))
    candidates_by_name = {}
    for candidate in candidates:
        observed = float(candidate.get("observed_cf_rate", 0.0))
        if boundary_only and not (lower <= observed <= upper):
            continue
        candidates_by_name[str(candidate.get("name", ""))] = candidate
    if not candidates_by_name:
        return []

    steps_path = path / "validate" / "observation_steps.jsonl"
    native_steps = _load_successful_native_steps(steps_path, max_steps_per_episode=max_steps_per_episode)
    perturbed_episode_keys = _load_perturbed_episode_keys(steps_path, set(candidates_by_name))

    rows: list[dict[str, Any]] = []
    for candidate_name, candidate in sorted(candidates_by_name.items()):
        for episode_key in sorted(perturbed_episode_keys.get(candidate_name, set())):
            native = native_steps.get(episode_key)
            if not native:
                continue
            observed = float(candidate.get("observed_cf_rate", 0.0))
            in_band = lower <= observed <= upper
            for step in native:
                rows.append(
                    {
                        "method": method,
                        "run": path.name,
                        "source_artifact": str(path),
                        "candidate_name": candidate_name,
                        "candidate_spec": str(candidate.get("spec", "")),
                        "perturbation_type": str(candidate.get("perturbation_type", "")),
                        "perturbation_params": json.dumps(candidate.get("perturbation_params", {}), sort_keys=True),
                        "observed_cf_rate": observed,
                        "in_boundary_band": in_band,
                        "suite": episode_key[0],
                        "task_idx": int(episode_key[1]),
                        "task_name": episode_key[2],
                        "episode_idx": int(episode_key[3]),
                        "init_state_idx": int(episode_key[4]),
                        "episode_uid": _episode_uid(episode_key),
                        "step_idx": int(step["step_idx"]),
                        "instruction": str(step.get("policy_instruction") or step.get("instruction") or ""),
                        "target_action": json.dumps(step["executed_action"]),
                        "target_token_ids": json.dumps(step.get("token_ids", [])),
                        "source_label": "successful_native_openvla_action",
                    }
                )
    return rows


def _export_native_anchors(path: Path, method: str, max_steps_per_episode: int) -> list[dict[str, Any]]:
    native_steps = _load_successful_native_steps(path / "validate" / "observation_steps.jsonl", max_steps_per_episode=max_steps_per_episode)
    candidate_name = f"native_{path.name}"
    rows: list[dict[str, Any]] = []
    for episode_key, steps in sorted(native_steps.items()):
        for step in steps:
            rows.append(
                {
                    "method": method,
                    "run": path.name,
                    "source_artifact": str(path),
                    "candidate_name": candidate_name,
                    "candidate_spec": "native:identity",
                    "perturbation_type": "identity",
                    "perturbation_params": "{}",
                    "observed_cf_rate": 0.0,
                    "in_boundary_band": True,
                    "suite": episode_key[0],
                    "task_idx": int(episode_key[1]),
                    "task_name": episode_key[2],
                    "episode_idx": int(episode_key[3]),
                    "init_state_idx": int(episode_key[4]),
                    "episode_uid": _episode_uid(episode_key),
                    "step_idx": int(step["step_idx"]),
                    "instruction": str(step.get("policy_instruction") or step.get("instruction") or ""),
                    "target_action": json.dumps(step["executed_action"]),
                    "target_token_ids": json.dumps(step.get("token_ids", [])),
                    "source_label": "successful_native_openvla_clean_anchor",
                }
            )
    return rows


def _load_successful_native_steps(path: Path, max_steps_per_episode: int) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    successful_keys: set[tuple[Any, ...]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("perturbation_type") != "identity":
                continue
            if not row.get("executed_action"):
                continue
            key = _episode_key(row)
            if bool(row.get("success_after_step", False)):
                successful_keys.add(key)
            if len(grouped[key]) >= max_steps_per_episode:
                continue
            grouped[key].append(row)
    return {
        key: sorted(items, key=lambda row: int(row["step_idx"]))
        for key, items in grouped.items()
        if items and key in successful_keys
    }


def _load_perturbed_episode_keys(path: Path, candidate_names: set[str]) -> dict[str, set[tuple[Any, ...]]]:
    keys: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            name = str(row.get("perturbation_name", ""))
            if name in candidate_names:
                keys[name].add(_episode_key(row))
    return keys


def _episode_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[key] for key in EPISODE_KEYS)


def _episode_uid(key: tuple[Any, ...]) -> str:
    suite, task_idx, task_name, episode_idx, init_state_idx = key
    return f"{suite}:task={task_idx}:{task_name}:episode={episode_idx}:init={init_state_idx}"


def _summarize(rows: list[dict[str, Any]], lower: float, upper: float, max_steps_per_episode: int) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    methods = {}
    for method, items in sorted(by_method.items()):
        candidates = {str(row["candidate_name"]) for row in items}
        episodes = {str(row["episode_uid"]) for row in items}
        methods[method] = {
            "rows": len(items),
            "candidates": len(candidates),
            "episodes": len(episodes),
            "boundary_rows": sum(bool(row["in_boundary_band"]) for row in items),
            "clean_anchor_rows": sum(str(row.get("perturbation_type", "")) == "identity" for row in items),
            "mean_observed_cf_rate": mean(float(row["observed_cf_rate"]) for row in items),
            "families": sorted({str(row["perturbation_type"]) for row in items}),
        }
    return {
        "num_rows": len(rows),
        "boundary_band": [lower, upper],
        "max_steps_per_episode": max_steps_per_episode,
        "methods": methods,
        "note": (
            "Rows contain successful native OpenVLA actions paired with validated perturbation specs. "
            "A downstream converter must replay the native actions in LIBERO, render each step, apply the "
            "candidate observation perturbation, and write RLDS episodes for OpenVLA-OFT."
        ),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: Path) -> None:
    path.write_text(
        "OpenVLA teacher-replay manifest for BGR fine-tuning.\n\n"
        "Each row pairs a validated boundary candidate with a target action from a successful native OpenVLA rollout. "
        "The artifact intentionally stops before RLDS creation because the source logs contain actions and token IDs, "
        "but not saved images. A downstream RLDS converter must replay the native action prefix in LIBERO to render "
        "observations, apply the candidate perturbation to the rendered image stream, and write OpenVLA-OFT episodes.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
