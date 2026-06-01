#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from bgr.libero_probe import patch_torch_load_for_libero


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a small PNG/action smoke set from an OpenVLA teacher-replay manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Optional manifest method filter. Can be passed multiple times, e.g. --method bgr_boundary.",
    )
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--selection", choices=["first", "first_per_family", "balanced_episodes"], default="first")
    parser.add_argument(
        "--episodes-per-family",
        type=int,
        default=1,
        help="For --selection balanced_episodes, render this many replay episodes per perturbation family.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help="For --selection balanced_episodes, truncate each selected replay episode to this many steps.",
    )
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--env-image-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-preprocess", choices=["raw", "official_libero"], default="official_libero")
    parser.add_argument("--camera-key", default="agentview_image")
    args = parser.parse_args()

    rows = _load_rows(
        Path(args.manifest),
        int(args.max_examples),
        str(args.selection),
        methods=tuple(str(method) for method in args.method),
        episodes_per_family=int(args.episodes_per_family),
        max_steps_per_episode=args.max_steps_per_episode,
    )
    if not rows:
        raise SystemExit("No manifest rows found.")

    out_dir = Path(args.out)
    image_dir = out_dir / "images"
    wrist_dir = out_dir / "wrist_images"
    array_dir = out_dir / "arrays"
    image_dir.mkdir(parents=True, exist_ok=True)
    wrist_dir.mkdir(parents=True, exist_ok=True)
    array_dir.mkdir(parents=True, exist_ok=True)
    examples = _render_examples(
        rows=rows,
        image_dir=image_dir,
        wrist_dir=wrist_dir,
        array_dir=array_dir,
        num_steps_wait=int(args.num_steps_wait),
        env_image_size=int(args.env_image_size),
        image_size=int(args.image_size),
        image_preprocess=str(args.image_preprocess),
        camera_key=str(args.camera_key),
    )
    (out_dir / "examples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in examples),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "requested_examples": int(args.max_examples),
                "rendered_examples": len(examples),
                "manifest": str(args.manifest),
                "methods": [str(method) for method in args.method],
                "image_size": int(args.image_size),
                "note": "PNG/NPZ action smoke set for the teacher-replay to RLDS path.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _load_rows(
    path: Path,
    max_examples: int,
    selection: str = "first",
    *,
    methods: tuple[str, ...] = (),
    episodes_per_family: int = 1,
    max_steps_per_episode: int | None = None,
) -> list[dict[str, Any]]:
    all_rows = [row for row in _iter_rows(path) if _matches_filters(row, methods)]
    if selection == "balanced_episodes":
        return _select_balanced_episode_rows(
            all_rows,
            max_examples=max_examples,
            episodes_per_family=episodes_per_family,
            max_steps_per_episode=max_steps_per_episode,
        )

    rows: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for row in all_rows:
        if _keep_row(row, selection, seen_families):
            rows.append(row)
        if len(rows) >= max_examples:
            break
    return rows


def _matches_filters(row: dict[str, Any], methods: tuple[str, ...]) -> bool:
    if methods and str(row.get("method", "")) not in set(methods):
        return False
    return True


def _iter_rows(path: Path):
    opener = path.open
    with opener(encoding="utf-8") as handle:
        if path.suffix == ".csv":
            for row in csv.DictReader(handle):
                yield dict(row)
        else:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _keep_row(row: dict[str, Any], selection: str, seen_families: set[str]) -> bool:
    if selection == "first":
        return True
    family = str(row.get("perturbation_type", ""))
    if family in seen_families:
        return False
    seen_families.add(family)
    return True


def _select_balanced_episode_rows(
    rows: list[dict[str, Any]],
    *,
    max_examples: int,
    episodes_per_family: int,
    max_steps_per_episode: int | None,
) -> list[dict[str, Any]]:
    if episodes_per_family <= 0:
        raise ValueError("episodes_per_family must be positive.")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    order: list[tuple[Any, ...]] = []
    for row in rows:
        key = _render_episode_key(row)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for key in order:
        items = sorted(grouped[key], key=lambda item: int(item["step_idx"]))
        if not items:
            continue
        family = str(items[0].get("perturbation_type", ""))
        if counts.get(family, 0) >= episodes_per_family:
            continue
        if max_steps_per_episode is not None:
            items = items[: int(max_steps_per_episode)]
        if not items:
            continue
        if max_examples > 0 and len(selected) + len(items) > max_examples:
            remaining = max_examples - len(selected)
            if remaining <= 0:
                break
            items = items[:remaining]
        selected.extend(items)
        counts[family] = counts.get(family, 0) + 1
        if max_examples > 0 and len(selected) >= max_examples:
            break
    return selected


def _render_episode_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["suite"]),
        int(row["task_idx"]),
        str(row["task_name"]),
        int(row["episode_idx"]),
        int(row["init_state_idx"]),
        str(row["candidate_name"]),
    )


def _render_examples(
    *,
    rows: list[dict[str, Any]],
    image_dir: Path,
    wrist_dir: Path,
    array_dir: Path,
    num_steps_wait: int,
    env_image_size: int,
    image_size: int,
    image_preprocess: str,
    camera_key: str,
) -> list[dict[str, Any]]:
    patch_torch_load_for_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    examples: list[dict[str, Any]] = []
    env_cache: dict[tuple[str, int], Any] = {}
    suite_cache: dict[str, Any] = {}
    current_episode: tuple[Any, ...] | None = None
    env = None
    obs = None
    try:
        for index, row in enumerate(rows):
            episode_key = (
                str(row["suite"]),
                int(row["task_idx"]),
                str(row["task_name"]),
                int(row["episode_idx"]),
                int(row["init_state_idx"]),
                str(row["candidate_name"]),
            )
            if episode_key != current_episode:
                if env is not None:
                    env.close()
                env, obs = _reset_env_for_row(row, suite_cache, env_cache, env_image_size)
                for _ in range(num_steps_wait):
                    obs, _reward, done, _info = env.step(_dummy_action())
                    if done:
                        break
                current_episode = episode_key
            assert env is not None and obs is not None
            target_step = int(row["step_idx"])
            while len(examples) > 0 and examples[-1].get("episode_key") == str(episode_key) and int(examples[-1]["step_idx"]) + 1 < target_step:
                break
            base_image = _preprocess_image(np.asarray(obs[camera_key]), image_preprocess, image_size)
            wrist_image = _preprocess_image(np.asarray(obs["robot0_eye_in_hand_image"]), image_preprocess, image_size)
            perturbed = _apply_perturbation(base_image, str(row.get("perturbation_type", "")), _json_dict(row.get("perturbation_params", {})))
            image_path = image_dir / f"example_{index:05d}.png"
            wrist_path = wrist_dir / f"example_{index:05d}.png"
            array_path = array_dir / f"example_{index:05d}.npz"
            Image.fromarray(perturbed).save(image_path)
            Image.fromarray(wrist_image).save(wrist_path)
            action = _json_list(row["target_action"])
            state = _libero_oft_state(obs)
            np.savez_compressed(
                array_path,
                image=perturbed,
                wrist_image=wrist_image,
                state=state.astype(np.float32),
                action=np.asarray(action, dtype=np.float32),
                language_instruction=np.asarray(str(row["instruction"])),
            )
            examples.append(
                {
                    "image": str(image_path.relative_to(image_dir.parent)),
                    "wrist_image": str(wrist_path.relative_to(image_dir.parent)),
                    "array": str(array_path.relative_to(image_dir.parent)),
                    "target_action": action,
                    "state": [float(x) for x in state.reshape(-1)],
                    "instruction": str(row["instruction"]),
                    "suite": str(row["suite"]),
                    "task_idx": int(row["task_idx"]),
                    "task_name": str(row["task_name"]),
                    "init_state_idx": int(row["init_state_idx"]),
                    "episode_idx": int(row["episode_idx"]),
                    "step_idx": target_step,
                    "candidate_name": str(row["candidate_name"]),
                    "candidate_spec": str(row.get("candidate_spec", "")),
                    "perturbation_type": str(row.get("perturbation_type", "")),
                    "method": str(row.get("method", "")),
                    "run": str(row.get("run", "")),
                    "episode_uid": str(row.get("episode_uid", "")),
                    "episode_key": str(episode_key),
                }
            )
            obs, _reward, done, _info = env.step(np.asarray(action, dtype=np.float32))
            if done:
                current_episode = None
        return examples
    finally:
        if env is not None:
            env.close()


def _reset_env_for_row(row: dict[str, Any], suite_cache: dict[str, Any], env_cache: dict[tuple[str, int], Any], image_size: int):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite_name = _suite_name(str(row["suite"]))
    task_idx = int(row["task_idx"])
    suite = suite_cache.get(suite_name)
    if suite is None:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        suite_cache[suite_name] = suite
    task = suite.get_task(task_idx)
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=image_size,
        camera_widths=image_size,
    )
    env.seed(0)
    obs = env.reset()
    init_states = suite.get_task_init_states(task_idx)
    init_state = np.asarray(init_states[int(row["init_state_idx"])], dtype=float)
    obs = env.set_init_state(init_state)
    return env, obs


def _suite_name(suite: str) -> str:
    if suite.startswith("libero_"):
        return suite
    if suite in {"object", "goal", "spatial", "10", "90"}:
        return f"libero_{suite}"
    return suite


def _preprocess_image(image: np.ndarray, mode: str, resize_size: int) -> np.ndarray:
    processed = image[::-1, ::-1] if mode == "official_libero" else image
    pil = Image.fromarray(np.asarray(processed, dtype=np.uint8)).convert("RGB")
    if pil.size != (resize_size, resize_size):
        pil = pil.resize((resize_size, resize_size), resample=Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def _libero_oft_state(obs: dict[str, Any]) -> np.ndarray:
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)[:3]
    axis_angle = _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1))
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[:2]
    return np.concatenate([eef_pos, axis_angle.astype(np.float32), gripper], axis=0)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(quat[3]) / den)).astype(np.float32)


def _apply_perturbation(image: np.ndarray, perturbation_type: str, params: dict[str, Any]) -> np.ndarray:
    pil = Image.fromarray(image).convert("RGB")
    if perturbation_type == "blur":
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(params.get("radius", 0.0))))
    elif perturbation_type == "brightness":
        pil = ImageEnhance.Brightness(pil).enhance(float(params.get("factor", 1.0)))
    elif perturbation_type == "occlusion":
        arr = np.asarray(pil, dtype=np.uint8).copy()
        fraction = max(0.0, min(1.0, float(params.get("fraction", 0.0))))
        side = max(1, int(round(min(arr.shape[0], arr.shape[1]) * fraction)))
        y0 = (arr.shape[0] - side) // 2
        x0 = (arr.shape[1] - side) // 2
        arr[y0 : y0 + side, x0 : x0 + side, :] = 0
        return arr
    elif perturbation_type == "shift":
        arr = np.asarray(pil, dtype=np.uint8)
        dx = int(round(float(params.get("dx_fraction", 0.0)) * arr.shape[1]))
        dy = int(round(float(params.get("dy_fraction", 0.0)) * arr.shape[0]))
        shifted = np.zeros_like(arr)
        src_x0 = max(0, -dx)
        src_y0 = max(0, -dy)
        dst_x0 = max(0, dx)
        dst_y0 = max(0, dy)
        width = arr.shape[1] - abs(dx)
        height = arr.shape[0] - abs(dy)
        if width > 0 and height > 0:
            shifted[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = arr[src_y0 : src_y0 + height, src_x0 : src_x0 + width]
        return shifted
    return np.asarray(pil, dtype=np.uint8)


def _dummy_action() -> np.ndarray:
    return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(value or "{}")


def _json_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(x) for x in json.loads(value)]


if __name__ == "__main__":
    raise SystemExit(main())
