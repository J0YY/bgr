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
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--env-image-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-preprocess", choices=["raw", "official_libero"], default="official_libero")
    parser.add_argument("--camera-key", default="agentview_image")
    args = parser.parse_args()

    rows = _load_rows(Path(args.manifest), int(args.max_examples))
    if not rows:
        raise SystemExit("No manifest rows found.")

    out_dir = Path(args.out)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    examples = _render_examples(
        rows=rows,
        image_dir=image_dir,
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
                "image_size": int(args.image_size),
                "note": "PNG/action smoke set for the teacher-replay to RLDS path.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _load_rows(path: Path, max_examples: int) -> list[dict[str, Any]]:
    rows = []
    opener = path.open
    with opener(encoding="utf-8") as handle:
        if path.suffix == ".csv":
            for row in csv.DictReader(handle):
                rows.append(dict(row))
                if len(rows) >= max_examples:
                    break
        else:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) >= max_examples:
                    break
    return rows


def _render_examples(
    *,
    rows: list[dict[str, Any]],
    image_dir: Path,
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
            perturbed = _apply_perturbation(base_image, str(row.get("perturbation_type", "")), _json_dict(row.get("perturbation_params", {})))
            image_path = image_dir / f"example_{index:05d}.png"
            Image.fromarray(perturbed).save(image_path)
            action = _json_list(row["target_action"])
            examples.append(
                {
                    "image": str(image_path.relative_to(image_dir.parent)),
                    "target_action": action,
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
