#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from bgr.metrics import critical_radius, recovery_auc


SIGMA_NORMALIZERS = {
    "blur": 2.5,
    "brightness": 0.5,
    "occlusion": 0.5,
    "shift": 0.15,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert closed-loop LIBERO/OpenVLA perturbation rollouts into BGR recovery-curve summaries."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing observation_episodes.jsonl.")
    parser.add_argument("--out", required=True, help="Output directory for compact BGR recovery summaries.")
    parser.add_argument("--source-name", default="libero_openvla_observation_object3_h220")
    parser.add_argument("--alpha", type=float, default=0.8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(input_dir / "observation_episodes.jsonl")
    summary = summarize(rows, source_name=args.source_name, alpha=float(args.alpha))
    _write_outputs(summary, out_dir)
    return 0


def summarize(rows: list[dict[str, Any]], source_name: str, alpha: float = 0.8) -> dict[str, Any]:
    native_by_state: dict[tuple[Any, ...], list[bool]] = defaultdict(list)
    family_by_state_sigma: dict[str, dict[tuple[Any, ...], dict[float, list[bool]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    perturbation_rows: list[dict[str, Any]] = []

    for row in rows:
        state_key = _state_key(row)
        perturbation_type = str(row.get("perturbation_type", ""))
        success = bool(row.get("success", False))
        if perturbation_type == "identity":
            native_by_state[state_key].append(success)
            continue
        sigma = _sigma_for(row)
        family_by_state_sigma[perturbation_type][state_key][sigma].append(success)
        perturbation_rows.append(
            {
                "family": perturbation_type,
                "perturbation_name": str(row.get("perturbation_name", "")),
                "sigma": sigma,
                "success": success,
                "suite": str(row.get("suite", "")),
                "task_idx": int(row.get("task_idx", -1)),
                "task_name": str(row.get("task_name", "")),
                "episode_idx": int(row.get("episode_idx", -1)),
                "init_state_idx": row.get("init_state_idx"),
            }
        )

    family_summaries: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for family, state_map in sorted(family_by_state_sigma.items()):
        all_sigmas = sorted({sigma for sigma_map in state_map.values() for sigma in sigma_map})
        state_metrics = []
        for state_key, sigma_map in sorted(state_map.items()):
            native = native_by_state.get(state_key, [])
            if not native:
                continue
            sigmas = [0.0]
            recoveries = [mean([float(x) for x in native])]
            for sigma in all_sigmas:
                observations = sigma_map.get(sigma, [])
                if observations:
                    sigmas.append(float(sigma))
                    recoveries.append(mean([float(x) for x in observations]))
            if len(sigmas) < 2:
                continue
            sig_arr = np.array(sigmas, dtype=float)
            rec_arr = _monotone_nonincreasing(np.array(recoveries, dtype=float))
            state_metric = {
                "family": family,
                "state_id": _state_id(state_key),
                "clean": float(rec_arr[0]),
                "rauc": recovery_auc(sig_arr, rec_arr, sigma_max=1.0),
                "r80": critical_radius(sig_arr, rec_arr, alpha=alpha),
                "r50": critical_radius(sig_arr, rec_arr, alpha=0.5),
                "num_radii": int(len(sig_arr)),
            }
            state_rows.append(state_metric)
            state_metrics.append(state_metric)

        mean_curve = _mean_curve_for_family(native_by_state, state_map, all_sigmas)
        for curve_row in mean_curve:
            curve_row["family"] = family
            curve_rows.append(curve_row)

        if state_metrics:
            family_summaries.append(
                {
                    "source": source_name,
                    "family": family,
                    "num_states": len(state_metrics),
                    "num_radii": 1 + len(all_sigmas),
                    "clean_mean": _field_mean(state_metrics, "clean"),
                    "rauc_mean": _field_mean(state_metrics, "rauc"),
                    "r80_mean": _field_mean(state_metrics, "r80"),
                    "r50_mean": _field_mean(state_metrics, "r50"),
                    "rauc_sem": _field_sem(state_metrics, "rauc"),
                    "r80_sem": _field_sem(state_metrics, "r80"),
                }
            )

    return {
        "source": source_name,
        "alpha": alpha,
        "num_episode_rows": len(rows),
        "families": family_summaries,
        "curves": curve_rows,
        "states": state_rows,
        "perturbations": perturbation_rows,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _state_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("suite"),
        row.get("task_idx"),
        row.get("task_name"),
        row.get("episode_idx"),
        row.get("init_state_idx"),
    )


def _state_id(key: tuple[Any, ...]) -> str:
    suite, task_idx, task_name, episode_idx, init_state_idx = key
    return f"{suite}:{task_idx}:{task_name}:episode={episode_idx}:init={init_state_idx}"


def _sigma_for(row: dict[str, Any]) -> float:
    perturbation_type = str(row.get("perturbation_type", ""))
    params = dict(row.get("perturbation_params") or {})
    if perturbation_type == "blur":
        raw = abs(float(params.get("radius", 0.0)))
    elif perturbation_type == "brightness":
        raw = abs(1.0 - float(params.get("factor", 1.0)))
    elif perturbation_type == "occlusion":
        raw = abs(float(params.get("fraction", 0.0)))
    elif perturbation_type == "shift":
        dx = float(params.get("dx_fraction", 0.0))
        dy = float(params.get("dy_fraction", 0.0))
        raw = float(np.sqrt(dx * dx + dy * dy))
    else:
        raw = 0.0
    return float(np.clip(raw / SIGMA_NORMALIZERS.get(perturbation_type, max(raw, 1e-9)), 0.0, 1.0))


def _monotone_nonincreasing(values: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(np.clip(values, 0.0, 1.0))


def _mean_curve_for_family(
    native_by_state: dict[tuple[Any, ...], list[bool]],
    state_map: dict[tuple[Any, ...], dict[float, list[bool]]],
    all_sigmas: list[float],
) -> list[dict[str, Any]]:
    rows = []
    native_values = [float(success) for state_key in state_map for success in native_by_state.get(state_key, [])]
    raw_recoveries = [mean(native_values) if native_values else 0.0]
    sigmas = [0.0]
    trial_counts = [len(native_values)]
    for sigma in all_sigmas:
        values = [float(success) for sigma_map in state_map.values() for success in sigma_map.get(sigma, [])]
        sigmas.append(float(sigma))
        raw_recoveries.append(mean(values) if values else 0.0)
        trial_counts.append(len(values))
    mono = _monotone_nonincreasing(np.array(raw_recoveries, dtype=float))
    for sigma, raw, monotone, trials in zip(sigmas, raw_recoveries, mono.tolist(), trial_counts, strict=True):
        rows.append(
            {
                "sigma": float(sigma),
                "success_rate_raw": float(raw),
                "success_rate_monotone": float(monotone),
                "trials": int(trials),
            }
        )
    return rows


def _field_mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(mean(float(row[key]) for row in rows))


def _field_sem(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _write_outputs(summary: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(out_dir / "summary.csv", summary["families"])
    _write_csv(out_dir / "curves.csv", summary["curves"])
    _write_csv(out_dir / "state_metrics.csv", summary["states"])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
