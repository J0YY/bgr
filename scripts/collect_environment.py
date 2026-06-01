"""Collect lightweight runtime metadata for reproducibility records."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGES = [
    "numpy",
    "pyyaml",
    "matplotlib",
    "torch",
    "transformers",
    "libero",
    "robosuite",
    "mujoco",
    "gymnasium",
]


def command_output(args: list[str], timeout: int = 20) -> dict[str, Any]:
    executable = shutil.which(args[0])
    if executable is None:
        return {"available": False}
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - depends on cluster tools.
        return {"available": True, "error": repr(exc)}
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def collect() -> dict[str, Any]:
    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "SLURM_CLUSTER_NAME",
        "SLURM_CPUS_ON_NODE",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_PARTITION",
        "SLURM_MEM_PER_NODE",
    ]
    return {
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
        },
        "environment": {key: os.environ.get(key) for key in env_keys},
        "packages": package_versions(),
        "commands": {
            "lscpu": command_output(["lscpu"]),
            "free_h": command_output(["free", "-h"]),
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version,cuda_version",
                    "--format=csv,noheader",
                ]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Path to write JSON metadata.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(collect(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
