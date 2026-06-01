from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class LiberoProbeRow:
    suite: str
    task_id: int
    task_name: str
    language: str
    init_state_id: int
    radius: float
    radius_meters: float
    trials: int
    valid_rate: float
    success_rate: float
    mean_reward: float
    num_free_object_joints: int
    error: str


def row_to_dict(row: LiberoProbeRow) -> dict[str, Any]:
    return asdict(row)


def patch_torch_load_for_libero() -> None:
    """Allow LIBERO's trusted local init-state files under PyTorch 2.6+.

    LIBERO stores pruned init states as legacy torch pickles containing NumPy
    arrays. PyTorch 2.6 changed torch.load's default to weights_only=True,
    which rejects these files. This patch is intentionally scoped to the
    current process and should only be used for trusted local LIBERO assets.
    """

    import torch

    if getattr(torch.load, "_bgr_libero_patched", False):
        return
    original_load = torch.load

    def load_with_legacy_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load_with_legacy_default._bgr_libero_patched = True  # type: ignore[attr-defined]
    torch.load = load_with_legacy_default  # type: ignore[assignment]


def finite_observation(obs: dict[str, Any]) -> bool:
    for value in obs.values():
        if isinstance(value, np.ndarray) and not bool(np.all(np.isfinite(value))):
            return False
    return True


def free_object_joint_names(sim) -> list[str]:
    names: list[str] = []
    for name in sim.model.joint_names:
        if name.startswith("robot0_") or name.startswith("gripper0_"):
            continue
        addr = sim.model.get_joint_qpos_addr(name)
        if _addr_width(addr) >= 7:
            names.append(str(name))
    return names


def perturb_free_joint_qpos(sim, joint_name: str, dx: float, dy: float) -> float:
    qpos = np.array(sim.data.get_joint_qpos(joint_name), dtype=float).copy()
    if qpos.size < 2:
        return 0.0
    qpos[0] += dx
    qpos[1] += dy
    sim.data.set_joint_qpos(joint_name, qpos)
    sim.forward()
    return float(math.hypot(dx, dy))


def probe_task(
    *,
    suite_name: str,
    task_id: int,
    init_state_ids: list[int],
    radii: list[float],
    trials_per_radius: int,
    max_radius_meters: float,
    settle_steps: int,
    image_size: int,
    seed: int,
) -> list[LiberoProbeRow]:
    patch_torch_load_for_libero()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import os

    rng = np.random.default_rng(seed)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = suite.get_task_init_states(task_id)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=image_size,
        camera_widths=image_size,
    )
    rows: list[LiberoProbeRow] = []
    try:
        env.seed(seed)
        env.reset()
        for init_state_id in init_state_ids:
            if init_state_id >= len(init_states):
                rows.extend(
                    _error_rows(
                        suite_name,
                        task_id,
                        task.name,
                        task.language,
                        init_state_id,
                        radii,
                        trials_per_radius,
                        max_radius_meters,
                        f"init_state_id {init_state_id} out of range {len(init_states)}",
                    )
                )
                continue
            base_state = np.array(init_states[init_state_id], dtype=float).copy()
            env.set_init_state(base_state)
            object_joints = free_object_joint_names(env.sim)
            for radius in radii:
                valid = 0
                successes = 0
                rewards: list[float] = []
                error = ""
                radius_m = float(radius) * max_radius_meters
                for _ in range(trials_per_radius):
                    try:
                        env.set_init_state(base_state)
                        if object_joints and radius_m > 0:
                            joint = str(rng.choice(object_joints))
                            angle = float(rng.uniform(0.0, 2.0 * math.pi))
                            perturb_free_joint_qpos(sim=env.sim, joint_name=joint, dx=radius_m * math.cos(angle), dy=radius_m * math.sin(angle))
                            obs = env.regenerate_obs_from_state(env.get_sim_state())
                        else:
                            obs = env.regenerate_obs_from_state(env.get_sim_state())
                        reward = 0.0
                        done = False
                        for _step in range(settle_steps):
                            obs, reward, done, _info = env.step([0.0] * 7)
                            if done:
                                break
                        is_valid = finite_observation(obs)
                        valid += int(is_valid)
                        rewards.append(float(reward))
                        successes += int(bool(env.check_success()) or float(reward) > 0.0)
                    except Exception as exc:  # pragma: no cover - exercised on cluster.
                        error = f"{type(exc).__name__}: {exc}"
                rows.append(
                    LiberoProbeRow(
                        suite=suite_name,
                        task_id=task_id,
                        task_name=task.name,
                        language=task.language,
                        init_state_id=init_state_id,
                        radius=float(radius),
                        radius_meters=radius_m,
                        trials=trials_per_radius,
                        valid_rate=valid / max(1, trials_per_radius),
                        success_rate=successes / max(1, trials_per_radius),
                        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
                        num_free_object_joints=len(object_joints),
                        error=error,
                    )
                )
    finally:
        env.close()
    return rows


def _addr_width(addr) -> int:
    if isinstance(addr, tuple):
        return int(addr[1] - addr[0])
    if isinstance(addr, slice):
        return int((addr.stop or 0) - (addr.start or 0))
    return 1


def _error_rows(
    suite_name: str,
    task_id: int,
    task_name: str,
    language: str,
    init_state_id: int,
    radii: list[float],
    trials_per_radius: int,
    max_radius_meters: float,
    error: str,
) -> list[LiberoProbeRow]:
    return [
        LiberoProbeRow(
            suite=suite_name,
            task_id=task_id,
            task_name=task_name,
            language=language,
            init_state_id=init_state_id,
            radius=float(radius),
            radius_meters=float(radius) * max_radius_meters,
            trials=trials_per_radius,
            valid_rate=0.0,
            success_rate=0.0,
            mean_reward=0.0,
            num_free_object_joints=0,
            error=error,
        )
        for radius in radii
    ]
