# Environment Snapshot v1

Collected on 2026-06-01 with `scripts/collect_environment.py` through
`~/remote_srun.sh` under `/work/anonymous/bgr`.

## Compute job

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/collect_environment.py --out runs/environment_v1/compute_environment.json
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780318399_415321575.out
```

Summary: `cnode301`, Intel Xeon E5-2660 v4 CPU, 251GiB host memory,
Ubuntu Linux 5.15, Python 3.10.19 from the `safesae` conda environment.

## GPU job

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:10:00 /work/anonymous/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/collect_environment.py --out runs/environment_v1/gpu_environment.json
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780318387_340655701.out
```

Summary: `c1-g4-04`, AMD EPYC 7542 CPU, NVIDIA RTX A6000 GPUs visible
through PCI/proc metadata, 503GiB host memory, Ubuntu Linux 5.15, Python
3.10.19 from the `safesae` conda environment.
