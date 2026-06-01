# Bifurcation-Guided Replay

This repository scaffolds the AAAI-27 submission:

**Bifurcation-Guided Replay: Learning at the Success-Failure Boundary of Decision Policies**

The current implementation contains a reusable BGR core and a Tier-0 synthetic recovery-margin benchmark. The benchmark is intentionally light: it validates the algorithmic mechanics, estimator, priority score, and reporting path before adding MiniGrid/LIBERO experiments.

## Repository Layout

```text
src/bgr/                 Core BGR data structures, estimators, metrics, and samplers
scripts/                 Experiment and plotting entry points
configs/                 Versioned experiment configs
tests/                   Unit tests for estimator/priority/metrics behavior
paper/                   AAAI-27 manuscript skeleton and official AuthorKit27
```

## Local Smoke Tests

```bash
python -m pytest
```

## Tier-0 Experiment

Heavy or repeated experiments should run on the cluster through `~/remote_srun.sh`; use `/work/joy/bgr` as the remote project directory to avoid home-directory disk pressure.

Dry run:

```bash
~/remote_srun.sh --dry-run /work/joy/bgr python scripts/run_toy_experiment.py --config configs/toy_bgr.yaml --out runs/toy
```

Real run:

```bash
~/remote_srun.sh --github-test --git-pull --log /work/joy/bgr python scripts/run_toy_experiment.py --config configs/toy_bgr.yaml --out runs/toy
```

## AAAI Sources

The official AAAI-27 page lists the 2027 timetable and links the AAAI-27 author kit. The kit in `paper/AuthorKit27` was downloaded from `https://aaai.org/authorkit27/` on 2026-06-01.
