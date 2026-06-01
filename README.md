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

## Procedural Grid Recovery

The grid benchmark is a dependency-light procedural decision benchmark with generated obstacle maps, replayable mid-path states, Manhattan-radius perturbations, an exact shortest-path feasibility witness, and a tabular recovery policy.

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 2 --mem 8G --time 01:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_bgr.yaml --out runs/grid_fast
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 01:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_bgr.yaml --out runs/grid_fast
```

See [results/README.md](results/README.md) for the current run ledger. The first grid result is a negative diagnostic: the environment is too easy after clean pretraining, so BGR does not yet beat broad replay there.

## AAAI Sources

The official AAAI-27 page lists the 2027 timetable and links the AAAI-27 author kit. The kit in `paper/AuthorKit27` was downloaded from `https://aaai.org/authorkit27/` on 2026-06-01.
