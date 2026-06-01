# Bifurcation-Guided Replay

This repository scaffolds the AAAI-27 submission:

**Bifurcation-Guided Replay: Learning at the Success-Failure Boundary of Decision Policies**

The current implementation contains a reusable BGR core, a Tier-0 synthetic recovery-margin benchmark, and a grid-backed procedural recovery-margin benchmark. The grid-margin benchmark uses generated obstacle maps for replay states and feasibility while preserving the state-conditioned recovery-margin object needed to evaluate BGR.

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
PYTHONPATH=src:. python3 -m unittest discover -s tests
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

The grid benchmarks are dependency-light procedural decision benchmarks with generated obstacle maps, replayable mid-path states, Manhattan-radius perturbations, and an exact shortest-path feasibility witness.

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 2 --mem 8G --time 01:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_bgr.yaml --out runs/grid_fast
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 01:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_bgr.yaml --out runs/grid_fast
```

The positive procedural benchmark is `grid_margin_bgr`, which evaluates state-conditioned margin expansion on grid-backed replay states:

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_bgr_full.yaml --out runs/grid_margin_full
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_bgr_full.yaml --out runs/grid_margin_full
```

See [results/README.md](results/README.md) for the current run ledger. The original tabular grid policy benchmark is retained as a negative diagnostic because broad replay saturates it after clean pretraining.

## AAAI Sources

The official AAAI-27 page lists the 2027 timetable and links the AAAI-27 author kit. The kit in `paper/AuthorKit27` was downloaded from `https://aaai.org/authorkit27/` on 2026-06-01.
