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

## Reproducibility Metadata

The repository is MIT licensed. Runtime environment snapshots can be collected
on the cluster and checked against `results/environment_v1`:

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/collect_environment.py --out runs/environment_v1/compute_environment.json
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/collect_environment.py --out runs/environment_v1/compute_environment.json

~/remote_srun.sh --dry-run --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:10:00 /work/joy/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/collect_environment.py --out runs/environment_v1/gpu_environment.json
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:10:00 /work/joy/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/collect_environment.py --out runs/environment_v1/gpu_environment.json
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

## Active Estimator Validation

This run isolates the recovery-curve estimator from policy training by comparing fixed-budget probes against dense reference curves.

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 4 --mem 12G --time 01:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_estimator_experiment.py --config configs/estimator_bgr_full.yaml --out runs/estimator_full_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 01:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_estimator_experiment.py --config configs/estimator_bgr_full.yaml --out runs/estimator_full_v1
```

## LIBERO Simulator Probe

The cluster has LIBERO and robosuite available. This probe validates resettable LIBERO task states and object-pose perturbations on GPU/EGL; it is not a policy-success experiment.

```bash
~/remote_srun.sh --dry-run --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 01:00:00 /work/joy/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/probe_libero_suffix_states.py --suite libero_goal --task-ids 0,1,2,3,4 --init-state-ids 0,1,2 --radii 0.0,0.25,0.5,0.75,1.0 --trials-per-radius 4 --settle-steps 5 --image-size 64 --out runs/libero_probe_v2
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 01:00:00 /work/joy/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/probe_libero_suffix_states.py --suite libero_goal --task-ids 0,1,2,3,4 --init-state-ids 0,1,2 --radii 0.0,0.25,0.5,0.75,1.0 --trials-per-radius 4 --settle-steps 5 --image-size 64 --out runs/libero_probe_v2
```

## Robot Suffix Strategy Comparison

This diagnostic compares BGR-Suffix radius distributions while keeping the same replay-state estimator.

```bash
~/remote_srun.sh --dry-run --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_strategy.yaml --out runs/suffix_strategy_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_strategy.yaml --out runs/suffix_strategy_v1
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

Grid-margin ablations isolate BGR priority terms and boundary-centered radius sampling:

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_ablation.yaml --out runs/grid_margin_ablation_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_ablation.yaml --out runs/grid_margin_ablation_v1
```

The tabular grid-policy configs are retained as negative diagnostics:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 00:45:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_policy_mixed.yaml --out runs/grid_policy_mixed_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 00:45:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_policy_coverage.yaml --out runs/grid_policy_coverage_v1
```

See [results/README.md](results/README.md) for the current run ledger. The original tabular grid policy benchmark is retained as a negative diagnostic because broad replay saturates it after clean pretraining.

## Result Aggregation and Paper Figures

```bash
python3 scripts/aggregate_results.py --results-dir results --out-dir paper/figures
python3 scripts/analyze_significance.py --results-dir results --out-csv paper/figures/significance_tests.csv --out-tex paper/figures/significance_table.tex
```

This writes CSV summaries, a LaTeX summary table, and bar-chart figures used by `paper/main.tex`.

## AAAI Sources

The official AAAI-27 page lists the 2027 timetable and links the AAAI-27 author kit. The kit in `paper/AuthorKit27` was downloaded from `https://aaai.org/authorkit27/` on 2026-06-01.
