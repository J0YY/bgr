# Experiment Results Ledger

All heavy runs are launched through `~/remote_srun.sh` and write outputs under `/work/joy/bgr/runs`.

## Completed Runs

### `toy_fast_v3`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 01:00:00 /work/joy/bgr env PYTHONPATH=src python scripts/run_toy_experiment.py --config configs/toy_bgr.yaml --out runs/toy_fast_v3
```

Remote log:

```text
/work/joy/bgr/logs/run_1780304197_887365905.out
```

Mean results over three seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.8962 | 0.3756 | 0.2829 | 0.3169 |
| Uniform | 0.8886 | 0.3659 | 0.2753 | 0.2987 |
| Fixed radius | 0.8240 | 0.2307 | 0.1294 | 0.2297 |
| Failure-only | 0.8209 | 0.2328 | 0.1347 | 0.2307 |
| PLR-loss proxy | 0.8224 | 0.2287 | 0.1314 | 0.2290 |

Interpretation: this diagnostic benchmark supports the core BGR claim under a controlled recovery-margin model.

### `grid_fast_v4`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 01:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_bgr.yaml --out runs/grid_fast_v4
```

Remote log:

```text
/work/joy/bgr/logs/run_1780305363_531953871.out
```

Mean results over three seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.9537 | 0.9241 | 0.9778 | 0.7605 |
| Uniform | 1.0000 | 0.9874 | 1.0000 | 0.8745 |
| Fixed radius | 1.0000 | 0.9856 | 1.0000 | 0.8785 |
| Failure-only | 1.0000 | 0.9928 | 1.0000 | 0.8785 |
| PLR-loss proxy | 1.0000 | 0.9936 | 1.0000 | 0.8810 |

Interpretation: this procedural grid setup is currently too easy after clean-suffix pretraining, and the BGR sampler is too narrow for saturated states. Do not use this as a positive main-paper result without changing the benchmark or sampler. The next iteration should either harden perturbations/generalization or add a saturated-state fallback that preserves coverage.

### `grid_margin_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 01:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_bgr.yaml --out runs/grid_margin_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780305895_462068628.out
```

Mean results over three seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.9288 | 0.4039 | 0.3246 | 0.3289 |
| Uniform | 0.8885 | 0.3609 | 0.2909 | 0.2911 |
| Failure-only | 0.8680 | 0.3199 | 0.2617 | 0.2685 |
| PLR-loss proxy | 0.8059 | 0.2123 | 0.1368 | 0.2129 |
| Fixed radius | 0.8041 | 0.2094 | 0.1351 | 0.2114 |

Interpretation: this grid-backed margin benchmark is the first positive procedural result. It retains grid-generated replay states and feasibility constraints while evaluating the recovery-margin object directly.

### `grid_margin_full_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_bgr_full.yaml --out runs/grid_margin_full_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780305974_747225749.out
```

Mean results over five seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.9474 | 0.4350 | 0.3442 | 0.3538 |
| Uniform | 0.8969 | 0.3979 | 0.3322 | 0.3145 |
| Failure-only | 0.8475 | 0.2919 | 0.2291 | 0.2549 |
| PLR-loss proxy | 0.8008 | 0.2117 | 0.1367 | 0.2139 |
| Fixed radius | 0.7981 | 0.2107 | 0.1354 | 0.2138 |

Interpretation: the larger five-seed procedural run preserves the BGR advantage. Compared with uniform replay, BGR improves final RAUC by 0.0371, RAUC AULC by 0.0393, median r80 by 0.0120, and clean success by 0.0505.
