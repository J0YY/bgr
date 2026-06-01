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

### `suffix_full_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_bgr_full.yaml --out runs/suffix_full_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780307573_760181637.out
```

Mean results over five seeds:

| Method | Clean | Object RAUC | Median r80 | EE-transfer RAUC | RAUC AULC |
|---|---:|---:|---:|---:|---:|
| BGR-Suffix | 0.8738 | 0.4730 | 0.4534 | 0.3029 | 0.3745 |
| Uniform suffix | 0.8368 | 0.4854 | 0.5032 | 0.3081 | 0.3716 |
| Clean FT | 0.8652 | 0.2630 | 0.2068 | 0.1889 | 0.2389 |
| Failure-only | 0.7582 | 0.4237 | 0.4761 | 0.2434 | 0.3031 |
| Loss priority | 0.7120 | 0.1704 | 0.1285 | 0.1447 | 0.1694 |
| Fixed radius | 0.6848 | 0.1585 | 0.1255 | 0.1379 | 0.1648 |

Interpretation: BGR-Suffix strongly beats clean-only, fixed-radius, failure-only, and loss-priority recovery training. Compared with uniform suffix replay, it improves clean success and sample efficiency, but uniform retains higher final object RAUC and transfer RAUC in this lightweight suffix simulator. Treat this as a robotics-style diagnostic rather than the final LIBERO/OpenVLA evidence requested by the full spec.

### `estimator_full_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 01:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_estimator_experiment.py --config configs/estimator_bgr_full.yaml --out runs/estimator_full_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780309253_221712673.out
```

Mean results over five seeds, with 17 Bernoulli probes per state and dense 201-point curves used only as references:

| Estimator | r80 MAE | RAUC MAE | Boundary hit rate |
|---|---:|---:|---:|
| Active BGR | 0.0787 | 0.0655 | 0.6824 |
| Coarse grid | 0.0717 | 0.0598 | 0.6535 |
| Uniform random | 0.1056 | 0.0647 | 0.6152 |

Interpretation: active probing improves boundary-hit rate over uniform and coarse probing under the same rollout budget. Coarse grids remain slightly better on mean absolute radius error in this one-dimensional synthetic setting, so the paper frames this as a probe-efficiency validation rather than as proof that active probing dominates all fixed grids.

### `grid_margin_ablation_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_ablation.yaml --out runs/grid_margin_ablation_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780309573_174287695.out
```

Mean results over five seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.9474 | 0.4350 | 0.3442 | 0.3538 |
| No uncertainty | 0.9467 | 0.4356 | 0.3429 | 0.3541 |
| No sharpness | 0.9479 | 0.4355 | 0.3449 | 0.3539 |
| Uniform radius | 0.8939 | 0.3845 | 0.3235 | 0.3068 |
| Uniform replay | 0.8969 | 0.3979 | 0.3322 | 0.3145 |

Interpretation: on the grid-margin benchmark, radius-level boundary sampling is the main BGR ingredient. Removing uncertainty or sharpness weighting has little effect, while replacing boundary-centered perturbation radii with uniform radii drops below uniform replay.
