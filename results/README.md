# Experiment Results Ledger

All heavy runs are launched through `~/remote_srun.sh` and write outputs under `/work/joy/bgr/runs`.

## Completed Runs

### `environment_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/collect_environment.py --out runs/environment_v1/compute_environment.json
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:10:00 /work/joy/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/collect_environment.py --out runs/environment_v1/gpu_environment.json
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780318399_415321575.out
/work/joy/bgr/logs/run_1780318387_340655701.out
```

Checked-in snapshots:

```text
results/environment_v1/compute_environment.json
results/environment_v1/gpu_environment.json
```

Interpretation: these snapshots record Slurm allocation details, hostnames, CPU model, host memory, OS/kernel, Python executable/version, selected package versions, and GPU hardware metadata for the LIBERO/EGL path. `nvidia-smi` is not on the job path, but the GPU node exposes NVIDIA RTX A6000 hardware through PCI/proc metadata.

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

### `grid_margin_pair_15seed_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_pair_15seed.yaml --out runs/grid_margin_pair_15seed_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780328088_151681007.out
```

Mean results over 15 paired seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.9461 | 0.4345 | 0.3441 | 0.3525 |
| Uniform | 0.8943 | 0.3961 | 0.3318 | 0.3129 |

Interpretation: this strengthens the primary procedural claim. BGR improves
all four reported metrics over uniform on every paired seed; paired exact
sign-flip tests give p=0.00006 for clean success, RAUC, median r80, and RAUC
AULC.

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

### `estimator_pair_15seed_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:30:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_estimator_experiment.py --config configs/estimator_pair_15seed.yaml --out runs/estimator_pair_15seed_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780327938_068713790.out
```

Mean results over 15 paired seeds:

| Estimator | r80 MAE | RAUC MAE | Boundary hit rate |
|---|---:|---:|---:|
| Active BGR | 0.0799 | 0.0645 | 0.6775 |
| Uniform | 0.1064 | 0.0656 | 0.6007 |

Interpretation: active probing improves boundary-hit rate and lowers r80 error
relative to uniform under the same 17-probe budget. Paired exact sign-flip
tests give p=0.00006 for both effects.

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

### `grid_policy_hard_fast_v1` and `grid_policy_mixed_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 00:45:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_policy_hard_fast.yaml --out runs/grid_policy_hard_fast_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 00:45:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_policy_mixed.yaml --out runs/grid_policy_mixed_v1
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780316465_289961157.out
/work/joy/bgr/logs/run_1780317123_541002217.out
```

Mean results over three seeds for the mixed run:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.7578 | 0.7096 | 0.7583 | 0.5473 |
| BGR mixed | 0.7839 | 0.7589 | 0.8463 | 0.5861 |
| Uniform | 0.9965 | 0.9641 | 0.9833 | 0.7838 |
| Fixed radius | 0.9931 | 0.9771 | 1.0000 | 0.7999 |
| PLR-loss proxy | 0.9991 | 0.9786 | 1.0000 | 0.7960 |

Interpretation: this is a negative policy-level diagnostic. Adding clean/uniform radius coverage improves tabular BGR, but fixed-radius and loss-priority replay saturate the tabular oracle-imitation grid policy much faster. The main paper should continue using `grid_margin_full_v1` as the positive procedural result and treat this policy-level tabular setup as a benchmark-design failure, not as evidence for BGR.

### `grid_policy_coverage_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 00:45:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_policy_coverage.yaml --out runs/grid_policy_coverage_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780319352_975146235.out
```

Mean results over three seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR mixed coverage | 0.9592 | 0.9117 | 0.9000 | 0.6914 |
| BGR | 0.9054 | 0.8730 | 0.8000 | 0.6448 |
| Uniform | 0.9965 | 0.9641 | 0.9833 | 0.7838 |
| Fixed radius | 0.9931 | 0.9771 | 1.0000 | 0.7999 |
| PLR-loss proxy | 0.9991 | 0.9786 | 1.0000 | 0.7960 |

Interpretation: coverage-aware BGR substantially improves the tabular policy diagnostic over the earlier mixed sampler (RAUC 0.9117 vs. 0.7589) and over narrow BGR in the same config (0.8730). The result still does not overturn the diagnostic failure: in this oracle-imitation grid policy, broad fixed-radius and loss-priority replay nearly saturate all states faster than BGR.

### `libero_probe_v2`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 01:00:00 /work/joy/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/probe_libero_suffix_states.py --suite libero_goal --task-ids 0,1,2,3,4 --init-state-ids 0,1,2 --radii 0.0,0.25,0.5,0.75,1.0 --trials-per-radius 4 --settle-steps 5 --image-size 64 --out runs/libero_probe_v2
```

Remote log:

```text
/work/joy/bgr/logs/run_1780311860_935283441.out
```

Probe summary:

| Suite | Tasks | Init states/task | Radii | Trials/radius | Valid rate | Zero-action success |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO-Goal | 5 | 3 | 5 | 4 | 1.0000 | 0.0000 |

Interpretation: the cluster can instantiate LIBERO environments on GPU with `MUJOCO_GL=egl`, load trusted local init states after patching PyTorch 2.6's `torch.load(weights_only=True)` default, and apply object free-joint perturbations across resettable LIBERO states. This is infrastructure evidence for BGR-Suffix on real LIBERO simulator states, not a policy result: this probe uses zero-action rollouts, which are not expected to solve tasks. A separate fixed-policy OpenVLA recovery summary is recorded in `libero_openvla_recovery_v1`.

### `libero_openvla_recovery_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/summarize_libero_openvla_recovery.py --input-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_object3_h220_bash --out runs/libero_openvla_recovery_v1 --source-name libero_openvla_observation_object3_h220_bash
```

Remote log:

```text
/work/joy/bgr/logs/run_1780320300_854946446.out
```

Source artifact:

```text
/work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_object3_h220_bash
```

Mean recovery metrics over nine closed-loop OpenVLA LIBERO-Object replay states:

| Perturbation family | Clean | RAUC | r80 | r50 |
|---|---:|---:|---:|---:|
| blur | 1.0000 | 0.4667 | 0.3067 | 0.4667 |
| brightness | 1.0000 | 0.8000 | 0.7333 | 0.8000 |
| occlusion | 1.0000 | 0.3889 | 0.2822 | 0.3889 |
| shift | 1.0000 | 0.5148 | 0.3793 | 0.5148 |

Interpretation: these are fixed-policy OpenVLA recovery curves, not BGR
fine-tuning results. They show that the BGR recovery-curve object is measurable
on learned VLA rollouts: the policy is clean-successful on the replay states,
but success drops sharply under blur, occlusion, and image shift perturbations.

### `libero_openvla_boundary_selection_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/summarize_openvla_boundary_selection.py --proposal-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_guided_h160 --proposal-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_guided_seed2_h160 --proposal-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_guided_seed3_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed1b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed2b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed3b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed4b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed5b_skip_lp2_h160 --out runs/libero_openvla_boundary_selection_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780320651_072406017.out
```

Boundary band is observed counterfactual failure rate in `[0.25, 0.75]`:

| Method | Runs | Mean CF rate | Boundary hit rate | Mean `abs(CF-0.5)` | Certificates |
|---|---:|---:|---:|---:|---:|
| Proposal-guided | 3 | 0.5278 | 0.8750 | 0.1667 | 25.33 |
| Random-balanced | 5 | 0.6438 | 0.5750 | 0.2563 | 20.60 |

Interpretation: proposal-guided selection finds perturbations closer to the
OpenVLA success-failure boundary than random-balanced selection. This supports
BGR's boundary-discovery premise on learned VLA rollouts. It is not a
fine-tuning result, and the comparison is diagnostic because proposal-guided
artifacts concentrate on blur/shift while random-balanced artifacts cover four
families.

### `libero_openvla_boundary_selection_balanced_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/summarize_openvla_boundary_selection.py --proposal-method-name bgr_boundary --proposal-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed1_lp2_h160 --proposal-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed2_lp2_h160 --proposal-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed3_lp2_h160 --proposal-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed4_lp2_h160 --proposal-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed5_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed1b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed2b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed3b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed4b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed5b_skip_lp2_h160 --out runs/libero_openvla_boundary_selection_balanced_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780321470_140661271.out
```

| Method | Runs | Mean CF rate | Boundary hit rate | Mean `abs(CF-0.5)` | Certificates |
|---|---:|---:|---:|---:|---:|
| BGR-boundary | 5 | 0.5500 | 0.6250 | 0.2958 | 16.00 |
| Random-balanced | 5 | 0.6438 | 0.5750 | 0.2563 | 20.60 |

Interpretation: this is the paper-table OpenVLA selection audit because both
methods cover the same four perturbation families over five runs. It supports
boundary-discovery as a measurable learned-policy diagnostic, but not yet
OpenVLA fine-tuning.

### `openvla_bgr_finetune_manifest_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/export_openvla_bgr_finetune_manifest.py --bgr-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed1_lp2_h160 --bgr-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed2_lp2_h160 --bgr-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed3_lp2_h160 --bgr-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed4_lp2_h160 --bgr-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed5_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed1b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed2b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed3b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed4b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed5b_skip_lp2_h160 --out runs/openvla_bgr_finetune_manifest_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780321861_807492940.out
```

Interpretation: this exports validated candidate-level manifests and OpenVLA-OFT
LoRA command templates. It identified 80 candidates total; BGR-boundary has
25/40 candidates in the boundary band, and random-balanced has 23/40. The
OpenVLA-OFT trainer still requires RLDS episodes, so this is a fine-tuning
scaffold rather than a completed fine-tuning result.

### `openvla_teacher_replay_manifest_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 12G --time 00:15:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/export_openvla_teacher_replay_manifest.py --boundary-only --max-steps-per-episode 64 --bgr-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed1_lp2_h160 --bgr-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed2_lp2_h160 --bgr-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed3_lp2_h160 --bgr-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed4_lp2_h160 --bgr-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed5_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed1b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed2b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed3b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed4b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed5b_skip_lp2_h160 --out runs/openvla_teacher_replay_manifest_v1
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780322114_709717949.out
/work/joy/bgr/logs/run_1780322189_156724835.out
```

Interpretation: this is the next bridge toward OpenVLA-OFT fine-tuning. It
exports 11,776 step-level rows pairing validated boundary candidates with target
actions from successful native OpenVLA rollouts. A downstream converter must
replay the native action prefix in LIBERO, render observations, apply the
candidate perturbation to the image stream, and write RLDS episodes.

### `openvla_teacher_oft_smoke_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:20:00 /work/joy/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_oft_smoke_v1 --max-examples 4 --selection first_per_family --num-steps-wait 10 --env-image-size 256 --image-size 224
```

Remote log:

```text
/work/joy/bgr/logs/run_1780322998_166835773.out
```

Interpretation: this validates that the teacher-replay manifest can be converted
into rendered OpenVLA-OFT-style examples under LIBERO GPU/EGL. The checked-in
smoke artifact contains one example for each visual perturbation family: blur,
brightness, shift, and occlusion. Each NPZ contains primary image
`(224,224,3)`, wrist image `(224,224,3)`, 8D LIBERO state, 7D action, and
language label. This is still a bridge artifact, not RLDS conversion or
fine-tuning.

### `openvla_oft_pack_smoke_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/pack_openvla_oft_examples.py --examples results/openvla_teacher_oft_smoke_v1/examples.jsonl --out runs/openvla_oft_pack_smoke_v1 --write-hdf5
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780323325_136783423.out
/work/joy/bgr/logs/run_1780323386_780730416.out
```

Interpretation: this validates and packs the rendered OFT-field examples into a
LIBERO-style HDF5 smoke dataset. The checked-in HDF5 has `data/demo_*` groups
with `actions`, `obs/agentview_rgb`, `obs/eye_in_hand_rgb`, `obs/ee_states`,
and `obs/gripper_states`. This is the next bridge artifact toward RLDS
conversion, not a completed RLDS dataset or fine-tuning run.

### `openvla_oft_tfds_smoke_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:20:00 /work/joy/bgr /work/joy/safesae-openvla/bin/python scripts/export_openvla_oft_tfds.py --examples results/openvla_teacher_oft_smoke_v1/examples.jsonl --out runs/openvla_oft_tfds_smoke_v1 --dataset-name bgr_libero_oft_smoke --version 1.0.0
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780323827_280717056.out
/work/joy/bgr/logs/run_1780323900_674014804.out
```

Interpretation: this exports the rendered OFT-field examples as a minimal
RLDS-style TFDS dataset and verifies readback with `tfds.builder_from_directory`.
The checked-in smoke dataset has four train episodes and four total steps, one
per perturbation family, with primary RGB, wrist RGB, 8D state, 7D action, and
language fields. This closes the smoke-tested TFDS conversion path, but it is
not yet a full OpenVLA-OFT fine-tuning run.

### `openvla_oft_tfds_libero_goal_smoke_v2`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:20:00 /work/joy/bgr /work/joy/safesae-openvla/bin/python scripts/export_openvla_oft_tfds.py --examples results/openvla_teacher_oft_smoke_v1/examples.jsonl --out runs/openvla_oft_tfds_libero_goal_smoke_v2 --dataset-name libero_goal_no_noops --version 1.0.0
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 12G --time 00:15:00 /work/joy/bgr env PYTHONPATH=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/python -c "from pathlib import Path; import tensorflow_datasets as tfds; from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs; from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds; kw=make_oxe_dataset_kwargs('libero_goal_no_noops', Path('/work/joy/bgr/runs/openvla_oft_tfds_libero_goal_smoke_v2'), load_camera_views=('primary','wrist'), load_proprio=True, load_language=True); ds, stats=make_dataset_from_rlds(train=True, shuffle=False, **kw); ex=next(iter(tfds.as_numpy(ds))); print('keys', sorted(ex.keys())); print('obs', {k:v.shape for k,v in ex['observation'].items()}); print('task', ex['task']['language_instruction'][0].decode()); print('action', ex['action'].shape); print('stats_action_mean', stats['action']['mean'].shape); print('stats_proprio_mean', stats['proprio']['mean'].shape)"
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780324900_610846337.out
/work/joy/bgr/logs/run_1780324943_628488822.out
```

Interpretation: this re-exports the TFDS smoke under the stock
`libero_goal_no_noops` name and validates it through OpenVLA-OFT's own
`make_oxe_dataset_kwargs` and `make_dataset_from_rlds` path. The loader computes
dataset statistics and yields primary/wrist image fields, 8D proprio, 7D action,
and language. This verifies that the BGR-rendered examples can enter the
unmodified OpenVLA-OFT RLDS loader; the remaining gap is scaling the render to a
training-sized dataset and running LoRA fine-tuning/evaluation.

### `openvla_teacher_oft_balanced64_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:40:00 /work/joy/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_oft_balanced64_v1 --max-examples 64 --selection balanced_episodes --episodes-per-family 1 --max-steps-per-episode 16 --num-steps-wait 10 --env-image-size 256 --image-size 224
```

Remote log:

```text
/work/joy/bgr/logs/run_1780325448_898879565.out
```

Interpretation: this scales the renderer from four isolated smoke frames to
four contiguous 16-step replay episodes, one per visual perturbation family.
Each rendered row contains primary RGB, wrist RGB, 8D LIBERO state, 7D teacher
action, and language.

### `openvla_oft_pack_balanced64_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/pack_openvla_oft_examples.py --examples runs/openvla_teacher_oft_balanced64_v1/examples.jsonl --out runs/openvla_oft_pack_balanced64_v1 --write-hdf5
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780325655_972521234.out
/work/joy/bgr/logs/run_1780325709_771885875.out
```

Interpretation: the balanced rendered rows pack into four HDF5 demos with
`actions (16,7)`, `obs/agentview_rgb (16,224,224,3)`, wrist RGB, EEF states,
and gripper states. This verifies episode-safe grouping for multi-step
candidate replays.

### `openvla_oft_tfds_libero_goal_balanced64_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:20:00 /work/joy/bgr /work/joy/safesae-openvla/bin/python scripts/export_openvla_oft_tfds.py --examples runs/openvla_teacher_oft_balanced64_v1/examples.jsonl --out runs/openvla_oft_tfds_libero_goal_balanced64_v1 --dataset-name libero_goal_no_noops --version 1.0.0
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 12G --time 00:15:00 /work/joy/bgr env PYTHONPATH=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/python -c "from pathlib import Path; import tensorflow_datasets as tfds; from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs; from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds; kw=make_oxe_dataset_kwargs('libero_goal_no_noops', Path('/work/joy/bgr/runs/openvla_oft_tfds_libero_goal_balanced64_v1'), load_camera_views=('primary','wrist'), load_proprio=True, load_language=True); ds, stats=make_dataset_from_rlds(train=True, shuffle=False, **kw); ex=next(iter(tfds.as_numpy(ds))); print('keys', sorted(ex.keys())); print('obs', {k:v.shape for k,v in ex['observation'].items()}); print('task', ex['task']['language_instruction'][0].decode()); print('action', ex['action'].shape); print('stats_action_mean', stats['action']['mean'].shape); print('stats_proprio_mean', stats['proprio']['mean'].shape)"
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780325681_170623263.out
/work/joy/bgr/logs/run_1780325731_654888892.out
```

Interpretation: this exports the 64-step balanced set under
`libero_goal_no_noops` and validates it through OpenVLA-OFT's unmodified RLDS
loader. The loader computes dataset statistics and yields trajectory chunks with
primary/wrist image fields, proprio `(16,8)`, action `(16,7)`, and language.
The remaining gap is now training-sized scale and LoRA fine-tuning/evaluation.

### `openvla_oft_finetune_balanced64_ckpt_smoke_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 8 --mem 64G --time 01:00:00 /work/joy/bgr bash -lc 'cd /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft && env WANDB_MODE=disabled HF_HOME=/work/joy/cache_home/huggingface TRANSFORMERS_CACHE=/work/joy/cache_home/huggingface/hub PYTHONPATH=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path openvla/openvla-7b --data_root_dir /work/joy/bgr/runs/openvla_oft_tfds_libero_goal_balanced64_v1 --dataset_name libero_goal_no_noops --run_root_dir /work/joy/bgr/runs/openvla_oft_finetune_balanced64_ckpt_smoke_v1 --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio True --batch_size 1 --learning_rate 5e-4 --num_steps_before_decay 100000 --max_steps 1 --save_freq 1 --save_latest_checkpoint_only True --image_aug False --lora_rank 8 --merge_lora_during_training False --shuffle_buffer_size 16 --wandb_entity disabled --wandb_project bgr --run_id_note bgr-balanced64-ckpt-smoke --wandb_log_freq 1'
```

Remote log:

```text
/work/joy/bgr/logs/run_1780326595_120788212.out
```

Interpretation: this is the first end-to-end OpenVLA-OFT training smoke on BGR
data. The run loaded OpenVLA 7B, initialized LoRA rank 8 plus proprio and L1
action heads, loaded the BGR `libero_goal_no_noops` TFDS dataset, completed a
one-step forward/backward/update loop, and wrote checkpoint artifacts under
`/work/joy/bgr/runs`. The checkpoint directory is 656M and is not checked into
git; checked-in metadata records the saved LoRA adapter, action head, proprio
projector, and dataset statistics. This is still a smoke test, not a
training-scale fine-tuning/evaluation result.

### `openvla_teacher_oft_random_balanced64_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:40:00 /work/joy/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_oft_random_balanced64_v1 --method random_balanced --max-examples 64 --selection balanced_episodes --episodes-per-family 1 --max-steps-per-episode 16 --num-steps-wait 10 --env-image-size 256 --image-size 224
```

Remote log:

```text
/work/joy/bgr/logs/run_1780327075_235451672.out
```

Interpretation: this renders the matched random-balanced baseline analog of
the BGR balanced64 bridge. Validation found 64 rows, all `random_balanced`,
split evenly across blur, brightness, shift, and occlusion with one 16-step
episode per family.

### `openvla_oft_pack_random_balanced64_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/pack_openvla_oft_examples.py --examples runs/openvla_teacher_oft_random_balanced64_v1/examples.jsonl --out runs/openvla_oft_pack_random_balanced64_v1 --write-hdf5
```

Remote log:

```text
/work/joy/bgr/logs/run_1780327145_234510679.out
```

Interpretation: the random-balanced rendered rows pack into four HDF5 demos
with 7D actions, 8D proprio/state, and primary/wrist image streams. This
confirms the baseline uses the same episode-safe bridge as BGR.

### `openvla_oft_tfds_libero_goal_random_balanced64_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:20:00 /work/joy/bgr /work/joy/safesae-openvla/bin/python scripts/export_openvla_oft_tfds.py --examples runs/openvla_teacher_oft_random_balanced64_v1/examples.jsonl --out runs/openvla_oft_tfds_libero_goal_random_balanced64_v1 --dataset-name libero_goal_no_noops --version 1.0.0
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 12G --time 00:15:00 /work/joy/bgr env PYTHONPATH=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/python -c "from pathlib import Path; import tensorflow_datasets as tfds; from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs; from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds; kw=make_oxe_dataset_kwargs('libero_goal_no_noops', Path('/work/joy/bgr/runs/openvla_oft_tfds_libero_goal_random_balanced64_v1'), load_camera_views=('primary','wrist'), load_proprio=True, load_language=True); ds, stats=make_dataset_from_rlds(train=True, shuffle=False, **kw); ex=next(iter(tfds.as_numpy(ds))); print('keys', sorted(ex.keys())); print('obs', {k:v.shape for k,v in ex['observation'].items()}); print('task', ex['task']['language_instruction'][0].decode()); print('action', ex['action'].shape); print('stats_action_mean', stats['action']['mean'].shape); print('stats_proprio_mean', stats['proprio']['mean'].shape)"
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780327170_100921796.out
/work/joy/bgr/logs/run_1780327240_909792233.out
```

Interpretation: this exports the matched random-balanced set under
`libero_goal_no_noops` and validates it through OpenVLA-OFT's unmodified RLDS
loader. The loader yields primary/wrist image fields, proprio `(16,8)`, action
`(16,7)`, and language, matching the BGR bridge.

### `openvla_oft_finetune_random_balanced64_ckpt_smoke_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 8 --mem 64G --time 01:00:00 /work/joy/bgr bash -lc 'cd /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft && env WANDB_MODE=disabled HF_HOME=/work/joy/cache_home/huggingface TRANSFORMERS_CACHE=/work/joy/cache_home/huggingface/hub PYTHONPATH=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path openvla/openvla-7b --data_root_dir /work/joy/bgr/runs/openvla_oft_tfds_libero_goal_random_balanced64_v1 --dataset_name libero_goal_no_noops --run_root_dir /work/joy/bgr/runs/openvla_oft_finetune_random_balanced64_ckpt_smoke_v1 --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio True --batch_size 1 --learning_rate 5e-4 --num_steps_before_decay 100000 --max_steps 1 --save_freq 1 --save_latest_checkpoint_only True --image_aug False --lora_rank 8 --merge_lora_during_training False --shuffle_buffer_size 16 --wandb_entity disabled --wandb_project bgr --run_id_note random-balanced64-ckpt-smoke --wandb_log_freq 1'
```

Remote log:

```text
/work/joy/bgr/logs/run_1780327341_446370402.out
```

Interpretation: this mirrors the BGR one-step OpenVLA-OFT checkpoint smoke on
the matched random-balanced baseline. It loaded OpenVLA 7B, initialized LoRA
rank 8 plus proprio and L1 action heads, loaded the baseline
`libero_goal_no_noops` TFDS dataset, completed one optimizer step, and wrote the
same class of checkpoint files under `/work/joy/bgr/runs`. The 656M checkpoint
weights are not checked into git.

### `openvla_teacher_oft_bgr_balanced2048_v1` and `openvla_teacher_oft_random_balanced2048_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 24G --time 01:30:00 /work/joy/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_oft_bgr_balanced2048_v1 --method bgr_boundary --max-examples 2048 --selection balanced_episodes --episodes-per-family 8 --max-steps-per-episode 64 --num-steps-wait 10 --env-image-size 256 --image-size 224
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 24G --time 01:30:00 /work/joy/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_oft_random_balanced2048_v1 --method random_balanced --max-examples 2048 --selection balanced_episodes --episodes-per-family 8 --max-steps-per-episode 64 --num-steps-wait 10 --env-image-size 256 --image-size 224
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780379901_989185344.out
/work/joy/bgr/logs/run_1780380209_343397743.out
```

Interpretation: this scales the OpenVLA render bridge to matched 2,048-step
BGR-boundary and random-balanced datasets. Each render has 32 replay episodes,
8 episodes per visual perturbation family, and 64 steps per episode. The full
render trees are 492M and 524M under `/work` and are not checked into git.

### `openvla_oft_pack_bgr_balanced2048_v1` and `openvla_oft_pack_random_balanced2048_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 16G --time 00:30:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/pack_openvla_oft_examples.py --examples runs/openvla_teacher_oft_bgr_balanced2048_v1/examples.jsonl --out runs/openvla_oft_pack_bgr_balanced2048_v1 --write-hdf5
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 16G --time 00:30:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/pack_openvla_oft_examples.py --examples runs/openvla_teacher_oft_random_balanced2048_v1/examples.jsonl --out runs/openvla_oft_pack_random_balanced2048_v1 --write-hdf5
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780380477_635336557.out
/work/joy/bgr/logs/run_1780380526_513758944.out
```

Interpretation: both matched 2,048-step renders pack into LIBERO-style HDF5
with 7D actions, 8D state, and primary/wrist image streams. The full HDF5 packs
are 383M and 399M under `/work`.

### `openvla_oft_tfds_libero_goal_bgr_balanced2048_v1`, `openvla_oft_tfds_libero_goal_random_balanced2048_v1`, and `openvla_oft_loader_balanced2048_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 16G --time 01:00:00 /work/joy/bgr /work/joy/safesae-openvla/bin/python scripts/export_openvla_oft_tfds.py --examples runs/openvla_teacher_oft_bgr_balanced2048_v1/examples.jsonl --out runs/openvla_oft_tfds_libero_goal_bgr_balanced2048_v1 --dataset-name libero_goal_no_noops --version 1.0.0
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 16G --time 01:00:00 /work/joy/bgr /work/joy/safesae-openvla/bin/python scripts/export_openvla_oft_tfds.py --examples runs/openvla_teacher_oft_random_balanced2048_v1/examples.jsonl --out runs/openvla_oft_tfds_libero_goal_random_balanced2048_v1 --dataset-name libero_goal_no_noops --version 1.0.0
```

Loader validation uses OpenVLA-OFT's unmodified `make_oxe_dataset_kwargs` and
`make_dataset_from_rlds` on each TFDS root.

Remote logs:

```text
/work/joy/bgr/logs/run_1780380577_704261678.out
/work/joy/bgr/logs/run_1780380831_695183835.out
/work/joy/bgr/logs/run_1780381106_407628579.out
/work/joy/bgr/logs/run_1780381221_375772232.out
```

Interpretation: both matched 2,048-step TFDS exports use the stock
`libero_goal_no_noops` dataset name and load through OpenVLA-OFT's RLDS loader.
The loader computes dataset statistics for 2,048 transitions and 32
trajectories, then yields 64-step chunks with primary/wrist images, proprio
`(64,8)`, action `(64,7)`, and language. This is a larger dataset
compatibility gate.

### `openvla_oft_finetune_bgr_balanced2048_step10_v1` and `openvla_oft_finetune_random_balanced2048_step10_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition low-prio-gpu --gres gpu:a6000:1 --cpus 8 --mem 64G --time 02:00:00 /work/joy/bgr bash -lc 'cd /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft && env WANDB_MODE=disabled HF_HOME=/work/joy/cache_home/huggingface TRANSFORMERS_CACHE=/work/joy/cache_home/huggingface/hub PYTHONPATH=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path openvla/openvla-7b --data_root_dir /work/joy/bgr/runs/openvla_oft_tfds_libero_goal_bgr_balanced2048_v1 --dataset_name libero_goal_no_noops --run_root_dir /work/joy/bgr/runs/openvla_oft_finetune_bgr_balanced2048_step10_v1 --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio True --batch_size 1 --learning_rate 5e-4 --num_steps_before_decay 100000 --max_steps 10 --save_freq 10 --save_latest_checkpoint_only True --image_aug False --lora_rank 8 --merge_lora_during_training False --shuffle_buffer_size 512 --wandb_entity disabled --wandb_project bgr --run_id_note bgr-balanced2048-step10 --wandb_log_freq 1'
~/remote_srun.sh --github-test --git-pull --log --partition low-prio-gpu --gres gpu:a6000:1 --cpus 8 --mem 64G --time 02:00:00 /work/joy/bgr bash -lc 'cd /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft && env WANDB_MODE=disabled HF_HOME=/work/joy/cache_home/huggingface TRANSFORMERS_CACHE=/work/joy/cache_home/huggingface/hub PYTHONPATH=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path openvla/openvla-7b --data_root_dir /work/joy/bgr/runs/openvla_oft_tfds_libero_goal_random_balanced2048_v1 --dataset_name libero_goal_no_noops --run_root_dir /work/joy/bgr/runs/openvla_oft_finetune_random_balanced2048_step10_v1 --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio True --batch_size 1 --learning_rate 5e-4 --num_steps_before_decay 100000 --max_steps 10 --save_freq 10 --save_latest_checkpoint_only True --image_aug False --lora_rank 8 --merge_lora_during_training False --shuffle_buffer_size 512 --wandb_entity disabled --wandb_project bgr --run_id_note random-balanced2048-step10 --wandb_log_freq 1'
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780381812_955203657.out
/work/joy/bgr/logs/run_1780382026_303226869.out
```

Interpretation: both matched 2,048-step TFDS exports now complete bounded
OpenVLA-OFT LoRA fine-tuning runs. Each run loads OpenVLA 7B, initializes LoRA
rank 8 plus proprio and L1 action heads, trains for 10 optimizer steps, and
writes a latest checkpoint under `/work/joy/bgr/runs`. The remote checkpoint
trees are 656M each and are not checked into git. This upgrades the OpenVLA
bridge from dataset compatibility to matched checkpoint generation; closed-loop
LIBERO evaluation of these checkpoints remains pending.

### `suffix_strategy_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_strategy.yaml --out runs/suffix_strategy_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780312179_117921719.out
```

Mean results over five seeds:

| Method | Clean | Object RAUC | Median r80 | EE-transfer RAUC | RAUC AULC |
|---|---:|---:|---:|---:|---:|
| Uniform suffix | 0.8368 | 0.4854 | 0.5032 | 0.3081 | 0.3716 |
| BGR-Suffix | 0.8738 | 0.4730 | 0.4534 | 0.3029 | 0.3745 |
| BGR-Suffix Boundary | 0.8724 | 0.4519 | 0.4242 | 0.2872 | 0.3658 |
| BGR-Suffix Broad | 0.8700 | 0.4815 | 0.4704 | 0.3175 | 0.3798 |
| BGR-Suffix Hard | 0.8657 | 0.4749 | 0.4689 | 0.3292 | 0.3865 |

Interpretation: in the suffix simulator, narrow boundary-only radius sampling undercovers high-radius recovery. BGR-Suffix Broad nearly closes the final object-RAUC gap to uniform while preserving much higher clean success and improving transfer RAUC and AULC. BGR-Suffix Hard gives the best transfer and learning-curve area but lower final object RAUC.

### `suffix_strategy_pair_15seed_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_strategy_pair_15seed.yaml --out runs/suffix_strategy_pair_15seed_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780328104_484552023.out
```

Mean results over 15 paired seeds:

| Method | Clean | Object RAUC | Median r80 | EE-transfer RAUC | RAUC AULC |
|---|---:|---:|---:|---:|---:|
| BGR-Broad | 0.8691 | 0.4815 | 0.4703 | 0.3165 | 0.3790 |
| Uniform | 0.8358 | 0.4844 | 0.5046 | 0.3075 | 0.3707 |

Interpretation: the 15-seed paired extension supports the clean-success,
transfer, and sample-efficiency claims for BGR-Broad (p=0.00006), while
preserving the limitation that uniform remains slightly higher on final
object-perturbation RAUC.
