# Boundary-Guided Replay

This repository contains the anonymous AAAI-27 submission package for:

**Boundary-Guided Replay: Learning at the Success-Failure Boundary of Decision Policies**

The included artifact contains the reusable BGR core, versioned experiment
configs, per-seed result summaries, generated paper tables/figures, OpenVLA
audit scripts, environment snapshots, and a SHA-256 submission manifest. The
main evidence includes a 30-seed synthetic mechanism check, active-estimator
validation, a completed 30-seed
procedural grid-margin full-baseline comparison, a held-out grid replication, a
30-seed robot-suffix coverage comparison, a held-out suffix full-baseline
replication, and a held-out suffix BGR-vs-uniform replication.
OpenVLA/LIBERO results are included as recovery-curve, selection, and
data-plumbing audits rather than robotics fine-tuning claims.

The anonymous submission archive contains `submission_manifest.json` plus the
files it declares. Only those archive entries should be treated as the anonymous
submission artifact. The manifest hashes the declared payload files; the
manifest itself is packaged as the verifier entry point and is intentionally not
self-hashed. Mirrored raw Slurm logs are diagnostic run records and are not part
of the anonymous submission artifact.
Cluster commands below are provenance recipes; any remote input paths they
mention are not reviewer evidence unless their generated summaries are declared
in `submission_manifest.json`.

To assemble only those files after the package gate passes:

```bash
PYTHONPATH=src:. python3 scripts/check_submission_package.py --root . --write-submission-zip bgr-aaai27-anonymous.zip
```

## Reviewer Navigation

Start with `paper/main.pdf` for the anonymous manuscript, then use
`results/README.md#submission-evidence-index` for the evidence map. The primary
evidence is the 30-seed synthetic mechanism check, the active-estimator
validation, the 30-seed grid-margin
comparison, the held-out grid replication, the 30-seed robot-suffix coverage
comparison, the held-out suffix full-baseline replication, the held-out suffix
BGR-vs-uniform replication, and
`paper/figures/significance_tests.csv`. OpenVLA/LIBERO entries are scoped audits
and should not be read as robotics fine-tuning claims.

For the primary paired comparisons, competing methods share experiment configs,
replayable-state pools, evaluation radius grids, learner/update budgets, and
paired seeds; the intended intervention is the replay state/radius selection
rule. Baseline rows in the generated tables come from the same scripts and
summary artifacts listed in the evidence index.

## Claim-Evidence Map

| Paper claim | Primary artifact evidence | Verification hook |
| --- | --- | --- |
| Controlled synthetic recovery-margin training validates the intended BGR sampler before higher-cost runs. | `results/toy_30seed_v1/summary.csv`, `results/toy_15seed_v1/summary.csv`, `paper/figures/significance_tests.csv` | `scripts/check_paper_claims.py` checks the synthetic RAUC, AULC, clean-success, and sign-test prose; `scripts/check_submission_package.py` checks the 30-seed synthetic comparison. |
| Boundary-centered replay expands recovery margins in the main procedural setting. | `results/grid_margin_full_30seed_v1/summary.csv`, `results/grid_margin_full_replication_30seed_v1/summary.csv`, `paper/figures/grid_margin_full_table.tex`, `paper/figures/significance_tests.csv` | `scripts/check_paper_claims.py` checks the numeric prose; `scripts/check_submission_package.py` checks paired seeds, ledgers, generated tables, and manifest hashes. |
| Active boundary probing estimates useful critical radii at a small fixed rollout budget. | `paper/figures/estimator_stats.csv`, `paper/figures/estimator_table.tex`, `results/estimator_pair_30seed_v1/summary.csv`, `paper/figures/significance_tests.csv` | `scripts/check_paper_claims.py` checks the estimator prose; `scripts/check_submission_package.py` checks the generated estimator table and the 30-seed estimator confirmation. |
| Radius-level boundary sampling is the important BGR ablation in the grid-margin benchmark. | `results/grid_margin_ablation_30seed_v1/summary.csv`, `results/grid_margin_ablation_replication_30seed_v1/summary.csv`, `results/grid_margin_ablation_15seed_v1/summary.csv`, `paper/figures/grid_margin_ablation_table.tex`, `paper/figures/significance_tests.csv` | `scripts/check_paper_claims.py` checks the ablation prose against the original and held-out 30-seed summaries; `scripts/check_submission_package.py` checks the 30-seed mechanism confirmation and regenerates aggregate tables and significance artifacts exactly. |
| Coverage-aware BGR-Suffix is positive manipulation-style evidence but not a final robotics claim. | `results/suffix_coverage_full_30seed_v1/summary.csv`, `results/suffix_coverage_full_replication_30seed_v1/summary.csv`, `results/suffix_strategy_coverage_30seed_v1/summary.csv`, `results/suffix_strategy_coverage_replication_30seed_v1/summary.csv`, `paper/figures/significance_tests.csv` | `scripts/check_paper_claims.py` checks the full-baseline RAUC rows, clean, transfer, AULC, and median-r80 caveat prose. |
| The learned-policy OpenVLA/LIBERO path is an audit, not a robotics fine-tuning claim. | `results/libero_probe_v2/summary.csv`, `results/openvla_teacher_replay_manifest_v1/summary.json`, and the packaged OpenVLA-OFT audit summaries listed below. | `scripts/check_submission_package.py` enforces paper-facing audit wording and keeps paper-negative scale-diagnostic outputs out of the anonymous manifest. |

Grid-margin robustness/scope diagnostic artifacts:

- 30-seed radius-level ablation/table: `results/grid_margin_ablation_30seed_v1/summary.csv`, `paper/figures/grid_margin_ablation_table.tex`
- held-out 30-seed radius-level ablation replication: `results/grid_margin_ablation_replication_30seed_v1/summary.csv`
- original 15-seed radius-level ablation: `results/grid_margin_ablation_15seed_v1/summary.csv`
- learning-curve stats/source: `paper/figures/grid_margin_learning_curve_stats.csv`, `results/grid_margin_full_15seed_v1/results.json`
- 30-seed target-margin sweep table/source: `paper/figures/grid_margin_target_sensitivity_stats.csv`, `results/grid_margin_target_sensitivity_30seed_v1/summary.csv`; 15-seed provenance: `results/grid_margin_target_sensitivity_15seed_v1/summary.csv`
- 30-seed learning-rate sweep table/source: `paper/figures/grid_margin_learning_rate_sensitivity_stats.csv`, `results/grid_margin_learning_rate_sensitivity_30seed_v1/summary.csv`; 15-seed provenance: `results/grid_margin_learning_rate_sensitivity_15seed_v1/summary.csv`
- 30-seed regime sweep table/source: `paper/figures/grid_margin_regime_sensitivity_stats.csv`, `results/grid_margin_regime_sensitivity_30seed_v1/summary.csv`; 15-seed provenance: `results/grid_margin_regime_sensitivity_15seed_v1/summary.csv`
- 30-seed stress sweep table/source: `paper/figures/grid_margin_stress_sensitivity_stats.csv`, `results/grid_margin_stress_sensitivity_30seed_v1/summary.csv`; 15-seed provenance: `results/grid_margin_stress_sensitivity_15seed_v1/summary.csv`

OpenVLA-OFT packaged audit summaries:

- OpenVLA selection/audit stats: `paper/figures/openvla_stats.csv`
- OpenVLA recovery audit source: `results/libero_openvla_recovery_v1/summary.csv`
- OpenVLA selection audit source: `results/libero_openvla_boundary_selection_balanced_v1/aggregate.csv`
- official-checkpoint sanity audit: `results/openvla_oft_sanity_eval_sanity_v1/summary.csv`
- 1,000-step balanced2048 data-plumbing audit: `results/openvla_oft_eval_balanced2048_step1000_v1/summary.csv`
- p1024 clean adaptation audit: `results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv`
- p1024 original perturbation audit: `results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv`
- p1024 offset-3 perturbation audit: `results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/summary.csv`
- p2048 clean adaptation audit: `results/openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv`
- p2048 original perturbation audit: `results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv`
- p2048 offset-3 perturbation audit: `results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/summary.csv`
- p2048 10-trial perturbation variance audit: `results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_10trials_v1/summary.csv`
- p2048 full-goal clean identity audit: `results/openvla_oft_clean_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1/summary.csv`
- p2048 full-goal visual perturbation audit: `results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1/summary.csv`

## Repository Layout

```text
src/bgr/                 Core BGR data structures, estimators, metrics, and samplers
scripts/                 Experiment and plotting entry points
configs/                 Versioned experiment configs
tests/                   Unit tests for estimator/priority/metrics behavior
paper/                   AAAI-27 manuscript, generated tables/figures, and official AuthorKit27
```

## Verification Commands

```bash
python3 -m pip install -e .
PYTHONPATH=src:. python3 -m unittest discover -s tests
PYTHONPATH=src:. python3 scripts/check_paper_claims.py --paper paper/main.tex --results-dir results --figures-dir paper/figures
PYTHONPATH=src:. python3 scripts/check_submission_package.py --root .
```

## Reproducibility Metadata

The repository is MIT licensed. Runtime environment snapshots can be collected
on the cluster and checked against `results/environment_v1`:

The included submission artifact records required-file integrity in
`submission_manifest.json`. The package gate verifies rendered PDFs, PDF
metadata hygiene, required artifacts, generated table synchronization,
double-blind hygiene, README framing, and the SHA-256 manifest with:

```bash
PYTHONPATH=src:. python3 scripts/check_submission_package.py --root .
```

After an intentional update to any required artifact, regenerate the manifest
before rerunning the package gate:

```bash
PYTHONPATH=src:. python3 scripts/check_submission_package.py --root . --write-required-manifest
```

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/collect_environment.py --out runs/environment_v1/compute_environment.json
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/collect_environment.py --out runs/environment_v1/compute_environment.json

~/remote_srun.sh --dry-run --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:10:00 /work/anonymous/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/collect_environment.py --out runs/environment_v1/gpu_environment.json
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:10:00 /work/anonymous/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/collect_environment.py --out runs/environment_v1/gpu_environment.json
```

## Tier-0 Experiment

Heavy or repeated experiments should run on the cluster through `~/remote_srun.sh`; use `/work/anonymous/bgr` as the remote project directory to avoid home-directory disk pressure.

Local smoke run after `python3 -m pip install -e .`:

```bash
PYTHONPATH=src:. python3 scripts/run_toy_experiment.py --config configs/toy_smoke.yaml --out runs/toy_smoke
```

Dry run:

```bash
~/remote_srun.sh --dry-run /work/anonymous/bgr python scripts/run_toy_experiment.py --config configs/toy_bgr_15seed.yaml --out runs/toy_15seed_v1
```

Real run:

```bash
~/remote_srun.sh --github-test --git-pull --log /work/anonymous/bgr python scripts/run_toy_experiment.py --config configs/toy_bgr_15seed.yaml --out runs/toy_15seed_v1
```

## Active Estimator Validation

This run isolates the recovery-curve estimator from policy training by comparing fixed-budget probes against dense reference curves.

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 4 --mem 12G --time 01:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_estimator_experiment.py --config configs/estimator_bgr_full.yaml --out runs/estimator_full_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 01:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_estimator_experiment.py --config configs/estimator_bgr_full.yaml --out runs/estimator_full_v1
```

## LIBERO Simulator Probe

The cluster has LIBERO and robosuite available. This probe validates resettable LIBERO task states and object-pose perturbations on GPU/EGL; it is not a policy-success experiment.

```bash
~/remote_srun.sh --dry-run --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 01:00:00 /work/anonymous/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/probe_libero_suffix_states.py --suite libero_goal --task-ids 0,1,2,3,4 --init-state-ids 0,1,2 --radii 0.0,0.25,0.5,0.75,1.0 --trials-per-radius 4 --settle-steps 5 --image-size 64 --out runs/libero_probe_v2
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 01:00:00 /work/anonymous/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/probe_libero_suffix_states.py --suite libero_goal --task-ids 0,1,2,3,4 --init-state-ids 0,1,2 --radii 0.0,0.25,0.5,0.75,1.0 --trials-per-radius 4 --settle-steps 5 --image-size 64 --out runs/libero_probe_v2
```

Existing closed-loop OpenVLA/LIBERO object-task rollouts can be converted into
BGR-style recovery curves:

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/summarize_libero_openvla_recovery.py --input-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_object3_h220_bash --out runs/libero_openvla_recovery_v1 --source-name libero_openvla_observation_object3_h220_bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/summarize_libero_openvla_recovery.py --input-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_object3_h220_bash --out runs/libero_openvla_recovery_v1 --source-name libero_openvla_observation_object3_h220_bash
```

Existing OpenVLA perturbation-selection artifacts can also be summarized as a
boundary-discovery diagnostic:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/summarize_openvla_boundary_selection.py --proposal-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_guided_h160 --proposal-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_guided_seed2_h160 --proposal-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_guided_seed3_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed1b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed2b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed3b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed4b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed5b_skip_lp2_h160 --out runs/libero_openvla_boundary_selection_v1
```

## Robot Suffix Strategy Comparison

This diagnostic compares BGR-Suffix radius distributions while keeping the same replay-state estimator.

```bash
~/remote_srun.sh --dry-run --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_full_15seed.yaml --out runs/suffix_full_15seed_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_full_15seed.yaml --out runs/suffix_full_15seed_v1
~/remote_srun.sh --dry-run --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_strategy_coverage_15seed.yaml --out runs/suffix_strategy_coverage_15seed_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_strategy_coverage_15seed.yaml --out runs/suffix_strategy_coverage_15seed_v1
~/remote_srun.sh --dry-run --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_strategy_coverage_30seed.yaml --out runs/suffix_strategy_coverage_30seed_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_strategy_coverage_30seed.yaml --out runs/suffix_strategy_coverage_30seed_v1
~/remote_srun.sh --dry-run --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_strategy.yaml --out runs/suffix_strategy_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_strategy.yaml --out runs/suffix_strategy_v1
```

## Procedural Grid Recovery

The grid benchmarks are dependency-light procedural decision benchmarks with generated obstacle maps, replayable mid-path states, Manhattan-radius perturbations, and an exact shortest-path feasibility witness.

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 2 --mem 8G --time 01:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_bgr.yaml --out runs/grid_fast
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 01:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_bgr.yaml --out runs/grid_fast
```

The positive procedural benchmark is `grid_margin_bgr`, which evaluates state-conditioned margin expansion on grid-backed replay states. The completed 30-seed full-baseline config is the primary grid comparison; the stored learning-curve history remains 15-seed, while the rendered ablation and sensitivity tables use the packaged 30-seed confirmations:

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_bgr_full.yaml --out runs/grid_margin_full
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_bgr_full.yaml --out runs/grid_margin_full
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_full_15seed.yaml --out runs/grid_margin_full_15seed_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_full_15seed.yaml --out runs/grid_margin_full_15seed_v1
PYTHONPATH=src:. python scripts/run_grid_margin_trial.py --config configs/grid_margin_full_30seed.yaml --out results/grid_margin_full_30seed_v1 --method bgr --seed 0
PYTHONPATH=src:. python scripts/merge_grid_margin_trials.py --config configs/grid_margin_full_30seed.yaml --out results/grid_margin_full_30seed_v1
```

The target-margin sensitivity sweep checks whether the grid-margin BGR result is tied to the reported `target_margin=0.38` setting:

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_target_sensitivity.py --config configs/grid_margin_target_sensitivity_15seed.yaml --out runs/grid_margin_target_sensitivity_15seed_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_target_sensitivity.py --config configs/grid_margin_target_sensitivity_15seed.yaml --out runs/grid_margin_target_sensitivity_15seed_v1
```

A learning-rate sensitivity sweep is retained as a scope diagnostic. It tests the same paired BGR/uniform setup at `learning_rate` values 0.015, 0.03, and 0.06:

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_learning_rate_sensitivity.py --config configs/grid_margin_learning_rate_sensitivity_15seed.yaml --out runs/grid_margin_learning_rate_sensitivity_15seed_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_learning_rate_sensitivity.py --config configs/grid_margin_learning_rate_sensitivity_15seed.yaml --out runs/grid_margin_learning_rate_sensitivity_15seed_v1
```

Grid-margin ablations isolate BGR priority terms and boundary-centered radius sampling:

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_ablation_15seed.yaml --out runs/grid_margin_ablation_15seed_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_ablation_15seed.yaml --out runs/grid_margin_ablation_15seed_v1
```

A grid-regime sensitivity runner is retained as a diagnostic. The packaged
`obstacle_prob`/`max_offset` sweep mostly reproduces the nominal margin dynamics,
so it should not be treated as separate robustness evidence without stronger
regime changes. The rendered table uses the 30-seed regime diagnostic
source, with 30/0 paired wins for BGR on final RAUC, RAUC AULC, clean
success, and median r80 in each tested regime:

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_regime_sensitivity.py --config configs/grid_margin_regime_sensitivity_15seed.yaml --out runs/grid_margin_regime_sensitivity_15seed_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_regime_sensitivity.py --config configs/grid_margin_regime_sensitivity_15seed.yaml --out runs/grid_margin_regime_sensitivity_15seed_v1
PYTHONPATH=src:. python3 scripts/run_grid_margin_regime_sensitivity.py --config configs/grid_margin_regime_sensitivity_30seed.yaml --out results/grid_margin_regime_sensitivity_30seed_v1
```

The grid-margin stress sweep changes the latent recovery geometry rather than
only the obstacle layout. It tests sharp low-margin states, diffuse recovery
boundaries, and lower feasible-radius mass:

```bash
~/remote_srun.sh --dry-run --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_stress_sensitivity.py --config configs/grid_margin_stress_sensitivity_15seed.yaml --out runs/grid_margin_stress_sensitivity_15seed_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_stress_sensitivity.py --config configs/grid_margin_stress_sensitivity_15seed.yaml --out runs/grid_margin_stress_sensitivity_15seed_v1
```

The artifact `results/grid_margin_stress_sensitivity_15seed_v1` run is
positive across all three stress cases: BGR final RAUC is 0.362--0.457 versus
uniform 0.324--0.430, and all RAUC/AULC paired sign tests are 15/0
($p=0.0001$).

The tabular grid-policy configs are retained as negative diagnostics:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 00:45:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_policy_mixed.yaml --out runs/grid_policy_mixed_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 00:45:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_policy_coverage.yaml --out runs/grid_policy_coverage_v1
```

See [results/README.md](results/README.md) for the packaged run ledger. The original tabular grid policy benchmark is retained as a negative diagnostic because broad replay saturates it after clean pretraining.

## Result Aggregation and Paper Figures

```bash
python3 scripts/aggregate_results.py --results-dir results --out-dir paper/figures
python3 scripts/analyze_significance.py --results-dir results --out-csv paper/figures/significance_tests.csv --out-tex paper/figures/significance_table.tex
PYTHONPATH=src:. python3 scripts/check_paper_claims.py --paper paper/main.tex --results-dir results --figures-dir paper/figures
```

This writes CSV summaries, a LaTeX summary table, and bar-chart figures used by `paper/main.tex`.
The claim check verifies the headline numeric prose against the generated CSV/JSON artifacts.

## OpenVLA/LIBERO Audit Summary

Detailed OpenVLA commands, Slurm IDs, copied artifacts, and summaries are in
`results/README.md`. The top-level claim is scoped: these runs audit
recovery curves, perturbation selection, and OpenVLA-OFT data plumbing rather
than robotics fine-tuning performance.

The packaged useful audit scale is p1024/p2048 clean-mix adaptation with official
training statistics, identity LoRA initialization, and low learning rate. At
p1024, BGR and matched random tie clean at 14/15; pooling the original and
offset-3 visual perturbation evals gives BGR 0.8550 vs. 0.8400 for random,
still trailing the unadapted official checkpoint at 0.8700. At p2048, the
full-goal identity audit gives 99/100 clean successes for BGR, matched random,
and the official checkpoint. The 10-task visual perturbation audit gives BGR
367/400 perturbed successes, tying official and trailing matched random by one
episode (368/400).

## AAAI Sources

The official AAAI-27 page lists the 2026 author-submission timetable for the
February 16--23, 2027 conference and links the AAAI-27 author kit: abstracts are
due July 21, 2026, full papers July 28, 2026, and supplementary material/code
July 31, 2026. The kit in `paper/AuthorKit27` was downloaded from
`https://aaai.org/authorkit27/` on 2026-06-01.
