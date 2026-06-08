# Experiment Results Ledger

All heavy runs are launched through `~/remote_srun.sh` and write outputs under
`/work/anonymous/bgr/runs`.

## Submission Evidence Index

Primary controlled evidence for the paper is:

- `results/toy_30seed_v1/summary.csv`: 30 paired seeds for the controlled
  synthetic recovery-margin benchmark.
- `results/toy_15seed_v1/summary.csv`: original 15 paired-seed synthetic
  mechanism run retained for provenance.
- `results/grid_margin_full_30seed_v1/summary.csv`: 30 paired seeds for the
  procedural grid-margin full-baseline comparison.
- `results/grid_margin_full_replication_30seed_v1/summary.csv`: held-out seeds
  30--59 replication for the grid-margin BGR-vs-uniform comparison.
- `results/suffix_coverage_full_30seed_v1/summary.csv`: 30 paired seeds for the
  coverage-aware robot-suffix full-baseline comparison against clean-only,
  fixed-radius, failure-only, loss-priority, and uniform replay.
- `results/suffix_coverage_full_replication_30seed_v1/summary.csv`: held-out
  seeds 30--59 suffix full-baseline replication.
- `results/suffix_strategy_coverage_30seed_v1/summary.csv`: 30 paired seeds for
  the coverage-aware robot-suffix comparison.
- `results/suffix_strategy_coverage_replication_30seed_v1/summary.csv`: held-out
  seeds 30--59 replication for the robot-suffix coverage comparison.
- `results/suffix_strategy_ablation_30seed_v1/summary.csv`: 30-seed suffix
  strategy ablation explaining why BGR-Coverage is the promoted suffix variant.
- `results/suffix_stress_sensitivity_30seed_v1/summary.csv`: 30-seed
  suffix stress sweep over low-teacher-quality, high-clutter, tight-feasibility,
  and diffuse-boundary regimes.
- `paper/figures/suffix_stress_sensitivity_stats.csv`: generated suffix stress
  table source used by the paper-facing claim checks.
- `paper/figures/significance_tests.csv`: paired exact sign tests for the
  central claims, ablations, sensitivity sweeps, and replications.
- `paper/figures/estimator_stats.csv` and `paper/figures/estimator_table.tex`:
  method-validation evidence that active boundary probing estimates useful
  critical radii at a small fixed rollout budget.
- `results/estimator_pair_30seed_v1/summary.csv`: 30-seed active-estimator
  confirmation.

Packaged grid-margin robustness/scope diagnostics are:

- `results/grid_margin_ablation_30seed_v1/summary.csv`: 30-seed radius-level
  ablation showing that boundary-radius sampling is the important BGR component.
- `results/grid_margin_ablation_replication_30seed_v1/summary.csv`: held-out
  seeds 30--59 replication of the radius-level ablation.
- `results/grid_margin_ablation_15seed_v1/summary.csv`: original 15-seed
  radius-level ablation retained for provenance.
- `paper/figures/grid_margin_learning_curve_stats.csv`,
  `paper/figures/grid_margin_learning_curve.pdf`, and
  `results/grid_margin_full_15seed_v1/results.json`: stored 15-seed
  learning-curve history, generated stats, and paper figure.
- `paper/figures/grid_margin_target_sensitivity_stats.csv` and
  `results/grid_margin_target_sensitivity_30seed_v1/summary.csv`: 30-seed
  target-margin table/source; `results/grid_margin_target_sensitivity_15seed_v1/summary.csv`
  is retained for provenance.
- `paper/figures/grid_margin_learning_rate_sensitivity_stats.csv` and
  `results/grid_margin_learning_rate_sensitivity_30seed_v1/summary.csv`:
  30-seed learning-rate table/source; `results/grid_margin_learning_rate_sensitivity_15seed_v1/summary.csv`
  is retained for provenance.
- `paper/figures/grid_margin_regime_sensitivity_stats.csv` and
  `results/grid_margin_regime_sensitivity_30seed_v1/summary.csv`: 30-seed
  regime table/source; `results/grid_margin_regime_sensitivity_15seed_v1/summary.csv`
  is retained for provenance.
- `paper/figures/grid_margin_stress_sensitivity_stats.csv` and
  `results/grid_margin_stress_sensitivity_30seed_v1/summary.csv`: 30-seed
  geometry-stress table/source; `results/grid_margin_stress_sensitivity_15seed_v1/summary.csv`
  is retained for provenance.
- `results/grid_margin_witness_sensitivity_30seed_v1/summary.csv`: 30-seed
  witness sensitivity diagnostic for the grid-margin boundary sampler. Exact
  witness acceptance has valid accepted samples at 1.0000; symmetric 10% and
  20% witness noise mainly reduces true-boundary recall rather than admitting
  invalid samples under this controlled boundary-candidate stream.

Secondary diagnostics are included to scope the claim rather than expand it.
OpenVLA/LIBERO rows are recovery-curve, selection, and data-plumbing audits; the
paper does not claim a stable OpenVLA fine-tuning gain over the official
checkpoint.

Internal pre-existing-dataset route scout:

- `results/openml_margin_scout_v0/summary.csv`: fixed 4-seed OpenML margin
  replay scout, run with
  `PYTHONPATH=src:. python3 tools/openml_margin_scout.py --out results/openml_margin_scout_v0`.
  This is not submission evidence. It uses OpenML version-1 ionosphere, sonar,
  diabetes, and spambase through `sklearn.datasets.fetch_openml`, median
  imputation, label encoding, standardized feature-space fixed-L2
  perturbations, and an online `SGDClassifier`. Diabetes at target radius 2.0
  clears only the scout gate: BGR reaches 0.7402 final RAUC versus uniform
  0.6797 (W/L/T=4/0/0), while fixed-radius replay is 0.6999. The fixed
  preregistered follow-up command is
  `PYTHONPATH=src:. python3 tools/openml_margin_scout.py --datasets diabetes --targets 2.0 --seeds 30 --out results/openml_diabetes_margin_30seed_v1`.
- `results/openml_margin_scout_v0/per_seed.csv` and
  `results/openml_margin_scout_v0/package_versions.json`: compact per-seed and
  package-version records for the same scout (`scikit-learn==1.8.0`,
  `numpy==2.4.2`, Python 3.14.3).

Completed external-package scope diagnostic:

- `results/minatar_breakout_recovery_calibration_12seed_v1/summary.json`:
  fixed MinAtar Breakout pre-method calibration, run with
  `PYTHONPATH=src:. /tmp/bgr_minatar_venv/bin/python tools/minatar_breakout_recovery_calibration.py --out results/minatar_breakout_recovery_calibration_12seed_v1`.
  This route uses MinAtar's package-owned Breakout dynamics, a fixed
  paddle-tracking controller, and signed paddle-cell offsets after a burn-in
  checkpoint. It clears the pre-method gate with clean success 1.0000,
  recovery range 0.6667--1.0000, RAUC 0.7000, and r80 0.6000. Treat it as
  permission for the fixed all-method screen below, not as BGR evidence.
- `results/minatar_breakout_recovery_calibration_12seed_v1/package_versions.json`
  and `results/minatar_breakout_recovery_calibration_12seed_v1/recovery_rows.csv`:
  compact package-version and row-level records for the same calibration
  (`MinAtar==1.0.15`, `numpy==2.4.6`).
- `results/minatar_breakout_recovery_probe_4seed_v1/summary.csv`: fixed
  MinAtar Breakout all-method screen, run with
  `PYTHONPATH=src:. /tmp/bgr_minatar_venv/bin/python tools/minatar_breakout_recovery_probe.py --out results/minatar_breakout_recovery_probe_4seed_v1`.
  This route is negative and should not be scaled or promoted: default BGR and
  BGR-Coverage both tie uniform at 0.8896 final RAUC with W/L/T=0/0/4, median
  r80 saturates at 5.0000, and failure-only has the best AULC at 0.7721.
- `results/minatar_breakout_recovery_probe_4seed_v1/aggregate.csv`,
  `results/minatar_breakout_recovery_probe_4seed_v1/history.csv`, and
  `results/minatar_breakout_recovery_probe_4seed_v1/package_versions.json`:
  compact aggregate, learning-history, and package-version records for the same
  negative scope diagnostic.

Completed external-package pre-method calibration:

- `results/handreach_recovery_calibration_8seed_v1/summary.json`: fixed
  Gymnasium-Robotics HandReach-v3 ShadowHand calibration, run with
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/handreach_recovery_calibration.py --out results/handreach_recovery_calibration_8seed_v1`.
  This route is rejected before method comparison: clean success is 0.0000,
  recovery is flat at 0.0000 across the 0.00--0.20 joint-perturbation grid,
  RAUC is 0.0000, and r80 is 0.2000. The package stack is
  `gymnasium==1.3.0`, `gymnasium_robotics==1.4.2`, `mujoco==3.9.0`, and
  `numpy==2.4.6`. Do not promote or scale this route without a new
  preregistered controller that first clears clean-success and non-flat
  recovery prerequisites.
- `results/handreach_recovery_calibration_8seed_v1/package_versions.json` and
  `results/handreach_recovery_calibration_8seed_v1/recovery_rows.csv`: compact
  package-version and row-level records for the same rejected calibration.

Completed external-package scope diagnostic:

- `results/minigrid_fourrooms_recovery_probe_maxr10_4seed_v1/summary.csv`:
  fixed MiniGrid FourRooms measurement-window follow-up, run with
  `PYTHONPATH=src:. /tmp/bgr_minigrid_venv/bin/python tools/minigrid_fourrooms_recovery_probe.py --out results/minigrid_fourrooms_recovery_probe_maxr10_4seed_v1 --max-radius 10`.
  This route is negative and should not be promoted: BGR-Coverage reaches
  0.1031 final RAUC vs. 0.1014 for uniform (+0.0017, W/L/T=2/2/0), barely
  exceeds BGR-uniform-radius at 0.0967, and median r80 remains saturated at
  1.0000 for both BGR-Coverage and uniform.
- `results/minigrid_fourrooms_recovery_probe_maxr10_4seed_v1/package_versions.json`:
  compact package-version record for the same scope diagnostic
  (`gymnasium==1.3.0`, `minigrid==3.1.0`).

Completed external-package pre-promotion route:

- `results/bsuite_deepsea_recovery_probe_4seed_v1/summary.csv`: fixed bsuite
  DeepSea 4-seed screen, preregistered before method outcomes on 2026-06-07.
  Run with
  `PYTHONPATH=src:. /tmp/bgr_bsuite_venv/bin/python tools/bsuite_deepsea_recovery_probe.py --out results/bsuite_deepsea_recovery_probe_4seed_v1`.
  The route uses `bsuite==0.3.6` in an isolated temporary environment,
  package-owned randomized DeepSea action mappings, exact restart states, and a
  fixed left-column perturbation family. The screen is negative: default BGR
  reaches 0.1125 final RAUC vs. 0.0844 for uniform, but wins only 2/4 paired
  seeds, trails the state-priority/uniform-radius ablation at 0.1266, and has
  lower median r80 than uniform. BGR-Coverage ties uniform on mean RAUC. Do not
  scale or promote this route without a genuinely new preregistered premise.
- `results/bsuite_deepsea_recovery_probe_4seed_v1/package_versions.json`:
  compact package-version record for the same screen.

Completed external-package pre-promotion route:

- `results/bsuite_catch_recovery_probe_4seed_v1/summary.csv`: fixed bsuite
  Catch 4-seed screen, preregistered before method outcomes on 2026-06-07.
  Run with
  `PYTHONPATH=src:. /tmp/bgr_bsuite_venv/bin/python tools/bsuite_catch_recovery_probe.py --out results/bsuite_catch_recovery_probe_4seed_v1`.
  The route uses `bsuite==0.3.6` in an isolated temporary environment,
  package-owned Catch dynamics, exact restart fields, and a fixed paddle-column
  perturbation family. The 4-seed screen passed the scale-up gate for default
  BGR: BGR reaches 0.9742 final RAUC vs. uniform 0.8388 (+0.1354, 4/0/0),
  fixed-radius 0.7767, failure-only 0.9336, TD-loss 0.7140, and
  BGR-uniform-radius 0.8982, with non-contradictory median r80. This was only
  permission to run the fixed 30-seed promotion screen.
- `results/bsuite_catch_recovery_probe_4seed_v1/package_versions.json`:
  compact package-version record for the same screen.
- `results/bsuite_catch_recovery_probe_30seed_v1/summary.csv`: fixed bsuite
  Catch 30-seed promotion screen from the preregistered command below. The
  scale-up is negative: default BGR reaches 0.8446 final RAUC vs. 0.8782 for
  uniform (14/16/0), BGR-Coverage reaches 0.8452 vs. 0.8782 (13/17/0),
  failure-only reaches 0.9676, and BGR-uniform-radius reaches 0.8588. Median
  r80 also contradicts the boundary-improvement story: default BGR 0.8367 and
  BGR-Coverage 0.8608 trail uniform 0.9233. Do not promote this route without a
  genuinely new preregistered premise.
- `results/bsuite_catch_recovery_probe_30seed_v1/package_versions.json`:
  compact package-version record for the 30-seed screen.

Fixed 30-seed bsuite Catch promotion command:

```bash
PYTHONPATH=src:. /tmp/bgr_bsuite_venv/bin/python tools/bsuite_catch_recovery_probe.py --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 --out results/bsuite_catch_recovery_probe_30seed_v1
```

Completed external-package pre-promotion route:

- `tools/bsuite_mountaincar_recovery_probe.py`: fixed bsuite MountainCar screen,
  preregistered before method outcomes on 2026-06-07. The route uses
  `bsuite==0.3.6` in `/tmp/bgr_bsuite_venv`, instantiates bsuite's
  package-owned MountainCar task, and steps exact private restart fields during
  recovery rollouts. Replay states are right-moving MountainCar states; larger
  perturbation radii move starts back toward the low-energy valley anchor.
- `results/bsuite_mountaincar_recovery_probe_4seed_v1/summary.csv`: fixed
  4-seed bsuite MountainCar screen from the preregistered command below. The
  route is negative and should not be scaled: default BGR reaches 0.0532 final
  RAUC versus 0.0497 for uniform (+0.0036, 3/1/0), below the preregistered +0.01
  threshold; BGR-Coverage reaches 0.0553, below fixed-radius replay (0.1420),
  failure-only replay (0.0653), and BGR-uniform-radius (0.0558). Median r80 is
  saturated at 1.0000 for BGR, BGR-Coverage, BGR-uniform-radius, failure-only,
  TD-loss, and uniform, so the screen does not measure useful boundary
  expansion.
- `results/bsuite_mountaincar_recovery_probe_4seed_v1/package_versions.json`:
  compact package-version record for the same screen.

Fixed 4-seed bsuite MountainCar pre-promotion command:

```bash
PYTHONPATH=src:. /tmp/bgr_bsuite_venv/bin/python tools/bsuite_mountaincar_recovery_probe.py --out results/bsuite_mountaincar_recovery_probe_4seed_v1
```

Completed external-package pre-promotion route:

- `tools/bsuite_cartpole_recovery_probe.py`: fixed bsuite Cartpole screen,
  preregistered before method outcomes on 2026-06-07. The route uses
  `bsuite==0.3.6` in `/tmp/bgr_bsuite_venv`, instantiates bsuite's
  package-owned three-action Cartpole task, and steps exact `CartpoleState`
  restarts with the package `step_cartpole` dynamics. Replay states are
  near-upright cart-pole states with nonzero cart, pole-angle, and velocity
  errors; larger radii increase bounded cart-position, cart-velocity,
  pole-angle, and pole-velocity perturbations while preserving bsuite's height
  and cart-position feasibility limits.
- `results/bsuite_cartpole_recovery_probe_4seed_v1/summary.csv`: fixed 4-seed
  bsuite Cartpole screen from the preregistered command below. The route is
  negative and should not be scaled: default BGR reaches 0.7464 final RAUC
  versus 0.7577 for uniform (-0.0113, 0/4/0), while BGR-Coverage reaches
  0.7559 versus uniform (-0.0018, 2/2/0). Both trail TD-loss replay (0.7694)
  and fixed-radius replay (0.7604); default BGR also trails the
  state-priority/uniform-radius ablation (0.7551). Median r80 is lower for
  default BGR than uniform (0.9667 vs. 0.9875), while BGR-Coverage is 0.9917 in
  a near-ceiling regime.
- `results/bsuite_cartpole_recovery_probe_4seed_v1/package_versions.json`:
  compact package-version record for the same screen.

Fixed 4-seed bsuite Cartpole pre-promotion command:

```bash
PYTHONPATH=src:. /tmp/bgr_bsuite_venv/bin/python tools/bsuite_cartpole_recovery_probe.py --out results/bsuite_cartpole_recovery_probe_4seed_v1
```

Promotion gate: this 4-seed screen can only justify a 30-seed scale-up if
default BGR or BGR-Coverage beats uniform, fixed-radius, failure-only,
TD/loss-priority, and BGR-uniform-radius on final RAUC with at least 3/4 paired
wins over uniform, a mean RAUC gap of at least +0.01, and non-contradictory,
non-saturated median-r80 evidence. It is not paper evidence.

Packaged FrozenLake diagnostic:

- `results/frozenlake_recovery_focused_30seed_v1/summary.csv` and
  `results/frozenlake_recovery_focused_30seed_v1/results.json`: 30 paired seeds
  on the canonical Gym FrozenLake8x8-v1 slippery map using
  `configs/frozenlake_recovery_focused_30seed.yaml`. This was run as an
  attempted independent standard-environment confirmation. It is included in
  the paper as a limitation, not as support for the method, because it does not
  give a clean BGR win: BGR gives final RAUC 0.5453 vs. 0.5312 for uniform and
  clean success 0.5581 vs. 0.5461, but the paired signs are 14/16 wins/losses on
  RAUC and 13/17 on clean success, median r80 is lower for BGR (0.7967 vs.
  0.8091), best RAUC is lower (0.6491 vs. 0.6565), and failure-only replay is
  stronger on final RAUC, median r80, AULC, and best RAUC in this run.
- `results/reacher_recovery_probe_12seed_v1/summary.csv`: fixed 12-seed
  Gymnasium MuJoCo Reacher-v5 all-method screen. It is included as a limitation,
  not support, because uniform replay has the highest final RAUC (0.3862), while
  BGR reaches 0.2907 with 4/8/0 paired wins/losses/ties against uniform and
  BGR-Coverage reaches 0.2721.

Packaged OpenVLA audit artifacts are:

- `results/libero_probe_v2/summary.csv`: resettable LIBERO radius-probe audit.
- `paper/figures/openvla_stats.csv`: OpenVLA recovery and selection audit stats.
- `results/libero_openvla_recovery_v1/summary.csv`: source recovery summary
  used to generate the OpenVLA audit stats.
- `results/libero_openvla_boundary_selection_balanced_v1/aggregate.csv`:
  source selection summary used to generate the OpenVLA audit stats.
- `results/openvla_teacher_replay_manifest_v1/summary.json`: teacher-replay
  data-plumbing audit.
- `results/openvla_action_tfds_validation_v1/summary.json`: compact
  action-label/TFDS plumbing audit validating 2,048-transition matched
  BGR/random exports with 7D actions, 8D state, stock loader ingestion, and
  matched 10-step checkpoint smokes.
- `results/openvla_oft_sanity_eval_sanity_v1/summary.csv`: official-checkpoint
  sanity audit.
- `results/openvla_oft_eval_balanced2048_step1000_v1/summary.csv`: 1,000-step
  balanced2048 data-plumbing audit.
- `results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv`:
  p1024 clean adaptation audit.
- `results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv`:
  p1024 original perturbation audit.
- `results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/summary.csv`:
  p1024 offset-3 perturbation audit.
- `results/openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv`:
  p2048 clean adaptation audit.
- `results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv`:
  p2048 original perturbation audit.
- `results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/summary.csv`:
  p2048 offset-3 perturbation audit.
- `results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_10trials_v1/summary.csv`:
  p2048 10-trial perturbation variance audit.
- `results/openvla_oft_clean_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1/summary.csv`:
  p2048 full-goal clean identity audit.
- `results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1/summary.csv`:
  p2048 full-goal visual perturbation audit.
- `results/openvla_oft_perturb_eval_cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1/summary.csv`:
  p2048 300-step image-augmentation continuation audit.
- `results/openvla_oft_perturb_eval_cleanmix_p2048_step51000_lr1em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1/summary.csv`:
  p2048 1,000-step low-LR image-augmentation continuation audit.
- `results/openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv`:
  completed weighted p2048unique perturbation audit. BGR is 367/400 on
  non-identity perturbations, tied with the official checkpoint at 367/400 and
  below matched random at 370/400, so the preregistered learned-policy
  promotion gate fails.
- `results/openvla_oft_perturb_eval_p2048unique_perturbonly_anchor_prereg_perturbonly_proxanchor_l2_5em0_step50300_lr2em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv`:
  completed perturb-only anchored p2048unique audit. Identity is 99/100 for
  BGR, official, and matched random; non-identity perturbation success is
  BGR 371/400, official 367/400, and matched random 372/400, so the
  preregistered learned-policy promotion gate fails.

The p4096 and common-availability sections below are retained as paper-negative
diagnostics in this ledger only. Their summary CSVs are not part of the
anonymous submission manifest or archive.

The detailed sections below are a historical provenance ledger. Reviewer-facing
evidence should be read from the index above and the anonymous manuscript; later
diagnostic sections are included for auditability.
Older troubleshooting sections may retain labels such as Queued command to
record original Slurm submissions; those labels are provenance, not active
experiment status.

## Internal FetchReach Goal-Recovery Calibration

This began as a pre-comparison calibration for an independent-benchmark route,
not paper evidence. It uses Gymnasium-Robotics FetchReach-v4 from the existing
isolated robotics environment (`gymnasium-robotics==1.4.2`,
`gymnasium==1.3.0`, `mujoco==3.9.0`) and keeps repo runtime dependencies
unchanged. The route differs from the failed MiniGrid and PointMaze screens:
replay states are package-sampled Fetch goals, perturbations are clipped 3D
goal offsets inside the package target range, and evaluation uses MuJoCo Fetch
dynamics. Both the default and hard-budget method screens are now completed
negative, so this route remains internal scope evidence only.

Calibration command:

```bash
PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/fetchreach_goal_recovery_calibration.py --out results/fetchreach_goal_recovery_calibration_gain2_h14_v1 --seeds 2 --replay-states 4 --trials 8 --horizon 14 --controller-gain 2.0 --direction-jitter 0.15
```

The selected calibration gives clean success 0.3750, RAUC 0.1969, median r80
0.0395, and recovery range 0.0625--0.3750. Compact artifacts:

- `results/fetchreach_goal_recovery_calibration_gain2_h14_v1/summary.json`
- `results/fetchreach_goal_recovery_calibration_gain2_h14_v1/recovery_rows.csv`
- `results/fetchreach_goal_recovery_calibration_gain2_h14_v1/package_versions.json`

This only establishes a non-saturated package-backed recovery curve. A full BGR
comparison must be preregistered before method results and must beat uniform,
fixed-radius, failure-only, loss-priority, and the state-priority/uniform-radius
ablation on final RAUC with non-contradictory median-r80 before any paper
promotion.

The full comparison tool is now fixed before method results at
`tools/fetchreach_goal_recovery_probe.py`. It uses a learned linear FetchReach
goal controller initialized at the calibrated weak setting and applies
teacher-action updates selected by the replay method. The preregistered 4-seed
command is:

```bash
PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/fetchreach_goal_recovery_probe.py --out results/fetchreach_goal_recovery_probe_4seed_v1
```

The screen compares uniform, fixed-radius, failure-only, TD-loss,
BGR-uniform-radius, BGR-Coverage, and default BGR. Do not scale or promote it
unless default BGR or BGR-Coverage beats every listed baseline on final RAUC,
wins at least 3/4 paired seeds against uniform, has a visible mean gap, and has
non-contradictory median r80.

The completed fixed 4-seed screen is negative and saturated after training:
failure-only reaches 0.9563 final RAUC, uniform and fixed-radius reach 0.9437,
TD-loss reaches 0.9375, BGR-uniform-radius reaches 0.9062, BGR-Coverage reaches
0.9000, and default BGR reaches 0.8938. Clean success is 1.0000 for every
method, and median r80 is saturated at the evaluation maximum 0.1500 for every
method. Compact artifacts:

- `results/fetchreach_goal_recovery_probe_4seed_v1/summary.csv`
- `results/fetchreach_goal_recovery_probe_4seed_v1/package_versions.json`

Keep this as an internal negative independent-benchmark screen; do not add it
to the paper unless the limitations table is expanded.

A hard-budget follow-up was calibrated before any new method comparison by
running uniform replay only. The weaker-controller probes with `--horizon 10`
and `--init-gain 1.2` collapsed to clean success 0.0000 and are rejected. The
fixed usable uniform-only calibration command is:

```bash
PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/fetchreach_goal_recovery_probe.py --out results/fetchreach_goal_recovery_hard_uniform_calibration_4seed_v1 --methods uniform --seeds 0,1,2,3 --iterations 20 --eval-every 10 --train-batch-size 2 --horizon 14 --init-gain 2.0 --init-noise 0.04 --teacher-gain 4.0 --learning-rate 0.20 --eval-trials 4 --record-trials 2 --quick-trials 2
```

This calibration gives uniform-only final clean 0.9375, final RAUC 0.6813, and
median r80 0.1185 over four seeds. It is not method evidence; it only fixes a
non-saturated measurement budget before comparison. Compact artifacts:

- `results/fetchreach_goal_recovery_hard_uniform_calibration_4seed_v1/summary.csv`
- `results/fetchreach_goal_recovery_hard_uniform_calibration_4seed_v1/package_versions.json`

The preregistered all-method hard-budget comparison command is:

```bash
PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/fetchreach_goal_recovery_probe.py --out results/fetchreach_goal_recovery_hard_probe_4seed_v1 --methods uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr --seeds 0,1,2,3 --iterations 20 --eval-every 10 --train-batch-size 2 --horizon 14 --init-gain 2.0 --init-noise 0.04 --teacher-gain 4.0 --learning-rate 0.20 --eval-trials 4 --record-trials 2 --quick-trials 2
```

Do not edit this command after seeing method results. Do not scale or promote
unless BGR or BGR-Coverage beats uniform, fixed-radius, failure-only, TD-loss,
and BGR-uniform-radius on final RAUC with at least 3/4 paired wins over
uniform, a visible mean gap, and non-contradictory non-saturated median r80.

The completed hard-budget screen is negative and should not be scaled: final
RAUC is 0.9437 for TD-loss, 0.8250 for failure-only, 0.6813 for uniform,
0.6438 for fixed-radius, 0.5188 for BGR-uniform-radius, 0.4625 for
BGR-Coverage, and 0.4438 for default BGR. BGR-Coverage loses to uniform
with W/L/T=(1,3,0), loses to failure-only and TD-loss on all four seeds, and
trails the state-priority/uniform-radius ablation. Compact artifacts:

- `results/fetchreach_goal_recovery_hard_probe_4seed_v1/summary.csv`
- `results/fetchreach_goal_recovery_hard_probe_4seed_v1/package_versions.json`

## Internal Fetch Object-Goal Calibrations

These are pre-method calibrations for harder Gymnasium-Robotics Fetch object
tasks. They use the exact reset-state and object-goal perturbation interface in
`tools/fetch_object_goal_recovery_calibration.py`; they are not BGR method
comparisons and should not be promoted into the paper.

FetchPush-v4 command:

```bash
PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/fetch_object_goal_recovery_calibration.py --out results/fetchpush_object_goal_calibration_2seed_v1 --env-id FetchPush-v4 --seeds 2 --replay-states 4 --trials 2 --radii 0.00,0.02,0.04,0.06,0.08,0.12 --horizon 80 --controller-gain 6.0 --direction-jitter 0.10
```

FetchPush is rejected as a method route: clean success is 0.2500, recovery is
flat at 0.2500 over all tested radii, RAUC is 0.2500, and median r80 is at the
evaluation maximum 0.1200.

FetchSlide-v4 command:

```bash
PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/fetch_object_goal_recovery_calibration.py --out results/fetchslide_object_goal_calibration_2seed_v1 --env-id FetchSlide-v4 --seeds 2 --replay-states 4 --trials 1 --radii 0.00,0.03,0.06,0.09,0.12,0.15 --horizon 80 --controller-gain 6.0 --direction-jitter 0.10
```

FetchSlide is also rejected as a method route: clean success is 0.2500, RAUC is
0.1875, recovery ranges from 0.1250 to 0.2500, and median r80 is 0.0720. The
curve is not fully flat, but the scripted controller fails the clean-success
prerequisite on most replay states.

FetchPickAndPlace-v4 command:

```bash
PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/fetch_object_goal_recovery_calibration.py --out results/fetchpickplace_object_goal_calibration_2seed_v1 --env-id FetchPickAndPlace-v4 --controller scripted_pick_place --seeds 2 --replay-states 4 --trials 1 --radii 0.00,0.03,0.06,0.09,0.12,0.15 --horizon 100 --controller-gain 6.0 --direction-jitter 0.10
```

FetchPickAndPlace is also rejected as a method route: clean success is 0.1250,
RAUC is 0.0625, recovery ranges from 0.0000 to 0.1250, and median r80 is
0.0660. The fixed pick-place controller does not provide a usable clean
recovery interface.

Compact artifacts:

- `results/fetchpush_object_goal_calibration_2seed_v1/summary.json`
- `results/fetchpush_object_goal_calibration_2seed_v1/recovery_rows.csv`
- `results/fetchpush_object_goal_calibration_2seed_v1/package_versions.json`
- `results/fetchslide_object_goal_calibration_2seed_v1/summary.json`
- `results/fetchslide_object_goal_calibration_2seed_v1/recovery_rows.csv`
- `results/fetchslide_object_goal_calibration_2seed_v1/package_versions.json`
- `results/fetchpickplace_object_goal_calibration_2seed_v1/summary.json`
- `results/fetchpickplace_object_goal_calibration_2seed_v1/recovery_rows.csv`
- `results/fetchpickplace_object_goal_calibration_2seed_v1/package_versions.json`

Do not build FetchPush, FetchSlide, or FetchPickAndPlace replay comparisons
around these scripted controller/interfaces unless a new preregistered
calibration first produces usable clean success and a non-saturated recovery
curve.

## Internal highway-env Parking Calibration

This is a pre-method calibration for a different external benchmark package,
`highway-env==1.10.1`. It uses the package-owned `parking-v0` goal-conditioned
environment, continuous vehicle dynamics, reward, and success predicate through
`tools/highway_parking_recovery_calibration.py`; it is not a BGR method
comparison.

Command:

```bash
PYTHONPATH=src:. /tmp/bgr_highway311_venv/bin/python tools/highway_parking_recovery_calibration.py --out results/highway_parking_recovery_calibration_12seed_v1 --seeds 12 --radii 0,1,2,3,4,5,6,8,10 --horizon 80
```

The route is rejected before method comparison: clean success is only 0.3333,
recovery ranges from 0.2500 to 0.5000, mean crash rate is 0.5370, RAUC is
0.3750, and median r80 is 9.8000. The fixed scripted controller therefore does
not provide a usable clean recovery interface, even though the package itself
is installable and exposes a real goal-conditioned task. Do not build a
highway-env parking replay comparison unless a new preregistered controller or
policy first clears the clean-success and non-saturated recovery prerequisites.

Compact artifacts:

- `results/highway_parking_recovery_calibration_12seed_v1/summary.json`
- `results/highway_parking_recovery_calibration_12seed_v1/recovery_rows.csv`
- `results/highway_parking_recovery_calibration_12seed_v1/package_versions.json`

## Internal Gymnasium Box2D LunarLander Calibration

This is the next pre-method calibration for a different official Gymnasium
task, `LunarLander-v3`. It uses Gymnasium's package-owned Box2D contact
dynamics, terrain, termination rules, and heuristic controller through
`tools/lunarlander_recovery_calibration.py`; it is not a BGR method comparison.
The reset interface rolls the package heuristic to a fixed descent checkpoint,
then perturbs the exact Box2D lander and leg body state before continuing the
same heuristic controller.

The isolated environment is `/tmp/bgr_lunar_venv` with `gymnasium==1.3.0`,
`box2d==2.3.10`, `pygame-ce==2.5.7`, `swig==4.4.1`, and `numpy==2.4.6`,
leaving repository runtime dependencies unchanged unless the result becomes
promotable.

Fixed pre-method calibration command:

```bash
PYTHONPATH=src:. /tmp/bgr_lunar_venv/bin/python tools/lunarlander_recovery_calibration.py --out results/lunarlander_recovery_calibration_12seed_v1
```

This calibration is only permission to implement a fixed all-method screen. Do
not promote LunarLander into the paper or run a method comparison until the
calibration clears clean success >= 0.80, recovery range >= 0.20, and
non-saturated median r80. Any later comparison must fix the replay-state pool,
perturbation radii, baselines, seeds, learner, and promotion gate before seeing
method results. Promotion should require default BGR or BGR-Coverage to beat
uniform, fixed-radius, failure-only, TD/loss-priority, and the
state-priority/uniform-radius ablation on final RAUC with a visible effect,
paired wins over uniform, and non-contradictory non-saturated radius metrics.

The fixed 12-seed calibration clears the pre-method gate: clean success is
0.9167, recovery ranges from 0.5833 to 0.9167, RAUC is 0.7722, median r80 is
0.5300 on the 0--1 perturbation grid, and the mean crash rate is 0.1620. This
is not paper evidence; it only permits a fixed all-method LunarLander screen.

Compact artifacts:

- `results/lunarlander_recovery_calibration_12seed_v1/summary.json`
- `results/lunarlander_recovery_calibration_12seed_v1/recovery_rows.csv`
- `results/lunarlander_recovery_calibration_12seed_v1/package_versions.json`

### LunarLander Comparison Screen

The fixed all-method LunarLander screen is implemented at
`tools/lunarlander_recovery_probe.py` before method-comparison results. It
keeps the same official `LunarLander-v3` package, Box2D dynamics, heuristic
teacher, fixed burn-in checkpoint, 0--1 perturbation grid, and isolated
`/tmp/bgr_lunar_venv` package versions as the calibration. The learner is a
fixed linear imitation policy trained on replayed perturbed states selected by
the replay method.

Preregistered command:

```bash
PYTHONPATH=src:. /tmp/bgr_lunar_venv/bin/python tools/lunarlander_recovery_probe.py --out results/lunarlander_recovery_probe_4seed_v1
```

Do not tune this protocol after seeing method outcomes. The 4-seed screen can
only justify a 30-seed scale-up if default BGR or BGR-Coverage beats uniform,
fixed-radius, failure-only, TD/loss-priority, and BGR-uniform-radius on final
RAUC with at least 3/4 paired wins over uniform and non-contradictory,
non-saturated median-r80 evidence.

The fixed 4-seed screen is negative under that gate. BGR-Coverage has the best
mean final RAUC, 0.7500, above uniform 0.6222, fixed-radius 0.7375,
failure-only 0.6799, TD-loss 0.7174, and BGR-uniform-radius 0.7160. It still
fails promotion because it wins only 2/4 paired seeds against uniform and its
median r80 is lower than uniform (0.4200 vs. 0.4825). Default BGR reaches
0.6479 final RAUC, below fixed-radius, failure-only, TD-loss, and
BGR-uniform-radius. Do not scale or promote this LunarLander protocol.

Compact artifacts:

- `results/lunarlander_recovery_probe_4seed_v1/summary.csv`
- `results/lunarlander_recovery_probe_4seed_v1/history.csv`
- `results/lunarlander_recovery_probe_4seed_v1/results.json`
- `results/lunarlander_recovery_probe_4seed_v1/package_versions.json`

## Internal Gymnasium MuJoCo Reacher Calibration

This is a pre-method calibration for a different official Gymnasium MuJoCo
task, `Reacher-v5`. It uses Gymnasium's package-owned two-link Reacher dynamics
and target sampling through `tools/reacher_recovery_calibration.py`; it is not a
BGR method comparison. The reset interface perturbs the two arm joint angles
from the package reset state and keeps the package target fixed. Evaluation
runs a fixed inverse-kinematics/PD controller with deliberately weak torque and
horizon settings so the calibration measures a recovery boundary rather than a
solved controller.

Command:

```bash
PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/reacher_recovery_calibration.py --out results/reacher_recovery_calibration_12seed_v1
```

The route clears the pre-method calibration gate: clean success is 0.8333,
recovery ranges from 0.5000 to 0.9167, RAUC is 0.7891, and r80 is 3.0000 on a
0--4 joint-perturbation grid. The isolated environment records
`gymnasium==1.3.0`, `mujoco==3.9.0`, and `numpy==2.4.6`.

Compact artifacts:

- `results/reacher_recovery_calibration_12seed_v1/summary.json`
- `results/reacher_recovery_calibration_12seed_v1/recovery_rows.csv`
- `results/reacher_recovery_calibration_12seed_v1/package_versions.json`

This calibration is only permission to implement a fixed all-method screen. Do
not promote Reacher into the paper or run a method comparison until the full
comparison tool, replay-state pool, perturbation radii, baselines, seeds,
learner, and promotion gate are fixed before seeing method results. Promotion
should require default BGR or BGR-Coverage to beat uniform, fixed-radius,
failure-only, TD/loss-priority, and the state-priority/uniform-radius ablation
on final RAUC with a visible effect, paired wins over uniform, and
non-contradictory non-saturated radius metrics.

### Reacher Comparison Result

The fixed all-method Reacher screen is implemented at
`tools/reacher_recovery_probe.py` before method-comparison results. It keeps the
same official `Reacher-v5` package dynamics, target sampling, exact MuJoCo state
resets, two-joint angular perturbation family, 0--4 evaluation grid, and
12-seed screen budget. The learner is a small linear controller initialized
below the calibration controller and trained by imitation of a stronger
inverse-kinematics teacher on replayed perturbed states selected by the replay
method.

Preregistered command:

```bash
PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/reacher_recovery_probe.py --out results/reacher_recovery_probe_12seed_v1
```

The fixed 12-seed screen is negative and should not be promoted or scaled.
Uniform replay has the highest final RAUC (0.3862). Default BGR reaches 0.2907
with 4/8/0 paired wins/losses/ties against uniform, BGR-Coverage reaches
0.2721 with 4/8/0 against uniform, BGR-uniform-radius reaches 0.2921,
failure-only reaches 0.3273, fixed-radius reaches 0.2330, and TD-loss reaches
0.2501. Median-r80 is also not supportive: BGR is 3.8375, BGR-Coverage is
3.6708, and uniform is 3.2437.

Compact artifacts:

- `results/reacher_recovery_probe_12seed_v1/summary.csv`
- `results/reacher_recovery_probe_12seed_v1/aggregate.csv`
- `results/reacher_recovery_probe_12seed_v1/history.csv`
- `results/reacher_recovery_probe_12seed_v1/package_versions.json`

## Internal Gymnasium MuJoCo InvertedPendulum Calibration

This is a pre-method calibration for another official Gymnasium MuJoCo task,
`InvertedPendulum-v5`. It uses Gymnasium's package-owned dynamics through
`tools/inverted_pendulum_recovery_calibration.py`; it is not a BGR method
comparison. The reset interface perturbs the pole angle from exact MuJoCo
states and evaluates a fixed PD balance controller over a 200-step survival
horizon.

Command:

```bash
PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/inverted_pendulum_recovery_calibration.py --out results/inverted_pendulum_recovery_calibration_12seed_v1
```

The route clears the pre-method calibration gate: clean success is 1.0000,
recovery ranges from 0.0000 to 1.0000, RAUC is 0.7500, and r80 is 0.2100 on a
0--0.30 pole-angle perturbation grid. The isolated environment records
`gymnasium==1.3.0`, `mujoco==3.9.0`, and `numpy==2.4.6`.

Compact artifacts:

- `results/inverted_pendulum_recovery_calibration_12seed_v1/summary.json`
- `results/inverted_pendulum_recovery_calibration_12seed_v1/recovery_rows.csv`
- `results/inverted_pendulum_recovery_calibration_12seed_v1/package_versions.json`

This calibration is only permission to run the fixed all-method screen. The
comparison tool was implemented before method-comparison results at
`tools/inverted_pendulum_recovery_probe.py`. It keeps the official
`InvertedPendulum-v5` package dynamics, exact MuJoCo state resets, pole-angle
perturbation family, 0--0.30 evaluation grid, and a 4-seed pre-promotion screen
budget. The learner is a small linear controller initialized below the
calibration controller and trained by imitation of a stronger PD teacher on
replayed perturbed states selected by the replay method.

Preregistered command:

```bash
PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/inverted_pendulum_recovery_probe.py --out results/inverted_pendulum_recovery_probe_4seed_v1
```

Do not scale or promote this route unless default BGR or BGR-Coverage beats
uniform, fixed-radius, failure-only, TD/loss-priority, and the
state-priority/uniform-radius ablation on final RAUC with a visible effect,
paired wins over uniform, and non-contradictory non-saturated median-r80
metrics.

The completed 4-seed screen is negative and should not be scaled. Every method
has the same final clean success (1.0000), final RAUC (0.7500), final median
r80 (0.2100), and best RAUC (0.7500). BGR has 0/0/4 paired wins/losses/ties
against uniform on final RAUC, so the route fails the visible-effect,
paired-win, strong-baseline, and state-priority-ablation gates.

Compact artifacts:

- `results/inverted_pendulum_recovery_probe_4seed_v1/summary.csv`
- `results/inverted_pendulum_recovery_probe_4seed_v1/aggregate.csv`
- `results/inverted_pendulum_recovery_probe_4seed_v1/history.csv`
- `results/inverted_pendulum_recovery_probe_4seed_v1/package_versions.json`

## Internal Gymnasium MuJoCo InvertedDoublePendulum Calibration

This began as a pre-method calibration route for another official Gymnasium
MuJoCo task, `InvertedDoublePendulum-v5`. It uses Gymnasium's package-owned
dynamics through `tools/inverted_double_pendulum_recovery_calibration.py`; the
calibration itself is not a BGR method comparison. The reset interface starts
from the package seeded reset state, applies a two-pole angular perturbation,
zeros velocities, and evaluates a fixed finite-difference LQR balance
controller over a 250-step survival horizon.

Command:

```bash
PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/inverted_double_pendulum_recovery_calibration.py --out results/inverted_double_pendulum_recovery_calibration_12seed_v1
```

The route clears the pre-method calibration gate: clean success is 1.0000,
recovery ranges from 0.0000 to 1.0000, RAUC is 0.4259, and r80 is 0.2825 on a
0--0.90 two-pole perturbation grid. The isolated environment records
`gymnasium==1.3.0`, `mujoco==3.9.0`, and `numpy==2.4.6`.

Compact artifacts:

- `results/inverted_double_pendulum_recovery_calibration_12seed_v1/summary.json`
- `results/inverted_double_pendulum_recovery_calibration_12seed_v1/recovery_rows.csv`
- `results/inverted_double_pendulum_recovery_calibration_12seed_v1/package_versions.json`

This calibration only gave permission to run the fixed all-method screen. The
comparison tool was fixed before method-comparison results at
`tools/inverted_double_pendulum_recovery_probe.py`. It keeps the official
`InvertedDoublePendulum-v5` package dynamics, exact MuJoCo state resets,
two-pole angular perturbation family, 0--0.90 evaluation grid, and a 4-seed
pre-promotion screen budget. The learner is a linear LQR-feature controller
initialized at 0.70 times the calibrated LQR gain and trained by imitation of
the full LQR teacher on replayed perturbed states selected by the replay
method.

Preregistered command:

```bash
PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/inverted_double_pendulum_recovery_probe.py --out results/inverted_double_pendulum_recovery_probe_4seed_v1
```

Do not tune the learner, replay-state count, perturbation radii, methods,
seeds, or promotion gate after seeing this result. Do not scale or promote this
route unless default BGR or BGR-Coverage beats uniform, fixed-radius,
failure-only, TD/loss-priority, and the state-priority/uniform-radius ablation
on final RAUC with a visible effect, paired wins over uniform, and
non-contradictory non-saturated median-r80 metrics.

The completed 4-seed screen is negative and should not be scaled. Default BGR
has the highest final RAUC (0.0833 vs. 0.0035 for uniform and 0.0000 for the
other baselines), but this comes with clean-success collapse: BGR final clean is
0.2500, BGR-Coverage is 0.0000, and uniform is 0.0208. The paired RAUC split
against uniform is only 1/0/3, and median-r80 is saturated at 0.9000 for every
method except the single surviving BGR seed. This is not acceptance evidence.

Compact artifacts:

- `results/inverted_double_pendulum_recovery_probe_4seed_v1/summary.csv`
- `results/inverted_double_pendulum_recovery_probe_4seed_v1/aggregate.csv`
- `results/inverted_double_pendulum_recovery_probe_4seed_v1/history.csv`
- `results/inverted_double_pendulum_recovery_probe_4seed_v1/package_versions.json`

## Internal Official PointMaze Diagnostic

The next preregistered external-package screen is official PointMaze U-Maze,
implemented at `tools/pointmaze_recovery_probe.py`. It runs in an isolated
`/tmp/bgr_pointmaze_venv` environment with `gymnasium-robotics==1.4.2`,
`gymnasium==1.3.0`, and `mujoco==3.9.0`, leaving repo runtime dependencies
unchanged unless the result becomes promotable. The probe uses
`gym.make("PointMaze_UMaze-v3", continuing_task=False, reset_target=False)`,
package maze layouts, package point dynamics, exact `PointEnv.set_state`
restarts, graph-distance perturbations over official free cells, and compares
uniform, fixed-radius, failure-only, TD-loss, BGR-uniform-radius,
BGR-Coverage, and default BGR.

Uniform-only seed-0 calibration was run before any BGR comparison. Medium Maze
and broad U-Maze bands either collapsed or had unusable radius metrics. The
fixed U-Maze near-goal band gives a usable baseline diagnostic: uniform clean
success 0.7500, final RAUC 0.4375, median r80 0.7000, and absolute r20 0.5333.
The preregistered 4-seed command is:
`PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/pointmaze_recovery_probe.py --out results/pointmaze_umaze_recovery_probe_4seed_v1 --env-id PointMaze_UMaze-v3 --max-steps 80 --q-init-blend 1.0 --q-init-noise 0.02 --perturb-cells 3 --replay-distance-min 1 --replay-distance-max 3 --iterations 60`.
Do not scale or promote this result unless default BGR or BGR-Coverage beats
uniform, fixed-radius, failure-only, TD-loss, and BGR-uniform-radius on final
RAUC with a visible gap, while median r80 and absolute r20 are non-saturated
and do not contradict the RAUC effect.

A topology-bottleneck replay-state follow-up is now fixed before method
comparison. It changes the PointMaze reset-state policy rather than the BGR
sampler: `--replay-selection bottleneck` chooses package-maze articulation
points, falling back to low-degree cells if needed. Uniform-only seed-0
calibration rejected the default bottleneck controller on U-Maze (clean success
0.0000) and the stronger controller on Medium Maze (final clean and RAUC both
0.0000). The fixed U-Maze bottleneck protocol uses the stronger controller and
is noncollapsed under uniform seed 0: final clean 0.5000, final RAUC 0.1875,
median r80 0.7000, and absolute r20 0.0667. The preregistered 4-seed command
is:
`PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/pointmaze_recovery_probe.py --out results/pointmaze_umaze_bottleneck_probe_4seed_v1 --env-id PointMaze_UMaze-v3 --methods uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr --replay-selection bottleneck --max-steps 120 --q-init-blend 2.0 --q-init-noise 0.01 --action-scale 0.6 --perturb-cells 3 --replay-distance-min 1 --replay-distance-max 5 --iterations 60 --eval-every 20`.
Do not scale it unless default BGR or BGR-Coverage beats uniform, fixed-radius,
failure-only, TD-loss, and BGR-uniform-radius on final RAUC with a visible gap,
and median r80 plus absolute r20 do not contradict the RAUC effect.

`results/pointmaze_umaze_bottleneck_probe_4seed_v1/summary.csv` is the
completed topology-bottleneck reset-interface screen. It is negative for
promotion: failure-only reaches 0.3500 final RAUC, ahead of fixed-radius
0.0896, default BGR 0.0854, uniform 0.0677, BGR-Coverage 0.0573,
BGR-uniform-radius 0.0167, and TD-loss 0.0156. Failure-only also leads absolute
r20 (0.1462), while default BGR ties uniform at 0.0167 and BGR-Coverage is
0.0000. The promotion checker rejects default BGR because it wins only 2/4
paired seeds against uniform, loses to fixed-radius on mean final RAUC, and
loses to failure-only with W/L/T=0/3/1. It rejects BGR-Coverage because it
loses to uniform, fixed-radius, and failure-only on mean final RAUC; the
absolute-r20 gate also contradicts BGR-Coverage. Do not scale this protocol.

`results/pointmaze_umaze_recovery_probe_4seed_v1/summary.csv` is the completed
4-seed screen. It is negative for promotion: final RAUC is 0.5458 for
failure-only, 0.2201 for uniform, 0.2073 for BGR-Coverage, 0.1406 for BGR,
0.1233 for BGR-uniform-radius, 0.0556 for TD-loss, and 0.0448 for fixed-radius
replay. Failure-only also leads absolute r20 (0.5472 vs. 0.2500 for uniform,
0.0750 for BGR-Coverage, and 0.0167 for BGR). The promotion checker rejects
default BGR and BGR-Coverage because neither beats uniform on mean final RAUC,
both lose to failure-only on all four paired seeds, and both have lower
absolute r20 than uniform. Do not scale this protocol.

The next preregistered algorithmic follow-up is BGR-Guarded, implemented as
`bgr_guarded` in `tools/pointmaze_recovery_probe.py`. It targets the observed
PointMaze failure mode where boundary replay collapses clean recovery while the
failure-only baseline preserves it: updates mix BGR boundary replay with the
same failure-mining candidate rule used by the explicit baseline
(`--guarded-failure-mix 0.55`) and apply clean replay on the selected state
with probability `--guarded-clean-mix 0.35`. It is only promotable if it beats
the original failure-only baseline, not merely old BGR. The preregistered
4-seed follow-up command is:
`PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/pointmaze_recovery_probe.py --out results/pointmaze_umaze_guarded_probe_4seed_v1 --env-id PointMaze_UMaze-v3 --methods uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr,bgr_guarded --max-steps 80 --q-init-blend 1.0 --q-init-noise 0.02 --perturb-cells 3 --replay-distance-min 1 --replay-distance-max 3 --iterations 60`.

`results/pointmaze_umaze_guarded_probe_4seed_v1/summary.csv` is the completed
guarded follow-up. It is negative: final RAUC is 0.5458 for failure-only,
0.2201 for uniform, 0.2073 for BGR-Coverage, 0.1406 for BGR, 0.1233 for
BGR-uniform-radius, 0.0566 for BGR-Guarded, 0.0556 for TD-loss, and 0.0448 for
fixed-radius replay. BGR-Guarded has floor-saturated absolute r20 (0.0000),
below uniform (0.2500) and failure-only (0.5472). The promotion checker rejects
BGR-Guarded because it loses to uniform on mean RAUC, loses to failure-only on
all four paired seeds, loses to the state-priority/uniform-radius ablation, and
has lower absolute r20 than uniform. Do not scale this sampler-mixing variant.

The next preregistered PointMaze follow-up is BGR-Clean-Shield, implemented as
`bgr_clean_shield` in `tools/pointmaze_recovery_probe.py`. It targets the
observed clean-recovery collapse rather than changing only the sampler: if the
selected replay state's current clean recovery is below
`--clean-shield-threshold 0.75`, the update is spent on the clean state
(`sigma=0`); otherwise the method uses default BGR boundary replay and applies
a clean anchor update with probability `--clean-shield-anchor-mix 0.25`. The
fixed 4-seed command is:
`PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/pointmaze_recovery_probe.py --out results/pointmaze_umaze_clean_shield_probe_4seed_v1 --env-id PointMaze_UMaze-v3 --methods uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr,bgr_clean_shield --max-steps 80 --q-init-blend 1.0 --q-init-noise 0.02 --perturb-cells 3 --replay-distance-min 1 --replay-distance-max 3 --iterations 60`.
Do not scale it unless BGR-Clean-Shield beats uniform, fixed-radius,
failure-only, TD-loss, and BGR-uniform-radius on final RAUC, and absolute r20
does not contradict the RAUC effect.

`results/pointmaze_umaze_clean_shield_probe_4seed_v1/summary.csv` is the
completed clean-shield follow-up. It is negative for promotion: BGR-Clean-Shield
improves over default BGR (0.2448 vs. 0.1406 final RAUC) and is above uniform
on mean final RAUC (0.2448 vs. 0.2201), but the paired split against uniform is
only 2/2, failure-only remains much stronger (0.5458 final RAUC), and absolute
r20 contradicts the mean-RAUC gain (0.1167 for BGR-Clean-Shield vs. 0.2500 for
uniform). Do not scale this update-schedule variant.

## Internal Official MiniGrid-FourRooms Diagnostic

`results/minigrid_fourrooms_recovery_probe_4seed_v1/summary.csv` is a 4-seed
pre-promotion screen using the external MiniGrid package rather than a local
environment clone. It ran in an isolated virtualenv with package versions
recorded in
`results/minigrid_fourrooms_recovery_probe_4seed_v1/package_versions.json`
(`minigrid==3.1.0`, `gymnasium==1.3.0`). The protocol uses
`gym.make("MiniGrid-FourRooms-v0")`, package-generated layouts/dynamics,
resettable package env state, bottleneck-adjacent replay states, and Manhattan
position restarts with direction preserved.

The result is promising but not paper-promotable under the preregistered gate.
Final RAUC is 0.1426 for BGR-Coverage, 0.1355 for BGR-uniform-radius, 0.0992
for BGR, 0.0962 for failure-only, 0.0863 for TD-loss, 0.0351 for uniform, and
0.0220 for fixed-radius replay. The promotion checker rejects default BGR
because it loses to the state-priority/uniform-radius ablation. It rejects
BGR-Coverage under the strict preregistered gate only because median r80 is
saturated at 1.0 for BGR-Coverage and uniform; with radius saturation explicitly
waived, BGR-Coverage clears the 4-seed RAUC/baseline screen. Do not promote or
scale this exact protocol until a preregistered follow-up defines a
non-saturated radius check or otherwise resolves the median-r80 issue.

`results/minigrid_fourrooms_recovery_probe_absr10_4seed_v1/summary.csv` is the
preregistered follow-up that keeps the same official package versions, task,
method set, seeds, replay states, perturbation family, and training budget, and
adds `final_abs_r10`, the median largest perturbation radius whose absolute
recovery probability is at least 0.10. The RAUC ordering is unchanged, with
BGR-Coverage retaining the 0.1426 vs. 0.0351 mean RAUC lead over uniform and
remaining above fixed-radius, failure-only, TD-loss, and the
state-priority/uniform-radius ablation. However, `final_abs_r10` is 0.0 for
every method, so this follow-up replaces ceiling saturation with floor
saturation rather than resolving the radius-metric weakness. The promotion
checker now rejects both ceiling- and floor-saturated radius metrics, and it
rejects this follow-up accordingly. Do not scale MiniGrid under this exact
protocol.

The next preregistered MiniGrid screen uses the same official package task but
restricts replay states to a calibrated shortest-path distance band before any
method comparison: `--replay-selection midband --replay-distance-min 2
--replay-distance-max 6`. This band was selected from uniform-only seed-0
calibration because it avoids the two endpoint failures seen so far: the
goalward selector was effectively solved, while wider midband/spread selectors
collapsed to floor-saturated absolute-radius metrics. The fixed 4-seed command
is:
`PYTHONPATH=src:. /tmp/bgr_minigrid_venv/bin/python tools/minigrid_fourrooms_recovery_probe.py --out results/minigrid_fourrooms_recovery_probe_midband_4seed_v1 --replay-selection midband --replay-distance-min 2 --replay-distance-max 6`.
Do not promote this result unless the full method comparison clears the same
baseline and non-saturated median-r80 gates.

`results/minigrid_fourrooms_recovery_probe_midband_4seed_v1/summary.csv` is the
completed midband follow-up. It is negative for promotion: final RAUC is 0.7940
for failure-only, 0.7587 for fixed-radius, 0.6665 for uniform, 0.6287 for
BGR-uniform-radius, 0.6170 for TD-loss, 0.6077 for BGR-Coverage, and 0.5538 for
BGR. The promotion checker rejects BGR-Coverage because it loses to uniform
(-0.0589 mean RAUC, 2/2 paired split), fixed-radius, failure-only, TD-loss, and
the state-priority/uniform-radius ablation. Median r80 is non-saturated but
contradicts the BGR-Coverage RAUC claim: 0.6050 for BGR-Coverage vs. 0.6799 for
uniform. Do not scale this protocol.

Two additional uniform-only replay-distance calibrations were run before any new
method comparison:

- `results/minigrid_fourrooms_recovery_probe_mid2_5_uniform_calibration_4seed_v1/summary.csv`
  uses `--replay-selection midband --replay-distance-min 2
  --replay-distance-max 5` and gives uniform-only clean success 0.7188, RAUC
  0.6190, and non-saturated median r80 0.6451 over four seeds.
- `results/minigrid_fourrooms_recovery_probe_mid3_7_uniform_calibration_4seed_v1/summary.csv`
  uses `--replay-selection midband --replay-distance-min 3
  --replay-distance-max 7` and gives uniform-only clean success 0.7422, RAUC
  0.6661, and non-saturated median r80 0.6687 over four seeds.

Because band 2--5 is harder while keeping relative median r80 non-saturated, the
fixed all-method screen was preregistered as:
`PYTHONPATH=src:. /tmp/bgr_minigrid_venv/bin/python tools/minigrid_fourrooms_recovery_probe.py --out results/minigrid_fourrooms_recovery_probe_mid2_5_4seed_v1 --replay-selection midband --replay-distance-min 2 --replay-distance-max 5`.

`results/minigrid_fourrooms_recovery_probe_mid2_5_4seed_v1/summary.csv` is the
completed all-method screen. It is negative for promotion. Default BGR improves
mean RAUC over uniform (0.6747 vs. 0.6190), but wins only 2/4 paired seeds,
trails fixed-radius replay (0.6779) and failure-only replay (0.7309), and has
lower non-saturated median r80 than uniform (0.5627 vs. 0.6451). BGR-Coverage
also fails: 0.5933 RAUC vs. 0.6190 for uniform, and below fixed-radius,
failure-only, TD-loss, and BGR-uniform-radius. Do not scale or promote this
protocol.

## Internal Official MiniGrid-DoorKey Diagnostic

The next preregistered external-package screen is official MiniGrid-DoorKey,
implemented at `tools/minigrid_doorkey_recovery_probe.py`. It uses the isolated
`/tmp/bgr_minigrid_venv` environment with `minigrid==3.1.0` and
`gymnasium==1.3.0`, leaving repo runtime dependencies unchanged unless the
result becomes promotable. The probe uses
`gym.make("MiniGrid-DoorKey-6x6-v0")`, package-generated layouts and MiniGrid
`env.step` dynamics, exact reset states with the key carried and the door still
closed, and position/direction perturbations around the door approach.

Uniform-only seed-0 calibration selected a distance band and budget that avoid
the FourRooms endpoint failures: with `--replay-distance-min 1
--replay-distance-max 6`, `--iterations 50`, `--train-batch-size 5`,
`--q-init-blend 0.015`, and `--absolute-radius-alpha 0.01`, uniform gives final
RAUC 0.4939, clean success 0.9167, median r80 0.3000, and absolute radius
0.7475. The fixed 4-seed command is:
`PYTHONPATH=src:. /tmp/bgr_minigrid_venv/bin/python tools/minigrid_doorkey_recovery_probe.py --out results/minigrid_doorkey_recovery_probe_4seed_v1 --iterations 50 --eval-every 25 --train-batch-size 5 --q-init-blend 0.015 --q-init-noise 0.05 --learning-rate 0.25 --epsilon 0.10 --absolute-radius-alpha 0.01 --replay-distance-min 1 --replay-distance-max 6`.
Do not scale it unless BGR or BGR-Coverage beats uniform, fixed-radius,
failure-only, TD-loss, and BGR-uniform-radius on final RAUC, and median r80 plus
absolute radius do not contradict the RAUC effect.

`results/minigrid_lavacrossing_recovery_probe_4seed_v1/summary.csv` is the
completed fixed 4-seed screen. It is negative for promotion: final RAUC is
0.4165 for uniform, 0.4042 for BGR-uniform-radius, 0.3547 for BGR-Coverage,
0.3153 for BGR, 0.2358 for failure-only, 0.2059 for TD-loss, and 0.1677 for
fixed-radius. The promotion checker rejects BGR-Coverage because it loses to
uniform (-0.0618 mean RAUC), loses to the state-priority/uniform-radius
ablation, and has lower final_abs_r10 than uniform (0.5047 vs. 0.6352). It
rejects default BGR because it loses to uniform (-0.1012 mean RAUC), loses to
the state-priority/uniform-radius ablation, and has lower final_abs_r10 than
uniform (0.4187 vs. 0.6352). Do not scale this protocol.

`results/minigrid_doorkey_recovery_probe_4seed_v1/summary.csv` is the completed
fixed 4-seed screen. It is negative for promotion: final RAUC is 0.6459 for
failure-only, 0.6384 for uniform, 0.5018 for TD-loss, 0.4939 for
BGR-uniform-radius, 0.4846 for BGR-Coverage, 0.3687 for BGR, and 0.2424 for
fixed-radius. The promotion checker rejects BGR-Coverage because it loses to
uniform (-0.1538 mean RAUC), failure-only, TD-loss, and the
state-priority/uniform-radius ablation; it rejects default BGR because it loses
to uniform (-0.2697 mean RAUC) and all strong baselines. Absolute-radius checks
also contradict promotion: final_abs_r10 is 0.5916 for BGR-Coverage and 0.4981
for BGR versus 0.7484 for uniform. Do not scale this protocol.

The next official MiniGrid external-package screen is MiniGrid-LavaGapS7. This
uses the same package-backed lava probe and extends only the allowed `--env-id`
choices in `tools/minigrid_lavacrossing_recovery_probe.py`; the environment is
`gym.make("MiniGrid-LavaGapS7-v0")` from `minigrid==3.1.0`. LavaGap is a
different fixed package geometry from LavaCrossing, with a single gap through a
lava barrier. Baseline-only calibration before method comparison rejected
LavaGapS5/S6 and default S7 budgets because uniform saturated final RAUC and
radius metrics. The fixed S7 hard budget was chosen from uniform-only
calibration: over seeds 0--3, uniform gives final clean 0.8917, final RAUC
0.4461, median r80 0.2453, and absolute r10 0.6336. The preregistered 4-seed
command is:
`PYTHONPATH=src:. /tmp/bgr_minigrid_venv/bin/python tools/minigrid_lavacrossing_recovery_probe.py --out results/minigrid_lavagap_s7_recovery_probe_4seed_v1 --env-id MiniGrid-LavaGapS7-v0 --methods uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr --replay-selection spread --replay-distance-min 2 --replay-distance-max 8 --iterations 60 --eval-every 20 --train-batch-size 5 --q-init-blend 0.015 --q-init-noise 0.08 --learning-rate 0.25 --epsilon 0.10 --rollout-horizon 40 --max-radius 6 --absolute-radius-alpha 0.10`.
Do not scale it unless BGR or BGR-Coverage beats uniform, fixed-radius,
failure-only, TD-loss, and BGR-uniform-radius on final RAUC with a visible gap,
and median r80 plus absolute r10 do not contradict the RAUC effect.

`results/minigrid_lavagap_s7_recovery_probe_4seed_v1/summary.csv` is the
completed fixed 4-seed LavaGap screen. It is negative for promotion: final RAUC
is 0.4627 for BGR-uniform-radius, 0.4461 for uniform, 0.4277 for BGR-Coverage,
0.4094 for TD-loss, 0.4031 for BGR, 0.3353 for failure-only, and 0.1435 for
fixed-radius. The promotion checker rejects default BGR because it loses to
uniform (1/4 paired wins, -0.0430 mean RAUC), TD-loss, and the
state-priority/uniform-radius ablation; absolute r10 also contradicts BGR
versus uniform (0.6047 vs. 0.6336). It rejects BGR-Coverage because it trails
uniform on mean RAUC (-0.0184), loses to the state-priority/uniform-radius
ablation, and has lower median r80 than uniform. Do not scale this protocol.

## Internal Official MiniGrid-LavaCrossing Diagnostic

The next preregistered external-package screen is official
MiniGrid-LavaCrossingS9N3, implemented at
`tools/minigrid_lavacrossing_recovery_probe.py`. It uses the isolated
`/tmp/bgr_minigrid_venv` environment with `minigrid==3.1.0` and
`gymnasium==1.3.0`, leaving repo runtime dependencies unchanged unless the
result becomes promotable. The probe uses
`gym.make("MiniGrid-LavaCrossingS9N3-v0")`, package-generated lava/goal layouts
and MiniGrid `env.step` dynamics, exact reset states on safe cells near the
package shortest path and lava hazards, and Manhattan position/direction
perturbations that preserve safe-cell validity. Lava cells are terminal
failures, not replay states.

Uniform-only seed-0 calibration selected a non-saturated pre-promotion budget:
with `--iterations 60`, `--train-batch-size 5`, `--replay-selection midband`,
`--replay-distance-min 4`, `--replay-distance-max 16`, `--q-init-blend 0.015`,
`--q-init-noise 0.06`, `--learning-rate 0.25`, `--epsilon 0.10`, and
`--absolute-radius-alpha 0.10`, uniform gives final RAUC 0.4878, clean success
0.9167, median r80 0.3000, and absolute radius 0.6500. The fixed 4-seed command
is:
`PYTHONPATH=src:. /tmp/bgr_minigrid_venv/bin/python tools/minigrid_lavacrossing_recovery_probe.py --out results/minigrid_lavacrossing_recovery_probe_4seed_v1 --env-id MiniGrid-LavaCrossingS9N3-v0 --iterations 60 --eval-every 20 --train-batch-size 5 --replay-selection midband --replay-distance-min 4 --replay-distance-max 16 --q-init-blend 0.015 --q-init-noise 0.06 --learning-rate 0.25 --epsilon 0.10 --absolute-radius-alpha 0.10`.
Do not scale it unless BGR or BGR-Coverage beats uniform, fixed-radius,
failure-only, TD-loss, and BGR-uniform-radius on final RAUC, and median r80 plus
absolute radius do not contradict the RAUC effect.

## Internal FourRooms Diagnostic

`results/fourrooms_recovery_probe_4seed_v1/summary.csv` is a 4-seed
pre-promotion diagnostic on a canonical 11x11 FourRooms grid with deterministic
shortest-path dynamics, cross walls, four doorways, resettable doorway-adjacent
replay states, and Manhattan restart perturbations. It was preregistered in
`docs/aaai_acceptance_gap.md` before the fixed 4-seed comparison ran. The
diagnostic is not promotable: final RAUC is 0.9994 for failure-only, 0.9958 for
TD-loss, 0.9900 for fixed-radius, 0.9777 for uniform, 0.9746 for
BGR-uniform-radius, 0.9695 for BGR-Coverage, and 0.9672 for BGR. The promotion
checker rejects default BGR because it loses to uniform on mean RAUC (-0.0106,
1/3 paired split), loses to fixed-radius, failure-only, TD-loss, and the
state-priority/uniform-radius ablation, and has saturated median r80 against
uniform. It rejects BGR-Coverage because it trails uniform and all strong
baselines and has a lower median r80 than uniform. Keep FourRooms out of the
paper except as a limitation unless a new fixed protocol first creates a
non-saturated boundary-radius effect.

## Internal CliffWalking Diagnostics

`results/cliffwalking_recovery_probe_4seed_v1/summary.csv` is a 4-seed
pre-promotion diagnostic on canonical CliffWalking-v0 dynamics with resettable
safe-corridor states and Manhattan restart perturbations. This default budget
saturates and is not promotable: fixed-radius, failure-only, and TD-loss replay
all reach 1.0000 final RAUC, uniform reaches 0.9945, BGR-uniform-radius reaches
0.9958, and BGR reaches 0.9917; clean success and median r80 saturate at 1.0.

`results/cliffwalking_recovery_probe_hard_4seed_v1/summary.csv` is a harder
undertrained diagnostic with fewer updates and lower optimal-Q initialization.
It is also negative for promotion: final RAUC is 0.9074 for failure-only, 0.7118
for TD-loss, 0.6758 for fixed-radius, 0.6490 for uniform, 0.6087 for
BGR-Coverage, 0.4900 for BGR, and 0.4533 for BGR-uniform-radius. BGR beats the
state-priority-only ablation in this harder variant but loses to uniform and all
strong baselines, so CliffWalking remains an internal negative
independent-benchmark probe rather than paper evidence.

## Internal Acrobot Diagnostic

`results/acrobot_recovery_probe_4seed_v1/summary.csv` is a 4-seed
pre-promotion diagnostic on canonical Acrobot-v1 dynamics implemented locally
without adding a Gym dependency. Replay states are near-swing-up continuous
states, perturbations interpolate toward the hanging state with bounded restart
noise, and all methods share the same coarse value-table initialization,
rollout horizon, and evaluation grid. The diagnostic is non-saturated but not
promotable. Final RAUC is 0.1471 for uniform, 0.1488 for BGR-Coverage, 0.1455
for BGR, 0.1383 for failure-only, 0.1305 for BGR-uniform-radius, 0.1292 for
TD-loss, and 0.1279 for fixed-radius replay. The promotion checker rejects
default BGR because it loses to uniform on mean RAUC (-0.0016, 2/2 paired
split). It also rejects BGR-Coverage because the uniform gap is only +0.0016
with a 2/1/1 paired split, below the pre-set 3/4 wins and +0.01 mean-delta
thresholds. Keep Acrobot out of the paper unless a new fixed protocol first
produces a visible boundary-radius effect.

## Internal Pendulum Diagnostic

`results/pendulum_recovery_probe_4seed_v1/summary.csv` is a 4-seed
pre-promotion diagnostic on canonical Pendulum-v1 dynamics implemented locally
without adding a Gym dependency. Replay states are near-upright angle/velocity
states, perturbations move starts toward a high-angle adverse anchor, and
training imitates a fixed PD controller. The diagnostic is not promotable:
endpoint recovery is near zero for all methods and median r80 saturates at 1.0.
Final RAUC is 0.0075 for failure-only, 0.0036 for BGR, 0.0007 for uniform,
0.0004 for BGR-Coverage, and 0.0000 for fixed-radius, TD-loss, and
BGR-uniform-radius replay. The promotion checker rejects default BGR because
it has only a +0.0029 RAUC gap over uniform with a 2/1/1 paired split, loses to
failure-only replay, and has saturated median r80 against uniform. It rejects
BGR-Coverage because it trails uniform and failure-only. Keep Pendulum out of
the paper unless a new fixed protocol first avoids endpoint collapse and
non-informative radius saturation.

## Packaged OpenVLA Action/TFDS Validation

`results/openvla_action_tfds_validation_v1/summary.json` is a compact,
path-free summary derived from the teacher-replay manifest, matched 2048-step
packing summaries, stock OpenVLA-OFT loader validation, and matched 10-step LoRA
checkpoint smokes. It validates OpenVLA teacher actions, 7D action labels, 8D
state/proprio fields, matched BGR/random TFDS exports, loader/stat shapes, and
checkpoint writeability. It is infrastructure evidence only; the paper still
treats OpenVLA as an audit because the learned adaptation does not stably beat
both matched random selection and the official checkpoint.

## Completed OpenVLA-OFT p2048 1,000-Step Low-LR Image-Augmentation Continuation

Queued and completed on 2026-06-04 in response to the current review risk: the manuscript can
only promote a learned-policy robotics claim if BGR beats both matched random
selection and the unadapted official checkpoint on a standard LIBERO-Goal audit.
This run keeps the p2048 clean-mix BGR/random datasets, official training/eval
statistics, identity-LoRA entry point, image augmentation, and 10-task/10-trial
evaluation scale from the completed 300-step audit, but reduces the adaptation
learning rate to `1e-7` and extends the continuation to 1,000 optimizer steps
(`MAX_STEPS=51000`). It remains ledger-only: the completed repair does not show
a stable BGR improvement over either matched random selection or the official
checkpoint.

Submitted adaptation command shape:

```bash
REMOTE_PROJECT=/work/anonymous/bgr \
REMOTE_LOG_DIR=/work/anonymous/bgr/logs \
REMOTE_RUN_ROOT=/work/anonymous/bgr/runs \
REMOTE_HF_HOME=/work/anonymous/cache_home/huggingface \
REMOTE_TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub \
OPENVLA_OFT_ROOT=/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft \
LIBERO_ROOT=/home/anonymous/LIBERO \
TRAIN_DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/.../dataset_statistics.json \
DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/.../dataset_statistics.json \
FINETUNE_SCRIPT=vla-scripts/finetune_identity_lora.py \
ADAPT_STEPS=1000 LR=1e-7 IMAGE_AUG=True TRAIN_TIME=16:00:00 \
TAG=cleanmix_p2048_step51000_lr1em7_identitylora_imageaug_officialtrainstats_v1 \
EVAL_ARTIFACT=openvla_oft_goal_adapt_eval_cleanmix_p2048_step51000_lr1em7_identitylora_imageaug_officialtrainstats_v1 \
BGR_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p2048_v1 \
RANDOM_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_cleanmix_p2048_v1 \
BGR_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p2048_step51000_lr1em7_identitylora_imageaug_officialtrainstats_v1 \
RANDOM_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p2048_step51000_lr1em7_identitylora_imageaug_officialtrainstats_v1 \
EVAL_TASKS=10 EVAL_TRIALS=10 EVAL_SEED=37 GIT_PULL=0 \
scripts/queue_openvla_oft_goal_adapt.sh --submit
```

Submitted perturbation-audit command shape:

```bash
REMOTE_LOG_DIR=/work/anonymous/bgr/logs \
REMOTE_RUN_ROOT=/work/anonymous/bgr/runs \
REMOTE_HF_HOME=/work/anonymous/cache_home/huggingface \
REMOTE_TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub \
OPENVLA_OFT_ROOT=/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft \
LIBERO_ROOT=/home/anonymous/LIBERO \
TAG=cleanmix_p2048_step51000_lr1em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1 \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p2048_step51000_lr1em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1 \
BGR_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p2048_step51000_lr1em7_identitylora_imageaug_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
RANDOM_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p2048_step51000_lr1em7_identitylora_imageaug_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
BGR_DEPENDENCY=afterok:765586 RANDOM_DEPENDENCY=afterok:765589 \
EVAL_TASKS=10 EVAL_TRIALS=10 EVAL_SEED=37 \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Initial Slurm submission:

```text
765585  BGR 1,000-step low-LR image-aug adapt
765586  BGR merge, afterok:765585
765587  BGR clean 10-task/10-trial eval, afterok:765586
765588  random 1,000-step low-LR image-aug adapt, afterok:765585
765589  random merge, afterok:765588
765590  random clean 10-task/10-trial eval, afterok:765589
765591--765595  official identity -> blur -> brightness -> occlusion -> shift comparator chain
765596--765600  BGR identity -> blur -> brightness -> occlusion -> shift, afterok:765586
765601--765605  random identity -> blur -> brightness -> occlusion -> shift, afterok:765589
```

Initial submission outcome:

```text
765585  BGR adapt completed, exit 0:0, elapsed 00:10:39
765586  BGR merge cancelled on c2-g4-19 during model load before a Python traceback or merged checkpoint
765587--765590  dependent BGR clean eval and random train/merge/eval cancelled after the merge cancellation
765591  official identity comparator completed, 99/100 successes
765592--765605  remaining official/BGR/random perturbation jobs cancelled before complete evidence
```

Repair submission excludes the cancelled merge node and uses a fresh tag so
complete summaries will not mix partial and repaired outputs:

```text
765627  BGR repair1 adapt
765628  BGR repair1 merge, afterok:765627
765629  BGR repair1 clean 10-task/10-trial eval, afterok:765628
765630  random repair1 adapt, afterok:765627
765631  random repair1 merge, afterok:765630
765632  random repair1 clean 10-task/10-trial eval, afterok:765631
765635--765639  official repair1 identity -> blur -> brightness -> occlusion -> shift comparator chain
765640--765644  BGR repair1 identity -> blur -> brightness -> occlusion -> shift, afterok:765628
765645--765649  random repair1 identity -> blur -> brightness -> occlusion -> shift, afterok:765631
```

Repair completion outcome:

```text
765627  BGR repair1 adapt completed, 00:10:42
765628  BGR repair1 merge completed, 00:01:44
765629  BGR repair1 clean eval completed, 98/100 = 0.9800
765630  random repair1 adapt completed, 00:10:44
765631  random repair1 merge completed, 00:01:31
765632  random repair1 clean eval completed, 99/100 = 0.9900
765635--765639  official perturbation chain completed
765640--765644  BGR perturbation chain completed
765645--765649  random perturbation chain completed
```

| Method | Identity | Blur | Brightness | Occlusion | Shift | Perturbed mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Official OpenVLA-OFT | 0.9900 | 0.9700 | 0.9800 | 0.7400 | 0.9800 | 0.9320 |
| BGR low-LR image-aug p2048 | 0.9800 | 0.9700 | 0.9800 | 0.7500 | 0.9600 | 0.9280 |
| Random low-LR image-aug p2048 | 0.9900 | 0.9800 | 0.9800 | 0.7700 | 0.9700 | 0.9380 |

Interpretation: this falsifies the low-learning-rate continuation as a
robotics fine-tuning claim. BGR trails matched random on clean identity
evaluation (98/100 vs. 99/100), trails the official checkpoint on aggregate
perturbation success (464/500 vs. 466/500), and trails matched random more
clearly (464/500 vs. 469/500). The only positive cell is a one-episode
occlusion edge over official (75/100 vs. 74/100), which is not enough to
promote into the paper.

## Completed OpenVLA-OFT p2048 300-Step Image-Augmentation Continuation

Launched on 2026-06-04 after the 100-step p2048 image-augmentation audit
separated BGR from matched random but still tied the unadapted official
checkpoint. This continuation keeps the same p2048 clean-mix datasets,
identity-LoRA entry point, official training/eval statistics, full-goal
10-task/10-trial audit scale, and image augmentation, while extending the
adaptation budget to 300 low-learning-rate steps (`MAX_STEPS=50300`,
`LR=5e-7`). It is a falsifiable learned-policy continuation: it should remain
ledger-only unless the completed summaries show a stable BGR improvement over
both matched random selection and the official checkpoint.

Submitted adaptation command shape:

```bash
REMOTE_PROJECT=/work/anonymous/bgr \
REMOTE_LOG_DIR=/work/anonymous/bgr/logs \
REMOTE_RUN_ROOT=/work/anonymous/bgr/runs \
REMOTE_HF_HOME=/work/anonymous/cache_home/huggingface \
REMOTE_TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub \
OPENVLA_OFT_ROOT=/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft \
LIBERO_ROOT=/home/anonymous/LIBERO \
TRAIN_DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/.../dataset_statistics.json \
DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/.../dataset_statistics.json \
FINETUNE_SCRIPT=vla-scripts/finetune_identity_lora.py \
ADAPT_STEPS=300 LR=5e-7 IMAGE_AUG=True \
TAG=cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_v1 \
EVAL_ARTIFACT=openvla_oft_goal_adapt_eval_cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_v1 \
BGR_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p2048_v1 \
RANDOM_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_cleanmix_p2048_v1 \
BGR_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_v1 \
RANDOM_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_v1 \
EVAL_TASKS=10 EVAL_TRIALS=10 EVAL_SEED=31 GIT_PULL=0 \
scripts/queue_openvla_oft_goal_adapt.sh --submit
```

Submitted perturbation-audit command shape:

```bash
REMOTE_LOG_DIR=/work/anonymous/bgr/logs \
REMOTE_RUN_ROOT=/work/anonymous/bgr/runs \
REMOTE_HF_HOME=/work/anonymous/cache_home/huggingface \
REMOTE_TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub \
OPENVLA_OFT_ROOT=/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft \
LIBERO_ROOT=/home/anonymous/LIBERO \
TAG=cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1 \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1 \
BGR_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
RANDOM_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
BGR_DEPENDENCY=afterok:764960 RANDOM_DEPENDENCY=afterok:764963 \
EVAL_TASKS=10 EVAL_TRIALS=10 EVAL_SEED=31 \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Initial Slurm submission:

```text
764938  BGR adapt, failed before training because the remote checkout had local artifact changes and `git pull --ff-only` refused to merge
764939--764943  first dependent BGR/random merge/eval chain, cancelled after 764938 failed
764944--764948  official identity -> blur -> brightness -> occlusion -> shift comparator chain
764949--764958  first dependent BGR/random perturbation chain, cancelled after 764938 failed
```

Repaired Slurm submission:

```text
764959  BGR 300-step image-aug adapt, superseded by repair2 after storage failure
764960  BGR merge, superseded by repair2 after storage failure
764961  BGR clean eval, failed before final summary during storage outage
764962  random 300-step image-aug adapt, superseded by repair1/repair2
764963  random merge, failed with partial shards during storage outage
764964  random clean eval, cancelled after dependency failure
764965--764969  BGR identity -> blur -> brightness -> occlusion -> shift, superseded by repair2
764970--764974  random identity -> blur -> brightness -> occlusion -> shift, superseded by repair2
```

Failure accounting and storage repair:

```text
764944  official identity eval failed on c2-g4-22 at 2026-06-04T15:22:48 after 43:57, before a final summary
764961  BGR clean eval failed on c2-g4-22 at 2026-06-04T15:22:48 after 35:30, before a final summary
764963  random merge failed on c2-g4-22 at 2026-06-04T15:22:48 after 33:07, after partial merged checkpoint shards
764965  BGR identity perturb eval failed on c2-g4-22 at 2026-06-04T15:22:48 after 35:30, before a final summary
764945--764948, 764964, 764966--764974  cancelled because their dependencies no longer had complete evidence
```

The failed logs did not contain Python tracebacks, out-of-memory signatures, or
complete `Overall success rate` summaries. A follow-up quota check showed the
shared run filesystem at 100% use, with partial merged checkpoint shards and
checkout-level rollout videos consuming writable space. Obsolete raw merged
checkpoint roots from superseded OpenVLA sweeps were removed after their
summaries had already been copied into the package ledger; the filesystem then
reported roughly 386 GB free. The clean and perturbation queue launchers now
route rollout videos through `BGR_EVAL_ROLLOUT_DIR` under each job's
`local_log_dir` instead of the shared OpenVLA-OFT checkout `./rollouts` path.

Second repair submission:

```text
764976  random repair1 adapt failed immediately while the filesystem was still full
764977--764978  cancelled after 764976 failed
764980  random repair1 adapt completed, exit 0:0, elapsed 03:48
764981  random repair1 merge completed, exit 0:0, elapsed 01:23
764982  random repair1 clean eval failed before evaluation because the repair command used an invalid LIBERO path
764983--764997  first repair1 perturb chain superseded because the submitted LIBERO path was invalid; identity jobs failed before evaluation
764998--765002  official repair2 identity -> blur -> brightness -> occlusion -> shift completed, exit 0:0
765003--765007  BGR repair2 identity -> blur -> brightness -> occlusion -> shift completed, exit 0:0
765008--765012  random repair2 identity -> blur -> brightness -> occlusion -> shift completed, exit 0:0, afterok:764981
```

Completed repair2 perturbation rows:

| Method | Identity | Blur | Brightness | Occlusion | Shift | Mean perturbed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BGR p2048 300-step image-aug | 0.9800 | 0.9700 | 0.9900 | 0.7500 | 0.9700 | 0.9200 |
| Official OpenVLA-OFT | 0.9900 | 0.9700 | 0.9800 | 0.7400 | 0.9800 | 0.9175 |
| Random p2048 300-step image-aug | 1.0000 | 0.9700 | 0.9800 | 0.7600 | 0.9700 | 0.9200 |

Interpretation: the 300-step continuation is paper-negative as a learned-policy
improvement claim. BGR ties matched random on the mean perturbed score (368/400
each) and is only one perturbed episode above the official checkpoint (367/400),
while losing the identity condition to both official (98/100 vs. 99/100) and
matched random (98/100 vs. 100/100). This is therefore a completed optimization
audit, not a promotable robotics fine-tuning result.

## Completed OpenVLA-OFT p2048 Image-Augmentation Adaptation Audit

Queued and completed on 2026-06-04 after the full-goal p2048 clean and perturbation audits
showed no stable gain over matched random or the official checkpoint. This is a
single-factor optimization test: it keeps the p2048 clean-mix data, identity-LoRA
entry point, official training/eval statistics, low learning rate, and 100-step
adaptation recipe fixed, and changes only OpenVLA-OFT `--image_aug` from `False`
to `True`.

Submitted adaptation command shape:

```bash
REMOTE_HF_HOME=/work/anonymous/cache_home/huggingface \
REMOTE_TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub \
TRAIN_DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/.../dataset_statistics.json \
DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/.../dataset_statistics.json \
FINETUNE_SCRIPT=vla-scripts/finetune_identity_lora.py \
ADAPT_STEPS=100 LR=1e-6 IMAGE_AUG=True \
TAG=cleanmix_p2048_step50100_lr1em6_identitylora_imageaug_officialtrainstats_v1 \
EVAL_ARTIFACT=openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_imageaug_officialtrainstats_v1 \
BGR_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p2048_v1 \
RANDOM_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_cleanmix_p2048_v1 \
BGR_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p2048_step50100_lr1em6_identitylora_imageaug_officialtrainstats_v1 \
RANDOM_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p2048_step50100_lr1em6_identitylora_imageaug_officialtrainstats_v1 \
EVAL_TASKS=10 EVAL_TRIALS=10 EVAL_SEED=29 \
scripts/queue_openvla_oft_goal_adapt.sh --submit
```

Submitted perturbation-audit command shape:

```bash
REMOTE_HF_HOME=/work/anonymous/cache_home/huggingface \
REMOTE_TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub \
TAG=cleanmix_p2048_step50100_lr1em6_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1 \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1 \
BGR_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p2048_step50100_lr1em6_identitylora_imageaug_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
RANDOM_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p2048_step50100_lr1em6_identitylora_imageaug_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
BGR_DEPENDENCY=afterok:764836 RANDOM_DEPENDENCY=afterok:764839 \
EVAL_TASKS=10 EVAL_TRIALS=10 EVAL_SEED=29 \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Slurm IDs:

```text
764813  BGR image-aug adapt, failed before training because the launcher still used an unwritable anonymous HF cache
764814--764818  first dependent merge/eval/random chain, cancelled after 764813 failed
764835  BGR image-aug adapt repair with REMOTE_HF_HOME/REMOTE_TRANSFORMERS_CACHE
764836  BGR image-aug merge, afterok:764835
764837  BGR image-aug clean eval, failed before eval because local_log_dir still used an unwritable anonymous run root
764838  random image-aug adapt repair, afterok:764835
764839  random image-aug merge, afterok:764838
764840  random image-aug clean eval, cancelled after 764837 exposed the local_log_dir bug
764819--764823  official identity -> blur -> brightness -> occlusion -> shift
764824--764833  first BGR/random perturbation chain, cancelled after the first adaptation chain failed
764841--764850  second BGR/random perturbation chain, cancelled after the clean-eval local_log_dir failure
764853--764857  BGR identity -> blur -> brightness -> occlusion -> shift, afterok:764836
764858--764862  random identity -> blur -> brightness -> occlusion -> shift, afterok:764839
```

Final perturbation eval accounting:

```text
764819--764823  official identity/blur/brightness/occlusion/shift completed, exit 0:0, elapsed 16:07--21:19
764853--764857  BGR identity/blur/brightness/occlusion/shift completed, exit 0:0, elapsed 11:40--15:04
764858--764862  random identity/blur/brightness/occlusion/shift completed, exit 0:0, elapsed 16:01--20:48
```

Summary rows:

| Method | Identity | Blur | Brightness | Occlusion | Shift | Mean perturbed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BGR image-aug p2048 | 0.9900 | 0.9800 | 0.9800 | 0.7300 | 0.9800 | 0.9175 |
| Official OpenVLA-OFT | 0.9900 | 0.9700 | 0.9800 | 0.7400 | 0.9800 | 0.9175 |
| Random image-aug p2048 | 0.9800 | 0.9800 | 0.9600 | 0.7500 | 0.9400 | 0.9075 |

Interpretation: image augmentation gives BGR a small aggregate edge over the
matched random selector (+4/400 perturbed episodes, plus +1/100 identity
episode), but BGR ties the unadapted official checkpoint on both identity and
mean perturbed success. Relative to the previous no-image-augmentation full-goal
BGR audit, the perturbed mean is unchanged at 0.9175; occlusion decreases from
0.7500 to 0.7300 while shift increases from 0.9600 to 0.9800. This is therefore
paper-negative as a robotics fine-tuning claim and remains a ledger-only
optimization audit.

## Completed OpenVLA-OFT p2048 10-Trial Perturbation Follow-Up

Queued and completed on 2026-06-04 to reduce variance in the p2048 perturbation
audit. The paper reports 15-episode original and offset-3 perturbation
diagnostics; this follow-up evaluates five LIBERO-Goal tasks with 10 initial
states each, giving 50 episodes per perturbation and method. It remains an
audit, not a promoted robotics fine-tuning claim: BGR has a small matched-random
edge on the perturbed mean but ties the unadapted official checkpoint.

Submitted command shape:

```bash
TAG=cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_10trials_v1 \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_10trials_v1 \
REMOTE_RUN_ROOT=/work/anonymous/bgr/runs \
REMOTE_HF_HOME=/work/anonymous/cache_home/huggingface \
BGR_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
RANDOM_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
EVAL_TRIALS=10 \
EVAL_SEED=17 \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Initial Slurm jobs:

```text
764720--764724  official identity/blur/brightness/occlusion/shift completed
764725          BGR identity failed during checkpoint config mutation
764726--764729  BGR blur/brightness/occlusion/shift cancelled before completion
764730--764734  random identity/blur/brightness/occlusion/shift cancelled before start
```

The BGR failure came from concurrent eval jobs mutating the same merged
checkpoint config. The BGR and random evals were resubmitted as per-checkpoint
serial chains:

```text
764735--764739  BGR identity -> blur -> brightness -> occlusion -> shift completed
764740--764744  random identity -> blur -> brightness -> occlusion -> shift completed
```

Summary rows:

| Method | Identity | Blur | Brightness | Occlusion | Shift | Mean perturbed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BGR clean-mix p2048 | 0.9800 | 0.9600 | 0.9600 | 0.6200 | 0.9600 | 0.8750 |
| Official OpenVLA-OFT | 0.9800 | 0.9600 | 0.9800 | 0.6000 | 0.9600 | 0.8750 |
| Random clean-mix p2048 | 0.9800 | 0.9800 | 0.9400 | 0.6000 | 0.9600 | 0.8700 |

Interpretation: this variance-reduction audit preserves the earlier p2048
pattern. BGR is one episode above matched random and official on occlusion,
one episode below random on blur, one episode below official on brightness, and
tied on identity/shift. Aggregated over perturbed conditions, BGR ties official
at 175/200 successes and is one episode above matched random at 174/200. This
supports keeping OpenVLA-OFT as an audit rather than promoting a stable robotics
fine-tuning gain.

Compact artifact:

```bash
PYTHONPATH=src:. python3 scripts/summarize_openvla_oft_perturb_eval.py \
  --logs-root results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_10trials_v1/logs \
  --out results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_10trials_v1
```

## Completed OpenVLA-OFT p2048 Full-Goal Clean Identity Audit

Queued and completed on 2026-06-04 after the p2048 10-trial perturbation variance audit tied
the unadapted official checkpoint and gave only a one-episode edge over matched
random. This follow-up evaluates the p2048 BGR, matched-random, and official
checkpoints under identity/no-perturbation observations across all 10
LIBERO-Goal tasks with 10 initial states each. It directly tests the paper's
remaining multi-task learned-policy gap without rerunning adaptation.

Submitted command shape:

```bash
TAG=cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1 \
EVAL_ARTIFACT=openvla_oft_clean_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1 \
REMOTE_RUN_ROOT=/work/anonymous/bgr/runs \
REMOTE_HF_HOME=/work/anonymous/cache_home/huggingface \
BGR_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
RANDOM_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
PERTURBATIONS='identity={}' \
EVAL_TASKS=10 \
EVAL_TRIALS=10 \
EVAL_SEED=23 \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Initial Slurm jobs:

```text
764778  official identity, completed, 00:13:37, 99/100 = 0.9900
764779  BGR identity, completed, 00:13:38, 99/100 = 0.9900
764780  random identity, completed, 00:13:25, 99/100 = 0.9900
```

Summary rows:

| Method | Tasks | Episodes | Successes | Clean success |
| --- | ---: | ---: | ---: | ---: |
| BGR clean-mix p2048 | 10 | 100 | 99 | 0.9900 |
| Official OpenVLA-OFT | 10 | 100 | 99 | 0.9900 |
| Random clean-mix p2048 | 10 | 100 | 99 | 0.9900 |

Interpretation: the full LIBERO-Goal identity audit confirms that the p2048
BGR adaptation preserves clean competence across the 10-task suite. It remains a
scoping audit rather than a promoted learned-policy improvement, because BGR
ties both matched random and the unadapted official checkpoint.

## Completed OpenVLA-OFT p2048 Full-Goal Visual Perturbation Audit

Queued and completed on 2026-06-04 after the full-goal clean identity audit tied
all methods at 99/100. This follow-up runs the same 10 LIBERO-Goal tasks with 10
initial states for identity, blur, brightness, occlusion, and shift. The launcher
serializes perturbations within each checkpoint chain, avoiding the shared
checkpoint config mutation race observed in the earlier p2048 10-trial audit,
while still running official, BGR, and random chains in parallel.

Submitted command shape:

```bash
TAG=cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_perturb_v1 \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1 \
REMOTE_RUN_ROOT=/work/anonymous/bgr/runs \
REMOTE_HF_HOME=/work/anonymous/cache_home/huggingface \
BGR_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
RANDOM_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
EVAL_TASKS=10 \
EVAL_TRIALS=10 \
EVAL_SEED=29 \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Initial Slurm jobs:

```text
764781--764785  official identity -> blur -> brightness -> occlusion -> shift completed
764786--764790  BGR identity -> blur -> brightness -> occlusion -> shift completed
764791--764795  random identity -> blur -> brightness -> occlusion -> shift completed
```

Summary rows:

| Method | Identity | Blur | Brightness | Occlusion | Shift | Mean perturbed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BGR clean-mix p2048 | 0.9900 | 0.9800 | 0.9800 | 0.7500 | 0.9600 | 0.9175 |
| Official OpenVLA-OFT | 0.9900 | 0.9700 | 0.9800 | 0.7400 | 0.9800 | 0.9175 |
| Random clean-mix p2048 | 0.9900 | 0.9900 | 0.9700 | 0.7400 | 0.9800 | 0.9200 |

Interpretation: the full-goal perturbation audit preserves the bounded OpenVLA
story. BGR is one episode above official and random on occlusion, but one below
random on blur and two below official/random on shift. Aggregated over the four
perturbed conditions, BGR ties official at 367/400 successes and trails matched
random by one episode at 368/400. This is useful robustness instrumentation and
variance evidence, not a promoted robotics fine-tuning gain.

## Completed OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic

Launched on 2026-06-02 after p2048 again tied matched random on clean success and
gave only a one-episode perturbation edge over matched random while tying the
unadapted official checkpoint. This run doubles the perturbation subset again,
to 4,096 rendered perturbation examples with sixteen perturbation episodes per
family, using the same corrected identity-LoRA and official-training-statistics
recipe as p1024/p2048. Treat it as a falsification/scale diagnostic for the
OpenVLA audit, not as a promised robotics claim.

Submitted prep/TFDS command:

```bash
TAG=p4096 \
MAX_PERTURB_EXAMPLES=4096 \
PERTURB_EPISODES_PER_FAMILY=16 \
TIME=16:00:00 \
scripts/queue_openvla_oft_clean_mix_prep.sh --submit
```

Submitted clean adaptation/eval continuation:

```bash
TRAIN_DEPENDENCY=afterok:763725 \
TRAIN_DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b/dataset_statistics.json \
DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b/dataset_statistics.json \
FINETUNE_SCRIPT=vla-scripts/finetune_identity_lora.py \
ADAPT_STEPS=100 LR=1e-6 \
TAG=cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1 \
EVAL_ARTIFACT=openvla_oft_goal_adapt_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1 \
BGR_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p4096_v1 \
RANDOM_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_cleanmix_p4096_v1 \
BGR_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1 \
RANDOM_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1 \
TRAIN_TIME=05:00:00 \
EVAL_TIME=06:00:00 \
scripts/queue_openvla_oft_goal_adapt.sh --submit
```

Submitted perturbation eval command:

```bash
TAG=cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1 \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1 \
BGR_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
RANDOM_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
BGR_DEPENDENCY=afterok:763727 \
RANDOM_DEPENDENCY=afterok:763730 \
EVAL_TIME=06:00:00 \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Slurm chain:

```text
763725  bgr-cleanmix-prep-p4096                                                               completed, 00:25:03; TFDS roots valid but perturb counts imbalanced
763726  bgr-goal-adapt-bgr-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1     afterok:763725
763727  bgr-goal-merge-bgr-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1     afterok:763726
763728  bgr-goal-eval-bgr-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1      afterok:763727
763729  bgr-goal-adapt-random-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1  afterok:763725, afterok:763726
763730  bgr-goal-merge-random-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1  afterok:763729
763731  bgr-goal-eval-random-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1   afterok:763730
763732--763736  official perturb evals, submitted immediately
763737--763741  BGR perturb evals, afterok:763727
763742--763746  random perturb evals, afterok:763730
```

Live status checked 2026-06-03:

```text
763725  bgr-cleanmix-prep-p4096                                                               completed, 00:25:03
763726  bgr-goal-adapt-bgr-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1     completed, 00:01:43
763727  bgr-goal-merge-bgr-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1     completed, 00:01:23
763728  bgr-goal-eval-bgr-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1      completed, 00:02:27, 14/15 = 0.9333
763729  bgr-goal-adapt-random-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1  completed, 00:01:43
763730  bgr-goal-merge-random-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1  completed, 00:01:26
763731  bgr-goal-eval-random-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1   completed, 00:02:24, 14/15 = 0.9333
763737  bgr-pert-eval-bgr-identity-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1  completed, 00:03:34, 14/15 = 0.9333
763738  bgr-pert-eval-bgr-blur-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1      completed, 00:02:10, 14/15 = 0.9333
763739  bgr-pert-eval-bgr-brightness-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1 completed, 00:02:32, 13/15 = 0.8667
763740  bgr-pert-eval-bgr-occlusion-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1  completed, 00:02:59, 6/15 = 0.4000
763741  bgr-pert-eval-bgr-shift-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1      completed, 00:02:16, 15/15 = 1.0000
763742  bgr-pert-eval-random-identity-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1 completed, 00:02:28, 14/15 = 0.9333
763743  bgr-pert-eval-random-blur-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1     completed, 00:03:33, 14/15 = 0.9333
763744  bgr-pert-eval-random-brightness-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1 completed, 00:02:29, 14/15 = 0.9333
763745  bgr-pert-eval-random-occlusion-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1 completed, 00:02:52, 7/15 = 0.4667
763746  bgr-pert-eval-random-shift-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1     completed, 00:02:27, 14/15 = 0.9333
763732  bgr-pert-eval-official-identity-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1  completed, 00:02:06, 14/15 = 0.9333
763733  bgr-pert-eval-official-blur-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1      completed, 00:02:11, 14/15 = 0.9333
763734  bgr-pert-eval-official-brightness-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1 completed, 00:02:07, 14/15 = 0.9333
763735  bgr-pert-eval-official-occlusion-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1  completed, 00:02:54, 7/15 = 0.4667
763736  bgr-pert-eval-official-shift-cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1      completed, 00:02:08, 14/15 = 0.9333
```

Prep completed at 2026-06-03T03:41:20+01:00. Local prep metadata is mirrored
under `results/openvla_oft_cleanmix_p4096_v1/`. The original diagnostic TFDS
roots are valid but imbalanced: BGR has 82 episodes / 5,248 steps, while random
has 76 episodes / 4,864 steps because random has only 512 available occlusion
perturbation examples. Treat this chain as informational; use the
common-availability repair below for a fair p4096 matched comparison.

Clean eval logs are mirrored under
`results/openvla_oft_goal_adapt_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1/`.
BGR and matched random again tie clean at 14/15 each. Perturbation evals are
complete, but this chain remains imbalanced because the original random TFDS
has fewer perturbation examples than BGR.

Perturbation logs are mirrored under
`results/openvla_oft_perturb_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1/`.
The local summary gives identity 0.9333 for official, BGR, and random. Mean
perturbed success across blur, brightness, occlusion, and shift is 0.8167 for
official, 0.8000 for BGR, and 0.8167 for random. This is still not a fair
matched p4096 BGR/random perturbation comparison; use the common-availability
repair below for that.

TFDS roots:

```text
/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p4096_v1
/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_cleanmix_p4096_v1
```

Local collection paths:

```text
results/openvla_oft_cleanmix_p4096_v1/
results/openvla_oft_goal_adapt_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1/
results/openvla_oft_perturb_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1/
```

## Completed OpenVLA-OFT p4096 Common-Availability Repair

Launched on 2026-06-03 after the p4096 scale diagnostic revealed a perturbation
availability imbalance: BGR rendered 4,096/4,096 perturb examples, but matched
random rendered 3,584/4,096 because the random replay manifest contains only
eight occlusion episodes, or 512 occlusion examples, at 64 steps each. The
original p4096 chain remains an imbalanced diagnostic. This repair creates a
fair matched comparison from the already-rendered examples by capping both
methods to the common perturbation availability:

```text
blur:       1,024 examples
brightness: 1,024 examples
occlusion:    512 examples
shift:      1,024 examples
total:      3,584 perturb examples per method
```

Clean examples remain the native clean-anchor counts, matching prior clean-mix
runs: 1,152 BGR clean examples and 1,280 random clean examples. The fair repair
therefore tests common perturbation availability rather than equal total clean
steps.

Submitted common-availability chain:

```text
763748  bgr-cleanmix-commonavail-p4096                                                        prep/filter/combine/TFDS export
763749  bgr-goal-adapt-bgr-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1     afterok:763748
763750  bgr-goal-merge-bgr-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1     afterok:763749
763751  bgr-goal-eval-bgr-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1      afterok:763750
763752  bgr-goal-adapt-random-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1  afterok:763748, afterok:763749
763753  bgr-goal-merge-random-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1  afterok:763752
763754  bgr-goal-eval-random-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1   afterok:763753
763755--763759  official perturb evals for common-availability tag, submitted immediately
763760--763764  BGR perturb evals for common-availability tag, afterok:763750
763765--763769  random perturb evals for common-availability tag, afterok:763753
```

Live status checked 2026-06-03:

```text
763748  bgr-cleanmix-commonavail-p4096  completed, 00:07:59; local prep metadata mirrored
763749  bgr-goal-adapt-bgr-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1     completed, 00:01:48
763750  bgr-goal-merge-bgr-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1     completed, 00:01:09
763751  bgr-goal-eval-bgr-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1      completed, 00:02:06, 14/15 = 0.9333
763752  bgr-goal-adapt-random-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1  completed, 00:01:47
763753  bgr-goal-merge-random-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1  completed, 00:01:20
763754  bgr-goal-eval-random-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1   completed, 00:02:15, 15/15 = 1.0000
763755  bgr-pert-eval-official-identity-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1    completed, 00:02:28, 14/15 = 0.9333
763756  bgr-pert-eval-official-blur-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1        completed, 00:02:10, 14/15 = 0.9333
763757  bgr-pert-eval-official-brightness-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1  completed, 00:02:31, 14/15 = 0.9333
763758  bgr-pert-eval-official-occlusion-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1   completed, 00:02:53, 7/15 = 0.4667
763759  bgr-pert-eval-official-shift-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1       completed, 00:02:06, 14/15 = 0.9333
763760  bgr-pert-eval-bgr-identity-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1         completed, 00:02:04, 14/15 = 0.9333
763761  bgr-pert-eval-bgr-blur-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1             completed, 00:02:08, 14/15 = 0.9333
763762  bgr-pert-eval-bgr-brightness-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1       completed, 00:04:09, 13/15 = 0.8667
763763  bgr-pert-eval-bgr-occlusion-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1        completed, 00:02:56, 7/15 = 0.4667
763764  bgr-pert-eval-bgr-shift-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1            completed, 00:01:55, 15/15 = 1.0000
763765  bgr-pert-eval-random-identity-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1      completed, 00:02:23, 15/15 = 1.0000
763766  bgr-pert-eval-random-blur-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1          completed, 00:03:18, 14/15 = 0.9333
763767  bgr-pert-eval-random-brightness-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1    completed, 00:02:30, 14/15 = 0.9333
763768  bgr-pert-eval-random-occlusion-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1     completed, 00:03:26, 7/15 = 0.4667
763769  bgr-pert-eval-random-shift-cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1         completed, 00:01:56, 15/15 = 1.0000
```

Common-availability prep completed at 2026-06-03T03:48:15+01:00. Local prep
metadata is mirrored under `results/openvla_oft_cleanmix_p4096_commonavail_v1/`.
Both methods have 3,584 perturb examples with matched perturbation family
counts. BGR TFDS has 74 episodes / 4,736 steps; random TFDS has 76 episodes /
4,864 steps because the native clean-anchor counts remain 1,152 vs. 1,280.
Perturbation logs are mirrored under
`results/openvla_oft_perturb_eval_cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1/`;
official identity is 0.9333 and official mean perturbed is 0.8167. Clean-eval
logs are mirrored under
`results/openvla_oft_goal_adapt_eval_cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1/`;
BGR clean is 14/15 = 0.9333; random clean is 15/15 = 1.0000. Mean perturbed
success is 0.8167 for official, 0.8167 for BGR, and 0.8333 for random. The
fair p4096 repair is therefore a negative/diagnostic OpenVLA-OFT result rather
than paper-facing evidence for BGR.

Common-availability collection paths:

```text
results/openvla_oft_cleanmix_p4096_commonavail_v1/
results/openvla_oft_goal_adapt_eval_cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1/
results/openvla_oft_perturb_eval_cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1/
```

## Completed OpenVLA-OFT p4096 Preregistered Image-Augmentation Audit

Launched on 2026-06-04 and completed on 2026-06-05 as the preregistered
learned-policy promotion attempt after the fair common-availability repair. The
adaptation recipe used the fair p4096 common-availability TFDS roots, official
OpenVLA-OFT LIBERO-Goal statistics, identity-LoRA initialization, image
augmentation, `ADAPT_STEPS=500`, `LR=5e-7`, and fixed 10-task x 10-trial
identity/perturbation evals. Promotion was preregistered to require BGR to beat
both matched random and the official OpenVLA-OFT checkpoint on the fixed
non-identity perturbation total by at least 10/400 episodes and at least 0.02
absolute success, while not trailing clean identity by more than 1/100.

The initial official identity job (`765765`) failed immediately because the
generated live scripts used a stale home-directory `LIBERO_ROOT`, which is
absent on the compute node. Pending eval jobs generated with that bad path were
canceled, and repaired evals were submitted with the same checkpoint paths,
perturbation families, 10-task x 10-trial protocol, seed, and promotion gate,
changing only the live LIBERO installation path.

Completed repaired perturb eval jobs:

```text
765782  official identity  completed, 99/100 = 0.9900
765785  official blur      completed, 97/100 = 0.9700
765787  official brightness completed, 98/100 = 0.9800
765791  official occlusion completed, 74/100 = 0.7400
765795  official shift     completed, 98/100 = 0.9800
765783  BGR identity       completed, 99/100 = 0.9900
765786  BGR blur           completed, 97/100 = 0.9700
765789  BGR brightness     completed, 98/100 = 0.9800
765792  BGR occlusion      completed, 74/100 = 0.7400
765794  BGR shift          completed, 97/100 = 0.9700
765781  random identity    completed, 98/100 = 0.9800
765784  random blur        completed, 98/100 = 0.9800
765788  random brightness  completed, 99/100 = 0.9900
765790  random occlusion   completed, 75/100 = 0.7500
765793  random shift       completed, 96/100 = 0.9600
```

Summary:

| Method | Identity | Blur | Brightness | Occlusion | Shift | Non-identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BGR prereg p4096 | 0.9900 | 0.9700 | 0.9800 | 0.7400 | 0.9700 | 366/400 = 0.9150 |
| Official OpenVLA-OFT | 0.9900 | 0.9700 | 0.9800 | 0.7400 | 0.9800 | 367/400 = 0.9175 |
| Random prereg p4096 | 0.9800 | 0.9800 | 0.9900 | 0.7500 | 0.9600 | 368/400 = 0.9200 |

Interpretation: BGR satisfies the clean-identity side condition by tying the
official checkpoint and beating matched random by one episode, but it fails the
actual promotion gate because it trails official by one non-identity episode and
matched random by two. This is therefore another negative/diagnostic
OpenVLA-OFT result, not paper-facing evidence for a robotics fine-tuning claim.
The local summaries are stored under
`results/openvla_oft_perturb_eval_cleanmix_p4096_commonavail_step50500_lr5em7_identitylora_imageaug_officialtrainstats_prereg_fullgoal10x10_v1/`.

## Preregistered OpenVLA-OFT Weighted Perturbation Curriculum

This is the next learned-policy intervention after the negative p4096
image-augmentation audit. It is not another step-count or seed rerun: the
training distribution changes by repeating perturbation examples under separate
TFDS `mix_source` labels before OpenVLA-OFT adaptation. The fixed recipe uses
2,048 unique perturbation examples per method, eight episodes per perturbation
family, `PERTURB_REPEAT=3`, official OpenVLA-OFT LIBERO-Goal statistics,
identity-LoRA, image augmentation, `ADAPT_STEPS=500`, `LR=5e-7`, and the same
10-task x 10-trial identity and visual-perturbation evals.

Fixed prep command:

```bash
scripts/queue_openvla_oft_preregistered_weighted_perturb.sh --prep-only --submit-prep
```

Submitted weighted prep on 2026-06-05 after the preregistration script was
pushed. It completed successfully:

```text
766799  bgr-cleanmix-prep-p2048unique_perturbrepeat3_prereg  COMPLETED  exit=0:0  elapsed=00:21:30
```

Prep metadata confirms that the weighted perturbation subsets are matched after
the `PERTURB_REPEAT=3` curriculum expansion. BGR exports 114 episodes and 7,296
steps; random exports 116 episodes and 7,424 steps. The combine summaries are:

```text
BGR examples=7296
BGR examples_by_source clean=1152 perturb_1=2048 perturb_2=2048 perturb_3=2048
BGR perturbation_types blur=1536 brightness=1536 identity=1152 occlusion=1536 shift=1536
random examples=7424
random examples_by_source clean=1280 perturb_1=2048 perturb_2=2048 perturb_3=2048
random perturbation_types blur=1536 brightness=1536 identity=1280 occlusion=1536 shift=1536
```

The non-identity perturbation-family counts match exactly between BGR and
random after weighting: blur, brightness, occlusion, and shift each contribute
1,536 examples per method. The clean identity count differs because the native
clean-anchor render has 1,152 BGR and 1,280 random examples.

The live submission uses the cluster user's writable workspace, source
artifacts, Hugging Face cache, OpenVLA-OFT checkout, and LIBERO checkout:

```bash
REMOTE_PROJECT=/work/<user>/bgr
REMOTE_RUN_ROOT=/work/<user>/bgr/runs
REMOTE_LOG_DIR=/work/<user>/bgr/logs
REMOTE_HF_HOME=/work/<user>/cache_home/huggingface
OPENVLA_OFT_ROOT=/work/<user>/external_validation/openvla_oft_smoke_746850/openvla-oft
LIBERO_ROOT=/work/<user>/external_validation/openvla_oft_smoke_746850/LIBERO
SOURCE_ARTIFACT_ROOT=/work/<user>/dreamaudit_jobs/artifacts
PERTURB_MANIFEST=/work/<user>/bgr/results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl
```

Fixed adaptation command after prep succeeds:

```bash
PREP_DEPENDENCY=afterok:<prep_job> \
scripts/queue_openvla_oft_preregistered_weighted_perturb.sh --adapt-only --submit-adapt
```

Submitted weighted adaptation on 2026-06-05 with `PREP_DEPENDENCY=afterok:766799`
and `GIT_PULL=0` because the live remote checkout is intentionally treated as
dirty. Slurm job chain:

```text
bgr train=766805 merge=766806 clean_eval=766807
random train=766808 merge=766809 clean_eval=766810
```

At submission audit time, BGR train `766805` was running on `c1-g4-03`; BGR
merge/eval and the random chain were pending on their declared dependencies.

Fixed perturbation command after BGR/random merge jobs exist:

```bash
BGR_DEPENDENCY=afterok:<bgr_merge> \
RANDOM_DEPENDENCY=afterok:<random_merge> \
scripts/queue_openvla_oft_preregistered_weighted_perturb.sh --perturb-only --submit-perturb
```

Submitted the fixed perturbation eval on 2026-06-05 with
`BGR_DEPENDENCY=afterok:766806` and `RANDOM_DEPENDENCY=afterok:766809`.
Per-method perturbations are serialized by the queue script. Slurm job map:

```text
official identity=766817 blur=766818 brightness=766819 occlusion=766820 shift=766821
bgr identity=766822 blur=766823 brightness=766824 occlusion=766825 shift=766826
random identity=766827 blur=766828 brightness=766829 occlusion=766830 shift=766831
```

At submission audit time, official identity `766817` was running. Official
blur/brightness/occlusion/shift were pending on the previous official
perturbation job. BGR perturbation jobs were pending on BGR merge `766806` and
their per-method serial dependencies; random perturbation jobs were pending on
random merge `766809` and their per-method serial dependencies.

Follow-up audit on 2026-06-05: BGR train `766805` completed with exit `0:0`
after 6m14s and produced the expected adapted checkpoint files. BGR merge
`766806`, random train `766808`, and official identity perturbation eval
`766817` were running; BGR clean eval `766807`, random merge/eval
`766809`/`766810`, and all BGR/random perturbation evals remained pending on
their declared dependencies. No weighted clean or perturbation summary was
available yet, so there are no paper-facing results to incorporate from this
intervention.

Later 2026-06-05 audit: BGR merge `766806`, random train `766808`, and random
merge `766809` completed with exit `0:0`. BGR clean eval `766807`, random clean
eval `766810`, official identity perturbation eval `766817`, BGR identity
perturbation eval `766822`, and random identity perturbation eval `766827` were
running. The remaining perturbation jobs were pending on their per-method
serial dependencies. No weighted clean or perturbation `summary.csv`/`summary.json`
files were present yet.

The clean identity evals then completed and were summarized on the cluster:
BGR clean identity is 99/100 and matched random clean identity is 99/100. The
identity perturbation logs also completed and the partial perturbation summary
shows BGR 99/100, official 99/100, and random 99/100. These results satisfy the
clean-floor part of the preregistered gate but do not address the non-identity
perturbation requirement. At the same audit, the blur perturbation jobs were
running for official (`766818`), BGR (`766823`), and random (`766828`), while
brightness, occlusion, and shift remained pending behind per-method serial
dependencies. The cluster-generated partial summaries retain live log paths and
are not anonymous package artifacts.

The next partial perturbation summary added the blur rows: BGR 98/100,
official 97/100, and matched random 99/100. This is not promotable because BGR
trails matched random by one episode on the first non-identity family. The full
promotion gate remains undecided until brightness, occlusion, and shift finish,
but BGR must now overcome a one-episode deficit to random while still beating
both random and official by at least 10/400 total non-identity episodes and at
least 0.02 absolute success.

The following completed family, brightness, gives BGR 99/100, official 98/100,
and matched random 99/100. The completed non-identity subtotal over blur and
brightness is therefore BGR 197/200, official 195/200, and matched random
198/200. This remains an incomplete audit rather than paper-positive evidence:
occlusion and shift are still required for the fixed 400-episode non-identity
gate, and the completed subtotal still trails matched random by one episode.

Occlusion then completed as BGR 75/100, official 74/100, and matched random
75/100. BGR and official shift also completed as 95/100 and 98/100. A later
Athena poll on 2026-06-06 03:05 PDT / 11:05 BST showed matched-random shift job
`766831` completed successfully, and log inspection found 97/100 success. The
complete weighted audit is therefore BGR 367/400, official 367/400, and
matched random 370/400 over non-identity perturbations, with identity at
99/100 for all three methods. This fails the preregistered +10/400 episode and
+0.02 absolute-success margins over both official and matched random. The paper
should therefore continue to treat this intervention as a negative
OpenVLA/LIBERO audit, not a robotics fine-tuning result.

The complete compact artifact is
`results/openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv`.
It was regenerated from the completed Athena text logs using
`scripts/summarize_openvla_oft_perturb_eval.py`, then stored locally without raw
log paths for double-blind hygiene.

Promotion gate: weighted BGR must beat weighted matched random and the official
checkpoint on the fixed non-identity perturbation total by at least 10/400
episodes and at least 0.02 absolute success rate, while not trailing clean
identity by more than 1/100. If prep metadata shows unmatched BGR/random
perturbation-family counts after weighting, the result is an audit only and
cannot be promoted.

## Preregistered OpenVLA-OFT Proximal-Anchor Adaptation

This is the next learned-policy intervention after the negative weighted
perturbation curriculum. It changes the optimization objective rather than the
data mix: the fixed recipe reuses the already prepared weighted perturbation
TFDS roots, keeps official OpenVLA-OFT LIBERO-Goal statistics, identity-LoRA,
image augmentation, `ADAPT_STEPS=500`, `LR=5e-7`, and 10-task x 10-trial
identity/visual-perturbation evals, but adds `PROXIMAL_ANCHOR_L2=1.0` to the
OpenVLA-OFT training objective.

The implementation is in
`scripts/queue_openvla_oft_preregistered_proximal_anchor.sh` plus the
`PROXIMAL_ANCHOR_L2` wrapper path in `scripts/queue_openvla_oft_goal_adapt.sh`.
The generated remote trainer snapshots all trainable parameters immediately
after resuming the official LIBERO-Goal checkpoint and penalizes L2 deviation
from those initial values while fitting the BGR-boundary or matched-random
examples. This is a proximal residual adaptation test: it can only become
paper-positive if the BGR branch improves perturbation success without losing
the official clean behavior and without being matched by the random branch.

Fixed adaptation command:

```bash
scripts/queue_openvla_oft_preregistered_proximal_anchor.sh --adapt-only --submit-adapt
```

Fixed perturbation command after BGR/random merge jobs exist:

```bash
BGR_DEPENDENCY=afterok:<bgr_merge> \
RANDOM_DEPENDENCY=afterok:<random_merge> \
scripts/queue_openvla_oft_preregistered_proximal_anchor.sh --perturb-only --submit-perturb
```

Promotion gate: proximal-anchor BGR must beat proximal-anchor matched random
and the official checkpoint on the fixed non-identity perturbation total by at
least 10/400 episodes and at least 0.02 absolute success rate, while not
trailing clean identity by more than 1/100. A tie, one-episode edge, or
official/random lead remains a negative audit.

Launched on 2026-06-05 13:18 PDT / 21:18 BST after verifying the remote
TFDS roots, OpenVLA-OFT checkout, Python environment, `torchrun`, and official
statistics file on `athena`.

Adaptation/merge/clean-eval jobs:

```text
BGR:    train=767128 merge=767129 clean_eval=767130
random: train=767131 merge=767132 clean_eval=767133
```

Fixed perturbation eval jobs:

```text
official: identity=767134 blur=767135 brightness=767136 occlusion=767137 shift=767138
BGR:      identity=767139 blur=767140 brightness=767141 occlusion=767142 shift=767143
random:   identity=767144 blur=767145 brightness=767146 occlusion=767147 shift=767148
```

The BGR perturbation rows were submitted with dependency `afterok:767129`, and
the matched-random perturbation rows were submitted with dependency
`afterok:767132`; each method serializes identity through shift. The immediate
Slurm audit showed all rows pending, with BGR/random perturb rows held on
dependencies and official perturb rows serialized by method. No result is
available yet, and this remains an audit until the fixed gate above is checked.

Fresh Athena poll on 2026-06-05 16:25 PDT / 2026-06-06 00:25 BST:

```text
767128 PENDING (ReqNodeNotAvail, UnavailableNodes:c1-g4-[01-05],c2-g4-[13,16-26],c2-g8-[01-03,05-08])
767134 PENDING (ReqNodeNotAvail, UnavailableNodes:c1-g4-[01-05],c2-g4-[13,16-26],c2-g8-[01-03,05-08])
767129-767133 PENDING (Dependency)
767135-767148 PENDING (Dependency)
StartTime=2026-06-07T14:27:51 for 767128 and 767134
TresPerNode=gres/gpu:a6000:1 for 767128 and 767134
```

`sacct` reported `PENDING`, `00:00:00` elapsed, and unknown start/end times for
all jobs `767128`-`767148`; there is still no proximal-anchor summary to sync
or promote. A scheduler diagnostic found idle `g2` nodes, but those nodes expose
`gpu:a4000`, while these OpenVLA jobs retain a typed A6000 GRES request. Do not
resubmit this route as a generic/A4000 retry unless the memory footprint is
separately changed and preregistered.

Follow-up Athena poll on 2026-06-06 03:05 PDT / 11:05 BST showed BGR train job
`767128` failed with exit code `1:0`. Log inspection found the failure at
`normalized_loss.backward()` in the proximal-anchor wrapper:
PyTorch DDP reported `Expected to mark a variable ready only once` for
`base_model.model.vision_backbone.fused_featurizer.attn_pool.mlp.fc2.lora_B.default.weight`.
BGR merge job `767129`, random adapt job `767131`, and downstream BGR/random
perturb jobs were dependency-held with `DependencyNeverSatisfied`. Official
identity job `767134` completed, but the proximal route has no valid BGR/random
summary and remains non-evidence until the wrapper is repaired under the same
preregistered protocol or the route is retired.

Execution repair on 2026-06-06: commit `cfecfd9` keeps the same
proximal-anchor objective but avoids the DDP double-ready path by logging the
proximal metric under `torch.no_grad()` and adding the analytically equivalent
proximal gradient to each trainable parameter after the normal
`normalized_loss.backward()` call. This is an execution repair, not a tuning
change. The repaired tag is `proxanchor_l2_1em0_ddpgradfix_v1`, giving
adaptation artifact
`openvla_oft_goal_adapt_eval_cleanmix_p2048unique_perturbrepeat3_prereg_proxanchor_l2_1em0_ddpgradfix_v1_step50500_lr5em7_identitylora_imageaug_officialtrainstats_v1`
and perturbation artifact
`openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_proxanchor_l2_1em0_ddpgradfix_v1_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1`.

Repaired adaptation/merge/clean-eval jobs:

```text
BGR:    train=767657 merge=767658 clean_eval=767659
random: train=767660 merge=767661 clean_eval=767662
```

`sacct` showed all six repaired adaptation/merge/clean-eval jobs completed with
exit code `0:0`. This confirms the wrapper repair gets through the previous
DDP failure point, but it is not paper evidence because the perturbation gate is
still incomplete.

Repaired fixed perturbation eval jobs:

```text
official: identity=767663 blur=767664 brightness=767665 occlusion=767666 shift=767667
BGR:      identity=767674 blur=767675 brightness=767676 occlusion=767677 shift=767678
random:   identity=767681 blur=767682 brightness=767683 occlusion=767684 shift=767685
```

All repaired perturbation jobs completed with exit code `0:0`. The first
BGR/random perturb submission attempt used a bare merge job id and produced an
invalid `afterany` dependency for BGR identity; that job (`767668`) was
canceled before execution. A retry with `afterok:` dependencies was rejected
because the merge jobs had already completed, so BGR and random perturb evals
were submitted only after their checkpoints existed.

Compact local artifacts:

```text
results/openvla_oft_goal_adapt_eval_cleanmix_p2048unique_perturbrepeat3_prereg_proxanchor_l2_1em0_ddpgradfix_v1_step50500_lr5em7_identitylora_imageaug_officialtrainstats_v1/summary.csv
results/openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_proxanchor_l2_1em0_ddpgradfix_v1_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv
```

Summary:

| Method | Identity | Blur | Brightness | Occlusion | Shift | Non-identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BGR proximal anchor | 98/100 | 98/100 | 99/100 | 73/100 | 98/100 | 368/400 = 0.9200 |
| Official OpenVLA-OFT | 99/100 | 97/100 | 98/100 | 74/100 | 98/100 | 367/400 = 0.9175 |
| Random proximal anchor | 98/100 | 99/100 | 98/100 | 73/100 | 98/100 | 368/400 = 0.9200 |

Interpretation: the repaired proximal-anchor route clears the execution bug but
does not clear the learned-policy promotion gate. BGR ties matched random on
non-identity perturbations and beats the official checkpoint by only 1/400
episode (+0.0025), far below the preregistered +10/400 and +0.02 margins. Clean
identity is within the side condition but trails official by one episode. This
result is incorporated into `paper/main.tex` as another negative OpenVLA audit,
not as robotics fine-tuning evidence.

Result ingestion helper:

```bash
scripts/sync_openvla_oft_proximal_anchor_results.sh --poll --no-check
scripts/sync_openvla_oft_proximal_anchor_results.sh --sync
```

The helper polls the fixed jobs, checks the remote compact summaries under
the configured remote run root, prints selected `scontrol show job -dd` details
including `TresPerNode`, syncs only `summary.csv` files when present, and then
runs the local perturbation and readiness gates. For the live internal cluster
workspace, set `REMOTE_RUN_ROOT=/work/<user>/bgr/runs`. For this repaired run,
the remote eval jobs did not write compact summaries directly; the local
compact CSVs above were generated from copied `*.txt` eval logs with
`scripts/summarize_openvla_oft_eval.py` and
`scripts/summarize_openvla_oft_perturb_eval.py`, then stripped to
double-blind-safe columns.

## Preregistered OpenVLA-OFT Perturb-Only Anchored Adaptation

This is the next learned-policy route after the completed negative clean-mix,
weighted clean-mix, and proximal-anchor clean-mix audits. It changes the
training data/objective combination: the RLDS datasets contain only rendered
boundary-band perturbation examples, with no clean anchor episodes mixed in,
and the trainer uses a stronger official-checkpoint proximal anchor
(`PROXIMAL_ANCHOR_L2=5.0`) to protect clean identity behavior.

The implementation is in
`scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh`. The fixed
recipe renders matched BGR-boundary and random-balanced perturbation examples
from `results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl`,
using `MAX_PERTURB_EXAMPLES=2048`, `PERTURB_EPISODES_PER_FAMILY=8`, and
`MAX_STEPS_PER_EPISODE=64`; exports perturb-only TFDS roots; adapts from the
official OpenVLA-OFT LIBERO-Goal checkpoint with identity-LoRA, official
dataset statistics, image augmentation, `ADAPT_STEPS=300`, and `LR=2e-7`; and
then runs the same official/BGR/random 10-task x 10-trial identity, blur,
brightness, occlusion, and shift evaluation.

Fixed prep command:

```bash
scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh --prep-only --submit-prep
```

Fixed adaptation command after prep succeeds:

```bash
TRAIN_DEPENDENCY=afterok:<prep_job> \
scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh --adapt-only --submit-adapt
```

Fixed perturbation command after BGR/random merge jobs exist:

```bash
BGR_DEPENDENCY=afterok:<bgr_merge> \
RANDOM_DEPENDENCY=afterok:<random_merge> \
scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh --perturb-only --submit-perturb
```

Promotion gate: perturb-only anchored BGR must beat both perturb-only anchored
matched random and the official checkpoint on the fixed non-identity
perturbation total by at least 10/400 episodes and at least 0.02 absolute
success rate, while not trailing clean identity by more than 1/100. A tie,
one-episode edge, official lead, or matched-random lead remains an audit.

Prep submission audit: submitted after preregistration commit `8b69ac7` on
2026-06-07 using the live cluster workspace:

```text
job=767789
REMOTE_PROJECT=/work/<user>/bgr
REMOTE_RUN_ROOT=/work/<user>/bgr/runs
REMOTE_HF_HOME=/work/<user>/cache_home/huggingface
OPENVLA_OFT_ROOT=/work/<user>/external_validation/openvla_oft_smoke_746850/openvla-oft
LIBERO_ROOT=/work/<user>/external_validation/openvla_oft_smoke_746850/LIBERO
stdout=/work/<user>/bgr/logs/bgr-perturbonly-prep-p2048unique_perturbonly_anchor_prereg-767789.out
```

Initial Slurm poll showed job `767789` running on `c1-g4-04` with
`gres/gpu:a6000:1`. The prepared TFDS roots, if the job succeeds, are:

```text
/work/<user>/bgr/runs/openvla_oft_tfds_libero_goal_bgr_perturbonly_p2048unique_perturbonly_anchor_prereg_v1
/work/<user>/bgr/runs/openvla_oft_tfds_libero_goal_random_perturbonly_p2048unique_perturbonly_anchor_prereg_v1
```

Dependent adaptation and perturbation eval submission audit: after the prep
poll above, the adaptation stage was queued with
`TRAIN_DEPENDENCY=afterok:767789` and `GIT_PULL=0`. The BGR chain is
train/merge/clean-eval jobs `767790`/`767791`/`767792`; the matched-random
chain is train/merge/clean-eval jobs `767793`/`767794`/`767795`. The fixed
perturbation evals were then queued with `BGR_DEPENDENCY=afterok:767791` and
`RANDOM_DEPENDENCY=afterok:767794`: official jobs `767796`-`767800`, BGR jobs
`767801`-`767805`, and matched-random jobs `767806`-`767810`. An immediate
Slurm poll showed prep `767789` and official identity `767796` running; all
adaptation jobs and BGR/random perturb evals were dependency-pending. These
queued jobs are not paper evidence unless compact summaries later clear the
fixed promotion gate. Poll and sync compact summaries with:

```bash
REMOTE_RUN_ROOT=/work/<user>/bgr/runs \
scripts/sync_openvla_oft_perturb_only_anchor_results.sh --poll

REMOTE_RUN_ROOT=/work/<user>/bgr/runs \
scripts/sync_openvla_oft_perturb_only_anchor_results.sh --sync
```

A helper poll at 2026-06-07 22:08:57 BST showed prep `767789` completed
cleanly at 22:06:54 BST, BGR adaptation `767790` running on `c1-g4-04`, and
official identity eval `767796` running on `c2-g4-24`; adapt and perturb compact
summary CSVs were still missing.

A later helper poll at 2026-06-07 22:19:57 BST showed BGR train/merge
`767790`/`767791`, random train/merge `767793`/`767794`, and official identity
`767796` completed with exit code `0:0`. BGR clean eval `767792`, random clean
eval `767795`, BGR identity perturb eval `767801`, random identity perturb eval
`767806`, and official blur `767797` were running; compact summaries were still
missing.

Clean adaptation eval logs were summarized locally into
`results/openvla_oft_goal_adapt_eval_p2048unique_perturbonly_anchor_prereg_perturbonly_proxanchor_l2_5em0_step50300_lr2em7_identitylora_imageaug_officialtrainstats_v1/summary.csv`.
BGR and matched random both score 99/100 clean episodes. This is a clean-floor
sanity check only; the learned-policy promotion gate remains pending until the
complete official/BGR/random perturbation summary exists.

Final perturbation jobs `767796`-`767810` completed with exit code `0:0`. The
remote compact summaries were not written at the expected paths, so
`scripts/sync_openvla_oft_perturb_only_anchor_results.sh --sync` copied the
remote eval logs and generated the compact local summary from those logs:

```text
results/openvla_oft_perturb_eval_p2048unique_perturbonly_anchor_prereg_perturbonly_proxanchor_l2_5em0_step50300_lr2em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv
```

Summary:

| Method | Identity | Blur | Brightness | Occlusion | Shift | Non-identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BGR perturb-only anchor | 99/100 | 98/100 | 99/100 | 76/100 | 98/100 | 371/400 = 0.9275 |
| Official OpenVLA-OFT | 99/100 | 97/100 | 98/100 | 74/100 | 98/100 | 367/400 = 0.9175 |
| Random perturb-only anchor | 99/100 | 99/100 | 99/100 | 76/100 | 98/100 | 372/400 = 0.9300 |

Interpretation: the perturb-only anchored route preserves the clean identity
side condition for all three methods, but it fails the learned-policy promotion
gate. BGR beats the official checkpoint by only 4/400 non-identity episodes
(+0.0100) and trails matched random by 1/400 (-0.0025), below the fixed
+10/400 and +0.02 margins. This is incorporated into the paper only as a
negative OpenVLA/LIBERO audit, not as robotics fine-tuning evidence.

## Completed OpenVLA-OFT p2048 Clean-Mix Scale-Up

Launched on 2026-06-02 after the p1024 offset-3 follow-up showed only a small
BGR-over-random perturbation edge and no official-checkpoint improvement. The
p2048 run tests whether doubling the perturbation subset again, to 2,048
rendered perturbation examples with eight perturbation episodes per family,
improves the matched adapted checkpoints or confirms that OpenVLA-OFT remains a
diagnostic audit in the paper.

Submitted prep/TFDS command:

```bash
TAG=p2048 \
MAX_PERTURB_EXAMPLES=2048 \
PERTURB_EPISODES_PER_FAMILY=8 \
TIME=12:00:00 \
scripts/queue_openvla_oft_clean_mix_prep.sh --submit
```

Slurm chain:

```text
763698  bgr-cleanmix-prep-p2048  completed, 00:19:18; TFDS roots valid
```

TFDS roots:

```text
/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p2048_v1
/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_cleanmix_p2048_v1
```

Completed prep outputs:

```text
BGR cleanmix examples:    3,200 = 1,152 clean + 2,048 perturb
Random cleanmix examples: 3,328 = 1,280 clean + 2,048 perturb
Perturbation balance: 512 examples each for blur, brightness, occlusion, and shift
BGR TFDS:    50 episodes, 3,200 steps, libero_goal_no_noops/1.0.0
Random TFDS: 52 episodes, 3,328 steps, libero_goal_no_noops/1.0.0
```

Local copied prep artifacts:

```text
results/openvla_oft_cleanmix_p2048_v1/render/bgr_clean_summary.json
results/openvla_oft_cleanmix_p2048_v1/render/random_clean_summary.json
results/openvla_oft_cleanmix_p2048_v1/render/bgr_perturb_summary.json
results/openvla_oft_cleanmix_p2048_v1/render/random_perturb_summary.json
results/openvla_oft_cleanmix_p2048_v1/render/bgr_cleanmix_summary.json
results/openvla_oft_cleanmix_p2048_v1/render/random_cleanmix_summary.json
results/openvla_oft_cleanmix_p2048_v1/tfds/bgr/bgr_export_summary.json
results/openvla_oft_cleanmix_p2048_v1/tfds/bgr/dataset_info.json
results/openvla_oft_cleanmix_p2048_v1/tfds/bgr/features.json
results/openvla_oft_cleanmix_p2048_v1/tfds/random/bgr_export_summary.json
results/openvla_oft_cleanmix_p2048_v1/tfds/random/dataset_info.json
results/openvla_oft_cleanmix_p2048_v1/tfds/random/features.json
results/openvla_oft_cleanmix_p2048_v1/slurm/bgr-cleanmix-prep-p2048-763698.out
```

Submitted clean adaptation/eval continuation:

```bash
TRAIN_DEPENDENCY=afterok:763698 \
TRAIN_DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b/dataset_statistics.json \
DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b/dataset_statistics.json \
FINETUNE_SCRIPT=vla-scripts/finetune_identity_lora.py \
ADAPT_STEPS=100 LR=1e-6 \
TAG=cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1 \
EVAL_ARTIFACT=openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1 \
BGR_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p2048_v1 \
RANDOM_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_cleanmix_p2048_v1 \
BGR_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1 \
RANDOM_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1 \
TRAIN_TIME=05:00:00 \
EVAL_TIME=06:00:00 \
scripts/queue_openvla_oft_goal_adapt.sh --submit
```

Submitted perturbation eval command:

```bash
TAG=cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1 \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1 \
BGR_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
RANDOM_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
BGR_DEPENDENCY=afterok:763700 \
RANDOM_DEPENDENCY=afterok:763703 \
EVAL_TIME=06:00:00 \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Completed Slurm jobs:

```text
763699  bgr-goal-adapt-bgr-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1        completed, 00:02:14
763700  bgr-goal-merge-bgr-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1        completed, 00:01:44
763701  bgr-goal-eval-bgr-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1         completed, 00:03:07, 14/15 = 0.9333
763702  bgr-goal-adapt-random-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1     completed, 00:02:14
763703  bgr-goal-merge-random-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1     completed, 00:01:40
763704  bgr-goal-eval-random-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1      completed, 00:03:54, 14/15 = 0.9333
763705  bgr-pert-eval-official-identity-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1   completed, 00:03:09, 14/15 = 0.9333
763706  bgr-pert-eval-official-blur-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1       completed, 00:03:14, 14/15 = 0.9333
763707  bgr-pert-eval-official-brightness-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1 completed, 00:03:12, 14/15 = 0.9333
763708  bgr-pert-eval-official-occlusion-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1  completed, 00:04:14, 7/15 = 0.4667
763709  bgr-pert-eval-official-shift-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1      completed, 00:03:12, 14/15 = 0.9333
763710  bgr-pert-eval-bgr-identity-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1        completed, 00:03:08, 14/15 = 0.9333
763711  bgr-pert-eval-bgr-blur-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1            completed, 00:03:13, 14/15 = 0.9333
763712  bgr-pert-eval-bgr-brightness-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1      completed, 00:03:08, 14/15 = 0.9333
763713  bgr-pert-eval-bgr-occlusion-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1       completed, 00:04:12, 7/15 = 0.4667
763714  bgr-pert-eval-bgr-shift-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1           completed, 00:03:05, 14/15 = 0.9333
763715  bgr-pert-eval-random-identity-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1     completed, 00:03:08, 14/15 = 0.9333
763716  bgr-pert-eval-random-blur-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1         completed, 00:03:16, 14/15 = 0.9333
763717  bgr-pert-eval-random-brightness-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1   completed, 00:02:50, 14/15 = 0.9333
763718  bgr-pert-eval-random-occlusion-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1    completed, 00:03:59, 6/15 = 0.4000
763719  bgr-pert-eval-random-shift-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1        completed, 00:02:49, 14/15 = 0.9333
```

Completed p2048 perturb calibration:

| Method | Identity | Blur | Brightness | Occlusion | Shift | Mean perturbed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Official OpenVLA-OFT | 0.9333 | 0.9333 | 0.9333 | 0.4667 | 0.9333 | 0.8167 |
| BGR-boundary | 0.9333 | 0.9333 | 0.9333 | 0.4667 | 0.9333 | 0.8167 |
| Matched random | 0.9333 | 0.9333 | 0.9333 | 0.4000 | 0.9333 | 0.8000 |

Local copied perturb artifacts:

```text
results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/logs/
results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/slurm/
results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv
results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.json
```

Completed p2048 clean adapted eval:

| Method | Clean success |
| --- | ---: |
| BGR-boundary | 14/15 = 0.9333 |
| Matched random | 14/15 = 0.9333 |

Local copied clean adapted artifacts:

```text
results/openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/logs/
results/openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/slurm/
results/openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv
results/openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.json
```

Collection/summarization commands used:

```bash
rsync -az athena:/work/anonymous/bgr/runs/openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/ \
  results/openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/
rsync -az athena:/work/anonymous/bgr/runs/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/ \
  results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/
PYTHONPATH=src:. python3 scripts/summarize_openvla_oft_eval.py \
  --method-log-dir bgr=results/openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/logs/bgr \
  --method-log-dir random=results/openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/logs/random \
  --out results/openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1
PYTHONPATH=src:. python3 scripts/summarize_openvla_oft_perturb_eval.py \
  --logs-root results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/logs \
  --out results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1
```

## Completed OpenVLA-OFT p2048 Offset-3 Perturbation Eval

Launched and completed on 2026-06-03 as the p2048 counterpart to the completed p1024 offset-3
follow-up. This reuses the completed p2048 BGR/random adapted checkpoints and
the unadapted official checkpoint, but evaluates seven trials per task starting
at init-state offset 3. Each method therefore gets 35 episodes per perturbation
and 175 episodes across identity, blur, brightness, occlusion, and shift. All
15 corrected jobs completed with Slurm exit code 0:0, and local summaries were
generated from the copied eval logs.

The first live submission, 763962--763976, was superseded before usable eval
logs were produced: jobs 763962--763969 failed immediately or after one second
because the generated Slurm scripts still used the anonymized output/cache root,
and jobs 763970--763976 were cancelled before starting. The corrected queue
script exposes live output/cache root overrides while retaining anonymized
defaults for reproducible command text.

Corrected submitted command:

```bash
TAG=cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1 \
BGR_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
RANDOM_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
EVAL_TRIALS=7 \
EVAL_INIT_STATE_OFFSET=3 \
EVAL_SEED=17 \
EVAL_TIME=08:00:00 \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Corrected Slurm chain:

```text
763978  bgr-pert-eval-official-identity-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials    completed, 00:04:08
763979  bgr-pert-eval-official-blur-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials        completed, 00:05:38
763980  bgr-pert-eval-official-brightness-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials  completed, 00:06:43
763981  bgr-pert-eval-official-occlusion-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials   completed, 00:07:17
763982  bgr-pert-eval-official-shift-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials       completed, 00:04:03
763983  bgr-pert-eval-bgr-identity-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials         completed, 00:05:32
763984  bgr-pert-eval-bgr-blur-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials             completed, 00:06:52
763985  bgr-pert-eval-bgr-brightness-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials       completed, 00:04:36
763986  bgr-pert-eval-bgr-occlusion-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials        completed, 00:07:23
763987  bgr-pert-eval-bgr-shift-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials            completed, 00:05:14
763989  bgr-pert-eval-random-identity-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials      completed, 00:04:25
763990  bgr-pert-eval-random-blur-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials          completed, 00:06:53
763991  bgr-pert-eval-random-brightness-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials    completed, 00:05:47
763992  bgr-pert-eval-random-occlusion-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials     completed, 00:07:06
763993  bgr-pert-eval-random-shift-cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials         completed, 00:05:00
```

Offset-3 visual perturbation results:

| Method | Identity | Blur | Brightness | Occlusion | Shift | Mean perturbed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BGR-boundary | 0.9714 | 1.0000 | 0.9714 | 0.5714 | 0.9429 | 0.8714 |
| Official OpenVLA-OFT | 0.9714 | 1.0000 | 1.0000 | 0.6000 | 0.9714 | 0.8929 |
| Random-balanced | 1.0000 | 0.9714 | 0.9714 | 0.5714 | 0.9714 | 0.8714 |

Pooled with the original 15-episode p2048 perturbation eval, excluding identity
perturbations:

| Method | Successes / episodes | Mean perturbed success |
| --- | ---: | ---: |
| BGR-boundary | 171 / 200 | 0.8550 |
| Official OpenVLA-OFT | 174 / 200 | 0.8700 |
| Random-balanced | 170 / 200 | 0.8500 |

Local copied artifacts:

```text
results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/logs/
results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/slurm/
results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/summary.csv
results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/summary.json
```

Collection/summarization commands used:

```bash
rsync -az athena:/work/anonymous/bgr/runs/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/ \
  results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/
PYTHONPATH=src:. python3 scripts/summarize_openvla_oft_perturb_eval.py \
  --logs-root results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/logs \
  --out results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1
```

## Completed OpenVLA-OFT p1024 Offset-3 Perturbation Eval

### `openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1`

Launched on 2026-06-02 to test whether the observed p1024 one-episode occlusion
edge survives out-of-sample LIBERO init states. This reuses the completed
p1024 BGR/random adapted checkpoints and the unadapted official checkpoint, but
evaluates seven additional trials per task starting at init-state offset 3
instead of repeating the original three-trial diagnostic. Each method therefore
gets 35 episodes per perturbation and 175 episodes across identity, blur,
brightness, occlusion, and shift. All 15 jobs completed with Slurm exit code
0:0, and local summaries were generated from the copied eval logs.

Submitted command:

```bash
TAG=cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1 \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1 \
BGR_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
RANDOM_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
EVAL_TRIALS=7 \
EVAL_INIT_STATE_OFFSET=3 \
EVAL_SEED=17 \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Slurm chain:

```text
763678  bgr-pert-eval-official-identity-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1   completed, 00:04:38
763679  bgr-pert-eval-official-blur-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1       completed, 00:04:48
763680  bgr-pert-eval-official-brightness-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1 completed, 00:04:37
763681  bgr-pert-eval-official-occlusion-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1  completed, 00:05:53
763682  bgr-pert-eval-official-shift-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1      completed, 00:05:55
763683  bgr-pert-eval-bgr-identity-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1        completed, 00:06:00
763684  bgr-pert-eval-bgr-blur-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1            completed, 00:04:46
763685  bgr-pert-eval-bgr-brightness-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1      completed, 00:04:39
763686  bgr-pert-eval-bgr-occlusion-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1       completed, 00:06:24
763687  bgr-pert-eval-bgr-shift-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1           completed, 00:04:27
763688  bgr-pert-eval-random-identity-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1     completed, 00:05:46
763689  bgr-pert-eval-random-blur-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1         completed, 00:06:02
763690  bgr-pert-eval-random-brightness-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1   completed, 00:04:50
763691  bgr-pert-eval-random-occlusion-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1    completed, 00:06:42
763692  bgr-pert-eval-random-shift-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1        completed, 00:04:24
```

Local copied artifacts:

```text
results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/logs/
results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/slurm/
results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/summary.csv
results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/summary.json
```

Collection/summarization command used:

```bash
rsync -az athena:/work/anonymous/bgr/runs/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/ \
  results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/
PYTHONPATH=src:. python3 scripts/summarize_openvla_oft_perturb_eval.py \
  --logs-root results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/logs \
  --out results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1
```

Offset-3 visual perturbation results:

| Method | Identity | Blur | Brightness | Occlusion | Shift | Mean perturbed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BGR-boundary | 0.9714 | 0.9714 | 1.0000 | 0.5714 | 0.9143 | 0.8643 |
| Official OpenVLA-OFT | 0.9714 | 1.0000 | 1.0000 | 0.6000 | 0.9714 | 0.8929 |
| Random-balanced | 1.0000 | 1.0000 | 0.9429 | 0.5429 | 0.9429 | 0.8571 |

Pooled with the original 15-episode p1024 perturbation eval, excluding identity
perturbations:

| Method | Successes / episodes | Mean perturbed success |
| --- | ---: | ---: |
| BGR-boundary | 171 / 200 | 0.8550 |
| Official OpenVLA-OFT | 174 / 200 | 0.8700 |
| Random-balanced | 168 / 200 | 0.8400 |

Interpretation: the offset-3 follow-up preserves only a very small BGR edge over
matched random selection and removes the earlier one-episode edge over the
unadapted official checkpoint. The honest OpenVLA framing is therefore:
diagnostic data plumbing and a small matched-random perturbation edge, not a
robotics fine-tuning improvement over the official checkpoint. Slurm logs include
post-final-result EGL destructor warnings, but all jobs completed with exit code
0:0 and each local eval log contains final results.

## Completed OpenVLA-OFT p1024 Perturbation-Mix Diagnostic

### `openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1`

Launched on 2026-06-02 after the completed p512 diagnostic tied matched random
on clean LIBERO-Goal success and had a small matched-random perturbation edge,
but only matched the unadapted official checkpoint. This p1024 run keeps the
same official-checkpoint resume, identity-LoRA wrapper, official training/eval
statistics, low learning rate, 100-step adaptation, clean eval, and matched
visual-perturbation protocol. The only intended experimental change was
doubling the perturbation subset again, from 512 to 1,024 rendered examples, by
allowing four perturbation episodes per family. Prep, combine, TFDS export,
matched BGR/random adaptation, clean evals, and matched visual-perturbation
evals all completed.

Submitted prep command:

```bash
TAG=p1024 \
MAX_PERTURB_EXAMPLES=1024 \
PERTURB_EPISODES_PER_FAMILY=4 \
TIME=10:00:00 \
scripts/queue_openvla_oft_clean_mix_prep.sh --submit
```

Submitted TFDS continuation command:

```bash
TAG=p1024 \
PREP_DEPENDENCY=afterany:763655 \
TIME=03:00:00 \
scripts/queue_openvla_oft_clean_mix_tfds_export.sh --submit
```

Submitted clean eval command:

```bash
TRAIN_DEPENDENCY=afterok:763656 \
TRAIN_DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b/dataset_statistics.json \
DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b/dataset_statistics.json \
FINETUNE_SCRIPT=vla-scripts/finetune_identity_lora.py \
ADAPT_STEPS=100 LR=1e-6 \
TAG=cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1 \
EVAL_ARTIFACT=openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1 \
BGR_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p1024_v1 \
RANDOM_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_cleanmix_p1024_v1 \
BGR_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1 \
RANDOM_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1 \
TRAIN_TIME=05:00:00 \
EVAL_TIME=06:00:00 \
scripts/queue_openvla_oft_goal_adapt.sh --submit
```

Submitted visual-perturbation eval command:

```bash
METHODS=bgr,random \
BGR_DEPENDENCY=afterok:763658 \
RANDOM_DEPENDENCY=afterok:763661 \
TAG=cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1 \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1 \
BGR_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
RANDOM_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
EVAL_TIME=06:00:00 \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Submitted official visual-perturbation comparator:

```bash
METHODS=official \
TAG=cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1 \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1 \
EVAL_TIME=06:00:00 \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Slurm chain:

```text
763655  bgr-cleanmix-prep-p1024                                                         completed
763656  bgr-cleanmix-tfds-p1024                                                         completed, TFDS roots already valid
763657  bgr-goal-adapt-bgr-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1    completed, checkpoint saved at step 50100
763658  bgr-goal-merge-bgr-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1    completed
763659  bgr-goal-eval-bgr-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1     completed
763660  bgr-goal-adapt-random-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1 completed, checkpoint saved at step 50100
763661  bgr-goal-merge-random-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1 completed
763662  bgr-goal-eval-random-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1  completed

763663  bgr perturb identity      completed
763664  bgr perturb blur          completed
763665  bgr perturb brightness    completed
763666  bgr perturb occlusion     completed
763667  bgr perturb shift         completed
763668  random perturb identity   completed
763669  random perturb blur       completed
763670  random perturb brightness completed
763671  random perturb occlusion  completed
763672  random perturb shift      completed
763673  official perturb identity completed
763674  official perturb blur     completed
763675  official perturb brightness completed
763676  official perturb occlusion completed
763677  official perturb shift    completed
```

Local copied artifacts:

```text
results/openvla_oft_cleanmix_p1024_v1/render/bgr_cleanmix_summary.json
results/openvla_oft_cleanmix_p1024_v1/render/random_cleanmix_summary.json
results/openvla_oft_cleanmix_p1024_v1/tfds/bgr_export_summary.json
results/openvla_oft_cleanmix_p1024_v1/tfds/random_export_summary.json
results/openvla_oft_cleanmix_p1024_v1/slurm/bgr-cleanmix-prep-p1024-763655.out
results/openvla_oft_cleanmix_p1024_v1/slurm/bgr-cleanmix-tfds-p1024-763656.out
results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/slurm/bgr-goal-adapt-bgr-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1-763657.out
results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/slurm/bgr-goal-merge-bgr-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1-763658.out
results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/slurm/bgr-goal-eval-bgr-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1-763659.out
results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/slurm/bgr-goal-adapt-random-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1-763660.out
results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/slurm/bgr-goal-merge-random-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1-763661.out
results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/slurm/bgr-goal-eval-random-cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1-763662.out
results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/logs/
results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv
results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.json
results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/logs/
results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/slurm/
results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv
results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.json
```

Prepared p1024 dataset sizes:

| Method | Rendered clean | Rendered perturb | TFDS episodes | TFDS steps |
|---|---:|---:|---:|---:|
| BGR-boundary | 1,152 | 1,024 | 34 | 2,176 |
| Random-balanced | 1,280 | 1,024 | 36 | 2,304 |

Clean LIBERO-Goal eval results:

| Method | Episodes | Successes | Success rate | Task success rates |
|---|---:|---:|---:|---|
| BGR-boundary | 15 | 14 | 0.9333 | `[1.0, 0.6666666666666666, 1.0, 1.0, 1.0]` |
| Random-balanced | 15 | 14 | 0.9333 | `[1.0, 0.6666666666666666, 1.0, 1.0, 1.0]` |

Visual-perturbation results:

| Method | Identity | Blur | Brightness | Occlusion | Shift | Mean perturbed |
|---|---:|---:|---:|---:|---:|---:|
| BGR-boundary | 0.9333 | 0.9333 | 0.9333 | 0.5333 | 0.9333 | 0.8333 |
| Official OpenVLA-OFT | 0.9333 | 0.9333 | 0.9333 | 0.4667 | 0.9333 | 0.8167 |
| Random-balanced | 0.9333 | 0.9333 | 0.9333 | 0.4000 | 0.9333 | 0.8000 |

Interpretation: p1024 is a useful intermediate OpenVLA-OFT diagnostic. Clean
success ties matched random at 14/15. Under visual perturbations, BGR has a
one-episode occlusion edge over both matched random and the unadapted official
checkpoint, giving mean perturbed success 0.8333 vs. 0.8000 for random and
0.8167 for official. This is still a small 15-episode diagnostic and should be
read as recovery-margin/data-plumbing evidence rather than as a final
robotics fine-tuning claim.

## Completed OpenVLA-OFT Perturbation-Mix Diagnostic

### `openvla_oft_goal_adapt_eval_cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1`

Launched on 2026-06-02 as a completed OpenVLA-OFT scale-control diagnostic after
the p256 official-train-stats run was negative. This run keeps the same
identity-LoRA finetune wrapper, official LIBERO-Goal train/eval statistics,
100-step low-LR adaptation, clean LIBERO-Goal eval, and matched visual
perturbation eval protocol. The only intended experimental change is the
clean-mix dataset: the perturbation subset is doubled from 256 to 512 rendered
examples by allowing two perturbation episodes per family.

Queued prep command:

```bash
TAG=p512 \
MAX_PERTURB_EXAMPLES=512 \
PERTURB_EPISODES_PER_FAMILY=2 \
scripts/queue_openvla_oft_clean_mix_prep.sh --submit
```

The original prep finished successfully but was close enough to walltime that
the pending dependent chain was replaced with a continuation-safe graph. Jobs
`763615`--`763630` were canceled before starting. The TFDS continuation job
waited on `afterany:763614`, validated the rendered p512 TFDS roots, found them
complete, and exited without rebuilding.

Queued TFDS continuation command:

```bash
TAG=p512 \
PREP_DEPENDENCY=afterany:763614 \
scripts/queue_openvla_oft_clean_mix_tfds_export.sh --submit
```

Replacement clean eval command:

```bash
TRAIN_DEPENDENCY=afterok:763636 \
TRAIN_DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b/dataset_statistics.json \
DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b/dataset_statistics.json \
FINETUNE_SCRIPT=vla-scripts/finetune_identity_lora.py \
ADAPT_STEPS=100 LR=1e-6 \
TAG=cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1 \
EVAL_ARTIFACT=openvla_oft_goal_adapt_eval_cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1 \
BGR_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p512_v1 \
RANDOM_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_cleanmix_p512_v1 \
BGR_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1 \
RANDOM_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1 \
scripts/queue_openvla_oft_goal_adapt.sh --submit
```

Replacement visual-perturbation eval command:

```bash
METHODS=bgr,random \
BGR_DEPENDENCY=afterok:763638 \
RANDOM_DEPENDENCY=afterok:763641 \
TAG=cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1 \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1 \
BGR_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
RANDOM_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Queued official visual-perturbation comparator:

```bash
METHODS=official \
TAG=cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1 \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1 \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Slurm chain:

```text
763614  bgr-cleanmix-prep-p512                                                           completed
763615--763630 original BGR/random train/eval/perturb chain                              canceled before start
763636  bgr-cleanmix-tfds-p512                                                           completed, TFDS roots already valid
763637  bgr-goal-adapt-bgr-cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1    completed
763638  bgr-goal-merge-bgr-cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1    completed
763639  bgr-goal-eval-bgr-cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1     completed
763640  bgr-goal-adapt-random-cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1 completed
763641  bgr-goal-merge-random-cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1 completed
763642  bgr-goal-eval-random-cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1  completed

763643  bgr perturb identity      completed
763644  bgr perturb blur          completed
763645  bgr perturb brightness    completed
763646  bgr perturb occlusion     completed
763647  bgr perturb shift         completed
763648  random perturb identity   completed
763649  random perturb blur       completed
763650  random perturb brightness completed
763651  random perturb occlusion  completed
763652  random perturb shift      completed
763631  official perturb identity completed
763632  official perturb blur     completed
763633  official perturb brightness completed
763634  official perturb occlusion completed
763635  official perturb shift    completed
```

Local copied interim artifacts:

```text
results/openvla_oft_cleanmix_p512_v1/render/bgr_cleanmix_summary.json
results/openvla_oft_cleanmix_p512_v1/render/random_cleanmix_summary.json
results/openvla_oft_cleanmix_p512_v1/tfds/bgr_export_summary.json
results/openvla_oft_cleanmix_p512_v1/tfds/random_export_summary.json
results/openvla_oft_cleanmix_p512_v1/slurm/bgr-cleanmix-prep-p512-763614.out
results/openvla_oft_cleanmix_p512_v1/slurm/bgr-cleanmix-tfds-p512-763636.out
results/openvla_oft_goal_adapt_eval_cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv
results/openvla_oft_perturb_eval_cleanmix_p512_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv
```

Prepared p512 dataset sizes:

| Method | Rendered clean | Rendered perturb | TFDS episodes | TFDS steps |
|---|---:|---:|---:|---:|
| BGR-boundary | 1,152 | 512 | 26 | 1,664 |
| Random-balanced | 1,280 | 512 | 28 | 1,792 |

Clean LIBERO-Goal results:

| Method | Episodes | Successes | Success Rate | Task Rates |
|---|---:|---:|---:|---|
| BGR-boundary, p512 | 15 | 14 | 0.9333 | `[1.0, 0.6666666666666666, 1.0, 1.0, 1.0]` |
| Random-balanced, p512 | 15 | 14 | 0.9333 | `[1.0, 0.6666666666666666, 1.0, 1.0, 1.0]` |

Visual-perturbation results:

| Method | Identity | Blur | Brightness | Occlusion | Shift | Mean perturbed |
|---|---:|---:|---:|---:|---:|---:|
| BGR-boundary | 0.9333 | 0.9333 | 0.9333 | 0.4667 | 0.9333 | 0.8167 |
| Official OpenVLA-OFT | 0.9333 | 0.9333 | 0.9333 | 0.4667 | 0.9333 | 0.8167 |
| Random-balanced | 0.9333 | 0.9333 | 0.9333 | 0.4000 | 0.9333 | 0.8000 |

Interpretation: p512 improves the OpenVLA diagnostic relative to the negative
p256 final-control run. Clean success ties matched random at 14/15, and BGR
has a one-episode edge over random under occlusion, giving mean perturbed
success 0.8167 vs. 0.8000. However, BGR only matches the unadapted official
checkpoint's mean perturbed success and does not exceed it. The result is
therefore a learned-policy bridge/audit with a modest matched-random
robustness edge, not as final robotics fine-tuning evidence.

## Superseded OpenVLA-OFT Training-Stats Diagnostic

### `openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2`

Queued and completed on 2026-06-02 as an OpenVLA-OFT normalization-control
diagnostic: whether corrected clean-mix adaptation changes when the official
LIBERO-Goal action/proprio statistics are forced during RLDS training
normalization, not only restored after merge for evaluation. The launcher
uses `TRAIN_DATASET_STATISTICS_SOURCE` to create a temporary wrapper around the
OpenVLA-OFT finetune script. The wrapper extracts the per-dataset inner stats
object expected by `make_dataset_from_rlds` and monkey-patches that loader
function, leaving the external OpenVLA-OFT checkout unchanged.

Two earlier wrapper attempts were canceled as plumbing failures, not model
results: job `763424` passed the outer keyed stats JSON directly and failed with
`KeyError: 'action'`; job `763441` injected `dataset_statistics` into
`dataset_kwargs_list` and failed with a duplicate-keyword error on the second
loader pass. The repaired `_v2` train jobs reached the training loop, forced
the inner stats path, saved step-50100 checkpoints, merged, and completed clean
evals plus matched BGR/random/official visual-perturbation evals successfully.

Queued clean eval command:

```bash
TRAIN_DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b/dataset_statistics.json \
DATASET_STATISTICS_SOURCE=/work/anonymous/cache_home/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b/dataset_statistics.json \
FINETUNE_SCRIPT=vla-scripts/finetune_identity_lora.py \
ADAPT_STEPS=100 LR=1e-6 \
TAG=cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2 \
EVAL_ARTIFACT=openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2 \
BGR_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p256_v1 \
RANDOM_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_cleanmix_p256_v1 \
BGR_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2 \
RANDOM_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2 \
scripts/queue_openvla_oft_goal_adapt.sh --submit
```

Queued visual-perturbation eval command:

```bash
METHODS=bgr,random \
BGR_DEPENDENCY=afterok:763458 \
RANDOM_DEPENDENCY=afterok:763461 \
TAG=cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2 \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2 \
BGR_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2/openvla-7b-oft-finetuned-libero-goal \
RANDOM_CKPT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2/openvla-7b-oft-finetuned-libero-goal \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Queued official visual-perturbation comparator:

```bash
METHODS=official \
TAG=cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2 \
EVAL_ARTIFACT=openvla_oft_perturb_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2 \
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Slurm chain:

```text
763457  bgr-goal-adapt-bgr-cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2    completed
763458  bgr-goal-merge-bgr-cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2    completed
763459  bgr-goal-eval-bgr-cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2     completed
763460  bgr-goal-adapt-random-cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2 completed
763461  bgr-goal-merge-random-cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2 completed
763463  bgr-goal-eval-random-cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2  completed

763464  bgr perturb identity      completed
763465  bgr perturb blur          completed
763466  bgr perturb brightness    completed
763467  bgr perturb occlusion     completed
763468  bgr perturb shift         completed
763469  random perturb identity   completed
763470  random perturb blur       completed
763471  random perturb brightness completed
763472  random perturb occlusion  completed
763473  random perturb shift      completed
763603  official perturb identity completed
763604  official perturb blur     completed
763605  official perturb brightness completed
763606  official perturb occlusion completed
763607  official perturb shift    completed
```

Local copied summaries:

```text
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2/summary.csv
results/openvla_oft_perturb_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialtrainstats_v2/summary.csv
```

Clean LIBERO-Goal results:

| Method | Episodes | Successes | Success Rate | Task Rates |
|---|---:|---:|---:|---|
| BGR-boundary, official train stats | 15 | 14 | 0.9333 | `[1.0, 0.6666666666666666, 1.0, 1.0, 1.0]` |
| Random-balanced, official train stats | 15 | 15 | 1.0000 | `[1.0, 1.0, 1.0, 1.0, 1.0]` |

Visual-perturbation results:

| Method | Identity | Blur | Brightness | Occlusion | Shift | Mean perturbed |
|---|---:|---:|---:|---:|---:|---:|
| BGR-boundary | 0.9333 | 0.9333 | 0.9333 | 0.4667 | 0.8667 | 0.8000 |
| Official OpenVLA-OFT | 0.9333 | 0.9333 | 0.9333 | 0.4667 | 0.9333 | 0.8167 |
| Random-balanced | 1.0000 | 0.9333 | 0.9333 | 0.4667 | 0.9333 | 0.8167 |

Interpretation: forcing official statistics during training closes the last
obvious normalization-control question, but it does not produce a BGR advantage.
BGR ties random and the official checkpoint on blur, brightness, and occlusion,
but is one episode lower on shift and one episode lower than random on clean
identity. The completed BGR/random/official comparison reinforces the main-paper
decision to treat OpenVLA as a recovery-curve and data-plumbing audit rather
than as robotics fine-tuning evidence.

## Superseded OpenVLA-OFT Clean-Mix Adaptation

### `openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_v1`

Submitted on 2026-06-02 as a clean-mix adaptation diagnostic after the
perturbation-only official-checkpoint adaptation degraded clean competence. The
prep job completed successfully: it exported successful native OpenVLA replay
episodes as `identity` clean anchors, rendered one perturbation episode per
family for BGR and random, combined clean and perturbed rendered examples, and
exported OpenVLA-OFT TFDS datasets. The dependent train, merge, and eval jobs
all completed successfully.

Submitted commands:

```bash
scripts/queue_openvla_oft_clean_mix_prep.sh --submit

TRAIN_DEPENDENCY=afterok:763352 \
TAG=cleanmix_p256_step50100 \
EVAL_ARTIFACT=openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_v1 \
BGR_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p256_v1 \
RANDOM_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_cleanmix_p256_v1 \
BGR_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p256_step50100_v1 \
RANDOM_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p256_step50100_v1 \
scripts/queue_openvla_oft_goal_adapt.sh --submit
```

Slurm chain:

```text
763352  bgr-cleanmix-prep-p256                   render/export clean-mix TFDS
763353  bgr-goal-adapt-bgr-cleanmix_p256_step50100     afterok:763352
763354  bgr-goal-merge-bgr-cleanmix_p256_step50100     afterok:763353
763355  bgr-goal-eval-bgr-cleanmix_p256_step50100      afterok:763354
763356  bgr-goal-adapt-random-cleanmix_p256_step50100  afterok:763352
763357  bgr-goal-merge-random-cleanmix_p256_step50100  afterok:763356
763358  bgr-goal-eval-random-cleanmix_p256_step50100   afterok:763357
```

Local copied prep summaries and Slurm log:

```text
results/openvla_oft_cleanmix_p256_v1/manifests/clean_anchor_summary.json
results/openvla_oft_cleanmix_p256_v1/render/bgr_cleanmix_summary.json
results/openvla_oft_cleanmix_p256_v1/render/random_cleanmix_summary.json
results/openvla_oft_cleanmix_p256_v1/tfds/bgr_export_summary.json
results/openvla_oft_cleanmix_p256_v1/tfds/random_export_summary.json
results/openvla_oft_cleanmix_p256_v1/slurm/bgr-cleanmix-prep-p256-763352.out
```

Clean-anchor manifest:

| Method | Clean Steps | Perturbation |
|---|---:|---|
| BGR-boundary | 1,152 | `identity` |
| Random-balanced | 1,280 | `identity` |

Clean-mix TFDS datasets:

| Method | Episodes | Steps | Mix |
|---|---:|---:|---|
| BGR-boundary | 22 | 1,408 | 1,152 clean + 256 perturb |
| Random-balanced | 24 | 1,536 | 1,280 clean + 256 perturb |

Local copied eval logs and parsed summary:

```text
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_v1/logs/bgr/EVAL-libero_goal-openvla-2026_06_02-18_07_59--bgr-cleanmix_p256_step50100.txt
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_v1/logs/random/EVAL-libero_goal-openvla-2026_06_02-18_07_59--random-cleanmix_p256_step50100.txt
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_v1/slurm/bgr-goal-adapt-bgr-cleanmix_p256_step50100-763353.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_v1/slurm/bgr-goal-merge-bgr-cleanmix_p256_step50100-763354.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_v1/slurm/bgr-goal-eval-bgr-cleanmix_p256_step50100-763355.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_v1/slurm/bgr-goal-adapt-random-cleanmix_p256_step50100-763356.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_v1/slurm/bgr-goal-merge-random-cleanmix_p256_step50100-763357.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_v1/slurm/bgr-goal-eval-random-cleanmix_p256_step50100-763358.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_v1/summary.csv
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_v1/summary.json
```

Parsed closed-loop LIBERO-Goal results:

| Method | Episodes | Successes | Success Rate | Task Rates |
|---|---:|---:|---:|---|
| BGR-boundary clean-mix official-adapt | 15 | 0 | 0.0000 | `[0.0, 0.0, 0.0, 0.0, 0.0]` |
| Random-balanced clean-mix official-adapt | 15 | 0 | 0.0000 | `[0.0, 0.0, 0.0, 0.0, 0.0]` |

Interpretation, superseded by the zero-change diagnostics below: this run used
OpenVLA-OFT's default Gaussian LoRA initialization and wrote clean-mix dataset
statistics into the merged checkpoint. Later no-change controls show those two
checkpoint-plumbing choices are sufficient to collapse LIBERO-Goal success even
when the optimizer learning rate is zero. Treat this 0/15 result as a flawed
adaptation setup, not evidence that clean anchors are harmful.

### `openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr1em6_v1`

Submitted on 2026-06-02 after the 100-step clean-mix run collapsed. This diagnostic
keeps the same clean-mix TFDS roots but reduces update strength to 10 optimizer
steps at learning rate `1e-6`.

Submitted command:

```bash
ADAPT_STEPS=10 \
LR=1e-6 \
SAVE_FREQ=10 \
TRAIN_TIME=01:00:00 \
TAG=cleanmix_p256_step50010_lr1em6 \
EVAL_ARTIFACT=openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr1em6_v1 \
BGR_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p256_v1 \
RANDOM_DATA_ROOT=/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_cleanmix_p256_v1 \
BGR_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p256_step50010_lr1em6_v1 \
RANDOM_RUN_ROOT=/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p256_step50010_lr1em6_v1 \
scripts/queue_openvla_oft_goal_adapt.sh --submit
```

Slurm chain:

```text
763359  bgr-goal-adapt-bgr-cleanmix_p256_step50010_lr1em6     failed before data loading
763360  bgr-goal-merge-bgr-cleanmix_p256_step50010_lr1em6     canceled
763361  bgr-goal-eval-bgr-cleanmix_p256_step50010_lr1em6      canceled
763362  bgr-goal-adapt-random-cleanmix_p256_step50010_lr1em6  completed
763363  bgr-goal-merge-random-cleanmix_p256_step50010_lr1em6  completed
763364  bgr-goal-eval-random-cleanmix_p256_step50010_lr1em6   completed
763365  bgr-goal-adapt-bgr-cleanmix_p256_step50010_lr1em6     completed rerun
763366  bgr-goal-merge-bgr-cleanmix_p256_step50010_lr1em6     completed rerun
763367  bgr-goal-eval-bgr-cleanmix_p256_step50010_lr1em6      completed rerun
```

The first BGR train job failed because concurrent train jobs both called
OpenVLA-OFT's `update_auto_map` on the shared Hugging Face checkpoint
`config.json`; the random train job succeeded. The launcher uses
`SERIAL_TRAIN=1` and supports `METHODS=bgr` or `METHODS=random` for
single-branch reruns. The BGR branch was rerun after random finished.

Local copied logs and parsed summary:

```text
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr1em6_v1/logs/bgr/EVAL-libero_goal-openvla-2026_06_02-18_23_58--bgr-cleanmix_p256_step50010_lr1em6.txt
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr1em6_v1/logs/random/EVAL-libero_goal-openvla-2026_06_02-18_21_00--random-cleanmix_p256_step50010_lr1em6.txt
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr1em6_v1/slurm/bgr-goal-adapt-bgr-cleanmix_p256_step50010_lr1em6-763359.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr1em6_v1/slurm/bgr-goal-adapt-bgr-cleanmix_p256_step50010_lr1em6-763365.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr1em6_v1/slurm/bgr-goal-adapt-random-cleanmix_p256_step50010_lr1em6-763362.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr1em6_v1/slurm/bgr-goal-merge-bgr-cleanmix_p256_step50010_lr1em6-763366.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr1em6_v1/slurm/bgr-goal-merge-random-cleanmix_p256_step50010_lr1em6-763363.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr1em6_v1/slurm/bgr-goal-eval-bgr-cleanmix_p256_step50010_lr1em6-763367.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr1em6_v1/slurm/bgr-goal-eval-random-cleanmix_p256_step50010_lr1em6-763364.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr1em6_v1/summary.csv
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr1em6_v1/summary.json
```

Parsed closed-loop LIBERO-Goal results:

| Method | Episodes | Successes | Success Rate | Task Rates |
|---|---:|---:|---:|---|
| BGR-boundary clean-mix weak official-adapt | 15 | 0 | 0.0000 | `[0.0, 0.0, 0.0, 0.0, 0.0]` |
| Random-balanced clean-mix weak official-adapt | 15 | 0 | 0.0000 | `[0.0, 0.0, 0.0, 0.0, 0.0]` |

Interpretation, superseded by the zero-change diagnostics below: reducing the
update to 10 steps at `1e-6` was not enough because the run still used Gaussian
LoRA initialization and clean-mix action statistics in the eval checkpoint.

### OpenVLA-OFT zero-change checkpoint diagnostics

Submitted on 2026-06-02 to isolate why clean-mix adaptation collapsed despite the
official checkpoint scoring 14/15. Each row starts from the official
LIBERO-Goal checkpoint, trains for 10 batches with learning rate `0`, merges,
and evaluates the BGR branch only.

| Artifact | LoRA Init | Eval Action Stats | Episodes | Successes | Success Rate | Task Rates |
|---|---|---|---:|---:|---:|---|
| `openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr0_v1` | Gaussian default | clean-mix | 15 | 0 | 0.0000 | `[0.0, 0.0, 0.0, 0.0, 0.0]` |
| `openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr0_identitylora_v1` | identity/default PEFT | clean-mix | 15 | 0 | 0.0000 | `[0.0, 0.0, 0.0, 0.0, 0.0]` |
| `openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr0_identitylora_officialstats_v1` | identity/default PEFT | official LIBERO-Goal | 15 | 14 | 0.9333 | `[1.0, 0.6666666666666666, 1.0, 1.0, 1.0]` |

Slurm chains:

```text
763368  bgr-goal-adapt-bgr-cleanmix_p256_step50010_lr0
763369  bgr-goal-merge-bgr-cleanmix_p256_step50010_lr0
763370  bgr-goal-eval-bgr-cleanmix_p256_step50010_lr0
763372  bgr-goal-adapt-bgr-cleanmix_p256_step50010_lr0_identitylora
763373  bgr-goal-merge-bgr-cleanmix_p256_step50010_lr0_identitylora
763374  bgr-goal-eval-bgr-cleanmix_p256_step50010_lr0_identitylora
763375  bgr-goal-adapt-bgr-cleanmix_p256_step50010_lr0_identitylora_officialstats
763376  bgr-goal-merge-bgr-cleanmix_p256_step50010_lr0_identitylora_officialstats
763377  bgr-goal-eval-bgr-cleanmix_p256_step50010_lr0_identitylora_officialstats
```

Local summaries:

```text
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr0_v1/summary.json
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr0_identitylora_v1/summary.json
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr0_identitylora_officialstats_v1/summary.json
```

Interpretation: the collapse is reproducible without any parameter update when
the default OpenVLA-OFT script merges a Gaussian-initialized LoRA adapter, and it
also persists with identity LoRA if the eval checkpoint uses clean-mix action
normalization statistics. Identity LoRA plus the official LIBERO-Goal action
statistics restores the exact official-checkpoint result, 14/15. Corrected BGR
experiments use `FINETUNE_SCRIPT=vla-scripts/finetune_identity_lora.py` and
restore `dataset_statistics.json` from the official checkpoint after merge.

### `openvla_oft_goal_adapt_eval_cleanmix_p256_step50010_lr1em6_identitylora_officialstats_v1`

Submitted on 2026-06-02 as the first corrected matched adaptation: 10 optimizer
steps at learning rate `1e-6`, identity/default PEFT LoRA initialization, and
official LIBERO-Goal action statistics restored after merge.

Slurm chain:

```text
763378  bgr-goal-adapt-bgr-cleanmix_p256_step50010_lr1em6_identitylora_officialstats
763379  bgr-goal-merge-bgr-cleanmix_p256_step50010_lr1em6_identitylora_officialstats
763380  bgr-goal-eval-bgr-cleanmix_p256_step50010_lr1em6_identitylora_officialstats
763381  bgr-goal-adapt-random-cleanmix_p256_step50010_lr1em6_identitylora_officialstats
763382  bgr-goal-merge-random-cleanmix_p256_step50010_lr1em6_identitylora_officialstats
763383  bgr-goal-eval-random-cleanmix_p256_step50010_lr1em6_identitylora_officialstats
```

Parsed closed-loop LIBERO-Goal results:

| Method | Episodes | Successes | Success Rate | Task Rates |
|---|---:|---:|---:|---|
| BGR-boundary corrected clean-mix adapt | 15 | 14 | 0.9333 | `[1.0, 0.6666666666666666, 1.0, 1.0, 1.0]` |
| Random-balanced corrected clean-mix adapt | 15 | 14 | 0.9333 | `[1.0, 0.6666666666666666, 1.0, 1.0, 1.0]` |

Interpretation: the corrected adaptation path preserves official checkpoint
competence but does not show a BGR advantage.

### `openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1`

Submitted on 2026-06-02 as a stronger corrected matched adaptation: 100 optimizer
steps at learning rate `1e-6`, identity/default PEFT LoRA initialization, and
official LIBERO-Goal action statistics restored after merge.

```text
763384  bgr-goal-adapt-bgr-cleanmix_p256_step50100_lr1em6_identitylora_officialstats
763385  bgr-goal-merge-bgr-cleanmix_p256_step50100_lr1em6_identitylora_officialstats
763386  bgr-goal-eval-bgr-cleanmix_p256_step50100_lr1em6_identitylora_officialstats
763387  bgr-goal-adapt-random-cleanmix_p256_step50100_lr1em6_identitylora_officialstats
763388  bgr-goal-merge-random-cleanmix_p256_step50100_lr1em6_identitylora_officialstats
763389  bgr-goal-eval-random-cleanmix_p256_step50100_lr1em6_identitylora_officialstats
```

Local copied logs and parsed summary:

```text
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/logs/bgr/EVAL-libero_goal-openvla-2026_06_02-19_05_47--bgr-cleanmix_p256_step50100_lr1em6_identitylora_officialstats.txt
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/logs/random/EVAL-libero_goal-openvla-2026_06_02-19_08_01--random-cleanmix_p256_step50100_lr1em6_identitylora_officialstats.txt
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/slurm/bgr-goal-adapt-bgr-cleanmix_p256_step50100_lr1em6_identitylora_officialstats-763384.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/slurm/bgr-goal-merge-bgr-cleanmix_p256_step50100_lr1em6_identitylora_officialstats-763385.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/slurm/bgr-goal-eval-bgr-cleanmix_p256_step50100_lr1em6_identitylora_officialstats-763386.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/slurm/bgr-goal-adapt-random-cleanmix_p256_step50100_lr1em6_identitylora_officialstats-763387.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/slurm/bgr-goal-merge-random-cleanmix_p256_step50100_lr1em6_identitylora_officialstats-763388.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/slurm/bgr-goal-eval-random-cleanmix_p256_step50100_lr1em6_identitylora_officialstats-763389.out
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/summary.csv
results/openvla_oft_goal_adapt_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/summary.json
```

Parsed closed-loop LIBERO-Goal results:

| Method | Episodes | Successes | Success Rate | Task Rates |
|---|---:|---:|---:|---|
| BGR-boundary corrected clean-mix adapt, 100 steps | 15 | 13 | 0.8667 | `[1.0, 0.6666666666666666, 1.0, 0.6666666666666666, 1.0]` |
| Random-balanced corrected clean-mix adapt, 100 steps | 15 | 14 | 0.9333 | `[1.0, 0.6666666666666666, 1.0, 1.0, 1.0]` |

Interpretation: 100 corrected steps at `1e-6` remain stable relative to the
flawed 0/15 runs, but BGR does not beat random on clean LIBERO-Goal success and
is one episode worse. Clean closed-loop success is probably the wrong metric for
the BGR robustness claim; the visual-perturbation and official-statistics
controls below keep this result as a troubleshooting record rather than a
paper-facing claim.

### `openvla_oft_perturb_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1`

Submitted on 2026-06-02 to evaluate the corrected 100-step BGR and random
checkpoints, plus the official checkpoint, under matched visual perturbations.
The patched eval modifies only the primary camera image before policy
preprocessing; wrist images, simulator state, task instructions, and init states
remain unchanged. A single official identity smoke job (`763406`) first matched
the known baseline at 14/15, then the full 15-job sweep completed successfully.

Submitted command:

```bash
scripts/queue_openvla_oft_perturb_eval.sh --submit
```

Slurm jobs:

```text
763407  official identity
763408  official blur
763409  official brightness
763410  official occlusion
763411  official shift
763412  bgr identity
763413  bgr blur
763414  bgr brightness
763415  bgr occlusion
763416  bgr shift
763417  random identity
763418  random blur
763419  random brightness
763420  random occlusion
763421  random shift
```

Local copied logs and parsed summary:

```text
results/openvla_oft_perturb_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/logs/
results/openvla_oft_perturb_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/slurm/
results/openvla_oft_perturb_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/summary.csv
results/openvla_oft_perturb_eval_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/summary.json
```

Parsed closed-loop LIBERO-Goal results:

| Method | Perturbation | Episodes | Successes | Success Rate |
|---|---|---:|---:|---:|
| BGR-boundary corrected adapt | identity | 15 | 13 | 0.8667 |
| BGR-boundary corrected adapt | blur | 15 | 14 | 0.9333 |
| BGR-boundary corrected adapt | brightness | 15 | 14 | 0.9333 |
| BGR-boundary corrected adapt | occlusion | 15 | 7 | 0.4667 |
| BGR-boundary corrected adapt | shift | 15 | 14 | 0.9333 |
| Official OpenVLA-OFT | identity | 15 | 14 | 0.9333 |
| Official OpenVLA-OFT | blur | 15 | 14 | 0.9333 |
| Official OpenVLA-OFT | brightness | 15 | 14 | 0.9333 |
| Official OpenVLA-OFT | occlusion | 15 | 7 | 0.4667 |
| Official OpenVLA-OFT | shift | 15 | 14 | 0.9333 |
| Random-balanced corrected adapt | identity | 15 | 14 | 0.9333 |
| Random-balanced corrected adapt | blur | 15 | 14 | 0.9333 |
| Random-balanced corrected adapt | brightness | 15 | 14 | 0.9333 |
| Random-balanced corrected adapt | occlusion | 15 | 7 | 0.4667 |
| Random-balanced corrected adapt | shift | 15 | 14 | 0.9333 |

Aggregate: BGR identity success is 0.8667, official and random identity are
0.9333, and all three methods have identical mean perturbed success 0.8167
over blur, brightness, occlusion, and shift. Interpretation: this falsifies a
near-term OpenVLA robustness advantage from the tested BGR-selected clean-mix
adaptation. Occlusion exposes shared brittleness, but BGR does not improve it
over matched random selection or the unadapted official checkpoint.

## Completed OpenVLA-OFT Sanity Check

### `openvla_oft_sanity_eval_sanity_v1`

Submitted on 2026-06-02 with:

```bash
scripts/queue_openvla_sanity_eval.sh --submit
```

Slurm jobs:

```text
763343  bgr-sanity-base-sanity_v1
763344  bgr-sanity-oft-goal-sanity_v1
763345  bgr-sanity-base-sanity_base_retry_v1
```

Evaluation scope matches the 1,000-step BGR/random pilot: LIBERO-Goal, five
tasks, three default init states per task, full suite horizon, and seed 7.
The `base` job evaluates `openvla/openvla-7b` with native single-image
OpenVLA action decoding (`use_l1_regression=False`, `use_proprio=False`,
`num_images_in_input=1`). The `oft-goal` job evaluates the official
`moojink/openvla-7b-oft-finetuned-libero-goal` checkpoint with the L1/proprio
two-image OpenVLA-OFT path.

Logs:

```text
/work/anonymous/bgr/logs/bgr-sanity-base-sanity_v1-763343.out
/work/anonymous/bgr/logs/bgr-sanity-oft-goal-sanity_v1-763344.out
/work/anonymous/bgr/logs/bgr-sanity-base-sanity_base_retry_v1-763345.out
/work/anonymous/bgr/runs/openvla_oft_sanity_eval_sanity_v1/logs/base
/work/anonymous/bgr/runs/openvla_oft_sanity_eval_sanity_v1/logs/oft-goal
```

Startup note: `763343` failed before evaluation because loading the
`openvla/openvla-7b` Hugging Face Hub repo used the original OpenVLA modeling
code, whose vision backbone lacks the OpenVLA-OFT fork's
`set_num_images_in_input` method. The retry `763345` uses the patched local
cached OpenVLA snapshot:

```text
/work/anonymous/cache_home/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f
```

That retry also failed before evaluation because the base OpenVLA checkpoint has
no LIBERO-Goal action unnormalization statistics:

```text
AssertionError: Action un-norm key libero_goal not found in VLA `norm_stats`!
```

Local copied logs and parsed summary:

```text
results/openvla_oft_sanity_eval_sanity_v1/logs/oft-goal/EVAL-libero_goal-openvla-2026_06_02-17_22_54--oft-goal-sanity_v1.txt
results/openvla_oft_sanity_eval_sanity_v1/slurm/bgr-sanity-base-sanity_v1-763343.out
results/openvla_oft_sanity_eval_sanity_v1/slurm/bgr-sanity-base-sanity_base_retry_v1-763345.out
results/openvla_oft_sanity_eval_sanity_v1/slurm/bgr-sanity-oft-goal-sanity_v1-763344.out
results/openvla_oft_sanity_eval_sanity_v1/summary.csv
results/openvla_oft_sanity_eval_sanity_v1/summary.json
```

Parsed closed-loop LIBERO-Goal result:

| Method | Episodes | Successes | Success Rate | Task Rates |
|---|---:|---:|---:|---|
| Official OpenVLA-OFT LIBERO-Goal | 15 | 14 | 0.9333 | `[1.0, 0.6666666666666666, 1.0, 1.0, 1.0]` |

Interpretation: the official OpenVLA-OFT LIBERO-Goal checkpoint succeeds under
the same five-task, three-init-state protocol. The 0/15 BGR/random pilot
therefore reflects the then-tested data/optimization setup, not a broken LIBERO
runtime. Later clean-mix audits adapt from a LIBERO-Goal-competent OpenVLA-OFT
checkpoint and keep the result scoped as a data-plumbing diagnostic.

## Superseded Official-Checkpoint Adaptation

### `openvla_oft_goal_adapt_{bgr,random}_balanced2048_step50100_v1`

Submitted on 2026-06-02 with:

```bash
scripts/queue_openvla_oft_goal_adapt.sh --submit
```

This is a short adaptation smoke from the official
`moojink/openvla-7b-oft-finetuned-libero-goal` checkpoint. Unlike the 1,000-step
pilot from raw `openvla/openvla-7b`, it runs the OpenVLA-OFT fine-tune script in
resume mode (`resume_step=50000`) so the official L1 action head and proprio
projector are loaded before applying 100 additional optimizer steps on the
matched BGR-boundary and random-balanced 2,048-step datasets.

Slurm chain:

```text
763346  bgr-goal-adapt-bgr-step50100       train, after official step 50000
763347  bgr-goal-merge-bgr-step50100       afterok:763346
763348  bgr-goal-eval-bgr-step50100        afterok:763347
763349  bgr-goal-adapt-random-step50100    train, after official step 50000
763350  bgr-goal-merge-random-step50100    afterok:763349
763351  bgr-goal-eval-random-step50100     afterok:763350
```

Checkpoint roots:

```text
/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_balanced2048_step50100_v1/openvla-7b-oft-finetuned-libero-goal
/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_balanced2048_step50100_v1/openvla-7b-oft-finetuned-libero-goal
```

Local copied logs and parsed summary:

```text
results/openvla_oft_goal_adapt_eval_step50100_v1/logs/bgr/EVAL-libero_goal-openvla-2026_06_02-17_39_01--bgr-step50100.txt
results/openvla_oft_goal_adapt_eval_step50100_v1/logs/random/EVAL-libero_goal-openvla-2026_06_02-17_39_20--random-step50100.txt
results/openvla_oft_goal_adapt_eval_step50100_v1/slurm/bgr-goal-adapt-bgr-step50100-763346.out
results/openvla_oft_goal_adapt_eval_step50100_v1/slurm/bgr-goal-adapt-random-step50100-763349.out
results/openvla_oft_goal_adapt_eval_step50100_v1/slurm/bgr-goal-merge-bgr-step50100-763347.out
results/openvla_oft_goal_adapt_eval_step50100_v1/slurm/bgr-goal-merge-random-step50100-763350.out
results/openvla_oft_goal_adapt_eval_step50100_v1/slurm/bgr-goal-eval-bgr-step50100-763348.out
results/openvla_oft_goal_adapt_eval_step50100_v1/slurm/bgr-goal-eval-random-step50100-763351.out
results/openvla_oft_goal_adapt_eval_step50100_v1/summary.csv
results/openvla_oft_goal_adapt_eval_step50100_v1/summary.json
```

Parsed closed-loop LIBERO-Goal results:

| Method | Episodes | Successes | Success Rate | Task Rates |
|---|---:|---:|---:|---|
| BGR-boundary official-adapt | 15 | 2 | 0.1333 | `[0.6666666666666666, 0.0, 0.0, 0.0, 0.0]` |
| Random-balanced official-adapt | 15 | 1 | 0.0667 | `[0.3333333333333333, 0.0, 0.0, 0.0, 0.0]` |

Interpretation: the resume-mode setup is technically correct: both training
logs load the official `proprio_projector--50000_checkpoint.pt` and
`action_head--50000_checkpoint.pt`, save step-50100 checkpoints, merge LoRA
weights, and complete the same LIBERO-Goal eval. However, 100 low-LR adaptation
steps on the tested 2,048-step perturbation datasets severely degrade the
official checkpoint from 14/15 to 2/15 for BGR and 1/15 for random. BGR has a
small edge over random in this destructive regime, but this is not usable
robotics evidence. Subsequent clean-mix and identity-LoRA controls preserve
clean official-policy behavior and supersede this destructive adaptation setup.

## Superseded OpenVLA-OFT Pilot

### `openvla_oft_finetune_{bgr,random}_balanced2048_step1000_v1`

Submitted on 2026-06-02 with:

```bash
MAX_STEPS=1000 SAVE_FREQ=500 scripts/queue_openvla_oft_long_run.sh --submit
```

Initial Slurm jobs:

```text
763317  bgr-oft-bgr_boundary-step1000
763318  bgr-oft-random_balanced-step1000
```

These first attempts ran on `c2-g4-21` and failed immediately because CUDA
reported no usable GPU and Slurm logged `Failed to get device handle for GPU 0`.
Logs:

```text
/work/anonymous/bgr/logs/bgr-oft-bgr_boundary-step1000-763317.out
/work/anonymous/bgr/logs/bgr-oft-random_balanced-step1000-763318.out
```

The retry excluded `c2-g4-21`. The first retry jobs:

```text
763319  bgr-oft-bgr_boundary-step1000
763320  bgr-oft-random_balanced-step1000
```

These failed in preflight because the remote workspace's branch configuration
made `git pull --ff-only` report `fatal: Cannot fast-forward to multiple
branches`. The successful retry used the copied batch script and existing
validated remote datasets instead of a remote git pull.

Successful retry Slurm jobs:

```text
763329  bgr-oft-bgr_boundary-step1000
763330  bgr-oft-random_balanced-step1000
```

Final retry state: both completed under `low-prio-gpu` despite the nominal
8-hour time limit. `763329` ran on `c2-g4-24` for 11:55 and `763330` ran on
`c1-g4-01` for 10:59. Run roots:

```text
/work/anonymous/bgr/runs/openvla_oft_finetune_bgr_balanced2048_step1000_v1
/work/anonymous/bgr/runs/openvla_oft_finetune_random_balanced2048_step1000_v1
```

Slurm logs:

```text
/work/anonymous/bgr/logs/bgr-oft-bgr_boundary-step1000-763329.out
/work/anonymous/bgr/logs/bgr-oft-random_balanced-step1000-763330.out
```

Both logs passed preflight, reached OpenVLA/LIBERO startup, saved the first
checkpoint, and then reached `Max step 1000 reached! Stopping training...`.
Final checkpoint roots:

```text
/work/anonymous/bgr/runs/openvla_oft_finetune_bgr_balanced2048_step1000_v1/openvla-7b+libero_goal_no_noops+b1+lr-0.0005+lora-r8+dropout-0.0--bgr-balanced2048-step1000
/work/anonymous/bgr/runs/openvla_oft_finetune_random_balanced2048_step1000_v1/openvla-7b+libero_goal_no_noops+b1+lr-0.0005+lora-r8+dropout-0.0--random-balanced2048-step1000
```

Each checkpoint root contains `lora_adapter/adapter_model.safetensors`,
`action_head--latest_checkpoint.pt`, and `proprio_projector--latest_checkpoint.pt`.

### `openvla_oft_eval_balanced2048_step1000_v1`

Submitted on 2026-06-02 with:

```bash
BGR_TRAIN_JOB_ID=763329 RANDOM_TRAIN_JOB_ID=763330 scripts/queue_openvla_oft_eval.sh --submit
```

Dependent Slurm jobs:

```text
763331  bgr-merge-bgr-step1000       afterok:763329
763332  bgr-merge-random-step1000    afterok:763330
763333  bgr-eval-bgr-step1000        afterok:763331
763334  bgr-eval-random-step1000     afterok:763332
```

Because the training jobs initially looked unlikely to finish all 1,000 steps
before walltime, an additional afterany salvage chain was submitted with
`TRAIN_DEPENDENCY_TYPE=afterany` after both latest checkpoints were present:

```bash
BGR_TRAIN_JOB_ID=763329 RANDOM_TRAIN_JOB_ID=763330 TRAIN_DEPENDENCY_TYPE=afterany scripts/queue_openvla_oft_eval.sh --submit
```

Salvage Slurm jobs:

```text
763335  bgr-merge-bgr-step1000       afterany:763329
763336  bgr-merge-random-step1000    afterany:763330
763337  bgr-eval-bgr-step1000        afterok:763335
763338  bgr-eval-random-step1000     afterok:763336
```

Evaluation scope is LIBERO-Goal, five tasks, three official init states per
task, full suite horizon (`max_steps_override=-1`), seed 7, and matched
OpenVLA-OFT checkpoints from the 1,000-step BGR-boundary and random-balanced
training runs. Eval log roots:

```text
/work/anonymous/bgr/runs/openvla_oft_eval_balanced2048_step1000_v1/logs/bgr
/work/anonymous/bgr/runs/openvla_oft_eval_balanced2048_step1000_v1/logs/random
```

Final state: both training jobs completed cleanly, so the original `afterok`
chain was used. The salvage merge jobs `763335` and `763336` had just started
after the completed training jobs; they were cancelled along with dependent eval
jobs `763337` and `763338` to avoid duplicate writes to the same checkpoint and
log roots. Original merge jobs `763331` and `763332` completed successfully,
then eval jobs `763333` and `763334` completed successfully.

Local eval logs and parsed summaries:

```text
results/openvla_oft_eval_balanced2048_step1000_v1/logs/bgr/EVAL-libero_goal-openvla-2026_06_02-17_10_15--bgr-step1000.txt
results/openvla_oft_eval_balanced2048_step1000_v1/logs/random/EVAL-libero_goal-openvla-2026_06_02-17_09_58--random-step1000.txt
results/openvla_oft_eval_balanced2048_step1000_v1/summary.csv
results/openvla_oft_eval_balanced2048_step1000_v1/summary.json
```

Parsed closed-loop LIBERO-Goal results:

| Method | Episodes | Successes | Success Rate | Task Rates |
|---|---:|---:|---:|---|
| BGR-boundary | 15 | 0 | 0.0000 | `[0.0, 0.0, 0.0, 0.0, 0.0]` |
| Random-balanced | 15 | 0 | 0.0000 | `[0.0, 0.0, 0.0, 0.0, 0.0]` |

Interpretation: this completes the first matched 1,000-step
OpenVLA-OFT train/merge/eval pipeline over five LIBERO-Goal tasks and three
default init states per task. It is not policy-quality evidence for BGR:
neither BGR-boundary nor random-balanced fine-tuning produced a successful
closed-loop task. The result identifies data scale, task coverage, and
OpenVLA-OFT optimization as the limiting factors rather than missing
infrastructure.

## Completed Runs

### `environment_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/collect_environment.py --out runs/environment_v1/compute_environment.json
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:10:00 /work/anonymous/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/collect_environment.py --out runs/environment_v1/gpu_environment.json
```

Remote logs:

```text
/work/anonymous/bgr/logs/run_1780318399_415321575.out
/work/anonymous/bgr/logs/run_1780318387_340655701.out
```

Included snapshots:

```text
results/environment_v1/compute_environment.json
results/environment_v1/gpu_environment.json
```

Interpretation: these snapshots record Slurm allocation details, hostnames, CPU model, host memory, OS/kernel, Python executable/version, selected package versions, and GPU hardware metadata for the LIBERO/EGL path. `nvidia-smi` is not on the job path, but the GPU node exposes NVIDIA RTX A6000 hardware through PCI/proc metadata.

### `toy_fast_v3`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 01:00:00 /work/anonymous/bgr env PYTHONPATH=src python scripts/run_toy_experiment.py --config configs/toy_bgr.yaml --out runs/toy_fast_v3
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780304197_887365905.out
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

### `toy_15seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_toy_experiment.py \
  --config configs/toy_bgr_15seed.yaml \
  --out results/toy_15seed_v1
```

Mean results over 15 paired seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.8920 | 0.3726 | 0.2864 | 0.3151 |
| Uniform | 0.8860 | 0.3642 | 0.2813 | 0.2987 |
| Failure-only | 0.8188 | 0.2324 | 0.1343 | 0.2306 |
| Fixed radius | 0.8219 | 0.2294 | 0.1304 | 0.2291 |
| PLR-loss proxy | 0.8207 | 0.2288 | 0.1328 | 0.2290 |

Paired exact sign tests support BGR over uniform on final RAUC
(+0.0083, p=0.0001) and RAUC AULC (+0.0163, p=0.0001). BGR also beats fixed,
failure-only, and PLR-loss baselines on final RAUC with p=0.0001. Interpretation:
this replaces the earlier three-seed Tier-0 result in the paper tables and
removes the weak synthetic significance row.

### Completed `toy_30seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_toy_experiment.py \
  --config configs/toy_bgr_30seed.yaml \
  --out results/toy_30seed_v1
```

Completed locally on 2026-06-04 to confirm the controlled synthetic
recovery-margin mechanism at 30 paired seeds under the same protocol as the
15-seed paper table.

Mean results over 30 paired seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.8899 | 0.3716 | 0.2860 | 0.3149 |
| Uniform | 0.8838 | 0.3633 | 0.2816 | 0.2984 |
| Failure-only | 0.8175 | 0.2331 | 0.1347 | 0.2312 |
| Fixed radius | 0.8205 | 0.2298 | 0.1309 | 0.2295 |
| PLR-loss proxy | 0.8190 | 0.2289 | 0.1336 | 0.2292 |

BGR improves final RAUC over uniform (0.3716 vs. 0.3633) with 29/1 paired
wins, while BGR improves RAUC AULC (0.3149 vs. 0.2984) and clean success
(0.8899 vs. 0.8838) with 30/0 paired wins on both metrics. BGR also beats
fixed-radius, failure-only, and PLR-loss baselines on final RAUC with 30/0
paired wins. Interpretation: this strengthens the controlled synthetic
mechanism check while preserving the paper-table claim scope.

### `grid_fast_v4`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 01:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_bgr.yaml --out runs/grid_fast_v4
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780305363_531953871.out
```

Mean results over three seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.9537 | 0.9241 | 0.9778 | 0.7605 |
| Uniform | 1.0000 | 0.9874 | 1.0000 | 0.8745 |
| Fixed radius | 1.0000 | 0.9856 | 1.0000 | 0.8785 |
| Failure-only | 1.0000 | 0.9928 | 1.0000 | 0.8785 |
| PLR-loss proxy | 1.0000 | 0.9936 | 1.0000 | 0.8810 |

Interpretation, superseded by the grid-margin studies below: this procedural
grid setup is too easy after clean-suffix pretraining, and the BGR sampler is
too narrow for saturated states. Treat this historical run as a benchmark-design
diagnostic. The reviewer-facing procedural evidence is the completed 30-seed
grid-margin full-baseline comparison, held-out grid replication, and scoped
sensitivity diagnostics listed in the Submission Evidence Index.

### `grid_margin_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 01:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_bgr.yaml --out runs/grid_margin_v1
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780305895_462068628.out
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
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_bgr_full.yaml --out runs/grid_margin_full_v1
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780305974_747225749.out
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

### `grid_margin_full_15seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_grid_margin_experiment.py \
  --config configs/grid_margin_full_15seed.yaml \
  --out results/grid_margin_full_15seed_v1
```

Mean results over 15 paired seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.9461 | 0.4345 | 0.3441 | 0.3525 |
| Uniform | 0.8943 | 0.3961 | 0.3318 | 0.3129 |
| Failure-only | 0.8453 | 0.2910 | 0.2307 | 0.2537 |
| PLR-loss proxy | 0.7977 | 0.2106 | 0.1367 | 0.2127 |
| Fixed radius | 0.7945 | 0.2094 | 0.1358 | 0.2124 |

Paired exact sign tests from `results/significance_tests_v3` support BGR
over fixed, failure-only, and PLR-loss baselines on final RAUC and RAUC AULC
with p=0.0001 for each comparison. Final RAUC deltas are +0.2251 vs fixed,
+0.1435 vs failure-only, and +0.2239 vs PLR-loss. Interpretation: the stronger
procedural full-baseline result supports the main mechanism claim at the same
15-seed level as the BGR-vs-uniform comparison. Neither hard failure mining,
loss-priority replay, nor a static fixed perturbation radius expands recovery
margins comparably to boundary-centered replay.

### Completed `grid_margin_full_30seed_v1`

Submitted and completed on 2026-06-03 to confirm the procedural grid-margin
full-baseline result at the same 30 paired-seed level used by the suffix
confirmation. This was
a high-value check because the main mechanism claim previously rested on a
15-seed grid full-baseline sweep plus sensitivity runs.

Completed on 2026-06-03 with 30 paired seeds for each configured method
(`uniform`, `fixed`, `failure_only`, `plr_loss`, and `bgr`). BGR confirms the
15-seed full-baseline pattern:

```text
method        clean   RAUC    median r80  AULC
BGR           0.9455  0.4344  0.3447      0.3526
Uniform       0.8939  0.3963  0.3321      0.3132
Failure-only  0.8449  0.2909  0.2314      0.2540
Loss-priority 0.7974  0.2109  0.1373      0.2130
Fixed radius  0.7944  0.2097  0.1362      0.2128
```

Exact paired sign tests from `paper/figures/significance_tests.csv` support all
10 configured 30-seed grid comparisons: BGR beats uniform on final RAUC, AULC,
clean success, and median $r_{80}$, and beats fixed, failure-only, and PLR-loss
on final RAUC and AULC. Every comparison has 30 paired BGR wins, 0 losses, 0
ties, and two-sided sign-test p-value `1.86265e-09`.

Serial command attempted first:

```bash
PYTHONPATH=src:. python3 scripts/run_grid_margin_experiment.py \
  --config configs/grid_margin_full_30seed.yaml \
  --out results/grid_margin_full_30seed_v1
```

Earlier job `763779` failed immediately before producing rows because the
remote grid environment source was stale. Serial job `763780` was then canceled
after reaching `fixed` seed 5 before it wrote a summary, and the run was
replaced with a Slurm array to shorten turnaround and preserve per-trial
artifacts.

Replacement array command:

```bash
PYTHONPATH=src:. python3 scripts/run_grid_margin_trial.py \
  --config configs/grid_margin_full_30seed.yaml \
  --out results/grid_margin_full_30seed_v1 \
  --method METHOD \
  --seed SEED
```

Array job `763781` runs the 150 method/seed trials as `0-149%30`, writing
per-trial JSON files under
`/work/anonymous/bgr/results/grid_margin_full_30seed_v1/trials`. Merge job `763782`
depends on `afterok:763781` and runs:

```bash
PYTHONPATH=src:. python3 scripts/merge_grid_margin_trials.py \
  --config configs/grid_margin_full_30seed.yaml \
  --out results/grid_margin_full_30seed_v1
```

Slurm output is under
`/work/anonymous/bgr/runs/slurm/bgr-grid-full30-array-763781_*.out` and
`/work/anonymous/bgr/runs/slurm/bgr-grid-full30-merge-763782.out`.

### Completed `grid_margin_full_replication_30seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_grid_margin_trial.py \
  --config configs/grid_margin_full_replication_30seed.yaml \
  --out results/grid_margin_full_replication_30seed_v1 \
  --method METHOD \
  --seed SEED
PYTHONPATH=src:. python3 scripts/merge_grid_margin_trials.py \
  --config configs/grid_margin_full_replication_30seed.yaml \
  --out results/grid_margin_full_replication_30seed_v1
```

This local held-out replication reruns the headline BGR-vs-uniform
grid-margin comparison on seeds 30-59, disjoint from the original 30 paired
seeds, while keeping the same grid-margin full-baseline hyperparameters.

Mean results over 30 paired seeds:

| Method | Clean | RAUC | Median r80 | AULC |
|---|---:|---:|---:|---:|
| BGR | 0.9453 | 0.4340 | 0.3446 | 0.3523 |
| Uniform | 0.8934 | 0.3967 | 0.3327 | 0.3137 |

Per-seed paired signs give 30/0 paired wins for BGR over uniform on clean
success, final RAUC (0.4340 vs. 0.3967), median r80, AULC, and best RAUC. Interpretation:
independent held-out seeds replicate the central procedural margin-expansion
claim before the paper moves to suffix and OpenVLA audits.

Stored learning-curve histories in `results.json` also show that BGR leads
uniform at every evaluation checkpoint after the first update. These rows are
exported by `scripts/aggregate_results.py` to
`paper/figures/grid_margin_learning_curve_stats.csv`, and exact sign tests
for every checkpoint are included in `paper/figures/significance_tests.csv`.

| Step | BGR RAUC | Uniform RAUC | Delta |
|---:|---:|---:|---:|
| 30 | 0.2607 | 0.2372 | +0.0235 |
| 60 | 0.2952 | 0.2583 | +0.0369 |
| 90 | 0.3226 | 0.2786 | +0.0440 |
| 120 | 0.3454 | 0.2979 | +0.0474 |
| 150 | 0.3649 | 0.3167 | +0.0482 |
| 180 | 0.3817 | 0.3344 | +0.0473 |
| 210 | 0.3967 | 0.3511 | +0.0456 |
| 240 | 0.4103 | 0.3670 | +0.0433 |
| 270 | 0.4229 | 0.3820 | +0.0408 |
| 300 | 0.4345 | 0.3961 | +0.0384 |

Each stepwise RAUC delta is positive on all 15 paired seeds; exact two-sided
sign tests report p=0.0001 after CSV formatting for every checkpoint.

### `grid_margin_target_sensitivity_15seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_grid_margin_target_sensitivity.py \
  --config configs/grid_margin_target_sensitivity_15seed.yaml \
  --out results/grid_margin_target_sensitivity_15seed_v1
```

Mean BGR results over 15 seeds for each target margin:

| Target margin | Clean | RAUC | Median r80 | RAUC AULC |
|---:|---:|---:|---:|---:|
| 0.26 | 0.9512 | 0.4429 | 0.3514 | 0.3587 |
| 0.32 | 0.9493 | 0.4390 | 0.3488 | 0.3558 |
| 0.38 | 0.9461 | 0.4345 | 0.3441 | 0.3525 |
| 0.46 | 0.9414 | 0.4266 | 0.3363 | 0.3467 |
| 0.54 | 0.9350 | 0.4152 | 0.3295 | 0.3388 |

Against the paired uniform rows from `grid_margin_full_15seed_v1`, every target
margin improves final RAUC and RAUC AULC on all 15 seeds. The robustness rows
are included in `paper/figures/significance_tests.csv`; mean final RAUC deltas
range from +0.0191 at target margin 0.54 to +0.0468 at target margin 0.26, with
exact two-sided sign-test p=0.0001 after CSV formatting for every target/metric
comparison. Interpretation: the reported 0.38 target is not a cherry-picked
optimum; lower targets perform even better in this benchmark, and BGR remains
above uniform across the tested 0.26--0.54 range.

### Completed `grid_margin_target_sensitivity_30seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_grid_margin_target_sensitivity.py \
  --config configs/grid_margin_target_sensitivity_30seed.yaml \
  --out results/grid_margin_target_sensitivity_30seed_v1
```

Completed locally on 2026-06-04 to confirm the grid-margin target-margin sweep
at 30 paired seeds against the completed full-baseline uniform rows.

Mean BGR results over 30 paired seeds:

| Target margin | Clean | RAUC | Median r80 | RAUC AULC |
|---:|---:|---:|---:|---:|
| 0.26 | 0.9507 | 0.4427 | 0.3516 | 0.3587 |
| 0.32 | 0.9488 | 0.4391 | 0.3488 | 0.3560 |
| 0.38 | 0.9455 | 0.4344 | 0.3447 | 0.3526 |
| 0.46 | 0.9405 | 0.4264 | 0.3376 | 0.3468 |
| 0.54 | 0.9344 | 0.4157 | 0.3302 | 0.3393 |

Against the paired uniform rows from `grid_margin_full_30seed_v1`, BGR improves
final RAUC, RAUC AULC, and clean success over the 30-seed uniform baseline with
30/0 paired wins at every target margin. Final RAUC mean deltas range from
+0.0194 at target 0.54 to +0.0463 at target 0.26, and RAUC AULC mean deltas
range from +0.0261 to +0.0455. Interpretation: target 0.38 is not a
cherry-picked optimum; lower targets perform even better while all tested
targets remain above uniform on the paper-facing RAUC/AULC/clean metrics.
Median r80 is not a promoted target-sensitivity claim: high target margins
soften that metric (0.46 gives 27/3 paired wins and 0.54 gives 9/21).

### `grid_margin_learning_rate_sensitivity_15seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_grid_margin_learning_rate_sensitivity.py \
  --config configs/grid_margin_learning_rate_sensitivity_15seed.yaml \
  --out results/grid_margin_learning_rate_sensitivity_15seed_v1
```

Mean BGR-vs-uniform results over 15 paired seeds:

| Learning rate | Method | Clean | RAUC | Median r80 | RAUC AULC |
|---:|---|---:|---:|---:|---:|
| 0.015 | BGR | 0.9457 | 0.3821 | 0.2999 | 0.3124 |
| 0.015 | Uniform | 0.8846 | 0.3198 | 0.2485 | 0.2689 |
| 0.030 | BGR | 0.9461 | 0.4345 | 0.3441 | 0.3525 |
| 0.030 | Uniform | 0.8943 | 0.3961 | 0.3318 | 0.3129 |
| 0.060 | BGR | 0.9442 | 0.4856 | 0.4076 | 0.3962 |
| 0.060 | Uniform | 0.8993 | 0.4908 | 0.4397 | 0.3785 |

Paired exact sign-test rows are included in
`paper/figures/significance_tests.csv` for clean, final RAUC, median r80, and
RAUC AULC at each learning rate. BGR improves final RAUC at low and nominal
rates (+0.0623 and +0.0384, 15/0 wins), while the high learning rate lets
uniform overtake final RAUC (-0.0052, 1/14 wins) and median r80 (-0.0321, 0/15
wins). BGR still improves clean success and RAUC AULC at all three rates, with
15/0 paired wins on those metrics. Interpretation: this is a scope diagnostic,
not a new robustness headline; high learning rate can saturate uniform replay
enough to erase BGR's final-RAUC advantage.

### Completed `grid_margin_learning_rate_sensitivity_30seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_grid_margin_learning_rate_sensitivity.py \
  --config configs/grid_margin_learning_rate_sensitivity_30seed.yaml \
  --out results/grid_margin_learning_rate_sensitivity_30seed_v1
```

Completed locally on 2026-06-04 to confirm the learning-rate scope diagnostic
over 30 paired seeds.

Mean BGR-vs-uniform results over 30 paired seeds:

| Learning rate | Method | Clean | RAUC | Median r80 | RAUC AULC |
|---:|---|---:|---:|---:|---:|
| 0.015 | BGR | 0.9453 | 0.3820 | 0.2997 | 0.3124 |
| 0.015 | Uniform | 0.8843 | 0.3200 | 0.2491 | 0.2693 |
| 0.030 | BGR | 0.9455 | 0.4344 | 0.3447 | 0.3526 |
| 0.030 | Uniform | 0.8939 | 0.3963 | 0.3321 | 0.3132 |
| 0.060 | BGR | 0.9437 | 0.4854 | 0.4077 | 0.3961 |
| 0.060 | Uniform | 0.8989 | 0.4908 | 0.4400 | 0.3787 |

At low and nominal learning rates, BGR improves final RAUC, RAUC AULC, clean
success, and median r80 with 30/0 paired wins. At learning rate 0.060, uniform
remains higher on final RAUC with 29/1 paired wins and median r80 with 30/0
paired wins. BGR still improves RAUC AULC and clean success at 0.060 with 30/0
paired wins. Interpretation: the 30-seed confirmation preserves the
optimization-scope caveat rather than turning the learning-rate sweep into a
new robustness headline.

### Completed `grid_margin_stress_sensitivity_30seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_grid_margin_stress_sensitivity.py \
  --config configs/grid_margin_stress_sensitivity_30seed.yaml \
  --out results/grid_margin_stress_sensitivity_30seed_v1
```

Completed locally on 2026-06-04 to confirm the geometry-stress diagnostic over
30 paired seeds.

Mean BGR-vs-uniform results over 30 paired seeds:

| Stress case | Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---|---:|---:|---:|---:|
| diffuse_boundary | BGR | 0.9282 | 0.4573 | 0.3260 | 0.3734 |
| diffuse_boundary | Uniform | 0.8785 | 0.4299 | 0.3261 | 0.3387 |
| low_feasibility | BGR | 0.9451 | 0.3616 | 0.2852 | 0.3047 |
| low_feasibility | Uniform | 0.8875 | 0.3234 | 0.2616 | 0.2708 |
| sharp_low_margin | BGR | 0.9621 | 0.3980 | 0.3293 | 0.3076 |
| sharp_low_margin | Uniform | 0.8938 | 0.3250 | 0.2919 | 0.2451 |

BGR improves final RAUC, RAUC AULC, and clean success with 30/0 paired wins in
every stress case. Median r80 is not a promoted stress-sensitivity claim:
diffuse-boundary median r80 is not promoted because it has a 14/16 paired split,
while low-feasibility and sharp-low-margin have 30/0 BGR wins. Interpretation:
the 30-seed confirmation strengthens the geometry-stress diagnostic on
paper-facing recovery metrics without overstating median-r80 behavior.

### `grid_margin_regime_sensitivity_15seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_grid_margin_regime_sensitivity.py \
  --config configs/grid_margin_regime_sensitivity_15seed.yaml \
  --out results/grid_margin_regime_sensitivity_15seed_v1
```

Mean BGR-vs-uniform results over 15 paired seeds:

| Regime | Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---|---:|---:|---:|---:|
| low_obstacle | BGR | 0.9461 | 0.4345 | 0.3444 | 0.3526 |
| low_obstacle | Uniform | 0.8943 | 0.3961 | 0.3318 | 0.3129 |
| nominal | BGR | 0.9461 | 0.4345 | 0.3441 | 0.3525 |
| nominal | Uniform | 0.8943 | 0.3961 | 0.3318 | 0.3129 |
| high_obstacle | BGR | 0.9458 | 0.4344 | 0.3445 | 0.3524 |
| high_obstacle | Uniform | 0.8943 | 0.3961 | 0.3318 | 0.3129 |

Interpretation: this is an inconclusive robustness diagnostic, not a main-paper
claim. Changing `obstacle_prob` and `max_offset` over this range barely changes
the procedural margin distribution after feasibility clipping, so the run mostly
reproduces the nominal BGR-vs-uniform result rather than demonstrating a
meaningfully different grid regime.

### Completed `grid_margin_regime_sensitivity_30seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_grid_margin_regime_sensitivity.py \
  --config configs/grid_margin_regime_sensitivity_30seed.yaml \
  --out results/grid_margin_regime_sensitivity_30seed_v1
```

Mean BGR-vs-uniform results over 30 paired seeds:

| Regime | Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---|---:|---:|---:|---:|
| high_obstacle | BGR | 0.9454 | 0.4346 | 0.3446 | 0.3526 |
| high_obstacle | Uniform | 0.8939 | 0.3963 | 0.3320 | 0.3132 |
| low_obstacle | BGR | 0.9456 | 0.4345 | 0.3448 | 0.3526 |
| low_obstacle | Uniform | 0.8939 | 0.3963 | 0.3321 | 0.3132 |
| nominal | BGR | 0.9455 | 0.4344 | 0.3447 | 0.3526 |
| nominal | Uniform | 0.8939 | 0.3963 | 0.3321 | 0.3132 |

BGR improves final RAUC, RAUC AULC, clean success, and median r80 with 30/0
paired wins in every regime. Interpretation: this remains a diagnostic rather
than separate robustness evidence because the sweep mostly reproduces the
nominal margin dynamics.

### `grid_margin_pair_15seed_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_pair_15seed.yaml --out runs/grid_margin_pair_15seed_v1
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780328088_151681007.out
```

Mean results over 15 paired seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.9461 | 0.4345 | 0.3441 | 0.3525 |
| Uniform | 0.8943 | 0.3961 | 0.3318 | 0.3129 |

Interpretation: this strengthens the primary procedural claim. BGR improves
all four reported metrics over uniform on every paired seed; paired exact
sign tests give p=0.00006 for clean success, RAUC, median r80, and RAUC
AULC.

### `suffix_full_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_bgr_full.yaml --out runs/suffix_full_v1
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780307573_760181637.out
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

Interpretation, superseded by the coverage-aware 30-seed suffix results above:
BGR-Suffix strongly beats clean-only, fixed-radius, failure-only, and
loss-priority recovery training. Compared with uniform suffix replay, it
improves clean success and sample efficiency, but uniform retains higher final
object RAUC and transfer RAUC in this lightweight suffix simulator. Treat this
historical run as a suffix-design diagnostic, not as the packaged
manipulation-style evidence.

### `suffix_full_15seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_suffix_experiment.py \
  --config configs/suffix_full_15seed.yaml \
  --out results/suffix_full_15seed_v1
```

Mean results over 15 paired seeds:

| Method | Clean | Object RAUC | Median r80 | EE-transfer RAUC | RAUC AULC |
|---|---:|---:|---:|---:|---:|
| BGR-Suffix | 0.8734 | 0.4723 | 0.4530 | 0.3015 | 0.3739 |
| Uniform suffix | 0.8358 | 0.4844 | 0.5046 | 0.3075 | 0.3707 |
| Clean FT | 0.8645 | 0.2629 | 0.2067 | 0.1886 | 0.2388 |
| Failure-only | 0.7571 | 0.4228 | 0.4742 | 0.2430 | 0.3026 |
| Loss priority | 0.7105 | 0.1695 | 0.1283 | 0.1441 | 0.1690 |
| Fixed radius | 0.6838 | 0.1583 | 0.1244 | 0.1375 | 0.1646 |

Paired exact sign tests support BGR-Suffix over clean-only, fixed-radius,
failure-only, and loss-priority recovery training on object RAUC, transfer RAUC,
and RAUC AULC (p=0.0001 for each object-RAUC comparison). Compared with uniform,
BGR-Suffix has higher clean success (+0.0376, p=0.0001) and higher AULC
(+0.0032, p=0.0003), but lower final object RAUC (-0.0122, p=0.0001) and lower
transfer RAUC (-0.0061, p=0.0001). Interpretation: this upgrades the suffix
full-baseline evidence to 15 seeds while preserving the honest diagnostic
tradeoff that motivated the broad-radius strategy run.

### Completed `suffix_coverage_full_30seed_v1`

Completed on 2026-06-03 to test the coverage-aware BGR-Coverage setting against
the full suffix baseline set, not only uniform suffix replay. This replaces the
mixed narrative where boundary-heavy BGR-Suffix is compared to
clean/fixed/failure/loss baselines and BGR-Coverage is compared only to uniform.

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_suffix_experiment.py \
  --config configs/suffix_coverage_full_30seed.yaml \
  --out results/suffix_coverage_full_30seed_v1
```

Cluster job `763773` completed on `cnode301` with Slurm output at
`/work/anonymous/bgr/runs/slurm/bgr-suffix-covfull30-763773.out`.

Mean results over 30 paired seeds:

| Method | Clean | Object RAUC | Median r80 | EE-transfer RAUC | RAUC AULC |
|---|---:|---:|---:|---:|---:|
| BGR-Coverage | 0.8644 | 0.4969 | 0.4982 | 0.3143 | 0.3825 |
| Uniform suffix | 0.8364 | 0.4854 | 0.5047 | 0.3083 | 0.3709 |
| Clean FT | 0.8646 | 0.2631 | 0.2065 | 0.1896 | 0.2389 |
| Failure-only | 0.7573 | 0.4229 | 0.4734 | 0.2434 | 0.3024 |
| Loss priority | 0.7110 | 0.1694 | 0.1279 | 0.1449 | 0.1689 |
| Fixed radius | 0.6841 | 0.1583 | 0.1239 | 0.1382 | 0.1646 |

Exact paired sign tests in `paper/figures/significance_tests.csv` give 30/0
paired wins for BGR-Coverage over clean-only, fixed-radius, failure-only,
loss-priority, and uniform suffix replay on final object RAUC. BGR-Coverage also
beats uniform on EE-transfer RAUC and RAUC AULC with 30/0 paired wins. Clean-only
has a negligible clean-success edge (0.8646 vs. 0.8644), and uniform remains
higher on median r80 (0.5047 vs. 0.4982), so this is a strong positive suffix
simulator result with the median-critical-radius caveat preserved.

### Completed `suffix_coverage_full_replication_30seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_suffix_experiment.py \
  --config configs/suffix_coverage_full_replication_30seed.yaml \
  --out results/suffix_coverage_full_replication_30seed_v1
```

This held-out suffix full-baseline replication reruns the six-method
coverage-aware suffix comparison on seeds 30-59.

Mean results over 30 paired seeds:

| Method | Clean | Object RAUC | Median r80 | EE-transfer RAUC | RAUC AULC |
|---|---:|---:|---:|---:|---:|
| BGR-Coverage | 0.8641 | 0.4972 | 0.4989 | 0.3156 | 0.3833 |
| Uniform suffix | 0.8370 | 0.4859 | 0.5049 | 0.3096 | 0.3724 |
| Clean FT | 0.8650 | 0.2634 | 0.2073 | 0.1899 | 0.2394 |
| Failure-only | 0.7577 | 0.4237 | 0.4749 | 0.2441 | 0.3035 |
| Loss priority | 0.7115 | 0.1699 | 0.1291 | 0.1452 | 0.1696 |
| Fixed radius | 0.6852 | 0.1590 | 0.1247 | 0.1387 | 0.1654 |

BGR-Coverage beats clean-only, fixed-radius, failure-only, loss-priority, and
uniform suffix replay on final object RAUC, EE-transfer RAUC, and RAUC AULC with
30/0 paired wins. It also beats fixed-radius, failure-only, loss-priority, and
uniform on clean success with 30/0 paired wins. As in the original full-baseline
run, clean-only retains a tiny clean-success edge (0.8650 vs. 0.8641; 22/8
paired split), and uniform remains higher on median r80 with 30/0 paired wins.
Interpretation: held-out seeds independently replicate the full suffix
comparator ordering while preserving the clean-only and median-r80 caveats.

### `suffix_strategy_coverage_30seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_suffix_experiment.py \
  --config configs/suffix_strategy_coverage_30seed.yaml \
  --out results/suffix_strategy_coverage_30seed_v1
```

Cluster job `763770` completed on 2026-06-03 through the same suffix simulator
and coverage-aware BGR-Coverage settings as the 15-seed run, expanded to 30
paired seeds.

Mean results over 30 paired seeds:

| Method | Clean | Object RAUC | Median r80 | EE-transfer RAUC | RAUC AULC |
|---|---:|---:|---:|---:|---:|
| BGR-Coverage | 0.8644 | 0.4969 | 0.4982 | 0.3143 | 0.3825 |
| Uniform suffix | 0.8364 | 0.4854 | 0.5047 | 0.3083 | 0.3709 |

Paired exact sign tests in `paper/figures/significance_tests.csv` support
BGR-Coverage over uniform on clean success, final object RAUC, EE-transfer
RAUC, and RAUC AULC with 30/0 paired wins on each metric. Uniform remains
higher on median r80 with 29/1 paired wins. Interpretation: this upgrades the
coverage-aware suffix simulator evidence from a 15-seed positive diagnostic to
a 30-seed positive manipulation-style confirmation, while preserving the median
critical-radius caveat.

### Completed `suffix_strategy_coverage_replication_30seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_suffix_experiment.py \
  --config configs/suffix_strategy_coverage_replication_30seed.yaml \
  --out results/suffix_strategy_coverage_replication_30seed_v1
```

This local held-out replication reruns the two-method coverage-aware suffix
comparison on seeds 30-59, disjoint from the original 30 paired seeds.

Mean results over 30 paired seeds:

| Method | Clean | Object RAUC | Median r80 | EE-transfer RAUC | RAUC AULC |
|---|---:|---:|---:|---:|---:|
| BGR-Coverage | 0.8641 | 0.4972 | 0.4989 | 0.3156 | 0.3833 |
| Uniform suffix | 0.8370 | 0.4859 | 0.5049 | 0.3096 | 0.3724 |

Per-seed paired signs give 30/0 paired wins for BGR-Coverage over uniform on
clean success, final object RAUC, EE-transfer RAUC, and RAUC AULC. As in the
original 30-seed run, uniform remains higher on median r80 with 30/0 paired
wins. Interpretation: independent held-out seeds replicate the positive
BGR-Coverage-vs-uniform suffix result and preserve the median-critical-radius
caveat.

### Completed `suffix_strategy_ablation_30seed_v1`

Queued and completed on 2026-06-04 to turn the suffix strategy scan into a
paired 30-seed ablation. This tests whether the packaged BGR-Coverage result is
specifically a coverage-aware replay strategy rather than a one-off tuned radius
distribution.

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_suffix_experiment.py \
  --config configs/suffix_strategy_ablation_30seed.yaml \
  --out results/suffix_strategy_ablation_30seed_v1
```

Slurm submission:

```text
765450  first suffix strategy ablation submit from commit 25e85e5; failed immediately because the Slurm environment lacked `python`
765452  repaired sequential submit from commit 25e85e5 using `python3`; canceled after normal progress because the 120-run ablation was better handled as an array
765453  repaired array submit, tasks 0-119 with max 30 concurrent tasks
765454  dependent merge job, afterok:765453
```

All 120 array tasks completed and the dependent merge wrote the compact summary.

Mean results over 30 paired seeds:

| Method | Clean | Object RAUC | Median r80 | EE-transfer RAUC | RAUC AULC |
|---|---:|---:|---:|---:|---:|
| Uniform suffix | 0.8364 | 0.4854 | 0.5047 | 0.3083 | 0.3709 |
| Boundary-heavy BGR | 0.8716 | 0.4587 | 0.4296 | 0.2825 | 0.3690 |
| BGR-Coverage | 0.8644 | 0.4969 | 0.4982 | 0.3143 | 0.3825 |
| Hard-radius BGR | 0.8659 | 0.4850 | 0.4740 | 0.3252 | 0.3928 |

BGR-Coverage is the only suffix strategy that improves final object RAUC over
uniform with 30/0 paired wins. Boundary-heavy BGR undercovers object RAUC
(0.4587 vs. 0.4854 for uniform) with 30/0 paired losses. Hard-radius BGR leads
transfer RAUC and RAUC AULC (0.3252 and 0.3928), both with 30/0 paired wins over
uniform, but it does not improve final object RAUC (14/16 paired split vs.
uniform). Interpretation: the promoted suffix variant needs broad radius and
uniform state coverage to maintain final object-recovery gains, while hard
radius emphasis is useful for transfer/sample-efficiency but not the primary
object-RAUC claim.

### Completed `suffix_stress_sensitivity_30seed_v1`

Completed locally on 2026-06-04 to test whether the promoted BGR-Coverage suffix
result survives targeted simulator stress cases rather than only the nominal
suffix generator. The sweep uses the same coverage-aware BGR and uniform suffix
methods as the packaged suffix comparison, with four stress cases: low-teacher
quality, high clutter, tight feasibility, and diffuse boundaries.

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_suffix_stress_sensitivity.py \
  --config configs/suffix_stress_sensitivity_30seed.yaml \
  --out results/suffix_stress_sensitivity_30seed_v1
```

Mean results over 30 paired seeds:

| Stress case | Method | Clean | Object RAUC | Median r80 | EE-transfer RAUC | RAUC AULC |
|---|---|---:|---:|---:|---:|---:|
| Diffuse boundary | BGR-Coverage | 0.8594 | 0.4805 | 0.4420 | 0.3089 | 0.3737 |
| Diffuse boundary | Uniform suffix | 0.8318 | 0.4693 | 0.4495 | 0.3037 | 0.3637 |
| High clutter | BGR-Coverage | 0.8644 | 0.4971 | 0.4983 | 0.3142 | 0.3823 |
| High clutter | Uniform suffix | 0.8364 | 0.4853 | 0.5047 | 0.3081 | 0.3709 |
| Low teacher | BGR-Coverage | 0.6892 | 0.3806 | 0.4613 | 0.2267 | 0.2814 |
| Low teacher | Uniform suffix | 0.6647 | 0.3755 | 0.4808 | 0.2205 | 0.2720 |
| Tight feasible | BGR-Coverage | 0.8617 | 0.3987 | 0.3851 | 0.2379 | 0.3349 |
| Tight feasible | Uniform suffix | 0.8325 | 0.3860 | 0.3863 | 0.2314 | 0.3241 |

BGR-Coverage beats uniform with 30/0 paired wins on clean success, final object
RAUC, EE-transfer RAUC, and RAUC AULC in every stress case. As in the nominal
suffix runs, uniform retains the median-r80 caveat: BGR-Coverage has 0/30,
1/29, 0/30, and 3/27 median-r80 paired splits in diffuse-boundary, high-clutter,
low-teacher, and tight-feasibility cases. Interpretation: suffix stress tests
strengthen the manipulation-style simulator evidence while preserving the
critical-radius caveat and not converting the OpenVLA audit into a robotics
fine-tuning claim.

### `suffix_strategy_coverage_15seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_suffix_experiment.py \
  --config configs/suffix_strategy_coverage_15seed.yaml \
  --out results/suffix_strategy_coverage_15seed_v1
```

This run keeps the same suffix simulator and 15 paired seeds as
`suffix_strategy_pair_15seed_v1`, but uses a coverage-aware BGR-Broad setting:
`broad_uniform_radius_prob=0.80`, `broad_clean_radius_prob=0.05`, and
`uniform_mix=0.75`.

Mean results over 15 paired seeds:

| Method | Clean | Object RAUC | Median r80 | EE-transfer RAUC | RAUC AULC |
|---|---:|---:|---:|---:|---:|
| BGR-Coverage | 0.8638 | 0.4961 | 0.4970 | 0.3139 | 0.3821 |
| Uniform suffix | 0.8358 | 0.4844 | 0.5046 | 0.3075 | 0.3707 |

Paired exact sign tests in `paper/figures/significance_tests.csv` support
BGR-Coverage over uniform on clean success (+0.0280), final object RAUC
(+0.0116), EE-transfer RAUC (+0.0064), and RAUC AULC (+0.0114), with p=0.0001
after CSV formatting for each comparison. Uniform remains higher on median r80
(BGR-Coverage delta -0.0076, p=0.0001). Interpretation: the suffix simulator is
positive on final object RAUC and learning-curve area once BGR keeps enough broad
radius and uniform state coverage, but it remains a diagnostic simulator rather
than a real-robot/OpenVLA fine-tuning result.

### `estimator_full_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 01:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_estimator_experiment.py --config configs/estimator_bgr_full.yaml --out runs/estimator_full_v1
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780309253_221712673.out
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
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:30:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_estimator_experiment.py --config configs/estimator_pair_15seed.yaml --out runs/estimator_pair_15seed_v1
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780327938_068713790.out
```

Mean results over 15 paired seeds:

| Estimator | r80 MAE | RAUC MAE | Boundary hit rate |
|---|---:|---:|---:|
| Active BGR | 0.0799 | 0.0645 | 0.6775 |
| Uniform | 0.1064 | 0.0656 | 0.6007 |

Interpretation: active probing improves boundary-hit rate and lowers r80 error
relative to uniform under the same 17-probe budget. Paired exact sign tests give
p=0.00006 for both effects.

### Completed `estimator_pair_30seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_estimator_experiment.py \
  --config configs/estimator_pair_30seed.yaml \
  --out results/estimator_pair_30seed_v1
```

Completed locally on 2026-06-04 to confirm the active-estimator validation at
30 paired seeds under the same 17-probe budget and 512-state protocol as the
paper table.

Mean results over 30 paired seeds:

| Estimator | r80 MAE | RAUC MAE | Boundary hit rate |
|---|---:|---:|---:|
| Active BGR | 0.0806 | 0.0645 | 0.6701 |
| Uniform | 0.1055 | 0.0661 | 0.5949 |

The 30-seed confirmation shows active probing improves boundary-hit rate
(0.6701 vs. 0.5949) and lowers mean r80 error (0.0806 vs. 0.1055) with 30/0
paired wins on both effects. RAUC MAE is similar, with active slightly lower on
average (0.0645 vs. 0.0661) and 24/6 paired wins. Interpretation: this
strengthens the method-validation claim without changing the paper table or
claim scope.

### Completed `grid_margin_ablation_replication_30seed_v1`

Queued and completed on 2026-06-04 to rerun the radius-level mechanism ablation
on held-out seeds 30-59, disjoint from the original ablation seeds. This is a
replication of the paper's mechanism claim that BGR's gains come from
boundary-centered radius sampling rather than only replay-state priority.

Command shape:

```bash
PYTHONPATH=src:. python3 scripts/run_grid_margin_experiment.py \
  --config configs/grid_margin_ablation_replication_30seed.yaml \
  --out results/grid_margin_ablation_replication_30seed_v1
```

Slurm submission:

```text
765072  first 150-task method/seed array failed immediately because the Slurm wrapper used Bash array syntax under `/bin/sh`
765073  first merge job became DependencyNeverSatisfied and was cancelled
765223  repaired 150-task method/seed array, `0-149%30`, running with early tasks completing exit 0:0
765224  repaired merge job, afterok:765223
```

Mean results over 30 held-out paired seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.9453 | 0.4340 | 0.3446 | 0.3523 |
| No uncertainty | 0.9456 | 0.4339 | 0.3440 | 0.3522 |
| No sharpness | 0.9461 | 0.4343 | 0.3445 | 0.3525 |
| Uniform radius | 0.8904 | 0.3818 | 0.3225 | 0.3051 |
| Uniform replay | 0.8934 | 0.3967 | 0.3327 | 0.3137 |

BGR beats the uniform-radius ablation by +0.0522 final RAUC and +0.0472
RAUC AULC with 30/0 paired wins. The uniform-radius ablation is again worse
than uniform replay on final RAUC and RAUC AULC with 30/0 paired losses. The
no-uncertainty and no-sharpness variants remain effectively tied with BGR,
matching the original ablation. Interpretation: held-out seeds replicate the
mechanism result that the radius-level boundary rule, not merely hard-state
priority, drives the procedural grid-margin gains.

### Completed `grid_margin_ablation_30seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_grid_margin_experiment.py \
  --config configs/grid_margin_ablation_30seed.yaml \
  --out results/grid_margin_ablation_30seed_v1
```

Completed locally on 2026-06-04 to confirm the radius-level mechanism ablation
at the same 30 paired-seed level as the headline grid full-baseline result.

Mean results over 30 paired seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.9455 | 0.4344 | 0.3447 | 0.3526 |
| No uncertainty | 0.9455 | 0.4349 | 0.3443 | 0.3528 |
| No sharpness | 0.9464 | 0.4346 | 0.3444 | 0.3528 |
| Uniform radius | 0.8910 | 0.3819 | 0.3223 | 0.3051 |
| Uniform replay | 0.8939 | 0.3963 | 0.3321 | 0.3132 |

BGR beats the uniform-radius ablation by +0.0525 final RAUC and +0.0475 RAUC
AULC with 30/0 paired wins on both metrics. The uniform-radius ablation is also
worse than uniform replay by -0.0144 final RAUC and -0.0081 RAUC AULC, again
with 30/0 paired wins in the lower-is-expected direction. The no-uncertainty and
no-sharpness variants remain effectively tied with BGR, matching the original
15-seed diagnosis. Interpretation: this confirms at 30 paired seeds that
radius-level boundary sampling, not merely hard-state priority, drives the
procedural grid-margin gains.

### `grid_margin_ablation_15seed_v1`

Command:

```bash
PYTHONPATH=src:. python3 scripts/run_grid_margin_experiment.py \
  --config configs/grid_margin_ablation_15seed.yaml \
  --out results/grid_margin_ablation_15seed_v1
```

Mean results over 15 paired seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.9461 | 0.4345 | 0.3441 | 0.3525 |
| No uncertainty | 0.9460 | 0.4345 | 0.3438 | 0.3525 |
| No sharpness | 0.9468 | 0.4342 | 0.3441 | 0.3523 |
| Uniform radius | 0.8910 | 0.3816 | 0.3217 | 0.3046 |
| Uniform replay | 0.8943 | 0.3961 | 0.3318 | 0.3129 |

Paired exact sign tests show that BGR beats the uniform-radius ablation on
final RAUC by +0.0529 and RAUC AULC by +0.0479 (p=0.0001). The uniform-radius
ablation is also worse than uniform replay on final RAUC by -0.0145 (p=0.0001).
BGR is statistically indistinguishable from the no-uncertainty and no-sharpness
variants on final RAUC. Interpretation: this 15-seed ablation strengthens the
mechanism claim that radius-level boundary sampling, not merely hard-state
priority, drives the procedural grid-margin gains.

### `grid_margin_ablation_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_ablation.yaml --out runs/grid_margin_ablation_v1
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780309573_174287695.out
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
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 00:45:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_policy_hard_fast.yaml --out runs/grid_policy_hard_fast_v1
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 00:45:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_policy_mixed.yaml --out runs/grid_policy_mixed_v1
```

Remote logs:

```text
/work/anonymous/bgr/logs/run_1780316465_289961157.out
/work/anonymous/bgr/logs/run_1780317123_541002217.out
```

Mean results over three seeds for the mixed run:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.7578 | 0.7096 | 0.7583 | 0.5473 |
| BGR mixed | 0.7839 | 0.7589 | 0.8463 | 0.5861 |
| Uniform | 0.9965 | 0.9641 | 0.9833 | 0.7838 |
| Fixed radius | 0.9931 | 0.9771 | 1.0000 | 0.7999 |
| PLR-loss proxy | 0.9991 | 0.9786 | 1.0000 | 0.7960 |

Interpretation: this is a negative policy-level diagnostic. Adding clean/uniform
radius coverage improves tabular BGR, but fixed-radius and loss-priority replay
saturate the tabular oracle-imitation grid policy much faster. This historical
policy-level setup is a benchmark-design failure, not evidence for BGR; the
reviewer-facing procedural claim is the completed 30-seed grid-margin
full-baseline comparison plus held-out grid replication listed in the Submission
Evidence Index.

### `grid_policy_coverage_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 00:45:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_grid_experiment.py --config configs/grid_policy_coverage.yaml --out runs/grid_policy_coverage_v1
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780319352_975146235.out
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
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 01:00:00 /work/anonymous/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/probe_libero_suffix_states.py --suite libero_goal --task-ids 0,1,2,3,4 --init-state-ids 0,1,2 --radii 0.0,0.25,0.5,0.75,1.0 --trials-per-radius 4 --settle-steps 5 --image-size 64 --out runs/libero_probe_v2
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780311860_935283441.out
```

Probe summary:

| Suite | Tasks | Init states/task | Radii | Trials/radius | Valid rate | Zero-action success |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO-Goal | 5 | 3 | 5 | 4 | 1.0000 | 0.0000 |

Interpretation: the cluster can instantiate LIBERO environments on GPU with `MUJOCO_GL=egl`, load trusted local init states after patching PyTorch 2.6's `torch.load(weights_only=True)` default, and apply object free-joint perturbations across resettable LIBERO states. This is infrastructure evidence for BGR-Suffix on real LIBERO simulator states, not a policy result: this probe uses zero-action rollouts, which are not expected to solve tasks. A separate fixed-policy OpenVLA recovery summary is recorded in `libero_openvla_recovery_v1`.

### `libero_openvla_recovery_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/summarize_libero_openvla_recovery.py --input-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_object3_h220_bash --out runs/libero_openvla_recovery_v1 --source-name libero_openvla_observation_object3_h220_bash
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780320300_854946446.out
```

Source artifact:

```text
/work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_object3_h220_bash
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
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/summarize_openvla_boundary_selection.py --proposal-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_guided_h160 --proposal-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_guided_seed2_h160 --proposal-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_guided_seed3_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed1b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed2b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed3b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed4b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed5b_skip_lp2_h160 --out runs/libero_openvla_boundary_selection_v1
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780320651_072406017.out
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
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/summarize_openvla_boundary_selection.py --proposal-method-name bgr_boundary --proposal-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed1_lp2_h160 --proposal-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed2_lp2_h160 --proposal-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed3_lp2_h160 --proposal-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed4_lp2_h160 --proposal-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed5_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed1b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed2b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed3b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed4b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed5b_skip_lp2_h160 --out runs/libero_openvla_boundary_selection_balanced_v1
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780321470_140661271.out
```

| Method | Runs | Mean CF rate | Boundary hit rate | Mean `abs(CF-0.5)` | Certificates |
|---|---:|---:|---:|---:|---:|
| BGR-boundary | 5 | 0.5500 | 0.6250 | 0.2958 | 16.00 |
| Random-balanced | 5 | 0.6438 | 0.5750 | 0.2563 | 20.60 |

Interpretation: this is the paper-table OpenVLA selection audit because both
methods cover the same four perturbation families over five runs. It supports
boundary-discovery as a measurable learned-policy diagnostic while staying
separate from the later OpenVLA-OFT fine-tuning smoke and audit results.

### `openvla_bgr_finetune_manifest_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/export_openvla_bgr_finetune_manifest.py --bgr-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed1_lp2_h160 --bgr-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed2_lp2_h160 --bgr-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed3_lp2_h160 --bgr-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed4_lp2_h160 --bgr-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed5_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed1b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed2b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed3b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed4b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed5b_skip_lp2_h160 --out runs/openvla_bgr_finetune_manifest_v1
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780321861_807492940.out
```

Interpretation: this exports validated candidate-level manifests and OpenVLA-OFT
LoRA command templates. It identified 80 candidates total; BGR-boundary has
25/40 candidates in the boundary band, and random-balanced has 23/40. The
later teacher-replay, TFDS, and LoRA smoke sections supersede this as a
fine-tuning-readiness milestone.

### `openvla_teacher_replay_manifest_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 12G --time 00:15:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/export_openvla_teacher_replay_manifest.py --boundary-only --max-steps-per-episode 64 --bgr-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed1_lp2_h160 --bgr-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed2_lp2_h160 --bgr-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed3_lp2_h160 --bgr-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed4_lp2_h160 --bgr-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed5_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed1b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed2b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed3b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed4b_skip_lp2_h160 --random-dir /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed5b_skip_lp2_h160 --out runs/openvla_teacher_replay_manifest_v1
```

Remote logs:

```text
/work/anonymous/bgr/logs/run_1780322114_709717949.out
/work/anonymous/bgr/logs/run_1780322189_156724835.out
```

Interpretation: this bridge artifact exports 11,776 step-level rows pairing
validated boundary candidates with target actions from successful native OpenVLA
rollouts. Later renderer, packing, TFDS, and LoRA smoke sections close the
downstream conversion path needed for OpenVLA-OFT audits.

### `openvla_teacher_oft_smoke_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:20:00 /work/anonymous/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_oft_smoke_v1 --max-examples 4 --selection first_per_family --num-steps-wait 10 --env-image-size 256 --image-size 224
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780322998_166835773.out
```

Interpretation: this validates that the teacher-replay manifest can be converted
into rendered OpenVLA-OFT-style examples under LIBERO GPU/EGL. The included
smoke artifact contains one example for each visual perturbation family: blur,
brightness, shift, and occlusion. Each NPZ contains primary image
`(224,224,3)`, wrist image `(224,224,3)`, 8D LIBERO state, 7D action, and
language label. This is still a bridge artifact, not RLDS conversion or
fine-tuning.

### `openvla_oft_pack_smoke_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/pack_openvla_oft_examples.py --examples results/openvla_teacher_oft_smoke_v1/examples.jsonl --out runs/openvla_oft_pack_smoke_v1 --write-hdf5
```

Remote logs:

```text
/work/anonymous/bgr/logs/run_1780323325_136783423.out
/work/anonymous/bgr/logs/run_1780323386_780730416.out
```

Interpretation: this validates and packs the rendered OFT-field examples into a
LIBERO-style HDF5 smoke dataset. The included HDF5 has `data/demo_*` groups
with `actions`, `obs/agentview_rgb`, `obs/eye_in_hand_rgb`, `obs/ee_states`,
and `obs/gripper_states`. Later TFDS exports supersede this HDF5 smoke as the
OpenVLA-OFT loader-facing artifact.

### `openvla_oft_tfds_smoke_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:20:00 /work/anonymous/bgr /work/anonymous/safesae-openvla/bin/python scripts/export_openvla_oft_tfds.py --examples results/openvla_teacher_oft_smoke_v1/examples.jsonl --out runs/openvla_oft_tfds_smoke_v1 --dataset-name bgr_libero_oft_smoke --version 1.0.0
```

Remote logs:

```text
/work/anonymous/bgr/logs/run_1780323827_280717056.out
/work/anonymous/bgr/logs/run_1780323900_674014804.out
```

Interpretation: this exports the rendered OFT-field examples as a minimal
RLDS-style TFDS dataset and verifies readback with `tfds.builder_from_directory`.
The included smoke dataset has four train episodes and four total steps, one
per perturbation family, with primary RGB, wrist RGB, 8D state, 7D action, and
language fields. This closes the smoke-tested TFDS conversion path, but it is
a TFDS compatibility check rather than a full OpenVLA-OFT fine-tuning run.

### `openvla_oft_tfds_libero_goal_smoke_v2`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:20:00 /work/anonymous/bgr /work/anonymous/safesae-openvla/bin/python scripts/export_openvla_oft_tfds.py --examples results/openvla_teacher_oft_smoke_v1/examples.jsonl --out runs/openvla_oft_tfds_libero_goal_smoke_v2 --dataset-name libero_goal_no_noops --version 1.0.0
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 12G --time 00:15:00 /work/anonymous/bgr env PYTHONPATH=/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/python -c "from pathlib import Path; import tensorflow_datasets as tfds; from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs; from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds; kw=make_oxe_dataset_kwargs('libero_goal_no_noops', Path('/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_smoke_v2'), load_camera_views=('primary','wrist'), load_proprio=True, load_language=True); ds, stats=make_dataset_from_rlds(train=True, shuffle=False, **kw); ex=next(iter(tfds.as_numpy(ds))); print('keys', sorted(ex.keys())); print('obs', {k:v.shape for k,v in ex['observation'].items()}); print('task', ex['task']['language_instruction'][0].decode()); print('action', ex['action'].shape); print('stats_action_mean', stats['action']['mean'].shape); print('stats_proprio_mean', stats['proprio']['mean'].shape)"
```

Remote logs:

```text
/work/anonymous/bgr/logs/run_1780324900_610846337.out
/work/anonymous/bgr/logs/run_1780324943_628488822.out
```

Interpretation: this re-exports the TFDS smoke under the stock
`libero_goal_no_noops` name and validates it through OpenVLA-OFT's own
`make_oxe_dataset_kwargs` and `make_dataset_from_rlds` path. The loader computes
dataset statistics and yields primary/wrist image fields, 8D proprio, 7D action,
and language. This verifies that the BGR-rendered examples can enter the
unmodified OpenVLA-OFT RLDS loader. Later balanced2048 sections scale this path
to matched larger datasets and LoRA fine-tuning/evaluation smoke runs.

### `openvla_teacher_oft_balanced64_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:40:00 /work/anonymous/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_oft_balanced64_v1 --max-examples 64 --selection balanced_episodes --episodes-per-family 1 --max-steps-per-episode 16 --num-steps-wait 10 --env-image-size 256 --image-size 224
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780325448_898879565.out
```

Interpretation: this scales the renderer from four isolated smoke frames to
four contiguous 16-step replay episodes, one per visual perturbation family.
Each rendered row contains primary RGB, wrist RGB, 8D LIBERO state, 7D teacher
action, and language.

### `openvla_oft_pack_balanced64_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/pack_openvla_oft_examples.py --examples runs/openvla_teacher_oft_balanced64_v1/examples.jsonl --out runs/openvla_oft_pack_balanced64_v1 --write-hdf5
```

Remote logs:

```text
/work/anonymous/bgr/logs/run_1780325655_972521234.out
/work/anonymous/bgr/logs/run_1780325709_771885875.out
```

Interpretation: the balanced rendered rows pack into four HDF5 demos with
`actions (16,7)`, `obs/agentview_rgb (16,224,224,3)`, wrist RGB, EEF states,
and gripper states. This verifies episode-safe grouping for multi-step
candidate replays.

### `openvla_oft_tfds_libero_goal_balanced64_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:20:00 /work/anonymous/bgr /work/anonymous/safesae-openvla/bin/python scripts/export_openvla_oft_tfds.py --examples runs/openvla_teacher_oft_balanced64_v1/examples.jsonl --out runs/openvla_oft_tfds_libero_goal_balanced64_v1 --dataset-name libero_goal_no_noops --version 1.0.0
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 12G --time 00:15:00 /work/anonymous/bgr env PYTHONPATH=/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/python -c "from pathlib import Path; import tensorflow_datasets as tfds; from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs; from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds; kw=make_oxe_dataset_kwargs('libero_goal_no_noops', Path('/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_balanced64_v1'), load_camera_views=('primary','wrist'), load_proprio=True, load_language=True); ds, stats=make_dataset_from_rlds(train=True, shuffle=False, **kw); ex=next(iter(tfds.as_numpy(ds))); print('keys', sorted(ex.keys())); print('obs', {k:v.shape for k,v in ex['observation'].items()}); print('task', ex['task']['language_instruction'][0].decode()); print('action', ex['action'].shape); print('stats_action_mean', stats['action']['mean'].shape); print('stats_proprio_mean', stats['proprio']['mean'].shape)"
```

Remote logs:

```text
/work/anonymous/bgr/logs/run_1780325681_170623263.out
/work/anonymous/bgr/logs/run_1780325731_654888892.out
```

Interpretation: this exports the 64-step balanced set under
`libero_goal_no_noops` and validates it through OpenVLA-OFT's unmodified RLDS
loader. The loader computes dataset statistics and yields trajectory chunks with
primary/wrist image fields, proprio `(16,8)`, action `(16,7)`, and language.
Later balanced2048 sections supersede this 64-step compatibility check for
training-sized scale and LoRA fine-tuning/evaluation smoke coverage.

### `openvla_oft_finetune_balanced64_ckpt_smoke_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 8 --mem 64G --time 01:00:00 /work/anonymous/bgr bash -lc 'cd /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft && env WANDB_MODE=disabled HF_HOME=/work/anonymous/cache_home/huggingface TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub PYTHONPATH=/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path openvla/openvla-7b --data_root_dir /work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_balanced64_v1 --dataset_name libero_goal_no_noops --run_root_dir /work/anonymous/bgr/runs/openvla_oft_finetune_balanced64_ckpt_smoke_v1 --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio True --batch_size 1 --learning_rate 5e-4 --num_steps_before_decay 100000 --max_steps 1 --save_freq 1 --save_latest_checkpoint_only True --image_aug False --lora_rank 8 --merge_lora_during_training False --shuffle_buffer_size 16 --wandb_entity disabled --wandb_project bgr --run_id_note bgr-balanced64-ckpt-smoke --wandb_log_freq 1'
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780326595_120788212.out
```

Interpretation: this is the first end-to-end OpenVLA-OFT training smoke on BGR
data. The run loaded OpenVLA 7B, initialized LoRA rank 8 plus proprio and L1
action heads, loaded the BGR `libero_goal_no_noops` TFDS dataset, completed a
one-step forward/backward/update loop, and wrote checkpoint artifacts under
`/work/anonymous/bgr/runs`. The checkpoint directory is 656M and is not included in
the compact artifact; included metadata records the saved LoRA adapter, action head, proprio
projector, and dataset statistics. This is still a smoke test, not a
training-scale fine-tuning/evaluation result.

### `openvla_teacher_oft_random_balanced64_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:40:00 /work/anonymous/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_oft_random_balanced64_v1 --method random_balanced --max-examples 64 --selection balanced_episodes --episodes-per-family 1 --max-steps-per-episode 16 --num-steps-wait 10 --env-image-size 256 --image-size 224
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780327075_235451672.out
```

Interpretation: this renders the matched random-balanced baseline analog of
the BGR balanced64 bridge. Validation found 64 rows, all `random_balanced`,
split evenly across blur, brightness, shift, and occlusion with one 16-step
episode per family.

### `openvla_oft_pack_random_balanced64_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/pack_openvla_oft_examples.py --examples runs/openvla_teacher_oft_random_balanced64_v1/examples.jsonl --out runs/openvla_oft_pack_random_balanced64_v1 --write-hdf5
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780327145_234510679.out
```

Interpretation: the random-balanced rendered rows pack into four HDF5 demos
with 7D actions, 8D proprio/state, and primary/wrist image streams. This
confirms the baseline uses the same episode-safe bridge as BGR.

### `openvla_oft_tfds_libero_goal_random_balanced64_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:20:00 /work/anonymous/bgr /work/anonymous/safesae-openvla/bin/python scripts/export_openvla_oft_tfds.py --examples runs/openvla_teacher_oft_random_balanced64_v1/examples.jsonl --out runs/openvla_oft_tfds_libero_goal_random_balanced64_v1 --dataset-name libero_goal_no_noops --version 1.0.0
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 12G --time 00:15:00 /work/anonymous/bgr env PYTHONPATH=/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/python -c "from pathlib import Path; import tensorflow_datasets as tfds; from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs; from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds; kw=make_oxe_dataset_kwargs('libero_goal_no_noops', Path('/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_balanced64_v1'), load_camera_views=('primary','wrist'), load_proprio=True, load_language=True); ds, stats=make_dataset_from_rlds(train=True, shuffle=False, **kw); ex=next(iter(tfds.as_numpy(ds))); print('keys', sorted(ex.keys())); print('obs', {k:v.shape for k,v in ex['observation'].items()}); print('task', ex['task']['language_instruction'][0].decode()); print('action', ex['action'].shape); print('stats_action_mean', stats['action']['mean'].shape); print('stats_proprio_mean', stats['proprio']['mean'].shape)"
```

Remote logs:

```text
/work/anonymous/bgr/logs/run_1780327170_100921796.out
/work/anonymous/bgr/logs/run_1780327240_909792233.out
```

Interpretation: this exports the matched random-balanced set under
`libero_goal_no_noops` and validates it through OpenVLA-OFT's unmodified RLDS
loader. The loader yields primary/wrist image fields, proprio `(16,8)`, action
`(16,7)`, and language, matching the BGR bridge.

### `openvla_oft_finetune_random_balanced64_ckpt_smoke_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 8 --mem 64G --time 01:00:00 /work/anonymous/bgr bash -lc 'cd /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft && env WANDB_MODE=disabled HF_HOME=/work/anonymous/cache_home/huggingface TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub PYTHONPATH=/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path openvla/openvla-7b --data_root_dir /work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_balanced64_v1 --dataset_name libero_goal_no_noops --run_root_dir /work/anonymous/bgr/runs/openvla_oft_finetune_random_balanced64_ckpt_smoke_v1 --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio True --batch_size 1 --learning_rate 5e-4 --num_steps_before_decay 100000 --max_steps 1 --save_freq 1 --save_latest_checkpoint_only True --image_aug False --lora_rank 8 --merge_lora_during_training False --shuffle_buffer_size 16 --wandb_entity disabled --wandb_project bgr --run_id_note random-balanced64-ckpt-smoke --wandb_log_freq 1'
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780327341_446370402.out
```

Interpretation: this mirrors the BGR one-step OpenVLA-OFT checkpoint smoke on
the matched random-balanced baseline. It loaded OpenVLA 7B, initialized LoRA
rank 8 plus proprio and L1 action heads, loaded the baseline
`libero_goal_no_noops` TFDS dataset, completed one optimizer step, and wrote the
same class of checkpoint files under `/work/anonymous/bgr/runs`. The 656M checkpoint
weights are not included in the compact artifact.

### `openvla_teacher_oft_bgr_balanced2048_v1` and `openvla_teacher_oft_random_balanced2048_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 24G --time 01:30:00 /work/anonymous/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_oft_bgr_balanced2048_v1 --method bgr_boundary --max-examples 2048 --selection balanced_episodes --episodes-per-family 8 --max-steps-per-episode 64 --num-steps-wait 10 --env-image-size 256 --image-size 224
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 24G --time 01:30:00 /work/anonymous/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_oft_random_balanced2048_v1 --method random_balanced --max-examples 2048 --selection balanced_episodes --episodes-per-family 8 --max-steps-per-episode 64 --num-steps-wait 10 --env-image-size 256 --image-size 224
```

Remote logs:

```text
/work/anonymous/bgr/logs/run_1780379901_989185344.out
/work/anonymous/bgr/logs/run_1780380209_343397743.out
```

Interpretation: this scales the OpenVLA render bridge to matched 2,048-step
BGR-boundary and random-balanced datasets. Each render has 32 replay episodes,
8 episodes per visual perturbation family, and 64 steps per episode. The full
render trees are 492M and 524M under `/work` and are not included in the compact artifact.

### `openvla_oft_pack_bgr_balanced2048_v1` and `openvla_oft_pack_random_balanced2048_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 16G --time 00:30:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/pack_openvla_oft_examples.py --examples runs/openvla_teacher_oft_bgr_balanced2048_v1/examples.jsonl --out runs/openvla_oft_pack_bgr_balanced2048_v1 --write-hdf5
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 16G --time 00:30:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/pack_openvla_oft_examples.py --examples runs/openvla_teacher_oft_random_balanced2048_v1/examples.jsonl --out runs/openvla_oft_pack_random_balanced2048_v1 --write-hdf5
```

Remote logs:

```text
/work/anonymous/bgr/logs/run_1780380477_635336557.out
/work/anonymous/bgr/logs/run_1780380526_513758944.out
```

Interpretation: both matched 2,048-step renders pack into LIBERO-style HDF5
with 7D actions, 8D state, and primary/wrist image streams. The full HDF5 packs
are 383M and 399M under `/work`.

### `openvla_oft_tfds_libero_goal_bgr_balanced2048_v1`, `openvla_oft_tfds_libero_goal_random_balanced2048_v1`, and `openvla_oft_loader_balanced2048_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 16G --time 01:00:00 /work/anonymous/bgr /work/anonymous/safesae-openvla/bin/python scripts/export_openvla_oft_tfds.py --examples runs/openvla_teacher_oft_bgr_balanced2048_v1/examples.jsonl --out runs/openvla_oft_tfds_libero_goal_bgr_balanced2048_v1 --dataset-name libero_goal_no_noops --version 1.0.0
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 16G --time 01:00:00 /work/anonymous/bgr /work/anonymous/safesae-openvla/bin/python scripts/export_openvla_oft_tfds.py --examples runs/openvla_teacher_oft_random_balanced2048_v1/examples.jsonl --out runs/openvla_oft_tfds_libero_goal_random_balanced2048_v1 --dataset-name libero_goal_no_noops --version 1.0.0
```

Loader validation uses OpenVLA-OFT's unmodified `make_oxe_dataset_kwargs` and
`make_dataset_from_rlds` on each TFDS root.

Remote logs:

```text
/work/anonymous/bgr/logs/run_1780380577_704261678.out
/work/anonymous/bgr/logs/run_1780380831_695183835.out
/work/anonymous/bgr/logs/run_1780381106_407628579.out
/work/anonymous/bgr/logs/run_1780381221_375772232.out
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
~/remote_srun.sh --github-test --git-pull --log --partition low-prio-gpu --gres gpu:a6000:1 --cpus 8 --mem 64G --time 02:00:00 /work/anonymous/bgr bash -lc 'cd /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft && env WANDB_MODE=disabled HF_HOME=/work/anonymous/cache_home/huggingface TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub PYTHONPATH=/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path openvla/openvla-7b --data_root_dir /work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_balanced2048_v1 --dataset_name libero_goal_no_noops --run_root_dir /work/anonymous/bgr/runs/openvla_oft_finetune_bgr_balanced2048_step10_v1 --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio True --batch_size 1 --learning_rate 5e-4 --num_steps_before_decay 100000 --max_steps 10 --save_freq 10 --save_latest_checkpoint_only True --image_aug False --lora_rank 8 --merge_lora_during_training False --shuffle_buffer_size 512 --wandb_entity disabled --wandb_project bgr --run_id_note bgr-balanced2048-step10 --wandb_log_freq 1'
~/remote_srun.sh --github-test --git-pull --log --partition low-prio-gpu --gres gpu:a6000:1 --cpus 8 --mem 64G --time 02:00:00 /work/anonymous/bgr bash -lc 'cd /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft && env WANDB_MODE=disabled HF_HOME=/work/anonymous/cache_home/huggingface TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub PYTHONPATH=/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path openvla/openvla-7b --data_root_dir /work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_balanced2048_v1 --dataset_name libero_goal_no_noops --run_root_dir /work/anonymous/bgr/runs/openvla_oft_finetune_random_balanced2048_step10_v1 --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio True --batch_size 1 --learning_rate 5e-4 --num_steps_before_decay 100000 --max_steps 10 --save_freq 10 --save_latest_checkpoint_only True --image_aug False --lora_rank 8 --merge_lora_during_training False --shuffle_buffer_size 512 --wandb_entity disabled --wandb_project bgr --run_id_note random-balanced2048-step10 --wandb_log_freq 1'
```

Remote logs:

```text
/work/anonymous/bgr/logs/run_1780381812_955203657.out
/work/anonymous/bgr/logs/run_1780382026_303226869.out
```

Interpretation: both matched 2,048-step TFDS exports complete scoped
OpenVLA-OFT LoRA fine-tuning smoke runs. Each run loads OpenVLA 7B,
initializes LoRA rank 8 plus proprio and L1 action heads, trains for 10
optimizer steps, and writes a latest checkpoint under
`/work/anonymous/bgr/runs`. The remote checkpoint trees are 656M each and are
not included in the compact artifact. This upgrades the OpenVLA bridge from
dataset compatibility to matched checkpoint generation.

### `openvla_oft_eval_balanced2048_step10_smoke_v1`

Commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition low-prio-gpu --gres gpu:a6000:1 --cpus 8 --mem 80G --time 02:00:00 /work/anonymous/bgr bash -lc 'cd /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft && env WANDB_MODE=disabled HF_HOME=/work/anonymous/cache_home/huggingface TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub PYTHONPATH=/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/python vla-scripts/merge_lora_weights_and_save.py --base_checkpoint openvla/openvla-7b --lora_finetuned_checkpoint_dir /work/anonymous/bgr/runs/openvla_oft_finetune_bgr_balanced2048_step10_v1/openvla-7b+libero_goal_no_noops+b1+lr-0.0005+lora-r8+dropout-0.0--bgr-balanced2048-step10'
~/remote_srun.sh --github-test --git-pull --log --partition low-prio-gpu --gres gpu:a6000:1 --cpus 8 --mem 80G --time 02:00:00 /work/anonymous/bgr bash -lc 'cd /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft && env WANDB_MODE=disabled HF_HOME=/work/anonymous/cache_home/huggingface TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub PYTHONPATH=/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft /work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/python vla-scripts/merge_lora_weights_and_save.py --base_checkpoint openvla/openvla-7b --lora_finetuned_checkpoint_dir /work/anonymous/bgr/runs/openvla_oft_finetune_random_balanced2048_step10_v1/openvla-7b+libero_goal_no_noops+b1+lr-0.0005+lora-r8+dropout-0.0--random-balanced2048-step10'
```

Eval commands use `run_libero_eval.py` with the merged checkpoint roots,
`task_suite_name=libero_goal`, `num_tasks=1`, `num_trials_per_task=1`, and
`max_steps_override=20`. The required runtime path is:

```text
PYTHONPATH=/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft:/home/anonymous/LIBERO:/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/lib/python3.10/site-packages
```

Remote logs:

```text
/work/anonymous/bgr/logs/run_1780382513_105504870.out
/work/anonymous/bgr/logs/run_1780382656_581424411.out
/work/anonymous/bgr/logs/run_1780383317_343334235.out
/work/anonymous/bgr/logs/run_1780383458_588753145.out
```

Interpretation: this closes an OpenVLA infrastructure gap. Both matched
10-step LoRA checkpoints merge into 15G local OpenVLA checkpoint roots, load
through the stock OpenVLA-OFT LIBERO eval script, instantiate LIBERO-Goal, run
one closed-loop rollout, save a rollout video, and log final results. Both score
0/1 in this tiny smoke, so this is not policy-quality evidence; it verifies the
merge/load/closed-loop evaluation path needed for longer matched evaluations.

### `suffix_strategy_v1`

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 02:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_strategy.yaml --out runs/suffix_strategy_v1
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780312179_117921719.out
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
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/anonymous/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_strategy_pair_15seed.yaml --out runs/suffix_strategy_pair_15seed_v1
```

Remote log:

```text
/work/anonymous/bgr/logs/run_1780328104_484552023.out
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

### `minatar_asterix_recovery_calibration_12seed_v1`

Command:

```bash
PYTHONPATH=src:. /tmp/bgr_minatar_venv/bin/python tools/minatar_asterix_recovery_calibration.py --out results/minatar_asterix_recovery_calibration_12seed_v1
```

Package environment: `/tmp/bgr_minatar_venv` with `MinAtar==1.0.15` and
`numpy==2.4.6`.

Result: the fixed Asterix pre-method calibration clears the clean/non-flat/
non-saturated prerequisites with clean success 0.8333, recovery range
0.5000--0.8333, RAUC 0.7188, and r80 5.3333 on the 0--8 seed-fixed
player-cell displacement grid. This is not BGR evidence; it only authorizes the
fixed all-method Asterix screen under the same package-backed reset interface.

### `minatar_asterix_recovery_probe_4seed_v1`

Command:

```bash
PYTHONPATH=src:. /tmp/bgr_minatar_venv/bin/python tools/minatar_asterix_recovery_probe.py --out results/minatar_asterix_recovery_probe_4seed_v1
```

Package environment: `/tmp/bgr_minatar_venv` with `MinAtar==1.0.15` and
`numpy==2.4.6`.

Mean results over four paired seeds:

| Method | Clean | RAUC | Median r80 | AULC |
|---|---:|---:|---:|---:|
| Failure-only | 1.0000 | 0.8625 | 6.2000 | 0.8209 |
| BGR-Coverage | 1.0000 | 0.8406 | 5.9500 | 0.7400 |
| BGR-uniform-radius | 1.0000 | 0.8313 | 5.9500 | 0.8143 |
| Uniform | 1.0000 | 0.8234 | 5.4500 | 0.8000 |
| BGR | 1.0000 | 0.8047 | 5.3250 | 0.7977 |
| TD-loss | 1.0000 | 0.8031 | 5.5750 | 0.8080 |
| Fixed-radius | 0.9250 | 0.7734 | 3.7000 | 0.7773 |

Interpretation: the fixed Asterix all-method screen is negative. BGR-Coverage
beats uniform on mean RAUC but loses to the failure-only baseline and has only
1/2/1 paired wins/losses/ties against uniform, so it fails the promotion gate
and should not be scaled or promoted under this protocol.
