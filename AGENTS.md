# AGENTS.md

This file is the project handoff contract for coding agents working in this
repository. Read it before editing code, running experiments, changing paper
claims, or committing results.

## Goal Snapshot

Last verified: 2026-06-07.

Persistent thread goal: iterate, queue experiments, and reframe the paper until
it is plausibly 90%+ likely to get into AAAI.

Current answer to "what is our goal?": make the BGR paper genuinely
main-track-defensible, not merely better phrased. The work is only successful
when the evidence gates support a high-confidence AAAI submission: independent
or pre-existing benchmark evidence, a learned-policy win under a fixed gate, a
stronger formal result, or presentation/theory improvements that materially
change reviewer risk.

Current acceptance answer: not ready. The internal readiness check still reports
`NOT_READY_FOR_90P_AAAI_CLAIM`; the controlled grid result is positive, but the
independent/pre-existing benchmark gate and OpenVLA/LIBERO learned-policy gate
are failing.

Acceptance reality: the project is not there yet. The current evidence supports
a careful mechanism study, not a high-confidence AAAI main-track claim. Make
progress by adding preregistered positive evidence, fixing standard-benchmark or
learned-policy failures, strengthening the theory, or improving required
figures/presentation. Do not treat softer wording around unchanged results as
acceptance progress.

Operational rule of thumb: commit and push clean, explicit checkpoints to
`origin/main`; do not use Docker for this project; and incorporate new results
into `paper/main.tex` only after the artifacts and claim/package checks support
the claim.

## Standing Instructions

- Active goal: iterate, queue experiments, and reframe the paper until it is
  plausibly 90%+ likely to clear the AAAI main-track bar.
- Current acceptance status: `NOT_READY_FOR_90P_AAAI_CLAIM`.
- Keep pushing clean checkpoints to GitHub.
- Do not use Docker. Use local Python environments, temporary virtualenvs, and
  the `athena` Slurm workflow already used by this repo.
- Incorporate new results into `paper/main.tex` only when they satisfy the
  evidence policy below and have source artifacts/checker coverage.
- Preserve unrelated local or user-created changes. Stage explicit files only.

## Working Objective

Short version: move BGR from a careful-but-not-main-track-ready mechanism study
to a genuinely defensible AAAI main-track submission, and do not claim success
until the readiness gates support that.

The active objective is to move this repository toward a genuinely high-confidence AAAI main-track paper. Do not treat wording-only reframing as success. The core blocker is evidence: a promoted result must beat strong baselines on a fixed or pre-existing benchmark with non-contradictory metrics.

Current target: iterate, queue experiments, and reframe the paper until the work is plausibly 90%+ likely to clear the AAAI main-track bar. Current status is below that bar: the strongest positive evidence is still the controlled grid-margin mechanism result, while standard-environment and learned-policy probes remain negative or non-promotable.

As of 2026-06-07, `PYTHONPATH=src:. python3 scripts/check_acceptance_readiness.py --root .` reports:

- PASS controlled grid mechanism: pooled RAUC 0.4342 vs 0.3965.
- FAIL independent/pre-existing benchmarks: FrozenLake, MiniGrid FourRooms, MiniGrid DoorKey, MiniGrid LavaCrossing, MiniGrid LavaGapS7, PointMaze, FetchReach, and Reacher remain non-promotable.
- FAIL learned-policy OpenVLA/LIBERO: the latest completed proximal-anchor audit has non-identity success BGR 368/400, official 367/400, and matched random 368/400, with identity BGR 98/100, official 99/100, and random 98/100. BGR ties matched random and beats official by only 1/400, so it fails the fixed +10/400 and +0.02 promotion gate. The earlier weighted perturbation audit is also negative at BGR 367/400, official 367/400, and matched random 370/400.
- Decision: `NOT_READY_FOR_90P_AAAI_CLAIM`.

The practical goal is not to make the paper sound accepted. The practical goal is to find or build defensible evidence that survives the acceptance criteria below, then incorporate only those results into `paper/main.tex`.
Use `PYTHONPATH=src:. python3 scripts/acceptance_scorecard.py --root . --out docs/acceptance_scorecard.md` to quantify current distances to the internal gates before choosing the next experiment.

## Current Acceptance Status

- The internal readiness gate is intentionally failing; do not report the paper as AAAI-ready.
- Controlled grid-margin evidence is positive and mechanistically useful, but it is not enough for a high-confidence main-track acceptance claim.
- FrozenLake, MiniGrid FourRooms, MiniGrid DoorKey, MiniGrid LavaCrossing, MiniGrid LavaGapS7, PointMaze, and OpenVLA/LIBERO are currently negative or non-promotable.
- FetchPush, FetchSlide, and FetchPickAndPlace object-goal calibrations are
  rejected pre-method routes under the current scripted controller/interface;
  do not build replay comparisons around them without a new preregistered
  calibration that first clears clean-success and recovery-curve prerequisites.
- The next acceptance-moving work must change the learned-policy intervention, use a truly different independent benchmark/reset interface, or materially strengthen theory/presentation. Do not spend more cycles on same-protocol MiniGrid/classic-control screens unless the premise changes. Do not spend more compute on the current OpenVLA-OFT clean-mix/visual-perturbation recipe family; the preregistered weighted perturbation curriculum already failed the official-checkpoint promotion gate.
- The PointMaze U-Maze topology-bottleneck reset-interface screen is completed and negative. Failure-only reaches 0.3500 final RAUC, while BGR reaches 0.0854 and BGR-Coverage reaches 0.0573; do not scale or promote this protocol.
- The MiniGrid-LavaGapS7 external-package screen is completed and negative. BGR-Coverage trails uniform on mean RAUC (0.4277 vs. 0.4461), default BGR is lower (0.4031), and the state-priority/uniform-radius ablation is highest (0.4627); do not scale or promote this protocol.
- The hard-budget FetchReach-v4 reset-interface follow-up is completed and
  negative. BGR-Coverage reaches 0.4625 final RAUC and default BGR reaches
  0.4438, below uniform 0.6813, failure-only 0.8250, and TD-loss 0.9437; do
  not scale or promote this protocol.
- The MiniGrid-FourRooms midband distance-2-to-5 follow-up is completed and
  negative. Default BGR improves mean RAUC over uniform (0.6747 vs. 0.6190)
  but wins only 2/4 paired seeds, trails fixed-radius replay (0.6779) and
  failure-only replay (0.7309), and has lower non-saturated median r80 than
  uniform (0.5627 vs. 0.6451). BGR-Coverage is also negative at 0.5933 RAUC;
  do not scale or promote this protocol.
- The highway-env parking-v0 external-package calibration is completed and
  rejected before method comparison. With `highway-env==1.10.1` in an isolated
  Python 3.11 venv, the fixed scripted parking controller gets clean success
  0.3333, recovery range 0.2500--0.5000, mean crash rate 0.5370, RAUC 0.3750,
  and median r80 9.8000 over 12 seeds. Do not build or scale a highway-env
  parking replay comparison unless a new preregistered controller or policy
  first clears clean-success and non-saturated recovery prerequisites.
- The completed Reacher independent-benchmark route is a pre-method Gymnasium
  MuJoCo Reacher-v5 calibration plus a negative method screen, not BGR
  evidence. It uses package-owned Reacher-v5
  dynamics and target sampling, exact MuJoCo state resets, two-joint angular
  perturbations, and a fixed weak inverse-kinematics/PD controller in
  `/tmp/bgr_pointmaze_venv` (`gymnasium==1.3.0`, `mujoco==3.9.0`). The compact
  artifact `results/reacher_recovery_calibration_12seed_v1/summary.json`
  reports clean success 0.8333, recovery range 0.5000--0.9167, RAUC 0.7891,
  and r80 3.0000 on a 0--4 perturbation grid. The fixed all-method comparison
  was implemented at `tools/reacher_recovery_probe.py` before method results
  and run with
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/reacher_recovery_probe.py --out results/reacher_recovery_probe_12seed_v1`.
  The result is negative: uniform final RAUC is 0.3862, BGR is 0.2907 with
  4/8/0 paired wins/losses/ties against uniform, and BGR-Coverage is 0.2721
  with 4/8/0. Do not scale or promote this Reacher route.
- The completed Gymnasium MuJoCo InvertedPendulum-v5 route uses the existing
  isolated `/tmp/bgr_pointmaze_venv`
  environment (`gymnasium==1.3.0`, `mujoco==3.9.0`, `numpy==2.4.6`), exact
  MuJoCo state resets, pole-angle perturbations, and a fixed PD balance
  controller. The compact pre-method calibration artifact
  `results/inverted_pendulum_recovery_calibration_12seed_v1/summary.json`
  reports clean success 1.0000, recovery range 0.0000--1.0000, RAUC 0.7500,
  and r80 0.2100 on a 0--0.30 perturbation grid. The fixed all-method
  comparison was implemented before method results at
  `tools/inverted_pendulum_recovery_probe.py` and run exactly as:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/inverted_pendulum_recovery_probe.py --out results/inverted_pendulum_recovery_probe_4seed_v1`.
  The result is negative and should not be scaled or promoted: every method
  ties at final RAUC 0.7500 and median r80 0.2100, with BGR 0/0/4 against
  uniform on final RAUC.
- The completed Gymnasium MuJoCo InvertedDoublePendulum-v5 route is also
  non-promotable. The pre-method calibration cleared the reset-interface screen
  with clean success 1.0000, recovery range 0.0000--1.0000, RAUC 0.4259, and
  r80 0.2825, but the fixed 4-seed method screen collapses clean success. BGR
  reaches final RAUC 0.0833 versus uniform 0.0035 with W/L/T 1/0/3, while
  BGR-Coverage is 0.0000 and clean success falls to 0.2500 for BGR and 0.0000
  for BGR-Coverage. Treat this as a retired calibration plus negative method
  audit, not acceptance evidence.

## Reviewer-Critique Priorities

Treat the following as the current paper-weakness backlog:

- No new positive evidence: the positive empirical core remains synthetic/grid/suffix, with the only practically visible effect on a constructed grid-margin benchmark.
- Standard environments are negative: the paper must not imply that transparent reframing solves the acceptance problem.
- Novelty is still incremental relative to PLR, PAIRED-ACCEL, prioritized replay, and regret/TD/loss-priority sampling.
- Author-defined metrics carry most of the positive case; median-r80 and absolute-radius checks must be shown even when they weaken the story.
- Fragility is part of the record: high-learning-rate reversal, BGR-Coverage rescue after suffix undercoverage, and untuned margin choices should be disclosed or fixed by preregistered evidence.
- To materially improve acceptance odds, prioritize independent positive evidence, a learned-policy win under a fixed gate, clearer visual intuition, or a stronger formal result. Do not spend effort merely softening language around unchanged negative evidence.

## Evidence Policy

- Do not promote a new result into `paper/main.tex` unless it was preregistered or fixed before method comparison.
- Paper-positive results should use at least 30 paired seeds unless explicitly recorded as a pre-promotion screen.
- A promotable benchmark must beat uniform, fixed-radius, failure-only, loss/TD-priority, and the state-priority/uniform-radius ablation on final RAUC with a visible mean gap.
- Median r80 and absolute-radius checks must not be saturated or contradict the RAUC claim.
- Negative screens belong in `docs/aaai_acceptance_gap.md` and `results/README.md`; keep them out of the paper unless they are explicit limitations.

## Experiment Workflow

- Before a full method comparison, record the exact command, package versions, promotion gate, and calibration notes in `docs/aaai_acceptance_gap.md` and `results/README.md`.
- Commit and push that preregistration before launching the comparison.
- Use isolated temporary virtualenvs for optional external packages, such as `/tmp/bgr_minigrid_venv` and `/tmp/bgr_pointmaze_venv`. Do not add runtime dependencies unless the result becomes promotable.
- Do not use Docker for this workflow.
- Commit compact artifacts such as `summary.csv` and `package_versions.json`. Leave raw `results.json`, Slurm logs, and scratch directories untracked unless there is a deliberate reason to package them.
- Use the `athena` Slurm workflow and repository scripts for heavy OpenVLA/LIBERO work. Do not rely on the dirty remote checkout being clean; prefer local wrapper scripts, explicit environment variables, and `GIT_PULL=0` where the remote tree is known to be dirty.
- The latest completed learned-policy follow-up is the proximal-anchor
  OpenVLA-OFT adaptation in
  `scripts/queue_openvla_oft_preregistered_proximal_anchor.sh`. It reuses the
  fixed weighted perturbation TFDS roots but changes the optimization objective
  with `PROXIMAL_ANCHOR_L2=1.0`, penalizing deviation of trainable parameters
  from their resumed official-checkpoint values. It is promotable only if BGR
  beats both proximal-anchor matched random and the official checkpoint by the
  fixed +10/400 and +0.02 non-identity perturbation gate while preserving clean
  identity within -1/100. The original adaptation chain was submitted on 2026-06-05
  13:18 PDT / 21:18 BST as BGR train/merge/clean-eval jobs
  `767128`/`767129`/`767130` and random train/merge/clean-eval jobs
  `767131`/`767132`/`767133`. The fixed perturbation evals were submitted with
  BGR dependency `afterok:767129` and random dependency `afterok:767132` as
  official jobs `767134`-`767138`, BGR jobs `767139`-`767143`, and random jobs
  `767144`-`767148`. A remote Athena poll on 2026-06-06 03:05 PDT / 11:05 BST
  showed BGR train job `767128` failed with exit code `1:0`; BGR merge job
  `767129`, random adapt job `767131`, and downstream BGR/random perturb jobs
  were dependency-held with `DependencyNeverSatisfied`. Official identity job
  `767134` completed, official blur job `767135` remained pending on
  unavailable GPU nodes, and other official perturb jobs were dependency-held.
  Remote proximal perturb/adapt summaries were still missing. Log inspection
  showed `767128` failed during `normalized_loss.backward()` with PyTorch DDP
  reporting `Expected to mark a variable ready only once` for
  `base_model.model.vision_backbone.fused_featurizer.attn_pool.mlp.fc2.lora_B.default.weight`.
  The wrapper was repaired in commit `cfecfd9` by logging the proximal metric
  under `torch.no_grad()` and adding the equivalent proximal gradient to
  `param.grad` after the normal DDP backward pass. The repaired execution tag is
  `proxanchor_l2_1em0_ddpgradfix_v1`. Repaired adaptation jobs
  BGR `767657`/`767658`/`767659` and random `767660`/`767661`/`767662`
  completed successfully. Repaired perturb evals were submitted as official
  `767663`-`767667`, BGR `767674`-`767678`, and random `767681`-`767685`;
  BGR/random were submitted after the repaired checkpoints existed because
  Slurm rejected dependencies on already-completed merge jobs. The compact
  local artifacts are
  `results/openvla_oft_goal_adapt_eval_cleanmix_p2048unique_perturbrepeat3_prereg_proxanchor_l2_1em0_ddpgradfix_v1_step50500_lr5em7_identitylora_imageaug_officialtrainstats_v1/summary.csv`
  and
  `results/openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_proxanchor_l2_1em0_ddpgradfix_v1_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv`.
  The gate is negative: non-identity BGR 368/400, official 367/400, and random
  368/400; identity BGR 98/100, official 99/100, and random 98/100. This has
  been incorporated into `paper/main.tex` and
  `paper/figures/openvla_adaptation_table.tex` as negative audit evidence, not
  as a robotics fine-tuning win.
- The prior completed weighted OpenVLA perturbation curriculum is also a
  negative audit: random-shift job `766831` completed successfully with 97/100
  success, producing non-identity totals BGR 367/400, official 367/400, and
  matched random 370/400. The compact local artifact is
  `results/openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv`.

## Paper Workflow

- If `paper/main.tex` changes, rebuild `paper/main.pdf` with pdfTeX on the remote builder, strip PDF metadata with `qpdf`, then run claim and package checks.
- Do not add positive claims without a corresponding source artifact and checker coverage.
- Keep OpenVLA/LIBERO phrased as audits unless BGR beats both matched random and the official checkpoint under a fixed promotion gate.
- Keep local tabular and classic-control probes as internal diagnostics unless
  a preregistered result clears the evidence policy. Do not name exploratory
  Taxi, CliffWalking, MountainCar, CartPole, Acrobot, or Pendulum probes in the
  manuscript as evidence.

## Required Checks

- Compile changed Python tools/scripts with `python3 -m py_compile ...`.
- Run `PYTHONPATH=src:. python3 scripts/check_paper_claims.py --paper paper/main.tex --results-dir results --figures-dir paper/figures` after paper-facing claim changes.
- Run `PYTHONPATH=src:. python3 scripts/check_submission_package.py --root . --write-required-manifest` and then `PYTHONPATH=src:. python3 scripts/check_submission_package.py --root .` before committing package-facing changes.
- Run `git diff --check` before committing.

## Git Hygiene

- Push clean checkpoints to `origin/main` frequently.
- Never revert unrelated or user-created changes.
- Stage explicit files only; avoid sweeping in untracked raw results.
