# AGENTS.md

This file is the project handoff contract for coding agents working in this
repository. Read it before editing code, running experiments, changing paper
claims, or committing results.

## Goal Snapshot

Persistent thread goal: iterate, queue experiments, and reframe the paper until
it is plausibly 90%+ likely to get into AAAI.

Acceptance reality: the project is not there yet. The current evidence supports
a careful mechanism study, not a high-confidence AAAI main-track claim. Make
progress by adding preregistered positive evidence, fixing standard-benchmark or
learned-policy failures, strengthening the theory, or improving required
figures/presentation. Do not treat softer wording around unchanged results as
acceptance progress.

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

As of 2026-06-05, `PYTHONPATH=src:. python3 scripts/check_acceptance_readiness.py --root .` reports:

- PASS controlled grid mechanism: pooled RAUC 0.4342 vs 0.3965.
- FAIL independent/pre-existing benchmarks: FrozenLake, MiniGrid FourRooms, MiniGrid DoorKey, MiniGrid LavaCrossing, MiniGrid LavaGapS7, and PointMaze remain non-promotable.
- FAIL learned-policy OpenVLA/LIBERO: the latest weighted perturbation audit has non-identity success BGR 367/400 and official 367/400, with matched-random 273/300 available rows and random shift job `766831` still pending on unavailable A6000 GPU nodes as of 2026-06-05 16:25 PDT / 2026-06-06 00:25 BST. The official-checkpoint gate is already impossible because the required margin is +10/400 and +0.02 absolute success.
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
- The latest completed learned-policy follow-up is the preregistered weighted OpenVLA perturbation curriculum. It is a negative audit: before the matched-random shift row finished, BGR's completed non-identity total was already 367/400, tied with the official checkpoint's 367/400, so it cannot clear the required +10/400 and +0.02 official-checkpoint margins. Poll job `766831` only for ledger completion; the 2026-06-05 16:25 PDT / 2026-06-06 00:25 BST remote poll still had it pending for unavailable A6000 GPU nodes with a Slurm start estimate of 2026-06-07T14:27:51. The remote `summary.csv` still has only the same 14 completed rows as local `summary_available.csv`. Use `scripts/sync_openvla_oft_weighted_perturb_results.sh --poll --no-check` to re-poll; the helper writes incomplete syncs to `summary_available.csv` and only writes `summary.csv` if the fixed gate rows are complete. Do not treat the final random-shift row as paper-positive evidence.
- The latest preregistered learned-policy route is the proximal-anchor
  OpenVLA-OFT adaptation in
  `scripts/queue_openvla_oft_preregistered_proximal_anchor.sh`. It reuses the
  fixed weighted perturbation TFDS roots but changes the optimization objective
  with `PROXIMAL_ANCHOR_L2=1.0`, penalizing deviation of trainable parameters
  from their resumed official-checkpoint values. It is promotable only if BGR
  beats both proximal-anchor matched random and the official checkpoint by the
  fixed +10/400 and +0.02 non-identity perturbation gate while preserving clean
  identity within -1/100. The adaptation chain was submitted on 2026-06-05
  13:18 PDT / 21:18 BST as BGR train/merge/clean-eval jobs
  `767128`/`767129`/`767130` and random train/merge/clean-eval jobs
  `767131`/`767132`/`767133`. The fixed perturbation evals were submitted with
  BGR dependency `afterok:767129` and random dependency `afterok:767132` as
  official jobs `767134`-`767138`, BGR jobs `767139`-`767143`, and random jobs
  `767144`-`767148`. At submission, all were pending; BGR/random perturb jobs
  were dependency-held, while official perturb jobs serialized by method. A
  remote Athena poll on 2026-06-05 16:25 PDT / 2026-06-06 00:25 BST still showed all jobs
  pending with no `sacct` start/end times: BGR train job `767128` and official
  identity job `767134` were waiting on unavailable A6000 GPU nodes with a
  Slurm start estimate of 2026-06-07 14:27:51 BST, and all other jobs were
  dependency-held. `scontrol show job -dd` confirmed
  `TresPerNode=gres/gpu:a6000:1`; idle g2 nodes are A4000s, so do not resubmit
  this OpenVLA route as a generic/A4000 job unless the memory requirement is
  separately changed and preregistered. Use
  `scripts/sync_openvla_oft_proximal_anchor_results.sh --poll --no-check` to
  re-poll the fixed job IDs and remote summary paths; use `--sync` only after
  the compact `summary.csv` files exist.

## Paper Workflow

- If `paper/main.tex` changes, rebuild `paper/main.pdf` with pdfTeX on the remote builder, strip PDF metadata with `qpdf`, then run claim and package checks.
- Do not add positive claims without a corresponding source artifact and checker coverage.
- Keep OpenVLA/LIBERO phrased as audits unless BGR beats both matched random and the official checkpoint under a fixed promotion gate.

## Required Checks

- Compile changed Python tools/scripts with `python3 -m py_compile ...`.
- Run `PYTHONPATH=src:. python3 scripts/check_paper_claims.py --paper paper/main.tex --results-dir results --figures-dir paper/figures` after paper-facing claim changes.
- Run `PYTHONPATH=src:. python3 scripts/check_submission_package.py --root . --write-required-manifest` and then `PYTHONPATH=src:. python3 scripts/check_submission_package.py --root .` before committing package-facing changes.

## Git Hygiene

- Push clean checkpoints to `origin/main` frequently.
- Never revert unrelated or user-created changes.
- Stage explicit files only; avoid sweeping in untracked raw results.
