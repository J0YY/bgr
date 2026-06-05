# AGENTS.md

## Working Objective

The active objective is to move this repository toward a genuinely high-confidence AAAI main-track paper. Do not treat wording-only reframing as success. The core blocker is evidence: a promoted result must beat strong baselines on a fixed or pre-existing benchmark with non-contradictory metrics.

Current target: iterate, queue experiments, and reframe the paper until the work is plausibly 90%+ likely to clear the AAAI main-track bar. Current status is below that bar: the strongest positive evidence is still the controlled grid-margin mechanism result, while standard-environment and learned-policy probes remain negative or non-promotable.

As of 2026-06-05, `PYTHONPATH=src:. python3 scripts/check_acceptance_readiness.py --root .` reports:

- PASS controlled grid mechanism: pooled RAUC 0.4342 vs 0.3965.
- FAIL independent/pre-existing benchmarks: FrozenLake, MiniGrid FourRooms, MiniGrid DoorKey, MiniGrid LavaCrossing, and PointMaze remain non-promotable; MiniGrid-LavaGapS7 is the next preregistered external-package screen.
- FAIL learned-policy OpenVLA/LIBERO: the latest weighted perturbation audit has non-identity success BGR 367/400 and official 367/400, with matched-random 273/300 available rows and random shift job `766831` still pending on unavailable GPU nodes. The official-checkpoint gate is already impossible because the required margin is +10/400 and +0.02 absolute success.
- Decision: `NOT_READY_FOR_90P_AAAI_CLAIM`.

The practical goal is not to make the paper sound accepted. The practical goal is to find or build defensible evidence that survives the acceptance criteria below, then incorporate only those results into `paper/main.tex`.

## Current Acceptance Status

- The internal readiness gate is intentionally failing; do not report the paper as AAAI-ready.
- Controlled grid-margin evidence is positive and mechanistically useful, but it is not enough for a high-confidence main-track acceptance claim.
- FrozenLake, MiniGrid FourRooms, MiniGrid DoorKey, MiniGrid LavaCrossing, PointMaze, and OpenVLA/LIBERO are currently negative or non-promotable.
- The next acceptance-moving work must change the learned-policy intervention, use a truly different independent benchmark/reset interface, or materially strengthen theory/presentation. Do not spend more cycles on same-protocol MiniGrid/classic-control screens unless the premise changes. Do not spend more compute on the current OpenVLA-OFT clean-mix/visual-perturbation recipe family; the preregistered weighted perturbation curriculum already failed the official-checkpoint promotion gate.
- The PointMaze U-Maze topology-bottleneck reset-interface screen is completed and negative. Failure-only reaches 0.3500 final RAUC, while BGR reaches 0.0854 and BGR-Coverage reaches 0.0573; do not scale or promote this protocol.
- The active external-package screen is official MiniGrid-LavaGapS7 via `tools/minigrid_lavacrossing_recovery_probe.py`. It was calibrated only on uniform before method comparison; run the fixed command in `results/README.md` and promote only if BGR or BGR-Coverage beats uniform, fixed-radius, failure-only, TD-loss, and BGR-uniform-radius with non-contradictory median r80 and absolute r10.

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
- The latest learned-policy follow-up is the preregistered weighted OpenVLA perturbation curriculum. It is a negative audit: before the matched-random shift row finished, BGR's completed non-identity total was already 367/400, tied with the official checkpoint's 367/400, so it cannot clear the required +10/400 and +0.02 official-checkpoint margins. Poll job `766831` only for ledger completion; do not treat the final random-shift row as paper-positive evidence.

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
