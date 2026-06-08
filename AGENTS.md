# AGENTS.md

This file is the project handoff contract for coding agents working in this
repository. Read it before editing code, running experiments, changing paper
claims, or committing results.

## Quick Answer

File status: this is the required project `AGENTS.md`. Keep it current whenever
the goal, acceptance status, experiment route, Docker policy, or push policy
changes.

Current goal: iterate, queue experiments, and reframe the BGR paper until it is
genuinely plausibly 90%+ likely to get into AAAI main track.

Current status: not there yet. The readiness gate still says
`NOT_READY_FOR_90P_AAAI_CLAIM`; the controlled grid mechanism result is the
main mechanism evidence, and OpenML diabetes, blood-transfusion, and phoneme
are now replicated positive pre-existing-dataset margin-replay evidence.
Standard-environment recovery screens and OpenVLA/LIBERO learned-policy evidence
remain failing or non-promotable. The latest bsuite Catch 30-seed scale-up,
MiniGrid FourRooms radius-10 rescue,
HandReach-v3 calibration, highway-fast-v0 lane calibration, and MinAtar
Breakout all-method screen are also negative, so they do not solve the
independent-benchmark evidence gap. MinAtar Asterix also completed negative
after its usable calibration, so there is no active independent route. The
internal sklearn-digits margin replay scout is also rejected before promotion:
its best BGR target gives only 0.8271 vs. 0.8123 RAUC against uniform with a
2/2/0 paired split, while fixed-radius replay is stronger at another target.
The internal sklearn tabular margin replay scout is also rejected: breast
cancer's best BGR target gives 0.9610 vs. 0.9516 RAUC against uniform, and
wine's best gives 0.9702 vs. 0.9563, below the +0.03 pre-registration screen.
The OpenML diabetes margin replay route was the first replicated positive
pre-existing-dataset signal in this thread: the fixed 30-seed follow-up gives
BGR 0.7062 vs. uniform 0.6689 RAUC (W/L/T=24/6/0) and vs. fixed-radius 0.6759,
and the held-out seeds 30--59 replication gives BGR 0.7056 vs. uniform 0.6673
(W/L/T=23/7/0) and vs. fixed-radius 0.6640. Pooled across 60 seeds, BGR is
0.7059 vs. uniform 0.6681 (+0.0378, W/L/T=47/13/0) and vs. fixed-radius
0.6699 (+0.0359, W/L/T=43/17/0). A fixed external numeric OpenML suite then
found a second replicated positive route on blood-transfusion-service-center:
the original fixed target-2.0 30-seed row gives BGR 0.7625 vs. uniform 0.6657
(W/L/T=30/0/0) and vs. fixed-radius 0.6920, and the held-out seeds 30--59
replication gives BGR 0.7595 vs. uniform 0.6846 (W/L/T=25/5/0) and vs.
fixed-radius 0.7133. The full external numeric OpenML suite also has a
held-out seeds 30--59 repeat at the same target radius: original/held-out macro
means are BGR 0.8055/0.8068, uniform 0.7939/0.7943, and fixed-radius
0.7864/0.7839; pooled across both suite runs, BGR is ahead on 6/8 dataset means
versus uniform and 7/8 versus fixed-radius. Pooled across 60
blood-transfusion seeds, BGR is 0.7610 vs. uniform 0.6751 (+0.0858,
W/L/T=55/5/0) and vs. fixed-radius 0.7026 (+0.0584, W/L/T=52/6/2). A held-out
phoneme replication has also completed:
the original fixed target-2.0 30-seed row gives BGR 0.7228 vs. uniform 0.6896
(W/L/T=21/9/0) and vs. fixed-radius 0.6704 (W/L/T=23/7/0); the held-out
seeds 30--59 replication gives BGR 0.7124 vs. uniform 0.6758 (W/L/T=21/9/0)
and vs. fixed-radius 0.6792 (W/L/T=25/5/0). Pooled across 60 phoneme seeds,
BGR is 0.7176 vs. uniform 0.6827 (+0.0349, W/L/T=42/18/0) and vs.
fixed-radius 0.6748 (+0.0428, W/L/T=48/12/0). This is acceptance-moving
supervised margin-replay evidence, but it still does not solve the
learned-policy or standard-environment failures by itself.
The new grid-margin witness-sensitivity diagnostic improves scope support for the
feasibility-witness assumption but is controlled mechanism evidence, not an
independent-benchmark win.

Immediate priority: fix the weaknesses called out by the review, especially the
lack of new independent positive evidence, the negative standard-environment
record, incremental novelty, author-defined metric dependence, and robustness
fragility. Do not just soften language around unchanged results.

Active learned-policy route: preregistered OpenVLA-OFT occlusion-bottleneck
adaptation in `scripts/queue_openvla_oft_preregistered_occlusion_bottleneck.sh`.
This route is motivated by the fixed full-goal visual audit: blur,
brightness, and shift are near saturated, while occlusion is the only clear
non-identity bottleneck. The prep builds matched BGR-boundary and random
clean-plus-occlusion TFDS roots with `OCCLUSION_CAP=512`,
`OCCLUSION_REPEAT=4`, official stats, identity-LoRA, image augmentation,
`PROXIMAL_ANCHOR_L2=5.0`, `LR=2e-7`, and `ADAPT_STEPS=400`. It is not a paper
claim unless the fixed gate passes: BGR must beat both matched random and the
official checkpoint by at least 10/400 non-identity episodes and at least 0.02
absolute success rate while not trailing clean identity by more than 1/100.
Submitted on `athena` in the live `/work/joy` workspace after this route was
implemented: prep job `767850`; BGR train/merge/clean-eval jobs
`767851`/`767852`/`767853`; matched-random train/merge/clean-eval jobs
`767854`/`767855`/`767856`; official perturb-eval jobs `767857`--`767861`;
BGR perturb-eval jobs `767862`--`767866`; matched-random perturb-eval jobs
`767868`/`767878`--`767881`. The submission used
`REMOTE_PROJECT=/work/joy/bgr`, `REMOTE_RUN_ROOT=/work/joy/bgr/runs`,
`REMOTE_HF_HOME=/work/joy/cache_home/huggingface`,
`OPENVLA_OFT_ROOT=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft`,
`LIBERO_ROOT=/work/joy/external_validation/openvla_oft_smoke_746850/LIBERO`,
`SOURCE_ARTIFACT_ROOT=/work/joy/dreamaudit_jobs/artifacts`, and `GIT_PULL=0`.
Initial `squeue` showed prep `767850` and official identity `767857` running,
with adaptation and BGR/random perturb jobs dependency-pending. Do not
incorporate this route into `paper/main.tex` until compact summaries exist and
the fixed gate passes.
Poll/sync helper:
`scripts/sync_openvla_oft_occlusion_bottleneck_results.sh --poll --no-check`
and, when summaries or logs exist,
`scripts/sync_openvla_oft_occlusion_bottleneck_results.sh --sync`. The first
poll at 2026-06-08 05:39:10 BST showed prep `767850` running on `c1-g4-02`,
official identity `767857` running on `c1-g4-04`, BGR/random adaptation jobs
pending on `afterok:767850` (with random also serialized behind BGR train),
BGR perturb jobs pending on `afterok:767852`, random perturb jobs pending on
`afterok:767855`, and both expected compact summaries still missing.
Latest poll at 2026-06-08 05:42:04 BST showed the same substantive state:
prep `767850` and official identity `767857` still running, all adaptation and
BGR/random perturb jobs still dependency-pending, and the compact summaries
still missing. The sync helper now treats existing-but-incomplete remote logs
as `[pending]` instead of failing the whole poll/sync command.
Latest poll at 2026-06-08 05:43:31 BST still showed prep `767850` running on
`c1-g4-02` and official identity `767857` running on `c1-g4-04`; all
adaptation jobs and BGR/random perturb jobs were still dependency-pending.
Both compact summary paths were missing, the perturb log directory existed but
was not summarizable yet, and the readiness decision remained
`NOT_READY_FOR_90P_AAAI_CLAIM`.
Latest poll at 2026-06-08 05:47:09 BST still showed prep `767850` running on
`c1-g4-02` and official identity `767857` running on `c1-g4-04`; all
adaptation and BGR/random perturb jobs were still dependency-pending. Remote
log tails were healthy: prep had advanced to matched-random perturb rendering,
and official identity eval had reached at least 64/100 episodes with 63
successes. No compact summaries were available, so no paper claim should be
changed.
Latest poll at 2026-06-08 05:50:33 BST showed official identity job `767857`
completed successfully with 99/100 clean successes, official blur job `767858`
running, and prep job `767850` still running. BGR/random adaptation and all
BGR/random perturb jobs remained dependency-pending. The sync produced only a
partial `summary_available.csv` row for official/identity; the full perturb
summary and adapt summary were still missing, and the promotion gate remained
incomplete rather than positive evidence.
Latest poll at 2026-06-08 05:52:55 BST showed prep `767850` still running on
`c1-g4-02`, official identity `767857` completed at 99/100, and official blur
`767858` running on `c1-g4-04`. BGR/random train, merge, clean-eval, and
perturb-eval jobs remained dependency-pending. Remote log tails were healthy:
prep had finished the BGR clean-plus-occlusion TFDS root and moved to
matched-random TFDS generation; official blur had reached roughly episode 28/29
with about 96% success so far. Only the partial official/identity
`summary_available.csv` exists locally; full perturb and adapt summaries are
still missing, so the route remains incomplete.

Latest paper checkpoint: OpenML diabetes, blood-transfusion, phoneme, and the
mixed full external OpenML suite repeat are now incorporated as replicated
pre-existing supervised margin-replay evidence. This improves the "no new
positive evidence" and cherry-picking weaknesses but does not change readiness
because the learned-policy gate still fails.
Latest poll at 2026-06-08 06:13:20 BST showed the occlusion route past data
prep and training: prep `767850`, BGR train/merge `767851`/`767852`, and random
train/merge `767854`/`767855` completed with exit `0:0`. BGR clean eval
`767853`, random clean eval `767856`, official brightness `767859`, BGR
identity perturb eval `767862`, and random identity perturb eval `767868` were
running. Official identity `767857` and blur `767858` completed; official
occlusion/shift, BGR blur/brightness/occlusion/shift, and random
blur/brightness/occlusion/shift remained dependency-pending. The sync wrote an
incomplete `summary_available.csv` from logs, but the full perturb and adapt
summaries were still missing; the promotion gate remained `[INCOMPLETE]`.
This is not paper evidence yet.
Latest poll at 2026-06-08 06:17:16 BST showed BGR clean eval `767853`
completed, official brightness `767859` completed, and BGR identity perturb
eval `767862` completed. Random clean eval `767856`, random identity perturb
eval `767868`, official occlusion `767860`, and BGR blur `767863` were running.
Official shift `767861`, BGR brightness/occlusion/shift `767864`--`767866`,
and random blur/brightness/occlusion/shift `767878`--`767881` remained
dependency-pending. The full perturb/adapt summaries were still missing, and
the fixed OpenVLA perturb promotion gate remained `[INCOMPLETE]` with missing
non-identity BGR/random rows and official occlusion/shift rows. This still
cannot be incorporated into `paper/main.tex`.
Latest poll at 2026-06-08 06:18:48 BST showed random clean eval `767856`
completed; the synced adapt summary has BGR clean `99/100` and random clean
`98/100`. The partial perturb summary still had only BGR identity `99/100`,
official identity `99/100`, official blur `97/100`, and official brightness
`98/100`. Official occlusion `767860`, BGR blur `767863`, and random identity
`767868` were still running or not summarizable, with downstream BGR/random
non-identity perturb jobs pending.
Latest poll at 2026-06-08 06:22:39 BST still left the fixed perturb gate
`[INCOMPLETE]`: BGR blur `767863` was running, random identity `767868` was
running, official occlusion `767860` appeared pending with `BeginTime`, official
shift `767861` was pending, and all BGR/random brightness/occlusion/shift plus
random blur jobs were still dependency-pending. The helper regenerated the
adapt summary and incomplete perturb `summary_available.csv`, but the full
perturb summary was missing. Do not incorporate this route into the paper until
`summary.csv` exists and `scripts/check_openvla_perturb_gate.py` passes the
fixed +10/400 and +0.02 gate.
Latest poll at 2026-06-08 06:24:35 BST had the same fixed-gate status:
BGR blur `767863` and random identity `767868` were running, official
occlusion `767860` was still pending on `BeginTime`, official shift `767861`
was pending, and downstream BGR/random non-identity perturb jobs were
dependency-pending. The perturb gate was still `[INCOMPLETE]`.
Latest poll at 2026-06-08 06:28:56 BST showed BGR blur `767863` completed at
`0:0`, BGR brightness `767864` running, official occlusion `767860` running,
and random identity `767868` still running. The partial perturb summary now has
BGR identity `99/100`, BGR blur `98/100`, official identity `99/100`, official
blur `97/100`, and official brightness `98/100`. Missing rows still include
BGR brightness/occlusion/shift, official occlusion/shift, and all random
non-identity perturbations, so the fixed perturb gate remains `[INCOMPLETE]`.
No paper claim should change until the full `summary.csv` exists and the gate
script returns a positive decision.
Latest poll at 2026-06-08 06:35:44 BST showed random identity `767868`
completed after 21:50, random blur `767878` running, BGR brightness `767864`
running, and official occlusion `767860` running. BGR occlusion/shift
`767865`/`767866`, official shift `767861`, and random
brightness/occlusion/shift `767879`--`767881` remained dependency-pending. The
partial perturb summary now has BGR identity `99/100`, BGR blur `98/100`,
official identity `99/100`, official blur `97/100`, official brightness
`98/100`, and random identity `98/100`. The fixed perturb gate remains
`[INCOMPLETE]`; missing rows still include BGR brightness/occlusion/shift,
official occlusion/shift, and all random non-identity perturbations.
Latest poll at 2026-06-08 06:38:14 BST showed no promotable change yet. BGR
brightness `767864`, official occlusion `767860`, and random blur `767878`
were running; BGR occlusion/shift `767865`/`767866`, official shift `767861`,
and random brightness/occlusion/shift `767879`--`767881` were still
dependency-pending. The adapt summary remains BGR clean `99/100` and matched
random clean `98/100`; the partial perturb summary remains BGR identity
`99/100`, BGR blur `98/100`, official identity `99/100`, official blur
`97/100`, official brightness `98/100`, and random identity `98/100`. The
fixed perturb gate is still `[INCOMPLETE]`, and readiness remains
`NOT_READY_FOR_90P_AAAI_CLAIM`.
Latest poll at 2026-06-08 06:39:41 BST showed a scheduler delay, not new
evidence: BGR brightness `767864`, official occlusion `767860`, and random
blur `767878` moved to pending with `BeginTime`, and the remaining official,
BGR, and random non-identity perturb jobs were still dependency-pending. A
direct `squeue` check at 2026-06-08 06:41:57 BST showed the pending reason
changed to `Priority`; estimated starts were official occlusion `767860` at
2026-06-09 00:28:08 BST and BGR brightness `767864` plus random blur `767878`
at 2026-06-09 18:31:42 BST. The synced partial summaries are unchanged, the
fixed perturb gate is still `[INCOMPLETE]`, and no `paper/main.tex` claim
should change.
Latest poll at 2026-06-08 07:14:45 BST showed the route still incomplete:
BGR brightness `767864`, random blur `767878`, and official occlusion `767860`
were pending on `Priority`; official shift `767861`, BGR occlusion/shift
`767865`/`767866`, and random brightness/occlusion/shift `767879`--`767881`
were dependency-pending. Completed rows available locally are BGR identity
`99/100`, BGR blur `98/100`, official identity `99/100`, official blur
`97/100`, official brightness `98/100`, and random identity `98/100`; adapt
clean is BGR `99/100` and matched random `98/100`. The full perturb summary is
still missing, so the fixed perturb gate remains `[INCOMPLETE]`.

Operational defaults:

- Keep committing and pushing clean checkpoints to `origin/main`.
- Do not use Docker for this project.
- Why no Docker: the submission/package gate forbids Docker artifacts, and the
  established workflows use local temporary virtualenvs plus `athena` Slurm for
  heavyweight LaTeX/OpenVLA/LIBERO work.
- Use local or temporary virtualenvs for external packages and `athena` for
  remote LaTeX/OpenVLA/LIBERO workflows.
- Stage explicit paths only; many `results/` files are intentionally untracked.
- Incorporate new results into `paper/main.tex` only after artifacts and checks
  justify the claim; negative route closures may appear only as scoped
  limitations with checker coverage.
- When the user asks "what is our goal?", answer plainly: get the BGR paper to
  a genuinely high-confidence AAAI-main case by fixing the review weaknesses
  with evidence, not by hiding negative results or rephrasing unchanged tables.

## Goal Snapshot

Last verified: 2026-06-08.

Persistent thread goal: iterate, queue experiments, and reframe the paper until
it is plausibly 90%+ likely to get into AAAI.

One-line goal: turn BGR from a careful mechanism study with mostly tailored
positive evidence into a main-track-defensible paper backed by independent or
learned-policy evidence that survives fixed gates.

Current answer to "what is our goal?": make the BGR paper genuinely
main-track-defensible, not merely better phrased. The work is only successful
when the evidence gates support a high-confidence AAAI submission: independent
or pre-existing benchmark evidence, a learned-policy win under a fixed gate, a
stronger formal result, or presentation/theory improvements that materially
change reviewer risk.

Current acceptance answer: not ready. The internal readiness check still reports
`NOT_READY_FOR_90P_AAAI_CLAIM`; the controlled grid result is positive and the
OpenML diabetes, blood-transfusion, and phoneme pre-existing-dataset
follow-up/replication evidence now clears the independent/pre-existing gate,
but OpenVLA/LIBERO learned-policy evidence is still failing and the
standard-environment recovery record is still negative.
The latest FourRooms radius-10, HandReach-v3, and highway-fast-v0 follow-ups
close simple rescue paths, but they are negative limitations. A 30-seed grid-margin witness
sensitivity diagnostic now quantifies one reviewer-facing interface assumption:
exact-witness accepted samples are valid at 1.0000, while symmetric 10%/20%
witness noise lowers true-boundary recall to 0.9001/0.7980.

Acceptance reality: the project is not there yet. The current evidence supports
a careful mechanism study, not a high-confidence AAAI main-track claim. Make
progress by adding preregistered positive evidence, fixing standard-benchmark or
learned-policy failures, strengthening the theory, or improving required
figures/presentation. Do not treat softer wording around unchanged results as
acceptance progress.
As of the latest checkpoint, MinAtar Breakout is retired as a negative
independent route: the 12-seed calibration was usable, but the fixed 4-seed
all-method screen ties uniform on final RAUC and saturates median r80. MinAtar
Asterix is also retired negative: the fixed 12-seed calibration cleared
clean/non-flat/non-saturated prerequisites, but the fixed 4-seed all-method
screen loses to failure-only replay.

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

Current target: iterate, queue experiments, and reframe the paper until the work is plausibly 90%+ likely to clear the AAAI main-track bar. Current status is below that bar: the strongest positive evidence now includes the controlled grid-margin mechanism result and replicated OpenML diabetes, blood-transfusion, and phoneme margin replay, while standard-environment and learned-policy probes remain negative or non-promotable.

As of 2026-06-07, `PYTHONPATH=src:. python3 scripts/check_acceptance_readiness.py --root .` reports:

- PASS controlled grid mechanism: pooled RAUC 0.4342 vs 0.3965.
- PASS independent/pre-existing benchmark: OpenML diabetes, blood-transfusion, and phoneme margin replay are positive and replicated. Pooled BGR-vs-uniform RAUC gaps are +0.0378 for diabetes, +0.0858 for blood-transfusion, and +0.0349 for phoneme; each also beats fixed-radius replay in the pooled comparison. Standard-environment recovery screens such as FrozenLake, MiniGrid, PointMaze, FetchReach, Reacher, bsuite, MinAtar, and LunarLander remain non-promotable or negative.
- FAIL learned-policy OpenVLA/LIBERO: the latest completed perturb-only anchored audit has non-identity success BGR 371/400, official 367/400, and matched random 372/400, with identity BGR 99/100, official 99/100, and random 99/100. BGR trails matched random by 1/400 and beats official by only 4/400, so it fails the fixed +10/400 and +0.02 promotion gate. Earlier weighted and proximal-anchor audits are also negative.
- Decision: `NOT_READY_FOR_90P_AAAI_CLAIM`.

The practical goal is not to make the paper sound accepted. The practical goal is to find or build defensible evidence that survives the acceptance criteria below, then incorporate only those results into `paper/main.tex`.
Use `PYTHONPATH=src:. python3 scripts/acceptance_scorecard.py --root . --out docs/acceptance_scorecard.md` to quantify current distances to the internal gates before choosing the next experiment.

## Current Acceptance Status

- The internal readiness gate is intentionally failing; do not report the paper as AAAI-ready.
- Controlled grid-margin evidence is positive and mechanistically useful, but it is not enough for a high-confidence main-track acceptance claim.
- FrozenLake, MiniGrid FourRooms, MiniGrid DoorKey, MiniGrid LavaCrossing, MiniGrid LavaGapS7, PointMaze, and OpenVLA/LIBERO are currently negative or non-promotable.
- MinAtar Breakout is completed negative scope evidence, not positive BGR
  evidence. The
  fixed pre-method calibration command is
  `PYTHONPATH=src:. /tmp/bgr_minatar_venv/bin/python tools/minatar_breakout_recovery_calibration.py --out results/minatar_breakout_recovery_calibration_12seed_v1`.
  It uses `MinAtar==1.0.15` and `numpy==2.4.6` in `/tmp/bgr_minatar_venv`,
  MinAtar's package-owned Breakout dynamics, a fixed paddle-tracking
  controller, and signed paddle-cell offsets after a burn-in checkpoint. The
  calibration clears the pre-method gate with clean success 1.0000, recovery
  range 0.6667--1.0000, RAUC 0.7000, and r80 0.6000. This only permits a
  fixed all-method screen; the completed screen at
  `results/minatar_breakout_recovery_probe_4seed_v1/summary.csv` is negative:
  BGR and BGR-Coverage both tie uniform at 0.8896 final RAUC with median r80
  saturated at 5.0000, and failure-only has the best AULC at 0.7721.
- MinAtar Asterix is completed negative scope evidence, not positive BGR
  evidence. The fixed calibration command is
  `PYTHONPATH=src:. /tmp/bgr_minatar_venv/bin/python tools/minatar_asterix_recovery_calibration.py --out results/minatar_asterix_recovery_calibration_12seed_v1`.
  It uses `MinAtar==1.0.15` and `numpy==2.4.6` in `/tmp/bgr_minatar_venv`,
  MinAtar's package-owned Asterix dynamics, a fixed gold-seeking controller
  with one-step enemy avoidance, and seed-fixed random player-cell
  displacements after a 30-step burn-in. The calibration clears the pre-method
  gate with clean success 0.8333, recovery range 0.5000--0.8333, RAUC 0.7188,
  and r80 5.3333. The fixed all-method screen at
  `results/minatar_asterix_recovery_probe_4seed_v1/summary.csv` is negative:
  failure-only reaches 0.8625 final RAUC, above BGR-Coverage 0.8406, BGR
  0.8047, and uniform 0.8234; BGR-Coverage wins only 1/4 paired seeds against
  uniform. Do not scale or promote this route without a genuinely new
  preregistered premise.
- FetchPush, FetchSlide, and FetchPickAndPlace object-goal calibrations are
  rejected pre-method routes under the current scripted controller/interface;
  do not build replay comparisons around them without a new preregistered
  calibration that first clears clean-success and recovery-curve prerequisites.
  A 2026-06-07 opt-in `scripted_push_far` controller scout for FetchPush
  improves the compact calibration but still fails the clean-success gate:
  `results/fetchpush_object_goal_calibration_far_push_2seed_v1/summary.json`
  reports clean 0.6250, recovery range 0.6250--0.8750, RAUC 0.8125, and
  r80 0.1200 under `gymnasium==1.3.0`, `gymnasium_robotics==1.4.2`, and
  `mujoco==3.9.0`. Treat it as a rejected calibration, not an active route.
- Gymnasium-Robotics HandReach-v3 is also rejected as a pre-method route under
  the fixed random-shooting ShadowHand controller in `/tmp/bgr_pointmaze_venv`.
  `results/handreach_recovery_calibration_8seed_v1/summary.json` reports clean
  success 0.0000, recovery range 0.0000--0.0000, RAUC 0.0000, and r80 0.2000
  under `gymnasium==1.3.0`, `gymnasium_robotics==1.4.2`, and `mujoco==3.9.0`.
  Do not build a HandReach replay comparison unless a new preregistered
  controller first clears the clean-success and non-flat recovery prerequisites.
- MiniGrid FourRooms has no remaining same-protocol radius-window rescue. The
  max-radius-10 follow-up at
  `results/minigrid_fourrooms_recovery_probe_maxr10_4seed_v1/summary.csv`
  gives BGR-Coverage 0.1031 final RAUC vs. uniform 0.1014 (W/L/T=2/2/0), only
  0.0064 above BGR-uniform-radius, with median r80 still saturated at 1.0000
  for both BGR-Coverage and uniform. Do not scale or promote this protocol.
- The sklearn digits margin replay scout is completed and rejected before
  paper promotion. The fixed command is
  `PYTHONPATH=src:. python3 tools/sklearn_digits_margin_scout.py --out results/sklearn_digits_margin_scout_v0`.
  It uses `sklearn.datasets.load_digits`, label-preserving fixed-L2 pixel
  perturbations, and an online `SGDClassifier` to compare uniform,
  fixed-radius, and BGR replay across target radii 0.5--2.0. Every target row
  is marked `reject-scout`; the best BGR row is target 1.0 with 0.8271 RAUC
  versus 0.8123 uniform and only W/L/T=2/2/0, while fixed-radius replay reaches
  0.8425 at target 0.8. Do not add this to the manuscript or scale it without a
  genuinely new preregistered premise.
- The sklearn tabular margin replay scout is also completed and rejected before
  paper promotion. The fixed command is
  `PYTHONPATH=src:. python3 tools/sklearn_tabular_margin_scout.py --out results/sklearn_tabular_margin_scout_v0`.
  It uses `sklearn.datasets.load_breast_cancer` and `load_wine`,
  label-preserving fixed-L2 standardized-feature perturbations, and an online
  `SGDClassifier` to compare uniform, fixed-radius, and BGR replay across
  target radii 0.5--2.0. Every target row is marked `reject-scout`; breast
  cancer's best BGR row is target 2.0 with 0.9610 RAUC versus 0.9516 uniform
  and W/L/T=3/1/0, while wine's best BGR row is target 0.5 with 0.9702 RAUC
  versus 0.9563 uniform and W/L/T=4/0/0. Both gaps are below the +0.03
  pre-registration threshold. Do not add this to the manuscript or scale it
  without a genuinely new preregistered premise.
- The OpenML margin replay scout is completed and opened one candidate route.
  The fixed command is
  `PYTHONPATH=src:. python3 tools/openml_margin_scout.py --out results/openml_margin_scout_v0`.
  It uses OpenML version-1 datasets ionosphere, sonar, diabetes, and spambase
  through `sklearn.datasets.fetch_openml`, median imputation, label encoding,
  standardized feature-space fixed-L2 perturbations, and an online
  `SGDClassifier`. OpenML diabetes target radius 2.0 clears the 4-seed scout
  screen with BGR 0.7402 final RAUC versus uniform 0.6797 and W/L/T=4/0/0;
  fixed-radius replay is 0.6999 at the same target. Ionosphere, sonar, and
  spambase are rejected or near-miss scouts. The fixed preregistered 30-seed
  command was:
  `PYTHONPATH=src:. python3 tools/openml_margin_scout.py --datasets diabetes --targets 2.0 --seeds 30 --out results/openml_diabetes_margin_30seed_v1`.
  It stayed positive: BGR 0.7062 vs. uniform 0.6689 (+0.0373, W/L/T=24/6/0)
  and vs. fixed-radius 0.6759 (+0.0303, W/L/T=19/11/0). The held-out
  replication command was:
  `PYTHONPATH=src:. python3 tools/openml_margin_scout.py --datasets diabetes --targets 2.0 --seed-start 30 --seeds 30 --out results/openml_diabetes_margin_replication_30seed_v1`.
  It also stayed positive: BGR 0.7056 vs. uniform 0.6673 (+0.0383,
  W/L/T=23/7/0) and vs. fixed-radius 0.6640 (+0.0416, W/L/T=24/6/0). A fixed
  external numeric OpenML suite at the same target radius 2.0 is mixed overall
  but stable under a full held-out repeat: the command
  `PYTHONPATH=src:. python3 tools/openml_margin_scout.py --external-validation-suite --targets 2.0 --seeds 30 --out results/openml_numeric_external_fixed_target2_30seed_v1`
  and held-out repeat
  `PYTHONPATH=src:. python3 tools/openml_margin_scout.py --external-validation-suite --targets 2.0 --seed-start 30 --seeds 30 --out results/openml_numeric_external_fixed_target2_replication_30seed_v1`
  give original/held-out macro means BGR 0.8055/0.8068, uniform
  0.7939/0.7943, and fixed-radius 0.7864/0.7839; pooled across both suite runs,
  BGR is ahead on 6/8 dataset means versus uniform and 7/8 versus fixed-radius.
  Blood-transfusion gives BGR 0.7625 vs. uniform 0.6657 (+0.0968,
  W/L/T=30/0/0) and vs. fixed-radius 0.6920 (+0.0705, W/L/T=27/1/2). The
  held-out command
  `PYTHONPATH=src:. python3 tools/openml_margin_scout.py --datasets blood-transfusion-service-center --targets 2.0 --seed-start 30 --seeds 30 --out results/openml_blood_transfusion_margin_replication_30seed_v1`
  gives BGR 0.7595 vs. uniform 0.6846 (+0.0749, W/L/T=25/5/0) and vs.
  fixed-radius 0.7133 (+0.0462, W/L/T=25/5/0). The held-out phoneme
  replication command was:
  `PYTHONPATH=src:. python3 tools/openml_margin_scout.py --datasets phoneme --targets 2.0 --seed-start 30 --seeds 30 --out results/openml_phoneme_margin_replication_30seed_v1`.
  It also stayed positive: BGR 0.7124 vs. uniform 0.6758 (+0.0366,
  W/L/T=21/9/0) and vs. fixed-radius 0.6792 (+0.0332, W/L/T=25/5/0), after
  the original fixed external suite row gave BGR 0.7228 vs. uniform 0.6896 and
  vs. fixed-radius 0.6704. This is incorporated only with careful framing as
  pre-existing supervised margin-replay evidence, not robotics evidence.
- The next acceptance-moving work must change the learned-policy intervention, use a truly different independent benchmark/reset interface, or materially strengthen theory/presentation. Do not spend more cycles on same-protocol MiniGrid/classic-control screens unless the premise changes. Do not spend more compute on the current OpenVLA-OFT clean-mix/visual-perturbation/perturb-only recipe family; the preregistered weighted, proximal-anchor, and perturb-only anchored audits all failed the learned-policy promotion gate.
- The grid-margin witness-sensitivity diagnostic is completed and paper-facing
  only as scope evidence for the feasibility-witness assumption. The fixed
  command is:
  `PYTHONPATH=src:. python3 tools/grid_margin_witness_sensitivity.py --config configs/grid_margin_full_30seed.yaml --out results/grid_margin_witness_sensitivity_30seed_v1`.
  `results/grid_margin_witness_sensitivity_30seed_v1/summary.csv` reports
  exact valid accepted samples at 1.0000; symmetric 10%/20% witness noise keeps
  valid-accept rates at 1.0000/0.9999 but lowers true-boundary recall to
  0.9001/0.7980. Treat this as controlled witness-scope support, not a broad
  robustness or independent-benchmark result.
- The latest completed independent route is Gymnasium Box2D `LunarLander-v3`
  in `/tmp/bgr_lunar_venv` with
  `gymnasium==1.3.0`, `box2d==2.3.10`, `pygame-ce==2.5.7`, `swig==4.4.1`,
  and `numpy==2.4.6`. The fixed calibration command is
  `PYTHONPATH=src:. /tmp/bgr_lunar_venv/bin/python tools/lunarlander_recovery_calibration.py --out results/lunarlander_recovery_calibration_12seed_v1`.
  This is not BGR evidence. The fixed 12-seed calibration cleared the
  pre-method gate with clean success 0.9167, recovery range 0.5833--0.9167,
  RAUC 0.7722, and median r80 0.5300. The fixed all-method screen in
  `tools/lunarlander_recovery_probe.py` is completed and negative under the
  preregistered gate: BGR-Coverage has the best mean final RAUC (0.7500 vs.
  0.6222 uniform, 0.7375 fixed, 0.6799 failure-only, 0.7174 TD-loss, and
  0.7160 BGR-uniform-radius), but it wins only 2/4 paired seeds against
  uniform and has lower median r80 than uniform (0.4200 vs. 0.4825). Do not
  scale or promote this LunarLander route without a genuinely new
  preregistered premise.
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
- The highway-env highway-fast-v0 lane-driving calibration is also rejected
  before method comparison. With `highway-env==1.10.1` in the isolated Python
  3.11 venv, the fixed idle lane-keep policy gets clean success 0.6667,
  recovery range 0.5833--0.6667, mean crash rate 0.3810, RAUC 0.6181, and
  saturated r80 6.0000 over 12 seeds. Do not build or scale a highway-env lane
  replay comparison around this controller/interface.
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
- The completed bsuite DeepSea route is negative. It uses `bsuite==0.3.6` in
  `/tmp/bgr_bsuite_venv`, package-owned randomized DeepSea action mappings,
  exact restart states, and fixed left-column perturbations. The fixed 4-seed
  screen in `results/bsuite_deepsea_recovery_probe_4seed_v1/summary.csv` gives
  default BGR 0.1125 final RAUC vs. uniform 0.0844, but BGR wins only 2/4
  paired seeds, trails the state-priority/uniform-radius ablation at 0.1266,
  and has lower median r80 than uniform (0.3625 vs. 0.5750). BGR-Coverage ties
  uniform on mean RAUC. Do not scale or promote this protocol.
- The completed bsuite Catch route is negative after scale-up. It uses
  `bsuite==0.3.6` in `/tmp/bgr_bsuite_venv`, package-owned Catch dynamics,
  exact restart fields, and fixed paddle-column perturbations. The compact
  4-seed artifact `results/bsuite_catch_recovery_probe_4seed_v1/summary.csv`
  initially cleared the fixed scale-up gate: default BGR 0.9742 final RAUC vs.
  uniform 0.8388 (+0.1354, 4/0/0), fixed-radius 0.7767, failure-only 0.9336,
  TD-loss 0.7140, and BGR-uniform-radius 0.8982, with non-contradictory median
  r80. The fixed 30-seed run in
  `results/bsuite_catch_recovery_probe_30seed_v1/summary.csv` did not survive:
  default BGR reaches 0.8446 final RAUC versus uniform 0.8782 (14/16/0),
  BGR-Coverage reaches 0.8452 versus uniform 0.8782 (13/17/0), failure-only is
  0.9676, and BGR-uniform-radius is 0.8588. Median r80 also contradicts the
  boundary-improvement story: default BGR 0.8367 and BGR-Coverage 0.8608 trail
  uniform 0.9233. Do not scale or promote this Catch route without a genuinely
  new preregistered premise.
- The completed bsuite MountainCar route is negative. It uses `bsuite==0.3.6`
  in `/tmp/bgr_bsuite_venv`, instantiates bsuite's package-owned MountainCar
  task, and steps exact private restart fields during recovery rollouts. Replay
  states are right-moving MountainCar states; larger perturbation radii move
  starts back toward the low-energy valley anchor. The fixed 4-seed command was
  `PYTHONPATH=src:. /tmp/bgr_bsuite_venv/bin/python tools/bsuite_mountaincar_recovery_probe.py --out results/bsuite_mountaincar_recovery_probe_4seed_v1`.
  Default BGR reaches 0.0532 final RAUC versus 0.0497 for uniform (+0.0036,
  3/1/0), below the preregistered +0.01 threshold; BGR-Coverage reaches 0.0553,
  below fixed-radius replay (0.1420), failure-only replay (0.0653), and
  BGR-uniform-radius (0.0558). Median r80 is saturated at 1.0000 for the
  BGR-family methods and uniform. Do not scale or promote this route without a
  genuinely new preregistered premise.
- The completed bsuite Cartpole route is negative. It uses `bsuite==0.3.6` in
  `/tmp/bgr_bsuite_venv`, package-owned three-action Cartpole dynamics, exact
  `CartpoleState` restarts, and the package `step_cartpole` function. Replay
  states are near-upright cart-pole states with nonzero cart, pole-angle, and
  velocity errors; larger radii perturb bounded cart-position, cart-velocity,
  pole-angle, and pole-velocity terms while preserving bsuite feasibility
  limits. The fixed 4-seed command was
  `PYTHONPATH=src:. /tmp/bgr_bsuite_venv/bin/python tools/bsuite_cartpole_recovery_probe.py --out results/bsuite_cartpole_recovery_probe_4seed_v1`.
  Default BGR reaches 0.7464 final RAUC versus 0.7577 for uniform (-0.0113,
  0/4/0), while BGR-Coverage reaches 0.7559 versus uniform (-0.0018, 2/2/0).
  Both trail TD-loss replay (0.7694) and fixed-radius replay (0.7604); default
  BGR also trails BGR-uniform-radius (0.7551). Median r80 is lower for default
  BGR than uniform (0.9667 vs. 0.9875), while BGR-Coverage is 0.9917 in a
  near-ceiling regime. Do not scale or promote this route without a genuinely
  new preregistered premise.

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
- A completed learned-policy follow-up is the proximal-anchor
  OpenVLA-OFT adaptation in
  `scripts/queue_openvla_oft_preregistered_proximal_anchor.sh`. It reuses the
  fixed weighted perturbation TFDS roots but changes the optimization objective
  with `PROXIMAL_ANCHOR_L2=1.0`, penalizing deviation of trainable parameters
  from their resumed official-checkpoint values. Its fixed promotion gate
  required BGR to beat both proximal-anchor matched random and the official
  checkpoint by +10/400 and +0.02 non-identity perturbation success while
  preserving clean identity within -1/100. The original adaptation chain was submitted on 2026-06-05
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
- The latest completed learned-policy route is perturb-only anchored
  OpenVLA-OFT adaptation in
  `scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh`. It changes
  the completed clean-mix recipe by training only on rendered boundary-band
  perturbation examples, with no clean anchor episodes mixed into the RLDS
  data, and adds a stronger official-checkpoint proximal L2 anchor
  (`PROXIMAL_ANCHOR_L2=5.0`) to protect clean identity behavior. Fixed command
  sequence: prep with
  `scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh --prep-only --submit-prep`,
  adapt after prep with
  `TRAIN_DEPENDENCY=afterok:<prep_job> scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh --adapt-only --submit-adapt`,
  and perturb-evaluate after BGR/random merge jobs with
  `BGR_DEPENDENCY=afterok:<bgr_merge> RANDOM_DEPENDENCY=afterok:<random_merge> scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh --perturb-only --submit-perturb`.
  It is promotable only if BGR beats both perturb-only anchored matched random
  and the official checkpoint by the fixed +10/400 and +0.02 non-identity
  perturbation gate while preserving clean identity within -1/100.
  Prep was submitted after commit `8b69ac7` on 2026-06-07 as Slurm job
  `767789`, writing to
  `/work/joy/bgr/logs/bgr-perturbonly-prep-p2048unique_perturbonly_anchor_prereg-767789.out`.
  Initial poll showed it running on `c1-g4-04` with `gres/gpu:a6000:1`. The
  dependent adaptation chain was then submitted with `TRAIN_DEPENDENCY=afterok:767789`
  and `GIT_PULL=0`: BGR train/merge/clean-eval jobs
  `767790`/`767791`/`767792`, and matched-random train/merge/clean-eval jobs
  `767793`/`767794`/`767795`. The fixed perturbation evals were submitted with
  `BGR_DEPENDENCY=afterok:767791` and
  `RANDOM_DEPENDENCY=afterok:767794`: official jobs `767796`-`767800`, BGR
  jobs `767801`-`767805`, and matched-random jobs `767806`-`767810`. A poll
  immediately after submission showed prep `767789` and official identity
  `767796` running, with all adaptation and BGR/random perturb jobs pending on
  dependencies. At that intermediate poll, these jobs were not paper evidence
  until compact summaries passed the fixed gate. Historical poll/sync commands:
  `REMOTE_RUN_ROOT=/work/joy/bgr/runs scripts/sync_openvla_oft_perturb_only_anchor_results.sh --poll`
  and
  `REMOTE_RUN_ROOT=/work/joy/bgr/runs scripts/sync_openvla_oft_perturb_only_anchor_results.sh --sync`.
  A helper poll at 2026-06-07 22:08:57 BST showed prep `767789` completed
  cleanly at 22:06:54 BST, BGR adaptation `767790` running on `c1-g4-04`, and
  official identity eval `767796` running on `c2-g4-24`; perturb-only adapt and
  perturb compact summaries were still missing.
  A later helper poll at 2026-06-07 22:19:57 BST showed BGR train/merge
  `767790`/`767791`, random train/merge `767793`/`767794`, and official
  identity `767796` completed with exit code `0:0`. BGR clean eval `767792`,
  random clean eval `767795`, BGR identity perturb eval `767801`, random
  identity perturb eval `767806`, and official blur `767797` were running; the
  remaining perturb jobs were still dependency-pending and compact summaries
  were still missing.
  The clean adaptation eval logs were summarized locally into
  `results/openvla_oft_goal_adapt_eval_p2048unique_perturbonly_anchor_prereg_perturbonly_proxanchor_l2_5em0_step50300_lr2em7_identitylora_imageaug_officialtrainstats_v1/summary.csv`;
  BGR and matched random both score 99/100 clean episodes. This clears the
  clean-floor sanity check for the adapted checkpoints but is not a promotion
  result without the full official/BGR/random perturbation summary.
  All perturbation jobs `767796`-`767810` then completed with exit code `0:0`;
  because the remote summaries were not written at the expected paths, the sync
  helper generated compact summaries from copied eval logs. The final perturb
  summary is
  `results/openvla_oft_perturb_eval_p2048unique_perturbonly_anchor_prereg_perturbonly_proxanchor_l2_5em0_step50300_lr2em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv`.
  The gate is negative: non-identity BGR 371/400, official 367/400, and random
  372/400; identity BGR 99/100, official 99/100, and random 99/100. BGR trails
  matched random by one episode and beats official by only four episodes, so it
  fails the fixed +10/400 and +0.02 learned-policy gate. This result should be
  incorporated only as negative audit evidence.

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
