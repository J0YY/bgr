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
main mechanism evidence, and OpenML diabetes, blood-transfusion, phoneme,
MagicTelescope, haberman, and jm1 are now replicated positive
pre-existing-dataset margin-replay evidence. A fixed all-binary numeric OpenML
target-1.5 aggregation over 32 datasets is also now complete and small
macro-positive across three independent 30-seed blocks: pooled BGR 0.7844 vs.
uniform 0.7766 and fixed-radius 0.7756, with BGR ahead on 22/32 dataset means
versus uniform and 24/32 versus fixed.
This strengthens the supervised pre-existing benchmark story but still does
not solve the standard-environment or learned-policy evidence gap.
Standard-environment recovery screens and OpenVLA/LIBERO learned-policy evidence
remain failing or non-promotable. The latest bsuite Catch 30-seed scale-up,
MiniGrid FourRooms radius-10 rescue, LunarLander 30-seed stress test,
HandReach-v3 calibration, highway-fast-v0 lane calibration, and MinAtar
Breakout all-method screen are also negative, so they do not solve the
independent-benchmark evidence gap. MinAtar Asterix also completed negative
after its usable calibration. New MinAtar Freeway and Space Invaders routes
cleared pre-method calibration but their fixed all-method screens tied every
method, so they are retired. A new MinAtar Seaquest package-state route also
cleared its 30-seed pre-method calibration, but the fixed 4-seed all-method
screen is saturated and negative: BGR-Coverage reaches 0.9017 RAUC versus
uniform 0.8958, fixed 0.9017, failure-only 0.9025, and BGR-uniform-radius
0.9192, while default BGR is 0.8892 and every method has median r80 5.0000.
Do not scale or promote Seaquest without a materially new premise. A new
Gymnasium Blackjack package-state recovery
scout completed negative on `athena` as Slurm job `774192`; this was only a
scout for a different independent reset interface, not paper evidence, and all
nine perturbation/target-radius configs are rejected before promotion. A new
Gymnasium Taxi-v3 package-state recovery screen is also completed negative:
the default 4-seed protocol is saturated and trails failure-only/TD-loss, while
the preregistered hard-budget follow-up calibrated from uniform-only runs gives
BGR-Coverage 0.5696 and BGR 0.5516 RAUC versus uniform 0.7596 and failure-only
0.9692. Do not scale or promote Taxi without a materially new premise. A
2026-06-11 deterministic-action bsuite DeepSea scout is also rejected: the
best `bgr_coverage` row at target 0.85/mix 0.80 gives 0.1594 RAUC versus
uniform 0.1031 and beats fixed, failure-only, TD-loss, and uniform-radius in
mean, but it has lower median `r80` than uniform. bsuite also warns this
deterministic action setting is debug mode, so do not scale it. A new MiniGrid
SimpleCrossing S9N3 scout is rejected before scale-up: the 8-seed default row
has `bgr_coverage` 0.5714 RAUC versus uniform 0.4746, failure-only 0.5223,
TD-loss 0.5547, and uniform-radius 0.5422, but paired wins are only 3/8 versus
uniform, 2/8 versus failure-only, 1/8 versus TD-loss, and 3/8 versus the
radius ablation. Harder-budget SimpleCrossing variants are negative or
dominated by uniform/failure-only/uniform-radius. Do not queue a SimpleCrossing
scale-up without a materially new fixed premise. The 2026-06-11 bsuite
Cartpole Swingup package-state recovery scout is also completed negative:
Slurm job `782844` finished with exit `0:0`, but default BGR reaches only
0.1044 RAUC versus uniform 0.0761, failure-only 0.1287, TD-loss 0.1456, and
BGR-uniform-radius 0.0732. BGR-Coverage is weaker at 0.0806 RAUC. Median r80
is saturated at 1.0000 for every method, so this route fails the strong-baseline
and radius gates and must not be scaled without a materially new fixed premise.
A broader
fixed OpenML numeric-suite
target-2.0 run and held-out seeds 30--59 replication completed on `athena` as
jobs `774312` and `774346`: the 10-dataset suite is mixed and not a macro win
(pooled macro BGR 0.7788 vs. uniform 0.7809 and fixed-radius 0.7830), but it
adds replicated dataset-level positives on MagicTelescope and haberman. This is
additional supervised pre-existing-dataset evidence only, not a
standard-environment or learned-policy win. The latest completed OpenVLA/LIBERO
hard-occlusion transfer diagnostics are also negative. At occlusion fraction
0.80, BGR reaches 305/400 non-identity successes versus official 296/400 and
matched random 296/400, but the +9 episode margin misses the fixed +10/400
gate and identity is BGR 391/400 versus official 393/400, violating the
identity side condition. The corrected router-specific hard-occlusion 0.80
training route is also completed negative: BGR reaches 297/400 versus official
298/400 and matched random 300/400, failing the occlusion criterion directly.
A held-out occlusion-only confirmation of this same 0.80 transfer checkpoint is
also completed negative as a router-style diagnostic, not a paper claim. The
first attempt, official/BGR/random `782604`/`782605`/`782606`, incorrectly used
`EVAL_INIT_STATE_OFFSET=40` with `EVAL_TRIALS=80`; LIBERO-Goal has only 50
initial states per task, so official failed at index 50 and BGR/random were
cancelled. The corrected held-out slice used the remaining 10 initial states per
task: official `782609`, BGR `782610`, and matched random `782611` evaluated
hard occlusion 0.80 with `EVAL_INIT_STATE_OFFSET=40`, `EVAL_TRIALS=10`,
`EVAL_SEED=137`, and `SAVE_ROLLOUTS=0`. The compact local summary at
`results/openvla_oft_perturb_eval_occlusion_bottleneck_hardocc080_transfer_step50400_lr2em7_heldout_offset40_trials10_v1/summary_available.csv`
has BGR 69/100, official 71/100, and matched random 71/100. The combined
original-plus-held-out readout is BGR 374/500 versus official/random 367/500,
only a +7 episode margin and below the +0.02 router-style requirement. Do not
formalize this fallback/router premise from the transfer checkpoint.
At occlusion fraction 0.65, BGR reaches 300/400
successes versus official 297/400 and matched random 296/400, only +3/+4
episodes and +0.0075/+0.0100 success rate rather than the fixed +10/400 and
+0.02 promotion margin; identity is again BGR 391/400, official 393/400, and
matched random 389/400. The 0.65 adaptation route has BGR identity 389/400
versus official 393/400 and BGR occlusion 301/400 versus official 297/400 and
matched random 304/400, failing both the identity side condition and the
matched-random comparison. The A40 adaptation fallback is also already closed
negative on identity side condition with BGR identity
391/400 versus official 393/400. The completed hard-occlusion 0.80
identity-anchored route is now also negative: BGR has identity 389/400 versus
official and matched random at 393/400, and occlusion 303/400 versus official
296/400 and matched random 302/400, so it fails the identity side condition
and beats matched random by only +1/400. Still-running identity-anchored
hard-occlusion routes remain incomplete and must not be incorporated into
`paper/main.tex` unless their full summaries pass the fixed gate. A new fixed
hard-occlusion
0.80 head-interpolation route was queued on 2026-06-10 after the completed
transfer route missed promotion by one occlusion episode and two identity
episodes. It copies the completed occlusion-bottleneck BGR and matched-random
checkpoints, interpolates trainable action/proprio heads toward the official
checkpoint with `ALPHA=0.75`, scales LoRA-B tensors by the same alpha, and
evaluates official, interpolated BGR, and interpolated matched random on
identity plus occlusion fraction 0.80 over 10 LIBERO-Goal tasks x 40 trials.
Jobs are prep `779973`, official identity/occlusion `779974`/`779975`, BGR
identity/occlusion `779976`/`779977`, and matched-random identity/occlusion
`779978`/`779979`. Latest poll/sync at 2026-06-10 22:03 BST showed prep
`779973` completed successfully at 21:16:12 and official identity `779974`
completed successfully at 393/400. BGR identity `779976` was still running on
`c1-g4-03` after 18:28, matched-random identity `779978` was running on
`c1-g4-05` after 2:53, official occlusion `779975` was priority-pending, and
BGR/random occlusion `779977`/`779979` remained dependency-pending behind their
identity evals. The synced incomplete summary has only the official identity
row; live log tails showed BGR identity around 89/94 and matched random around
20/20, which is not gateable evidence. This route is not paper evidence unless
the full `summary.csv` passes the unchanged fixed gate:
BGR must beat both official and matched random by at least 10/400 occlusion
episodes and at least 0.02 absolute success rate while not trailing the best
identity comparator by more than one episode. Latest poll/sync at 2026-06-10
22:13 BST closed this route early as non-promotable on identity: BGR identity
reached 134/144, so even perfect success afterward could finish no better than
390/400 against official identity 393/400. The remaining route-A jobs were
cancelled to save cluster time: official occlusion `779975`, BGR identity
`779976`, BGR occlusion `779977`, random identity `779978`, and random
occlusion `779979` are all cancelled, while official identity `779974`
completed at 393/400. This route is negative and must not be incorporated into
`paper/main.tex`. A stronger head-only repair
variant was queued at 2026-06-10 21:12 BST before any head-interpolation
summary was available: it uses the same `ALPHA=0.75` head interpolation but
keeps adapted LoRA-B tensors at full scale with `LORA_B_SCALE=1.0`, testing
whether identity can be repaired without shrinking the occlusion adaptation
signal. Jobs are prep `780059`, official identity/occlusion `780060`/`780061`,
BGR identity/occlusion `780062`/`780063`, and matched-random
identity/occlusion `780064`/`780065`. Latest poll/sync at 2026-06-10
22:03 BST showed prep `780059` running after starting at 22:03:06, official
identity `780060` pending on resources with estimated start
2026-06-11 06:21:43 BST, and all downstream official occlusion and BGR/random
evals dependency-pending. No logs or summary were available yet. This is also
not evidence unless
the same fixed +10/400, +0.02, and identity-preservation gate passes. Latest
poll/sync at 2026-06-10 22:13 BST showed this LoRA-full route past prep:
official identity `780060` was running, BGR identity `780062` and random
identity `780064` were priority-pending, and all occlusion jobs were
dependency-pending. Latest poll/sync at 2026-06-10 22:20 BST showed the
LoRA-full route still incomplete but active: official identity `780060`, BGR
identity `780062`, and matched-random identity `780064` were running, while
official/BGR/random occlusion jobs `780061`/`780063`/`780065` remained
dependency-pending. Live identity log tails at 2026-06-10 22:20 BST showed
official 131/137, BGR 51/53, and matched random 26/27; this is not gateable
evidence and does not yet rule the route out. Do not launch another
overlapping OpenVLA variant while this LoRA-full route remains unresolved.
Latest poll/sync at 2026-06-10 22:33 BST showed all three identity jobs
requeued or preempted back to `PENDING` on `Priority`: official `780060` and
BGR `780062` have estimated starts at 2026-06-11 10:04:52 BST, and
matched-random `780064` has estimated start at 2026-06-11 17:11:00 BST. The
occlusion jobs `780061`/`780063`/`780065` remain dependency-pending, the full
remote `summary.csv` is still missing, and the synced local file is only
`summary_available.csv`. Direct log tails before preemption reached official
178/185, BGR 112/119, and matched random 65/68 identity successes. This still
is not gateable evidence; BGR has seven identity failures by 119 episodes, so
identity-side feasibility is weak but not yet mathematically closed because
the comparator identity rows are also incomplete. Latest poll/sync at
2026-06-10 22:53 BST showed the LoRA-full route closed by infrastructure
failure rather than a complete gate result: official/BGR/matched-random
identity jobs `780060`/`780062`/`780064` restarted and failed quickly with
exit `1:0` after 00:50/00:23/00:28, leaving occlusion jobs
`780061`/`780063`/`780065` stuck on `DependencyNeverSatisfied`. Those
dependent occlusion jobs were cancelled at 22:54 BST. No full `summary.csv`
exists, so this route is not a paper result and should not be rerun unchanged
without first diagnosing why the restarted identity evals failed. The restart
failure was diagnosed as an infrastructure/quota issue rather than a policy
result: `/work/joy` was full, the failed jobs stopped without a Python
traceback, and `sacct` showed low MaxRSS. The queue path now supports
`SAVE_ROLLOUTS=0` and patches the LIBERO video hook to skip MP4 writes while
keeping text logs. Reproducible alpha-0.75 head-interpolation checkpoint copies
and a UV temp cache were removed on `athena`, raising free `/work/joy` space
from about 1.6G to 5.6G.
Completed learned-policy repair route (negative): fixed no-video alpha-0.0
official-head/full-LoRA repair,
`TAG=occlusion_bottleneck_hardocc080_transfer_headinterp000_lorafull_novideo_v1`,
`ALPHA=0.0`, `LORA_B_SCALE=1.0`, and
`SAVE_ROLLOUTS=0`. This keeps adapted LoRA tensors but restores official
action/proprio heads exactly, targeting the identity side-condition without
discarding the occlusion adaptation. Submitted jobs were prep `782410`,
official identity/occlusion `782411`/`782412`, BGR identity/occlusion
`782413`/`782414`, and matched-random identity/occlusion `782415`/`782416`,
with `EXCLUDE=c2-g4-17,c2-g4-18,c2-g4-19,c2-g4-21,c2-g4-23`. Initial poll at
2026-06-11 01:33:40 BST showed prep completed, identity jobs running, and
occlusion jobs dependency-pending. The early partial summary had only identity
rows over 8--14 episodes, all at 100%, so it is not gateable evidence. Latest
poll/sync at 2026-06-11 01:48:36 BST still showed official/BGR/matched-random
identity jobs `782411`/`782413`/`782415` running and occlusion jobs
`782412`/`782414`/`782416` dependency-pending. The synced incomplete
`summary_available.csv` has only identity rows: BGR 136/143, official
141/147, and matched random 138/144. Direct log tails were slightly ahead of
the compact parse at BGR 137/144, official 143/149, and matched random
139/145. This is still incomplete non-evidence; BGR is currently one identity
failure behind the best comparator, so the identity side-condition remains
precarious but not mathematically closed. The final identity rows closed the
route negative before occlusion: BGR identity `391/400`, official `393/400`,
and matched random `392/400`. BGR trails the best identity comparator by two
episodes, exceeding the fixed side-condition of at most one episode. Occlusion
jobs `782412`/`782414`/`782416` were cancelled at 2026-06-11 02:14 BST after
running only tiny partial fragments, so no occlusion comparison should be
interpreted. The compact local closure artifact is
`results/openvla_oft_perturb_eval_occlusion_bottleneck_hardocc080_transfer_headinterp000_lorafull_novideo_v1/summary_available.csv`.
Do not poll this route except to audit logs; use:
`ARTIFACT=openvla_oft_perturb_eval_occlusion_bottleneck_hardocc080_transfer_headinterp000_lorafull_novideo_v1 JOB_IDS=782410,782411,782412,782413,782414,782415,782416 DETAIL_JOB_IDS=782410,782411,782412,782413,782414,782415,782416 ROUTE_LABEL='Hard-occlusion 0.80 alpha0 official-head/full-LoRA no-video OpenVLA-OFT repair' scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh --poll --sync --no-check`.
Do not incorporate this route into `paper/main.tex`; it is a completed
negative/non-promotable learned-policy repair attempt. After this closure, old
dependency-held OpenVLA jobs from superseded routes were cancelled on Athena;
`squeue -u $(whoami)` was empty at 2026-06-11 02:16 BST. No active
learned-policy cluster jobs remained queued at that checkpoint. A new
occlusion-only learned-policy scout was then submitted at 2026-06-11
03:35 BST to test a genuinely different intervention premise: an
official-identity fallback/router would use the official checkpoint for clean
identity and only use the adapted branch on known hard-occlusion inputs. This
scout evaluated only hard-occlusion 0.80 for the same alpha-0 official-head/
full-LoRA no-video checkpoints. It is completed non-promotable and not paper
evidence because the router was not formalized and the occlusion margin is too
small. Jobs official/BGR/matched-random `782556`/`782557`/`782558` all
completed with exit `0:0`. The compact local summary at
`results/openvla_oft_perturb_eval_occlusion_bottleneck_hardocc080_transfer_headinterp000_lorafull_novideo_occscout_v1/summary_available.csv`
has BGR 301/400, official 298/400, and matched random 298/400, only a +3
episode margin over both comparators and far below the required +10/400 and
+0.02 router-style threshold. Do not formalize this alpha-0 fallback/router
premise. The first router-specific occlusion-only training attempt
`p1024unique_occonly_hardocc080_router_prereg` also failed before adaptation:
prep `782649` rendered zero matched-random occlusion examples before the
post-render family filter, exited `1:0`, and left dependent jobs
`782650`--`782655`/`782657`/`782658` unsatisfied; those jobs and orphaned
official eval `782656` were cancelled. The fix is now in code:
`scripts/render_openvla_teacher_examples.py` supports a pre-selection
`--include-family` filter, and
`scripts/queue_openvla_oft_preregistered_occlusion_bottleneck.sh` passes
`--include-family occlusion` to both BGR and matched-random perturb renders.
Because the remote manifest has 1,600 BGR occlusion rows but only 512
matched-random occlusion rows, the corrected fair route uses cap 512 and
repeat 2. Corrected active route queued at 2026-06-11 04:55 BST:
prep `782671`, BGR train/merge/clean-eval `782672`/`782673`/`782674`,
matched-random train/merge/clean-eval `782675`/`782676`/`782677`, and
official/BGR/random hard-occlusion evals `782679`/`782680`/`782681`, with
`PREP_TAG=p512unique_occonly_hardocc080_router_randfix_prereg`,
`ADAPT_TAG=occonly_p512unique_hardocc080_router_randfix_step50300_lr5em7_identitylora_imageaug_officialtrainstats_v1`,
`PERTURB_TAG=occonly_p512unique_hardocc080_router_randfix_step50300_lr5em7_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1`,
`OCCLUSION_CAP=512`, `OCCLUSION_REPEAT=2`,
`OCCLUSION_FRACTION_OVERRIDE=0.80`, `PROXIMAL_ANCHOR_L2=0`,
`ADAPT_STEPS=300`, and `LR=5e-7`. It is only router-premise evidence unless
BGR beats both official and matched random by at least +10/400 and +0.02 on
hard occlusion. Poll/sync with:
`PREP_TAG=p512unique_occonly_hardocc080_router_randfix_prereg ADAPT_TAG=occonly_p512unique_hardocc080_router_randfix_step50300_lr5em7_identitylora_imageaug_officialtrainstats_v1 PERTURB_TAG=occonly_p512unique_hardocc080_router_randfix_step50300_lr5em7_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1 JOB_IDS=782671,782672,782673,782674,782675,782676,782677,782679,782680,782681 DETAIL_JOB_IDS=782671,782672,782673,782675,782676,782679,782680,782681 ROUTE_LABEL='Hard-occlusion 0.80 occlusion-only router-trained OpenVLA-OFT randfix premise' GATE_PERTURBATIONS=occlusion scripts/sync_openvla_oft_occlusion_bottleneck_results.sh --poll --sync --no-check`.
Final poll/sync at 2026-06-11 06:34 BST closed the corrected router-specific
occlusion-only route as negative. Prep, BGR/random train/merge/clean evals,
and official/BGR/random hard-occlusion evals all completed with exit `0:0`.
The remote wrapper still did not write a full perturb `summary.csv`, but the
sync helper reconstructed a complete local nested-log summary at
`results/openvla_oft_perturb_eval_occonly_p512unique_hardocc080_router_randfix_step50300_lr5em7_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1/summary_available.csv`.
The clean/adapt summary has BGR 386/400 and matched random 388/400. The
completed hard-occlusion rows are BGR 297/400, official 298/400, and matched
random 300/400. BGR trails official by 1 episode and matched random by 3
episodes, so it fails the decisive router occlusion criterion before any
identity-side interpretation. Do not incorporate this route into
`paper/main.tex`; it is another learned-policy negative result.
Completed low-priority learned-policy diagnostic: because the canceled
hard-occlusion 0.90 alpha-0 no-video scout had unequal partial rows and a small
BGR edge over matched random, a fixed full 400-episode rerun was submitted on
2026-06-11 to close that severity cleanly. This was only a router-premise
diagnostic, not a paper claim. It used
`TAG=occlusion_bottleneck_hardocc090_transfer_headinterp000_lorafull_novideo_fullrerun_v1`,
`ALPHA=0.0`, `LORA_B_SCALE=1.0`, `PERTURBATIONS='occlusion={"fraction":0.90}'`,
`EVAL_TASKS=10`, `EVAL_TRIALS=40`, `EVAL_SEED=37`, and `SAVE_ROLLOUTS=0` through
`scripts/queue_openvla_oft_hard_occlusion080_headinterp_results.sh --submit`.
Submitted jobs are prep `782931`, official occlusion `782932`, BGR occlusion
`782933`, and matched-random occlusion `782935`; all completed with exit
`0:0`. Final reconstructed local summary at
`results/openvla_oft_perturb_eval_occlusion_bottleneck_hardocc090_transfer_headinterp000_lorafull_novideo_fullrerun_v1/summary_available.csv`
has BGR 307/400, official 305/400, and matched random 303/400. BGR beats both
comparators but only by +2 and +4 episodes (+0.005 and +0.010), far below the
fixed +10/400 and +0.02 router gate, so this route is negative and must not be
incorporated into `paper/main.tex`.
Completed learned-policy diagnostic: hard-occlusion 0.90 occlusion-only
router-trained OpenVLA-OFT premise. This was a changed training distribution,
not another same-checkpoint re-evaluation: it trained matched BGR and random
branches only on hard-occlusion 0.90 examples while a hypothetical router would
use the official checkpoint for clean identity. Fixed configuration:
`PREP_TAG=p512unique_occonly_hardocc090_router_prereg`,
`ADAPT_TAG=occonly_p512unique_hardocc090_router_step50300_lr5em7_identitylora_imageaug_officialtrainstats_v1`,
`PERTURB_TAG=occonly_p512unique_hardocc090_router_step50300_lr5em7_identitylora_imageaug_officialtrainstats_hardocc090_occonly_fullgoal10x40_v1`,
`OCCLUSION_CAP=512`, `OCCLUSION_REPEAT=2`, `INCLUDE_CLEAN_ANCHORS=0`,
`OCCLUSION_FRACTION_OVERRIDE=0.90`, `PROXIMAL_ANCHOR_L2=0`, `ADAPT_STEPS=300`,
`LR=5e-7`, `EVAL_TASKS=10`, `EVAL_TRIALS=40`, `EVAL_SEED=37`, and
`PERTURBATIONS='occlusion={"fraction":0.90}'`. Prep, BGR train/merge/clean
eval, random train/merge/clean eval, and official/BGR/random hard-occlusion
eval jobs `783002`--`783008` and `783034`--`783036` all completed with exit
`0:0`.
Earlier eval jobs `783009`--`783023` and `783028`--`783030` were cancelled
because they were submitted under a contaminated/default perturb artifact; do
not sync or interpret those. Final poll/sync at 2026-06-11 10:23:38 BST
closed the route negative. The clean/adapt summary has BGR 391/400 and matched
random 391/400. The hard-occlusion summary has BGR 305/400, official 305/400,
and matched random 301/400. BGR ties official and beats matched random by only
+4/400 (+0.0100), far below the +10/400 and +0.02 router gate. Reproduction
sync:
`PREP_TAG=p512unique_occonly_hardocc090_router_prereg ADAPT_TAG=occonly_p512unique_hardocc090_router_step50300_lr5em7_identitylora_imageaug_officialtrainstats_v1 PERTURB_TAG=occonly_p512unique_hardocc090_router_step50300_lr5em7_identitylora_imageaug_officialtrainstats_hardocc090_occonly_fullgoal10x40_v1 JOB_IDS=783002,783003,783004,783005,783006,783007,783008,783034,783035,783036 DETAIL_JOB_IDS=783002,783003,783004,783006,783007,783034,783035,783036 ROUTE_LABEL='Hard-occlusion 0.90 occlusion-only router-trained OpenVLA-OFT premise' GATE_PERTURBATIONS=occlusion scripts/sync_openvla_oft_occlusion_bottleneck_results.sh --poll --sync --no-check`.
New low-cost learned-policy diagnostic queued after the 0.90 trained-router
partial looked weak: an occlusion-fraction 0.75 scout of the original
occlusion-bottleneck transfer checkpoints, with no rollout videos and a fresh
artifact `openvla_oft_perturb_eval_occlusion_bottleneck_transfer_occ075_scout_v1`.
Submitted jobs are official/BGR/matched-random `783104`/`783105`/`783106`.
Latest poll/sync at 2026-06-11 10:20:00 BST showed official `783104` and BGR
`783105` running, matched random `783106` pending on unavailable nodes, and a
weakened incomplete summary of BGR 77/159 versus official 82/165. This is a
severity-window scout only, not a paper claim and not a replacement for the
fixed gate. Use it only to decide whether a future preregistered router-style
gate is worth running; promotion would still require BGR to beat both official
and matched random by at least +10/400 and +0.02 on a fixed 400-episode
readout.
Do not modify separate `rl4vla-*` jobs on Athena for this project.
The
latest 0.80 identity-anchored base route is closed negative with complete
rows: BGR identity/occlusion are 389/400 and 303/400, official is 393/400 and
296/400, and matched random is 393/400 and 302/400. The fixed gate reports
episode_margin=1, rate_margin=+0.0025 against the best occlusion comparator,
and identity_deficit=4. The micro and strict 0.80 variants are also already
non-promotable on partial identity rows: BGR identity is 387/400 for the micro
route and 388/400 for the strict route, each trailing official identity
393/400 by more than one episode. The micro route still has only BGR identity
387/400 and official identity 393/400 summarized, with official occlusion
`777039`, BGR occlusion `777041`, and matched-random identity `777042` running.
The strict route partial summary now has official occlusion 296/400, while BGR
occlusion `776551` and matched-random occlusion `776554` were running and
matched-random identity has completed at 391/400; the micro and strict routes
cannot be promoted because their identity rows already fail the side condition.
The 0.80 A40 fallback still only has official identity 393/400 summarized,
while the 0.90 strict A40 route has official identity 393/400 summarized and
downstream rows dependency-pending. The shared sync helper now parses both
final total lines and repeated progress lines, and selects the highest-episode
retry log per method/perturbation so duplicate partial logs no longer hide
completed compact summaries. The earlier completed full
perturbation
occlusion-bottleneck route was negative as well: BGR reaches 365/400
non-identity successes versus official 367/400 and matched random 369/400,
with identity BGR 99/100, official 99/100, and random 98/100. The
new broad CPU supervised check is a fixed all-binary numeric OpenML target-1.5
sweep over all current numeric binary OpenML datasets already registered in
`tools/openml_margin_scout.py`. Original seeds 0--29 were submitted as Slurm
job `780049`, and held-out seeds 30--59 were submitted as job `780050`; both
started on 2026-06-10 at 21:02 BST with `TARGETS=1.5`, `SEEDS=30`,
`PREPROCESSING=numeric`, and 32 datasets. Both jobs completed successfully on
2026-06-10 (`780049` at 21:34:24 BST and `780050` at 21:34:48 BST) and synced
to `results/openml_all_binary_numeric_target15_30seed_v1_780049/` and
`results/openml_all_binary_numeric_target15_replication_30seed_v1_780050/`.
The fixed readout is a small broad supervised macro win: original/held-out
macro means are BGR 0.7842/0.7859, uniform 0.7774/0.7764, and fixed-radius
0.7790/0.7741; pooled means are BGR 0.7851 vs. uniform 0.7769 and fixed-radius
0.7766, with BGR ahead on 22/32 dataset means versus uniform and 24/32 versus
fixed. The pooled simple-screen positive rows are MagicTelescope,
blood-transfusion-service-center, diabetes, haberman, jm1, and kc2, but kc2 is
still below the stricter +0.03 fixed-radius margin used for earlier
dataset-level promotion. Treat this as a small macro-positive supervised
pre-existing benchmark aggregation, not a learned-policy or standard-control
win. The deterministic readout is saved at
`results/openml_all_binary_numeric_target15_analysis_780049_780050.txt`. A
third independent seed block initially failed as monolithic Slurm job `781423`
and several opaque signal-53 retries while `/work/joy` was full. The OpenML
queue path now submits materialized remote `.sbatch` files and
`tools/openml_margin_scout.py` has checkpoint/resume support. After deleting
remote generated OpenVLA rollout videos to free quota, a Slurm smoke `781672`
passed, and the fixed third block was rerun in four checkpointed chunks:
`781682`/`781683`/`781684`/`781685`. All completed with exit `0:0`, were
synced locally as intermediate chunk directories, and were merged into
`results/openml_all_binary_numeric_target15_thirdsplit_30seed_v1_781682_781685/`.
The third block remains small macro-positive: BGR 0.7831 vs. uniform 0.7762
and fixed-radius 0.7736, ahead on 24/32 dataset means against each comparator.
Pooled over seeds 0--89, BGR is 0.7844 vs. uniform 0.7766 and fixed-radius
0.7756, ahead on 22/32 and 24/32 dataset means. This strengthens the
supervised pre-existing benchmark aggregation but still does not solve the
standard-control or learned-policy gap. The
internal sklearn-digits margin replay scout is also rejected before promotion:
its best BGR target gives only 0.8271 vs. 0.8123 RAUC against uniform with a
2/2/0 paired split, while fixed-radius replay is stronger at another target.
The internal sklearn tabular margin replay scout is also rejected: breast
cancer's best BGR target gives 0.9610 vs. 0.9516 RAUC against uniform, and
wine's best gives 0.9702 vs. 0.9563, below the +0.03 pre-registration screen.
A 2026-06-10 FetchPush object-goal controller/horizon calibration follow-up
fixed the local calibration harness so `--horizon` is honored via
`max_episode_steps` and added a new `scripted_push_sweep` controller, but it is
also rejected before method comparison: the best compact/wide scouts reach
clean success 0.8750 with recovery 0.7500--0.8750, yet the critical radius is
saturated at the tested maximum (`decision:
reject-calibration-radius-saturated`). Do not build a FetchPush method
comparison from this interface without a new perturbation premise that first
creates a non-saturated boundary.
A materially different 2026-06-10 FetchPush object-state perturbation
calibration now gives one usable reset-interface route, but not paper evidence
yet. The fixed command
`PYTHONPATH=src:. /tmp/bgr_fetch_venv/bin/python tools/fetch_object_goal_recovery_calibration.py --env-id FetchPush-v4 --seeds 2 --replay-states 4 --trials 2 --radii 0.00,0.02,0.04,0.06,0.08,0.12,0.16,0.20 --horizon 250 --controller scripted_push_sweep --controller-gain 8.0 --perturbation-target object --out results/fetchpush_object_state_calibration_sweep_g8_h250_2seed_v1`
reports clean success 0.8750, recovery 0.2500--0.8750, RAUC 0.4125, r80
0.0140, and `decision: usable-calibration`. This is only a pre-method
calibration. A first local linear-imitator method-probe scaffold was added in
`tools/fetchpush_object_state_recovery_probe.py`, but its viability checks are
rejected before promotion: uniform training stayed at zero clean success and
zero RAUC even with teacher-forced behavior cloning. Do not scale that learner;
the active route requires a better preregistered learned controller or a
separate fixed method comparison that first proves nonzero clean recovery.
Follow-up implementation added opt-in MLP, KNN, and trajectory-library policies
to the same probe and fixed the probe seed offset to match the calibrated
`121000` reset family. The MLP/KNN behavior-cloning variants are still rejected
by local viability checks, but the trajectory-library policy with common
warm-start demonstrations gives a nonzero local viability row on calibrated
states: uniform seed 0, clean 0.7500 and RAUC 0.5312 under
`--policy trajectory --warmstart-policy`. A bounded all-baseline CPU Slurm
scout was submitted on `athena` as job `777783` at 2026-06-10 16:49 BST using
`scripts/queue_fetchpush_object_state_probe.sh`. It writes to
`/work/joy/bgr/runs/fetchpush_object_state_recovery_probe_scout_v1_777783`
and logs to `/work/joy/bgr/logs/bgr-fetchpush-object-state-777783.out`; sync
with `JOB_ID=777783 scripts/sync_fetchpush_object_state_probe.sh`. This is not
paper evidence unless `summary.csv` exists and the fixed candidate-promotion
checks beat uniform, fixed, failure-only, TD-loss, and the uniform-radius
ablation.
Latest partial FetchPush object-state method evidence is not promotable. The
single serial scout job `777783` and split per-method jobs `777896`--`777901`
are still running or partially complete, but the sparse-probe BGR rows are now
rejected before promotion. The paired sparse summary has uniform seeds 0--3 at
RAUC 0.3125/0.3000/0.1500/0.2500; sparse BGR-Coverage seeds 0--3 at
0.1875/0.2250/0.1250/0.2125; sparse BGR seeds 0--3 at
0.1750/0.2375/0.1500/0.0500; BGR-uniform-radius seeds 0--3 at
0.3250/0.2750/0.1500/0.2125; TD-loss seeds 0--3 at
0.3250/0.3625/0.1375/0.2000; and failure-only seeds 0--2 at
0.3000/0.3875/0.1500. `tools/check_candidate_promotion.py` rejects sparse
BGR-Coverage at 0.1875 mean RAUC versus uniform 0.2531
(W/L/T=0/4/0), TD-loss 0.2562, and BGR-uniform-radius 0.2406; it rejects
sparse BGR at 0.1531 versus uniform 0.2531 (W/L/T=0/3/1), fixed 0.1750,
TD-loss 0.2562, and BGR-uniform-radius 0.2406. A corrected dense-probe BGR
diagnostic was submitted as job `777969` after canceling mislabeled job
`777958`; it uses `INITIAL_PROBES=0.00,0.02,0.08,0.20`,
`TARGET_RADIUS=0.046`, and `RADIUS_BANDWIDTH=0.050`. Its first three
available BGR-Coverage rows are 0.3125/0.3875/0.1750 RAUC on seeds 0/1/2, but
dense warm-start probes can affect all methods. To avoid an unfair
dense-vs-sparse comparison, matched dense-probe per-method jobs were submitted
on `athena` at 2026-06-10 17:30 BST:
`778100` uniform, `778101` fixed, `778102` failure-only, `778103` TD-loss,
`778104` BGR-uniform-radius, `778105` BGR-Coverage, and `778106` BGR. They use
the same dense probes, `TARGET_RADIUS=0.046`, `RADIUS_BANDWIDTH=0.050`, and
write to `/work/joy/bgr/runs/fetchpush_object_state_recovery_probe_densecommon_<method>_v1_<job>`.
The matched dense-probe BGR-Coverage row is now completed negative before
promotion: dense uniform seeds 0--3 have RAUC 0.3875/0.3375/0.1750/0.3000
(mean 0.3000), dense fixed has 0.3375/0.3625/0.1500/0.1750 (mean 0.2563),
dense BGR-uniform-radius has 0.3625/0.3625/0.1500/0.2750 (mean 0.2875), and
dense BGR-Coverage has 0.3125/0.3875/0.1750/0.2500 (mean 0.2812).
`tools/check_candidate_promotion.py` rejects dense BGR-Coverage versus uniform
(W/L/T=1/2/1, delta -0.0188), versus the uniform-radius ablation (delta
-0.0063), and on the median-r80 contradiction. The dense default-BGR job
`778106`, failure-only job `778102`, and TD-loss job `778103` were still
running at the 2026-06-10 17:57--18:00 BST poll, with only seed-0 rows locally.
Treat dense default BGR as incomplete and not paper evidence unless the
completed matched dense common-protocol summary clears the candidate-promotion
checks.
Latest poll at 2026-06-10 18:05 BST showed the same dense FetchPush status:
failure-only `778102`, TD-loss `778103`, and default BGR `778106` were still
running on CPU nodes after roughly 34 minutes. Local summaries remain complete
only for uniform, fixed, BGR-uniform-radius, and BGR-Coverage; failure-only,
TD-loss, and default BGR still have only seed 0 locally. A combined partial
candidate check again rejects dense BGR-Coverage, and rejects default BGR as
incomplete with seed 0 already below uniform, fixed, failure-only, and
BGR-uniform-radius. Do not retune the dense FetchPush target/bandwidth while
these fixed common-protocol jobs are still incomplete.
Follow-up sync at 2026-06-10 18:08 BST closed dense TD-loss and dense default
BGR, while failure-only `778102` remained running. Dense default BGR is now
rejected before promotion: mean RAUC is 0.2750 versus uniform 0.3000
(W/L/T=0/2/2), fixed 0.2563, TD-loss 0.2687, and BGR-uniform-radius 0.2875,
with median-r80 lower than uniform. Dense BGR-Coverage remains rejected at
0.2812 versus uniform 0.3000 and BGR-uniform-radius 0.2875. The only missing
row for final dense route closure is the completed four-seed failure-only
summary; sync `778102` when it finishes, but neither BGR-family treatment can
be promoted because both already fail versus uniform.
Final dense FetchPush object-state closure: failure-only `778102` completed at
2026-06-10 18:10:47 BST and synced cleanly. The completed dense common-protocol
means are uniform 0.3000, failure-only 0.2938, BGR-uniform-radius 0.2875,
BGR-Coverage 0.2812, default BGR 0.2750, TD-loss 0.2687, and fixed 0.2563
RAUC. `tools/check_candidate_promotion.py` rejects BGR-Coverage versus uniform
(delta -0.0188, W/L/T=1/2/1), failure-only (delta -0.0125), and
BGR-uniform-radius (delta -0.0063), with median-r80 contradiction. It rejects
default BGR versus uniform (delta -0.0250, W/L/T=0/2/2), failure-only (delta
-0.0187), and BGR-uniform-radius (delta -0.0125), also with median-r80
contradiction. Treat the dense FetchPush object-state route as completed
negative, not paper evidence.
New active CPU independent-benchmark scout: a fixed mixed-type binary OpenML
suite was added to `tools/openml_margin_scout.py` behind
`--mixed-binary-suite`, using one-hot categorical preprocessing instead of the
existing numeric-only pipeline. The suite is credit-g, kr-vs-kp, tic-tac-toe,
mushroom, bank-marketing, adult, PhishingWebsites, and credit-approval at
target radii 0.5, 1.0, 1.5, and 2.0 with the same uniform/fixed/BGR methods.
This is a scout for a materially different pre-existing benchmark interface,
not paper evidence. A local smoke on credit-g and tic-tac-toe with
`--preprocessing mixed` passed. The fixed 4-seed Athena scout was submitted at
2026-06-10 18:17 BST as job `778553` via
`scripts/queue_openml_mixed_binary_suite.sh`; it writes to
`/work/joy/bgr/runs/openml_mixed_binary_scout_v1_778553` and logs to
`/work/joy/bgr/logs/bgr-openml-mixed-binary-778553.out`. Initial log tails
showed healthy credit-g rows, including some BGR-positive target-1.0/1.5 rows,
but no conclusion should be drawn until `summary.csv` exists. If candidates
clear the 4-seed scout gate, run fixed 30-seed and held-out 30-seed follow-ups
before making any manuscript claim.
The 4-seed mixed OpenML scout `778553` completed with exit `0:0` and synced to
`results/openml_mixed_binary_scout_v1_778553/`. No row cleared the strict scout
gate because every positive-looking row had at least one paired loss versus
uniform. The top BGR rows were credit-approval target 2.0, BGR 0.8447 vs.
uniform 0.7849 (+0.0598, W/L/T=3/1/0) and fixed 0.7949; credit-approval target
1.5, BGR 0.8358 vs. uniform 0.7849 (+0.0509, W/L/T=3/1/0); adult target 2.0,
BGR 0.7963 vs. uniform 0.7648 (+0.0315, W/L/T=3/1/0) and fixed 0.7628; and
bank-marketing target 2.0, BGR 0.8847 vs. uniform 0.8534 (+0.0312,
W/L/T=3/1/0) and fixed 0.8631. Treat these as near-misses, not candidates. To
avoid cherry-picking a single near-miss row, fixed all-dataset/all-target
30-seed mixed-suite diagnostics were submitted before any manuscript use:
original seeds 0--29 job `778596`, writing to
`/work/joy/bgr/runs/openml_mixed_binary_target_sensitivity_30seed_v1_778596`,
and held-out seeds 30--59 job `778597`, writing to
`/work/joy/bgr/runs/openml_mixed_binary_target_sensitivity_replication_30seed_v1_778597`.
Initial poll showed `778596` running on `cnode401` and `778597` pending on
resources. These jobs are route-closing/route-promoting diagnostics only; no
paper claim should change unless fixed summaries pass the follow-up gates.
The mixed-suite 30-seed diagnostics completed at 2026-06-10 18:59 BST with
exit `0:0` and synced via `scripts/sync_openml_mixed_binary_results.sh` to
`results/openml_mixed_binary_target_sensitivity_30seed_v1_778596/` and
`results/openml_mixed_binary_target_sensitivity_replication_30seed_v1_778597/`.
The overall mixed suite is not a macro win: pooled macro means are BGR 0.7891,
uniform 0.7936, and fixed-radius 0.8000, with BGR ahead on 17/32
dataset-target means versus uniform and 16/32 versus fixed. The one clean
replicated row by the fixed follow-up screen is adult target radius 1.5:
original BGR 0.7981 vs. uniform 0.7544 (+0.0437, W/L/T=22/8/0) and fixed
0.7677 (+0.0304), held-out BGR 0.7901 vs. uniform 0.7550 (+0.0350,
W/L/T=21/9/0) and fixed 0.7557 (+0.0344), pooled BGR 0.7941 vs. uniform
0.7547 (+0.0394, W/L/T=43/17/0) and fixed 0.7617 (+0.0324, W/L/T=38/22/0).
Credit-approval and credit-g have positive BGR-vs-uniform rows but do not
clear the stricter fixed-radius margin. Treat the mixed-suite result as
supervised margin-replay evidence only, not a standard-environment or robotics
win, and do not make a manuscript claim until the additional confirmation
below completes and the claim/checker policy is updated.
Locked adult-only confirmation: an initial job `778905` failed immediately
because the Slurm wrapper quoted the dataset override as one argument. The
wrapper was fixed, and corrected job `778912` ran with `DATASETS=adult`,
`TARGETS=1.5`, `SEED_START=60`, `SEEDS=60`, writing to
`/work/joy/bgr/runs/openml_mixed_adult_target15_confirmation_60seed_v2_778912`
and syncing to
`results/openml_mixed_adult_target15_confirmation_60seed_v2_778912/`. This
third split is weak and rejects adult target-1.5 as new paper evidence:
confirmation BGR is 0.7935 vs. uniform 0.7809 (+0.0126, W/L/T=37/23/0) and
fixed 0.7879 (+0.0055, W/L/T=29/31/0). Pooled over all 120 adult target-1.5
seeds, BGR is 0.7938 vs. uniform 0.7678 (+0.0260, W/L/T=80/40/0) and fixed
0.7748 (+0.0190, W/L/T=67/53/0), below the +0.03 fixed follow-up standard.
Treat the mixed-type OpenML route as a fragility/near-miss diagnostic, not an
acceptance-moving independent benchmark win.
Completed low-priority supervised third-split target-2.0 check: because mixed
credit-approval target radius 2.0 was positive in the original and held-out
30-seed target-sensitivity blocks, a fixed no-retuning third block was queued on
2026-06-11 for `credit-approval,credit-g` at target radius 2.0, seeds 60--89.
This is not the standard-environment or learned-policy evidence gap, and it did
not add a promotable mixed-feature OpenML positive. Athena job `782899`
completed with exit `0:0`, writing to
`results/openml_mixed_credit_target2_thirdsplit_30seed_v1_782899/`. Launch:
`OUT_PREFIX=openml_mixed_credit_target2_thirdsplit_30seed_v1 DATASETS=credit-approval,credit-g TARGETS=2.0 SEED_START=60 SEEDS=30 STEPS=8 BATCH_SIZE=64 CANDIDATE_COUNT=128 EVAL_EXAMPLES=250 TIME_LIMIT=12:00:00 MEMORY=24G CPUS=4 scripts/queue_openml_mixed_binary_suite.sh`.
Credit-approval gives BGR 0.8236, uniform 0.7954, and fixed-radius 0.7998, so
it misses the +0.03 uniform margin at +0.0282. Credit-g gives BGR 0.7022,
uniform 0.6670, and fixed-radius 0.6842, clearing uniform by +0.0352 but fixed
by only +0.0180. Treat this as another mixed-feature supervised
fragility/near-miss diagnostic, not a paper headline or an acceptance-moving
independent benchmark win.

Completed MiniGrid DynamicObstacles independent-benchmark scout: the official
package-owned reset interface in
`tools/minigrid_dynamic_obstacles_recovery_probe.py` uses
`MiniGrid-Dynamic-Obstacles-6x6-v0`, exact restoration of the package
grid/obstacle list, stochastic package obstacle moves, and teacher-action
replay from reset states near the package-generated shortest path. Local
seed-0 scouts were non-promotable, and the fixed 4-seed CPU cluster screen
submitted as Slurm job `779232` completed negative with exit `0:0`, synced to
`results/minigrid_dynamic_obstacles_recovery_probe_4seed_v1_779232/`. Mean
RAUC is failure-only 0.6513, TD-loss 0.6195, uniform 0.6082, fixed 0.6047,
BGR-uniform-radius 0.5923, default BGR 0.5689, and BGR-Coverage 0.5307.
`tools/check_candidate_promotion.py` rejects BGR-Coverage versus uniform
(delta -0.0775, W/L/T=0/4/0), fixed, failure-only, TD-loss, and the
uniform-radius ablation; it also rejects default BGR versus uniform
(delta -0.0394, W/L/T=0/4/0), failure-only, TD-loss, and BGR-uniform-radius,
with a median-r80 contradiction. Treat this fixed DynamicObstacles route as
completed negative, not paper evidence.
Completed DynamicObstacles clean-preservation follow-up: a separate
`bgr_clean_shield` scout tested a new premise after the fixed screen showed
BGR-family clean recovery collapse. The clean-shield rule trains at sigma 0
when selected-state clean success falls below 0.65 and adds a clean anchor
after boundary updates with probability 0.25. The bounded 4-seed common
baseline scout completed on `athena` as Slurm job `779412` and synced to
`results/minigrid_dynamic_obstacles_clean_shield_probe_4seed_v1_779412/`.
It is negative: mean RAUC is failure-only 0.6513, TD-loss 0.6195, uniform
0.6082, fixed 0.6047, BGR-uniform-radius 0.5923, BGR-Clean-Shield 0.5689,
default BGR 0.5689, and BGR-Coverage 0.5307. The candidate checker rejects
BGR-Clean-Shield versus uniform (delta -0.0394, W/L/T=1/3/0), fixed,
failure-only, TD-loss, and BGR-uniform-radius; it also rejects BGR-Coverage and
default BGR. Treat this route as completed negative, not paper evidence.

Latest OpenVLA poll/sync at 2026-06-10 20:31 BST: the hard-occlusion 0.80
transfer route is complete and negative under the fixed gate, and the A6000
0.80 identity-anchored route is now already closed negative on the fixed
identity side condition. The 0.80 transfer summary has BGR identity 391/400,
official identity 393/400, matched-random identity 389/400, BGR occlusion
305/400, official occlusion 296/400, and matched-random occlusion 296/400;
`scripts/check_openvla_perturb_gate.py` reports `[FAIL]` because BGR is only
+9/400 over both comparators on hard occlusion and trails official identity by
two. The A6000 0.80 identity-anchored `summary_available.csv` has complete
identity rows with BGR 389/400, official 393/400, and matched-random 393/400,
so the route cannot pass the fixed gate even though official/BGR occlusion were
still running and matched-random occlusion was priority-pending. The 0.80 micro
A6000 route has official identity complete and BGR identity running; the 0.80
strict A6000 route has official occlusion and BGR identity running; the 0.80
identity-anchor A40, 0.80 micro A40, and 0.90 strict A40 variants remain
pending or dependency-pending. Keep polling/syncing for closure, but do not
promote any partial route. No new same-protocol standard-environment job was
launched: the current scorecard rejects that route family, and the active queue
already covers the current identity-preserving hard-occlusion learned-policy
premises.
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
versus uniform and 7/8 versus fixed-radius. A 60-seed target-radius sensitivity
check over the three positive OpenML datasets is now a fragility caveat: pooled
BGR-minus-uniform gaps for diabetes/blood/phoneme are +0.002/-0.002/-0.007 at
radius 1.0, +0.032/+0.065/+0.014 at 1.5, and +0.038/+0.086/+0.035 at 2.0.
Pooled across 60
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
A completed fixed broad numeric OpenML suite added two more replicated
pre-existing-dataset positives while also sharpening the macro caveat. The
original/held-out broad-suite macro means are uniform 0.7817/0.7800,
fixed-radius 0.7815/0.7845, and BGR 0.7756/0.7820; pooled across both runs,
BGR is ahead on 6/10 dataset means versus uniform and 6/10 versus fixed-radius,
so this is not a broad macro win. The replicated positive dataset rows are:
MagicTelescope original BGR 0.7395 vs. uniform 0.6884 (+0.0511, W/L/T=25/5/0)
and fixed 0.6827, held-out BGR 0.7339 vs. uniform 0.6694 (+0.0645,
W/L/T=26/4/0) and fixed 0.6780; haberman original BGR 0.7258 vs. uniform
0.6155 (+0.1103, W/L/T=29/0/1) and fixed 0.6644, held-out BGR 0.7275 vs.
uniform 0.6327 (+0.0948, W/L/T=27/3/0) and fixed 0.6690. Treat these as
additional supervised margin-replay evidence, not the missing standard-
environment or learned-policy result.
Completed secondary numeric OpenML suite: another predeclared active
version-1 numeric OpenML set was run through the existing median-impute plus
standardized numeric-feature pipeline: kc2, pc2, pc3, pc4, mc1, jm1,
hill-valley, madelon, gina_agnostic, and electricity. Slurm jobs `776728` and
`776729` completed with exit `0:0` and were synced to
`results/openml_secondary_numeric_target2_30seed_v1/` and
`results/openml_secondary_numeric_target2_replication_30seed_v1/`. The suite
is macro-neutral/negative, not a broad win: pooled macro means are BGR 0.7597,
uniform 0.7600, and fixed-radius 0.7616, with BGR ahead on 4/10 dataset means
versus uniform and 4/10 versus fixed. It adds one replicated dataset-level
positive, jm1: original BGR 0.7894 vs. uniform 0.7196 (+0.0698, W/L/T=28/2/0)
and fixed 0.7468 (+0.0426, W/L/T=23/7/0), held-out BGR 0.7888 vs. uniform
0.7081 (+0.0806, W/L/T=26/4/0) and fixed 0.7358 (+0.0529, W/L/T=26/4/0).
kc2 is a near-miss rather than a promoted positive: pooled BGR is +0.0320 vs.
uniform but only +0.0270 vs. fixed, and the original split is below the +0.03
fixed-radius margin. Use
`PYTHONPATH=src:. python3 tools/analyze_openml_margin_suite.py --original results/openml_secondary_numeric_target2_30seed_v1/per_seed.csv --replication results/openml_secondary_numeric_target2_replication_30seed_v1/per_seed.csv`
for the deterministic readout. Treat jm1 as supervised pre-existing-dataset
margin evidence only, not the missing standard-environment or learned-policy
win.
Completed robustness follow-up: jm1/kc2 target-radius sensitivity. Slurm jobs
`776811` and `776812` completed with exit `0:0` and were synced to
`results/openml_secondary_positive_target_sensitivity_30seed_v1/` and
`results/openml_secondary_positive_target_sensitivity_replication_30seed_v1/`.
The result supports jm1 at radii 1.5 and 2.0 but not at 1.0: pooled
BGR-minus-uniform gaps are +0.023/+0.083/+0.075 at radii 1.0/1.5/2.0, with
BGR-minus-fixed +0.025/+0.074/+0.048. kc2 remains a near-miss: pooled
BGR-minus-uniform is +0.008/+0.034/+0.032, but BGR-minus-fixed is only
+0.024/+0.027/+0.027. Treat this as a target-radius caveat for jm1 and a
rejection of kc2 before promotion.
Completed robustness diagnostic: MagicTelescope/haberman target-radius
sensitivity. This is a caveat/robustness check for the two new broad-suite
positives, not a new headline route. Initial jobs `774495`/`774496` failed
immediately because Slurm had `python3` but not `python` on PATH; follow-up
jobs `774509`/`774510` then failed because bare `python3` did not have
`sklearn`. The corrected OpenML-venv jobs `774520` and `774521` completed with
exit `0:0` and were synced to
`results/openml_broad_positive_target_sensitivity_30seed_v1/` and
`results/openml_broad_positive_target_sensitivity_replication_30seed_v1/`.
The result strengthens MagicTelescope/haberman at target radii 1.5 and 2.0 but
keeps a target-1.0 fragility caveat. Pooled across original and held-out seeds:
MagicTelescope BGR-minus-uniform gaps are +0.014/+0.047/+0.058 at radii
1.0/1.5/2.0, with BGR-minus-fixed +0.008/+0.043/+0.056; haberman gaps are
+0.016/+0.089/+0.103 versus uniform and +0.002/+0.054/+0.060 versus fixed.
Use
`PYTHONPATH=src:. python3 tools/analyze_openml_margin_suite.py --original results/openml_broad_positive_target_sensitivity_30seed_v1/per_seed.csv --replication results/openml_broad_positive_target_sensitivity_replication_30seed_v1/per_seed.csv`
for the full readout.
Completed negative pre-existing-dataset route: fixed numeric multiclass OpenML
suite. This was a materially different supervised benchmark route from the
binary OpenML sweeps because it used multiclass OpenML version-1 datasets with
the same median-impute plus standardized numeric-feature perturbation pipeline:
optdigits, pendigits, satimage, segment, letter, vehicle, texture,
mfeat-fourier, mfeat-karhunen, and mfeat-pixel. Original and held-out
target-2.0 30-seed jobs `774591` and `774592` completed with exit `0:0` and
were synced to `results/openml_multiclass_numeric_target2_30seed_v1/` and
`results/openml_multiclass_numeric_target2_replication_30seed_v1/`. The route
is rejected: pooled macro means are BGR 0.6418, uniform 0.6948, and
fixed-radius 0.6837; BGR is ahead on 0/10 dataset means versus uniform and
2/10 versus fixed-radius, with no promotable-like rows. Treat this as a
negative supervised scope result, not manuscript evidence or an active route.
Full readout:
`PYTHONPATH=src:. python3 tools/analyze_openml_margin_suite.py --original results/openml_multiclass_numeric_target2_30seed_v1/per_seed.csv --replication results/openml_multiclass_numeric_target2_replication_30seed_v1/per_seed.csv`.
Completed near-miss/rejected supervised follow-up route: OpenML heart-statlog
fixed target-2.0 seed extension. This was motivated by the broad numeric
suite's only unresolved candidate-like row: original seeds 0--29 gave BGR
0.8075 vs. uniform 0.7662 (+0.0412, W/L/T=21/9/0), but held-out seeds 30--59
gave only 0.7979 vs. 0.7707 (+0.0272, W/L/T=19/11/0), below the +0.03
follow-up screen. The no-retuning seed extension ran on `athena` as Slurm job
`774696` and is synced to
`results/openml_heart_statlog_target2_extension_60seed_v1/`. Extension seeds
60--119 give BGR 0.7987 vs. uniform 0.7733 (+0.0255, W/L/T=41/18/1) and
fixed-radius 0.7656 (+0.0332, W/L/T=44/15/1). Pooled across all 120
heart-statlog seeds, BGR is 0.8007 vs. uniform 0.7709 (+0.0298, W/L/T=81/38/1)
and fixed-radius 0.7605 (+0.0402, W/L/T=88/31/1). Because the extension and
pooled uniform gaps remain just below the +0.03 screen, retire this as a
near-miss supervised route, not paper evidence.
The new grid-margin witness-sensitivity diagnostic improves scope support for the
feasibility-witness assumption but is controlled mechanism evidence, not an
independent-benchmark win.
Completed internal package-free CartPole and MountainCar scouts are also
negative and should not be promoted. The fixed CartPole scout command
`PYTHONPATH=src:. python3 tools/cartpole_recovery_probe.py --out results/cartpole_recovery_probe_4seed_v1`
gives saturated clean success 1.0000 for every method and tightly tied final
RAUC means: BGR-Coverage 0.9026, failure-only 0.9006, TD-loss 0.9001,
uniform 0.8985, fixed 0.8985, and BGR 0.8987. The fixed MountainCar scout
command
`PYTHONPATH=src:. python3 tools/mountaincar_recovery_probe.py --out results/mountaincar_recovery_probe_4seed_v1`
is dominated by fixed-radius replay and has poor clean success: fixed-radius
0.1420 RAUC and 0.5208 clean versus BGR-Coverage 0.0553 RAUC, BGR 0.0532,
uniform 0.0497, and failure-only 0.0653. Treat both as internal diagnostics
only; they do not justify an official-package scale-up without a materially new
premise.

Immediate priority: fix the weaknesses called out by the review, especially the
lack of new independent positive evidence, the negative standard-environment
record, incremental novelty, author-defined metric dependence, and robustness
fragility. Do not just soften language around unchanged results.

Newest active learned-policy route: hard-occlusion 0.80 micro
identity-anchored OpenVLA-OFT adaptation. This was queued after the 0.80
transfer route became a near miss on occlusion (+9/400 over official) but
failed the fixed identity side condition by trailing official 391/400 vs.
393/400. The route reuses the completed hard-occlusion 0.80 identity-anchor
TFDS roots and tests a smaller, more strongly anchored update:
`ADAPT_STEPS=50`, `LR=5e-8`, and `PROXIMAL_ANCHOR_L2=100.0`, with
identity-LoRA, image augmentation, official stats, and fixed identity plus
occlusion fraction 0.80 evaluation over 10 LIBERO-Goal tasks with 40 trials per
task. It is not a relaxed protocol: BGR must beat both official and matched
random by at least 10/400 occlusion episodes and at least 0.02 absolute success
rate while not trailing the best identity comparator by more than one episode.
Submitted on `athena` at 2026-06-10 15:14 BST after canceling an accidental
default blur/brightness/shift perturb submission and replacing it with
identity+occlusion-only evals. BGR train/merge/clean-eval jobs are
`776998`/`777000`/`777001`; matched-random train/merge/clean-eval jobs are
`777003`/`777004`/`777006`; official identity/occlusion eval jobs are
`777037`/`777039`; BGR identity/occlusion eval jobs are `777040`/`777041`;
matched-random identity/occlusion eval jobs are `777042`/`777043`.
Initial `squeue` showed BGR train `776998` and official identity `777037`
priority-pending, with BGR/random merges, clean evals, and dependent perturb
jobs pending. Poll/sync with
`scripts/sync_openvla_oft_hard_occlusion080_identityanchor_micro_results.sh --poll --sync --no-check`.
Do not incorporate this route into `paper/main.tex` unless a complete
`summary.csv` exists and the fixed gate passes.

Latest all-route OpenVLA poll at 2026-06-10 15:25 BST found no new complete
gateable summary. The micro route remains scheduler-limited: BGR train
`776998` and official identity `777037` are priority-pending with estimated
starts at 2026-06-13 14:07:41 BST, all micro merge/clean/perturb children are
dependency-pending, and no logs or compact summaries exist. The 0.80
identity-anchored A6000 route also remains incomplete: random train `776036`
is priority-pending for 2026-06-10 20:08:00 BST, BGR identity `776042` for
2026-06-11 15:23:58 BST, and BGR clean `776035` plus official identity
`776040` for 2026-06-13 14:07:41 BST; the remote perturb-log directory has
only incomplete official-identity output and no summarizable compact row. The
0.80 strict A6000 route has BGR train `776541` and official identity `776548`
priority-pending for 2026-06-13 14:07:41 BST, with children dependency-pending
and no logs. The 0.90 strict A40 route remains earlier in the A40 queue with
prep `776601` and official identity `776611` priority-pending for
2026-06-11 22:02:14 BST. The 0.80 identity-anchor A40 fallback remains
priority/dependency-pending with BGR train `776291` and official identity
`776300` estimated for 2026-06-11 22:02:14 BST. The 0.80 transfer route is
still incomplete because matched-random occlusion `774923` is priority-pending
for 2026-06-11 08:07:00 BST; partial rows are unchanged and already
non-promotable under the fixed gate: BGR identity/occlusion 391/400 and
305/400 versus official 393/400 and 296/400, with matched-random identity
389/400. The 0.65 A6000 adaptation remains incomplete because matched-random
occlusion `774729` is priority-pending, and already fails the identity side
condition and official occlusion margin: BGR identity/occlusion 389/400 and
301/400 versus official 393/400 and 297/400, with matched-random identity
390/400. The 0.65 A40 fallback remains incomplete because replacement BGR
occlusion `775103` and matched-random occlusion `774851` are priority-pending;
partial rows are BGR identity 391/400, official identity 393/400, official
occlusion 295/400, and matched-random identity 392/400, so it also cannot
become a positive result under the identity side condition. No paper claim
should change until a complete `summary.csv` exists and the fixed gate passes.
Latest all-route OpenVLA poll at 2026-06-10 15:33 BST again found no complete
gateable summary. The micro route is unchanged: BGR train `776998` and
official identity `777037` are priority-pending with estimated starts at
2026-06-13 14:07:41 BST, all children are dependency-pending, and no logs or
summaries exist. The 0.80 identity-anchored A6000 route remains incomplete:
matched-random train `776036` is priority-pending for 2026-06-10 20:24:00 BST,
BGR identity `776042` for 2026-06-11 15:23:58 BST, and BGR clean `776035`
plus official identity `776040` for 2026-06-13 14:07:41 BST; only an
incomplete official-identity log exists. The 0.80 strict A6000 route remains
priority/dependency-pending with BGR train `776541` and official identity
`776548` estimated for 2026-06-13 14:07:41 BST, while the 0.90 strict A40 route
and 0.80 identity-anchor A40 fallback remain priority/dependency-pending with
front jobs estimated for 2026-06-11 22:02:14 BST. The 0.80 transfer route now
has completed official identity/occlusion, BGR identity/occlusion, and
matched-random identity rows, but matched-random occlusion `774923` remains
priority-pending for 2026-06-11 08:25:00 BST; partial rows are BGR
identity/occlusion 391/400 and 305/400, official 393/400 and 296/400, and
matched-random identity 389/400, so the route is still incomplete and already
violates the fixed identity side condition. The 0.65 A6000 adaptation now has
completed BGR occlusion 301/400 in addition to official occlusion 297/400 and
all identity rows, but matched-random occlusion `774729` remains
priority-pending; it is still incomplete and already non-promotable because BGR
identity is 389/400 vs. official 393/400. The 0.65 A40 fallback has completed
replacement BGR identity 391/400, official identity 393/400, official
occlusion 295/400, and matched-random identity 392/400; replacement BGR
occlusion `775103` and matched-random occlusion `774851` remain
priority-pending for 2026-06-11 22:02:14 BST, so the route is incomplete and
already fails the identity side condition. No new same-protocol
standard-environment job was launched: the current scorecard still rejects that
route family by saturation, stronger-baseline losses, or state-priority
ablation failures, and the only acceptance-moving active work remains the
pending hard-occlusion identity-anchored OpenVLA routes.
Latest all-route OpenVLA poll at 2026-06-10 15:36 BST showed no material
change from the 15:33 checkpoint and no complete gateable summary. Micro
`776998`/`777037`, 0.80 strict A6000 `776541`/`776548`, 0.90 strict A40
`776601`/`776611`, and 0.80 identity-anchor A40 `776291`/`776300` remain
priority/dependency-pending with the same estimated front-job starts. The main
0.80 identity-anchor route remains incomplete with matched-random train
`776036` priority-pending for 2026-06-10 20:24:00 BST, BGR identity `776042`
for 2026-06-11 15:23:58 BST, and official identity `776040` for
2026-06-13 14:07:41 BST; only incomplete official-identity output exists. The
0.80 transfer route is still missing matched-random occlusion `774923`, and
the 0.65 A6000 route is still missing matched-random occlusion `774729`; both
remain incomplete and already violate the fixed identity side condition. The
0.65 A40 fallback is still missing replacement BGR occlusion `775103` and
matched-random occlusion `774851`, and already violates the identity side
condition. Explicit partial gate checks on the three available
`summary_available.csv` files returned `[INCOMPLETE]`.
Latest all-route OpenVLA poll at 2026-06-10 15:53 BST synced one complete
negative route and otherwise found no new gateable positive result. The fixed
0.65 hard-occlusion transfer route is complete at
`results/openvla_oft_perturb_eval_occlusion_bottleneck_hardocc065_transfer_step50400_lr2em7_v1/summary.csv`;
the fixed gate fails with BGR identity/occlusion 391/400 and 300/400, official
393/400 and 297/400, and matched random 389/400 and 296/400. The 0.80
transfer route remains incomplete because matched-random occlusion `774923` is
priority-pending with an estimated start of 2026-06-11 08:44:00 BST; partial
rows are BGR identity/occlusion 391/400 and 305/400, official 393/400 and
296/400, and matched-random identity 389/400, so it already violates the fixed
identity side condition. The 0.65 A6000 adaptation remains incomplete because
matched-random occlusion `774729` is priority-pending; available rows are BGR
identity/occlusion 389/400 and 301/400, official identity/occlusion 393/400
and 297/400, and matched-random identity 390/400, so it is already
non-promotable. The 0.65 A40 fallback remains incomplete because replacement
BGR occlusion `775103` and matched-random occlusion `774851` are
priority-pending, and it already has BGR identity 391/400 versus official
393/400 and matched-random 392/400. The 0.80 micro A6000 route remains
priority/dependency-pending with front jobs `776998` and `777037` estimated
for 2026-06-13 14:07:41 BST; the 0.80 micro A40 fallback front jobs `777254`
and `777264`, the 0.80 identity-anchor A40 fallback front jobs `776291` and
`776300`, and the 0.90 strict A40 front jobs `776601` and `776611` are
estimated for 2026-06-11 22:02:14 BST. The main 0.80 identity-anchor A6000
route remains mixed-priority/dependency pending, with random train `776036`
estimated for 2026-06-10 20:44:00 BST and official/BGR identity rows delayed.
No paper claim should change.
Latest all-route OpenVLA poll at 2026-06-10 16:02 BST again found no complete
gateable positive result. The 0.80 transfer route is still incomplete because
matched-random occlusion `774923` is priority-pending with estimated start
2026-06-11 08:44:00 BST; available rows are BGR identity/occlusion 391/400 and
305/400, official 393/400 and 296/400, and matched-random identity 389/400,
so it already violates the fixed identity side condition despite the partial
+9/400 occlusion edge over official. The 0.65 A6000 adaptation is incomplete
because matched-random occlusion `774729` is priority-pending; available rows
are BGR identity/occlusion 389/400 and 301/400, official 393/400 and 297/400,
and matched-random identity 390/400, so it is already non-promotable. The 0.65
A40 fallback remains incomplete because replacement BGR occlusion `775103` and
matched-random occlusion `774851` are priority-pending, while available
identity rows already put BGR at 391/400 behind official 393/400 and matched
random 392/400. The 0.80 micro A40 fallback, 0.80 identity-anchor A40 fallback,
and 0.90 strict A40 route are still priority/dependency-pending with front jobs
estimated for 2026-06-11 22:02:14 BST and no logs or summaries. The A6000
micro and strict routes remain priority/dependency-pending with front jobs
estimated for 2026-06-13 14:07:41 BST; the main 0.80 identity-anchor A6000
route has random train `776036` estimated for 2026-06-10 20:00:00 BST and
official/BGR identity work delayed to 2026-06-13 14:07:41 BST. Explicit gate
checks on the available partial summaries returned `[INCOMPLETE]`. No new
same-protocol standard-environment or duplicate OpenVLA job was launched in
this checkpoint: the existing scorecard already rejects the obvious
standard-environment route family, and the active queue already covers the
current identity-preserving hard-occlusion learned-policy premises. The next
empirical addition should be a materially different external reset interface or
a genuinely different learned-policy intervention, not another minor rerun of
the exhausted recipes.
Latest all-route OpenVLA poll at 2026-06-10 16:10--16:11 BST still found no
complete gateable positive result. The 0.80 transfer route remains incomplete
because matched-random occlusion `774923` is priority-pending with estimated
start 2026-06-11 08:06:00 BST; available rows remain BGR identity/occlusion
391/400 and 305/400, official 393/400 and 296/400, and matched-random identity
389/400, so identity already blocks promotion. The 0.65 A6000 adaptation is
incomplete because matched-random occlusion `774729` is priority-pending; its
available BGR identity/occlusion rows are 389/400 and 301/400 versus official
393/400 and 297/400 and matched-random identity 390/400, so it is already
non-promotable. The 0.65 A40 fallback remains incomplete because replacement
BGR occlusion `775103` and matched-random occlusion `774851` are
priority-pending, while BGR identity 391/400 trails official 393/400 and
matched random 392/400. The 0.80 identity-anchor A6000 route remains
priority/dependency-pending, with random train `776036` estimated for
2026-06-10 20:00:00 BST and official/BGR identity work estimated for
2026-06-13 14:07:41 BST. The A40 identity-anchor, A40 micro, and 0.90 strict
A40 routes remain priority/dependency-pending with front jobs estimated for
2026-06-11 22:02:14 BST, and the A6000 micro/strict routes remain
priority/dependency-pending with front jobs estimated for
2026-06-13 14:07:41 BST. No OpenVLA paper claim should change.
Latest all-route OpenVLA poll at 2026-06-10 17:36 BST still found no complete
gateable summary. The 0.80 transfer route remains incomplete because
matched-random occlusion `774923` is priority-pending with estimated start
2026-06-11 12:24:00 BST; available rows remain insufficient and identity
already blocks promotion. The 0.80 identity-anchor A6000 route is still
priority/dependency-pending, with random train `776036` estimated for
2026-06-11 00:22:00 BST, BGR identity `776042` for 2026-06-11 00:25:00 BST,
and official identity `776040` for 2026-06-11 12:28:00 BST. The 0.80
identity-anchor A40 fallback, A40 micro fallback, and 0.90 strict A40 route are
also priority/dependency-pending with front jobs estimated for 2026-06-11
22:02:14 BST. The A6000 strict and A6000 micro routes remain
priority/dependency-pending with front jobs estimated for 2026-06-11
14:21:02--15:23:58 BST and later perturb jobs still dependency-pending. No
OpenVLA paper claim should change.
Latest all-route OpenVLA poll at 2026-06-10 18:05 BST still found no complete
gateable summary. The hard-occlusion 0.80 transfer route synced an incomplete
`summary_available.csv`: official identity/occlusion and BGR
identity/occlusion are complete, matched-random identity is complete, but
matched-random occlusion `774923` remains priority-pending with estimated start
2026-06-11 10:04 BST. The 0.80 identity-anchor A6000 route remains
priority/dependency-pending, with random train `776036`, BGR identity `776042`,
and official identity `776040` now estimated around 2026-06-10 22:56--23:04
BST; downstream occlusion rows are dependency-pending. The 0.80 identity-anchor
A40 fallback and 0.90 strict A40 route remain priority/dependency-pending with
front jobs estimated for 2026-06-11 22:02:14 BST. No new same-protocol OpenVLA
job was launched in this checkpoint because the active queue already covers the
current identity-preserving hard-occlusion premises.
Latest all-route OpenVLA poll at 2026-06-10 18:38 BST still found no complete
gateable summary. The hard-occlusion 0.80 transfer route remains incomplete:
matched-random occlusion `774923` is priority-pending with estimated start
2026-06-11 08:25 BST, while available identity rows already block promotion.
The 0.80 identity-anchor A6000 route has front jobs now estimated around
2026-06-10 20:18--20:22 BST, but all perturb summaries are still missing. The
0.80 micro A6000 route has BGR train `776998` estimated for
2026-06-11 07:35 BST and official identity `777037` for 08:22 BST; the 0.80
strict A6000 route has BGR train `776541` estimated for 07:32 BST and official
identity `776548` for 07:35 BST. The 0.80 identity-anchor A40 fallback, 0.80
micro A40 fallback, and 0.90 strict A40 route remain priority/dependency
pending with front jobs estimated for 2026-06-11 22:02:14 BST. No OpenVLA
paper claim should change.
Latest all-route OpenVLA poll/sync at 2026-06-10 20:39 BST still found no
complete gateable hard-occlusion summary. The 0.80 identity-anchor A6000 route
has identity rows complete locally at BGR 389/400, official 393/400, and
matched random 393/400, so it is already non-promotable on the fixed identity
side condition; its official and BGR occlusion eval jobs `776041`/`776043` are
running, and matched-random occlusion `776045` is priority-pending with
estimated start 2026-06-11 07:15:44 BST. The 0.80 micro A6000 route has only
official identity complete at 393/400; BGR identity `777040` is running,
official occlusion `777039` and random identity `777042` are priority-pending,
and BGR/random occlusion jobs are dependency-pending. The 0.80 strict A6000
route has official identity complete at 393/400, official occlusion `776549`,
BGR identity `776550`, and random identity `776553` running, with BGR/random
occlusion dependency-pending. The 0.80 A40 fallback has official identity
`776300` running while its BGR/random adaptation chains are still
priority/dependency-pending; the 0.80 micro A40 fallback and 0.90 strict A40
route remain pending without local logs or summaries. No new same-premise
OpenVLA job was launched because the active queue already covers the
identity-preserving hard-occlusion variants; do not incorporate any of these
routes into `paper/main.tex` unless a complete `summary.csv` exists and the
fixed gate passes.

Newest acceleration route: hard-occlusion 0.80 micro identity-anchored
OpenVLA-OFT A40 fallback. This is a resource fallback for the already fixed
micro premise, not a new claim: it reuses the completed hard-occlusion 0.80
identity-anchor TFDS roots but requests A40 GPUs under separate artifact tags
with the same `ADAPT_STEPS=50`, `LR=5e-8`, `PROXIMAL_ANCHOR_L2=100.0`,
identity-LoRA, image augmentation, official stats, and identity plus occlusion
fraction 0.80 evaluation over 10 LIBERO-Goal tasks x 40 trials. Submitted on
`athena` at 2026-06-10 15:47 BST: BGR train/merge/clean-eval
`777254`/`777255`/`777256`; matched-random train/merge/clean-eval
`777257`/`777259`/`777261`; official identity/occlusion `777264`/`777265`;
BGR identity/occlusion `777266`/`777268`; matched-random identity/occlusion
`777269`/`777270`. Initial poll showed BGR train `777254` and official
identity `777264` priority-pending with estimated starts at
2026-06-11 22:02:14 BST, all dependent jobs pending, and no remote logs or
compact summary yet. Poll/sync with
`scripts/sync_openvla_oft_hard_occlusion080_identityanchor_micro_a40_results.sh --poll --sync --no-check`.
Do not incorporate this route into `paper/main.tex` unless a complete
`summary.csv` exists and the fixed +10/400, +0.02, and identity-side-condition
gate passes.

Newest active learned-policy route: hard-occlusion 0.90 strict
identity-anchored OpenVLA-OFT adaptation. This was queued after the 0.65
transfer gate completed negative and the 0.80 transfer route showed a near-miss
occlusion gain with an identity deficit. The route trains on hard occlusion
fraction 0.90 with a strict identity-preservation update (`ADAPT_STEPS=100`,
`LR=5e-8`, `PROXIMAL_ANCHOR_L2=50.0`) and uses the unchanged promotion gate:
BGR must beat both official and matched random by at least 10/400 occlusion
episodes and at least 0.02 absolute success rate while not trailing the best
identity comparator by more than one episode. Submitted on `athena` at
2026-06-10 14:18 BST: prep `776601`; BGR train/merge/clean-eval
`776602`/`776603`/`776604`; matched-random train/merge/clean-eval
`776605`/`776606`/`776607`; official identity/occlusion `776611`/`776613`;
BGR identity/occlusion `776615`/`776616`; matched-random identity/occlusion
`776617`/`776619`. Initial poll showed prep `776601` and official identity
`776611` priority-pending with estimated starts at 2026-06-11 22:02:14 BST,
all other jobs dependency-pending, and no remote logs or compact summaries.
Latest poll at 2026-06-10 14:22:28 BST showed the same substantive state:
prep `776601` and official identity `776611` were still priority-pending with
estimated starts at 2026-06-11 22:02:14 BST, all adaptation/eval children were
dependency-pending, and no remote logs or compact summaries existed.
Poll/sync with
`scripts/sync_openvla_oft_hard_occlusion090_identityanchor_strict_results.sh --poll --sync --no-check`.
Do not incorporate this route into `paper/main.tex` unless a complete
`summary.csv` exists and the fixed gate passes.

Active companion learned-policy route: hard-occlusion 0.80 identity-anchored
OpenVLA-OFT adaptation. This was queued after the 0.80 transfer and A6000
hard-occlusion adaptation partial identity rows already made those routes
non-promotable under the fixed clean-identity side condition. The route trains
on hard occlusion fraction 0.80 with a shorter, more strongly anchored update
(`ADAPT_STEPS=200`, `LR=1e-7`, `PROXIMAL_ANCHOR_L2=20.0`) and uses the same
promotion gate. It is not paper evidence unless `summary.csv` exists and the
fixed gate passes. Poll/sync with
`scripts/sync_openvla_oft_hard_occlusion080_identityanchor_results.sh --poll --sync --no-check`.
Latest poll/sync at 2026-06-10 14:22:11 BST showed this A6000 0.80
identity-anchored route still scheduler-limited rather than gateable:
official identity `776040` and BGR identity `776042` were priority-pending
with estimated starts at 2026-06-11 17:55:00 BST, official/BGR occlusion and
all random perturb evals were dependency-pending, and no compact summary
exists. The same-protocol A40 fallback remained earlier in the queue: BGR train
`776291` and official identity `776300` were priority-pending with estimated
starts at 2026-06-11 22:02:14 BST, all downstream A40 jobs were
dependency-pending, and no remote logs or summary existed. The 0.80 transfer
route also remains incomplete: matched-random occlusion `774923` is still
priority-pending, while BGR already has only a +9/400 occlusion edge over
official and a 2-episode identity deficit. Do not incorporate either 0.80
identity-anchored route into
`paper/main.tex` unless a complete `summary.csv` exists and the fixed
identity/occlusion gate passes.

Latest hard-occlusion status at 2026-06-10 14:22:28 BST also closes two
nearby learned-policy routes as non-promotable even though their full summaries
are incomplete. The A6000 0.65 adaptation route has BGR identity 389/400,
official identity 393/400, matched-random identity 390/400, BGR occlusion
301/400, and official occlusion 297/400; matched-random occlusion `774729`
is still priority-pending. It already violates the fixed identity side
condition and beats official occlusion by only +4/400. The A40 0.65 fallback
has replacement BGR identity 391/400, official identity 393/400, and
matched-random identity 392/400, so it also fails the fixed identity side
condition by trailing the best identity comparator by two episodes; official
occlusion `774847` completed at 295/400, replacement BGR occlusion `775103` and
matched-random occlusion `774851` were priority-pending, and the original
failed BGR child `774849` remains ignored. Treat both as negative/incomplete
scope information, not paper evidence.

Completed learned-policy diagnostic route (negative): fixed hard-occlusion transfer eval of
the completed OpenVLA-OFT occlusion-bottleneck checkpoints. This is a
preregistered diagnostic/route scout, not a paper claim: the previous full
perturbation gate failed because blur, brightness, and shift were near
saturated while occlusion was the only clear bottleneck. The fixed evaluation
uses identity plus occlusion fraction 0.65 on 10 LIBERO-Goal tasks with 40
trials per task, comparing official, BGR, and matched random. Promotion would
require BGR to beat both official and matched random by at least 10/400
occlusion episodes and at least 0.02 absolute success rate while not trailing
the best comparator identity count by more than 1 episode. Submitted on
`athena` with
`EVAL_ARTIFACT=openvla_oft_perturb_eval_occlusion_bottleneck_hardocc065_transfer_step50400_lr2em7_v1`,
`PERTURBATIONS='identity={};occlusion={"fraction":0.65}'`, `EVAL_TASKS=10`,
`EVAL_TRIALS=40`, `REMOTE_RUN_ROOT=/work/joy/bgr/runs`,
`REMOTE_HF_HOME=/work/joy/cache_home/huggingface`,
`OPENVLA_OFT_ROOT=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft`,
and `LIBERO_ROOT=/work/joy/external_validation/openvla_oft_smoke_746850/LIBERO`.
Slurm jobs are official identity/occlusion `774711`/`774712`, BGR
identity/occlusion `774713`/`774714`, and matched-random identity/occlusion
`774715`/`774716`. Initial `squeue` showed identity jobs `774711`, `774713`,
and `774715` running, with occlusion jobs dependency-pending. Poll/sync with
`scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh --poll --no-check`
and, when logs or summaries exist,
`scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh --sync`.
Latest poll at 2026-06-10 09:45:12 BST showed identity jobs `774711`,
`774713`, and `774715` still running at about 20:30 elapsed; occlusion jobs
`774712`, `774714`, and `774716` remained dependency-pending behind the
corresponding identity jobs. Remote identity logs exist, but
`summary.csv` is still missing, so no gate can be run yet.
Latest poll at 2026-06-10 10:04:47 BST showed the same route still in
identity-eval progress: `774711`, `774713`, and `774715` running at about
40:05 elapsed on A6000 nodes, with `774712`, `774714`, and `774716`
dependency-pending behind them. Remote log tails are healthy rather than stuck:
BGR identity had reached 236/244 successes, official identity 215/222, and
matched-random identity 230/240. The compact `summary.csv` is still missing,
and the partial logs are incomplete by construction, so the fixed hard-occlusion
gate cannot be run yet.
Latest poll at 2026-06-10 10:14:46 BST showed the same 0.65 transfer route
still incomplete: identity jobs `774711`, `774713`, and `774715` were running
at about 50:04 elapsed, and occlusion jobs `774712`, `774714`, and `774716`
remained dependency-pending. Remote identity logs exist but the compact
`summary.csv` is still missing, so no fixed hard-occlusion gate can be run.
Latest poll/sync at 2026-06-10 10:19:18 BST showed the 0.65 transfer route
still in identity-only progress: `774711`, `774713`, and `774715` were running
at about 54:36 elapsed, while occlusion jobs `774712`, `774714`, and `774716`
were still dependency-pending. The sync found only identity logs and could not
build a compact summary yet, so the route remains incomplete.
Latest poll/sync at 2026-06-10 10:21:07 BST still had the 0.65 transfer route
in identity-only progress: `774711`, `774713`, and `774715` were running at
about 56:25 elapsed, and occlusion jobs `774712`, `774714`, and `774716`
remained dependency-pending. Log tails looked healthy: BGR identity had reached
370/378 successes, official identity 328/335, and matched random identity
365/375. No compact summary exists and no gate can be run yet.
Latest poll/sync at 2026-06-10 10:23:53 BST still had the 0.65 transfer route
in identity-only progress: `774711`, `774713`, and `774715` were running at
about 59:11 elapsed, and occlusion jobs `774712`, `774714`, and `774716`
remained dependency-pending. Log tails showed BGR identity 383/392, official
identity 348/355, and matched-random identity 378/389. No compact summary
exists yet. A transient rsync drop during the incomplete-log sync left no
evidence artifact after cleanup; the sync helper now treats rsync drops and
incomplete logs as pending and removes temporary paths.
Latest poll/sync at 2026-06-10 10:30:56 BST showed the 0.65 transfer route
past two identity rows but still incomplete: BGR identity `774713` completed
with 391/400 successes and matched-random identity `774715` completed with
389/400 successes. Official identity `774711` was still marked running, though
the log tail had reached 393/400 successes. BGR and matched-random occlusion
jobs `774714` and `774716` had just started on `c2-g4-23`; official occlusion
`774712` remained dependency-pending. Direct log tails showed the first BGR and
matched-random hard-occlusion episodes were failures, which is not yet a
summary. The compact `summary.csv` is still missing, the fixed gate remains
incomplete, and readiness is unchanged.
Latest poll/sync at 2026-06-10 10:35:15 BST showed all 0.65 transfer identity
rows completed: official identity `774711` completed with 393/400 successes,
BGR identity `774713` with 391/400, and matched-random identity `774715` with
389/400. All three hard-occlusion jobs were running: official `774712` on
`c1-g4-03`, and BGR/random `774714`/`774716` on `c2-g4-23`. Direct log tails
had early occlusion partials only: official 9/11, BGR 16/18, and matched
random 16/18. The remote compact `summary.csv` is still missing, the fixed
gate remains incomplete, and the route is not paper evidence.
Latest poll/sync at 2026-06-10 10:43:45 BST showed the 0.65 transfer route
still in hard-occlusion progress: official `774712` was running at 11:31,
BGR `774714` at 13:26, and matched random `774716` at 13:26. The local
`summary_available.csv` has only completed identity rows so far: official
393/400, BGR 391/400, and matched random 389/400. Direct hard-occlusion tails
were early and not gateable: official 48/56, BGR 47/59, and matched random
49/59. The remote compact `summary.csv` is still missing, so no fixed gate can
be run and no paper claim changes.
Latest poll/sync at 2026-06-10 10:47:48 BST showed the 0.65 transfer route
still running all three hard-occlusion evals: official `774712` at 15:34,
BGR `774714` at 17:29, and matched random `774716` at 17:29. Direct tails were
still partial and not gateable: official 60/74, BGR 58/76, and matched random
59/76. The remote compact `summary.csv` is still missing, and no paper claim
changes.
Latest poll/sync at 2026-06-10 10:54:30 BST showed the 0.65 transfer route
still running all three hard-occlusion evals: official `774712` at 22:16 on
`c1-g4-03`, BGR `774714` at 24:11 on `c2-g4-23`, and matched random `774716`
at 24:11 on `c2-g4-23`. The synced `summary_available.csv` still contains
only completed identity rows: official 393/400, BGR 391/400, and matched
random 389/400. Direct hard-occlusion tails were partial and not gateable:
official 73/101, BGR 73/104, and matched random 74/105. The compact
`summary.csv` is still missing, so no fixed gate can be run and no paper claim
changes.
Latest poll/sync at 2026-06-10 11:01:16 BST showed the 0.65 transfer route
still running all three hard-occlusion evals: official `774712` at 29:02, BGR
`774714` at 30:57, and matched random `774716` at 30:57. The remote
`summary.csv` was still missing; the live rsync of logs dropped with code
12/255 and was treated as pending. Direct hard-occlusion tails were still
partial and not gateable: official 79/122, BGR 83/125, and matched random
82/125. The fixed gate remains incomplete and no paper claim changes.
Latest poll/sync at 2026-06-10 11:03:34 BST showed the 0.65 transfer route
still running all three hard-occlusion evals: official `774712` at 31:20, BGR
`774714` at 33:15, and matched random `774716` at 33:15. The helper synced
only an incomplete `summary_available.csv`, still with identity rows only.
Direct hard-occlusion tails were partial and not gateable: official 79/129,
BGR 84/132, and matched random 83/132. The compact `summary.csv` is still
missing, the fixed gate remains incomplete, and no paper claim changes.
Latest poll/sync at 2026-06-10 11:09:06 BST still had the 0.65 transfer route
in hard-occlusion progress: official `774712` was running at 36:52 on
`c1-g4-03`, while BGR `774714` and matched random `774716` were running at
38:47 on `c2-g4-23`. The synced `summary_available.csv` still contains only
identity rows: official 393/400, BGR 391/400, and matched random 389/400.
Direct hard-occlusion tails were still partial and not gateable: official
83/146, BGR 86/148, and matched random 84/148. The compact `summary.csv` is
still missing, the fixed gate remains incomplete, and no paper claim changes.
Latest poll/sync at 2026-06-10 11:12:56 BST still had the 0.65 transfer route
in hard-occlusion progress: official `774712` was running at 40:42, BGR
`774714` at 42:37, and matched random `774716` at 42:37. The synced
`summary_available.csv` remained identity-only. Direct hard-occlusion tails
were partial and not gateable: official 83/158, BGR 86/158, and matched random
84/158. The compact `summary.csv` is still missing, the fixed gate remains
incomplete, and no paper claim changes.
Latest poll/sync at 2026-06-10 11:17:49 BST still had the 0.65 transfer route
in hard-occlusion progress: official `774712` was running at 45:35, BGR
`774714` at 47:30, and matched random `774716` at 47:30. The remote
`summary.csv` was missing and a live-log rsync dropped with code 12/255, so the
local `summary_available.csv` remained identity-only. A direct log-tail check
afterward showed partial hard-occlusion tails official 110/188, BGR 111/187,
and matched random 109/187. These partials are not gateable; the fixed gate
remains incomplete and no paper claim changes.
Latest poll/sync at 2026-06-10 11:24:29 BST still had the 0.65 transfer route
in hard-occlusion progress: official `774712` was running at 52:15, BGR
`774714` at 54:10, and matched random `774716` at 54:10. The helper synced
only an identity-row `summary_available.csv` with official 393/400, BGR
391/400, and matched random 389/400; the compact `summary.csv` was still
missing. Direct hard-occlusion tails were partial and not gateable: official
133/222, BGR 134/220, and matched random 132/221. No paper claim changes.
Latest poll/sync at 2026-06-10 11:28:56 BST still had the 0.65 transfer route
in hard-occlusion progress: official `774712` was running at 56:42, BGR
`774714` at 58:37, and matched random `774716` at 58:37. The helper again
synced only an identity-row `summary_available.csv`; the compact `summary.csv`
was missing. Direct tails remained partial and not gateable: official 145/241,
BGR 147/238, and matched random 143/238.
Latest poll/sync at 2026-06-10 11:33:01 BST still had the 0.65 transfer route
in hard-occlusion progress: official `774712` was running at 1:00:47, BGR
`774714` at 1:02:42, and matched random `774716` at 1:02:42. The compact
`summary.csv` was still missing; the synced `summary_available.csv` remained
identity-only. No fixed gate can be run.
Latest poll/sync at 2026-06-10 11:36:17 BST still had the 0.65 transfer route
in hard-occlusion progress: official `774712` was running at 1:04:03, BGR
`774714` at 1:05:58, and matched random `774716` at 1:05:58. The compact
`summary.csv` was still missing; the synced `summary_available.csv` remained
identity-only. No fixed gate can be run.
Latest poll/sync at 2026-06-10 11:39:18 BST still had the 0.65 transfer route
in hard-occlusion progress: official `774712` was running at 1:07:04, BGR
`774714` at 1:08:59, and matched random `774716` at 1:08:59. The compact
`summary.csv` was still missing; the synced `summary_available.csv` remained
identity-only. No fixed gate can be run.
Latest poll/sync at 2026-06-10 11:42:32 BST still had the 0.65 transfer route
in hard-occlusion progress: official `774712` was running at 1:10:18, BGR
`774714` at 1:12:13, and matched random `774716` at 1:12:13. The compact
`summary.csv` was still missing; the synced `summary_available.csv` remained
identity-only. No fixed gate can be run.
Latest poll/sync at 2026-06-10 11:46:00 BST still had the 0.65 transfer route
in hard-occlusion progress: official `774712` was running at 1:13:46 on
`c1-g4-03`, while BGR `774714` and matched random `774716` were both running
at 1:15:41 on `c2-g4-23`. The compact `summary.csv` was still missing; the
synced `summary_available.csv` remained identity-only with official 393/400,
BGR 391/400, and matched random 389/400. No fixed gate can be run.
Latest poll/sync at 2026-06-10 11:50:24 BST still had the 0.65 transfer route
in hard-occlusion progress: official `774712` was running at 1:18:10, while
BGR `774714` and matched random `774716` were both running at 1:20:05. The
compact `summary.csv` was still missing and the synced `summary_available.csv`
remained identity-only. No fixed gate can be run.
Latest poll/sync at 2026-06-10 11:54:13 BST still had the 0.65 transfer route
in hard-occlusion progress: official `774712` was running at 1:21:59, while
BGR `774714` and matched random `774716` were both running at 1:23:54. The
compact `summary.csv` was still missing and the synced `summary_available.csv`
remained identity-only. No fixed gate can be run.
Latest poll/sync at 2026-06-10 11:57:34 BST completed the 0.65 transfer route:
official/BGR/matched-random occlusion jobs `774712`/`774714`/`774716` all
completed with exit `0:0`, and the compact
`results/openvla_oft_perturb_eval_occlusion_bottleneck_hardocc065_transfer_step50400_lr2em7_v1/summary.csv`
now exists. The fixed gate fails: BGR is 300/400 on hard occlusion, official is
297/400, and matched random is 296/400, so BGR leads by only +3/+4 episodes
and +0.0075/+0.0100 success rate; identity is BGR 391/400, official 393/400,
matched random 389/400, so BGR also has a 2-episode identity deficit. This is
a completed negative/non-promotable learned-policy transfer diagnostic, not a
paper claim.

Active learned-policy diagnostic route: fixed hard-occlusion 0.80 transfer
eval of the completed OpenVLA-OFT occlusion-bottleneck checkpoints. This was
queued before any 0.65 occlusion rows were available to test whether a harder
visual bottleneck avoids saturation. It is a diagnostic/route scout, not a
paper claim. The fixed evaluation uses identity plus occlusion fraction 0.80 on
10 LIBERO-Goal tasks with 40 trials per task, comparing official, BGR, and
matched random. Promotion uses the same hard-occlusion gate: BGR must beat both
official and matched random by at least 10/400 occlusion episodes and at least
0.02 absolute success rate while not trailing the best identity comparator by
more than 1 episode. Submitted with
`EVAL_ARTIFACT=openvla_oft_perturb_eval_occlusion_bottleneck_hardocc080_transfer_step50400_lr2em7_v1`
and `PERTURBATIONS='identity={};occlusion={"fraction":0.8}'`. Slurm jobs are
official identity/occlusion `774917`/`774919`, BGR identity/occlusion
`774920`/`774921`, and matched-random identity/occlusion `774922`/`774923`.
Initial poll at 2026-06-10 10:14:59 BST showed official identity `774917`
running on `c2-g4-17` at 4:43 elapsed, BGR identity `774920` pending on
resources with estimated start 2026-06-10 12:56:05 BST, matched-random identity
`774922` pending on priority with estimated start 2026-06-10 22:13:17 BST, and
all occlusion jobs dependency-pending. Poll/sync with
`scripts/sync_openvla_oft_hard_occlusion080_transfer_results.sh --poll --no-check`
and, when logs or summaries exist,
`scripts/sync_openvla_oft_hard_occlusion080_transfer_results.sh --sync`.
Latest poll/sync at 2026-06-10 10:19:18 BST showed official identity `774917`
running at 9:02 elapsed and BGR identity `774920` running at 0:31 elapsed,
both on `c2-g4-17`; matched-random identity `774922` remained priority-pending
with estimated start 2026-06-10 12:56:05 BST, and occlusion jobs `774919`,
`774921`, and `774923` remained dependency-pending. The sync found only
identity logs and could not build a compact summary yet.
Latest poll/sync at 2026-06-10 10:21:07 BST showed the 0.80 transfer route
still incomplete but moving: official identity `774917` was running at 10:51
elapsed, BGR identity `774920` at 2:20 elapsed, matched-random identity
`774922` was still priority-pending with estimated start 2026-06-10 12:56:05
BST, and all occlusion jobs remained dependency-pending. Log tails showed
official identity 80/84 and BGR identity 12/12 so far. No compact summary
exists yet.
Latest poll/sync at 2026-06-10 10:23:53 BST showed the 0.80 transfer route
still incomplete: official identity `774917` was running at 13:37 elapsed, BGR
identity `774920` at 5:06 elapsed, matched-random identity `774922` was still
priority-pending with estimated start 2026-06-10 22:18:47 BST, and all
occlusion jobs remained dependency-pending. Log tails showed official identity
101/107 and BGR identity 32/32. No compact summary exists yet.
Latest poll/sync at 2026-06-10 10:30:56 BST showed the 0.80 transfer route
moving but still identity-only: official identity `774917` was running at
20:40, BGR identity `774920` at 12:09, and matched-random identity `774922`
had started on `c2-g8-07` at 0:25. Occlusion jobs `774919`, `774921`, and
`774923` remained dependency-pending. Log tails showed official identity
150/157, BGR identity 99/104, and matched-random identity 2/2 so far. No
compact summary exists and no gate can be run.
Latest poll/sync at 2026-06-10 10:35:15 BST showed the 0.80 transfer route
still identity-only and incomplete: official identity `774917` was running at
24:59, BGR identity `774920` at 16:28, matched-random identity `774922` was
pending on `BeginTime`, and all occlusion jobs remained dependency-pending.
Direct tails showed official identity 180/187, BGR identity 124/129, and
matched-random identity 9/9. No compact summary exists and no gate can be run.
Latest poll/sync at 2026-06-10 10:41:25 BST showed the 0.80 transfer route
still identity-only: official identity `774917` was running at 31:09, BGR
identity `774920` at 22:38, and matched-random identity `774922` at 5:36.
Occlusion jobs `774919`, `774921`, and `774923` remained dependency-pending.
No compact summary exists and no gate can be run.
Latest poll/sync at 2026-06-10 10:47:48 BST showed the 0.80 transfer route
still identity-only: official identity `774917` was running at 37:32, BGR
identity `774920` at 29:01, and matched-random identity `774922` at 11:59.
Occlusion jobs `774919`, `774921`, and `774923` remained dependency-pending.
Direct tails showed official identity 293/300, BGR identity 214/222, and
matched-random identity 93/97 in the active log. No compact summary exists and
no gate can be run.
Latest poll/sync at 2026-06-10 10:54:30 BST showed the 0.80 transfer route
still identity-only: official identity `774917` was running at 44:14, BGR
identity `774920` at 35:43, and matched-random identity `774922` at 18:41.
Occlusion jobs `774919`, `774921`, and `774923` remained dependency-pending.
Direct tails showed official identity 366/373, BGR identity 280/288, and the
active matched-random identity log 140/149; an earlier matched-random identity
log remains stale at 9/9. No compact summary exists and no gate can be run.
Latest poll/sync at 2026-06-10 11:01:16 BST showed the 0.80 transfer route
past official identity but still incomplete: official identity `774917`
completed successfully after 49:06, official occlusion `774919` started and
was running at 1:52, BGR identity `774920` was running at 42:29, and
matched-random identity `774922` was running at 25:27. BGR and matched-random
occlusion jobs `774921`/`774923` were dependency-pending. Direct tails showed
official identity complete at 393/400, official occlusion 6/9, BGR identity
349/357, and active matched-random identity 194/204. No compact summary exists
and no gate can be run.
Latest poll/sync at 2026-06-10 11:03:34 BST showed the 0.80 transfer route
still incomplete: official occlusion `774919` was running at 4:10, BGR
identity `774920` at 44:47, and matched-random identity `774922` at 27:45;
BGR and matched-random occlusion jobs `774921`/`774923` were still
dependency-pending. The helper synced an incomplete `summary_available.csv`
with only official identity complete at 393/400. Direct tails were official
occlusion 18/23, BGR identity 367/375, and active matched-random identity
215/225. No compact summary exists and no gate can be run.
Latest poll/sync at 2026-06-10 11:09:06 BST showed the 0.80 transfer route
partly into occlusion evaluation but still incomplete: official identity
`774917` completed at 393/400, BGR identity `774920` completed at 391/400,
official occlusion `774919` was running at 9:42, BGR occlusion `774921` had
started and was running at 1:11, matched-random identity `774922` was still
running at 33:17, and matched-random occlusion `774923` remained
dependency-pending. Direct tails were official occlusion 45/58, BGR occlusion
5/7, and the active matched-random identity log 271/281. The helper synced
only an incomplete `summary_available.csv`; no compact `summary.csv` exists
and no gate can be run.
Latest poll/sync at 2026-06-10 11:12:56 BST showed the 0.80 transfer route
still incomplete: official occlusion `774919` was running at 13:32, BGR
occlusion `774921` at 5:01, matched-random identity `774922` at 37:07, and
matched-random occlusion `774923` was still dependency-pending. Direct tails
were official occlusion 55/78, BGR occlusion 28/32, and active matched-random
identity 317/327. The synced `summary_available.csv` still contains only
official/BGR identity rows, so no compact `summary.csv` exists and no gate can
be run.
Latest poll/sync at 2026-06-10 11:17:49 BST showed the 0.80 transfer route
still incomplete and affected by scheduling: official occlusion `774919` was
running at 18:25, BGR occlusion `774921` at 9:54, matched-random identity
`774922` had returned to pending on `Priority` with an estimated start
2026-06-11 22:02:14 BST, and matched-random occlusion `774923` remained
dependency-pending. Direct tails were official occlusion 68/103, BGR occlusion
47/60, and the interrupted matched-random identity log 317/327. The compact
`summary.csv` is still missing and no gate can be run.
Latest poll/sync at 2026-06-10 11:24:53 BST still left the 0.80 transfer route
incomplete: official occlusion `774919` was running at 25:29, BGR occlusion
`774921` at 16:58, matched-random identity `774922` was pending on `Priority`
with estimated start 2026-06-11 10:59:00 BST, and matched-random occlusion
`774923` was dependency-pending. The synced `summary_available.csv` still has
only official/BGR identity rows at 393/400 and 391/400. Direct occlusion tails
were official 74/135 and BGR 72/103, while the active matched-random identity
log remains partial/stale at 317/327. The compact `summary.csv` is still
missing, so no fixed gate can be run.
Latest poll/sync at 2026-06-10 11:29:24 BST still left the 0.80 transfer route
incomplete: official occlusion `774919` was running at 30:00, BGR occlusion
`774921` at 21:29, matched-random identity `774922` was pending on `Priority`
with estimated start 2026-06-11 11:25:00 BST, and matched-random occlusion
`774923` remained dependency-pending. Direct partial tails showed official
occlusion 78/152 and BGR occlusion 77/123; matched-random still has no
complete identity or occlusion row, so the apparent BGR-official partial gap is
not gateable. The compact `summary.csv` is still missing.
Latest poll/sync at 2026-06-10 11:33:26 BST still left the 0.80 transfer route
incomplete: official occlusion `774919` was running at 34:02, BGR occlusion
`774921` at 25:31, matched-random identity `774922` remained priority-pending
with estimated start 2026-06-11 11:25:00 BST, and matched-random occlusion
`774923` remained dependency-pending. The compact `summary.csv` was still
missing; no gate can be run without the matched-random comparator.
Latest poll/sync at 2026-06-10 11:36:43 BST still left the 0.80 transfer route
incomplete: official occlusion `774919` was running at 37:19, BGR occlusion
`774921` at 28:48, matched-random identity `774922` remained priority-pending
with estimated start 2026-06-11 11:25:00 BST, and matched-random occlusion
`774923` remained dependency-pending. The compact `summary.csv` was still
missing; no gate can be run without the matched-random comparator.
Latest poll/sync at 2026-06-10 11:39:44 BST still left the 0.80 transfer route
incomplete: official occlusion `774919` was running at 40:20, BGR occlusion
`774921` at 31:49, matched-random identity `774922` remained priority-pending
with estimated start 2026-06-11 11:36:00 BST, and matched-random occlusion
`774923` remained dependency-pending. The compact `summary.csv` was still
missing; no gate can be run without the matched-random comparator.
Latest poll/sync at 2026-06-10 11:43:00 BST still left the 0.80 transfer route
incomplete: official occlusion `774919` was running at 43:36, BGR occlusion
`774921` at 35:05, matched-random identity `774922` remained priority-pending
with estimated start 2026-06-11 11:36:00 BST, and matched-random occlusion
`774923` remained dependency-pending. The compact `summary.csv` was still
missing; no gate can be run without the matched-random comparator.
Latest poll/sync at 2026-06-10 11:46:29 BST still left the 0.80 transfer route
incomplete: official occlusion `774919` was running at 47:05 on `c2-g4-17`,
BGR occlusion `774921` was running at 38:34 on `c2-g4-17`, matched-random
identity `774922` remained priority-pending with estimated start
2026-06-11 11:36:00 BST, and matched-random occlusion `774923` remained
dependency-pending. The compact `summary.csv` was still missing; no gate can
be run without the matched-random comparator.
Latest poll/sync at 2026-06-10 11:50:52 BST still left the 0.80 transfer route
incomplete: official occlusion `774919` was running at 51:28 and BGR occlusion
`774921` at 42:57, both on `c2-g4-17`; matched-random identity `774922`
remained priority-pending with estimated start 2026-06-11 11:36:00 BST, and
matched-random occlusion `774923` remained dependency-pending. The compact
`summary.csv` was still missing; no gate can be run without the matched-random
comparator.
Latest poll/sync at 2026-06-10 11:54:39 BST still left the 0.80 transfer route
incomplete: official occlusion `774919` was running at 55:15 and BGR occlusion
`774921` at 46:44, both on `c2-g4-17`; matched-random identity `774922`
remained priority-pending with estimated start 2026-06-11 22:02:14 BST, and
matched-random occlusion `774923` remained dependency-pending. The compact
`summary.csv` was still missing; no gate can be run without the matched-random
comparator.
Latest poll/sync at 2026-06-10 11:58:18 BST still left the 0.80 transfer route
incomplete: official occlusion `774919` was running at 58:54 and BGR occlusion
`774921` at 50:23, both on `c2-g4-17`; matched-random identity `774922`
remained priority-pending with estimated start 2026-06-11 11:36:00 BST, and
matched-random occlusion `774923` remained dependency-pending. The compact
`summary.csv` was still missing; no gate can be run without the matched-random
comparator.
Latest poll/sync at 2026-06-10 12:02:04 BST still left the 0.80 transfer route
incomplete: official occlusion `774919` was running at 1:02:40 and BGR
occlusion `774921` at 54:09, both on `c2-g4-17`; matched-random identity
`774922` remained priority-pending with estimated start 2026-06-11 11:36:00
BST, and matched-random occlusion `774923` remained dependency-pending. The
compact `summary.csv` was still missing; no gate can be run without the
matched-random comparator.
Latest poll/sync at 2026-06-10 12:04:56 BST still left the 0.80 transfer route
incomplete: official occlusion `774919` was running at 1:05:32 and BGR
occlusion `774921` at 57:01, both on `c2-g4-17`; matched-random identity
`774922` remained priority-pending with estimated start 2026-06-11 11:36:00
BST, and matched-random occlusion `774923` remained dependency-pending. The
compact `summary.csv` was still missing; no gate can be run without the
matched-random comparator.
Latest poll/sync at 2026-06-10 12:07:56 BST still left the 0.80 transfer route
incomplete: official occlusion `774919` completed successfully with exit `0:0`
after 1:05:37, BGR occlusion `774921` was still running at 1:00:01, and
matched-random identity `774922` was resource-pending with estimated start
2026-06-10 23:36:39 BST. Matched-random occlusion `774923` remained
dependency-pending. The compact `summary.csv` was still missing; no gate can be
run without BGR occlusion and matched-random rows.
Latest poll/sync at 2026-06-10 12:13:23 BST showed the 0.80 transfer route
still incomplete but closer to a comparator check: official identity/occlusion
`774917`/`774919` and BGR identity/occlusion `774920`/`774921` completed
successfully. The incomplete local `summary_available.csv` has BGR identity
391/400, BGR occlusion 305/400, official identity 393/400, and official
occlusion 296/400. Matched-random identity `774922` had just started on
`c2-g4-17`, and matched-random occlusion `774923` remained dependency-pending.
The compact `summary.csv` is still missing, so the fixed hard-occlusion gate
cannot be run and this is not paper evidence.
Latest poll/sync at 2026-06-10 12:20:17 BST left the 0.80 transfer route
unchanged in substance: matched-random identity `774922` was running on
`c2-g4-17` at 7:32, matched-random occlusion `774923` remained
dependency-pending, and the compact `summary.csv` was still missing. No fixed
gate can be run yet.

Active learned-policy intervention route: fixed hard-occlusion 0.80
identity-anchored OpenVLA-OFT adaptation. This is a new training route
targeting the observed identity-regression failure in the hard-occlusion
routes: the 0.80 transfer route already has BGR identity 391/400 vs. official
393/400, and the A6000 0.65 adaptation has BGR identity 389/400 vs. official
393/400 and matched random 390/400. The new route keeps the occlusion-specific
premise but uses a shorter, more strongly anchored update:
`PREP_TAG=p2048unique_hardocc080_identityanchor_prereg`,
`ADAPT_TAG=cleanmix_p2048unique_hardocc080_identityanchor_prereg_proxanchor_l2_2em1_step50200_lr1em7_identitylora_imageaug_officialtrainstats_v1`,
`PERTURB_TAG=cleanmix_p2048unique_hardocc080_identityanchor_prereg_proxanchor_l2_2em1_step50200_lr1em7_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1`,
`OCCLUSION_FRACTION_OVERRIDE=0.80`, `OCCLUSION_CAP=512`,
`OCCLUSION_REPEAT=4`, `ADAPT_STEPS=200`, `LR=1e-7`,
`PROXIMAL_ANCHOR_L2=20.0`, identity-LoRA, image augmentation, official stats,
and fixed identity plus occlusion fraction 0.80 evaluation over 10 LIBERO-Goal
tasks with 40 trials per task. Promotion uses the same hard-occlusion gate:
BGR must beat both official and matched random by at least 10/400 occlusion
episodes and at least 0.02 absolute success rate while not trailing the best
identity comparator by more than one episode. Submitted on `athena`: prep
`776029`; BGR train/merge/clean-eval `776033`/`776034`/`776035`;
matched-random train/merge/clean-eval `776036`/`776037`/`776038`; official
identity/occlusion eval `776040`/`776041`; BGR identity/occlusion eval
`776042`/`776043`; matched-random identity/occlusion eval `776044`/`776045`.
Initial poll/sync at 2026-06-10 12:30:34 BST showed prep `776029` pending on
priority with estimated start 2026-06-10 15:15:13 BST, official identity
`776040` priority-pending with estimated start 2026-06-10 23:36:39 BST, and
all BGR/random adaptation plus dependent perturb-eval jobs dependency-pending.
No remote logs or compact summaries exist yet. Poll/sync with
`scripts/sync_openvla_oft_hard_occlusion080_identityanchor_results.sh --poll --sync --no-check`.
Latest poll/sync at 2026-06-10 12:37:32 BST showed no substantive change:
prep `776029` remained priority-pending with estimated start 2026-06-10
15:15:13 BST, official identity `776040` remained priority-pending with
estimated start 2026-06-10 23:36:39 BST, and every BGR/random adaptation and
dependent perturb-eval job remained dependency-pending. There are still no
remote logs or compact summaries, so the route is not gateable.
Latest poll/sync at 2026-06-10 12:45:39 BST showed prep `776029` running since
12:39:19 on an A6000 node; direct log tail was healthy and had advanced to
`Rendering BGR boundary perturbations at 2026-06-10T12:43:36+01:00`.
Official identity `776040` remained priority-pending with estimated start
2026-06-10 15:15:13 BST, while all BGR/random adaptation, merge, clean-eval,
identity-eval, and occlusion-eval jobs remained dependency-pending. No compact
`summary.csv` exists, so this route remains the live acceptance-moving
candidate but is not gateable or paper evidence.
Latest poll/sync at 2026-06-10 12:48:57 BST showed the same scheduler state:
prep `776029` was still running, official identity `776040` was
priority-pending, and every downstream BGR/random adaptation and perturb eval
job was dependency-pending. Direct prep-log inspection remained healthy and had
advanced to `Rendering matched-random perturbations at
2026-06-10T12:46:53+01:00`. No compact `summary.csv` or perturb logs exist
yet for this route, so it remains incomplete.
Latest poll/sync at 2026-06-10 12:51:17 BST showed the identity-anchored route
still in prep: `776029` was running, official identity `776040` was
priority-pending, and all downstream BGR/random adaptation and perturb-eval
jobs were dependency-pending. Direct prep-log tail showed the route had moved
past rendering into TFDS generation, with `Generating train examples...` at
26 examples. There is still no compact `summary.csv`, so this route remains
the live acceptance-moving candidate but is not gateable.
Latest poll/sync at 2026-06-10 12:53:42 BST showed the same scheduler state
for the identity-anchored route: prep `776029` running, official identity
`776040` priority-pending, and all downstream BGR/random training and eval jobs
dependency-pending. Direct prep-log tail showed TFDS generation progressing to
49 train examples. There is still no compact `summary.csv`; no paper claim can
change.
Latest poll/sync at 2026-06-10 12:56:06 BST showed concrete progress on the
identity-anchored route: prep `776029` completed successfully at 12:54:41, and
official identity eval `776040` started at 12:54:52 on `c1-g4-02`. BGR
adaptation `776033` became priority-pending with estimated start
2026-06-10 15:15:13 BST; BGR/random merge, clean-eval, identity-eval, and
occlusion-eval children remained dependency-pending. Direct official identity
tail showed 6/6 successes. No compact `summary.csv` exists, so this route is
still incomplete and not paper evidence.
Latest poll/sync at 2026-06-10 13:00:40 BST showed the identity-anchored route
now actively training/evaluating: BGR adaptation `776033` started at 12:59:53
on `c1-g4-03`, and official identity `776040` was still running on `c1-g4-02`
after 5:48 elapsed. BGR/random merge, clean-eval, identity-eval, and
occlusion-eval children remained dependency-pending. Direct official identity
log parsing showed 41/42 successes so far. No compact `summary.csv` exists, so
the route is still incomplete and cannot support a paper claim.
Latest poll/sync at 2026-06-10 13:04:14 BST showed BGR adaptation `776033`
completed successfully at 13:03:27 and BGR merge `776034` running since
13:03:53 on `c1-g4-03`. Matched-random adaptation `776036` was priority-pending
with estimated start 2026-06-10 15:03:53 BST; BGR/random clean and perturb eval
children remained dependency-pending. Official identity `776040` was still
running with direct parsed tail 71/74. The compact `summary.csv` is still
missing and the route remains incomplete.
Latest direct poll at 2026-06-10 13:07:59 BST showed BGR clean eval `776035`
running after BGR merge completed, BGR identity eval `776042`
priority-pending, matched-random adaptation `776036` priority-pending, and all
occlusion children still dependency-pending. Official identity `776040` was
still running with direct parsed tail 101/107. The active BGR clean-eval log
had started but had not yet emitted parseable totals. No compact `summary.csv`
exists, so the route remains incomplete.
Latest poll/sync at 2026-06-10 13:15:05 BST showed a scheduler/preemption
delay rather than new evidence: official identity `776040`, BGR clean eval
`776035`, BGR identity eval `776042`, and matched-random adaptation `776036`
were all priority-pending with zero current elapsed time; official occlusion
`776041`, BGR occlusion `776043`, and matched-random eval children
`776044`/`776045` remained dependency-pending. Estimated starts had slipped to
2026-06-11 00:14--14:21 BST for the runnable jobs. The remote perturb-log
directory exists but contains no complete perturb-eval logs, so no
`summary_available.csv` or `summary.csv` exists locally for this route.
Latest poll/sync at 2026-06-10 13:25:30 BST showed the same non-gateable
state for the A6000 identity-anchored route: BGR clean eval `776035`,
matched-random adaptation `776036`, official identity `776040`, and BGR
identity `776042` were priority-pending, while official/BGR/random occlusion
and matched-random identity jobs were dependency-pending. Estimated starts
remained 2026-06-11 00:14--14:21 BST for the runnable jobs. The remote perturb
log directory exists with only incomplete official-identity output, no compact
summary is available, and no paper claim should change.

Companion learned-policy route: fixed hard-occlusion 0.80 identity-anchored
OpenVLA-OFT A40 fallback. This reuses the completed
`p2048unique_hardocc080_identityanchor_prereg` TFDS roots and keeps the same
fixed identity/occlusion-0.80 gate, but requests A40 GPUs and writes separate
checkpoint and perturb artifacts:
`ADAPT_TAG=cleanmix_p2048unique_hardocc080_identityanchor_a40_prereg_proxanchor_l2_2em1_step50200_lr1em7_identitylora_imageaug_officialtrainstats_v1`,
`PERTURB_TAG=cleanmix_p2048unique_hardocc080_identityanchor_a40_prereg_proxanchor_l2_2em1_step50200_lr1em7_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1`.
It is not a new claim or relaxed protocol. Submitted on `athena`: BGR
train/merge/clean-eval `776291`/`776292`/`776294`;
matched-random train/merge/clean-eval `776295`/`776296`/`776297`; official
identity/occlusion `776300`/`776301`; BGR identity/occlusion
`776302`/`776303`; matched-random identity/occlusion `776304`/`776305`.
Initial poll at 2026-06-10 13:25:17 BST showed BGR training `776291` and
official identity `776300` priority-pending, with all dependent merge,
clean-eval, BGR perturb, and matched-random jobs dependency-pending. Poll/sync
with
`scripts/sync_openvla_oft_hard_occlusion080_identityanchor_a40_results.sh --poll --sync --no-check`.
Do not incorporate this route into `paper/main.tex` unless a complete
`summary.csv` exists and the fixed +10/400, +0.02, and identity side-condition
gate passes.

Companion learned-policy route: fixed hard-occlusion 0.80 strict
identity-anchored OpenVLA-OFT adaptation. This reuses the completed
`p2048unique_hardocc080_identityanchor_prereg` TFDS roots and targets the
observed identity-regression failure more conservatively than the 200-step
identity-anchor route: `ADAPT_STEPS=100`, `LR=5e-8`, and
`PROXIMAL_ANCHOR_L2=50.0`, with identity-LoRA, image augmentation, official
stats, and the same fixed identity plus occlusion fraction 0.80 evaluation over
10 LIBERO-Goal tasks with 40 trials per task. It is not a relaxed protocol:
BGR must still beat both official and matched random by at least 10/400
occlusion episodes and at least 0.02 absolute success rate while not trailing
the best identity comparator by more than one episode. Submitted on `athena` at
2026-06-10 14:07 BST: BGR train/merge/clean-eval `776541`/`776542`/`776543`;
matched-random train/merge/clean-eval `776544`/`776545`/`776546`; official
identity/occlusion `776548`/`776549`; BGR identity/occlusion `776550`/`776551`;
matched-random identity/occlusion `776553`/`776554`. Initial poll showed BGR
train `776541` and official identity `776548` priority-pending with estimated
starts at 2026-06-11 14:21:02 BST, all dependent strict-route jobs
dependency-pending, and no remote logs or compact summaries. Poll/sync with
`scripts/sync_openvla_oft_hard_occlusion080_identityanchor_strict_results.sh --poll --sync --no-check`.
Do not incorporate this route into `paper/main.tex` unless a complete
`summary.csv` exists and the fixed gate passes.

Active learned-policy intervention route: fixed hard-occlusion OpenVLA-OFT
adaptation. This is a genuinely new training route, not just a transfer
diagnostic: both BGR and matched-random TFDS roots render occlusion examples
with `OCCLUSION_FRACTION_OVERRIDE=0.65`, then train from the official
checkpoint with the same clean-plus-occlusion mix, official stats,
identity-LoRA, image augmentation, `PROXIMAL_ANCHOR_L2=5.0`, `LR=2e-7`, and
`ADAPT_STEPS=400`. The fixed evaluation is identity plus occlusion fraction
0.65 on 10 LIBERO-Goal tasks with 40 trials per task. Promotion requires BGR
to beat both official and matched random by at least 10/400 occlusion episodes
and at least 0.02 absolute success rate while not trailing the best identity
comparator by more than 1 episode. The route required a renderer update:
`scripts/render_openvla_teacher_examples.py` now supports
`--override-perturbation-param occlusion.fraction=0.65`, and
`scripts/queue_openvla_oft_preregistered_occlusion_bottleneck.sh` forwards
`OCCLUSION_FRACTION_OVERRIDE` into both BGR and random perturb renders. Submitted
on `athena`: prep `774717`; BGR train/merge/clean-eval `774718`/`774719`/`774720`;
matched-random train/merge/clean-eval `774721`/`774722`/`774723`; official
identity/occlusion eval `774724`/`774725`; BGR identity/occlusion eval
`774726`/`774727`; matched-random identity/occlusion eval `774728`/`774729`.
Initial `squeue` showed prep `774717` and official identity `774724` pending
on unavailable nodes, with all adapted BGR/random jobs dependency-pending.
Poll/sync with
`scripts/sync_openvla_oft_hard_occlusion_adapt_results.sh --poll --no-check`
and, when logs or summaries exist,
`scripts/sync_openvla_oft_hard_occlusion_adapt_results.sh --sync`.
Latest poll at 2026-06-10 09:45:12 BST still showed prep `774717` and
official identity `774724` pending on unavailable nodes, with estimated starts
unchanged from the earlier scheduler report (`774717` at 2026-06-10 10:59:06
BST and `774724` at 2026-06-10 21:24:42 BST). All adapted BGR/random jobs and
occlusion eval jobs remain dependency-pending; no logs or summary exist for the
adapted route yet.
Latest poll at 2026-06-10 10:04:47 BST still showed the A6000 adaptation
blocked by unavailable nodes: prep `774717` and official identity `774724`
pending on `ReqNodeNotAvail`, with the same estimated starts (`774717` at
2026-06-10 10:59:06 BST and `774724` at 2026-06-10 21:24:42 BST). Every
adaptation, merge, clean-eval, and occlusion-eval child job remained
dependency-pending, and there were still no route logs or summaries.
Latest poll at 2026-06-10 10:14:46 BST showed no change for the A6000
adaptation route: prep `774717` and official identity `774724` were still
pending on `ReqNodeNotAvail`, all child jobs were dependency-pending, and no
logs or compact summaries existed.
Latest poll/sync at 2026-06-10 10:18:27 BST showed the A6000 adaptation prep
`774717` now pending on `Resources` instead of unavailable nodes, with
estimated start still 2026-06-10 10:59:06 BST. Official identity `774724` was
still pending on `ReqNodeNotAvail`; every child job remained dependency-pending
and no route logs or summary existed.
Latest poll/sync at 2026-06-10 10:21:07 BST showed the A6000 adaptation route
unchanged from the prior poll: prep `774717` was still resource-pending,
official identity `774724` was still unavailable-node pending, every child job
was dependency-pending, and no route logs or summary existed.
Latest poll/sync at 2026-06-10 10:23:53 BST showed the A6000 adaptation route
still unchanged: prep `774717` remained resource-pending, official identity
`774724` remained unavailable-node pending, all child jobs were
dependency-pending, and no route logs or summary existed.
Latest poll/sync at 2026-06-10 10:30:56 BST showed the A6000 adaptation route
finally started: prep `774717` was running on `c2-g4-20` at 5:23, and official
identity `774724` was running on `c2-g4-20` at 4:59. BGR/random adaptation,
merge, clean-eval, identity eval, and occlusion eval children remained
dependency-pending. The official identity log tail was healthy at 30/30
successes so far, but no compact summary exists and the route is not evidence
yet.
Latest poll/sync at 2026-06-10 10:35:15 BST showed the A6000 adaptation route
still in prep/official-identity progress: prep `774717` was running at 9:42,
official identity `774724` at 9:18, and all BGR/random adaptation, merge,
clean-eval, identity-eval, and occlusion-eval child jobs were still
dependency-pending. The official identity tail reached 49/51 successes. No
compact summary exists.
Latest poll/sync at 2026-06-10 10:41:25 BST showed the A6000 adaptation route
still in prep/official-identity progress: prep `774717` was running at 15:52
on `c2-g4-20`, official identity `774724` was running at 15:28, and every
BGR/random adaptation, merge, clean-eval, identity-eval, and hard-occlusion
child remained dependency-pending. The route still has no compact summary.
Latest poll/sync at 2026-06-10 10:47:48 BST showed the A6000 adaptation route
past prep: prep `774717` completed with exit `0:0`; BGR adaptation `774718`
was running at 4:14; official identity `774724` was running at 21:51; BGR
merge/clean, matched-random adaptation/merge/clean, and all BGR/random
identity plus hard-occlusion eval jobs remained dependency-pending. The
official identity direct tail reached 130/136. The route still has no compact
summary.
Latest poll/sync at 2026-06-10 10:54:30 BST showed the A6000 adaptation route
past BGR training and merge: prep `774717`, BGR adaptation `774718`, and BGR
merge `774719` completed with exit `0:0`; BGR clean eval `774720` was running
at 3:44; official identity `774724` was running at 28:34; matched-random
adaptation `774721` was resource-pending with estimated start
2026-06-10 22:25:57 BST; BGR identity `774726` was priority-pending with
estimated start 2026-06-10 22:30:19 BST; and the remaining matched-random and
occlusion jobs were dependency-pending. The official identity direct tail
reached 166/173. The route still has no compact summary.
Latest poll/sync at 2026-06-10 11:01:16 BST showed the A6000 adaptation route
still incomplete: BGR clean eval `774720` was running at 10:29, official
identity `774724` was running at 35:19, matched-random adaptation `774721`
remained resource-pending, BGR identity `774726` remained priority-pending,
and the remaining matched-random and occlusion jobs were dependency-pending.
The official identity direct tail reached 211/218. The route still has no
compact summary.
Latest poll/sync at 2026-06-10 11:03:34 BST showed the A6000 adaptation route
still incomplete: BGR clean eval `774720` was running at 12:47, official
identity `774724` at 37:37, matched-random adaptation `774721` remained
resource-pending with estimated start 2026-06-10 22:25:57 BST, and BGR
identity `774726` remained priority-pending with estimated start
2026-06-10 22:30:19 BST. The official identity direct tail reached 227/234.
The route still has no compact summary.
Latest poll/sync at 2026-06-10 11:09:06 BST showed the A6000 adaptation route
past BGR clean eval but still incomplete: prep `774717`, BGR train/merge
`774718`/`774719`, and BGR clean eval `774720` completed with exit `0:0`;
matched-random adaptation `774721` was running at 3:02 on `c2-g4-20`;
official identity `774724` was running at 43:09 on `c2-g4-20`; BGR identity
`774726` was pending on resources with estimated start
2026-06-10 22:25:57 BST; official occlusion `774725` and all matched-random
merge/clean/eval children remained dependency-pending. The official identity
direct tail reached 267/274. No compact summary exists and no gate can be run.
Latest poll/sync at 2026-06-10 11:12:56 BST showed the A6000 adaptation route
advancing but still incomplete: matched-random adaptation `774721` completed
successfully after 5:54, matched-random merge `774722` was pending on
resources, official identity `774724` was running at 46:59, and BGR identity
`774726` had started at 11:11:58 BST. Official occlusion `774725`, BGR
occlusion `774727`, and matched-random identity/occlusion `774728`/`774729`
remained dependency-pending. Direct tails were official identity 302/309 and
BGR identity 6/6. No compact summary exists and no gate can be run.
Latest poll/sync at 2026-06-10 11:17:49 BST showed the A6000 adaptation route
still incomplete: matched-random merge `774722` was running at 2:46, official
identity `774724` was running at 51:52, BGR identity `774726` was running at
5:51, and official/BGR/matched-random occlusion plus matched-random identity
children remained dependency-pending. Direct tails were official identity
347/354 and BGR identity 33/33. No compact summary exists and no gate can be
run.
Latest poll/sync at 2026-06-10 11:25:14 BST showed the A6000 adaptation route
past data prep and both adaptation/merge chains, but still far from a gate:
random clean eval `774723` was running at 6:29, official identity `774724` was
still reported running at 59:17 even though its direct log tail had reached the
complete 393/400 final result, BGR identity `774726` was running at 13:16, and
matched-random identity `774728` was pending on `Resources` with estimated
start 2026-06-10 15:15:13 BST. Official/BGR/matched-random occlusion evals
remained dependency-pending. Direct tails were BGR identity 87/91 and official
identity complete at 393/400. No compact summary exists and no gate can be run.
Latest poll/sync at 2026-06-10 11:29:45 BST showed the A6000 adaptation route
still incomplete but moving: official identity `774724` completed with 393/400,
BGR identity `774726` was running at 17:47, matched-random identity `774728`
had started and was running at 3:46, random clean `774723` was still running,
official occlusion `774725` was pending on `Priority`, and BGR/random occlusion
jobs `774727`/`774729` were dependency-pending. The synced
`summary_available.csv` has only official identity. Direct identity tails were
BGR 113/120 and matched random 23/23. No compact summary exists and no gate can
be run.
Latest poll/sync at 2026-06-10 11:33:47 BST showed the A6000 adaptation route
still incomplete but healthy: random clean `774723` was running at 15:02, BGR
identity `774726` was running at 21:49, matched-random identity `774728` was
running at 7:48, official occlusion `774725` was pending on `Priority`, and
BGR/random occlusion jobs `774727`/`774729` remained dependency-pending. The
synced `summary_available.csv` still contains only official identity 393/400,
so no fixed gate can be run.
Latest poll/sync at 2026-06-10 11:37:05 BST showed the A6000 adaptation route
advancing into occlusion but still incomplete: random clean `774723` completed
with exit `0:0`, official occlusion `774725` started and was running at 0:26,
BGR identity `774726` was running at 25:07, matched-random identity `774728`
was running at 11:06, and BGR/random occlusion jobs `774727`/`774729` remained
dependency-pending. The synced `summary_available.csv` still contains only
official identity 393/400, so no fixed gate can be run.
Latest poll/sync at 2026-06-10 11:40:07 BST showed the A6000 adaptation route
still progressing but incomplete: official occlusion `774725` was running at
3:28, BGR identity `774726` was running at 28:09, matched-random identity
`774728` was running at 14:08, and BGR/random occlusion jobs `774727`/`774729`
remained dependency-pending. The synced `summary_available.csv` still contains
only official identity 393/400, so no fixed gate can be run.
Latest poll/sync at 2026-06-10 11:43:23 BST showed the A6000 adaptation route
still progressing but incomplete: official occlusion `774725` was running at
6:45, BGR identity `774726` was running at 31:26, matched-random identity
`774728` was running at 17:25, and BGR/random occlusion jobs `774727`/`774729`
remained dependency-pending. The synced `summary_available.csv` still contains
only official identity 393/400, so no fixed gate can be run.
Latest poll/sync at 2026-06-10 11:47:49 BST showed the A6000 adaptation route
past prep, both training/merge chains, and both clean evals, but still
incomplete: official occlusion `774725` was running at 11:10, BGR identity
`774726` was running at 35:51, and matched-random identity `774728` was running
at 21:50. BGR and matched-random occlusion jobs `774727`/`774729` remained
dependency-pending. The compact `summary.csv` was missing; the synced
`summary_available.csv` was incomplete and no fixed gate can be run.
Latest poll/sync at 2026-06-10 11:51:14 BST showed the A6000 adaptation route
still incomplete: official occlusion `774725` was running at 14:35, BGR
identity `774726` was running at 39:16, and matched-random identity `774728`
was running at 25:15. BGR and matched-random occlusion jobs `774727`/`774729`
remained dependency-pending. The compact `summary.csv` was missing; the synced
`summary_available.csv` still could not support a fixed gate.
Latest poll/sync at 2026-06-10 11:55:02 BST showed the A6000 adaptation route
still incomplete: official occlusion `774725` was running at 18:23, BGR
identity `774726` was running at 43:04, and matched-random identity `774728`
was running at 29:03. BGR and matched-random occlusion jobs `774727`/`774729`
remained dependency-pending. The compact `summary.csv` was missing; the synced
`summary_available.csv` still could not support a fixed gate.
Latest poll/sync at 2026-06-10 11:58:42 BST showed the A6000 adaptation route
still incomplete: official occlusion `774725` was running at 22:03, BGR
identity `774726` was running at 46:44, and matched-random identity `774728`
was running at 32:43. BGR and matched-random occlusion jobs `774727`/`774729`
remained dependency-pending. The compact `summary.csv` was missing; the synced
`summary_available.csv` still could not support a fixed gate.
Latest poll/sync at 2026-06-10 12:02:28 BST showed the A6000 adaptation route
still incomplete: official occlusion `774725` was running at 25:49, BGR
identity `774726` was running at 50:30, and matched-random identity `774728`
was running at 36:29. BGR and matched-random occlusion jobs `774727`/`774729`
remained dependency-pending. The compact `summary.csv` was missing; the synced
`summary_available.csv` still could not support a fixed gate.
Latest poll/sync at 2026-06-10 12:05:20 BST showed the A6000 adaptation route
still incomplete: official occlusion `774725` was running at 28:41, BGR
identity `774726` was running at 53:22, and matched-random identity `774728`
was running at 39:21. BGR and matched-random occlusion jobs `774727`/`774729`
remained dependency-pending. The compact `summary.csv` was missing; the synced
`summary_available.csv` still could not support a fixed gate.
Latest poll/sync at 2026-06-10 12:08:20 BST showed the A6000 adaptation route
still incomplete: official occlusion `774725` was running at 31:41, BGR
identity `774726` was running at 56:22, and matched-random identity `774728`
was running at 42:21. BGR and matched-random occlusion jobs `774727`/`774729`
remained dependency-pending. The compact `summary.csv` was missing; the synced
`summary_available.csv` still could not support a fixed gate.
Latest poll/sync at 2026-06-10 12:13:47 BST showed the A6000 adaptation route
still incomplete: prep/train/merge/clean chains completed with exit `0:0`, and
official identity `774724` completed at 393/400. Official occlusion `774725`,
BGR identity `774726`, and matched-random identity `774728` were running; BGR
and matched-random occlusion jobs `774727`/`774729` remained
dependency-pending. The compact `summary.csv` was still missing, and the
synced `summary_available.csv` still contains only official identity, so no
fixed gate can be run.
Latest poll/sync at 2026-06-10 12:20:41 BST showed the A6000 adaptation route
advancing but still incomplete: BGR identity `774726` completed successfully
after 1:02:39, and BGR occlusion `774727` started on `c2-g4-23`. Official
occlusion `774725` and matched-random identity `774728` were still running,
matched-random occlusion `774729` remained dependency-pending, and the compact
`summary.csv` was still missing. The fixed gate remains incomplete.
Latest poll/sync at 2026-06-10 12:37:32 BST showed this A6000 adaptation route
still incomplete and already non-promotable under the clean-identity side
condition: BGR identity is 389/400, official identity is 393/400, and
matched-random identity is 390/400, so BGR trails the best identity comparator
by 4 episodes before occlusion rows are complete. Official occlusion `774725`,
BGR occlusion `774727`, and matched-random occlusion `774729` were running;
matched-random identity `774728` completed with 390/400. The synced
`summary_available.csv` has only identity rows, and no complete `summary.csv`
exists, so no positive paper claim can be made from this route.
Latest poll/sync at 2026-06-10 12:45:39 BST still had the A6000 0.65
adaptation route incomplete and non-promotable by the fixed identity side
condition: official identity `774724` completed at 393/400, BGR identity
`774726` completed at 389/400, and matched-random identity `774728` completed
at 390/400. Official/BGR/matched-random occlusion jobs `774725`/`774727`/
`774729` were running. Direct partial occlusion tails were official 224/323,
BGR 79/130, and matched random 76/99; these are not gateable, but they do not
rescue the identity failure.
Latest poll/sync at 2026-06-10 12:48:57 BST still left the A6000 0.65
adaptation route incomplete: official/BGR/matched-random occlusion jobs
`774725`/`774727`/`774729` were running. Direct partial tails were official
247/346, BGR 81/139, and matched random 78/108. The identity rows remain
BGR 389/400, official 393/400, and matched random 390/400, so this route is
already non-promotable regardless of the final occlusion rows.
Latest poll/sync at 2026-06-10 12:51:17 BST left the A6000 0.65 adaptation
route incomplete and still non-promotable by identity. Official/BGR/random
occlusion jobs were running, with direct partial tails official 263/362, BGR
82/146, and matched random 85/119. The partial occlusion evidence also points
away from a BGR win, but the formal blocker is already the identity side
condition.
Latest poll/sync at 2026-06-10 12:53:42 BST still left the A6000 0.65
adaptation route incomplete. Direct occlusion tails were official 275/375,
BGR 82/154, and matched random 86/126. This route remains non-promotable by
identity and the partial occlusion trajectory is also unfavorable to BGR.
Latest poll/sync at 2026-06-10 12:56:06 BST still left the A6000 0.65
adaptation route incomplete and non-promotable. Official/BGR/random occlusion
jobs `774725`/`774727`/`774729` were still running; direct tails were official
286/387, BGR 83/161, and matched random 88/134.
Latest poll/sync at 2026-06-10 13:00:40 BST kept the A6000 0.65 adaptation
route incomplete and non-promotable. Official occlusion `774725` completed
successfully at 297/400, while BGR occlusion `774727` and matched-random
occlusion `774729` were still running with direct parsed tails of 119/198 and
88/147. The completed identity rows remain BGR 389/400, official 393/400, and
matched random 390/400, so the route already violates the identity side
condition.
Latest poll/sync at 2026-06-10 13:04:14 BST still left the A6000 0.65
adaptation route incomplete and non-promotable. The synced incomplete
`summary_available.csv` now has official occlusion complete at 297/400 plus all
three identity rows. Direct parsed tails for the still-running occlusion jobs
were BGR 123/209 and matched random 88/156. The identity side condition remains
failed.
Latest poll/sync at 2026-06-10 13:15:05 BST still left the A6000 0.65
adaptation route incomplete and already non-promotable by identity. Official
occlusion remains complete at 297/400, BGR occlusion `774727` was running at
1:00:19, and matched-random occlusion `774729` was priority-pending. The
synced incomplete local summary still has only BGR/official/random identity
rows plus official occlusion; the BGR identity row is 389/400 vs. official
393/400 and random 390/400, so the fixed identity side condition is already
violated.
Latest poll/sync at 2026-06-10 13:35:04 BST showed BGR occlusion `774727`
completed successfully at 301/400. The A6000 0.65 adaptation route remains
incomplete because matched-random occlusion `774729` is still priority-pending,
and it remains non-promotable regardless: identity is BGR 389/400, official
393/400, and matched random 390/400, while BGR's completed occlusion row is
only +4/400 over official and below the fixed +10/400 and +0.02 promotion
threshold.

Operational sync update: `scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh`
now summarizes complete remote text logs in place over SSH and streams only the
compact CSV rows needed for local `summary_available.csv` generation. This
avoids repeated rsync code 12/255 drops from live log trees and preserves the
local gate path without copying `rollouts/` media.
Experiment-routing note at 2026-06-10 13:08 BST: the generated scorecard still
warns not to start another same-protocol MiniGrid, classic-control, PointMaze,
or FetchReach screen while the active learned-policy route is pending; existing
screens already fail by saturation, stronger-baseline loss, or state-priority
ablation. No duplicate standard-environment cluster job was launched.

Active learned-policy intervention fallback: fixed hard-occlusion OpenVLA-OFT
adaptation on A40 GPUs, queued under separate artifact tags after the A6000
adaptation route remained pending on unavailable nodes. This is the same fixed
hard-occlusion training/eval premise, not a new claim or a relaxed gate: BGR
must still beat both official and matched random by at least 10/400 occlusion
episodes and at least 0.02 absolute success rate while not trailing the best
identity comparator by more than 1 episode. The A40 fallback uses
`PREP_TAG=p2048unique_hardocc065_a40_prereg`,
`ADAPT_TAG=cleanmix_p2048unique_hardocc065_a40_prereg_proxanchor_l2_5em0_step50400_lr2em7_identitylora_imageaug_officialtrainstats_v1`,
`PERTURB_TAG=hardocc065_a40_adapt_step50400_lr2em7_v1`,
`OCCLUSION_FRACTION_OVERRIDE=0.65`, `ADAPT_STEPS=400`, `LR=2e-7`,
`PARTITION=low-prio-gpu`, and `GRES=gpu:a40:1`. Submitted jobs are prep
`774816`; BGR train/merge/clean-eval `774817`/`774818`/`774819`;
matched-random train/merge/clean-eval `774820`/`774821`/`774822`; official
identity/occlusion eval `774846`/`774847`; BGR identity/occlusion eval
`774848`/`774849`; and matched-random identity/occlusion eval
`774850`/`774851`. An initial wrapper submission mistakenly queued default
blur/brightness/shift eval jobs `774826`, `774828`--`774843`; these were
canceled immediately and replaced with the fixed direct perturb eval using
`PERTURBATIONS='identity={};occlusion={"fraction":0.65}'`. Initial A40 poll at
2026-06-10 09:57:47 BST showed prep `774816` running on `c2-g4-17`, official
identity `774846` pending on resources, and all BGR/random adaptation and
perturb jobs dependency-pending. Poll/sync with
`scripts/sync_openvla_oft_hard_occlusion_adapt_a40_results.sh --poll --no-check`
and, when logs or summaries exist,
`scripts/sync_openvla_oft_hard_occlusion_adapt_a40_results.sh --sync`.
Latest A40 poll at 2026-06-10 10:04:47 BST showed prep `774816` still running
on `c2-g4-17` at 9:54 elapsed; official identity `774846` was pending on
resources with estimated start 2026-06-10 12:56:05 BST; all BGR/random
adaptation and perturb jobs were dependency-pending. The prep log was healthy:
it had advanced through BGR perturb rendering and into matched-random
perturbation rendering at 2026-06-10 10:03:40 BST. No A40 perturb logs or
summary existed yet, so this route is still incomplete and not paper evidence.
Latest A40 poll at 2026-06-10 10:14:47 BST showed prep `774816` completed
successfully with exit `0:0`, BGR adaptation `774817` running on `c2-g4-17`,
and official identity `774846` running on `c2-g4-17`. BGR merge/clean,
matched-random adaptation/merge/clean, and all dependent occlusion eval jobs
remained dependency-pending. Remote perturb logs now exist for official
identity, but the compact `summary.csv` is still missing and the fixed gate
remains incomplete.
Latest A40 poll/sync at 2026-06-10 10:19:18 BST showed BGR adaptation `774817`
completed successfully with exit `0:0` after reaching step 50400 and saving the
checkpoint. BGR merge `774818` and matched-random adaptation `774820` were
pending on priority; official identity `774846` was still running at 14:09
elapsed; all BGR/random perturb evals and occlusion evals were still
dependency-pending. The sync found only official identity logs and could not
build a compact summary yet.
Latest A40 poll/sync at 2026-06-10 10:21:07 BST showed no gateable change:
prep `774816` and BGR adaptation `774817` were completed with exit `0:0`;
BGR merge `774818` and matched-random adaptation `774820` were still
priority-pending; official identity `774846` was running at 15:58 elapsed and
had reached 118/124 successes in the log tail; all BGR/random perturb evals and
all occlusion evals were still dependency-pending. No compact summary exists,
so the fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 10:23:53 BST showed no gateable change:
prep `774816` and BGR adaptation `774817` remained completed, BGR merge
`774818` and matched-random adaptation `774820` were still priority-pending,
official identity `774846` was running at 18:44 elapsed, and all BGR/random
perturb evals plus all occlusion evals were dependency-pending. The official
identity log tail reached 133/139 successes. No compact summary exists, so the
fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 10:30:56 BST showed more movement but still
no gateable summary: prep `774816` and BGR adaptation `774817` remained
completed, BGR merge `774818` was running, matched-random adaptation `774820`
was running, and official identity `774846` was running at 25:48 elapsed. BGR
clean eval `774819`, random merge/clean `774821`/`774822`, BGR/random perturb
evals `774848`--`774851`, and official occlusion `774847` were still
dependency-pending. The official identity log tail reached 196/203 successes.
No compact summary exists, so the fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 10:35:15 BST showed BGR merge `774818`
completed successfully after 2:38. Official identity `774846` was still running
at 30:06 with a log tail of 221/228 successes. BGR clean eval `774819`,
matched-random adaptation `774820`, and BGR identity eval `774848` were pending
on `BeginTime`; random merge/clean, BGR/random occlusion evals, and official
occlusion remained dependency-pending. No compact summary exists, so the fixed
gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 10:43:45 BST showed the original BGR
identity eval `774848` failed after 25 seconds with exit `1:0`, leaving the
original BGR occlusion job `774849` in `DependencyNeverSatisfied`. The Slurm
stdout shows an OpenVLA startup race while reading the merged checkpoint
`config.json`; the file was valid JSON immediately after the failure, so this
is infrastructure, not a scientific result. Replacement BGR identity/occlusion
jobs `775102`/`775103` were submitted with the same fixed perturbation protocol,
with `775102` depending on BGR clean eval `774819` and `775103` depending on
both `774819` and `775102`, to avoid concurrent checkpoint config mutation.
At the same poll, BGR clean `774819`, random clean `774822`, official identity
`774846`, and random identity `774850` were running; official occlusion
`774847` and random occlusion `774851` were dependency-pending. The compact
summary is still missing and the fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 10:47:48 BST showed the replacement chain
correctly waiting: BGR clean `774819` was still running at 11:59, replacement
BGR identity `775102` was pending on `afterok:774819`, and replacement BGR
occlusion `775103` was pending on `afterok:774819,afterok:775102`. Official
identity `774846`, random clean `774822`, and random identity `774850` were
running; official occlusion `774847` and random occlusion `774851` remained
dependency-pending. Direct tails were official identity 352/359 and matched
random identity 38/39. The compact summary is still missing and the fixed gate
remains incomplete.
Latest A40 poll/sync at 2026-06-10 10:54:30 BST showed more movement but no
gateable summary. Prep, BGR adaptation/merge/clean, matched-random
adaptation/merge/clean, and official identity completed successfully; official
identity was 393/400 and is the only synced available row. Official occlusion
`774847` had just started and was running; matched-random identity `774850`
and replacement BGR identity `775102` were running; matched-random occlusion
`774851` and replacement BGR occlusion `775103` were dependency-pending. The
original failed BGR identity job `774848` and its child `774849` remain
ignored in favor of replacement jobs `775102`/`775103`. Direct partial tails
were BGR identity 55/57, official occlusion 1/3, and matched-random identity
107/112. The compact summary is still missing and the fixed gate remains
incomplete.
Latest A40 poll/sync at 2026-06-10 11:01:16 BST still had no compact summary.
Official occlusion `774847` was running at 6:53, replacement BGR identity
`775102` was running at 12:56, and matched-random identity `774850` was
running at 18:19. Replacement BGR occlusion `775103` and matched-random
occlusion `774851` were dependency-pending; the original failed BGR occlusion
child `774849` remained `DependencyNeverSatisfied` and ignored. Direct partial
tails were BGR identity 115/119, official occlusion 38/44, and matched-random
identity 146/152. The fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 11:03:34 BST still had no compact summary.
Official occlusion `774847` was running at 9:11, replacement BGR identity
`775102` at 15:14, and matched-random identity `774850` at 20:37. Replacement
BGR occlusion `775103` and matched-random occlusion `774851` remained
dependency-pending; original failed child `774849` remained ignored. Direct
partial tails were BGR identity 131/135, official occlusion 49/59, and
matched-random identity 168/175. The fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 11:09:06 BST still had no compact summary.
Prep `774816`, BGR train/merge/clean `774817`/`774818`/`774819`,
matched-random train/merge/clean `774820`/`774821`/`774822`, and official
identity `774846` completed successfully; official identity is 393/400 and is
still the only synced available row. Official occlusion `774847` was running
at 14:43, replacement BGR identity `775102` at 20:46, and matched-random
identity `774850` at 26:09. Replacement BGR occlusion `775103` and
matched-random occlusion `774851` remained dependency-pending; original failed
child `774849` remains ignored. Direct partial tails were BGR identity
167/174, official occlusion 70/92, and matched-random identity 220/227. The
fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 11:12:56 BST still had no compact summary.
Official occlusion `774847` was running at 18:33, replacement BGR identity
`775102` at 24:36, and matched-random identity `774850` at 29:59. Replacement
BGR occlusion `775103` and matched-random occlusion `774851` remained
dependency-pending, and original failed child `774849` remains ignored. Direct
partials were BGR identity 204/211, official occlusion 73/107, and
matched-random identity 256/263. The fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 11:17:49 BST still had no compact summary
and showed scheduler interruption on the adapted identity evals: official
occlusion `774847` was running at 23:26, replacement BGR identity `775102` was
pending on `Priority` after reaching 204/211 in the log, and matched-random
identity `774850` was pending on `Resources` after reaching 256/263.
Replacement BGR occlusion `775103` and matched-random occlusion `774851`
remained dependency-pending; original failed child `774849` remains ignored.
The official occlusion direct tail reached 78/128. The fixed gate remains
incomplete.
Latest A40 poll/sync at 2026-06-10 11:25:28 BST still had no compact summary:
official occlusion `774847` was running at 31:05, replacement BGR identity
`775102` was pending on `Priority` with estimated start 2026-06-11 22:02:14
BST, and matched-random identity `774850` was pending on `Priority` with
estimated start 2026-06-10 22:59:24 BST. Replacement BGR occlusion `775103`
and matched-random occlusion `774851` remained dependency-pending, and the
original failed BGR identity/occlusion chain `774848`/`774849` remains ignored.
Direct partial tails were official occlusion 82/158, replacement BGR identity
204/211, and matched-random identity 256/263. The fixed gate remains
incomplete.
Latest A40 poll/sync at 2026-06-10 11:30:01 BST still had no compact summary:
official occlusion `774847` was running at 35:38, replacement BGR identity
`775102` was pending on `Priority` with estimated start 2026-06-11 22:02:14
BST, matched-random identity `774850` was pending on `Resources` with estimated
start 2026-06-10 23:25:59 BST, and replacement BGR/matched-random occlusions
were still dependency-pending. The original failed BGR chain `774848`/`774849`
remains ignored. The synced `summary_available.csv` still contains only
official identity 393/400, and the fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 11:34:03 BST still had no compact summary:
official occlusion `774847` was running at 39:40, replacement BGR identity
`775102` was priority-pending with estimated start 2026-06-11 22:02:14 BST,
matched-random identity `774850` was resource-pending with estimated start
2026-06-10 23:25:59 BST, and replacement BGR/matched-random occlusions remained
dependency-pending. The original failed BGR chain `774848`/`774849` remains
ignored. The fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 11:37:22 BST still had no compact summary:
official occlusion `774847` was running at 42:59, replacement BGR identity
`775102` was priority-pending with estimated start 2026-06-11 22:02:14 BST,
matched-random identity `774850` was resource-pending with estimated start
2026-06-10 23:36:39 BST, and replacement BGR/matched-random occlusions remained
dependency-pending. The original failed BGR chain `774848`/`774849` remains
ignored. The fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 11:40:24 BST still had no compact summary:
official occlusion `774847` was running at 46:01, replacement BGR identity
`775102` was priority-pending with estimated start 2026-06-11 22:02:14 BST,
matched-random identity `774850` was resource-pending with estimated start
2026-06-10 23:36:39 BST, and replacement BGR/matched-random occlusions remained
dependency-pending. The original failed BGR chain `774848`/`774849` remains
ignored. The fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 11:43:41 BST still had no compact summary:
official occlusion `774847` was running at 49:18, replacement BGR identity
`775102` was priority-pending with estimated start 2026-06-11 22:02:14 BST,
matched-random identity `774850` was resource-pending with estimated start
2026-06-10 23:36:39 BST, and replacement BGR/matched-random occlusions remained
dependency-pending. The original failed BGR chain `774848`/`774849` remains
ignored. The fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 11:48:06 BST still had no compact summary:
official occlusion `774847` was running at 53:44 on `c2-g4-17`, replacement
BGR identity `775102` was priority-pending with estimated start
2026-06-11 22:02:14 BST, matched-random identity `774850` was resource-pending
with estimated start 2026-06-10 23:36:39 BST, and replacement BGR/matched-random
occlusions remained dependency-pending. The original failed BGR chain
`774848`/`774849` remains ignored. The fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 11:51:33 BST still had no compact summary:
official occlusion `774847` was running at 57:10 on `c2-g4-17`, replacement
BGR identity `775102` was priority-pending with estimated start
2026-06-11 22:02:14 BST, matched-random identity `774850` was resource-pending
with estimated start 2026-06-10 23:36:39 BST, and replacement BGR/matched-random
occlusions remained dependency-pending. The original failed BGR chain
`774848`/`774849` remains ignored. The fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 11:55:21 BST still had no compact summary:
official occlusion `774847` had returned to pending on `BeginTime`,
replacement BGR identity `775102` was priority-pending with estimated start
2026-06-11 22:02:14 BST, matched-random identity `774850` was resource-pending
with estimated start 2026-06-10 23:36:39 BST, and replacement BGR/matched-random
occlusions remained dependency-pending. The original failed BGR chain
`774848`/`774849` remains ignored. The fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 11:59:00 BST still had no compact summary:
official occlusion `774847` was priority-pending, replacement BGR identity
`775102` was priority-pending with estimated start 2026-06-11 22:02:14 BST,
matched-random identity `774850` was resource-pending with estimated start
2026-06-10 23:36:39 BST, and replacement BGR/matched-random occlusions remained
dependency-pending. The original failed BGR chain `774848`/`774849` remains
ignored. The fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 12:02:47 BST still had no compact summary:
official occlusion `774847` was priority-pending, replacement BGR identity
`775102` was priority-pending with estimated start 2026-06-11 22:02:14 BST,
matched-random identity `774850` was resource-pending with estimated start
2026-06-10 23:36:39 BST, and replacement BGR/matched-random occlusions remained
dependency-pending. The original failed BGR chain `774848`/`774849` remains
ignored. The fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 12:05:43 BST still had no compact summary,
but matched-random identity `774850` had started on `c2-g4-17` and was running
at 0:29. Official occlusion `774847` and replacement BGR identity `775102`
remained priority-pending, with `775102` now estimated for
2026-06-11 11:36:00 BST; replacement BGR/matched-random occlusions remained
dependency-pending. The original failed BGR chain `774848`/`774849` remains
ignored. The fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 12:08:40 BST still had no compact summary.
Matched-random identity `774850` was running on `c2-g4-17` at 3:26. Official
occlusion `774847` and replacement BGR identity `775102` remained
priority-pending, with `775102` estimated for 2026-06-11 11:36:00 BST;
replacement BGR/matched-random occlusions remained dependency-pending. The
original failed BGR chain `774848`/`774849` remains ignored. The fixed gate
remains incomplete.
Latest A40 poll/sync at 2026-06-10 12:14:06 BST still had no compact summary.
Matched-random identity `774850` was running on `c2-g4-17` at 8:52. Official
occlusion `774847` was priority-pending, replacement BGR identity `775102` was
resource-pending with estimated start 2026-06-11 02:41:01 BST, replacement BGR
occlusion `775103` and matched-random occlusion `774851` remained
dependency-pending, and original failed child `774849` remained
`DependencyNeverSatisfied` and ignored. The synced `summary_available.csv`
still contains only official identity 393/400, so the fixed gate remains
incomplete.
Latest A40 poll/sync at 2026-06-10 12:20:56 BST still had no compact summary.
Matched-random identity `774850` was running on `c2-g4-17` at 15:42; official
occlusion `774847` and replacement BGR identity `775102` remained pending,
with `775102` still estimated for 2026-06-11 02:41:01 BST. Replacement BGR
occlusion `775103` and matched-random occlusion `774851` remained
dependency-pending, and original failed child `774849` remained ignored. The
fixed gate remains incomplete.
Latest A40 poll/sync at 2026-06-10 12:37:32 BST still had no compact summary:
matched-random identity `774850` was running on `c2-g4-17` at 32:18, official
occlusion `774847` was priority-pending, replacement BGR identity `775102`
was resource-pending with estimated start 2026-06-11 02:41:01 BST, replacement
BGR occlusion `775103` and matched-random occlusion `774851` remained
dependency-pending, and original failed child `774849` remained ignored. The
synced `summary_available.csv` still contains only official identity 393/400,
so the fixed gate remains incomplete.

Completed learned-policy route (negative, not active): preregistered OpenVLA-OFT occlusion-bottleneck
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
Final sync on 2026-06-10 reconstructed compact summaries from completed remote
logs. All listed jobs completed with exit `0:0`, but the fixed promotion gate
failed: non-identity totals are BGR 365/400, official 367/400, and random
369/400; identity rows are BGR 99/100, official 99/100, and random 98/100.
Per perturbation, BGR is blur 98/100, brightness 98/100, occlusion 75/100,
and shift 94/100; official is 97/98/74/98 and random is 99/98/74/98. Treat
this as negative learned-policy evidence, not an active route or paper claim.
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

Latest paper checkpoint: OpenML diabetes, blood-transfusion, phoneme,
MagicTelescope, haberman, jm1, the mixed full external/broad/secondary OpenML
suite repeats, the all-binary numeric OpenML target-1.5 32-dataset aggregation,
and OpenML positive-dataset target-sensitivity caveats are now incorporated as
replicated pre-existing supervised margin-replay evidence plus scope/fragility
evidence. This improves the "no new positive evidence",
cherry-picking, and target-fragility weaknesses but does not
change readiness because the learned-policy gate still fails.
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

Completed independent-route closure: MinAtar Freeway. The fixed pre-method
calibration command
`PYTHONPATH=src:. /tmp/bgr_minatar_venv/bin/python tools/minatar_freeway_recovery_calibration.py --out results/minatar_freeway_recovery_calibration_20seed_v1`
cleared the calibration gate with clean 0.9000, recovery range
0.4000--0.9000, RAUC 0.5667, and r80 1.8667 under `MinAtar==1.0.15`.
The authorized all-method screen
`PYTHONPATH=src:. /tmp/bgr_minatar_venv/bin/python tools/minatar_freeway_recovery_probe.py --out results/minatar_freeway_recovery_probe_4seed_v1`
is a complete tie: BGR, BGR-Coverage, BGR-uniform-radius, uniform, fixed,
failure-only, and TD-loss all have clean 0.9000, final RAUC 0.5667, median
r80 3.7000, AULC 0.5667, and best RAUC 0.5667. Low-data exploratory Freeway
scouts with init steps 0/20/100 and iterations 20/50 also tied. Do not scale
or promote Freeway without a genuinely new preregistered premise.

Latest independent-route closure: MinAtar Space Invaders. The fixed
pre-method calibration command
`PYTHONPATH=src:. /tmp/bgr_minatar_venv/bin/python tools/minatar_space_invaders_recovery_calibration.py --out results/minatar_space_invaders_recovery_calibration_20seed_v1`
cleared the calibration gate with clean 1.0000, recovery range
0.0000--1.0000, RAUC 0.4167, and r80 2.2000 under `MinAtar==1.0.15`.
The authorized all-method screen
`PYTHONPATH=src:. /tmp/bgr_minatar_venv/bin/python tools/minatar_space_invaders_recovery_probe.py --out results/minatar_space_invaders_recovery_probe_4seed_v1`
is a complete tie: BGR, BGR-Coverage, BGR-uniform-radius, uniform, fixed,
failure-only, and TD-loss all have clean 1.0000, final RAUC 0.4167, median
r80 2.2000, AULC 0.4167, and best RAUC 0.4167. Do not scale or promote Space
Invaders without a genuinely new preregistered premise.

Rejected pre-method/method scout: MinAtar Seaquest. A broad local scout found a
fragile candidate with the original heuristic controller, downward submarine-y
perturbations, burn-in 10, horizon 40, and reward threshold 1. The same window
gives clean 0.8000, recovery range 0.0000--0.8000, RAUC 0.2000, and r80 1.2000
over 20 seeds, but falls to clean 0.7333 over 30 seeds. A 4-seed method scout
is not promotable: fixed-radius replay reaches 0.1667 RAUC with clean 0.6042,
while BGR reaches only 0.0677 RAUC with clean 0.2708 and uniform reaches 0.0938
RAUC. Do not build a formal Seaquest screen unless a new preregistered
controller first clears a stable clean-success gate and avoids fixed-radius
dominance.

Completed independent-route scout: Gymnasium Blackjack package-state recovery.
This route is deliberately different from the prior MiniGrid/classic-control
screens: it uses Gymnasium `Blackjack-v1` package dynamics, exact internal
player/dealer state resets, stochastic card draws, and player-total
perturbations around decision states. A first local scout was too slow, so a
bounded CPU Slurm scout was submitted to `athena` on 2026-06-10 as job
`774192` after an initial environment-setup failure in job `774184`
(`numpy==2.4.6` is unavailable for Athena's Python 3.10). The resubmitted job
uses `/work/joy/bgr/.venv-blackjack-scout` with `gymnasium==1.3.0` and
`numpy==2.2.6`, runs 9 perturbation/target-radius configs over 8 seeds and all
required baselines, and writes to
`/work/joy/bgr/runs/blackjack_recovery_scout_20260610_080506_774192`, synced
locally to `results/blackjack_recovery_scout_8seed_v1/`. All nine configs are
negative with `candidate=False`. The best BGR-family rows by mode are:
`up` target 3.0 BGR-Coverage 0.5036 vs. uniform 0.5071 and failure-only
0.5188; `down` target 3.0 BGR-Coverage 0.3859 vs. uniform 0.3790 but
failure-only 0.3898 and only 5/3 paired wins; and `dealer_signed` target 3.0
BGR-Coverage 0.4402 vs. uniform 0.4414, failure-only 0.4498, and
BGR-uniform-radius 0.4514. A smaller local diagnostic script now exists at
`tools/blackjack_recovery_probe.py`; its default 4-seed run at
`results/blackjack_recovery_probe_4seed_scout_v1/` is also negative
(BGR-Coverage 0.3850 vs. uniform 0.3787, failure-only 0.3954, and
BGR-uniform-radius 0.3851; default BGR 0.3778). Treat the script as a reusable
diagnostic only. Do not scale or promote Blackjack without a materially new
preregistered premise.

Completed independent-route scout: Gymnasium Taxi-v3 package-state recovery.
This route uses exact tabular Taxi dynamics, resettable package states, and
Manhattan taxi-position perturbations around replay states. The default fixed
4-seed command was:
`PYTHONPATH=src:. python3 tools/taxi_recovery_probe.py --out results/taxi_recovery_probe_4seed_v1`.
It is negative and saturated: failure-only reaches 1.0000 RAUC, uniform 0.9963,
TD-loss 0.9960, BGR-uniform-radius 0.9763, BGR-Coverage 0.9650, and BGR
0.9578, with median r80 saturated at 1.0000. A uniform-only hard-budget
calibration fixed the non-saturated follow-up at 70 iterations, q-init blend
0.05, q-init noise 0.12, learning rate 0.25, and epsilon 0.10:
`PYTHONPATH=src:. python3 tools/taxi_recovery_probe.py --out results/taxi_recovery_uniform_calibration_iter70_blend005_4seed_v1 --methods uniform --iterations 70 --eval-every 35 --q-init-blend 0.05 --q-init-noise 0.12 --learning-rate 0.25 --epsilon 0.10`.
The calibration gives uniform clean 0.8458, RAUC 0.7596, and median r80 0.4007.
The fixed all-method hard-budget command was:
`PYTHONPATH=src:. python3 tools/taxi_recovery_probe.py --out results/taxi_recovery_hard_probe_4seed_v1 --methods uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr --iterations 70 --eval-every 35 --q-init-blend 0.05 --q-init-noise 0.12 --learning-rate 0.25 --epsilon 0.10`.
This is also negative: failure-only reaches 0.9692 RAUC, uniform 0.7596, fixed
0.7497, TD-loss 0.5812, BGR-Coverage 0.5696, BGR 0.5516, and
BGR-uniform-radius 0.5345; BGR and BGR-Coverage lose to uniform on all four
paired seeds. Treat Taxi as a closed negative independent route unless a
materially new preregistered premise is introduced.

Completed pre-existing-dataset broadening run: fixed broad numeric OpenML suite.
`tools/openml_margin_scout.py` now has `--broad-numeric-suite`, a predeclared
set of numeric binary OpenML datasets that pass the existing median-impute plus
standardized numeric-feature pipeline without adding categorical preprocessing:
heart-statlog, qsar-biodeg, mammography, breast-w, haberman, MagicTelescope,
eeg-eye-state, ozone-level-8hr, Bioresponse, and steel-plates-fault. A one-seed
local smoke on heart-statlog succeeded. The intended Athena run is fixed
target radius 2.0, 30 seeds, and the same uniform/fixed/BGR methods:
`PYTHONPATH=/work/joy/bgr/src:/work/joy/bgr python /work/joy/bgr/tools/openml_margin_scout.py --broad-numeric-suite --targets 2.0 --seeds 30 --out /work/joy/bgr/runs/openml_broad_numeric_target2_30seed_${SLURM_JOB_ID}`.
The first submission `774306` failed before running because Slurm executed the
wrapper under `/bin/sh` and rejected `set -o pipefail`. The corrected Bash
submission was Slurm job `774312`, completed with exit `0:0` and synced to
`results/openml_broad_numeric_target2_30seed_v1/`.
To avoid post-hoc replication selection, a held-out seeds 30--59 repeat was
also submitted before seeing the first suite summary as Slurm job `774346`:
`PYTHONPATH=/work/joy/bgr/src:/work/joy/bgr python /work/joy/bgr/tools/openml_margin_scout.py --broad-numeric-suite --targets 2.0 --seed-start 30 --seeds 30 --out /work/joy/bgr/runs/openml_broad_numeric_target2_replication_30seed_${SLURM_JOB_ID}`.
It completed with exit `0:0` and was synced to
`results/openml_broad_numeric_target2_replication_30seed_v1/`. Use
`PYTHONPATH=src:. python3 tools/analyze_openml_margin_suite.py --original results/openml_broad_numeric_target2_30seed_v1/per_seed.csv --replication results/openml_broad_numeric_target2_replication_30seed_v1/per_seed.csv`
for the deterministic readout. The broad suite is mixed and macro-negative
(pooled BGR 0.7788 vs. uniform 0.7809 and fixed 0.7830), but MagicTelescope
and haberman replicate as positive dataset-level signals. Do not present it as
a standard-environment, learned-policy, or broad macro win.

Active pre-existing-dataset broadening run: fixed secondary numeric OpenML
suite. `tools/openml_margin_scout.py` now has `--secondary-numeric-suite`, a
predeclared set of active numeric OpenML version-1 datasets that pass the
existing numeric pipeline without adding categorical preprocessing: kc2, pc2,
pc3, pc4, mc1, jm1, hill-valley, madelon, gina_agnostic, and electricity.
Local one-seed smokes over kc2 and the full secondary suite succeeded before
submission. The queued Athena route is fixed target radius 2.0, 30 seeds, and
the same uniform/fixed/BGR methods:
`PYTHONPATH=/work/joy/bgr/src:/work/joy/bgr /work/joy/bgr/.venv-openml-broad/bin/python /work/joy/bgr/tools/openml_margin_scout.py --secondary-numeric-suite --targets 2.0 --seeds 30 --out /work/joy/bgr/runs/openml_secondary_numeric_target2_30seed_${SLURM_JOB_ID}`.
To avoid post-hoc replication selection, the held-out seeds 30--59 repeat was
submitted in the same launcher invocation:
`PYTHONPATH=/work/joy/bgr/src:/work/joy/bgr /work/joy/bgr/.venv-openml-broad/bin/python /work/joy/bgr/tools/openml_margin_scout.py --secondary-numeric-suite --targets 2.0 --seed-start 30 --seeds 30 --out /work/joy/bgr/runs/openml_secondary_numeric_target2_replication_30seed_${SLURM_JOB_ID}`.
Slurm jobs are `776728` for the original run and `776729` for the held-out
replication. At submission, `776728` was running on `cnode403` and `776729`
was pending on resources. Treat this as an active broadening route only, not a
paper claim.

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
  A 2026-06-10 follow-up fixed the calibration script so `--horizon` controls
  Gymnasium `max_episode_steps` and added a materially different
  `scripted_push_sweep` controller. The named compact artifact
  `results/fetchpush_object_goal_calibration_sweep_g8_h250_2seed_v1/summary.json`
  gives clean 0.8750, recovery range 0.7500--0.8750, RAUC 0.7812, and
  `decision=reject-calibration-radius-saturated`; the wider-radius artifact
  `results/fetchpush_object_goal_calibration_sweep_g8_h250_xwide_2seed_v1/summary.json`
  gives the same clean/range with RAUC 0.7583 and
  `decision=reject-calibration-radius-saturated`. This improves controller
  clean success but still does not create a non-saturated boundary for BGR.
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
  A 60-seed target-radius sensitivity run over the three positive OpenML
  datasets has also completed:
  `PYTHONPATH=src:. python3 tools/openml_margin_scout.py --datasets diabetes,blood-transfusion-service-center,phoneme --targets 1.0,1.5,2.0 --seeds 30 --out results/openml_positive_target_sensitivity_30seed_v1`.
  Its held-out repeat is:
  `PYTHONPATH=src:. python3 tools/openml_margin_scout.py --datasets diabetes,blood-transfusion-service-center,phoneme --targets 1.0,1.5,2.0 --seed-start 30 --seeds 30 --out results/openml_positive_target_sensitivity_replication_30seed_v1`.
  Treat it as a fragility caveat, not a new headline: pooled BGR-minus-uniform
  gaps for diabetes/blood/phoneme are +0.002/-0.002/-0.007 at radius 1.0,
  +0.032/+0.065/+0.014 at 1.5, and +0.038/+0.086/+0.035 at 2.0.
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
  `tools/lunarlander_recovery_probe.py` first completed as a negative 4-seed
  near miss: BGR-Coverage had the best mean final RAUC (0.7500 vs. 0.6222
  uniform, 0.7375 fixed, 0.6799 failure-only, 0.7174 TD-loss, and 0.7160
  BGR-uniform-radius), but it won only 2/4 paired seeds against uniform and had
  lower median r80 than uniform (0.4200 vs. 0.4825). A fixed 30-seed stress
  test was then queued on `athena` as split method jobs `782056`--`782062`
  after adding `scripts/queue_lunarlander_probe.sh` and
  `scripts/sync_lunarlander_probe.sh`. The fast method jobs initially failed
  after writing `results.json` because the reused remote venv had `pygame`
  metadata rather than `pygame-ce`; the summaries were reconstructed from the
  completed JSON rows, and `tools/lunarlander_recovery_calibration.py` now
  accepts either package metadata name. The slow `failure_only` job completed
  normally. The merged 30-seed artifact is
  `results/lunarlander_recovery_probe_30seed_v3_782056_782062/`. It closes the
  route negative: BGR-Coverage is 0.7193 vs. uniform 0.7006, fixed 0.6730,
  failure-only 0.6196, TD-loss 0.7056, and BGR-uniform-radius 0.7031, but
  paired signs are only W/L/T=15/15/0 against uniform and median r80 is lower
  than uniform (0.3650 vs. 0.3863). Default BGR is worse at 0.6742. Do not
  scale or promote this LunarLander route without a genuinely new
  preregistered premise. The fixed target-0.70 follow-up is now also completed
  negative. It was queued because the earlier 4-seed target-radius scout
  repaired the median-r80 contradiction on its small sample while keeping high
  BGR-Coverage RAUC. The queue wrapper accepts `EXTRA_ARGS`; target-0.70 split
  all-method jobs ran on Athena as uniform/fixed/failure-only/
  TD-loss/BGR-uniform-radius/BGR-Coverage/BGR `782561`--`782567`, with
  `EXTRA_ARGS='--target-radius 0.70'` and 30 seeds. The merged artifact
  `results/lunarlander_recovery_probe_30seed_target070_merged/` is rejected by
  the promotion checker: BGR-Coverage is 0.6886 final RAUC versus uniform
  0.7006 (W/L/T=11/19/0), TD-loss 0.7056, fixed 0.6730, failure-only 0.6196,
  and BGR-uniform-radius 0.6777. Median r80 is no longer contradictory
  (0.4143 vs. uniform 0.3863), but the method loses to uniform and TD-loss.
  This route is closed negative and is not paper evidence.
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
- Additional same-family DeepSea changed-premise scouts on 2026-06-10 also do
  not open an independent-benchmark route. `size20_lowrows` is all-zero and
  saturated. `size20_boundaryheavy` gives BGR-Coverage 0.0219 RAUC vs. uniform
  0.0156 but trails TD-loss 0.0266 and is below the +0.01 scout gap.
  `size16_late` is effectively tied across methods. `size10_short` gives
  default BGR 0.1808 vs. uniform 0.1406, but wins only 1/4 paired seeds, trails
  TD-loss 0.1875, and has lower median r80 than uniform. Treat these untracked
  scouts as rejected diagnostics, not paper evidence or scale-up candidates.
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
