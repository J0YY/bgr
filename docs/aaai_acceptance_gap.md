# AAAI Acceptance Gap

This is an internal working note, not part of the anonymous submission package.

## Current Assessment

The paper is now framed as a mechanism study and the strongest overclaim risks
are explicit in the title, introduction, limitations, README, and package
guards. That improves reviewability, but it does not by itself make the paper a
90%+ AAAI main-track accept. The remaining acceptance blocker is evidence:

A read-only independent Codex review on 2026-06-11 scored the current paper
`3/6`, below the working target of at least `5/6`. The review agreed with the
internal readiness gate: the paper is honest and coherent, but it still needs a
clean standard-environment recovery win or a learned-policy win over both the
official checkpoint and matched random to move from mechanism study to likely
AAAI accept.

- no clear positive result on a standard recovery environment, despite
  replicated OpenML diabetes, blood-transfusion, and phoneme
  pre-existing-dataset margin results;
- no stable learned-policy OpenVLA/LIBERO improvement over both matched random
  and the official checkpoint;
- suffix and synthetic gains remain small even though paired consistency is
  strong;
- the strongest result is still the controlled procedural grid-margin
  mechanism benchmark.

Audit this current status from artifacts with:
`PYTHONPATH=src:. python3 scripts/check_acceptance_readiness.py --root .`.
Quantify the distances to those gates with:
`PYTHONPATH=src:. python3 scripts/acceptance_scorecard.py --root . --out docs/acceptance_scorecard.md`.
The checker is intentionally stricter than the package guard: package checks
verify that claims match artifacts, while this readiness check verifies whether
the evidence has actually cleared the independent-benchmark and learned-policy
promotion gates.

The generated scorecard now reports why each standard-environment recovery
screen fails the gate. As of the 2026-06-07 Asterix refresh, the MinAtar
Asterix route cleared calibration, but the fixed all-method screen is negative
because failure-only replay is the strongest baseline. The earlier MinAtar
Breakout calibration also cleared pre-method checks, but its fixed all-method
screen is tied/saturated negative.
Two new 2026-06-11 CPU scouts also fail before scale-up. A deterministic-action
bsuite DeepSea scout exposes `--no-randomize-actions` in
`tools/bsuite_deepsea_recovery_probe.py`; its best row
`results/bsuite_deepsea_deterministic_t085_mix080_scout_4seed_v1/summary.csv`
has `bgr_coverage` 0.1594 RAUC versus uniform 0.1031 and positive mean gaps
against fixed, failure-only, TD-loss, and the uniform-radius ablation, but the
candidate checker rejects it because median `r80` is lower than uniform. The
bsuite package also warns deterministic actions are debug mode. A MiniGrid
SimpleCrossing S9N3 scout, enabled by expanding
`tools/minigrid_lavacrossing_recovery_probe.py`, has an 8-seed mean gap for
`bgr_coverage` (0.5714 vs. 0.4746 uniform), but paired wins are weak: 3/8 vs.
uniform, 2/8 vs. failure-only, 1/8 vs. TD-loss, and 3/8 vs.
BGR-uniform-radius. Harder-budget SimpleCrossing variants are dominated by
uniform/failure-only/uniform-radius. Do not scale either route without a
materially new fixed premise.
The 2026-06-11 bsuite Cartpole Swingup scout is completed negative before
scale-up. It differs from the retired upright bsuite Cartpole route by using
bsuite's `CartpoleSwingup` dynamics, exact `CartpoleState` restarts, and a
fixed near-upright recovery interface. The fixed 4-seed Slurm job `782844`
completed with exit `0:0`; the synced summary is
`results/bsuite_cartpole_swingup_recovery_probe_4seed_v1_782844/summary.csv`.
Default BGR reaches 0.1044 RAUC versus uniform 0.0761 with W/L/T=3/1/0, but it
trails failure-only 0.1287 and TD-loss 0.1456. BGR-Coverage reaches only
0.0806 RAUC and fails the uniform gate. Both BGR variants have median r80
1.0000, tied with uniform, so the route also fails by radius ceiling
saturation. Do not scale Cartpole Swingup without a materially new fixed
premise.
The 2026-06-10 Freeway route also does not change this: it cleared pre-method
calibration under MinAtar package dynamics, but the fixed all-method screen is
a complete tie across BGR, BGR-Coverage, BGR-uniform-radius, uniform, fixed,
failure-only, and TD-loss at 0.5667 final RAUC, clean 0.9000, median r80
3.7000, and AULC 0.5667. Low-data Freeway scouts also tied, so this route
should not be scaled without a new premise.
The same 2026-06-10 MinAtar pass also closes Space Invaders. Space
Invaders cleared pre-method calibration with clean 1.0000, recovery range
0.0000--1.0000, RAUC 0.4167, and r80 2.2000, but the fixed all-method screen
is a complete tie across BGR, BGR-Coverage, BGR-uniform-radius, uniform, fixed,
failure-only, and TD-loss at 0.4167 final RAUC, clean 1.0000, median r80
2.2000, and AULC 0.4167.
The 2026-06-11 MinAtar Seaquest package-state route also fails before scale-up.
The new fixed 30-seed pre-method calibration
`results/minatar_seaquest_recovery_calibration_30seed_v1/summary.json` is
usable, with clean survival 1.0000, recovery range 0.5333--1.0000, RAUC
0.9267, and r80 4.3333 under leftward submarine-column perturbations. The
fixed 4-seed method screen
`results/minatar_seaquest_recovery_probe_4seed_v1/summary.csv` is negative:
BGR-Coverage gives 0.9017 RAUC versus uniform 0.8958, fixed 0.9017,
failure-only 0.9025, TD-loss 0.8992, and BGR-uniform-radius 0.9192, with only
2/4 paired wins versus uniform and saturated median r80 5.0000 for every
method. Default BGR is lower at 0.8892. Do not scale Seaquest without a
materially new preregistered premise.
New 2026-06-11 independent-benchmark route opened: Gymnasium Box2D
`BipedalWalker-v3` mid-gait recovery. Athena calibration job `783139`
completed with exit `0:0` and synced to
`results/bipedalwalker_recovery_calibration_12seed_v1_783139/`. The fixed
12-seed pre-method readout is usable: clean success 1.0000, recovery range
0.0000--1.0000, mean RAUC 0.7024, and median r80 0.7000 using exact Box2D
body-state restoration after an 80-step package-heuristic burn-in, Gymnasium's
`BipedalWalkerHeuristics`, and an 80-step no-fall/progress recovery target.
This only permitted the fixed all-method screen. The plain BipedalWalker-v3
screen is completed negative after split jobs
`783152`/`783153`/`783158`/`783159`/`783160`/`783161`/`783162`. The merged
summary is
`results/bipedalwalker_recovery_probe_4seed_split_merged_v1_783152_783162/summary.csv`.
Default BGR reaches 0.3743 RAUC versus uniform 0.3519 but has only 2/4 paired
wins, loses to fixed-radius 0.4312, and has a median-r80 contradiction.
BGR-Coverage is 0.2716 and loses to uniform, fixed, failure-only, TD-loss, and
BGR-uniform-radius. Treat the route as negative scope evidence, not a standard
benchmark win. A harder same-package follow-up was run only because
`BipedalWalkerHardcore-v3` also cleared pre-method calibration: Athena job
`783167` reports clean success 0.9167, recovery range 0.0000--1.0000, mean
RAUC 0.7560, and median r80 1.0000. The fixed all-method split jobs
`783168`--`783174` are also completed negative. The merged hardcore summary is
`results/bipedalwalker_hardcore_recovery_probe_4seed_split_merged_v1_783168_783174/summary.csv`:
fixed 0.4405, BGR-uniform-radius 0.4319, TD-loss 0.2604, failure-only 0.2526,
uniform 0.2314, BGR-Coverage 0.2303, and default BGR 0.1868 mean final RAUC.
The candidate checker rejects both BGR variants against uniform, fixed,
failure-only, TD-loss, and the uniform-radius ablation. Do not scale either
BipedalWalker route without a genuinely new preregistered premise.
The latest OpenVLA/LIBERO occlusion-bottleneck route is completed negative:
BGR reaches 365/400 non-identity successes versus official 367/400 and matched
random 369/400, with identity rows BGR 99/100, official 99/100, and random
98/100. This replaces the older perturb-only audit as the latest
learned-policy gate read and remains a failure against the +10/400 and +0.02
promotion rule.
The 2026-06-10 hard-occlusion follow-ups are also not promotable. The completed
0.80 transfer route has BGR occlusion 305/400 versus official and matched
random at 296/400, one episode short of the +10/400 margin gate, and identity
BGR 391/400 versus official 393/400. The completed 0.65 adapted route has BGR
identity/occlusion 389/400 and 301/400, official 393/400 and 297/400, and
matched random 390/400 and 304/400; it fails by identity, matched-random
occlusion, and the small official margin. The A40 0.65 adapted fallback remains
already non-promotable on partial identity rows. The completed 0.80
identity-anchored route is also negative: BGR identity/occlusion are 389/400
and 303/400, official is 393/400 and 296/400, and matched random is 393/400
and 302/400. It fails the fixed gate with episode_margin=1,
rate_margin=+0.0025 against the best occlusion comparator, and
identity_deficit=4. Remaining identity-anchored 0.80/0.90 variants are
incomplete or already non-promotable on partial identity rows and must not be
promoted unless complete summaries pass the fixed gate.
Because the 0.80 transfer route was the closest learned-policy result, a
held-out occlusion-only confirmation was queued on 2026-06-11 as a router-style
diagnostic rather than a paper claim. It evaluates the same completed 0.80
transfer BGR and matched-random checkpoints plus the official checkpoint on
hard occlusion 0.80 only. The first attempt, official/BGR/random
`782604`/`782605`/`782606`, incorrectly used `EVAL_INIT_STATE_OFFSET=40` with
`EVAL_TRIALS=80`; LIBERO-Goal has only 50 initial states per task, so official
failed at index 50 and BGR/random were cancelled. The corrected held-out slice
uses the remaining 10 initial states per task: official `782609`, BGR `782610`,
and matched random `782611` run with `EVAL_INIT_STATE_OFFSET=40`,
`EVAL_TRIALS=10`, `EVAL_TASKS=10`, `EVAL_SEED=137`, and `SAVE_ROLLOUTS=0`.
The interpretable readout is the combined 500 unique occlusion episodes from
the original 400 plus this held-out 100. This route can only motivate a
preregistered full router-style gate if combined BGR beats both comparators by
at least 0.02 absolute occlusion success rate; identity success would come from
the official fallback branch and must not be silently mixed into the existing
non-router gate. Poll/sync with:
`ARTIFACT=openvla_oft_perturb_eval_occlusion_bottleneck_hardocc080_transfer_step50400_lr2em7_heldout_offset40_trials10_v1 JOB_IDS=782609,782610,782611 DETAIL_JOB_IDS=782609,782610,782611 ROUTE_LABEL='Hard-occlusion 0.80 transfer held-out offset40 trials10 confirmation' scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh --poll --sync --no-check`.
Final 2026-06-11 04:38 BST sync closes this held-out confirmation as negative:
official and matched random are both 71/100, while BGR is 69/100. The combined
original-plus-held-out readout is BGR 374/500 versus official/random 367/500,
a +7 episode margin below the +10 and +0.02 router-style requirement. Treat the
0.80 transfer confirmation as closed negative unless a later audit discovers a
parsing error.
The alpha-0 official-head/full-LoRA no-video occlusion-only scout is a separate
fallback diagnostic. Latest 2026-06-11 04:22 BST partial is BGR 216/312,
official 209/308, and matched random 213/312 on hard occlusion 0.80. This is
incomplete and the margin is far below the router-style promotion requirement,
so it is not paper evidence. Do not formalize that alpha-0 fallback unless the
complete row clears the same held-out occlusion margin.
Final 2026-06-11 04:33 BST sync closes that hard-occlusion 0.80 alpha-0
no-video occlusion-only scout as non-promotable: BGR is 301/400, official is
298/400, and matched random is 298/400, only a +3 episode margin over both
comparators and far below the required +10/400 and +0.02 router-style
threshold. Treat this as another learned-policy near miss, not a paper claim.
A harder occlusion-only alpha-0 no-video scout was queued as a
new diagnostic severity, reusing the same official-head/full-LoRA checkpoints
without retraining and evaluating only occlusion fraction 0.90. Submitted jobs
are official `782638`, BGR `782639`, and matched random `782640`, writing to
`/work/joy/bgr/runs/openvla_oft_perturb_eval_occlusion_bottleneck_hardocc090_transfer_headinterp000_lorafull_novideo_occscout_v1`.
This remains a router-premise scout only; it is useful only if BGR beats both
comparators by at least +10/400 and +0.02 on hard occlusion. Poll/sync with:
`ARTIFACT=openvla_oft_perturb_eval_occlusion_bottleneck_hardocc090_transfer_headinterp000_lorafull_novideo_occscout_v1 JOB_IDS=782638,782639,782640 DETAIL_JOB_IDS=782638,782639,782640 ROUTE_LABEL='Hard-occlusion 0.90 alpha0 no-video occlusion-only fallback scout' scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh --poll --sync --no-check`.
Final local sync before cancellation has BGR 123/194, official 93/159, and
matched random 82/132. This was still incomplete and not a fixed 400-episode
result; BGR's partial edge is below the required +10/400 and +0.02 full-gate
standard, so jobs `782638`--`782640` were cancelled to free GPUs for the
trained router route. Treat the 0.90 alpha-0 no-video scout as closed
non-promotable partial evidence for the router premise.
A fixed full 400-episode rerun of the same hard-occlusion 0.90 alpha-0
no-video scout was queued on 2026-06-11 only to close the unequal canceled
partial cleanly. This was a router-premise diagnostic, not paper evidence.
It used
`TAG=occlusion_bottleneck_hardocc090_transfer_headinterp000_lorafull_novideo_fullrerun_v1`,
`ALPHA=0.0`, `LORA_B_SCALE=1.0`, `PERTURBATIONS='occlusion={"fraction":0.90}'`,
`EVAL_TASKS=10`, `EVAL_TRIALS=40`, `EVAL_SEED=37`, and `SAVE_ROLLOUTS=0`.
Submitted jobs are prep `782931`, official occlusion `782932`, BGR occlusion
`782933`, and matched-random occlusion `782935`; all completed with exit
`0:0`. Final reconstructed local summary at
`results/openvla_oft_perturb_eval_occlusion_bottleneck_hardocc090_transfer_headinterp000_lorafull_novideo_fullrerun_v1/summary_available.csv`
has BGR 307/400, official 305/400, and matched random 303/400. BGR beats both
comparators, but only by +2 and +4 episodes (+0.005 and +0.010), far below the
fixed +10/400 and +0.02 router gate. Treat this as another learned-policy
near miss, not a paper claim.

Because the hard-0.90 severity produced a small non-promotable BGR edge, a
router-specific hard-0.90 occlusion-only training premise was queued on
2026-06-11 and is now closed negative. This was not another same-checkpoint
re-evaluation: it trained matched BGR and random branches on hard-occlusion
0.90 examples only, while a hypothetical router would keep the official
checkpoint for clean identity. Fixed
configuration:
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
not sync or interpret those rows. Final poll/sync at 2026-06-11 10:23:38 BST
closed the route negative. The clean/adapt summary has BGR 391/400 and matched
random 391/400. The hard-occlusion summary has BGR 305/400, official 305/400,
and matched random 301/400. BGR ties official and beats matched random by only
+4/400 (+0.0100), far below the +10/400 and +0.02 router gate. Reproduction
sync:
`PREP_TAG=p512unique_occonly_hardocc090_router_prereg ADAPT_TAG=occonly_p512unique_hardocc090_router_step50300_lr5em7_identitylora_imageaug_officialtrainstats_v1 PERTURB_TAG=occonly_p512unique_hardocc090_router_step50300_lr5em7_identitylora_imageaug_officialtrainstats_hardocc090_occonly_fullgoal10x40_v1 JOB_IDS=783002,783003,783004,783005,783006,783007,783008,783034,783035,783036 DETAIL_JOB_IDS=783002,783003,783004,783006,783007,783034,783035,783036 ROUTE_LABEL='Hard-occlusion 0.90 occlusion-only router-trained OpenVLA-OFT premise' GATE_PERTURBATIONS=occlusion scripts/sync_openvla_oft_occlusion_bottleneck_results.sh --poll --sync --no-check`.
A low-cost occlusion-fraction 0.75 scout was also queued for the original
occlusion-bottleneck transfer checkpoints after the 0.90 trained-router partial
looked weak. It uses a fresh artifact
`openvla_oft_perturb_eval_occlusion_bottleneck_transfer_occ075_scout_v1`,
`PERTURBATIONS='occlusion={"fraction":0.75}'`, `EVAL_TASKS=10`,
`EVAL_TRIALS=40`, `EVAL_SEED=37`, and `SAVE_ROLLOUTS=0`; submitted jobs are
official/BGR/matched-random `783104`/`783105`/`783106`. Latest poll/sync at
2026-06-11 11:42:26 BST showed official `783104` and BGR `783105` completed
with exit `0:0`, while matched random `783106` was still pending after a
delayed partial log. The incomplete summary has BGR 293/400, official 295/400,
and matched random 279/378, so BGR already trails official on the fixed
400-episode readout and trails matched-random by success rate. Because the
scout was already non-promotable, `783106` was cancelled to avoid spending
more GPU time on a dead comparator. This scout is not a basis for another
fixed route. This is only a severity-window diagnostic for whether a future
preregistered router-style gate is worth running, not paper evidence and not a
moved gate.
A learned-policy scout was queued after that closure: combined
occlusion+shift visual corruption for the completed occlusion-bottleneck
transfer checkpoints. This is a different perturbation family, implemented by
adding `occlusion_shift` support to `scripts/queue_openvla_oft_perturb_eval.sh`
so the existing central occlusion and zero-padded image shift are applied in
sequence. The scout artifact is
`openvla_oft_perturb_eval_occlusion_bottleneck_combo_occ080_shift015_scout_v1`,
with
`PERTURBATIONS='occlusion_shift={"fraction":0.80,"dx_fraction":0.15,"dy_fraction":0.0}'`,
`EVAL_TASKS=10`, `EVAL_TRIALS=10`, `EVAL_SEED=237`, and `SAVE_ROLLOUTS=0`.
Athena jobs are official `783312`, BGR `783314`, and matched random `783315`;
the route closed early as non-promotable at the 2026-06-11 13:20:29 BST sync.
Official completed with exit `0:0` and 69/100 successes. BGR had only 66/98
successes, so even a perfect finish could reach at most 68/100 and could not
beat official, much less clear the +5/100 and +0.05 route-selection threshold.
Matched random had 28/51 successes. BGR and matched-random jobs were cancelled
at 13:20 BST to save GPU time. The remote full summary is missing; the local
incomplete summary is
`results/openvla_oft_perturb_eval_occlusion_bottleneck_combo_occ080_shift015_scout_v1/summary_available.csv`.
This is negative/non-promotable route-selection evidence only. Do not submit
`scripts/queue_openvla_oft_occlusion_shift_combo_gate.sh --submit` for this
premise, and do not incorporate this route into `paper/main.tex`.
The checker command remains:
`scripts/check_openvla_route_scout.py results/openvla_oft_perturb_eval_occlusion_bottleneck_combo_occ080_shift015_scout_v1/summary_available.csv`
for reproducing the closure. It returns incomplete because the BGR/random jobs
were intentionally cancelled after the route became mathematically
non-promotable; it is not a promotion signal.
A pre-artifact CarRacing-v3 controller smoke was also rejected on 2026-06-11:
the available `/tmp/bgr_lunar_venv` Gymnasium Box2D install can instantiate
`CarRacing-v3`, but a simple track-following controller visited only about
1.5% of track tiles from clean starts, and a small steering-sign/gain/lookahead
sweep produced no setting above 5%. Do not create a CarRacing recovery route
without first finding a materially better controller or policy.
Three low-cost independent-benchmark CPU scouts were queued at 2026-06-11
10:45 BST and completed negative by 10:49 BST using the existing official
MiniGrid package-state recovery harness and new reusable wrappers
`scripts/queue_minigrid_lavacrossing_probe.sh` and
`scripts/sync_minigrid_lavacrossing_probe.sh`. They tested unscaled,
materially different official variants after the SimpleCrossing S9N3 rows
failed paired and radius checks: SimpleCrossing S9N2 job `783136`,
SimpleCrossing S9N1 job `783137`, and LavaCrossing S11N5 job `783138`. All use
seeds 0--7 and the full all-method set. All fail the candidate checker: S9N2
BGR-Coverage has 0.7226 vs. 0.7042 uniform but only 2/8 paired wins and trails
TD-loss; S9N1 is dominated by uniform, failure-only, TD-loss, and
BGR-uniform-radius; S11N5 BGR-Coverage has 0.5052 vs. 0.5590 uniform and BGR
has 0.5087 vs. 0.5590 uniform. Do not scale or promote these variants without
a materially new fixed premise.
A new router-specific occlusion-only training premise was queued on
2026-06-11 after the 0.80 held-out confirmation failed. This is not another
same-checkpoint re-evaluation: `scripts/queue_openvla_oft_preregistered_occlusion_bottleneck.sh`
now supports `INCLUDE_CLEAN_ANCHORS=0`, so the matched BGR and random branches
train only on hard-occlusion 0.80 replay examples while a hypothetical router
would keep the official checkpoint for clean identity. The fixed configuration
uses `PREP_TAG=p1024unique_occonly_hardocc080_router_prereg`,
`OCCLUSION_CAP=1024`, `OCCLUSION_REPEAT=1`, `OCCLUSION_FRACTION_OVERRIDE=0.80`,
`PROXIMAL_ANCHOR_L2=0`, `ADAPT_STEPS=300`, `LR=5e-7`, official stats,
identity-LoRA, image augmentation, and a 10-task x 40-trial occlusion-only eval
at `EVAL_SEED=37`. Submitted jobs are prep `782649`, BGR train/merge/clean-eval
`782650`/`782651`/`782652`, matched-random train/merge/clean-eval
`782653`/`782654`/`782655`, and hard-occlusion evals official/BGR/random
`782656`/`782657`/`782658`. It can only matter if BGR beats both official and
matched random by at least +10/400 and +0.02 on the fixed occlusion readout.
Poll/sync with:
`PREP_TAG=p1024unique_occonly_hardocc080_router_prereg ADAPT_TAG=occonly_p1024unique_hardocc080_router_step50300_lr5em7_identitylora_imageaug_officialtrainstats_v1 PERTURB_TAG=occonly_p1024unique_hardocc080_router_step50300_lr5em7_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1 JOB_IDS=782649,782650,782651,782652,782653,782654,782655,782656,782657,782658 DETAIL_JOB_IDS=782649,782650,782651,782653,782654,782656,782657,782658 ROUTE_LABEL='Hard-occlusion 0.80 occlusion-only router-trained OpenVLA-OFT premise' GATE_PERTURBATIONS=occlusion scripts/sync_openvla_oft_occlusion_bottleneck_results.sh --poll --sync --no-check`.

The first router-training attempt failed in prep before any adaptation claim:
matched-random perturbation rendering selected zero occlusion examples before
the post-render family filter, so prep `782649` exited `1:0`; dependent jobs
`782650`--`782655`/`782657`/`782658` were cancelled, and official-only eval
`782656` was cancelled as an orphan. The fix is now in the queue path:
`scripts/render_openvla_teacher_examples.py` supports a pre-selection
`--include-family occlusion` filter, and
`scripts/queue_openvla_oft_preregistered_occlusion_bottleneck.sh` passes it to
both BGR and matched-random perturb renders. Remote manifest counts show 1,600
BGR occlusion rows but only 512 matched-random occlusion rows, so the corrected
fair route uses a matched cap of 512 and repeats both sources twice rather than
letting BGR train on twice as many unique occlusion examples. Corrected active
route, queued 2026-06-11 04:55 BST: prep `782671`, BGR train/merge/clean-eval
`782672`/`782673`/`782674`, matched-random train/merge/clean-eval
`782675`/`782676`/`782677`, and official/BGR/random hard-occlusion evals
`782679`/`782680`/`782681`. It uses
`PREP_TAG=p512unique_occonly_hardocc080_router_randfix_prereg`,
`OCCLUSION_CAP=512`, `OCCLUSION_REPEAT=2`, `INCLUDE_CLEAN_ANCHORS=0`,
`OCCLUSION_FRACTION_OVERRIDE=0.80`, `PROXIMAL_ANCHOR_L2=0`,
`ADAPT_STEPS=300`, `LR=5e-7`, and the same 10-task x 40-trial occlusion-only
eval. It is still only router-premise evidence unless BGR beats both official
and matched random by at least +10/400 and +0.02 on hard occlusion. Poll/sync:
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
`paper/main.tex`; it is another learned-policy negative result. No active
learned-policy cluster jobs remain queued for BGR after this closure; current
Athena GPU jobs are separate `rl4vla-*` work and should not be modified by this
project.
A fixed head-interpolation follow-up was queued on 2026-06-10 to test whether
the near-miss 0.80 transfer route can preserve the occlusion gain while
recovering identity success. It copies the completed BGR and matched-random
occlusion-bottleneck checkpoints, interpolates trainable heads toward the
official checkpoint with `ALPHA=0.75`, scales LoRA-B tensors by the same alpha,
and evaluates identity plus occlusion fraction 0.80 over 10 LIBERO-Goal tasks x
40 trials. Submitted jobs are prep `779973`, official `779974`/`779975`, BGR
`779976`/`779977`, and matched random `779978`/`779979`. Latest poll/sync at
2026-06-10 22:03 BST showed prep `779973` completed at 21:16:12 and official
identity `779974` completed at 393/400; BGR identity `779976` was still
running after 18:28 on `c1-g4-03`, matched-random identity `779978` was
running after 2:53 on `c1-g4-05`, official occlusion `779975` was
priority-pending, and BGR/random occlusion evals were dependency-pending behind
their identity evals. The synced incomplete summary has only the official
identity row; live log tails showed BGR identity around 89/94 and matched
random around 20/20, which is not gateable evidence. Latest poll/sync at
2026-06-10 22:13 BST closed Route A early as non-promotable on identity:
BGR identity reached 134/144, so even perfect success afterward could finish no
better than 390/400 against official identity 393/400. The remaining Route A
jobs were cancelled to save cluster time: official occlusion `779975`, BGR
identity `779976`, BGR occlusion `779977`, matched-random identity `779978`,
and matched-random occlusion `779979` are all cancelled. Route A is negative
and not paper evidence. A second fixed
head-only repair route was queued before any head-interpolation summary was
available: it keeps the same `ALPHA=0.75` action/proprio head interpolation but
sets `LORA_B_SCALE=1.0`, preserving the adapted LoRA-B tensors rather than
shrinking them with the heads. This tests the concrete failure mode from the
0.80 transfer near miss: repair identity without discarding the occlusion
adaptation. Submitted jobs are prep `780059`, official `780060`/`780061`, BGR
`780062`/`780063`, and matched random `780064`/`780065`; latest poll/sync at
2026-06-10 22:03 BST showed prep `780059` running after starting at 22:03:06,
official identity `780060` pending on resources with estimated start
2026-06-11 06:21:43 BST, and downstream evals dependency-pending, with no logs
or summary. Latest poll/sync at 2026-06-10 22:13 BST showed official identity
`780060` running, BGR identity `780062` and matched-random identity `780064`
priority-pending, and all occlusion jobs
dependency-pending. Latest poll/sync at 2026-06-10 22:20 BST showed official
identity `780060`, BGR identity `780062`, and matched-random identity `780064`
running, with occlusion jobs `780061`/`780063`/`780065` still
dependency-pending. Live identity tails were official 131/137, BGR 51/53, and
matched random 26/27, so the route is neither promotable nor impossible yet.
Do not launch another overlapping OpenVLA variant while this LoRA-full
head-interpolation route is unresolved. Latest poll/sync at 2026-06-10
22:33 BST showed all three identity jobs requeued or preempted back to
`PENDING` on `Priority`: official `780060` and BGR `780062` have estimated
starts at 2026-06-11 10:04:52 BST, matched-random `780064` has estimated start
at 2026-06-11 17:11:00 BST, and all occlusion jobs remain dependency-pending.
The full summary is still missing; direct identity log tails before preemption
reached official 178/185, BGR 112/119, and matched random 65/68. This remains
incomplete non-evidence; BGR's seven identity failures weaken the side
condition but do not yet mathematically close the route because comparator
identity rows are incomplete. Latest poll/sync at 2026-06-10 22:53 BST showed
the LoRA-full route closed by infrastructure failure rather than a complete
gate result: official/BGR/matched-random identity jobs
`780060`/`780062`/`780064` restarted and failed quickly with exit `1:0` after
00:50/00:23/00:28, leaving occlusion jobs `780061`/`780063`/`780065` on
`DependencyNeverSatisfied`. Those dependent occlusion jobs were cancelled at
22:54 BST. No full `summary.csv` exists, so this route is not a paper result
and should not be rerun unchanged without first diagnosing the restarted
identity-eval failures. The restart failure was diagnosed as infrastructure:
`/work/joy` was full, the jobs stopped without a Python traceback, and
`sacct` showed low MaxRSS rather than OOM. The perturb-eval queue now supports
`SAVE_ROLLOUTS=0`, patching the remote LIBERO video hook to skip MP4 writes.
Obsolete reproducible alpha-0.75 interpolation copies and a UV temp cache were
removed on Athena, raising free `/work/joy` space from about 1.6G to 5.6G.
A fixed alpha-0.0 official-head/full-LoRA no-video repair was queued as a
new premise rather than an unchanged rerun. It restored official action/proprio
heads exactly while preserving adapted LoRA tensors (`ALPHA=0.0`,
`LORA_B_SCALE=1.0`, `SAVE_ROLLOUTS=0`). Submitted jobs were prep `782410`,
official `782411`/`782412`, BGR `782413`/`782414`, and matched random
`782415`/`782416`, excluding `c2-g4-17,c2-g4-18,c2-g4-19,c2-g4-21,c2-g4-23`.
Latest poll/sync at 2026-06-11 01:33:40 BST showed prep completed, official,
BGR, and matched-random identity jobs running, and occlusion jobs still
dependency-pending. The early partial summary had only identity rows over
8--14 episodes, all at 100%, so this is not paper evidence. Latest poll/sync
at 2026-06-11 01:48:36 BST still showed identity jobs
`782411`/`782413`/`782415` running and occlusion jobs `782412`/`782414`/`782416`
dependency-pending. The synced incomplete summary has only identity rows:
BGR 136/143, official 141/147, and matched random 138/144; direct live tails
were slightly ahead at BGR 137/144, official 143/149, and matched random
139/145. The route is not gateable yet, and BGR's identity side-condition is
precarious because it currently has one more failure than the best comparator.
The final identity rows then closed the route negative before occlusion:
BGR identity `391/400`, official `393/400`, and matched random `392/400`.
BGR trails the best identity comparator by two episodes, so the fixed
identity-preservation side-condition fails. Occlusion jobs
`782412`/`782414`/`782416` were cancelled at 2026-06-11 02:14 BST after only
tiny partial fragments, so no occlusion comparison should be interpreted. The
local compact closure artifact is
`results/openvla_oft_perturb_eval_occlusion_bottleneck_hardocc080_transfer_headinterp000_lorafull_novideo_v1/summary_available.csv`.
Use the sync command only for audit/log reproduction:
`ARTIFACT=openvla_oft_perturb_eval_occlusion_bottleneck_hardocc080_transfer_headinterp000_lorafull_novideo_v1 JOB_IDS=782410,782411,782412,782413,782414,782415,782416 DETAIL_JOB_IDS=782410,782411,782412,782413,782414,782415,782416 ROUTE_LABEL='Hard-occlusion 0.80 alpha0 official-head/full-LoRA no-video OpenVLA-OFT repair' scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh --poll --sync --no-check`.
Do not incorporate this route; it is a completed negative learned-policy
repair attempt. After cancelling stale dependency-held OpenVLA jobs from
superseded routes, `squeue -u $(whoami)` was empty at 2026-06-11 02:16 BST.
No active learned-policy cluster jobs remained queued at that checkpoint. A
new occlusion-only scout was then submitted at 2026-06-11 03:35 BST for a
different premise: an official-identity fallback/router would use the official
checkpoint for clean identity and the adapted branch only when hard occlusion
is known. This scout evaluates only hard-occlusion 0.80 for the same alpha-0
official-head/full-LoRA no-video checkpoints, so it is not paper evidence and
not a full gate. It is only worth formalizing if BGR beats both official and
matched random on occlusion by at least +10/400 episodes and +0.02 absolute
success rate. Submitted jobs are official `782556`, BGR `782557`, and
matched random `782558`; initial poll at 2026-06-11 03:35:39 BST showed all
three running with no parseable logs yet. Latest local compact partial after
the 2026-06-11 03:57:37 BST poll had BGR occlusion `73/126`, official
`75/128`, and matched random `76/130`; BGR is still trailing both comparators on the
occlusion-only scout. This is not a complete 400-episode gate result, but the
fallback/router premise is not trending positive. Poll/sync with:
`ARTIFACT=openvla_oft_perturb_eval_occlusion_bottleneck_hardocc080_transfer_headinterp000_lorafull_novideo_occscout_v1 JOB_IDS=782556,782557,782558 DETAIL_JOB_IDS=782556,782557,782558 ROUTE_LABEL='Hard-occlusion 0.80 alpha0 no-video occlusion-only fallback scout' scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh --poll --sync --no-check`.
Do not incorporate this scout into `paper/main.tex`; close the fallback/router
premise if it fails the occlusion margin, and preregister a full router-style
learned-policy gate before making any paper claim if it clears. Both
head-interpolation routes remain non-evidence or negative unless a genuinely
different follow-up passes the same fixed +10/400, +0.02, and
identity-preservation gate. The
latest
0.80 identity-anchored base route is closed negative with complete rows:
BGR identity/occlusion 389/400 and 303/400, official 393/400 and 296/400, and
matched random 393/400 and 302/400. The sync helper previously left this route
as incomplete because an old partial identity retry log sat beside the complete
log; it now parses both total and progress-line formats and selects the
highest-episode retry log per method/perturbation. The micro and strict 0.80
routes already fail the identity side condition: BGR identity is 387/400 for
the micro route and 388/400 for the strict route, each trailing official
identity 393/400 by more than one episode. The strict route partial summary now
has official occlusion 296/400 and matched-random identity 391/400; these
routes are already non-promotable on identity. A 2026-06-11 12:06 BST sync
closed the stale A40 variants as incomplete infrastructure closures rather
than live candidates: the 0.80 micro A40 route has only official identity
393/400 summarized, the 0.80 identity-anchor A40 route has official identity
393/400 plus partial official occlusion 241/342, and the 0.90 strict A40 route
has only official identity 393/400. Their remaining BGR/random rows failed or
were cancelled, so they are non-evidence and not promotable comparisons.
An internal sklearn-digits margin scout was also opened as a genuinely
pre-existing supervised dataset route, but it is rejected before paper
promotion: the best BGR target in
`results/sklearn_digits_margin_scout_v0/summary.csv` is target radius 1.0 with
0.8271 final RAUC versus 0.8123 for uniform, only W/L/T=2/2/0, and below the
+0.03 screen threshold. This is not evidence to add to the manuscript.
A second sklearn tabular scout over built-in breast-cancer and wine datasets is
also rejected: breast cancer's best BGR row gives 0.9610 vs. 0.9516 RAUC
against uniform (W/L/T=3/1/0), and wine's best gives 0.9702 vs. 0.9563
(W/L/T=4/0/0), but both gaps are below the +0.03 pre-registration screen. This
does not solve the independent pre-existing benchmark weakness; the later
OpenML diabetes, blood-transfusion, and phoneme follow-ups below are the routes
that change that part of the record.
An OpenML margin scout opened one pre-existing-dataset route, and its fixed
30-seed follow-up plus held-out replication are now positive: OpenML diabetes
at target radius 2.0 gives BGR 0.7062 RAUC versus uniform 0.6689 (+0.0373,
W/L/T=24/6/0) in the first 30 seeds, and BGR 0.7056 versus uniform 0.6673
(+0.0383, W/L/T=23/7/0) on seeds 30--59. BGR also beats fixed-radius replay in
both runs. This is real acceptance-moving evidence for a pre-existing
supervised margin-replay benchmark, but it does not solve the learned-policy
failure or standard-environment record by itself.
A fixed external numeric OpenML suite at target radius 2.0 then added
replicated pre-existing-dataset signals. The suite is mixed overall, but the
full held-out repeat is stable at the suite level: original/held-out macro
means are BGR 0.8055/0.8068, uniform 0.7939/0.7943, and fixed-radius
0.7864/0.7839; pooled across both runs, BGR is ahead on 6/8 dataset means
versus uniform and 7/8 versus fixed-radius. Blood-transfusion-service-center
gives BGR 0.7625 versus uniform 0.6657 (+0.0968, W/L/T=30/0/0) and fixed-radius
0.6920 in the first 30 seeds; held-out seeds 30--59 give BGR 0.7595 versus
uniform 0.6846 (+0.0749, W/L/T=25/5/0) and fixed-radius 0.7133. Phoneme also
clears after held-out replication: original BGR 0.7228 versus uniform 0.6896
and fixed-radius 0.6704, held-out BGR 0.7124 versus uniform 0.6758 (+0.0366,
W/L/T=21/9/0) and fixed-radius 0.6792 (+0.0332, W/L/T=25/5/0).
A broader fixed 10-dataset numeric OpenML suite then reduced some
cherry-picking risk but also added a macro caveat. The suite was submitted as
job `774312`, and the held-out seeds 30--59 replication was submitted as job
`774346` before the first summary was known. Both completed with exit `0:0`.
Original/held-out macro means are BGR 0.7756/0.7820, uniform 0.7817/0.7800,
and fixed-radius 0.7815/0.7845; pooled across both runs, BGR is ahead on 6/10
dataset means versus uniform and 6/10 versus fixed-radius. This is therefore
not a broad macro win. It does add two replicated dataset-level positives:
MagicTelescope reaches BGR 0.7395 vs. uniform 0.6884 and fixed-radius 0.6827
in the original run, then BGR 0.7339 vs. uniform 0.6694 and fixed-radius
0.6780 on held-out seeds; haberman reaches BGR 0.7258 vs. uniform 0.6155 and
fixed-radius 0.6644 in the original run, then BGR 0.7275 vs. uniform 0.6327
and fixed-radius 0.6690 on held-out seeds. This strengthens the supervised
pre-existing-dataset story but still does not solve the standard-environment or
learned-policy gap.
A fixed target-radius sensitivity diagnostic has completed for the two new
broad-suite positives, MagicTelescope and haberman. Initial Athena jobs
`774495`/`774496` failed immediately because the Slurm batch environment had
`python3` but not `python` on PATH; jobs `774509`/`774510` then failed because
bare `python3` lacked `sklearn`. Corrected jobs `774520` and `774521` used
`/work/joy/bgr/.venv-openml-broad/bin/python` and completed with exit `0:0`.
The result is supportive at target radii 1.5 and 2.0 but keeps a target-1.0
fragility caveat: pooled BGR-minus-uniform gaps for MagicTelescope are
+0.014/+0.047/+0.058 at radii 1.0/1.5/2.0, and haberman gives
+0.016/+0.089/+0.103. Versus fixed-radius replay, pooled gaps are
+0.008/+0.043/+0.056 and +0.002/+0.054/+0.060, respectively. Treat this as a
robustness/caveat check for the supervised OpenML story, not a
standard-environment or learned-policy claim.
A 60-seed target-radius sensitivity check over the three positive OpenML
datasets is now recorded as a fragility caveat rather than a new headline:
pooled BGR-minus-uniform gaps for diabetes/blood/phoneme are
+0.002/-0.002/-0.007 at radius 1.0, +0.032/+0.065/+0.014 at 1.5, and
+0.038/+0.086/+0.035 at 2.0.
The newest standard-environment sequence sharpened the negative record:
LunarLander is now a completed 30-seed negative stress test, not an unresolved
near miss,
bsuite DeepSea trails the state-priority/uniform-radius ablation and has lower
median r80, the positive 4-seed bsuite Catch screen failed its fixed 30-seed
scale-up, bsuite MountainCar has only a tiny sub-threshold RAUC edge with
saturated r80, and bsuite Cartpole loses to uniform and TD-loss. Any further
acceptance-moving benchmark attempt therefore needs a materially different
reset/interface premise, not another rerun of the same small recovery-probe
family.
Gymnasium Blackjack was tested as a materially different package-backed reset
interface with stochastic card draws and exact internal player/dealer state
resets. The 8-seed Athena scout `774192` completed all nine perturbation and
target-radius configs, and every config is rejected before promotion. The
closest row is `down` target radius 3.0, where BGR-Coverage reaches 0.3859
RAUC versus uniform 0.3790, but it still trails failure-only 0.3898 and has
only a 5/3 paired split; the `dealer_signed` target 3.0 row trails uniform,
failure-only, and the BGR-uniform-radius ablation. Treat Blackjack as a
negative scope result, not an active route. A smaller local diagnostic script,
`tools/blackjack_recovery_probe.py`, also has a negative default 4-seed run at
`results/blackjack_recovery_probe_4seed_scout_v1/`: BGR-Coverage reaches
0.3850 RAUC versus uniform 0.3787, but trails failure-only 0.3954 and
BGR-uniform-radius 0.3851; default BGR is 0.3778. The script is useful for
diagnostics only and does not reopen Blackjack as a promotion route.
A fixed numeric multiclass OpenML suite was tested as a materially different
pre-existing supervised benchmark route from the binary OpenML sweeps. It used
the existing median-impute plus standardized numeric-feature perturbation
pipeline on OpenML version-1 optdigits, pendigits, satimage, segment, letter,
vehicle, texture, mfeat-fourier, mfeat-karhunen, and mfeat-pixel. All ten
datasets were validated as numeric arrays under the Athena OpenML venv, and
optdigits passed local and Athena smokes. Jobs `774591` and `774592` completed
original and held-out seeds at fixed target radius 2.0 with exit `0:0`. The
result is negative: pooled macro means are BGR 0.6418, uniform 0.6948, and
fixed-radius 0.6837; BGR is ahead on 0/10 dataset means versus uniform and
2/10 versus fixed-radius. There are no promotable-like rows. Treat this as a
negative supervised scope result, not a standard-environment or learned-policy
win.
The only remaining broad OpenML candidate-like row, heart-statlog at fixed
target radius 2.0, was tested with a no-retuning seed extension and is now a
near miss rather than a promotable route. The original broad-suite row gives
BGR 0.8075 versus uniform 0.7662 (+0.0412, W/L/T=21/9/0), but held-out seeds
30--59 give BGR 0.7979 versus uniform 0.7707 (+0.0272, W/L/T=19/11/0), below
the +0.03 follow-up screen. The fixed extension, submitted to `athena` as job
`774696` and synced to
`results/openml_heart_statlog_target2_extension_60seed_v1/`, gives BGR 0.7987
versus uniform 0.7733 (+0.0255, W/L/T=41/18/1) and fixed-radius 0.7656
(+0.0332, W/L/T=44/15/1) over seeds 60--119. Pooled across all 120
heart-statlog seeds, BGR is 0.8007 versus uniform 0.7709 (+0.0298,
W/L/T=81/38/1) and fixed-radius 0.7605 (+0.0402, W/L/T=88/31/1). Because the
extension and pooled uniform gaps remain just below the +0.03 screen, retire
heart-statlog as a near-miss supervised route, not paper evidence.

After the weak-reject style review, the immediate paper-defense priority is not
to amplify p-values or add more authored toy wins. The manuscript should instead
make the evidence contract unavoidable:

- lead with the grid-margin mechanism effect because it is the only visibly
  large effect;
- describe synthetic and suffix results as modest scoped support, not broad
  robustness evidence;
- use the new 30-seed grid-margin witness diagnostic only to scope the
  feasibility-witness assumption: exact acceptance has valid accepted samples at
  1.0000, while symmetric 10%/20% witness noise lowers true-boundary recall to
  0.9001/0.7980 without creating a broad robustness guarantee;
- keep standard-environment diagnostics in the limitations unless they satisfy
  the promotion criteria below;
- keep OpenVLA/LIBERO framed as an audit until BGR beats both matched random and
  the official checkpoint on the same fixed evaluation;
- foreground mean differences, clean success, and median critical radius before
  sign-test p-values;
- treat the local boundary proposition as design intuition, not a theory claim.

Latest learned-policy audit, completed 2026-06-07: the perturb-only anchored
OpenVLA-OFT route preserves clean identity success for BGR, matched random, and
the official checkpoint at 99/100 each, but fails the promotion gate on
non-identity perturbations. BGR reaches 371/400, official reaches 367/400, and
matched random reaches 372/400. The official gap is only +4/400 (+0.0100) and
BGR trails matched random by 1/400 (-0.0025), below the fixed +10/400 and +0.02
requirement. This sharpens the acceptance problem: the current learned-policy
evidence remains an audit, not positive robotics evidence.

Completed independent-benchmark route, opened and evaluated 2026-06-07:
MinAtar Breakout. This
route is materially different from the retired MiniGrid/Gymnasium/bsuite
screens because it uses MinAtar's package-owned compact Atari-style Breakout
dynamics, the package's internal state interface, a fixed paddle-tracking
controller, and signed paddle-cell perturbations after a burn-in checkpoint.
The isolated environment is `/tmp/bgr_minatar_venv` with `MinAtar==1.0.15` and
`numpy==2.4.6`; MinAtar is an optional temporary calibration dependency and is
not vendored.

Fixed pre-method calibration command:
`PYTHONPATH=src:. /tmp/bgr_minatar_venv/bin/python tools/minatar_breakout_recovery_calibration.py --out results/minatar_breakout_recovery_calibration_12seed_v1`

Calibration result: the fixed 12-seed run cleared the pre-method gate with clean
success 1.0000, recovery range 0.6667--1.0000, RAUC 0.7000, and r80 0.6000 on
the 0--5 paddle-cell offset grid. This was not BGR evidence; it only permitted
a fixed all-method MinAtar Breakout screen.

The fixed all-method MinAtar screen was implemented at
`tools/minatar_breakout_recovery_probe.py` and preregistered with:
`PYTHONPATH=src:. /tmp/bgr_minatar_venv/bin/python tools/minatar_breakout_recovery_probe.py --out results/minatar_breakout_recovery_probe_4seed_v1`.
It used uniform, fixed-radius, failure-only, TD-loss, BGR-uniform-radius,
BGR-Coverage, and default BGR baselines on the same package-backed restored
Breakout states. The result is negative: default BGR and BGR-Coverage both tie
uniform at 0.8896 final RAUC with W/L/T=0/0/4, median r80 saturates at 5.0000,
and failure-only has the best AULC at 0.7721. Do not scale or promote this
route without a genuinely new preregistered premise.

Completed independent-benchmark route, opened and evaluated 2026-06-07:
MinAtar Asterix. This is still MinAtar, but it uses different package-owned game dynamics from Breakout:
moving enemies/gold, vertical and horizontal player motion, and an
entity-avoidance controller. It is a pre-method calibration only, not BGR
evidence. The isolated environment is `/tmp/bgr_minatar_venv` with
`MinAtar==1.0.15` and `numpy==2.4.6`.

Fixed pre-method calibration command:
`PYTHONPATH=src:. /tmp/bgr_minatar_venv/bin/python tools/minatar_asterix_recovery_calibration.py --out results/minatar_asterix_recovery_calibration_12seed_v1`

Calibration result: the fixed 12-seed run clears the pre-method gate with clean
success 0.8333, recovery range 0.5000--0.8333, RAUC 0.7188, and r80 5.3333 on
the 0--8 seed-fixed player-cell displacement grid after a 30-step burn-in and
60-step recovery horizon. This only permitted a fixed all-method Asterix
screen; it was not positive BGR evidence.

The fixed all-method MinAtar Asterix screen was implemented at
`tools/minatar_asterix_recovery_probe.py` and preregistered with:
`PYTHONPATH=src:. /tmp/bgr_minatar_venv/bin/python tools/minatar_asterix_recovery_probe.py --out results/minatar_asterix_recovery_probe_4seed_v1`.
It used calibration-clean replay seeds 0,1,3,4,5,6,7,8,9,11 and compared
uniform, fixed-radius, failure-only, TD-loss, BGR-uniform-radius,
BGR-Coverage, and default BGR. The result is negative: failure-only has the
best final RAUC at 0.8625. BGR-Coverage reaches 0.8406, BGR reaches 0.8047,
uniform reaches 0.8234, and BGR-Coverage has W/L/T=1/2/1 against uniform.
Do not scale or promote this route without a genuinely new preregistered
premise.

Rejected internal route scout, opened 2026-06-07: sklearn digits margin replay.
This route was checked because it is a pre-existing dataset rather than another
authored recovery environment. It treats `sklearn.datasets.load_digits` images
as replay states, uses label-preserving fixed-L2 pixel perturbations as the
radius family, and compares uniform, fixed-radius, and boundary-guided replay
with an online `SGDClassifier`. The fixed scout command is:
`PYTHONPATH=src:. python3 tools/sklearn_digits_margin_scout.py --out results/sklearn_digits_margin_scout_v0`.
It uses `scikit-learn==1.8.0`, `numpy==2.4.2`, and Python 3.14.3 from the local
environment. The result is not promotable: across target radii 0.5, 0.8, 1.0,
1.2, 1.5, 1.8, and 2.0, every row is marked `reject-scout`. The best BGR row is
target radius 1.0 with final RAUC 0.8271 versus 0.8123 for uniform, but the
paired split is only W/L/T=2/2/0 and the gap is +0.0148, below the +0.03
candidate threshold. Fixed-radius replay is strongest at target 0.8 (0.8425),
so the route does not support the boundary-radius mechanism without post-hoc
tuning. Do not add this to the paper or scale it without a genuinely new
preregistered premise.

Rejected internal route scout, opened 2026-06-07: sklearn tabular margin
replay. This route extended the pre-existing supervised-dataset check from
digits to built-in tabular datasets, using standardized feature-space
fixed-radius perturbations and the same online `SGDClassifier` replay screen.
The fixed scout command is:
`PYTHONPATH=src:. python3 tools/sklearn_tabular_margin_scout.py --out results/sklearn_tabular_margin_scout_v0`.
It uses `scikit-learn==1.8.0`, `numpy==2.4.2`, and Python 3.14.3 from the local
environment. The result is not promotable: every breast-cancer and wine target
row is marked `reject-scout`. Breast cancer's best BGR row is target radius 2.0
with final RAUC 0.9610 versus 0.9516 for uniform (W/L/T=3/1/0, gap +0.0094).
Wine's best BGR row is target radius 0.5 with final RAUC 0.9702 versus 0.9563
for uniform (W/L/T=4/0/0, gap +0.0139). Both are below the +0.03 candidate
threshold, so this does not support a preregistered paper-facing route. Do not
add this to the paper or scale it without a genuinely new preregistered
premise.

Active internal route scout, opened 2026-06-07: OpenML margin replay. This
route broadens the pre-existing supervised-dataset check beyond built-in sklearn
datasets. It uses `sklearn.datasets.fetch_openml` on version-1 OpenML datasets
ionosphere, sonar, diabetes, and spambase; median-imputes numeric features;
label-encodes targets; standardizes the training split; and compares uniform,
fixed-radius, and boundary-guided replay with an online `SGDClassifier`.

Fixed scout command:
`PYTHONPATH=src:. python3 tools/openml_margin_scout.py --out results/openml_margin_scout_v0`

The scout uses `scikit-learn==1.8.0`, `numpy==2.4.2`, and Python 3.14.3 from
the local environment. OpenML diabetes at target radius 2.0 clears the 4-seed
candidate gate with BGR final RAUC 0.7402 versus uniform 0.6797, gap +0.0605,
and W/L/T=4/0/0; fixed-radius replay is 0.6999 at the same target. Ionosphere,
sonar, and spambase do not clear the gate; sonar at target 1.5 is a near miss
at +0.0289 with W/L/T=4/0/0.

This scout permitted the following fixed preregistered follow-up, with no
target retuning or dataset expansion:
`PYTHONPATH=src:. python3 tools/openml_margin_scout.py --datasets diabetes --targets 2.0 --seeds 30 --out results/openml_diabetes_margin_30seed_v1`.
The 30-seed result stayed positive: BGR final RAUC 0.7062 versus uniform
0.6689 (gap +0.0373, W/L/T=24/6/0) and fixed-radius 0.6759 (gap +0.0303,
W/L/T=19/11/0).

Held-out replication command:
`PYTHONPATH=src:. python3 tools/openml_margin_scout.py --datasets diabetes --targets 2.0 --seed-start 30 --seeds 30 --out results/openml_diabetes_margin_replication_30seed_v1`.
The held-out seeds 30--59 replication also stayed positive: BGR final RAUC
0.7056 versus uniform 0.6673 (gap +0.0383, W/L/T=23/7/0) and fixed-radius
0.6640 (gap +0.0416, W/L/T=24/6/0). Pooled over both 30-seed runs, BGR is
0.7059 versus uniform 0.6681 (gap +0.0378, W/L/T=47/13/0) and fixed-radius
0.6699 (gap +0.0359, W/L/T=43/17/0).

This result should be considered for paper incorporation as scoped,
pre-existing supervised margin-replay evidence. It should not be presented as a
standard-environment recovery result, a robotics result, or a replacement for
the negative OpenVLA/LIBERO audit.

Completed independent-benchmark route, opened and evaluated 2026-06-07:
official Gymnasium Box2D
`LunarLander-v3`. This route is materially different from the retired
MiniGrid, PointMaze, FetchReach, highway parking, and MuJoCo balance/reacher
screens: it uses package-owned Box2D contact dynamics, the package heuristic
controller, and an exact body-state perturbation at a fixed descent checkpoint.
The isolated environment is `/tmp/bgr_lunar_venv` with `gymnasium==1.3.0`,
`box2d==2.3.10`, `pygame-ce==2.5.7`, `swig==4.4.1`, and `numpy==2.4.6`.

Fixed pre-method calibration command:
`PYTHONPATH=src:. /tmp/bgr_lunar_venv/bin/python tools/lunarlander_recovery_calibration.py --out results/lunarlander_recovery_calibration_12seed_v1`

The calibration was only permission to implement a fixed all-method screen. It
was usable only if clean success was at least 0.80, the recovery range was at
least 0.20, and median r80 was not saturated at the maximum radius. It cleared,
but the later fixed all-method screen failed the promotion criteria below.

Calibration result: the fixed 12-seed run cleared the pre-method gate with
clean success 0.9167, recovery range 0.5833--0.9167, RAUC 0.7722, and median
r80 0.5300. This allowed a fixed all-method LunarLander screen; it did not
itself create BGR evidence.

The fixed all-method LunarLander screen was implemented at
`tools/lunarlander_recovery_probe.py` and preregistered with:
`PYTHONPATH=src:. /tmp/bgr_lunar_venv/bin/python tools/lunarlander_recovery_probe.py --out results/lunarlander_recovery_probe_4seed_v1`.
It could only justify a 30-seed scale-up if default BGR or BGR-Coverage beat
uniform, fixed-radius, failure-only, TD/loss-priority, and
BGR-uniform-radius on final RAUC with at least 3/4 paired wins over uniform and
non-contradictory, non-saturated median-r80 evidence.

Result: the fixed 4-seed LunarLander screen is a negative near miss. BGR-Coverage
has the best mean final RAUC at 0.7500, above uniform 0.6222, fixed-radius
0.7375, failure-only 0.6799, TD-loss 0.7174, and BGR-uniform-radius 0.7160.
The preregistered promotion checker still rejects it because BGR-Coverage wins
only 2/4 paired seeds against uniform and median r80 is lower than uniform
(0.4200 vs. 0.4825). Do not scale or promote this route without a genuinely new
preregistered premise.

Follow-up stress test, completed 2026-06-10/11: to close the near miss rather
than leave it ambiguous, the same fixed LunarLander protocol was run at 30
seeds as split Athena method jobs `782056`--`782062`. The merged artifact is
`results/lunarlander_recovery_probe_30seed_v3_782056_782062/`, with checker
transcript `promotion_check.txt`. This does not promote. BGR-Coverage has a
small mean RAUC edge over uniform, 0.7193 vs. 0.7006, and beats fixed 0.6730,
failure-only 0.6196, TD-loss 0.7056, and BGR-uniform-radius 0.7031 on mean
RAUC, but paired signs against uniform are only W/L/T=15/15/0, far below the
24/30 gate, and median r80 is lower than uniform (0.3650 vs. 0.3863). Default
BGR is 0.6742 and fails the uniform, TD-loss, and BGR-uniform-radius
comparisons. Treat LunarLander as closed negative under this protocol.

The fixed target-radius 0.70 LunarLander premise is now completed negative.
The previous small target-radius scout suggested BGR-Coverage might keep the
RAUC advantage while avoiding the median-r80 contradiction, so
`scripts/queue_lunarlander_probe.sh` was extended with `EXTRA_ARGS` and a
30-seed target-0.70 all-method screen was run on Athena as split jobs:
uniform `782561`, fixed `782562`, failure-only `782563`, TD-loss `782564`,
BGR-uniform-radius `782565`, BGR-Coverage `782566`, and BGR `782567`. The
merged artifact is
`results/lunarlander_recovery_probe_30seed_target070_merged/`. The promotion
checker rejects it: BGR-Coverage is 0.6886 final RAUC versus uniform 0.7006
(W/L/T=11/19/0), TD-loss 0.7056, fixed 0.6730, failure-only 0.6196, and
BGR-uniform-radius 0.6777. Median r80 is no longer contradictory
(0.4143 vs. uniform 0.3863), but the method loses to uniform and TD-loss.
Treat target-0.70 LunarLander as closed negative under this protocol.

Completed independent-benchmark route, opened and evaluated 2026-06-07:
official bsuite `deep_sea`. This route is materially different from the retired local
classic-control, MiniGrid, PointMaze, FetchReach, highway parking, Box2D, and
MuJoCo screens: it uses bsuite's package-owned sparse-reward DeepSea task and
randomized action mapping. The recovery interface uses exact restart states on
the DeepSea chain and a fixed adverse perturbation family that shifts the
restart column leftward away from the narrow rewarding path. The isolated
environment is `/tmp/bgr_bsuite_venv` with `bsuite==0.3.6`.

The fixed all-method pre-promotion command is:
`PYTHONPATH=src:. /tmp/bgr_bsuite_venv/bin/python tools/bsuite_deepsea_recovery_probe.py --out results/bsuite_deepsea_recovery_probe_4seed_v1`.

This can only justify a 30-seed scale-up if default BGR or BGR-Coverage beats
uniform, fixed-radius, failure-only, TD/loss-priority, and
BGR-uniform-radius on final RAUC with at least 3/4 paired wins over uniform and
non-contradictory, non-saturated median-r80 evidence. Passing this 4-seed
screen is not paper evidence; it is only permission to run the fixed 30-seed
promotion screen.

Result: the fixed 4-seed bsuite DeepSea screen is negative and should not be
scaled under this protocol. Default BGR has a mean RAUC edge over uniform
(0.1125 vs. 0.0844), but it wins only 2/4 paired seeds, trails the
state-priority/uniform-radius ablation (0.1266), and has lower median r80 than
uniform (0.3625 vs. 0.5750). BGR-Coverage ties uniform on mean RAUC (0.0844),
loses to TD-loss (0.0984) and the uniform-radius ablation, and wins only 1/4
paired seeds against uniform. This directly fails the novelty and
complementary-metric gates.

Completed independent-benchmark route, opened and evaluated 2026-06-07:
official bsuite `catch`. This route differs from the retired bsuite DeepSea
chain screen: it uses
bsuite's package-owned Catch task definition with falling-ball/paddle dynamics,
exact restart fields, and a fixed perturbation family that shifts the paddle
away from the ball while preserving a feasible catch before the terminal row.
The isolated environment is `/tmp/bgr_bsuite_venv` with `bsuite==0.3.6`.

The fixed all-method pre-promotion command is:
`PYTHONPATH=src:. /tmp/bgr_bsuite_venv/bin/python tools/bsuite_catch_recovery_probe.py --out results/bsuite_catch_recovery_probe_4seed_v1`.

This can only justify a 30-seed scale-up if default BGR or BGR-Coverage beats
uniform, fixed-radius, failure-only, TD/loss-priority, and
BGR-uniform-radius on final RAUC with at least 3/4 paired wins over uniform and
non-contradictory, non-saturated median-r80 evidence. Passing this 4-seed
screen is not paper evidence; it is only permission to run the fixed 30-seed
promotion screen.

Result: the fixed 4-seed bsuite Catch screen passed the pre-promotion scale-up
gate for default BGR. BGR reaches 0.9742 final RAUC vs. uniform 0.8388
(+0.1354, 4/0/0), fixed-radius 0.7767, failure-only 0.9336, TD-loss 0.7140,
and BGR-uniform-radius 0.8982. Median r80 is non-contradictory (BGR 1.0000 vs.
uniform 0.9437). BGR-Coverage is not promotable because it trails failure-only
and BGR-uniform-radius. This 4-seed screen is still not paper evidence.

The fixed 30-seed promotion command is:
`PYTHONPATH=src:. /tmp/bgr_bsuite_venv/bin/python tools/bsuite_catch_recovery_probe.py --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 --out results/bsuite_catch_recovery_probe_30seed_v1`.

Result: the fixed 30-seed bsuite Catch promotion screen is negative and should
not be promoted. The 4-seed result did not survive scale-up: default BGR reaches
0.8446 final RAUC versus 0.8782 for uniform (14/16/0), BGR-Coverage reaches
0.8452 versus 0.8782 (13/17/0), failure-only reaches 0.9676, and
BGR-uniform-radius reaches 0.8588. Median r80 also contradicts the boundary
improvement story: default BGR 0.8367 and BGR-Coverage 0.8608 trail uniform
0.9233. Do not scale or promote this Catch route without a genuinely new
preregistered premise.

Completed independent-benchmark route, opened and evaluated 2026-06-07:
official bsuite `mountain_car`. This route differs from the retired local
MountainCar diagnostic by instantiating bsuite's package-owned MountainCar task
and stepping its private restart fields for the recovery rollouts. The
perturbation family was fixed before method outcomes: replay states are
right-moving MountainCar states, and larger radii move starts back toward the
low-energy valley anchor, requiring the learner to recover momentum and
position before reaching the package goal. The isolated environment is
`/tmp/bgr_bsuite_venv` with `bsuite==0.3.6`.

The fixed all-method pre-promotion command is:
`PYTHONPATH=src:. /tmp/bgr_bsuite_venv/bin/python tools/bsuite_mountaincar_recovery_probe.py --out results/bsuite_mountaincar_recovery_probe_4seed_v1`.

This can only justify a 30-seed scale-up if default BGR or BGR-Coverage beats
uniform, fixed-radius, failure-only, TD/loss-priority, and
BGR-uniform-radius on final RAUC with at least 3/4 paired wins over uniform, a
mean RAUC gap of at least +0.01, and non-contradictory, non-saturated
median-r80 evidence. Passing this 4-seed screen is not paper evidence; it is
only permission to run the fixed 30-seed promotion screen.

Result: the fixed 4-seed bsuite MountainCar screen is negative and should not
be scaled under this protocol. Default BGR reaches 0.0532 final RAUC versus
0.0497 for uniform (+0.0036, 3/1/0), but the gap is below the preregistered
+0.01 threshold; BGR-Coverage reaches 0.0553, still below fixed-radius replay
(0.1420), failure-only replay (0.0653), and BGR-uniform-radius (0.0558). Median
r80 is saturated at 1.0000 for BGR, BGR-Coverage, BGR-uniform-radius,
failure-only, TD-loss, and uniform, so the screen does not measure useful
boundary expansion.

Completed independent-benchmark route, opened and evaluated 2026-06-07:
official bsuite `cartpole`. This route uses bsuite's package-owned three-action
Cartpole task (`left`, `stay`, `right`), exact `CartpoleState` restarts, and the
package `step_cartpole` dynamics. Replay states are near-upright cart-pole
states with nonzero cart, pole-angle, and velocity errors. The fixed
perturbation family increases bounded cart-position, cart-velocity,
pole-angle, and pole-velocity error while clipping starts inside bsuite's
height and cart-position feasibility limits. The isolated environment is
`/tmp/bgr_bsuite_venv` with `bsuite==0.3.6`.

The fixed all-method pre-promotion command is:
`PYTHONPATH=src:. /tmp/bgr_bsuite_venv/bin/python tools/bsuite_cartpole_recovery_probe.py --out results/bsuite_cartpole_recovery_probe_4seed_v1`.

This can only justify a 30-seed scale-up if default BGR or BGR-Coverage beats
uniform, fixed-radius, failure-only, TD/loss-priority, and
BGR-uniform-radius on final RAUC with at least 3/4 paired wins over uniform, a
mean RAUC gap of at least +0.01, and non-contradictory, non-saturated
median-r80 evidence. Passing this 4-seed screen is not paper evidence; it is
only permission to run the fixed 30-seed promotion screen.

Result: the fixed 4-seed bsuite Cartpole screen is negative and should not be
scaled under this protocol. Default BGR reaches 0.7464 final RAUC versus 0.7577
for uniform (-0.0113, 0/4/0), while BGR-Coverage reaches 0.7559 versus uniform
(-0.0018, 2/2/0). Both trail TD-loss replay (0.7694) and fixed-radius replay
(0.7604); default BGR also trails the state-priority/uniform-radius ablation
(0.7551). Median r80 does not rescue the result: default BGR reaches 0.9667
versus 0.9875 for uniform, while BGR-Coverage is 0.9917 in a near-ceiling
regime. Do not scale or promote this Cartpole route without a genuinely new
preregistered premise.

## Promotion Criteria For A New Independent Benchmark

A new benchmark result should be promoted into `paper/main.tex` only if it meets
all of these criteria before paper writing:

- The environment is pre-existing and recognizable, or the perturbation protocol
  is fixed before seeing results.
- The run uses at least 30 paired seeds for BGR and each baseline.
- BGR beats uniform replay on final RAUC with at least 24/30 paired wins and a
  practically visible mean gap.
- BGR also beats fixed-radius, failure-only, and loss/loss-proxy replay on
  final RAUC.
- Median critical radius does not contradict the RAUC claim, or the paper
  reports the contradiction as a limitation rather than a win.
- Median critical radius is not saturated at the evaluation maximum for both
  BGR and uniform; otherwise the benchmark is not measuring boundary expansion.
- The same configuration is either replicated on held-out seeds or passes a
  preregistered stress sweep.
- The result is checked by `scripts/check_paper_claims.py` and protected by
  `scripts/check_submission_package.py` rendered-PDF guards.

## Do Not Promote

Do not add another result to the manuscript if it has any of these properties:

- BGR wins only after choosing a post hoc variant or metric.
- BGR beats uniform but loses to failure-only, fixed-radius, or loss-priority on
  the headline metric.
- The mean gap is tiny and the argument depends mainly on paired sign-test
  significance.
- The result is another authored mechanism environment rather than an
  independent check.
- The learned-policy result fails to beat the official checkpoint.
- The learned-policy result fails to beat matched random under the fixed gate,
  even if it narrowly beats the official checkpoint.

## Next Experiment Candidates

Stop-rule after the 2026-06-07 standard-benchmark refresh: do not add more
local classic-control, MiniGrid, bsuite, or small tabular recovery probes under
this evidence plan unless the reset/interface premise materially changes.
FrozenLake, Taxi, CliffWalking, FourRooms, MountainCar, CartPole, Acrobot,
Pendulum, LunarLander, Reacher, FetchReach, PointMaze, and the bsuite
DeepSea/Catch/MountainCar/Cartpole sequence now cover the obvious low-cost
standard-environment checks, and none meets the promotion gate. Further
acceptance-moving evidence should use either (a) a materially different
external benchmark/reset interface with package version recorded before any
result is run, (b) a genuinely different learned-policy intervention, or (c) a
stronger theory/presentation contribution that materially changes reviewer
risk.

1. Taxi-v3 recovery replay, with taxi-position perturbations and fixed
   passenger/destination state. An optimized internal probe now exists at
   `tools/taxi_recovery_probe.py`, but the first diagnostics are negative for
   promotion:
   - At the normal 4-seed diagnostic budget, final RAUC is 0.9963 for uniform,
     1.0000 for failure-only, 0.9960 for TD-loss, 0.9763 for
     BGR-uniform-radius, 0.9650 for BGR-coverage, and 0.9578 for BGR. This
     protocol saturates simple baselines.
   - At a harder undertrained budget (`--iterations 80 --train-batch-size 6
     --max-steps 40 --q-init-blend 0.02 --q-init-noise 0.04
     --learning-rate 0.30`), BGR-family replay beats uniform, but the winning
     variant is BGR-uniform-radius (0.0794 final RAUC), ahead of BGR-coverage
     (0.0640) and BGR (0.0632). Uniform is 0.0058, failure-only is 0.0187,
     TD-loss is 0.0083, and fixed-radius is 0.0037. This suggests hard-state
     replay can help in Taxi, but boundary-radius sampling is not the important
     ingredient.
   Taxi should stay out of the paper unless a new pre-registered protocol makes
   the boundary-radius rule itself beat the state-priority-only ablation.
   Recheck candidate summaries with:
   `PYTHONPATH=src:. python3 tools/check_candidate_promotion.py <summary.csv>`.
2. A faster discrete control benchmark with exact reset states and a
   pre-registered perturbation family. CliffWalking and FrozenLake currently
   look negative for BGR under the tested protocols.
   - FourRooms recovery replay now has an internal diagnostic at
     `tools/fourrooms_recovery_probe.py`. The fixed protocol uses a canonical
     11x11 FourRooms grid with cross walls and four doorways, deterministic
     shortest-path dynamics, exact resettable doorway-adjacent replay states,
     and Manhattan restart perturbations that can move states across room
     bottlenecks. The 4-seed pre-promotion screen is negative: final RAUC is
     0.9994 for failure-only, 0.9958 for TD-loss, 0.9900 for fixed-radius,
     0.9777 for uniform, 0.9746 for BGR-uniform-radius, 0.9695 for
     BGR-Coverage, and 0.9672 for BGR. The promotion checker rejects BGR
     because it loses to uniform (-0.0106 mean RAUC, 1/3 paired split), all
     strong baselines, and the state-priority/uniform-radius ablation, with
     median r80 saturated against uniform. It rejects BGR-Coverage because it
     trails uniform and all strong baselines and has lower median r80 than
     uniform. Do not scale FourRooms to 30 seeds under this protocol.
   - CliffWalking-v0 recovery replay now has an internal diagnostic at
     `tools/cliffwalking_recovery_probe.py`. The pre-promotion protocol uses
     canonical 4x12 CliffWalking dynamics, exact resettable safe-corridor
     replay states, and Manhattan restart perturbations that can cross the
     cliff boundary. The default 4-seed diagnostic saturates: fixed-radius,
     failure-only, and TD-loss all reach 1.0000 final RAUC, uniform reaches
     0.9945, BGR-uniform-radius reaches 0.9958, and BGR reaches 0.9917, with
     clean success and median r80 saturated at 1.0. A harder undertrained
     variant (`--iterations 60 --train-batch-size 4 --max-steps 24
     --q-init-blend 0.02 --q-init-noise 0.04 --learning-rate 0.30`) is also
     negative: final RAUC is 0.9074 for failure-only, 0.7118 for TD-loss,
     0.6758 for fixed-radius, 0.6490 for uniform, 0.6087 for BGR-Coverage,
     0.4900 for BGR, and 0.4533 for BGR-uniform-radius. The promotion checker
     rejects it because BGR loses to uniform and all strong baselines, despite
     beating the state-priority-only ablation in the harder variant. Keep
     CliffWalking out of the paper.
   - MountainCar-v0 recovery replay now has an internal diagnostic at
     `tools/mountaincar_recovery_probe.py`. The pre-promotion protocol uses
     canonical MountainCar dynamics, right-moving replay states, and an adverse
     perturbation family that moves states back toward the low-energy valley
     anchor. A 4-seed diagnostic is negative for promotion: final RAUC is
     0.1420 for fixed-radius replay, 0.0653 for failure-only, 0.0558 for
     BGR-uniform-radius, 0.0553 for BGR-Coverage, 0.0532 for BGR, 0.0497 for
     uniform, and 0.0447 for TD-loss. The promotion checker rejects it because
     BGR has only 3/4 wins over uniform, a +0.0036 mean RAUC gap, loses to
     fixed-radius and failure-only replay, loses to the uniform-radius ablation,
     and has saturated median critical radius against uniform. Keep
     MountainCar out of the paper unless a new pre-registered protocol makes
     boundary-radius replay beat fixed-radius replay and the state-priority-only
     ablation without saturated radius metrics.
   - CartPole-v1 recovery replay now has an internal diagnostic at
     `tools/cartpole_recovery_probe.py`. The pre-promotion protocol uses
     canonical CartPole dynamics, resettable perturbed states, and linear
     teacher-action replay. A 4-seed diagnostic is also negative for promotion:
     final RAUC is 0.9006 for failure-only, 0.9001 for TD-loss, 0.8987 for BGR,
     0.8985 for fixed-radius, 0.8985 for uniform, and 0.8976 for
     BGR-uniform-radius. Clean success is saturated at 1.0 for every method.
     The promotion checker rejects it because BGR has only 2/4 wins over
     uniform, a +0.0001 mean RAUC gap, loses to failure-only and TD-loss replay,
     and has a lower median critical radius than uniform. Keep CartPole out of
     the paper unless a pre-registered non-saturated protocol creates a
     meaningful recovery-boundary metric before method comparison.
   - Acrobot-v1 recovery replay now has an internal diagnostic at
     `tools/acrobot_recovery_probe.py`. The pre-promotion protocol uses
     canonical Acrobot-v1 dynamics implemented locally, near-swing-up restart
     states, and adverse perturbations toward the hanging state. A 4-seed
     diagnostic is non-saturated but negative for promotion: final RAUC is
     0.1488 for BGR-Coverage, 0.1471 for uniform, 0.1455 for BGR, 0.1383 for
     failure-only, 0.1305 for BGR-uniform-radius, 0.1292 for TD-loss, and
     0.1279 for fixed-radius replay. The promotion checker rejects BGR because
     it loses to uniform on mean RAUC (-0.0016, 2/2 paired split), and rejects
     BGR-Coverage because the uniform gap is only +0.0016 with a 2/1/1 paired
     split, below the pre-set 3/4 wins and +0.01 mean-delta gates. Keep Acrobot
     out of the paper unless a new fixed protocol creates a visible effect
     before method comparison.
   - The official Gymnasium Acrobot package-state scout is completed negative.
     It was preregistered on 2026-06-11 as a different pre-existing reset
     interface, using `--dynamics-backend gymnasium` to step Gymnasium's
     package-owned `Acrobot-v1` transitions and termination after exact
     `env.unwrapped.state` restoration. Slurm job `783971` completed with exit
     `0:0` and synced to
     `results/acrobot_package_recovery_probe_4seed_v1_783971/`. Mean final
     RAUC is uniform 0.1501, BGR-Coverage 0.1393, default BGR 0.1383,
     failure-only 0.1390, fixed 0.1276, TD-loss 0.1273, and
     BGR-uniform-radius 0.1351. The preregistered gate rejects BGR-Coverage
     and default BGR because both lose to uniform with W/L/T=1/3/0, so there is
     no fixed 30-seed follow-up under this premise.
   - Pendulum-v1 recovery replay now has an internal diagnostic at
     `tools/pendulum_recovery_probe.py`. The pre-promotion protocol uses
     canonical Pendulum-v1 dynamics implemented locally, near-upright restart
     states, adverse angle/velocity perturbations, and fixed PD-controller
     imitation. A 4-seed diagnostic is negative and mostly non-informative at
     the endpoint: final RAUC is 0.0075 for failure-only, 0.0036 for BGR,
     0.0007 for uniform, 0.0004 for BGR-Coverage, and 0.0000 for fixed,
     TD-loss, and BGR-uniform-radius replay. The promotion checker rejects BGR
     because the uniform gap is only +0.0029 with a 2/1/1 paired split, BGR
     loses to failure-only, and median r80 saturates at 1.0 for both BGR and
     uniform. Keep Pendulum out of the paper unless a new fixed protocol first
     avoids endpoint collapse and saturated radius metrics.
3. MiniGrid-DoorKey or MiniGrid-FourRooms only if the official `minigrid`
   package is installed and the probe uses package task definitions rather than
   a local environment clone. The preregistered interface should use exact
   resettable states near doors/keys, perturb only the agent position/direction
   while preserving object layout and inventory validity, and compare BGR,
   BGR-Coverage, uniform, fixed-radius, failure-only, TD/loss-priority, and the
   state-priority/uniform-radius ablation. It should not run beyond a 4-seed
   screen unless BGR or BGR-Coverage beats every baseline, avoids clean-success
   and median-r80 saturation, and clears the same promotion checker used above.
   If the package is not installed, do not substitute another local gridworld;
   that would be another authored diagnostic, not the independent benchmark win
   reviewers are asking for.
   - Official MiniGrid-FourRooms was the fixed 4-seed screen at
     `tools/minigrid_fourrooms_recovery_probe.py`. The package was installed in
     an isolated `/tmp/bgr_minigrid_venv` environment as `minigrid==3.1.0` with
     `gymnasium==1.3.0` on 2026-06-05, leaving repo runtime dependencies
     unchanged unless the result becomes promotable. The probe uses
     `gym.make("MiniGrid-FourRooms-v0")`, package-generated layouts/dynamics,
     resettable package env state, bottleneck-adjacent replay states, and
     Manhattan position restarts with direction preserved. The preregistered
     screen command is:
     `PYTHONPATH=src:. /tmp/bgr_minigrid_venv/bin/python tools/minigrid_fourrooms_recovery_probe.py --out results/minigrid_fourrooms_recovery_probe_4seed_v1`.
     A one-method calibration before preregistration showed that the original
     default budget saturated uniform final RAUC (0.9040) and median r80, so the
     fixed screen uses the script defaults `--iterations 80 --eval-every 20
     --train-batch-size 8 --q-init-blend 0.03 --rollout-horizon 50`. A uniform
     one-method check under those defaults gave final RAUC 0.0666 and clean
     success 0.0938, but still saturated relative median r80; therefore any
     saturation or contradiction in the full 4-seed screen blocks promotion.
     The completed 4-seed screen is the first external-package diagnostic with
     a meaningful BGR-family RAUC lead: final RAUC is 0.1426 for BGR-Coverage,
     0.1355 for BGR-uniform-radius, 0.0992 for BGR, 0.0962 for failure-only,
     0.0863 for TD-loss, 0.0351 for uniform, and 0.0220 for fixed-radius
     replay. The strict promotion checker rejects default BGR because it loses
     to the state-priority/uniform-radius ablation, and rejects BGR-Coverage
     because median r80 saturates at 1.0 for both BGR-Coverage and uniform.
     With radius saturation explicitly waived, BGR-Coverage clears the 4-seed
     RAUC/baseline screen. Do not add this result to the paper or scale the same
     protocol to 30 seeds until a preregistered follow-up resolves the
     non-saturated radius-metric requirement.
     The preregistered follow-up is
     `PYTHONPATH=src:. /tmp/bgr_minigrid_venv/bin/python tools/minigrid_fourrooms_recovery_probe.py --out results/minigrid_fourrooms_recovery_probe_absr10_4seed_v1`.
     It keeps the same official package versions, task, replay states,
     perturbation family, methods, seeds, and training budget, and adds
     `final_abs_r10`, the median largest perturbation radius whose absolute
     recovery probability is at least 0.10. This follow-up can only justify a
     30-seed scale-up if BGR-Coverage keeps the RAUC/baseline lead and
     `final_abs_r10` is not saturated for both BGR-Coverage and uniform and
     does not contradict the RAUC effect.
     The completed follow-up keeps the RAUC lead but fails that radius gate:
     `final_abs_r10` is 0.0 for every method, so the radius evidence is
     floor-saturated. `tools/check_candidate_promotion.py` now rejects
     floor-saturated radius metrics as well as ceiling-saturated ones. MiniGrid
     remains the most promising independent benchmark path because the
     BGR-Coverage RAUC effect is visible on an external package task.
     A later fixed measurement-window audit widened the official MiniGrid
     FourRooms perturbation support to Manhattan radius 10 before seeing the
     new method outcomes:
     `PYTHONPATH=src:. /tmp/bgr_minigrid_venv/bin/python tools/minigrid_fourrooms_recovery_probe.py --out results/minigrid_fourrooms_recovery_probe_maxr10_4seed_v1 --max-radius 10`.
     This closes the simple "wider radius window" rescue. BGR-Coverage reaches
     only 0.1031 final RAUC versus 0.1014 for uniform (+0.0017, W/L/T=2/2/0),
     barely exceeds BGR-uniform-radius at 0.0967, and median r80 remains
     saturated at 1.0000 for both BGR-Coverage and uniform. Do not scale or
     promote the max-radius-10 FourRooms protocol.
     A baseline-only replay-distance calibration was run before any method
     comparison for the next screen. The original spread selector was too hard
     at small absolute radii, the goalward selector was too easy, and a
     mid-distance band of shortest-path distances 2--6 gave a usable seed-0
     uniform diagnostic: final RAUC 0.5652, clean success 0.7188, and median
     r80 0.55. The preregistered 4-seed follow-up command was:
     `PYTHONPATH=src:. /tmp/bgr_minigrid_venv/bin/python tools/minigrid_fourrooms_recovery_probe.py --out results/minigrid_fourrooms_recovery_probe_midband_4seed_v1 --replay-selection midband --replay-distance-min 2 --replay-distance-max 6`.
     This screen can only justify a 30-seed scale-up if BGR-Coverage beats
     uniform, fixed-radius, failure-only, TD-loss, and BGR-uniform-radius on
     final RAUC and does not lose the non-saturated median-r80 comparison.
     The completed midband screen is negative and should not be scaled:
     failure-only reaches 0.7940 final RAUC, fixed-radius 0.7587, uniform
     0.6665, BGR-uniform-radius 0.6287, TD-loss 0.6170, BGR-Coverage 0.6077,
     and BGR 0.5538. The checker rejects BGR-Coverage because it loses to
     uniform on mean RAUC and paired signs, loses to fixed-radius,
     failure-only, TD-loss, and the state-priority/uniform-radius ablation,
	     and has lower non-saturated median r80 than uniform (0.6050 vs. 0.6799).
	     A second baseline-only replay-distance calibration was then run before
	     any new method comparison. On four uniform-only seeds, the shorter
	     distance band 2--5 gives clean success 0.7188, RAUC 0.6190, and
	     non-saturated median r80 0.6451; band 3--7 gives clean success 0.7422,
	     RAUC 0.6661, and non-saturated median r80 0.6687. Because band 2--5 is
	     harder while preserving a non-saturated relative-radius metric, the
	     preregistered all-method screen was:
	     `PYTHONPATH=src:. /tmp/bgr_minigrid_venv/bin/python tools/minigrid_fourrooms_recovery_probe.py --out results/minigrid_fourrooms_recovery_probe_mid2_5_4seed_v1 --replay-selection midband --replay-distance-min 2 --replay-distance-max 5`.
	     The completed screen is negative and should not be scaled. Default BGR
	     improves mean RAUC over uniform (0.6747 vs. 0.6190) but wins only 2/4
	     paired seeds, trails fixed-radius replay (0.6779) and failure-only replay
	     (0.7309), and has lower non-saturated median r80 than uniform (0.5627
	     vs. 0.6451). BGR-Coverage is also negative: 0.5933 RAUC vs. 0.6190 for
	     uniform, and it trails fixed-radius, failure-only, TD-loss, and
	     BGR-uniform-radius. The paper-facing consequence remains unchanged:
	     MiniGrid belongs only in the limitations/scope audit.
4. PointMaze/D4RL-style continuous navigation only if a real installed benchmark
   package is available. This is the best mechanistic fit because resettable
   continuous states, corridor bottlenecks, and distance-to-goal perturbations
   should expose recovery margins without relying on image-level policy
   fine-tuning. The promotion gate is the same: no 30-seed scale-up unless a
   fixed 4-seed screen beats uniform and hard-state/loss baselines with a
   visible final-RAUC gap and non-saturated critical-radius metrics.
   - Official PointMaze was the fixed external-package screen at
     `tools/pointmaze_recovery_probe.py`. The package was installed in an
     isolated `/tmp/bgr_pointmaze_venv` environment as
     `gymnasium-robotics==1.4.2`, `gymnasium==1.3.0`, and `mujoco==3.9.0`,
     leaving repo runtime dependencies unchanged unless the result becomes
     promotable. The probe uses `gym.make("PointMaze_UMaze-v3",
     continuing_task=False, reset_target=False)`, package maze layouts,
     package point dynamics, exact `PointEnv.set_state` resets, and
     graph-distance perturbations over official free cells. Before any method
     comparison, uniform-only seed-0 calibration rejected Medium Maze and broad
     U-Maze bands as collapsed or radius-saturated. The fixed U-Maze near-goal
     band gave a usable baseline diagnostic: uniform clean success 0.7500,
     final RAUC 0.4375, median r80 0.7000, and absolute r20 0.5333. The
     preregistered 4-seed screen command is:
     `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/pointmaze_recovery_probe.py --out results/pointmaze_umaze_recovery_probe_4seed_v1 --env-id PointMaze_UMaze-v3 --max-steps 80 --q-init-blend 1.0 --q-init-noise 0.02 --perturb-cells 3 --replay-distance-min 1 --replay-distance-max 3 --iterations 60`.
     This screen can only justify 30-seed scale-up or paper promotion if
     default BGR or BGR-Coverage beats uniform, fixed-radius, failure-only,
     TD-loss, and BGR-uniform-radius on final RAUC with a visible gap, while
     median r80 and absolute r20 are non-saturated and do not contradict the
     RAUC effect.
     The completed 4-seed screen is negative and should not be scaled:
     failure-only reaches 0.5458 final RAUC and 0.5472 absolute r20, far ahead
     of uniform (0.2201 RAUC, 0.2500 absolute r20), BGR-Coverage (0.2073 RAUC,
     0.0750 absolute r20), and default BGR (0.1406 RAUC, 0.0167 absolute r20).
     The checker rejects default BGR because it loses to uniform on mean RAUC
     and paired signs, loses to failure-only on all four seeds, and has a lower
     absolute r20 than uniform. It rejects BGR-Coverage because it also loses to
     uniform on mean RAUC, loses to failure-only on all four seeds, and has
     lower absolute r20 than uniform. PointMaze therefore does not fix the
     independent-benchmark evidence gap under this protocol.
     The next algorithmic follow-up is BGR-Guarded, implemented before any
     full-result screen in `tools/pointmaze_recovery_probe.py` as method
     `bgr_guarded`. It addresses the observed PointMaze failure mode where
     boundary replay collapses clean recovery while failure-only preserves it:
     each update either uses the same failure-mining candidate rule as the
     explicit failure-only baseline (`--guarded-failure-mix 0.55`) or the BGR
     boundary sampler, and then applies clean replay on the selected state with
     probability `--guarded-clean-mix 0.35`. This is only promotable if it
     beats the original failure-only baseline, not merely if it beats old BGR.
     The preregistered 4-seed follow-up command is:
     `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/pointmaze_recovery_probe.py --out results/pointmaze_umaze_guarded_probe_4seed_v1 --env-id PointMaze_UMaze-v3 --methods uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr,bgr_guarded --max-steps 80 --q-init-blend 1.0 --q-init-noise 0.02 --perturb-cells 3 --replay-distance-min 1 --replay-distance-max 3 --iterations 60`.
     The completed guarded follow-up is worse than old BGR on endpoint RAUC:
     BGR-Guarded reaches 0.0566 final RAUC and 0.0000 absolute r20, versus
     failure-only 0.5458 final RAUC and 0.5472 absolute r20, uniform 0.2201
     RAUC and 0.2500 absolute r20, and BGR-Coverage 0.2073 RAUC and 0.0750
     absolute r20. The checker rejects BGR-Guarded because it loses to uniform
     on mean RAUC and paired signs, loses to failure-only on all four seeds,
     loses to the state-priority/uniform-radius ablation, and has lower
     absolute r20 than uniform. This rules out a sampler-mixing fix for the
     PointMaze failure mode; the next attempt must change the update objective,
     reset-state policy, or learned-policy intervention.
   - The preregistered PointMaze update-schedule follow-up was
     BGR-Clean-Shield, implemented as `bgr_clean_shield` in
     `tools/pointmaze_recovery_probe.py`. It uses the same BGR priority and
     boundary-radius sampling as default BGR, but if the selected replay state's
     current clean recovery is below `--clean-shield-threshold 0.75`, it spends
     that update on the clean state (`sigma=0`) rather than another perturbed
     boundary state; otherwise it applies a clean anchor update with probability
     `--clean-shield-anchor-mix 0.25`. This tests an update-schedule hypothesis,
     not another sampler-only rescue. The fixed 4-seed command is:
     `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/pointmaze_recovery_probe.py --out results/pointmaze_umaze_clean_shield_probe_4seed_v1 --env-id PointMaze_UMaze-v3 --methods uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr,bgr_clean_shield --max-steps 80 --q-init-blend 1.0 --q-init-noise 0.02 --perturb-cells 3 --replay-distance-min 1 --replay-distance-max 3 --iterations 60`.
     It is only worth scaling if BGR-Clean-Shield beats uniform, fixed-radius,
     failure-only, TD-loss, and BGR-uniform-radius on final RAUC, and if
     absolute r20 does not contradict the RAUC effect.
     The completed 4-seed screen improves over default BGR but is still
     negative: BGR-Clean-Shield reaches 0.2448 final RAUC, above uniform
     (0.2201), fixed-radius (0.0448), TD-loss (0.0556), and BGR-uniform-radius
     (0.1233), but below failure-only (0.5458). It wins only 2/4 paired seeds
     against uniform and has lower absolute r20 than uniform (0.1167 vs.
     0.2500). The checker rejects it because the effect is inconsistent versus
     uniform, failure-only remains much stronger, and absolute r20 contradicts
     the RAUC gain. Do not scale this update-schedule variant.
   - The next PointMaze reset-interface follow-up is a topology-bottleneck
     replay-state selector, implemented before method comparison in
     `tools/pointmaze_recovery_probe.py` as `--replay-selection bottleneck`.
     It keeps the official `gymnasium-robotics==1.4.2` PointMaze task and exact
     `PointEnv.set_state` resets, but chooses replay states from package maze
     articulation points, falling back to low-degree cells if a maze has no
     articulation points. This changes the reset-state policy rather than the
     BGR sampler or metric. Uniform-only seed-0 calibration rejected the
     default bottleneck controller because U-Maze clean success collapsed to
     0.0000, and rejected Medium Maze even after a stronger controller because
     final clean and RAUC were both 0.0000. The fixed U-Maze bottleneck screen
     uses the stronger controller selected before method comparison:
     `--max-steps 120 --q-init-blend 2.0 --q-init-noise 0.01 --action-scale 0.6`.
     Uniform-only seed-0 under this protocol gives final clean 0.5000, final
     RAUC 0.1875, median r80 0.7000, and absolute r20 0.0667. The
     preregistered 4-seed command is:
     `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/pointmaze_recovery_probe.py --out results/pointmaze_umaze_bottleneck_probe_4seed_v1 --env-id PointMaze_UMaze-v3 --methods uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr --replay-selection bottleneck --max-steps 120 --q-init-blend 2.0 --q-init-noise 0.01 --action-scale 0.6 --perturb-cells 3 --replay-distance-min 1 --replay-distance-max 5 --iterations 60 --eval-every 20`.
     Do not scale it unless default BGR or BGR-Coverage beats uniform,
     fixed-radius, failure-only, TD-loss, and BGR-uniform-radius on final RAUC
     with a visible gap, and median r80 plus absolute r20 do not contradict the
     RAUC effect.
     The completed 4-seed screen is negative and should not be scaled:
     failure-only reaches 0.3500 final RAUC and 0.1462 absolute r20, while
     default BGR reaches 0.0854 final RAUC and 0.0167 absolute r20, and
     BGR-Coverage reaches 0.0573 final RAUC and 0.0000 absolute r20. Default
     BGR wins only 2/4 paired seeds against uniform, loses to fixed-radius on
     mean final RAUC, and loses to failure-only with W/L/T=0/3/1. BGR-Coverage
     loses to uniform, fixed-radius, and failure-only on mean final RAUC, and
     its absolute r20 contradicts the RAUC effect. Keep this as a paper-negative
     reset-interface screen.
5. A larger OpenVLA/LIBERO adaptation only if the recipe changes in a way that
   plausibly beats both matched random and the official checkpoint, not merely a
   different perturbation score.
   - The preregistered candidate was
     `scripts/queue_openvla_oft_preregistered_goal_adapt.sh`. It uses the
     completed fair p4096 common-availability TFDS roots, official OpenVLA-OFT
     LIBERO-Goal statistics, identity-LoRA, image augmentation, `ADAPT_STEPS=500`,
     `LR=5e-7`, and fixed 10-task x 10-trial clean and perturbation evals.
   - Promotion requires BGR to beat both matched random and the official
     checkpoint on the fixed non-identity perturbation total by at least 10/400
     episodes and at least 0.02 absolute success rate, while not trailing clean
     identity by more than 1/100. A tie, one-episode edge, or gain over random
     that still trails official remains a negative audit.
   - Queue the adaptation chain only after the preregistration script is pushed.
     Queue perturbation evals only after BGR and random merge jobs exist, passing
     `BGR_DEPENDENCY=afterok:<bgr_merge>` and
     `RANDOM_DEPENDENCY=afterok:<random_merge>`.
   - Submitted on 2026-06-04 under the live `/work/joy` cluster workspace after
     the preregistration script was pushed. The archived anonymous paths from
     `results/README.md` are not writable in the live account, so the submitted
     jobs use `/work/joy/bgr`, `/work/joy/cache_home`, and
     `/work/joy/external_validation`. Remote `/work/joy/bgr` is dirty and behind
     `origin/main`, so the generated Slurm scripts were submitted with
     `GIT_PULL=0`; this chain uses only the pre-existing TFDS data roots,
     OpenVLA-OFT checkout, and official stats file.
     - Adapt/merge/clean eval: BGR `765759 -> 765760 -> 765761`; random
       `765762 -> 765763 -> 765764`.
     - Perturb evals: official `765765 -> 765766 -> 765767 -> 765768 -> 765769`;
       BGR `765770 -> 765771 -> 765772 -> 765773 -> 765774` after
       `afterok:765760`; random `765775 -> 765776 -> 765777 -> 765778 -> 765779`
       after `afterok:765763`.
     - Initial `squeue` check showed all jobs pending with the intended
       dependencies.
     - Follow-up scheduler check on 2026-06-04/05 still showed all 21 jobs
       pending, not failed. The two root jobs were pending for `Priority`, with
       Slurm backfill start estimates around 2026-06-05T02:12 for BGR adapt
       (`765759`) and 2026-06-05T02:37 for official identity perturb eval
       (`765765`). `scontrol show job` confirmed valid `/work/joy` stdout paths,
       `gres/gpu:a6000:1`, excluded nodes `c2-g4-[19,21]`, and scheduled nodes
       `c1-g4-01` / `c1-g4-05`. Do not resubmit duplicates unless these jobs
       fail or disappear from accounting.
     - Latest poll from the local workspace on 2026-06-04 showed the same
       state: all 21 jobs remain pending; `sacct` reports no start/end times;
       the expected step50500 output directories have not appeared yet. There
       are therefore no new preregistered OpenVLA results to incorporate into
       `paper/main.tex` as of this poll.
     - Follow-up poll while the queue was running showed BGR adapt `765759`
       completed and BGR merge `765760` completed. Random adapt `765762` had
       started, with random merge `765763` still pending on it. The original
       official identity perturb eval `765765` failed immediately with
       `ModuleNotFoundError: No module named 'libero'` because the generated
       live scripts used a stale home-directory `LIBERO_ROOT`, which is absent
       on the compute node; the valid live LIBERO installation is under the
       cluster user directory. The pending
       clean-eval and perturb-eval jobs generated with the bad path were
       canceled (`765761`, `765764`, `765766`--`765779`). Repaired perturb evals
       were submitted with the same preregistered checkpoint paths, perturbation
       families, 10-task x 10-trial protocol, seed, and promotion gate, changing
       only the live LIBERO installation path: official
       `765782 -> 765785 -> 765787 -> 765791 -> 765795`; BGR
       `765783 -> 765786 -> 765789 -> 765792 -> 765794`; random
       `765781 -> 765784 -> 765788 -> 765790 -> 765793` after
       `afterok:765763`. `scontrol` confirmed the repaired scripts include the
       live LIBERO installation in `PYTHONPATH`.
     - Completed poll on 2026-06-05 showed all repaired perturb evals completed
       successfully. The fixed non-identity totals are BGR 366/400 = 0.9150,
       official 367/400 = 0.9175, and matched random 368/400 = 0.9200. Clean
       identity is BGR 99/100, official 99/100, and random 98/100. The clean
       gate is acceptable for BGR, but the preregistered promotion gate fails
       because BGR trails both official and matched random on non-identity
       perturbations instead of beating each by at least 10/400 episodes and
       0.02 absolute success. This result is another negative OpenVLA audit and
       should not be promoted into the paper as robotics evidence.

## Immediate Engineering Work

- Do not add another local classic-control/tabular probe under the existing
  recovery-replay protocol. Taxi, CliffWalking, FourRooms, MountainCar,
  CartPole, Acrobot, Pendulum, and FrozenLake already serve as negative scope
  diagnostics.
- Do not run more MiniGrid/PointMaze/FetchReach screens under the same reset
  interface and tabular/linear update recipe. Official MiniGrid-FourRooms,
  DoorKey, LavaCrossing, LavaGap, PointMaze U-Maze variants, FetchReach, and
  hard-budget FetchReach are completed negative or non-promotable.
- The fixed independent pre-method calibration for Gymnasium MuJoCo
  `Swimmer-v5` is completed negative. This changed the reset interface and
  benchmark family rather than retuning a retired screen, using package-owned
  Gymnasium/MuJoCo dynamics, exact `env.unwrapped.set_state` restoration of
  `qpos`/`qvel`, a fixed phase-preserving sinusoidal controller, and progress
  over a 120-step horizon after perturbing restored package state. Slurm job
  `784458` completed with exit `0:0` and synced to
  `results/swimmer_recovery_calibration_12seed_v1_784458/`. The pre-method
  gate rejects the route because clean success is 0.7417, below the required
  0.80; recovery range is 0.0000--0.7417, RAUC is 0.0464, and r80 is 0.0500.
  Do not build a Swimmer method screen without a genuinely new preregistered
  controller or success definition.
- The next acceptance-moving empirical route must change the premise: either a
  genuinely different pre-existing benchmark package/reset interface, or a
  genuinely different learned-policy intervention that is preregistered before
  data generation and can plausibly beat both matched random and the official
  checkpoint. The current OpenVLA-OFT clean-mix/visual-perturbation recipe
  family is exhausted for acceptance purposes.
- The broader fixed OpenML numeric-suite run has completed and should be
  framed narrowly. It reduces cherry-picking risk by adding replicated
  MagicTelescope and haberman positives, but the 10-dataset macro result is
  mixed/negative and remains supervised margin-replay evidence only. The fixed
  readout command is
  `PYTHONPATH=src:. python3 tools/analyze_openml_margin_suite.py --original results/openml_broad_numeric_target2_30seed_v1/per_seed.csv --replication results/openml_broad_numeric_target2_replication_30seed_v1/per_seed.csv`.
- The target-radius sensitivity diagnostic for MagicTelescope/haberman is
  complete and synced to
  `results/openml_broad_positive_target_sensitivity_30seed_v1/` and
  `results/openml_broad_positive_target_sensitivity_replication_30seed_v1/`.
  It supports radii 1.5 and 2.0 but leaves radius 1.0 as a fragility caveat.
- The fixed multiclass numeric OpenML suite completed negative in Athena jobs
  `774591` and `774592`; it should not be scaled or promoted without a new
  premise because BGR trails uniform on every pooled dataset mean.
- The fixed heart-statlog target-2.0 seed extension completed as Slurm job
  `774696` and is synced locally. It is a near miss: BGR clears fixed-radius
  but misses the +0.03 uniform gap on the extension and pooled 120-seed
  readout. Retire the row unless a genuinely new preregistered premise exists.
- A fixed all-binary numeric OpenML target-1.5 sweep was submitted on
  2026-06-10 as a broad CPU check while OpenVLA GPU jobs wait. It combines all
  numeric binary OpenML datasets currently registered in
  `tools/openml_margin_scout.py` from the default, external-validation, broad,
  and secondary numeric suites, uses `TARGETS=1.5` with the existing numeric
  preprocessing, and runs original seeds 0--29 plus held-out seeds 30--59.
  Slurm jobs `780049` and `780050` both started at 21:02 BST and completed with
  exit `0:0` at 21:34:24/21:34:48 BST. The synced fixed readout is a small
  broad supervised macro win: original/held-out macro means are BGR
  0.7842/0.7859, uniform 0.7774/0.7764, and fixed-radius 0.7790/0.7741. A
  third no-retuning seed block, rerun as checkpointed chunks after the initial
  monolithic Slurm failure, gives BGR 0.7831, uniform 0.7762, and fixed-radius
  0.7736. Pooled over seeds 0--89, BGR is 0.7844, uniform 0.7766, and
  fixed-radius 0.7756. BGR is ahead on 22/32 pooled dataset means versus
  uniform and 24/32 versus fixed. This is
  useful pre-existing supervised benchmark evidence, but the effect is small
  and the suite still contains large negative rows, so do not use it as a
  learned-policy, robotics, or standard-control claim.
- A local aggregation of existing target-2.0 numeric OpenML chunks on
  2026-06-11 is not a stronger broad route: across the 29 datasets covered by
  the external, broad, secondary, and diabetes target-2.0 runs, pooled macro
  means are BGR 0.7772, uniform 0.7734, and fixed-radius 0.7723, with BGR ahead
  on only 17/29 dataset means versus uniform and 18/29 versus fixed. Do not
  queue a duplicate all-binary target-2.0 run as a path to the requested
  standard-environment or learned-policy win.
- The third all-binary numeric OpenML target-1.5 seed block first failed on
  2026-06-10 as monolithic Slurm job `781423`, using `SEED_START=60`,
  `SEEDS=30`, and the same 32 registered numeric binary datasets. It ran on
  `cnode401` but failed after 7:48 with exit `1:0` during `eeg-eye-state`, and
  no completed summary was produced because `tools/openml_margin_scout.py`
  wrote outputs only at the end. Split retries `781530`/`781536`/`781532`/
  `781531`, plus sequential chunk retry `781548`, were killed immediately by
  Slurm with `RaisedSignal:53` before writing logs while `/work/joy` was at
  100% use. On 2026-06-10, 35,696 remote generated OpenVLA rollout `.mp4`
  files were deleted from `/work/joy/bgr/runs` to free quota, the OpenML queue
  wrapper was changed to submit materialized remote `.sbatch` files, and
  `tools/openml_margin_scout.py` gained checkpoint/resume output. A smoke job
  `781672` completed successfully, then four fixed third-block chunks
  `781682`/`781683`/`781684`/`781685` completed with exit `0:0` and 720 rows
  each. The merged artifact is
  `results/openml_all_binary_numeric_target15_thirdsplit_30seed_v1_781682_781685/`,
  with deterministic readout
  `results/openml_all_binary_numeric_target15_thirdsplit_analysis_781682_781685.txt`.
- The Blackjack independent-route scout completed negative: all nine configs in
  `results/blackjack_recovery_scout_8seed_v1/config_summary.csv` have
  `candidate=False`. Do not scale or promote it without a materially new
  preregistered premise.
- If no such empirical route is ready, work on theory/presentation only when it
  directly answers a cited weakness: novelty over state-priority replay,
  estimator/sample-complexity guarantees, metric transparency, or clearer
  recovery-boundary visualization. Do not treat wording-only reframing as an
  acceptance substitute.
- Official MiniGrid-DoorKey was a preregistered external-package screen
  implemented at `tools/minigrid_doorkey_recovery_probe.py`. It uses
  `gym.make("MiniGrid-DoorKey-6x6-v0")`, package-generated layouts and
  MiniGrid `env.step` dynamics, exact reset states with the key carried and the
  door still closed, and position/direction perturbations around the door
  approach. Uniform-only seed-0 calibration selected a distance band that avoids
  the FourRooms endpoint failures: with `--replay-distance-min 1
  --replay-distance-max 6`, `--iterations 50`, `--train-batch-size 5`,
  `--q-init-blend 0.015`, and `--absolute-radius-alpha 0.01`, uniform gives
  final RAUC 0.4939, clean success 0.9167, median r80 0.3000, and absolute
  radius 0.7475. The fixed 4-seed command is:
  `PYTHONPATH=src:. /tmp/bgr_minigrid_venv/bin/python tools/minigrid_doorkey_recovery_probe.py --out results/minigrid_doorkey_recovery_probe_4seed_v1 --iterations 50 --eval-every 25 --train-batch-size 5 --q-init-blend 0.015 --q-init-noise 0.05 --learning-rate 0.25 --epsilon 0.10 --absolute-radius-alpha 0.01 --replay-distance-min 1 --replay-distance-max 6`.
  Do not scale it unless BGR or BGR-Coverage beats uniform, fixed-radius,
  failure-only, TD-loss, and BGR-uniform-radius on final RAUC, and median r80
  plus absolute radius do not contradict the RAUC effect.
  The completed 4-seed screen is negative and should not be scaled: final RAUC
  is 0.6459 for failure-only, 0.6384 for uniform, 0.5018 for TD-loss, 0.4939
  for BGR-uniform-radius, 0.4846 for BGR-Coverage, 0.3687 for BGR, and 0.2424
  for fixed-radius. BGR-Coverage loses to uniform, failure-only, TD-loss, and
  the state-priority/uniform-radius ablation; default BGR loses to uniform and
  all strong baselines. Absolute-radius checks also contradict promotion:
  final_abs_r10 is 0.5916 for BGR-Coverage and 0.4981 for BGR versus 0.7484
  for uniform.
- Official MiniGrid-LavaCrossingS9N3 was a preregistered external-package
  screen implemented at
  `tools/minigrid_lavacrossing_recovery_probe.py`. It uses
  `gym.make("MiniGrid-LavaCrossingS9N3-v0")`, package-generated lava/goal
  layouts and MiniGrid `env.step` dynamics, exact reset states on safe cells
  near the package shortest path and lava hazards, and Manhattan
  position/direction perturbations that preserve safe-cell validity. Lava cells
  are terminal failures, not replay states. Uniform-only seed-0 calibration
  selected a non-saturated pre-promotion budget: with `--iterations 60`,
  `--train-batch-size 5`, `--replay-selection midband`,
  `--replay-distance-min 4`, `--replay-distance-max 16`,
  `--q-init-blend 0.015`, `--q-init-noise 0.06`, `--learning-rate 0.25`,
  `--epsilon 0.10`, and `--absolute-radius-alpha 0.10`, uniform gives final
  RAUC 0.4878, clean success 0.9167, median r80 0.3000, and absolute radius
  0.6500. The fixed 4-seed command is:
  `PYTHONPATH=src:. /tmp/bgr_minigrid_venv/bin/python tools/minigrid_lavacrossing_recovery_probe.py --out results/minigrid_lavacrossing_recovery_probe_4seed_v1 --env-id MiniGrid-LavaCrossingS9N3-v0 --iterations 60 --eval-every 20 --train-batch-size 5 --replay-selection midband --replay-distance-min 4 --replay-distance-max 16 --q-init-blend 0.015 --q-init-noise 0.06 --learning-rate 0.25 --epsilon 0.10 --absolute-radius-alpha 0.10`.
  Do not scale it unless BGR or BGR-Coverage beats uniform, fixed-radius,
  failure-only, TD-loss, and BGR-uniform-radius on final RAUC, and median r80
  plus absolute radius do not contradict the RAUC effect.
  The completed 4-seed screen is negative and should not be scaled: final RAUC
  is 0.4165 for uniform, 0.4042 for BGR-uniform-radius, 0.3547 for
  BGR-Coverage, 0.3153 for BGR, 0.2358 for failure-only, 0.2059 for TD-loss,
  and 0.1677 for fixed-radius. BGR-Coverage and default BGR beat some weak
  baselines but lose to uniform and the state-priority/uniform-radius ablation.
  Absolute-radius checks also contradict promotion: final_abs_r10 is 0.5047
  for BGR-Coverage and 0.4187 for BGR versus 0.6352 for uniform.
- Official MiniGrid-LavaGapS7 was a preregistered external-package screen using
  the same package-backed lava probe after extending
  `tools/minigrid_lavacrossing_recovery_probe.py` to allow LavaGap env IDs.
  The environment is `gym.make("MiniGrid-LavaGapS7-v0")` from
  `minigrid==3.1.0`, a fixed package geometry with a single gap through a lava
  barrier. This changes the official task geometry, not the BGR sampler or
  metric. Baseline-only calibration before method comparison rejected
  LavaGapS5/S6 and default S7 budgets because uniform saturated final RAUC and
  radius metrics. The fixed hard-budget S7 calibration over uniform-only seeds
  0--3 gives final clean 0.8917, final RAUC 0.4461, median r80 0.2453, and
  absolute r10 0.6336. The preregistered 4-seed command is:
  `PYTHONPATH=src:. /tmp/bgr_minigrid_venv/bin/python tools/minigrid_lavacrossing_recovery_probe.py --out results/minigrid_lavagap_s7_recovery_probe_4seed_v1 --env-id MiniGrid-LavaGapS7-v0 --methods uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr --replay-selection spread --replay-distance-min 2 --replay-distance-max 8 --iterations 60 --eval-every 20 --train-batch-size 5 --q-init-blend 0.015 --q-init-noise 0.08 --learning-rate 0.25 --epsilon 0.10 --rollout-horizon 40 --max-radius 6 --absolute-radius-alpha 0.10`.
  Do not scale it unless BGR or BGR-Coverage beats uniform, fixed-radius,
  failure-only, TD-loss, and BGR-uniform-radius on final RAUC with a visible
  gap, and median r80 plus absolute r10 do not contradict the RAUC effect.
  The completed 4-seed screen is negative and should not be scaled: final RAUC
  is 0.4627 for BGR-uniform-radius, 0.4461 for uniform, 0.4277 for
  BGR-Coverage, 0.4094 for TD-loss, 0.4031 for BGR, 0.3353 for failure-only,
  and 0.1435 for fixed-radius. Default BGR loses to uniform with only 1/4
  paired wins, loses to TD-loss, loses to the state-priority/uniform-radius
  ablation, and has lower absolute r10 than uniform. BGR-Coverage gets 3/4
  wins over uniform but still trails on mean RAUC, loses to the
  state-priority/uniform-radius ablation, and has lower median r80 than
  uniform. Keep LavaGap as a paper-negative external-package scope screen.
- Keep scratch negative runs out of the anonymous package unless they are being
  used as explicit limitations.
- Do not spend more robotics compute on the current OpenVLA recipe family unless
  the next attempt changes the learned-policy intervention in a preregistered
  way that plausibly beats both official and matched random; the latest
  preregistered run reinforces the current negative audit.
- The completed weighted learned-policy audit was the weighted
  perturbation curriculum, implemented in
  `scripts/queue_openvla_oft_preregistered_weighted_perturb.sh`. It changes the
  OpenVLA-OFT training distribution rather than only changing step count, seed,
  or evaluation: render 2,048 unique perturbation examples per method with eight
  episodes per perturbation family, duplicate the perturbation subset three
  times under distinct TFDS `mix_source` labels, keep the same identity-LoRA,
  image augmentation, official statistics, `ADAPT_STEPS=500`, `LR=5e-7`, and
  10-task x 10-trial identity/perturbation evals. The fixed prep command is:
  `scripts/queue_openvla_oft_preregistered_weighted_perturb.sh --prep-only --submit-prep`.
  The fixed adaptation command, after prep succeeds, is:
  `PREP_DEPENDENCY=afterok:<prep_job> scripts/queue_openvla_oft_preregistered_weighted_perturb.sh --adapt-only --submit-adapt`.
  The fixed perturbation command, after BGR/random merge jobs exist, is:
  `BGR_DEPENDENCY=afterok:<bgr_merge> RANDOM_DEPENDENCY=afterok:<random_merge> scripts/queue_openvla_oft_preregistered_weighted_perturb.sh --perturb-only --submit-perturb`.
  Promotion requires weighted BGR to beat both weighted matched random and the
  official checkpoint on the fixed non-identity perturbation total by at least
  10/400 episodes and at least 0.02 absolute success rate, while not trailing
  clean identity by more than 1/100. If prep metadata shows unmatched
  BGR/random perturbation-family counts after weighting, the run is an audit
  only and cannot be promoted. The narrow promotion gate is now checked by:
  `PYTHONPATH=src:. python3 scripts/check_openvla_perturb_gate.py --perturb-summary <perturb_summary.csv> --clean-summary <clean_summary.csv>`.
  This intervention is now a negative audit for the official-checkpoint gate:
  before the final matched-random shift row finished, BGR's completed
  non-identity total was already tied with the official checkpoint at 367/400,
  so it cannot clear the required +10/400 and +0.02 margins. The weighted prep
  job was submitted after the preregistration commit on
  2026-06-05 as Slurm job `766799` and completed successfully with exit `0:0`
  after 21m30s. Prep metadata validates the intended perturbation weighting:
  BGR has 7,296 examples and random has 7,424 examples; both have exactly 1,536
  examples each for blur, brightness, occlusion, and shift after the three
  perturbation repeats. Identity differs because native clean-anchor counts are
  1,152 for BGR and 1,280 for random. This is acceptable for queuing adaptation,
  but the result remains audit-only until the fixed evaluation clears the
  promotion gate.
  Weighted adaptation was submitted on 2026-06-05 with
  `PREP_DEPENDENCY=afterok:766799` and `GIT_PULL=0`. The Slurm chain is:
  BGR train `766805`, BGR merge `766806`, BGR clean eval `766807`, random train
  `766808`, random merge `766809`, random clean eval `766810`. At submission
  audit time, BGR train `766805` was running and the remaining jobs were
  pending on dependencies.
  The fixed perturbation eval was submitted on 2026-06-05 before the merge jobs
  completed, using dependency gates `BGR_DEPENDENCY=afterok:766806` and
  `RANDOM_DEPENDENCY=afterok:766809`. Official perturbation jobs are `766817`
  through `766821` for identity, blur, brightness, occlusion, and shift; BGR
  jobs are `766822` through `766826`; random jobs are `766827` through `766831`.
  At submission audit time, official identity `766817` was running, the other
  official perturbations were serialized behind it, and BGR/random perturbation
  jobs were pending on their merge and per-method serial dependencies.
  A later 2026-06-05 audit found BGR train `766805` completed with exit `0:0`
  after 6m14s, with the adapted checkpoint files present. BGR merge `766806`,
  random train `766808`, and official identity eval `766817` were still
  running, while the dependent clean and perturbation evals were still pending.
  There are not yet any weighted clean or perturbation summaries to incorporate
  into the paper.
  A subsequent audit found BGR merge `766806`, random train `766808`, and
  random merge `766809` completed successfully. BGR clean eval `766807`, random
  clean eval `766810`, official identity perturbation eval `766817`, BGR
  identity perturbation eval `766822`, and random identity perturbation eval
  `766827` were running. No weighted clean or perturbation summaries were
  available yet, so the paper should remain unchanged.
  The next audit found the clean identity evals complete and summarized on the
  cluster: BGR 99/100 and matched random 99/100. Identity perturbation also
  completed for all three methods: BGR 99/100, official 99/100, and random
  99/100. This clears the clean-floor check but leaves the promotion gate
  undecided because the required non-identity perturbation totals are still
  running. Official blur `766818`, BGR blur `766823`, and random blur `766828`
  were running; brightness, occlusion, and shift remained pending.
  The first completed non-identity family is blur: BGR 98/100, official 97/100,
  and matched random 99/100. This partial result is not promotable because BGR
  trails random by one episode. The full gate remains open until brightness,
  occlusion, and shift complete, but BGR now needs to overcome this random
  deficit and still clear the preregistered +10/400 and +0.02 absolute
  non-identity margins.
  Brightness then completed for all three methods: BGR 99/100, official
  98/100, and matched random 99/100. Across the completed non-identity
  families so far (blur and brightness), BGR is 197/200, official is 195/200,
  and matched random is 198/200. The gate remains incomplete until occlusion
  and shift complete, and the partial aggregate remains non-promotable because
  BGR still trails matched random by one episode.
  Occlusion then completed as BGR 75/100, official 74/100, and matched random
  75/100. BGR and official shift also completed as 95/100 and 98/100. A later
  Athena poll on 2026-06-06 03:05 PDT / 11:05 BST showed matched-random shift
  job `766831` completed successfully, and log inspection found 97/100 success.
  The complete weighted audit is therefore BGR 367/400, official 367/400, and
  matched random 370/400 over non-identity perturbations, with identity at
  99/100 for all three methods. This fails the preregistered requirement that
  BGR beat both the official checkpoint and matched random by at least 10/400
  episodes and 0.02 absolute success. The complete compact artifact is
  `results/openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1/summary.csv`,
  stored without raw log paths for double-blind hygiene. This intervention is
  complete negative evidence, not a robotics fine-tuning win.
- The completed proximal-anchor learned-policy audit changed the optimization
  objective rather than the data mix:
  `scripts/queue_openvla_oft_preregistered_proximal_anchor.sh`. It reuses the
  already fixed weighted perturbation TFDS roots but adds
  `PROXIMAL_ANCHOR_L2=1.0` inside the OpenVLA-OFT trainer. The generated
  training wrapper snapshots all trainable parameters after resuming the
  official LIBERO-Goal checkpoint and adds an L2 penalty on deviation from
  those initial values while fitting the BGR-boundary or matched-random replay
  examples. This tests whether BGR can improve perturbation behavior without
  drifting away from the official checkpoint. Fixed adaptation command:
  `scripts/queue_openvla_oft_preregistered_proximal_anchor.sh --adapt-only --submit-adapt`.
  Fixed perturbation command after BGR/random merge jobs exist:
  `BGR_DEPENDENCY=afterok:<bgr_merge> RANDOM_DEPENDENCY=afterok:<random_merge> scripts/queue_openvla_oft_preregistered_proximal_anchor.sh --perturb-only --submit-perturb`.
  Promotion uses the same strict learned-policy gate: proximal-anchor BGR must
  beat both proximal-anchor matched random and the official checkpoint by at
  least 10/400 non-identity perturbation episodes and at least 0.02 absolute
  success, while clean identity is no worse than -1/100. Anything weaker stays
  an audit.
  Submitted the fixed adaptation chain on 2026-06-05 13:18 PDT / 21:18 BST
  after verifying the `/work/joy` TFDS roots, OpenVLA-OFT checkout, Python
  environment, `torchrun`, and official statistics file on `athena`: BGR
  train/merge/clean-eval jobs are `767128`/`767129`/`767130`, and matched-random
  train/merge/clean-eval jobs are `767131`/`767132`/`767133`. The fixed
  perturbation evals were then submitted with BGR dependency `afterok:767129`
  and random dependency `afterok:767132`: official identity/blur/brightness/
  occlusion/shift jobs `767134`-`767138`, BGR jobs `767139`-`767143`, and
  matched-random jobs `767144`-`767148`. Slurm immediately reported all jobs
  pending; BGR/random perturb rows were dependency-held, and official
  perturb rows serialized identity through shift. A fresh Athena poll on
  2026-06-05 16:25 PDT / 2026-06-06 00:25 BST still showed all jobs pending with no
  `sacct` start/end times: BGR train job `767128` and official identity job
  `767134` were waiting on unavailable A6000 GPU nodes
  (`ReqNodeNotAvail, UnavailableNodes:c1-g4-[01-05],c2-g4-[13,16-26],c2-g8-[01-03,05-08]`),
  and every other proximal job was dependency-held. `scontrol show job -dd`
  reported `StartTime=2026-06-07T14:27:51` and
  `TresPerNode=gres/gpu:a6000:1` for jobs `767128` and `767134`. Idle g2 nodes
  expose `gpu:a4000`, so a generic/A4000 resubmission is not a protocol-neutral
  acceleration for OpenVLA unless the memory footprint is separately changed
  and preregistered. A later poll on 2026-06-06 03:05 PDT / 11:05 BST showed
  BGR train job `767128` failed with exit code `1:0`; log inspection found a
  PyTorch DDP `Expected to mark a variable ready only once` error at
  `normalized_loss.backward()` in the proximal-anchor wrapper, naming
  `base_model.model.vision_backbone.fused_featurizer.attn_pool.mlp.fc2.lora_B.default.weight`.
  BGR merge, random adapt, and downstream BGR/random perturb jobs were
  dependency-held with `DependencyNeverSatisfied`. The perturbation and
  adaptation compact summaries are still missing, so there is nothing to sync
  or promote until the wrapper is repaired under the same preregistered
  protocol or the route is retired.
  The wrapper was repaired in commit `cfecfd9` without changing the
  preregistered objective: the proximal metric is computed under
  `torch.no_grad()`, and the equivalent proximal gradient is added to
  `param.grad` after the normal DDP backward pass. The repaired execution tag is
  `proxanchor_l2_1em0_ddpgradfix_v1`. Repaired adaptation/merge/clean-eval jobs
  BGR `767657`/`767658`/`767659` and matched random
  `767660`/`767661`/`767662` completed with exit code `0:0`, resolving the
  prior ready-twice execution failure. Repaired fixed perturbation evals were
  submitted as official jobs `767663`-`767667`, BGR jobs `767674`-`767678`, and
  matched-random jobs `767681`-`767685`. The first BGR perturb retry created an
  overly loose `afterany` dependency for job `767668`; that job was canceled
  before execution, and BGR/random perturb evals were then submitted after the
  repaired checkpoints existed. All repaired perturbation jobs completed with
  exit code `0:0`. The compact local summaries show non-identity totals BGR
  368/400, official 367/400, and matched random 368/400, with identity BGR
  98/100, official 99/100, and matched random 98/100. This route therefore
  fails the fixed +10/400 and +0.02 learned-policy gate: BGR ties random and is
  only one episode above the official checkpoint. The result has been added to
  `paper/main.tex` as negative OpenVLA audit evidence, not as a robotics
  fine-tuning claim.
- The completed perturb-only anchored learned-policy audit changed the
  data/objective combination rather than the clean-mix weighting:
  adaptation in
  `scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh`. This route
  trains only on rendered boundary-band perturbation examples, with no clean
  anchor episodes mixed into the RLDS data, and adds a stronger
  official-checkpoint proximal L2 anchor (`PROXIMAL_ANCHOR_L2=5.0`) to protect
  clean identity behavior. It uses the existing BGR-boundary and matched-random
  teacher replay manifest, balanced perturbation rendering with
  `MAX_PERTURB_EXAMPLES=2048`, `PERTURB_EPISODES_PER_FAMILY=8`, official
  OpenVLA-OFT LIBERO-Goal statistics, identity-LoRA, image augmentation,
  `ADAPT_STEPS=300`, `LR=2e-7`, and the same fixed 10-task x 10-trial
  perturbation evaluation. Fixed prep command:
  `scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh --prep-only --submit-prep`.
  Fixed adaptation command after prep succeeds:
  `TRAIN_DEPENDENCY=afterok:<prep_job> scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh --adapt-only --submit-adapt`.
  Fixed perturbation command after BGR/random merge jobs exist:
  `BGR_DEPENDENCY=afterok:<bgr_merge> RANDOM_DEPENDENCY=afterok:<random_merge> scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh --perturb-only --submit-perturb`.
  Promotion uses the same strict learned-policy gate: perturb-only anchored BGR
  must beat both perturb-only anchored matched random and the official
  checkpoint by at least 10/400 non-identity perturbation episodes and at least
  0.02 absolute success, while clean identity is no worse than -1/100.
  Anything weaker remains an audit. Do not incorporate this route into
  `paper/main.tex` unless compact summaries exist and clear that fixed gate.
  Prep was submitted after preregistration commit `8b69ac7` on 2026-06-07 as
  Slurm job `767789` using the live `/work/joy` workspace
  (`REMOTE_PROJECT=/work/joy/bgr`, `REMOTE_RUN_ROOT=/work/joy/bgr/runs`,
  `REMOTE_HF_HOME=/work/joy/cache_home/huggingface`,
  `OPENVLA_OFT_ROOT=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft`,
  `LIBERO_ROOT=/work/joy/external_validation/openvla_oft_smoke_746850/LIBERO`).
  Initial Slurm poll showed job `767789` running on `c1-g4-04` with
  `gres/gpu:a6000:1` and stdout
  `/work/joy/bgr/logs/bgr-perturbonly-prep-p2048unique_perturbonly_anchor_prereg-767789.out`.
  The dependent adaptation chain was submitted after that poll with
  `TRAIN_DEPENDENCY=afterok:767789` and `GIT_PULL=0`: BGR train/merge/clean-eval
  jobs `767790`/`767791`/`767792`, and matched-random train/merge/clean-eval
  jobs `767793`/`767794`/`767795`. The fixed perturbation evals were also
  queued with `BGR_DEPENDENCY=afterok:767791` and
  `RANDOM_DEPENDENCY=afterok:767794`: official jobs `767796`-`767800`, BGR jobs
  `767801`-`767805`, and matched-random jobs `767806`-`767810`. An immediate
  Slurm poll showed prep `767789` and official identity `767796` running, with
  the adaptation jobs and BGR/random perturb evals dependency-pending. At that
  intermediate point the route was not paper evidence until compact summaries
  cleared the fixed learned-policy promotion gate. The historical poll and
  sync commands were:
  `REMOTE_RUN_ROOT=/work/joy/bgr/runs scripts/sync_openvla_oft_perturb_only_anchor_results.sh --poll`
  and
  `REMOTE_RUN_ROOT=/work/joy/bgr/runs scripts/sync_openvla_oft_perturb_only_anchor_results.sh --sync`.
  A helper poll at 2026-06-07 22:08:57 BST showed prep `767789` completed
  cleanly at 22:06:54 BST, BGR adaptation `767790` running on `c1-g4-04`, and
  official identity eval `767796` running on `c2-g4-24`; at that poll both
  compact summary paths were still missing.
  A later helper poll at 2026-06-07 22:19:57 BST showed BGR train/merge
  `767790`/`767791`, random train/merge `767793`/`767794`, and official
  identity `767796` completed cleanly. BGR clean eval `767792`, random clean
  eval `767795`, BGR identity perturb eval `767801`, random identity perturb
  eval `767806`, and official blur `767797` were running; at that poll compact
  summaries were still missing.
  The clean adaptation eval logs were then summarized locally into
  `results/openvla_oft_goal_adapt_eval_p2048unique_perturbonly_anchor_prereg_perturbonly_proxanchor_l2_5em0_step50300_lr2em7_identitylora_imageaug_officialtrainstats_v1/summary.csv`.
  BGR and matched random both score 99/100 clean episodes. This satisfies the
  clean-floor sanity check for the adapted checkpoints but does not affect the
  learned-policy promotion gate until the complete perturbation summary exists.
- After the official MiniGrid-DoorKey and MiniGrid-LavaCrossing negatives, do
  not add more MiniGrid screens under the same tabular recovery-replay protocol.
  The standard-environment route has produced scope evidence, not acceptance
  evidence. Further acceptance-moving work should either change the
  learned-policy intervention, add a truly independent benchmark with a
  different reset/replay interface, or strengthen the theory/presentation enough
  to make the mechanism-study framing stand on its own.
- Gymnasium-Robotics FetchReach-v4 was the next independent-benchmark route
  because it changed both the package and reset interface: replay states are
  package-sampled Fetch goals, perturbations are clipped 3D goal offsets inside
  the package target range, and evaluation uses MuJoCo Fetch dynamics rather
  than grid/tabular transitions. The completed default and hard-budget screens
  are both negative, so this route is now scope evidence rather than acceptance
  evidence. The probe package was verified in the existing isolated
  `/tmp/bgr_pointmaze_venv` environment as `gymnasium-robotics==1.4.2`,
  `gymnasium==1.3.0`, and `mujoco==3.9.0`.
  A pre-comparison calibration tool is implemented at
  `tools/fetchreach_goal_recovery_calibration.py`. The saturated default
  controller (`--horizon 18 --controller-gain 4.0`) was rejected. The fixed
  usable calibration command is:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/fetchreach_goal_recovery_calibration.py --out results/fetchreach_goal_recovery_calibration_gain2_h14_v1 --seeds 2 --replay-states 4 --trials 8 --horizon 14 --controller-gain 2.0 --direction-jitter 0.15`.
  It gives clean success 0.3750, RAUC 0.1969, median r80 0.0395, and recovery
  range 0.0625--0.3750. This is not method evidence. It only establishes a
  non-saturated package-backed recovery curve suitable for implementing a
  preregistered full comparison. Do not run a BGR comparison until the full
  FetchReach tool fixes the learner, replay-state pool, perturbation directions,
  baselines, and promotion gate before seeing method results. Promotion should
  require BGR or BGR-Coverage to beat uniform, fixed-radius, failure-only,
  loss-priority, and state-priority/uniform-radius on final RAUC with
  non-contradictory median-r80 and visible mean effect.
  The full comparison tool is now fixed before method results at
  `tools/fetchreach_goal_recovery_probe.py`. It uses the same
  Gymnasium-Robotics package task, package-sampled replay goals, adverse
  clipped goal offsets, a learned linear goal controller initialized at the
  calibrated weak setting, and teacher-action updates selected by the replay
  method. The preregistered 4-seed command is:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/fetchreach_goal_recovery_probe.py --out results/fetchreach_goal_recovery_probe_4seed_v1`.
  Do not edit learner hyperparameters, replay-state count, perturbation radii,
  methods, seeds, or promotion gate after seeing this result. Do not scale to
  30 seeds or promote to the paper unless default BGR or BGR-Coverage beats
  uniform, fixed-radius, failure-only, TD-loss, and BGR-uniform-radius on final
  RAUC with at least 3/4 paired wins over uniform, a visible mean gap, and
  non-contradictory median r80.
  The completed 4-seed screen is negative and saturated after training:
  failure-only reaches 0.9563 final RAUC, uniform and fixed-radius reach
  0.9437, TD-loss reaches 0.9375, BGR-uniform-radius reaches 0.9062,
  BGR-Coverage reaches 0.9000, and default BGR reaches 0.8938. All methods
  have clean success 1.0000 and median r80 at the evaluation maximum 0.1500.
  This means the learned linear controller saturates the FetchReach recovery
  curve for simple baselines, while BGR-family replay trails uniform and
  failure-only. Do not scale or promote this FetchReach protocol.
  A hard-budget FetchReach follow-up was calibrated before any new method
  comparison by running uniform replay only. The first weaker-controller probes
  (`--iterations 10/20/30 --horizon 10 --init-gain 1.2`) collapsed to clean
  success 0.0000 and were rejected. Keeping the calibrated controller and
  reducing only update budget produced a usable uniform-only 4-seed calibration:
  final clean 0.9375, final RAUC 0.6813, and median r80 0.1185 with
  `--iterations 20 --eval-every 10 --train-batch-size 2 --horizon 14
  --init-gain 2.0 --init-noise 0.04 --teacher-gain 4.0 --learning-rate 0.20
  --eval-trials 4 --record-trials 2 --quick-trials 2`. This is not method
  evidence; it only fixes a non-saturated measurement budget before comparison.
  The preregistered all-method command is:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/fetchreach_goal_recovery_probe.py --out results/fetchreach_goal_recovery_hard_probe_4seed_v1 --methods uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr --seeds 0,1,2,3 --iterations 20 --eval-every 10 --train-batch-size 2 --horizon 14 --init-gain 2.0 --init-noise 0.04 --teacher-gain 4.0 --learning-rate 0.20 --eval-trials 4 --record-trials 2 --quick-trials 2`.
  Do not edit this command after seeing method results. Do not scale to
  30 seeds or promote unless BGR or BGR-Coverage beats uniform, fixed-radius,
  failure-only, TD-loss, and BGR-uniform-radius on final RAUC with at least 3/4
  paired wins over uniform, a visible mean gap, and non-contradictory
  non-saturated median r80.
  The completed hard-budget screen is negative and should not be scaled:
  TD-loss reaches 0.9437 final RAUC, failure-only reaches 0.8250, uniform
  reaches 0.6813, fixed-radius reaches 0.6438, BGR-uniform-radius reaches
  0.5188, BGR-Coverage reaches 0.4625, and default BGR reaches 0.4438.
  BGR-Coverage loses to uniform with W/L/T=(1,3,0), loses to failure-only on
  all four seeds, loses to TD-loss on all four seeds, and trails the
  state-priority/uniform-radius ablation. Default BGR is lower and has
  contradictory median r80 versus uniform. Keep this as another negative
  independent-benchmark screen.
- FetchPush-v4 was the harder Gymnasium-Robotics calibration route with an
  exact object-state reset interface. This was deliberately a pre-method
  calibration, not BGR evidence: it checks whether a fixed scripted push
  controller can preserve clean object-goal success and produce non-saturated
  recovery curves before any replay-training comparison is implemented. The
  preregistered calibration command is:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/fetch_object_goal_recovery_calibration.py --out results/fetchpush_object_goal_calibration_2seed_v1 --env-id FetchPush-v4 --seeds 2 --replay-states 4 --trials 2 --radii 0.00,0.02,0.04,0.06,0.08,0.12 --horizon 80 --controller-gain 6.0 --direction-jitter 0.10`.
  Do not implement or run a FetchPush method comparison unless this calibration
  gives usable clean success, a nontrivial recovery drop across radii, and no
  obvious controller-induced destruction of already-successful states.
  The completed calibration is negative: clean success is 0.2500, recovery is
  flat at 0.2500 for every tested radius, RAUC is 0.2500, and median r80 is at
  the evaluation maximum 0.1200. The successful rows come from replay states
  that are already solved across all tested perturbations, while the remaining
  states fail clean and perturbed goals. This does not define a trainable
  success-failure boundary, so do not build or scale a FetchPush replay
  comparison around this scripted controller/interface.
  A 2026-06-07 opt-in controller scout added `scripted_push_far`, which
  approaches farther behind the object and uses a longer fixed horizon:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/fetch_object_goal_recovery_calibration.py --env-id FetchPush-v4 --seeds 2 --replay-states 4 --trials 1 --horizon 200 --controller scripted_push_far --controller-gain 10.0 --out results/fetchpush_object_goal_calibration_far_push_2seed_v1`.
  This improves the compact calibration but is still rejected: clean success is
  0.6250, recovery range is 0.6250--0.8750, RAUC is 0.8125, and r80 is 0.1200.
  Do not reopen FetchPush unless a preregistered controller first clears the
  0.80 clean-success gate on the same fixed calibration logic.
  A 2026-06-10 follow-up found and fixed a calibration-harness issue: the
  `--horizon` argument had not overridden Gymnasium's default TimeLimit, so the
  tool now passes `max_episode_steps=args.horizon` to `gym.make`. The same
  follow-up added a materially different `scripted_push_sweep` controller that
  re-aligns behind the object and pushes through the goal. The fixed compact
  command was:
  `PYTHONPATH=src:. /tmp/bgr_fetch_venv/bin/python tools/fetch_object_goal_recovery_calibration.py --env-id FetchPush-v4 --seeds 2 --replay-states 4 --trials 2 --radii 0.00,0.02,0.04,0.06,0.08,0.12 --horizon 250 --controller scripted_push_sweep --controller-gain 8.0 --out results/fetchpush_object_goal_calibration_sweep_g8_h250_2seed_v1`.
  It improves clean success to 0.8750 with recovery range 0.7500--0.8750 and
  RAUC 0.7812, but is still rejected because r80 is saturated at the tested
  maximum and the summary decision is `reject-calibration-radius-saturated`.
  The wider-radius check
  `PYTHONPATH=src:. /tmp/bgr_fetch_venv/bin/python tools/fetch_object_goal_recovery_calibration.py --env-id FetchPush-v4 --seeds 2 --replay-states 4 --trials 2 --radii 0.00,0.08,0.16,0.24,0.32,0.40,0.50,0.60 --horizon 250 --controller scripted_push_sweep --controller-gain 8.0 --out results/fetchpush_object_goal_calibration_sweep_g8_h250_xwide_2seed_v1`
  keeps clean 0.8750 and recovery 0.7500--0.8750, but remains saturated with
  r80 at 0.6000 and the same rejection decision. This route should not proceed
  to replay-method comparison without a new perturbation premise that produces
  a non-saturated success-failure boundary.
  A materially different 2026-06-10 object-state perturbation calibration does
  produce a non-saturated boundary by moving `object0:joint` after reset while
  keeping the seeded FetchPush goal fixed:
  `PYTHONPATH=src:. /tmp/bgr_fetch_venv/bin/python tools/fetch_object_goal_recovery_calibration.py --env-id FetchPush-v4 --seeds 2 --replay-states 4 --trials 2 --radii 0.00,0.02,0.04,0.06,0.08,0.12,0.16,0.20 --horizon 250 --controller scripted_push_sweep --controller-gain 8.0 --perturbation-target object --out results/fetchpush_object_state_calibration_sweep_g8_h250_2seed_v1`.
  It is a usable calibration only: clean success is 0.8750, recovery is
  0.2500--0.8750, RAUC is 0.4125, r80 is 0.0140, and the summary decision is
  `usable-calibration`. A first local method-probe scaffold,
  `tools/fetchpush_object_state_recovery_probe.py`, uses a linear
  teacher-forced imitator of the sweep controller, but viability checks stayed
  at zero clean success and zero RAUC. Do not scale or promote that learner
  without a better preregistered controller; the object-state calibration is
  only an active reset-interface candidate.
  The follow-up probe now also supports opt-in MLP, KNN, and
  trajectory-library policies. MLP and KNN behavior cloning remained zero in
  local viability checks, but after aligning the probe default seed offset with
  the calibrated `121000` reset family, the trajectory-library policy with
  common warm-start demonstrations produced a nonzero sanity row: uniform seed
  0, clean 0.7500 and RAUC 0.5312. A bounded all-baseline scout was therefore
  queued on `athena` as Slurm job `777783` using
  `scripts/queue_fetchpush_object_state_probe.sh`. The remote output path is
  `/work/joy/bgr/runs/fetchpush_object_state_recovery_probe_scout_v1_777783`,
  the log is `/work/joy/bgr/logs/bgr-fetchpush-object-state-777783.out`, and
  the sync command is `JOB_ID=777783 scripts/sync_fetchpush_object_state_probe.sh`.
  Treat this as an in-flight method scout only; no paper claim follows unless
  the completed `summary.csv` clears the existing candidate-promotion checks
  against uniform, fixed radius, failure-only, TD-loss, and the BGR
  uniform-radius ablation.
  Sparse-probe method rows are now rejected before promotion: uniform seeds
  0--3 are 0.3125/0.3000/0.1500/0.2500 RAUC, sparse BGR-Coverage seeds 0--3
  are 0.1875/0.2250/0.1250/0.2125, sparse BGR seeds 0--3 are
  0.1750/0.2375/0.1500/0.0500, BGR-uniform-radius seeds 0--3 are
  0.3250/0.2750/0.1500/0.2125, TD-loss seeds 0--3 are
  0.3250/0.3625/0.1375/0.2000, and failure-only seeds 0--2 are
  0.3000/0.3875/0.1500. `tools/check_candidate_promotion.py` rejects sparse
  BGR-Coverage at 0.1875 mean RAUC versus uniform 0.2531 (W/L/T=0/4/0) and
  rejects sparse BGR at 0.1531 versus uniform 0.2531 (W/L/T=0/3/1), with both
  variants also losing to TD-loss and BGR-uniform-radius. A corrected
  dense-probe BGR diagnostic is running as job `777969` with
  `INITIAL_PROBES=0.00,0.02,0.08,0.20`, `TARGET_RADIUS=0.046`, and
  `RADIUS_BANDWIDTH=0.050`; its first three BGR-Coverage rows are
  0.3125/0.3875/0.1750 RAUC on seeds 0/1/2. Because the dense initial probes
  also warm-start the trajectory-library policy, this diagnostic must not be
  compared directly against sparse-probe baselines. A matched dense-probe
  common-protocol comparison was therefore submitted on Athena at 2026-06-10
  17:30 BST: jobs `778100` uniform, `778101` fixed, `778102` failure-only,
  `778103` TD-loss, `778104` BGR-uniform-radius, `778105` BGR-Coverage, and
  `778106` BGR. These jobs use the same dense probes, `TARGET_RADIUS=0.046`,
  and `RADIUS_BANDWIDTH=0.050`. The matched dense-probe BGR-Coverage row is
  now completed negative before promotion: dense uniform has mean RAUC 0.3000,
  dense fixed 0.2563, dense BGR-uniform-radius 0.2875, and dense BGR-Coverage
  0.2812. `tools/check_candidate_promotion.py` rejects dense BGR-Coverage
  versus uniform (W/L/T=1/2/1, delta -0.0188), versus the uniform-radius
  ablation (delta -0.0063), and on the median-r80 contradiction. The dense
  default-BGR, failure-only, and TD-loss jobs were still running at the
  2026-06-10 17:57--18:00 BST poll, so default BGR remains incomplete rather
  than promoted. Do not promote or scale FetchPush object-state unless the
  completed matched dense summary clears the fixed candidate-promotion checks.
  A follow-up poll at 2026-06-10 18:05 BST showed the same state: jobs
  `778102`, `778103`, and `778106` were still running, while the partial
  combined checker rejects dense BGR-Coverage and rejects default BGR as
  incomplete with only seed 0 available. Do not retune this same target/band
  while the fixed common-protocol rows are still incomplete.
  The 2026-06-10 18:08 BST sync then completed dense TD-loss and dense default
  BGR. Dense default BGR is rejected before promotion at 0.2750 mean RAUC
  versus uniform 0.3000 (W/L/T=0/2/2) and BGR-uniform-radius 0.2875, with a
  lower median-r80 than uniform. Failure-only `778102` is still running, but
  the BGR-family dense route is already not promotable because both default
  BGR and BGR-Coverage fail the uniform comparison.
  Final sync at 2026-06-10 18:10 BST completed failure-only, closing the dense
  common-protocol route negative. Completed mean RAUC is uniform 0.3000,
  failure-only 0.2938, BGR-uniform-radius 0.2875, BGR-Coverage 0.2812, default
  BGR 0.2750, TD-loss 0.2687, and fixed 0.2563. The promotion checker rejects
  both BGR-family treatments versus uniform, failure-only, and the
  BGR-uniform-radius ablation, with median-r80 contradictions. This route is
  negative scope evidence, not an independent benchmark win.
- A new fixed mixed-type binary OpenML scout was opened on 2026-06-10 as a
  materially different pre-existing benchmark interface from the existing
  numeric-only OpenML sweeps. The implementation adds `--mixed-binary-suite` to
  `tools/openml_margin_scout.py`, using one-hot categorical preprocessing for
  credit-g, kr-vs-kp, tic-tac-toe, mushroom, bank-marketing, adult,
  PhishingWebsites, and credit-approval. A local smoke on credit-g and
  tic-tac-toe passed. The fixed 4-seed Athena scout was submitted as job
  `778553` via `scripts/queue_openml_mixed_binary_suite.sh`, using target
  radii 0.5, 1.0, 1.5, and 2.0. This is not paper evidence unless a row clears
  the scout gate and then survives fixed 30-seed plus held-out replication.
  Job `778553` completed with exit `0:0`. No row cleared the strict scout gate:
  the largest BGR gains, credit-approval target 2.0 (+0.0598), credit-approval
  target 1.5 (+0.0509), adult target 2.0 (+0.0315), and bank-marketing target
  2.0 (+0.0312), all had W/L/T=3/1/0 versus uniform. To avoid single-row
  cherry-picking, fixed all-dataset/all-target 30-seed diagnostics were queued:
  original seeds 0--29 as job `778596` and held-out seeds 30--59 as job
  `778597`. Treat these as route-closing/route-promoting diagnostics only, not
  manuscript evidence. Jobs `778596`/`778597` completed with exit `0:0` and
  synced locally. The mixed suite is macro-negative: pooled macro means are BGR
  0.7891, uniform 0.7936, and fixed-radius 0.8000. The fixed all-target check
  did find one replicated adult row at target radius 1.5: original BGR 0.7981
  vs. uniform 0.7544 and fixed 0.7677, held-out BGR 0.7901 vs. uniform 0.7550
  and fixed 0.7557, pooled BGR 0.7941 vs. uniform 0.7547 (+0.0394) and fixed
  0.7617 (+0.0324). This is supervised margin-replay evidence only. A locked
  third-split adult-only confirmation was submitted as job `778912` after
  failed wrapper attempt `778905`; it uses adult target 1.5, seeds 60--119, and
  the same mixed preprocessing. The confirmation is weak and closes the route
  as a near-miss rather than paper evidence: BGR 0.7935 vs. uniform 0.7809
  (+0.0126, W/L/T=37/23/0) and fixed 0.7879 (+0.0055, W/L/T=29/31/0). Pooled
  over all 120 adult target-1.5 seeds, BGR reaches 0.7938 vs. uniform 0.7678
  (+0.0260) and fixed 0.7748 (+0.0190), below the +0.03 fixed follow-up
  standard.
  A narrower mixed-feature third-block confirmation was preregistered on
  2026-06-11 for the remaining stable positive-looking mixed rows rather than
  as a broad route claim: adult, credit-approval, and credit-g at fixed target
  radius 1.5, held-out seeds 60--89, same mixed preprocessing and OpenML
  version-1 datasets. It is not standard-environment or robotics evidence and
  should be used only if it improves the pre-existing supervised margin-replay
  case without contradicting the earlier mixed-suite fragility. Fixed launch:
  `DATASETS=adult,credit-approval,credit-g PREPROCESSING=mixed TARGETS=1.5 SEED_START=60 SEEDS=30 OUT_PREFIX=openml_mixed_positive_target15_thirdsplit_30seed_v1 scripts/queue_openml_mixed_binary_suite.sh`.
  Submitted on Athena as job `782625`, writing to
  `/work/joy/bgr/runs/openml_mixed_positive_target15_thirdsplit_30seed_v1_782625`
  and logging to `/work/joy/bgr/logs/bgr-openml-mixed-binary-782625.out`.
  Promotion bar for this narrow diagnostic: each reported dataset must beat
  both uniform and fixed-radius replay by at least +0.03 RAUC in the new block
  and remain positive when pooled with the original and held-out 60 seeds;
  otherwise treat it as another mixed-type fragility caveat.
  Job `782625` completed successfully at 2026-06-11 04:27:53 BST and synced to
  `results/openml_mixed_positive_target15_thirdsplit_30seed_v1_782625/`. It
  does not clear the fixed promotion bar. Adult gives BGR 0.7818, uniform
  0.7689, and fixed-radius 0.7870, so BGR loses to fixed. Credit-approval
  gives BGR 0.8253, uniform 0.7954, and fixed 0.7926; it clears fixed by
  +0.0327 but is just under +0.03 vs. uniform at +0.0299. Credit-g gives BGR
  0.7053, uniform 0.6670, and fixed 0.6760; it clears uniform by +0.0383 but
  is just under +0.03 vs. fixed at +0.0293. Treat this as mixed-feature
  supervised fragility/near-miss evidence, not a new headline or a solution to
  the standard-environment or learned-policy gap.
  A lower-priority fixed third-block target-2.0 check was queued on 2026-06-11
  only because credit-approval target 2.0 was positive in the original and
  held-out 30-seed target-sensitivity blocks. It also included credit-g target
  2.0 as the other remaining mixed-feature positive-looking row at that target.
  This is still supervised OpenML evidence, not the standard-environment or
  learned-policy win requested by the review. Launch command:
  `OUT_PREFIX=openml_mixed_credit_target2_thirdsplit_30seed_v1 DATASETS=credit-approval,credit-g TARGETS=2.0 SEED_START=60 SEEDS=30 STEPS=8 BATCH_SIZE=64 CANDIDATE_COUNT=128 EVAL_EXAMPLES=250 TIME_LIMIT=12:00:00 MEMORY=24G CPUS=4 scripts/queue_openml_mixed_binary_suite.sh`.
  Athena job `782899` completed with exit `0:0` and synced locally to
  `results/openml_mixed_credit_target2_thirdsplit_30seed_v1_782899/`. It does
  not clear the fixed promotion bar. Credit-approval gives BGR 0.8236, uniform
  0.7954, and fixed-radius 0.7998, missing the +0.03 uniform margin at +0.0282.
  Credit-g gives BGR 0.7022, uniform 0.6670, and fixed-radius 0.6842, clearing
  uniform by +0.0352 but fixed by only +0.0180. Treat this as another
  mixed-feature supervised fragility/near-miss diagnostic, not a new headline
  or a solution to the standard-environment or learned-policy gap.
- FetchSlide-v4 was the next Gymnasium-Robotics object calibration with the
  same exact reset-state and object-goal perturbation interface. It was
  pre-method calibration, not method evidence. The fixed command is:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/fetch_object_goal_recovery_calibration.py --out results/fetchslide_object_goal_calibration_2seed_v1 --env-id FetchSlide-v4 --seeds 2 --replay-states 4 --trials 1 --radii 0.00,0.03,0.06,0.09,0.12,0.15 --horizon 80 --controller-gain 6.0 --direction-jitter 0.10`.
  Do not implement a FetchSlide replay comparison unless this calibration first
  shows usable clean success and a non-flat recovery curve under the fixed
  scripted controller.
  The completed calibration is negative: clean success is 0.2500, RAUC is
  0.1875, recovery ranges from 0.1250 to 0.2500, and median r80 is 0.0720.
  The curve is not fully flat, but the fixed controller fails the clean-success
  prerequisite on most replay states. Do not build or scale a FetchSlide replay
  comparison around this scripted controller/interface.
- FetchPickAndPlace-v4 was the Gymnasium-Robotics object calibration using the
  same exact reset-state and object-goal perturbation interface but a
  separate fixed scripted pick-place controller. This was pre-method calibration
  only. The fixed command is:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/fetch_object_goal_recovery_calibration.py --out results/fetchpickplace_object_goal_calibration_2seed_v1 --env-id FetchPickAndPlace-v4 --controller scripted_pick_place --seeds 2 --replay-states 4 --trials 1 --radii 0.00,0.03,0.06,0.09,0.12,0.15 --horizon 100 --controller-gain 6.0 --direction-jitter 0.10`.
  Do not implement a FetchPickAndPlace replay comparison unless this
  calibration first shows usable clean success and a non-flat recovery curve
  under the fixed scripted controller.
  The completed calibration is negative: clean success is 0.1250, RAUC is
  0.0625, recovery ranges from 0.0000 to 0.1250, and median r80 is 0.0660.
- Gymnasium-Robotics HandReach-v3 was checked as a materially different
  package-owned ShadowHand route after the object-goal Fetch calibrations and
  other standard screens failed. This was a pre-method calibration, not BGR
  evidence: it asks whether a fixed random-shooting controller over active
  finger and thumb action dimensions can preserve clean HandReach success and
  expose a non-flat recovery curve before any replay comparison is implemented.
  The fixed command is:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/handreach_recovery_calibration.py --out results/handreach_recovery_calibration_8seed_v1`.
  The completed calibration is rejected: clean success is 0.0000, recovery is
  flat at 0.0000 across the 0.00--0.20 joint-perturbation grid, RAUC is 0.0000,
  and r80 is 0.2000 under `gymnasium==1.3.0`,
  `gymnasium_robotics==1.4.2`, `mujoco==3.9.0`, and `numpy==2.4.6`.
  Do not build or scale a HandReach replay comparison around this
  controller/interface unless a new preregistered controller first clears the
  0.80 clean-success gate and produces a non-flat recovery curve.
- highway-env parking-v0 was checked as a genuinely different external package
  route after the same-protocol MiniGrid/classic-control/PointMaze/FetchReach
  screens failed. The package was installed in an isolated Python 3.11
  environment as `highway-env==1.10.1` with `gymnasium==1.3.0`; the default
  Python 3.14 environment could not resolve the dependency stack, so do not use
  it for this route. The fixed pre-method command is:
  `PYTHONPATH=src:. /tmp/bgr_highway311_venv/bin/python tools/highway_parking_recovery_calibration.py --out results/highway_parking_recovery_calibration_12seed_v1 --seeds 12 --radii 0,1,2,3,4,5,6,8,10 --horizon 80`.
  This uses highway-env's package-owned `parking-v0` goal observation, vehicle
  dynamics, reward, and success predicate, then perturbs the initial ego pose
  before running a fixed scripted parking controller. The completed calibration
  is rejected before method comparison: clean success is 0.3333, recovery ranges
  from 0.2500 to 0.5000, mean crash rate is 0.5370, RAUC is 0.3750, and median
  r80 is 9.8000. This is a controller/interface failure rather than BGR
  evidence. Do not build or scale a highway-env parking replay comparison unless
  a new preregistered controller or policy first clears clean success and
  non-saturated recovery prerequisites.
- highway-env highway-fast-v0 lane driving was checked as a second highway-env
  route because it uses package-owned traffic dynamics, collision termination,
  kinematic observations, and discrete lane/speed actions rather than the
  parking goal interface. This was pre-method calibration, not BGR evidence.
  The fixed command is:
  `PYTHONPATH=src:. /tmp/bgr_highway311_venv/bin/python tools/highway_lane_recovery_calibration.py --out results/highway_lane_recovery_calibration_12seed_v1`.
  The completed calibration is rejected before method comparison: clean success
  is 0.6667, recovery ranges from 0.5833 to 0.6667, mean crash rate is 0.3810,
  RAUC is 0.6181, and median r80 saturates at 6.0000. Do not build or scale a
  highway-env lane replay comparison around the current idle lane-keep
  controller/interface.
- Gymnasium MuJoCo Reacher-v5 was a completed independent-benchmark route that
  changed both package and reset interface relative to the failed
  MiniGrid, PointMaze, FetchReach, and highway screens. The calibration uses
  Gymnasium's package-owned Reacher-v5 dynamics and target sampling, exact
  MuJoCo state resets, two-joint angular perturbations, and a fixed weak
  inverse-kinematics/PD controller. The fixed command is:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/reacher_recovery_calibration.py --out results/reacher_recovery_calibration_12seed_v1`.
  It clears the pre-method gate with clean success 0.8333, recovery range
  0.5000--0.9167, RAUC 0.7891, and r80 3.0000 on a 0--4 perturbation grid in
  the isolated environment (`gymnasium==1.3.0`, `mujoco==3.9.0`,
  `numpy==2.4.6`). This is not BGR evidence. Do not run or promote a Reacher
  method comparison until the full all-method tool fixes the learner,
  replay-state pool, perturbation radii, baselines, seeds, and promotion gate
  before seeing method results. Promotion should require default BGR or
  BGR-Coverage to beat uniform, fixed-radius, failure-only, TD/loss-priority,
  and the state-priority/uniform-radius ablation on final RAUC with a visible
  effect, paired wins over uniform, and non-contradictory non-saturated radius
  metrics.
  The full comparison tool is now fixed before method results at
  `tools/reacher_recovery_probe.py`. It uses the same official Reacher-v5
  package dynamics and target sampling, exact MuJoCo state resets, two-joint
  angular perturbations, a small linear controller initialized below the
  calibration controller, and teacher-action imitation updates selected by the
  replay method. The preregistered 12-seed command is:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/reacher_recovery_probe.py --out results/reacher_recovery_probe_12seed_v1`.
  The fixed 12-seed result is negative and should not be scaled or promoted:
  uniform has the highest final RAUC (0.3862), while BGR reaches 0.2907
  (4/8/0 paired wins/losses/ties against uniform), BGR-Coverage reaches 0.2721
  (4/8/0), BGR-uniform-radius reaches 0.2921, failure-only reaches 0.3273,
  fixed-radius reaches 0.2330, and TD-loss reaches 0.2501. Median-r80 is also
  not supportive: BGR 3.8375, BGR-Coverage 3.6708, and uniform 3.2437. Treat
  this as scope evidence that the Reacher route did not solve the independent
  benchmark acceptance gap.
- Gymnasium MuJoCo InvertedPendulum-v5 was a completed independent pre-method
  route that changed both task family and reset dynamics relative to the
  failed MiniGrid, PointMaze, FetchReach, highway, and Reacher screens. The
  calibration uses Gymnasium's package-owned InvertedPendulum-v5 dynamics,
  exact MuJoCo state resets, one-dimensional pole-angle perturbations, and a
  fixed PD balance controller. The fixed calibration command is:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/inverted_pendulum_recovery_calibration.py --out results/inverted_pendulum_recovery_calibration_12seed_v1`.
  It clears the pre-method gate with clean success 1.0000, recovery range
  0.0000--1.0000, RAUC 0.7500, and r80 0.2100 on a 0--0.30 perturbation grid
  in the isolated environment (`gymnasium==1.3.0`, `mujoco==3.9.0`,
  `numpy==2.4.6`). This is not BGR evidence.
  The full comparison tool is fixed before method results at
  `tools/inverted_pendulum_recovery_probe.py`. It keeps the same official
  package dynamics, exact MuJoCo state resets, pole-angle perturbation family,
  small linear controller initialized below the calibration controller, and
  teacher-action imitation updates selected by the replay method. The
  preregistered 4-seed screen command is:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/inverted_pendulum_recovery_probe.py --out results/inverted_pendulum_recovery_probe_4seed_v1`.
  Do not scale it unless default BGR or BGR-Coverage beats uniform,
  fixed-radius, failure-only, TD/loss-priority, and BGR-uniform-radius on final
  RAUC with a visible gap, paired wins over uniform, and non-contradictory
  non-saturated median-r80 metrics.
  The completed 4-seed screen is negative and should not be scaled: every
  method ties at final clean success 1.0000, final RAUC 0.7500, final median
  r80 0.2100, and best RAUC 0.7500. BGR has 0/0/4 paired wins/losses/ties
  against uniform on final RAUC, so this route fails the visible-effect,
  paired-win, strong-baseline, and state-priority-ablation gates.
- Gymnasium MuJoCo InvertedDoublePendulum-v5 was the next independent
  pre-method route because it changed the MuJoCo task, perturbation geometry,
  and controller family relative to the retired Reacher-v5 and
  InvertedPendulum-v5 routes. The calibration uses Gymnasium's package-owned
  `InvertedDoublePendulum-v5` dynamics, exact MuJoCo state resets from package
  seeded reset states, two-pole angular perturbations, and a fixed
  finite-difference LQR balance controller. The fixed calibration command is:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/inverted_double_pendulum_recovery_calibration.py --out results/inverted_double_pendulum_recovery_calibration_12seed_v1`.
  It clears the pre-method gate with clean success 1.0000, recovery range
  0.0000--1.0000, RAUC 0.4259, and r80 0.2825 on a 0--0.90 perturbation grid
  in the isolated environment (`gymnasium==1.3.0`, `mujoco==3.9.0`,
  `numpy==2.4.6`). This is not BGR evidence.
  The full comparison tool was fixed before method results at
  `tools/inverted_double_pendulum_recovery_probe.py`. It keeps the same
  official package dynamics, exact MuJoCo reset interface, two-pole
  perturbation family, 0--0.90 evaluation grid, 4-seed pre-promotion screen
  budget, and linear LQR-feature learner initialized at 0.70 times the
  calibrated LQR gain. The preregistered 4-seed screen command is:
  `PYTHONPATH=src:. /tmp/bgr_pointmaze_venv/bin/python tools/inverted_double_pendulum_recovery_probe.py --out results/inverted_double_pendulum_recovery_probe_4seed_v1`.
  Do not tune the learner, replay-state count, perturbation radii, methods,
  seeds, or promotion gate after seeing this result. Do not scale or promote
  it unless default BGR or BGR-Coverage beats uniform, fixed-radius,
  failure-only, TD/loss-priority, and BGR-uniform-radius on final RAUC with a
  visible gap, paired wins over uniform, and non-contradictory non-saturated
  median-r80 metrics.
  The completed 4-seed screen is negative and should not be scaled: default
  BGR has the highest final RAUC (0.0833 vs. 0.0035 for uniform and 0.0000 for
  the other baselines), but final clean success collapses to 0.2500 for BGR,
  0.0000 for BGR-Coverage, and 0.0208 for uniform. The paired RAUC split
  against uniform is only 1/0/3, and median-r80 is saturated at 0.9000 for
  every method except the single surviving BGR seed. Treat this as scope
  evidence that this MuJoCo route does not solve the independent-benchmark
  acceptance gap.
