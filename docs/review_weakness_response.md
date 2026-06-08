# Review Weakness Response

This is an internal triage note for the weak-reject review. It tracks what is
now addressed in the manuscript versus what still requires new evidence.

## Current Acceptance Read

The paper is more defensible as an honest mechanism study, but it is not yet a
90%+ AAAI main-track accept. The main blocker is still evidence, not packaging:
there is no promoted positive independent benchmark and no learned-policy
OpenVLA/LIBERO improvement over both matched random and the official checkpoint.
Run `PYTHONPATH=src:. python3 scripts/check_acceptance_readiness.py --root .`
to reproduce this read from the current result artifacts.

## Reviewer Weaknesses

| Review concern | Current paper defense | Remaining gap | Priority |
| --- | --- | --- | --- |
| Robotics promise not delivered | Title, abstract, introduction, evidence table, OpenVLA table, and conclusion now frame OpenVLA/LIBERO as an audit rather than a robotics claim. The preregistered weighted image-augmentation and proximal-anchor audits are included as negative. | Still no stable learned-policy win. Do not spend more compute on the same recipe family. | High |
| Tiny effect sizes hidden by p-values | Protocol section says effect sizes are primary and sign tests are consistency checks. Main text reports absolute RAUC/AULC/clean/r80 differences before p-values. | Synthetic and suffix effects remain small. They should stay scoped support. | High |
| Benchmarks constructed for BGR | Abstract, synthetic section, grid section, evidence contract, and limitations explicitly say synthetic/grid-margin establish mechanism rather than broad dominance. | Need a positive pre-existing benchmark before making stronger main-track claims. | High |
| Fragile/post-hoc variants | Suffix text explains the first boundary-heavy run undercovered, treats coverage-aware BGR-Suffix as separate manipulation-style support, and says it is not evidence that boundary-only replay is robust. Grid learning-rate caveat remains in text. | The coverage-aware suffix result is still a rescue-style variant; only new preregistered evidence can remove that robustness concern. | Medium |
| Incremental novelty | Related work and grid ablation now frame the uniform-radius run as a negative control: state priority is held fixed and only the radius rule changes, so the effect cannot be attributed to hard-state prioritization alone. | Novelty still depends on this ablation and the mechanism benchmark. A pre-existing win would make it stronger. | High |
| Proposition too strong | Proposition is renamed local boundary intuition and explicitly says it is not a convergence or global robustness theorem. The method section now adds a one-step local margin-shift corollary plus a finite-grid estimator sample-complexity guarantee over a buffer of states: Hoeffding/union-bound error controls every monotone recovery estimate, and critical-radius error is controlled by grid spacing, local slope, threshold error, and the number of Bernoulli probes. | This still does not prove global learner improvement; it only strengthens the local sampler/estimator rationale. | Medium |
| Metrics favor BGR | Protocol states RAUC/AULC are author-defined summaries and reports median r80 disagreements directly. Suffix median-r80 caveat remains in abstract/table text/limitations. | Independent metrics still matter. Any promoted benchmark must avoid saturated or contradictory r80. | High |
| Feasibility witness is a hidden requirement | Problem setting now states that the witness is a real interface assumption, not free supervision or a learned success oracle; without a reliable witness, BGR is only an audit tool or should not be applied. Promotion is limited to exact or stress-tested witnesses. A 30-seed grid-margin witness diagnostic now shows symmetric 10%/20% witness noise preserves valid accepted rates near 1.0 but lowers true-boundary recall to 0.9001/0.7980. | This is controlled mechanism evidence, not a broad guarantee for learned or noisy simulator witnesses. | Medium |
| Results dump/no intuition | Paper now has boundary-intuition and grid learning-curve figures, plus an evidence-contract table. The boundary-intuition panel now overlays seed-0 recovery curves for BGR, uniform, failure-only, fixed-radius, and PLR-loss replay in the same footprint. | This improves readability but does not add independent positive evidence. | Medium |

## Immediate Paper Policy

- Do not promote FourRooms, Acrobot, Pendulum, CliffWalking, Taxi, MountainCar,
  CartPole, FrozenLake, or OpenVLA adaptation as positive evidence under
  current results.
- Do not promote the newer standard-environment sequence either:
  LunarLander-v3 is a near miss rejected by paired signs and median-r80,
  bsuite DeepSea trails the state-priority/uniform-radius ablation, the
  positive 4-seed bsuite Catch screen failed its fixed 30-seed scale-up,
  bsuite MountainCar has only a tiny sub-threshold RAUC edge with saturated
  r80, and bsuite Cartpole loses to uniform and TD-loss. These are limitations,
  not acceptance-moving evidence.
- Keep FourRooms internal: the preregistered 4-seed screen has BGR and
  BGR-Coverage below uniform, fixed-radius, failure-only, TD-loss, and the
  state-priority/uniform-radius ablation, with saturated or contradictory
  median-r80 evidence.
- Keep Acrobot internal: the 4-seed diagnostic is non-saturated, but default
  BGR trails uniform and BGR-Coverage has only a +0.0016 RAUC edge with a
  2/1/1 paired split.
- Keep Pendulum internal: the 4-seed diagnostic has near-zero endpoints,
  saturated median r80, and failure-only replay above BGR.
- Keep CliffWalking in limitations only: default protocol saturates, and the
  harder undertrained protocol has BGR below uniform and all strong baselines.
- Do not add another authored toy benchmark unless it replaces, not expands,
  the current mechanism evidence.
- Do not run another local classic-control/tabular probe after FourRooms; the
  next acceptance-moving benchmark attempt needs a genuinely different external
  package/reset interface with package version recorded before results.
- Official MiniGrid-FourRooms is the first external-package screen with a
  BGR-family RAUC lead, but it is not paper evidence yet: BGR-Coverage clears
  the 4-seed RAUC/baseline screen only if radius saturation is waived, and the
  strict gate rejects it because median r80 saturates at 1.0.
- The MiniGrid absolute-r10 follow-up also stays internal: the RAUC lead
  persists, but `final_abs_r10` is 0.0 for every method, so the auxiliary
  radius metric floor-saturates.
- The MiniGrid FourRooms max-radius-10 measurement-window follow-up also stays
  internal: BGR-Coverage reaches only 0.1031 final RAUC vs. 0.1014 for uniform
  (2/2/0), barely exceeds BGR-uniform-radius at 0.0967, and median r80 remains
  saturated at 1.0000 for both BGR-Coverage and uniform. This closes the simple
  wider-radius-window rescue.
- Official MiniGrid-DoorKey is internal and paper-limitation evidence only:
  BGR-Coverage and default BGR lose to uniform and strong baselines, with
  lower absolute-radius checks than uniform.
- Official MiniGrid-LavaCrossingS9N3 was a preregistered external-package
  screen because it gave package-defined lava hazards, safe-cell reset states,
  and a non-saturated uniform-only calibration before method comparison.
- The completed LavaCrossingS9N3 screen is also internal and negative:
  BGR-Coverage and default BGR lose to uniform and to the
  state-priority/uniform-radius ablation, with lower absolute-radius checks
  than uniform.
- Stop adding MiniGrid/classic-control screens under the same tabular recovery
  protocol. They now support limitations and scope, not an AAAI-main acceptance
  case.
- Gymnasium-Robotics FetchReach-v4 is now completed negative, including the
  hard-budget follow-up. The default screen saturates after training:
  BGR-Coverage 0.9000 and BGR 0.8938 trail uniform/fixed 0.9437 and
  failure-only 0.9563, while every method has clean 1.0000 and median r80
  0.1500. The hard-budget screen is also negative: TD-loss 0.9437,
  failure-only 0.8250, uniform 0.6813, fixed-radius 0.6438,
  BGR-uniform-radius 0.5188, BGR-Coverage 0.4625, and BGR 0.4438. Do not scale
  or promote either FetchReach protocol.
- Gymnasium-Robotics HandReach-v3 is a rejected pre-method route, not paper
  evidence. The fixed 8-seed calibration with a random-shooting ShadowHand
  controller has clean success 0.0000, recovery range 0.0000--0.0000, RAUC
  0.0000, and r80 0.2000. Do not build a replay comparison around this
  controller/interface unless a new preregistered controller first clears clean
  success and produces a non-flat recovery curve.
- The next acceptance-moving result must be a fixed-protocol positive result
  on a truly different external package/reset interface, a genuinely different
  preregistered learned-policy intervention that can beat both official and
  matched random, or a stronger theory/presentation contribution. Do not keep
  rerunning the same MiniGrid/PointMaze/FetchReach/bsuite protocol family.
- An internal sklearn-digits margin replay scout checked a genuinely
  pre-existing supervised dataset route, but it is rejected before any paper
  promotion. The fixed command
  `PYTHONPATH=src:. python3 tools/sklearn_digits_margin_scout.py --out results/sklearn_digits_margin_scout_v0`
  compares uniform, fixed-radius, and boundary-guided replay over
  label-preserving pixel perturbations. Every target row is marked
  `reject-scout`; the best BGR target is 1.0 with 0.8271 RAUC versus 0.8123 for
  uniform, only W/L/T=2/2/0, while fixed-radius replay is stronger at target
  0.8 (0.8425). Do not use this route to answer the independent-benchmark
  critique without a new preregistered premise.
- An internal sklearn tabular margin replay scout checked the same idea on
  built-in breast-cancer and wine datasets, and it is also rejected before any
  paper promotion. The fixed command
  `PYTHONPATH=src:. python3 tools/sklearn_tabular_margin_scout.py --out results/sklearn_tabular_margin_scout_v0`
  marks every target row `reject-scout`; breast cancer's best BGR row is
  0.9610 vs. 0.9516 uniform (W/L/T=3/1/0), and wine's best is 0.9702 vs.
  0.9563 uniform (W/L/T=4/0/0), both below the +0.03 pre-registration screen.
  Do not use this route to answer the independent-benchmark critique without a
  new preregistered premise.
- An OpenML margin replay scout now gives one active pre-existing-dataset
  candidate, but not paper evidence. The fixed command
  `PYTHONPATH=src:. python3 tools/openml_margin_scout.py --out results/openml_margin_scout_v0`
  checks OpenML version-1 ionosphere, sonar, diabetes, and spambase with the
  same online classifier and fixed-L2 standardized-feature perturbation
  interface. OpenML diabetes at target radius 2.0 clears the 4-seed scout gate:
  BGR reaches 0.7402 RAUC versus uniform 0.6797 (W/L/T=4/0/0), while
  fixed-radius replay reaches 0.6999. Ionosphere, sonar, and spambase remain
  rejected or near-miss scouts. The only valid next step is the fixed 30-seed
  command
  `PYTHONPATH=src:. python3 tools/openml_margin_scout.py --datasets diabetes --targets 2.0 --seeds 30 --out results/openml_diabetes_margin_30seed_v1`;
  do not add this route to the paper unless that follow-up remains positive.
- MinAtar Breakout is completed negative scope evidence, not paper-positive
  evidence. The fixed 12-seed calibration in
  `results/minatar_breakout_recovery_calibration_12seed_v1/summary.json`
  clears clean/non-flat/non-saturated prerequisites under `MinAtar==1.0.15`
  with clean 1.0000, recovery range 0.6667--1.0000, RAUC 0.7000, and r80
  0.6000. The fixed all-method screen at
  `results/minatar_breakout_recovery_probe_4seed_v1/summary.csv` is negative:
  BGR and BGR-Coverage both tie uniform at 0.8896 final RAUC with W/L/T=0/0/4,
  median r80 saturates at 5.0000, and failure-only has the best AULC. Do not
  scale or promote this route without a genuinely new preregistered premise.
- MinAtar Asterix is completed negative scope evidence, not paper evidence. The fixed 12-seed calibration in
  `results/minatar_asterix_recovery_calibration_12seed_v1/summary.json` clears
  clean/non-flat/non-saturated prerequisites under `MinAtar==1.0.15` with clean
  0.8333, recovery range 0.5000--0.8333, RAUC 0.7188, and r80 5.3333. The
  fixed all-method screen at
  `results/minatar_asterix_recovery_probe_4seed_v1/summary.csv` is negative:
  failure-only reaches 0.8625 final RAUC, above BGR-Coverage 0.8406, default
  BGR 0.8047, and uniform 0.8234; BGR-Coverage wins only 1/4 paired seeds
  against uniform. Do not scale or promote this route without a genuinely new
  preregistered premise.
- Gymnasium MuJoCo Reacher-v5 is completed negative scope evidence, not paper
  evidence. Its fixed calibration cleared clean/non-flat prerequisites
  (clean 0.8333, recovery range 0.5000--0.9167, RAUC 0.7891, r80 3.0000 on a
  0--4 grid) under `gymnasium==1.3.0` and `mujoco==3.9.0`. The full
  all-method comparison in `tools/reacher_recovery_probe.py` is negative:
  uniform reaches 0.3862 final RAUC, while BGR reaches 0.2907 and BGR-Coverage
  reaches 0.2721. This is scope evidence, not paper evidence.
- Gymnasium Box2D LunarLander-v3 is completed negative scope evidence, not
  paper evidence. Its fixed calibration cleared the pre-method gate
  (clean 0.9167, recovery range 0.5833--0.9167, RAUC 0.7722, r80 0.5300), but
  the preregistered 4-seed method screen fails promotion: BGR-Coverage has the
  best mean final RAUC (0.7500 vs. 0.6222 uniform) but wins only 2/4 paired
  seeds against uniform and has lower median r80 (0.4200 vs. 0.4825). Do not
  scale or promote this route without a genuinely new preregistered premise.
- highway-env highway-fast-v0 lane driving is a rejected pre-method
  calibration, not paper evidence. Its fixed idle lane-keep policy has clean
  success 0.6667, recovery range 0.5833--0.6667, RAUC 0.6181, and saturated
  r80 6.0000 over 12 seeds, so it fails before any BGR comparison is allowed.
  Do not build or scale this route around the current controller/interface.
- The weighted perturbation curriculum in
  `scripts/queue_openvla_oft_preregistered_weighted_perturb.sh` is now a
  negative learned-policy audit for the fixed learned-policy gate. It changed
  the OpenVLA-OFT training distribution by repeating perturbation examples
  under separate TFDS `mix_source` labels, but the completed non-identity
  totals are BGR 367/400, official 367/400, and matched random 370/400. BGR
  ties the official checkpoint and trails matched random, so do not treat this
  recipe family as an active acceptance-moving route.
- The repaired proximal-anchor route in
  `scripts/queue_openvla_oft_preregistered_proximal_anchor.sh` is also a
  completed negative learned-policy audit. It changed the optimization
  objective with an official-checkpoint proximal L2 anchor and completed after
  the DDP wrapper repair, but the final non-identity totals are BGR 368/400,
  official 367/400, and matched random 368/400, with identity BGR 98/100
  versus official 99/100. This ties matched random and beats official by only
  one episode, far below the fixed +10/400 and +0.02 gate. Do not spend more
  compute on clean-mix visual-perturb OpenVLA-OFT variants unless the
  intervention changes materially.
- The perturb-only anchored OpenVLA-OFT route in
  `scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh` is complete
  and negative. It removed clean anchors from the RLDS training data and used a
  stronger proximal anchor (`PROXIMAL_ANCHOR_L2=5.0`), but the final audit has
  BGR at 371/400 non-identity successes, official at 367/400, and matched
  random at 372/400, with identity 99/100 for all three. It fails the same
  +10/400 and +0.02 gate over both official and matched random. Do not treat
  this recipe family as active acceptance-moving work.
