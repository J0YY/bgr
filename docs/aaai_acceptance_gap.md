# AAAI Acceptance Gap

This is an internal working note, not part of the anonymous submission package.

## Current Assessment

The paper is now framed as a mechanism study and the strongest overclaim risks
are explicit in the title, introduction, limitations, README, and package
guards. That improves reviewability, but it does not by itself make the paper a
90%+ AAAI main-track accept. The remaining acceptance blocker is evidence:

- no clear positive result on an independent pre-existing benchmark;
- no stable learned-policy OpenVLA/LIBERO improvement over both matched random
  and the official checkpoint;
- suffix and synthetic gains remain small even though paired consistency is
  strong;
- the strongest result is still the controlled procedural grid-margin
  mechanism benchmark.

After the weak-reject style review, the immediate paper-defense priority is not
to amplify p-values or add more authored toy wins. The manuscript should instead
make the evidence contract unavoidable:

- lead with the grid-margin mechanism effect because it is the only visibly
  large effect;
- describe synthetic and suffix results as modest scoped support, not broad
  robustness evidence;
- keep standard-environment diagnostics in the limitations unless they satisfy
  the promotion criteria below;
- keep OpenVLA/LIBERO framed as an audit until BGR beats both matched random and
  the official checkpoint on the same fixed evaluation;
- foreground mean differences, clean success, and median critical radius before
  sign-test p-values;
- treat the local boundary proposition as design intuition, not a theory claim.

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

## Next Experiment Candidates

Stop-rule after the FourRooms screen: do not add more local classic-control or
small tabular recovery probes under this evidence plan. FrozenLake, Taxi,
CliffWalking, FourRooms, MountainCar, CartPole, Acrobot, and Pendulum now cover
the obvious low-cost standard-environment checks, and none meets the promotion
gate. Further acceptance-moving evidence should use either (a) an installed
external benchmark package with recognizable task definitions and no local
reimplementation of the environment, or (b) a genuinely different learned-policy
intervention. A local dependency check on 2026-06-05 found `gymnasium`, `gym`,
`minigrid`, `d4rl`, `metaworld`, `procgen`, `mujoco`, and
`stable_baselines3` absent from the current Python environment, so the next
external benchmark attempt must start by installing/verifying the benchmark
package and recording its version before any result is run.

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
   - Official MiniGrid-FourRooms is now the next fixed 4-seed screen at
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
     A baseline-only replay-distance calibration was run before any method
     comparison for the next screen. The original spread selector was too hard
     at small absolute radii, the goalward selector was too easy, and a
     mid-distance band of shortest-path distances 2--6 gave a usable seed-0
     uniform diagnostic: final RAUC 0.5652, clean success 0.7188, and median
     r80 0.55. The next preregistered 4-seed screen is therefore:
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
     The paper-facing consequence is unchanged: MiniGrid belongs only in the
     limitations/scope audit unless a different preregistered external package
     benchmark clears the promotion gate before method comparison.
4. PointMaze/D4RL-style continuous navigation only if a real installed benchmark
   package is available. This is the best mechanistic fit because resettable
   continuous states, corridor bottlenecks, and distance-to-goal perturbations
   should expose recovery margins without relying on image-level policy
   fine-tuning. The promotion gate is the same: no 30-seed scale-up unless a
   fixed 4-seed screen beats uniform and hard-state/loss baselines with a
   visible final-RAUC gap and non-saturated critical-radius metrics.
   - Official PointMaze is now the next fixed external-package screen at
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
5. A larger OpenVLA/LIBERO adaptation only if the recipe changes in a way that
   plausibly beats both matched random and the official checkpoint, not merely a
   different perturbation score.
   - The next preregistered candidate is
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

- If Taxi is revisited, change the pre-registered perturbation/training budget
  first, then rerun a small diagnostic before any 30-seed scale-up.
- Add a small smoke test for Taxi transition dynamics against the Gym Taxi map
  only if the diagnostic is promoted into `src/`.
- Treat the CliffWalking probe as an internal negative and paper limitation:
  the default protocol saturates, while the harder undertrained protocol has
  BGR below uniform and all strong baselines.
- Treat the Acrobot probe as an internal negative unless a new protocol first
  creates a visible BGR or BGR-Coverage effect over uniform before scale-up.
- Treat the Pendulum probe as an internal negative unless a new protocol first
  avoids endpoint collapse and saturated median-r80 metrics.
- Treat the MountainCar probe as an internal negative unless a new protocol is
  pre-registered before rerunning; do not tune it into the paper post hoc.
- Treat the CartPole probe as an internal negative unless a new protocol first
  fixes clean-success saturation and defines a non-contradictory radius metric.
- Do not add another local classic-control/tabular probe after FourRooms. The
  next benchmark attempt must use an external package such as MiniGrid or
  PointMaze/D4RL, with the package/version recorded before any result is run.
- Treat official MiniGrid-FourRooms as the next highest-leverage follow-up, but
  only after preregistering a radius check that does not saturate when clean
  success is low. The current 4-seed BGR-Coverage RAUC lead is not yet a paper
  claim.
- The first MiniGrid absolute-radius follow-up failed because `final_abs_r10`
  floor-saturates at 0.0 for every method. Do not scale MiniGrid until the
  replay-state selection or training budget yields a non-saturated absolute
  radius check before method comparison.
- Keep scratch negative runs out of the anonymous package unless they are being
  used as explicit limitations.
- Do not spend more robotics compute on the current OpenVLA recipe family unless
  the next attempt changes the learned-policy intervention in a preregistered
  way that plausibly beats both official and matched random; the latest
  preregistered run reinforces the current negative audit.
