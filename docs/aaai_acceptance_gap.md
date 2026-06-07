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

Audit this current status from artifacts with:
`PYTHONPATH=src:. python3 scripts/check_acceptance_readiness.py --root .`.
Quantify the distances to those gates with:
`PYTHONPATH=src:. python3 scripts/acceptance_scorecard.py --root . --out docs/acceptance_scorecard.md`.
The checker is intentionally stricter than the package guard: package checks
verify that claims match artifacts, while this readiness check verifies whether
the evidence has actually cleared the independent-benchmark and learned-policy
promotion gates.

The generated scorecard now reports why each independent-benchmark screen fails
the gate. As of the 2026-06-05 refresh, the closest screen is the
official-package MiniGrid FourRooms BGR-Coverage run: it has a visible RAUC lead
over uniform and required baselines, but fails because both the original
relative-radius metric and the absolute-r10 follow-up are saturated. The
non-saturated midband distance-2-to-5 follow-up is also negative because default
BGR trails fixed-radius and failure-only replay, wins only 2/4 paired seeds over
uniform, and has lower median r80 than uniform. Any further independent-benchmark
attempt should therefore fix non-saturated radius evidence before method
comparison and avoid rerunning the same MiniGrid protocol unless the premise
materially changes.

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

Latest learned-policy audit, completed 2026-06-07: the perturb-only anchored
OpenVLA-OFT route preserves clean identity success for BGR, matched random, and
the official checkpoint at 99/100 each, but fails the promotion gate on
non-identity perturbations. BGR reaches 371/400, official reaches 367/400, and
matched random reaches 372/400. The official gap is only +4/400 (+0.0100) and
BGR trails matched random by 1/400 (-0.0025), below the fixed +10/400 and +0.02
requirement. This sharpens the acceptance problem: the current learned-policy
evidence remains an audit, not positive robotics evidence.

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
- The next acceptance-moving empirical route must change the premise: either a
  genuinely different pre-existing benchmark package/reset interface, or a
  genuinely different learned-policy intervention that is preregistered before
  data generation and can plausibly beat both matched random and the official
  checkpoint. The current OpenVLA-OFT clean-mix/visual-perturbation recipe
  family is exhausted for acceptance purposes.
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
- The latest preregistered learned-policy intervention is the weighted
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
- The next preregistered learned-policy route changes the optimization
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
- The next preregistered learned-policy route changes the data/objective
  combination rather than the clean-mix weighting: perturb-only anchored
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
  the adaptation jobs and BGR/random perturb evals dependency-pending. This
  route remains in flight and should not be added to `paper/main.tex` unless
  compact summaries clear the fixed learned-policy promotion gate. Poll and
  sync the route with
  `REMOTE_RUN_ROOT=/work/joy/bgr/runs scripts/sync_openvla_oft_perturb_only_anchor_results.sh --poll`
  and
  `REMOTE_RUN_ROOT=/work/joy/bgr/runs scripts/sync_openvla_oft_perturb_only_anchor_results.sh --sync`.
  A helper poll at 2026-06-07 22:08:57 BST showed prep `767789` completed
  cleanly at 22:06:54 BST, BGR adaptation `767790` running on `c1-g4-04`, and
  official identity eval `767796` running on `c2-g4-24`; both compact summary
  paths were still missing, so the route remains unevaluated.
  A later helper poll at 2026-06-07 22:19:57 BST showed BGR train/merge
  `767790`/`767791`, random train/merge `767793`/`767794`, and official
  identity `767796` completed cleanly. BGR clean eval `767792`, random clean
  eval `767795`, BGR identity perturb eval `767801`, random identity perturb
  eval `767806`, and official blur `767797` were running; compact summaries
  were still missing, so the route still cannot be scored.
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
  The fixed pick-place controller does not provide a usable clean recovery
  interface, so do not build or scale a FetchPickAndPlace replay comparison
  around this controller/interface.
- Gymnasium MuJoCo Reacher-v5 is now the active pre-method calibration route
  because it changes both package and reset interface relative to the failed
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
- Gymnasium MuJoCo InvertedPendulum-v5 is the next independent pre-method
  route because it changes both task family and reset dynamics relative to the
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
