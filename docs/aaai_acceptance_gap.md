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

## Promotion Criteria For A New Independent Benchmark

A new benchmark result should be promoted into `paper/main.tex` only if it meets
all of these criteria before paper writing:

- The environment is pre-existing and recognizable, or the perturbation protocol
  is fixed before seeing results.
- The run uses at least 30 paired seeds for BGR and each baseline.
- BGR beats uniform replay on final RAUC with at least 24/30 paired wins and a
  practically visible mean gap.
- BGR also beats failure-only and loss/loss-proxy replay on final RAUC.
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
3. A larger OpenVLA/LIBERO adaptation only if the recipe changes in a way that
   plausibly beats both matched random and the official checkpoint, not merely a
   different perturbation score.

## Immediate Engineering Work

- If Taxi is revisited, change the pre-registered perturbation/training budget
  first, then rerun a small diagnostic before any 30-seed scale-up.
- Add a small smoke test for Taxi transition dynamics against the Gym Taxi map
  only if the diagnostic is promoted into `src/`.
- Keep scratch negative runs out of the anonymous package unless they are being
  used as explicit limitations.
