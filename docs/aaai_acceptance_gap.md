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
   passenger/destination state. This is promising because the environment is
   canonical, resettable, and has exact feasibility, but the scratch prototype
   must be optimized before it can produce reliable evidence.
2. A faster discrete control benchmark with exact reset states and a
   pre-registered perturbation family. CliffWalking and FrozenLake currently
   look negative for BGR under the tested protocols.
3. A larger OpenVLA/LIBERO adaptation only if the recipe changes in a way that
   plausibly beats both matched random and the official checkpoint, not merely a
   different perturbation score.

## Immediate Engineering Work

- Implement a cached Taxi-v3 diagnostic before running more seeds.
- Add a small smoke test for Taxi transition dynamics against the Gym Taxi map
  if the diagnostic is promoted.
- Keep scratch negative runs out of the anonymous package unless they are being
  used as explicit limitations.
