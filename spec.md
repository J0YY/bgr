# Boundary-Guided Replay Design Note

This file is a short design note for the current repository state. The active
submission manuscript is `paper/main.tex`; result provenance is recorded in
`results/README.md`; executable experiment entry points are listed in
`README.md`.

Boundary-Guided Replay (BGR) trains decision policies from replayable states
near an estimated success-failure boundary. For each replayable state, BGR
estimates a recovery curve over perturbation radius, derives a critical radius,
and samples training perturbations near that state-conditioned boundary while
mixing in clean and broader coverage examples.

The current evidence is tiered:

- Synthetic experiments support the intended recovery-margin mechanism, with a
  15-seed rendered study and a 30-seed confirmation in the anonymous artifact.
- Procedural grid-margin experiments support the main recovery-margin claim. The
  completed 30-seed full-baseline comparison and held-out 30-seed replication
  pool to 60/0 paired RAUC wins for BGR over uniform.
- Robot-suffix experiments are positive manipulation-style evidence. The
  coverage-aware BGR-Suffix variant improves clean success, object RAUC,
  transfer RAUC, and AULC over uniform suffix replay across original and held-out
  30-seed sweeps and four suffix stress regimes, while uniform remains higher on
  median critical radius.
- OpenVLA/LIBERO experiments are audits of recovery-curve measurement,
  boundary selection, action-label/TFDS plumbing, and bounded full-goal
  evaluation. The compact OpenVLA validation artifact verifies 2,048-transition
  matched BGR/random exports with 7D actions and 8D state, but these audits are
  not positive BGR fine-tuning evidence because BGR does not stably outperform
  both matched random selection and the unadapted official checkpoint.

The paper therefore frames BGR as a boundary-centered replay principle with
controlled procedural evidence, positive manipulation-style simulator evidence,
and learned-policy audits, not as a completed robotics fine-tuning claim.
