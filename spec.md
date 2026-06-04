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

- Synthetic and procedural grid-margin experiments support the main
  recovery-margin mechanism under 15 paired seeds.
- Robot-suffix experiments are manipulation-style diagnostics. The
  coverage-aware BGR-Suffix variant improves clean success, object RAUC,
  transfer RAUC, and AULC over uniform suffix replay, while uniform remains
  higher on median critical radius.
- OpenVLA/LIBERO experiments are audits of recovery-curve measurement,
  boundary selection, and data plumbing. They are not positive BGR fine-tuning
  evidence.

The paper therefore frames BGR as a boundary-centered replay principle with
controlled procedural evidence and learned-policy audits, not as a completed
robotics fine-tuning claim.
