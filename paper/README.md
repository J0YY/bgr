# AAAI-27 Paper

`AuthorKit27` is the official AAAI-27 author kit downloaded from `https://aaai.org/authorkit27/` on 2026-06-01. The AAAI-27 conference page lists the author kit link and the main timetable at `https://aaai.org/conference/aaai/aaai-27/`.

AAAI rule sources were rechecked on 2026-06-04. On that date, the official
AAAI-27 page linked the AAAI-27 Author Kit and gave the July 21, 2026 abstract,
July 28, 2026 full-paper, and July 31, 2026 supplementary material/code
deadlines. The AAAI-27 page and bundled AAAI-27 AuthorKit are therefore the
package authorities for timetable, style, and source-build checks. The package
gate treats the kit's submission-mode anonymity, letter-paper geometry,
embedded-font hygiene, embedded checklist placement, and the 7-page
technical-content limit as submission-critical checks.

Build the main paper from this directory when `latexmk` is available:

```bash
latexmk -pdf main.tex
```

The AAAI style requires pdfTeX; Tectonic/XeTeX is not a suitable fallback for
this AuthorKit. On hosts without `pdflatex`, use the repository package gate
below to verify the included PDFs for page limits, double-blind hygiene,
embedded-checklist order, PDF metadata hygiene, and rendered/source claim sync.
Clean rebuilt PDFs with `qpdf --remove-info --remove-metadata` before rerunning
the package gate; rebuild on a pdfTeX-enabled host before replacing the PDFs.

Build the reproducibility checklist separately:

```bash
pdflatex -interaction=nonstopmode ReproducibilityChecklist.tex
```

From the repository root, verify the paper-facing claims and complete
submission package before submission:

```bash
PYTHONPATH=src:. python3 scripts/check_paper_claims.py --paper paper/main.tex --results-dir results --figures-dir paper/figures
PYTHONPATH=src:. python3 scripts/check_submission_package.py --root .
```

The packaged `main.tex` is an anonymous AAAI submission manuscript with
synthetic mechanism, estimator-validation, procedural grid-margin, grid-scope
diagnostic, OpenVLA audit, and embedded checklist evidence. The rendered
synthetic table checks the intended recovery-margin sampler over 30 paired
seeds from `results/toy_30seed_v1/summary.csv`. The rendered
active-estimator validation checks that boundary-focused probes recover useful
critical radii at a small fixed rollout budget over 30 paired seeds from
`results/estimator_pair_30seed_v1/summary.csv`. The procedural grid-margin
section reports the completed 30-seed full-baseline confirmation, while retaining 15-seed
target-margin sensitivity over targets 0.26--0.54 and stepwise learning-curve
sign tests showing BGR leads uniform at every post-update evaluation checkpoint.
The anonymous artifact also includes the 30-seed target-margin confirmation
`results/grid_margin_target_sensitivity_30seed_v1/summary.csv`, which shows
30/0 paired wins over the 30-seed uniform baseline on final RAUC, RAUC AULC,
and clean success at every tested target margin.
It also includes the 30-seed learning-rate confirmation
`results/grid_margin_learning_rate_sensitivity_30seed_v1/summary.csv`, which
preserves the high-learning-rate final-RAUC caveat while confirming 30/0 paired
wins for RAUC AULC and clean success at all tested rates.
The 30-seed regime diagnostic confirmation
`results/grid_margin_regime_sensitivity_30seed_v1/summary.csv` shows 30/0 paired
wins for final RAUC, RAUC AULC, clean success, and median r80 in each tested
`obstacle_prob`/`max_offset` regime, while preserving the caveat that this sweep
mostly reproduces the nominal margin dynamics rather than separate robustness
evidence.
The 30-seed stress confirmation
`results/grid_margin_stress_sensitivity_30seed_v1/summary.csv` shows 30/0 paired
wins for final RAUC, RAUC AULC, and clean success under diffuse-boundary,
low-feasibility, and sharp-low-margin stress cases, while leaving diffuse
median r80 as a non-promoted caveat.
A held-out seeds 30-59 grid-margin replication again gives BGR higher RAUC than
uniform (0.4340 vs. 0.3967) with 30/0 paired wins. Pooling the original and
held-out grid sweeps gives 0.4342 vs. 0.3965 RAUC with 60/0 paired wins. The
robot-suffix simulator reports
a 30-seed coverage-aware BGR-Suffix variant that beats uniform on final object
RAUC, clean success, transfer RAUC, and AULC, while still trailing uniform on median
critical radius.
The held-out suffix full-baseline replication
`results/suffix_coverage_full_replication_30seed_v1/summary.csv` repeats the
six-method comparison on seeds 30--59; BGR-Coverage again beats clean-only,
fixed-radius, failure-only, loss-priority, and uniform replay on final object
RAUC, transfer RAUC, and AULC with 30/0 paired wins, while preserving the
clean-only clean-success and uniform median-r80 caveats.
A held-out seeds 30-59 replication is included in the anonymous artifact; it
again gives BGR-Coverage higher final object RAUC than uniform (0.4972 vs.
0.4859) with 30/0 paired wins, while preserving the median-r80 caveat.
Pooling the original and held-out suffix sweeps gives object RAUC 0.4971 vs.
0.4856, clean success 0.8642 vs. 0.8367, transfer RAUC 0.3150 vs. 0.3089, and
AULC 0.3829 vs. 0.3717, while uniform remains higher on median r80.
The 30-seed suffix stress confirmation
`results/suffix_stress_sensitivity_30seed_v1/summary.csv` varies teacher
quality, clutter, feasible radii, and boundary sharpness. BGR-Coverage beats
uniform with 30/0 paired wins on clean success, final object RAUC, transfer
RAUC, and AULC in all four stress cases, while preserving the uniform
median-r80 caveat.
The OpenVLA-OFT bridge includes corrected clean-mix diagnostics that preserve
competence while keeping the claim scoped. The packaged action-label/TFDS
plumbing audit validates 2,048-transition matched BGR/random exports with 7D
actions and 8D state. The pooled p1024
visual-perturbation evidence gives BGR 0.8550 vs. 0.8400 for random, trailing
official at 0.8700. At p2048, the full-goal identity audit gives 99/100 clean
successes for BGR, matched random, and the official checkpoint. The 10-task
visual perturbation audit gives BGR 367/400 perturbed successes, tying official
and trailing matched random by one episode (368/400). The 300-step
image-augmentation continuation gives BGR and matched random 368/400 perturbed
successes each, only one episode above official (367/400), while BGR trails both
on identity. The 1,000-step low-learning-rate continuation is also negative:
BGR gives 366/400 non-identity perturbation successes, trailing official at
367/400 and matched random at 370/400. The follow-up weighted perturbation
curriculum is also negative: BGR and official tie at 367/400 non-identity
successes while matched random reaches 370/400. OpenVLA therefore remains
an audit of recovery curves, matched action/TFDS construction, and OpenVLA-OFT plumbing
rather than a robotics success claim.
