# Significance Tests v1

This artifact records paired exact sign-flip tests over the checked-in
per-seed summaries used in the current paper draft.

Generation command:

```bash
PYTHONPATH=src:. python3 scripts/analyze_significance.py --results-dir results --out-csv results/significance_tests_v1/significance_tests.csv --out-tex paper/figures/significance_table.tex
```

The five-seed comparisons are directionally consistent for the main procedural,
suffix-strategy, and estimator claims, but the exact sign-flip test has a
minimum nonzero two-sided value of 0.0625 at `n=5`. The 15-seed paired extension
configs in `configs/*_pair_15seed.yaml` are intended to replace these main
pairwise significance rows once the cluster runs complete.
