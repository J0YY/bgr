# Significance Tests v2

This artifact records paired exact sign-flip tests after adding the 15-seed
paired extensions for the main procedural, suffix, and estimator comparisons.

Generation command:

```bash
PYTHONPATH=src:. python3 scripts/analyze_significance.py --results-dir results --out-csv results/significance_tests_v2/significance_tests.csv --out-tex paper/figures/significance_table.tex
```

The key 15-seed rows now have p=0.0001 in the rounded CSV output
(exact value 2/32768 = 0.000061 for all-paired-direction effects). The suffix
final object-RAUC row remains negative for BGR-Broad and is intentionally kept
in the CSV as a limitation.
