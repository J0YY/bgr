# Grid Margin Pair 15-Seed v1

This paired extension reruns the main procedural grid-margin comparison for
`bgr` and `uniform` over 15 paired seeds. It is intended to strengthen the
primary statistical claim while the five-seed full-baseline and ablation runs
remain available as supporting diagnostics.

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_grid_margin_experiment.py --config configs/grid_margin_pair_15seed.yaml --out runs/grid_margin_pair_15seed_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780328088_151681007.out
```

Mean results over 15 paired seeds:

| Method | Clean | RAUC | Median r80 | RAUC AULC |
|---|---:|---:|---:|---:|
| BGR | 0.9461 | 0.4345 | 0.3441 | 0.3525 |
| Uniform | 0.8943 | 0.3961 | 0.3318 | 0.3129 |

Paired exact sign-flip tests give `p=0.00006` for BGR improving clean success,
RAUC, median r80, and RAUC AULC relative to uniform.
