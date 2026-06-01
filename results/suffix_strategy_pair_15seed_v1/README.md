# Suffix Strategy Pair 15-Seed v1

This paired extension reruns the robot-suffix strategy comparison for
`bgr_broad` and `uniform` over 15 paired seeds. It supports the clean-success,
sample-efficiency, and perturbation-family transfer claims while preserving the
limitation that uniform still has slightly higher final object-perturbation
RAUC.

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 12G --time 04:00:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_suffix_experiment.py --config configs/suffix_strategy_pair_15seed.yaml --out runs/suffix_strategy_pair_15seed_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780328104_484552023.out
```

Mean results over 15 paired seeds:

| Method | Clean | Object RAUC | Median r80 | EE-transfer RAUC | RAUC AULC |
|---|---:|---:|---:|---:|---:|
| BGR-Broad | 0.8691 | 0.4815 | 0.4703 | 0.3165 | 0.3790 |
| Uniform | 0.8358 | 0.4844 | 0.5046 | 0.3075 | 0.3707 |

Paired exact sign-flip tests give `p=0.00006` for BGR-Broad improving clean
success, transfer RAUC, and RAUC AULC. Uniform remains higher on final
object-perturbation RAUC, so this result is framed as a diagnostic robotics
instantiation rather than a completed robotics win.
