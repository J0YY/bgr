# Estimator Pair 15-Seed v1

This paired extension reruns the boundary-estimator comparison for `uniform`
and `active` over 15 seeds with the same 17-probe budget per state.

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:30:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/run_estimator_experiment.py --config configs/estimator_pair_15seed.yaml --out runs/estimator_pair_15seed_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780327938_068713790.out
```

Mean results over 15 paired seeds:

| Method | r80 MAE | RAUC MAE | Boundary hit rate |
|---|---:|---:|---:|
| Active BGR | 0.0799 | 0.0645 | 0.6775 |
| Uniform | 0.1064 | 0.0656 | 0.6007 |

Paired exact sign-flip tests give `p=0.00006` for active improving boundary-hit
rate and lowering r80 MAE relative to uniform.
