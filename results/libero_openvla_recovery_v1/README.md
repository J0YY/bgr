# LIBERO/OpenVLA Recovery Summary v1

This run converts existing closed-loop OpenVLA/LIBERO object-task rollouts into
BGR-style recovery curves. It is a learned-policy recovery audit, not a BGR
fine-tuning result.

Source artifact:

```text
/work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_object3_h220_bash
```

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/summarize_libero_openvla_recovery.py --input-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_object3_h220_bash --out runs/libero_openvla_recovery_v1 --source-name libero_openvla_observation_object3_h220_bash
```

Remote log:

```text
/work/joy/bgr/logs/run_1780320300_854946446.out
```

Mean state-conditioned recovery metrics over nine LIBERO-Object replay states:

| Perturbation family | Clean | RAUC | r80 | r50 |
|---|---:|---:|---:|---:|
| blur | 1.0000 | 0.4667 | 0.3067 | 0.4667 |
| brightness | 1.0000 | 0.8000 | 0.7333 | 0.8000 |
| occlusion | 1.0000 | 0.3889 | 0.2822 | 0.3889 |
| shift | 1.0000 | 0.5148 | 0.3793 | 0.5148 |

Interpretation: a real closed-loop OpenVLA policy succeeds on the unperturbed
LIBERO-Object replay states but has measurable success-failure boundaries under
image perturbation families. This supports the BGR premise that recovery curves
are meaningful for learned VLA policies. It does not show BGR-Suffix training
improvement because these artifacts evaluate a fixed OpenVLA policy.
