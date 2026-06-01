# LIBERO/OpenVLA Boundary Selection v1

This diagnostic summarizes existing OpenVLA/LIBERO perturbation-selection
artifacts as a boundary-discovery test. It compares proposal-guided candidate
selection against family-balanced random selection under the same eight-candidate
validation budget. The target boundary band is observed counterfactual failure
rate in `[0.25, 0.75]`, centered on the success-failure transition rather than
maximal failure.

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/summarize_openvla_boundary_selection.py --proposal-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_guided_h160 --proposal-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_guided_seed2_h160 --proposal-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_guided_seed3_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed1b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed2b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed3b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed4b_skip_lp2_h160 --random-dir /work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed5b_skip_lp2_h160 --out runs/libero_openvla_boundary_selection_v1
```

Remote log:

```text
/work/joy/bgr/logs/run_1780320651_072406017.out
```

Aggregate results:

| Method | Runs | Mean CF rate | Boundary hit rate | Mean abs(CF-0.5) | Certificates |
|---|---:|---:|---:|---:|---:|
| proposal-guided | 3 | 0.5278 | 0.8750 | 0.1667 | 25.33 |
| random-balanced | 5 | 0.6438 | 0.5750 | 0.2563 | 20.60 |

Interpretation: proposal-guided selection finds perturbations much closer to
the observed success-failure boundary than random-balanced selection. This is
OpenVLA active-selection evidence for BGR's boundary-discovery premise, not a
BGR-Suffix training result. Caveat: the proposal-guided artifacts concentrate on
blur/shift families, while random-balanced artifacts cover blur, brightness,
occlusion, and shift.
