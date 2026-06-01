Balanced OpenVLA boundary-selection audit over existing LIBERO/OpenVLA rollout artifacts.

Inputs:
- BGR-boundary: `/work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed{1..5}_lp2_h160`
- Random-balanced: `/work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed{1..5}b_skip_lp2_h160`

Command:
`scripts/summarize_openvla_boundary_selection.py --proposal-method-name bgr_boundary ... --out runs/libero_openvla_boundary_selection_balanced_v1`

This is a fixed-policy selection diagnostic over validated perturbation candidates, not an OpenVLA fine-tuning run.
