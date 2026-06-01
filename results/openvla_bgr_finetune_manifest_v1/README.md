OpenVLA BGR fine-tuning manifest exported from validated balanced boundary-selection artifacts.

Inputs:
- BGR-boundary: `/work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed{1..5}_lp2_h160`
- Random-balanced: `/work/joy/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed{1..5}b_skip_lp2_h160`

Outputs:
- `manifest.jsonl` and `manifest.csv`: validated perturbation candidates with boundary-band labels.
- `openvla_oft_finetune_plan.sh`: OpenVLA-OFT LoRA command template for matched BGR-boundary and random-balanced datasets.

This is a scaffold for the learned-policy BGR-Suffix experiment. It is not a completed fine-tuning result because OpenVLA-OFT consumes RLDS episodes, and no matching RLDS dataset for these replay states was found under `/work`.
