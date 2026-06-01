OpenVLA teacher-replay manifest for BGR fine-tuning.

Each row pairs a validated boundary candidate with a target action from a successful native OpenVLA rollout. The artifact intentionally stops before RLDS creation because the source logs contain actions and token IDs, but not saved images. A downstream RLDS converter must replay the native action prefix in LIBERO to render observations, apply the candidate perturbation to the rendered image stream, and write OpenVLA-OFT episodes.
