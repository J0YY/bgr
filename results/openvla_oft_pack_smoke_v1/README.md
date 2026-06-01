OpenVLA-OFT LIBERO HDF5 smoke pack from rendered BGR teacher examples.

Command:
`scripts/pack_openvla_oft_examples.py --examples results/openvla_teacher_oft_smoke_v1/examples.jsonl --out runs/openvla_oft_pack_smoke_v1 --write-hdf5`

This validates and packs the four rendered OFT-field examples into a LIBERO-style HDF5 file with `data/demo_*` groups, `actions`, `obs/agentview_rgb`, `obs/eye_in_hand_rgb`, `obs/ee_states`, and `obs/gripper_states`. This is a smoke artifact for the HDF5-to-RLDS bridge, not a completed RLDS dataset or fine-tuning run.
