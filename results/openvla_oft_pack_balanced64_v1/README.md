# OpenVLA-OFT Balanced-64 HDF5 Pack

This artifact validates and packs `openvla_teacher_oft_balanced64_v1` into a
LIBERO-style HDF5 dataset with four `data/demo_*` groups, one 16-step demo per
perturbation family.

Generation command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/pack_openvla_oft_examples.py --examples runs/openvla_teacher_oft_balanced64_v1/examples.jsonl --out runs/openvla_oft_pack_balanced64_v1 --write-hdf5
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780325655_972521234.out
/work/joy/bgr/logs/run_1780325709_771885875.out
```

The readback smoke found 4 demos. Each demo has `actions (16,7)`,
`obs/agentview_rgb (16,224,224,3)`, `obs/eye_in_hand_rgb (16,224,224,3)`,
`obs/ee_states (16,6)`, and `obs/gripper_states (16,2)`.
