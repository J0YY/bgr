# OpenVLA-OFT Random-Balanced-64 HDF5 Pack

This artifact validates and packs `openvla_teacher_oft_random_balanced64_v1`
into a LIBERO-style HDF5 dataset with four `data/demo_*` groups, one 16-step
demo per perturbation family.

Generation command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/pack_openvla_oft_examples.py --examples runs/openvla_teacher_oft_random_balanced64_v1/examples.jsonl --out runs/openvla_oft_pack_random_balanced64_v1 --write-hdf5
```

Remote log:

```text
/work/joy/bgr/logs/run_1780327145_234510679.out
```

The pack summary reports 64 examples, four visual perturbation families, 7D
actions, 8D state, and both `agentview` and `eye_in_hand` image streams.
