# OpenVLA-OFT BGR-Boundary Balanced-2048 HDF5 Pack

This remote artifact validates and packs
`openvla_teacher_oft_bgr_balanced2048_v1` into a LIBERO-style HDF5 dataset.

Generation command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 16G --time 00:30:00 /work/joy/bgr env PYTHONPATH=src:. python scripts/pack_openvla_oft_examples.py --examples runs/openvla_teacher_oft_bgr_balanced2048_v1/examples.jsonl --out runs/openvla_oft_pack_bgr_balanced2048_v1 --write-hdf5
```

Remote log:

```text
/work/joy/bgr/logs/run_1780380477_635336557.out
```

The pack summary reports 2,048 examples, four visual perturbation families, 7D
actions, 8D state, and both `agentview` and `eye_in_hand` image streams. The
full HDF5 pack is 383M and remains under `/work`.
