# OpenVLA-OFT TFDS Smoke Export

This artifact is a four-episode RLDS-style TFDS smoke dataset exported from the
rendered OpenVLA-OFT field examples in `results/openvla_teacher_oft_smoke_v1`.
It covers one example each for blur, brightness, shift, and occlusion.

Generation command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:20:00 /work/joy/bgr /work/joy/safesae-openvla/bin/python scripts/export_openvla_oft_tfds.py --examples results/openvla_teacher_oft_smoke_v1/examples.jsonl --out runs/openvla_oft_tfds_smoke_v1 --dataset-name bgr_libero_oft_smoke --version 1.0.0
```

Readback smoke:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 2 --mem 8G --time 00:10:00 /work/joy/bgr /work/joy/safesae-openvla/bin/python -c "import tensorflow_datasets as tfds; b=tfds.builder_from_directory('runs/openvla_oft_tfds_smoke_v1/bgr_libero_oft_smoke/1.0.0'); print(b.info.full_name); print(b.info.splits); ex=next(iter(tfds.as_numpy(b.as_dataset(split='train')))); step=next(iter(ex['steps'])); print(ex['episode_metadata']['task_name'].decode()); print(step['observation']['image'].shape, step['observation']['wrist_image'].shape, step['observation']['state'].shape, step['action'].shape, step['language_instruction'].decode())"
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780323827_280717056.out
/work/joy/bgr/logs/run_1780323900_674014804.out
```

The readback validates `tfds.builder_from_directory`, four train examples, and
step tensors with image `(224,224,3)`, wrist image `(224,224,3)`, state `(8,)`,
and action `(7,)`.
