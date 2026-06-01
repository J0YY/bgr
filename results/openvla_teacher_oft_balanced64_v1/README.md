# OpenVLA Teacher OFT Balanced-64 Render

This artifact renders 64 contiguous OpenVLA-OFT-style training rows from the
teacher-replay manifest: one 16-step replay episode for each perturbation
family, covering blur, brightness, shift, and occlusion.

Generation command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:40:00 /work/joy/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_oft_balanced64_v1 --max-examples 64 --selection balanced_episodes --episodes-per-family 1 --max-steps-per-episode 16 --num-steps-wait 10 --env-image-size 256 --image-size 224
```

Remote log:

```text
/work/joy/bgr/logs/run_1780325448_898879565.out
```

Validation confirmed 64 rows, 4 replay episodes, 16 steps per family, and NPZ
fields with primary image `(224,224,3)`, wrist image `(224,224,3)`, state
`(8,)`, action `(7,)`, and language instruction.
