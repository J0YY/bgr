# OpenVLA Teacher OFT BGR-Boundary Balanced-2048 Render

This remote artifact scales the OpenVLA-OFT render bridge from 64 rows to a
matched BGR-boundary dataset with 2,048 rendered teacher-action rows. It selects
8 replay episodes per visual perturbation family, truncates each episode to 64
steps, and covers blur, brightness, shift, and occlusion.

Generation command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 24G --time 01:30:00 /work/joy/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_oft_bgr_balanced2048_v1 --method bgr_boundary --max-examples 2048 --selection balanced_episodes --episodes-per-family 8 --max-steps-per-episode 64 --num-steps-wait 10 --env-image-size 256 --image-size 224
```

Remote log:

```text
/work/joy/bgr/logs/run_1780379901_989185344.out
```

Validation confirmed 2,048 rows, all from `bgr_boundary`, with 32 replay
episodes, 8 episodes per family, and 64 steps per episode. The full PNG/NPZ
render tree is 492M and remains under `/work`.
