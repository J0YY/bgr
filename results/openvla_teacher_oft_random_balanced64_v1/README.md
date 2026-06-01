# OpenVLA Teacher OFT Random-Balanced-64 Render

This artifact renders the random-balanced baseline analog of the BGR balanced64
OpenVLA-OFT bridge: one contiguous 16-step replay episode for each perturbation
family, covering blur, brightness, shift, and occlusion.

Generation command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 4 --mem 16G --time 00:40:00 /work/joy/bgr env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. python scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_oft_random_balanced64_v1 --method random_balanced --max-examples 64 --selection balanced_episodes --episodes-per-family 1 --max-steps-per-episode 16 --num-steps-wait 10 --env-image-size 256 --image-size 224
```

Remote log:

```text
/work/joy/bgr/logs/run_1780327075_235451672.out
```

Validation confirmed 64 rows, all from `random_balanced`, with 4 replay
episodes and 16 steps per family. Each row contains primary RGB, wrist RGB, 8D
LIBERO state, 7D teacher action, and language instruction.
