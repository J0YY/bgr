# OpenVLA-OFT Balanced-2048 Step-10 Merge and Eval Smoke

This artifact records the first closed-loop evaluation smoke for the matched
BGR-boundary and random-balanced 2,048-step OpenVLA-OFT checkpoints. It is not
a performance result: both policies were trained for only 10 optimizer steps and
were evaluated on one short LIBERO-Goal rollout. The purpose is to verify the
remaining path from BGR-selected TFDS data to merged OpenVLA checkpoints and a
stock LIBERO closed-loop rollout.

Merge commands:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition low-prio-gpu --gres gpu:a6000:1 --cpus 8 --mem 80G --time 02:00:00 /work/joy/bgr bash -lc 'cd /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft && env WANDB_MODE=disabled HF_HOME=/work/joy/cache_home/huggingface TRANSFORMERS_CACHE=/work/joy/cache_home/huggingface/hub PYTHONPATH=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/python vla-scripts/merge_lora_weights_and_save.py --base_checkpoint openvla/openvla-7b --lora_finetuned_checkpoint_dir /work/joy/bgr/runs/openvla_oft_finetune_bgr_balanced2048_step10_v1/openvla-7b+libero_goal_no_noops+b1+lr-0.0005+lora-r8+dropout-0.0--bgr-balanced2048-step10'
~/remote_srun.sh --github-test --git-pull --log --partition low-prio-gpu --gres gpu:a6000:1 --cpus 8 --mem 80G --time 02:00:00 /work/joy/bgr bash -lc 'cd /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft && env WANDB_MODE=disabled HF_HOME=/work/joy/cache_home/huggingface TRANSFORMERS_CACHE=/work/joy/cache_home/huggingface/hub PYTHONPATH=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/python vla-scripts/merge_lora_weights_and_save.py --base_checkpoint openvla/openvla-7b --lora_finetuned_checkpoint_dir /work/joy/bgr/runs/openvla_oft_finetune_random_balanced2048_step10_v1/openvla-7b+libero_goal_no_noops+b1+lr-0.0005+lora-r8+dropout-0.0--random-balanced2048-step10'
```

Eval commands use the `safesae` conda environment from `remote_srun.sh`, plus
the OpenVLA-OFT source tree, the LIBERO source tree, and the OpenVLA-OFT venv
site-packages on `PYTHONPATH`:

```bash
PYTHONPATH=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft:/athenahomes/joy/LIBERO:/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/lib/python3.10/site-packages
```

Remote Slurm logs:

```text
/work/joy/bgr/logs/run_1780382513_105504870.out  # BGR merge
/work/joy/bgr/logs/run_1780382656_581424411.out  # random merge
/work/joy/bgr/logs/run_1780383317_343334235.out  # BGR eval
/work/joy/bgr/logs/run_1780383458_588753145.out  # random eval
```

Key evidence:

- Both LoRA checkpoints were merged into local OpenVLA checkpoint roots.
- Each merged checkpoint root is 15G under `/work/joy/bgr/runs`.
- Both merged checkpoints load through the stock OpenVLA-OFT LIBERO eval code.
- Both eval runs instantiate LIBERO-Goal, run task 0 episode 0 with a 20-step
  horizon, save a rollout MP4 under the external OpenVLA-OFT tree, and log final
  results.
- BGR and random each scored 0/1 in this deliberately tiny smoke.

