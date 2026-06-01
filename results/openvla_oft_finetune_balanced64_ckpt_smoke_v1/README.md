# OpenVLA-OFT Balanced-64 Fine-Tune Checkpoint Smoke

This run is a one-step OpenVLA-OFT LoRA fine-tuning smoke on the BGR balanced64
TFDS dataset. It verifies that the BGR-rendered `libero_goal_no_noops` dataset
can drive the unmodified OpenVLA-OFT training script through model loading,
RLDS loading, forward/backward optimization, and checkpoint writing.

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition gpu --gres gpu:1 --cpus 8 --mem 64G --time 01:00:00 /work/joy/bgr bash -lc 'cd /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft && env WANDB_MODE=disabled HF_HOME=/work/joy/cache_home/huggingface TRANSFORMERS_CACHE=/work/joy/cache_home/huggingface/hub PYTHONPATH=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path openvla/openvla-7b --data_root_dir /work/joy/bgr/runs/openvla_oft_tfds_libero_goal_balanced64_v1 --dataset_name libero_goal_no_noops --run_root_dir /work/joy/bgr/runs/openvla_oft_finetune_balanced64_ckpt_smoke_v1 --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio True --batch_size 1 --learning_rate 5e-4 --num_steps_before_decay 100000 --max_steps 1 --save_freq 1 --save_latest_checkpoint_only True --image_aug False --lora_rank 8 --merge_lora_during_training False --shuffle_buffer_size 16 --wandb_entity disabled --wandb_project bgr --run_id_note bgr-balanced64-ckpt-smoke --wandb_log_freq 1'
```

Remote log:

```text
/work/joy/bgr/logs/run_1780326595_120788212.out
```

Key evidence from the log:

- OpenVLA 7B loaded from the Hugging Face cache.
- LoRA initialized with rank 8.
- Trainable modules included LoRA, proprio projector, and L1 action head.
- OpenVLA-OFT loaded the BGR TFDS dataset and reused its 64-transition,
  4-trajectory statistics.
- The run reached `Max step 1 reached! Stopping training...`.
- Checkpoint files were written under `/work/joy/bgr/runs`.

Large checkpoint weights are intentionally not checked into git. The remote
checkpoint directory is:

```text
/work/joy/bgr/runs/openvla_oft_finetune_balanced64_ckpt_smoke_v1/openvla-7b+libero_goal_no_noops+b1+lr-0.0005+lora-r8+dropout-0.0--bgr-balanced64-ckpt-smoke
```

The checked-in `dataset_statistics.json` is the small metadata file saved by
OpenVLA-OFT for this run.
