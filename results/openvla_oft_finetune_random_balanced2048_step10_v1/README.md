# OpenVLA-OFT Random-Balanced-2048 Step-10 Fine-Tune

This artifact records the matched random-balanced baseline for
`openvla_oft_finetune_bgr_balanced2048_step10_v1`. It uses the same
OpenVLA-OFT command shape and hyperparameters, replacing only the TFDS root,
run root, and run note.

Command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition low-prio-gpu --gres gpu:a6000:1 --cpus 8 --mem 64G --time 02:00:00 /work/joy/bgr bash -lc 'cd /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft && env WANDB_MODE=disabled HF_HOME=/work/joy/cache_home/huggingface TRANSFORMERS_CACHE=/work/joy/cache_home/huggingface/hub PYTHONPATH=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path openvla/openvla-7b --data_root_dir /work/joy/bgr/runs/openvla_oft_tfds_libero_goal_random_balanced2048_v1 --dataset_name libero_goal_no_noops --run_root_dir /work/joy/bgr/runs/openvla_oft_finetune_random_balanced2048_step10_v1 --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio True --batch_size 1 --learning_rate 5e-4 --num_steps_before_decay 100000 --max_steps 10 --save_freq 10 --save_latest_checkpoint_only True --image_aug False --lora_rank 8 --merge_lora_during_training False --shuffle_buffer_size 512 --wandb_entity disabled --wandb_project bgr --run_id_note random-balanced2048-step10 --wandb_log_freq 1'
```

Remote log:

```text
/work/joy/bgr/logs/run_1780382026_303226869.out
```

Key evidence:

- Slurm allocated job 763255 on `c2-g4-22`.
- OpenVLA 7B loaded from the Hugging Face cache.
- LoRA rank 8, proprio projector, and L1 action head were initialized.
- OpenVLA-OFT loaded the random-balanced `libero_goal_no_noops` TFDS dataset
  with 2,048 transitions and 32 trajectories.
- The run reached `Max step 10 reached! Stopping training...`.
- Checkpoint files were written under `/work/joy/bgr/runs`.

Large checkpoint weights are intentionally not checked into git. The remote
checkpoint directory is:

```text
/work/joy/bgr/runs/openvla_oft_finetune_random_balanced2048_step10_v1/openvla-7b+libero_goal_no_noops+b1+lr-0.0005+lora-r8+dropout-0.0--random-balanced2048-step10
```

