#!/usr/bin/env bash
set -euo pipefail

# Convert manifest.jsonl into two RLDS datasets before launching these commands:
#   /work/joy/bgr/openvla_rlds/bgr_boundary_libero_object
#   /work/joy/bgr/openvla_rlds/random_balanced_libero_object
# The manifest contains validated perturbation specs and source artifact pointers,
# but the OpenVLA-OFT trainer consumes RLDS episodes with images, actions, and language.

OPENVLA_OFT_ROOT=/work/joy/external_validation/openvla_oft_smoke_746787/openvla-oft
RLDS_ROOT=/work/joy/bgr/openvla_rlds
RUN_ROOT=/work/joy/bgr/openvla_oft_runs
BASE_CHECKPOINT=openvla/openvla-7b-finetuned-libero-object

python ${OPENVLA_OFT_ROOT}/vla-scripts/finetune.py \
  --vla_path ${BASE_CHECKPOINT} \
  --data_root_dir ${RLDS_ROOT} \
  --dataset_name bgr_boundary_libero_object \
  --run_root_dir ${RUN_ROOT} \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --max_steps 1000 \
  --save_freq 500 \
  --lora_rank 16 \
  --shuffle_buffer_size 2048 \
  --run_id_note bgr_boundary

python ${OPENVLA_OFT_ROOT}/vla-scripts/finetune.py \
  --vla_path ${BASE_CHECKPOINT} \
  --data_root_dir ${RLDS_ROOT} \
  --dataset_name random_balanced_libero_object \
  --run_root_dir ${RUN_ROOT} \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --max_steps 1000 \
  --save_freq 500 \
  --lora_rank 16 \
  --shuffle_buffer_size 2048 \
  --run_id_note random_balanced
