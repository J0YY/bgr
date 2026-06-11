#!/usr/bin/env bash
set -euo pipefail

export REMOTE_HOST="${REMOTE_HOST:-athena}"
export REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/work/joy/bgr/logs}"
export REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/work/joy/bgr/runs}"
export REMOTE_HF_HOME="${REMOTE_HF_HOME:-/work/joy/cache_home/huggingface}"
export OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft}"
export LIBERO_ROOT="${LIBERO_ROOT:-/work/joy/external_validation/openvla_oft_smoke_746850/LIBERO}"

export TAG="${TAG:-occlusion_bottleneck_combo_occ080_shift015_fullgate_v1}"
export EVAL_ARTIFACT="${EVAL_ARTIFACT:-openvla_oft_perturb_eval_${TAG}}"
export OFFICIAL_CKPT="${OFFICIAL_CKPT:-moojink/openvla-7b-oft-finetuned-libero-goal}"
export BGR_CKPT="${BGR_CKPT:-${REMOTE_RUN_ROOT}/openvla_oft_goal_adapt_bgr_cleanmix_p2048unique_occlusion_bottleneck_prereg_proxanchor_l2_5em0_step50400_lr2em7_identitylora_imageaug_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal}"
export RANDOM_CKPT="${RANDOM_CKPT:-${REMOTE_RUN_ROOT}/openvla_oft_goal_adapt_random_cleanmix_p2048unique_occlusion_bottleneck_prereg_proxanchor_l2_5em0_step50400_lr2em7_identitylora_imageaug_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal}"

export METHODS="${METHODS:-official,bgr,random}"
if [[ -z "${PERTURBATIONS:-}" ]]; then
  export PERTURBATIONS='identity={};occlusion_shift={"fraction":0.80,"dx_fraction":0.15,"dy_fraction":0.0}'
else
  export PERTURBATIONS
fi
export EVAL_TASKS="${EVAL_TASKS:-10}"
export EVAL_TRIALS="${EVAL_TRIALS:-40}"
export EVAL_TASK_OFFSET="${EVAL_TASK_OFFSET:-0}"
export EVAL_INIT_STATE_OFFSET="${EVAL_INIT_STATE_OFFSET:-0}"
export EVAL_SEED="${EVAL_SEED:-237}"
export EVAL_TIME="${EVAL_TIME:-12:00:00}"
export PARTITION="${PARTITION:-low-prio-gpu}"
export GRES="${GRES:-gpu:a6000:1}"
export CPUS="${CPUS:-8}"
export MEM="${MEM:-90G}"
export EXCLUDE="${EXCLUDE:-c2-g4-21}"
export SAVE_ROLLOUTS="${SAVE_ROLLOUTS:-0}"

exec scripts/queue_openvla_oft_perturb_eval.sh "$@"
