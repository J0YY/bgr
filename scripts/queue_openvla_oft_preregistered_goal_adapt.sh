#!/usr/bin/env bash
set -euo pipefail

SUBMIT_ADAPT=0
SUBMIT_PERTURB=0
RUN_ADAPT=1
RUN_PERTURB=1

REMOTE_PROJECT="${REMOTE_PROJECT:-/work/anonymous/bgr}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/work/anonymous/bgr/logs}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/work/anonymous/bgr/runs}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/work/anonymous/cache_home/huggingface}"
REMOTE_TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE:-${REMOTE_HF_HOME}/hub}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft}"
LIBERO_ROOT="${LIBERO_ROOT:-/home/anonymous/LIBERO}"
OFFICIAL_STATS="${OFFICIAL_STATS:-${REMOTE_HF_HOME}/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b/dataset_statistics.json}"

ADAPT_TAG="${ADAPT_TAG:-cleanmix_p4096_commonavail_step50500_lr5em7_identitylora_imageaug_officialtrainstats_prereg_v1}"
PERTURB_TAG="${PERTURB_TAG:-cleanmix_p4096_commonavail_step50500_lr5em7_identitylora_imageaug_officialtrainstats_prereg_fullgoal10x10_perturb_v1}"
ADAPT_ARTIFACT="${ADAPT_ARTIFACT:-openvla_oft_goal_adapt_eval_${ADAPT_TAG}}"
PERTURB_ARTIFACT="${PERTURB_ARTIFACT:-openvla_oft_perturb_eval_cleanmix_p4096_commonavail_step50500_lr5em7_identitylora_imageaug_officialtrainstats_prereg_fullgoal10x10_v1}"

BGR_DATA_ROOT="${BGR_DATA_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_tfds_libero_goal_bgr_cleanmix_p4096_commonavail_v1}"
RANDOM_DATA_ROOT="${RANDOM_DATA_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_tfds_libero_goal_random_cleanmix_p4096_commonavail_v1}"
BGR_RUN_ROOT="${BGR_RUN_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_goal_adapt_bgr_${ADAPT_TAG}}"
RANDOM_RUN_ROOT="${RANDOM_RUN_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_goal_adapt_random_${ADAPT_TAG}}"
BGR_CKPT="${BGR_CKPT:-${BGR_RUN_ROOT}/openvla-7b-oft-finetuned-libero-goal}"
RANDOM_CKPT="${RANDOM_CKPT:-${RANDOM_RUN_ROOT}/openvla-7b-oft-finetuned-libero-goal}"

ADAPT_METHODS="${ADAPT_METHODS:-bgr,random}"
PERTURB_METHODS="${PERTURB_METHODS:-official,bgr,random}"
ADAPT_STEPS="${ADAPT_STEPS:-500}"
LR="${LR:-5e-7}"
IMAGE_AUG="${IMAGE_AUG:-True}"
TRAIN_TIME="${TRAIN_TIME:-12:00:00}"
MERGE_TIME="${MERGE_TIME:-02:00:00}"
EVAL_TIME="${EVAL_TIME:-12:00:00}"
EVAL_TASKS="${EVAL_TASKS:-10}"
EVAL_TRIALS="${EVAL_TRIALS:-10}"
EVAL_SEED="${EVAL_SEED:-37}"
EXCLUDE="${EXCLUDE:-c2-g4-21,c2-g4-19}"
GIT_PULL="${GIT_PULL:-1}"
ALLOW_IMMEDIATE_PERTURB="${ALLOW_IMMEDIATE_PERTURB:-0}"

usage() {
  cat <<USAGE
Usage: scripts/queue_openvla_oft_preregistered_goal_adapt.sh [options]

Dry-run by default. This preregisters the next learned-policy OpenVLA/LIBERO
attempt before results are available:
  - common-availability p4096 BGR/random data roots from the completed fair repair
  - official OpenVLA-OFT LIBERO-Goal dataset statistics
  - identity-LoRA entry point, image augmentation, LR=${LR}, ADAPT_STEPS=${ADAPT_STEPS}
  - full LIBERO-Goal 10-task x 10-trial clean and visual-perturbation evals

Options:
  --adapt-only       print or submit only the adaptation/merge/clean-eval chain
  --perturb-only     print or submit only the perturbation eval chain
  --submit-adapt     submit the adaptation chain through queue_openvla_oft_goal_adapt.sh
  --submit-perturb   submit perturbation evals; requires BGR_DEPENDENCY and
                     RANDOM_DEPENDENCY unless ALLOW_IMMEDIATE_PERTURB=1
  -h, --help         show this message

Promotion gate:
  Promote this result only if BGR beats both matched random and official on the
  fixed non-identity perturbation total by at least 10/400 episodes and at least
  0.02 absolute success rate, while not trailing clean identity by more than
  1/100. A tie, one-episode edge, or official/random lead remains an audit.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapt-only) RUN_ADAPT=1; RUN_PERTURB=0; shift ;;
    --perturb-only) RUN_ADAPT=0; RUN_PERTURB=1; shift ;;
    --submit-adapt) SUBMIT_ADAPT=1; shift ;;
    --submit-perturb) SUBMIT_PERTURB=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

print_header() {
  cat <<EOF
### Preregistered OpenVLA-OFT p4096 common-availability continuation
ADAPT_TAG=${ADAPT_TAG}
PERTURB_TAG=${PERTURB_TAG}
BGR_DATA_ROOT=${BGR_DATA_ROOT}
RANDOM_DATA_ROOT=${RANDOM_DATA_ROOT}
OFFICIAL_STATS=${OFFICIAL_STATS}
Promotion gate: BGR must beat random and official by >=10/400 non-identity episodes and >=0.02, with clean identity no worse than -1/100.
EOF
}

run_adapt() {
  local command=(scripts/queue_openvla_oft_goal_adapt.sh)
  if [[ "${SUBMIT_ADAPT}" -eq 1 ]]; then
    command+=(--submit)
  fi

  env \
    REMOTE_PROJECT="${REMOTE_PROJECT}" \
    REMOTE_LOG_DIR="${REMOTE_LOG_DIR}" \
    REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT}" \
    REMOTE_HF_HOME="${REMOTE_HF_HOME}" \
    REMOTE_TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE}" \
    OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT}" \
    LIBERO_ROOT="${LIBERO_ROOT}" \
    TRAIN_DATASET_STATISTICS_SOURCE="${OFFICIAL_STATS}" \
    DATASET_STATISTICS_SOURCE="${OFFICIAL_STATS}" \
    FINETUNE_SCRIPT="vla-scripts/finetune_identity_lora.py" \
    ADAPT_STEPS="${ADAPT_STEPS}" \
    LR="${LR}" \
    IMAGE_AUG="${IMAGE_AUG}" \
    TRAIN_TIME="${TRAIN_TIME}" \
    MERGE_TIME="${MERGE_TIME}" \
    EVAL_TIME="${EVAL_TIME}" \
    TAG="${ADAPT_TAG}" \
    EVAL_ARTIFACT="${ADAPT_ARTIFACT}" \
    BGR_DATA_ROOT="${BGR_DATA_ROOT}" \
    RANDOM_DATA_ROOT="${RANDOM_DATA_ROOT}" \
    BGR_RUN_ROOT="${BGR_RUN_ROOT}" \
    RANDOM_RUN_ROOT="${RANDOM_RUN_ROOT}" \
    METHODS="${ADAPT_METHODS}" \
    EVAL_TASKS="${EVAL_TASKS}" \
    EVAL_TRIALS="${EVAL_TRIALS}" \
    EVAL_SEED="${EVAL_SEED}" \
    EXCLUDE="${EXCLUDE}" \
    GIT_PULL="${GIT_PULL}" \
    "${command[@]}"
}

run_perturb() {
  if [[ "${SUBMIT_PERTURB}" -eq 1 && "${ALLOW_IMMEDIATE_PERTURB}" != "1" ]]; then
    if [[ -z "${BGR_DEPENDENCY:-}" || -z "${RANDOM_DEPENDENCY:-}" ]]; then
      echo "Refusing --submit-perturb without BGR_DEPENDENCY and RANDOM_DEPENDENCY; set ALLOW_IMMEDIATE_PERTURB=1 if checkpoints already exist." >&2
      exit 2
    fi
  fi

  local command=(scripts/queue_openvla_oft_perturb_eval.sh)
  if [[ "${SUBMIT_PERTURB}" -eq 1 ]]; then
    command+=(--submit)
  fi

  env \
    REMOTE_LOG_DIR="${REMOTE_LOG_DIR}" \
    REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT}" \
    REMOTE_HF_HOME="${REMOTE_HF_HOME}" \
    REMOTE_TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE}" \
    OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT}" \
    LIBERO_ROOT="${LIBERO_ROOT}" \
    TAG="${PERTURB_TAG}" \
    EVAL_ARTIFACT="${PERTURB_ARTIFACT}" \
    BGR_CKPT="${BGR_CKPT}" \
    RANDOM_CKPT="${RANDOM_CKPT}" \
    METHODS="${PERTURB_METHODS}" \
    EVAL_TASKS="${EVAL_TASKS}" \
    EVAL_TRIALS="${EVAL_TRIALS}" \
    EVAL_SEED="${EVAL_SEED}" \
    EVAL_TIME="${EVAL_TIME}" \
    EXCLUDE="${EXCLUDE}" \
    BGR_DEPENDENCY="${BGR_DEPENDENCY:-}" \
    RANDOM_DEPENDENCY="${RANDOM_DEPENDENCY:-}" \
    "${command[@]}"
}

print_header
if [[ "${RUN_ADAPT}" -eq 1 ]]; then
  run_adapt
fi
if [[ "${RUN_PERTURB}" -eq 1 ]]; then
  run_perturb
fi
