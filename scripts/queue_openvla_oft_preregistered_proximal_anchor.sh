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

DATA_TAG="${DATA_TAG:-p2048unique_perturbrepeat3_prereg}"
PROXIMAL_ANCHOR_L2="${PROXIMAL_ANCHOR_L2:-1.0}"
PROXIMAL_TAG="${PROXIMAL_TAG:-proxanchor_l2_1em0}"
ADAPT_TAG="${ADAPT_TAG:-cleanmix_${DATA_TAG}_${PROXIMAL_TAG}_step50500_lr5em7_identitylora_imageaug_officialtrainstats_v1}"
PERTURB_TAG="${PERTURB_TAG:-cleanmix_${DATA_TAG}_${PROXIMAL_TAG}_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1}"
ADAPT_ARTIFACT="${ADAPT_ARTIFACT:-openvla_oft_goal_adapt_eval_${ADAPT_TAG}}"
PERTURB_ARTIFACT="${PERTURB_ARTIFACT:-openvla_oft_perturb_eval_${PERTURB_TAG}}"

BGR_DATA_ROOT="${BGR_DATA_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_tfds_libero_goal_bgr_cleanmix_${DATA_TAG}_v1}"
RANDOM_DATA_ROOT="${RANDOM_DATA_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_tfds_libero_goal_random_cleanmix_${DATA_TAG}_v1}"
BGR_RUN_ROOT="${BGR_RUN_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_goal_adapt_bgr_${ADAPT_TAG}}"
RANDOM_RUN_ROOT="${RANDOM_RUN_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_goal_adapt_random_${ADAPT_TAG}}"
BGR_CKPT="${BGR_CKPT:-${BGR_RUN_ROOT}/openvla-7b-oft-finetuned-libero-goal}"
RANDOM_CKPT="${RANDOM_CKPT:-${RANDOM_RUN_ROOT}/openvla-7b-oft-finetuned-libero-goal}"

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
TRAIN_DEPENDENCY="${TRAIN_DEPENDENCY:-}"
ALLOW_IMMEDIATE_PERTURB="${ALLOW_IMMEDIATE_PERTURB:-0}"

usage() {
  cat <<USAGE
Usage: scripts/queue_openvla_oft_preregistered_proximal_anchor.sh [options]

Dry-run by default. This preregisters a learned-policy intervention that changes
the OpenVLA-OFT optimization objective after the negative clean-mix and weighted
perturbation data-curriculum audits:
  - start from the official OpenVLA-OFT LIBERO-Goal checkpoint;
  - use the already fixed BGR-boundary and matched-random weighted perturbation
    TFDS roots from DATA_TAG=${DATA_TAG};
  - keep official stats, identity-LoRA, image augmentation, LR=${LR},
    ADAPT_STEPS=${ADAPT_STEPS}, and fixed 10-task x 10-trial evals;
  - add an explicit proximal L2 penalty with PROXIMAL_ANCHOR_L2=${PROXIMAL_ANCHOR_L2}
    that keeps all trainable parameters near their resumed official-checkpoint
    values while fitting the replay examples.

Options:
  --adapt-only       print or submit only adaptation/merge/clean-eval
  --perturb-only     print or submit only perturbation evals
  --submit-adapt     submit adaptation chain
  --submit-perturb   submit perturbation evals; requires BGR_DEPENDENCY and
                     RANDOM_DEPENDENCY unless ALLOW_IMMEDIATE_PERTURB=1
  -h, --help         show this message

Promotion gate:
  Promote this result only if proximal-anchor BGR beats both proximal-anchor
  matched random and the official checkpoint on the fixed non-identity
  perturbation total by at least 10/400 episodes and at least 0.02 absolute
  success rate, while not trailing clean identity by more than 1/100. A tie,
  one-episode edge, or official/random lead remains an audit.
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
### Preregistered OpenVLA-OFT proximal-anchor adaptation
DATA_TAG=${DATA_TAG}
ADAPT_TAG=${ADAPT_TAG}
PERTURB_TAG=${PERTURB_TAG}
BGR_DATA_ROOT=${BGR_DATA_ROOT}
RANDOM_DATA_ROOT=${RANDOM_DATA_ROOT}
PROXIMAL_ANCHOR_L2=${PROXIMAL_ANCHOR_L2}
Promotion gate: proximal-anchor BGR must beat proximal-anchor random and official by >=10/400 non-identity episodes and >=0.02, with clean identity no worse than -1/100.
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
    PROXIMAL_ANCHOR_L2="${PROXIMAL_ANCHOR_L2}" \
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
    METHODS="bgr,random" \
    EVAL_TASKS="${EVAL_TASKS}" \
    EVAL_TRIALS="${EVAL_TRIALS}" \
    EVAL_SEED="${EVAL_SEED}" \
    EXCLUDE="${EXCLUDE}" \
    GIT_PULL="${GIT_PULL}" \
    TRAIN_DEPENDENCY="${TRAIN_DEPENDENCY}" \
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
    METHODS="official,bgr,random" \
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
