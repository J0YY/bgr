#!/usr/bin/env bash
set -euo pipefail

SUBMIT_PREP=0
SUBMIT_ADAPT=0
SUBMIT_PERTURB=0
RUN_PREP=1
RUN_ADAPT=1
RUN_PERTURB=1

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/anonymous/bgr}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/work/anonymous/bgr/logs}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/work/anonymous/bgr/runs}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/work/anonymous/cache_home/huggingface}"
REMOTE_TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE:-${REMOTE_HF_HOME}/hub}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft}"
OPENVLA_OFT_PY="${OPENVLA_OFT_PY:-${OPENVLA_OFT_ROOT}/.venv-oft/bin/python}"
OPENVLA_OFT_SITE="${OPENVLA_OFT_SITE:-${OPENVLA_OFT_ROOT}/.venv-oft/lib/python3.10/site-packages}"
LIBERO_ROOT="${LIBERO_ROOT:-/home/anonymous/LIBERO}"
OFFICIAL_STATS="${OFFICIAL_STATS:-${REMOTE_HF_HOME}/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b/dataset_statistics.json}"
PERTURB_MANIFEST="${PERTURB_MANIFEST:-${REMOTE_PROJECT}/results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl}"

PREP_TAG="${PREP_TAG:-p2048unique_perturbonly_anchor_prereg}"
ANCHOR_TAG="${ANCHOR_TAG:-perturbonly_proxanchor_l2_5em0}"
ADAPT_TAG="${ADAPT_TAG:-${PREP_TAG}_${ANCHOR_TAG}_step50300_lr2em7_identitylora_imageaug_officialtrainstats_v1}"
PERTURB_TAG="${PERTURB_TAG:-${PREP_TAG}_${ANCHOR_TAG}_step50300_lr2em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1}"
ADAPT_ARTIFACT="${ADAPT_ARTIFACT:-openvla_oft_goal_adapt_eval_${ADAPT_TAG}}"
PERTURB_ARTIFACT="${PERTURB_ARTIFACT:-openvla_oft_perturb_eval_${PERTURB_TAG}}"

BGR_RENDER_ROOT="${BGR_RENDER_ROOT:-${REMOTE_RUN_ROOT}/openvla_teacher_oft_bgr_perturbonly_${PREP_TAG}_v1}"
RANDOM_RENDER_ROOT="${RANDOM_RENDER_ROOT:-${REMOTE_RUN_ROOT}/openvla_teacher_oft_random_perturbonly_${PREP_TAG}_v1}"
BGR_DATA_ROOT="${BGR_DATA_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_tfds_libero_goal_bgr_perturbonly_${PREP_TAG}_v1}"
RANDOM_DATA_ROOT="${RANDOM_DATA_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_tfds_libero_goal_random_perturbonly_${PREP_TAG}_v1}"
BGR_RUN_ROOT="${BGR_RUN_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_goal_adapt_bgr_${ADAPT_TAG}}"
RANDOM_RUN_ROOT="${RANDOM_RUN_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_goal_adapt_random_${ADAPT_TAG}}"
BGR_CKPT="${BGR_CKPT:-${BGR_RUN_ROOT}/openvla-7b-oft-finetuned-libero-goal}"
RANDOM_CKPT="${RANDOM_CKPT:-${RANDOM_RUN_ROOT}/openvla-7b-oft-finetuned-libero-goal}"

MAX_PERTURB_EXAMPLES="${MAX_PERTURB_EXAMPLES:-2048}"
PERTURB_EPISODES_PER_FAMILY="${PERTURB_EPISODES_PER_FAMILY:-8}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-64}"
DATASET_NAME="${DATASET_NAME:-libero_goal_no_noops}"
PROXIMAL_ANCHOR_L2="${PROXIMAL_ANCHOR_L2:-5.0}"
ADAPT_STEPS="${ADAPT_STEPS:-300}"
LR="${LR:-2e-7}"
IMAGE_AUG="${IMAGE_AUG:-True}"
PREP_TIME="${PREP_TIME:-12:00:00}"
TRAIN_TIME="${TRAIN_TIME:-10:00:00}"
MERGE_TIME="${MERGE_TIME:-02:00:00}"
EVAL_TIME="${EVAL_TIME:-12:00:00}"
EVAL_TASKS="${EVAL_TASKS:-10}"
EVAL_TRIALS="${EVAL_TRIALS:-10}"
EVAL_SEED="${EVAL_SEED:-37}"
PARTITION="${PARTITION:-low-prio-gpu}"
GRES="${GRES:-gpu:a6000:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-90G}"
EXCLUDE="${EXCLUDE:-c2-g4-21,c2-g4-19}"
SYNC_CODE="${SYNC_CODE:-1}"
GIT_PULL="${GIT_PULL:-1}"
TRAIN_DEPENDENCY="${TRAIN_DEPENDENCY:-}"
ALLOW_IMMEDIATE_PERTURB="${ALLOW_IMMEDIATE_PERTURB:-0}"

usage() {
  cat <<USAGE
Usage: scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh [options]

Dry-run by default. This preregisters a learned-policy intervention that is
different from the completed clean-mix, weighted clean-mix, and proximal
clean-mix audits:
  - train only on rendered boundary-band perturbation examples, with no clean
    anchor episodes mixed into the RLDS data;
  - keep BGR-boundary and matched-random perturbation data roots matched by the
    same family budget (${PERTURB_EPISODES_PER_FAMILY} episodes/family);
  - add a stronger official-checkpoint proximal L2 anchor
    PROXIMAL_ANCHOR_L2=${PROXIMAL_ANCHOR_L2} to protect clean identity behavior;
  - use identity-LoRA, image augmentation, official dataset statistics,
    LR=${LR}, ADAPT_STEPS=${ADAPT_STEPS}, and the fixed 10-task x 10-trial
    LIBERO-Goal perturbation evaluation.

Options:
  --prep-only        print or submit only perturb-only TFDS prep
  --adapt-only       print or submit only adaptation/merge/clean-eval
  --perturb-only     print or submit only perturbation evals
  --submit-prep      submit perturb-only TFDS prep
  --submit-adapt     submit adaptation chain; uses TRAIN_DEPENDENCY if set
  --submit-perturb   submit perturbation evals; requires BGR_DEPENDENCY and
                     RANDOM_DEPENDENCY unless ALLOW_IMMEDIATE_PERTURB=1
  -h, --help         show this message

Promotion gate:
  Promote this result only if perturb-only anchored BGR beats both perturb-only
  anchored matched random and the official checkpoint on the fixed non-identity
  perturbation total by at least 10/400 episodes and at least 0.02 absolute
  success rate, while not trailing clean identity by more than 1/100. Anything
  weaker remains an audit.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prep-only) RUN_PREP=1; RUN_ADAPT=0; RUN_PERTURB=0; shift ;;
    --adapt-only) RUN_PREP=0; RUN_ADAPT=1; RUN_PERTURB=0; shift ;;
    --perturb-only) RUN_PREP=0; RUN_ADAPT=0; RUN_PERTURB=1; shift ;;
    --submit-prep) SUBMIT_PREP=1; shift ;;
    --submit-adapt) SUBMIT_ADAPT=1; shift ;;
    --submit-perturb) SUBMIT_PERTURB=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

exclude_directive() {
  if [[ -n "${EXCLUDE}" ]]; then
    printf '#SBATCH --exclude=%s\n' "${EXCLUDE}"
  fi
}

sync_code() {
  if [[ "${SYNC_CODE}" != "1" ]]; then
    return
  fi
  rsync -a \
    scripts/render_openvla_teacher_examples.py \
    scripts/export_openvla_oft_tfds.py \
    scripts/pack_openvla_oft_examples.py \
    "${REMOTE_HOST}:${REMOTE_PROJECT}/scripts/"
}

print_header() {
  cat <<EOF
### Preregistered OpenVLA-OFT perturb-only anchored adaptation
PREP_TAG=${PREP_TAG}
ADAPT_TAG=${ADAPT_TAG}
PERTURB_TAG=${PERTURB_TAG}
BGR_DATA_ROOT=${BGR_DATA_ROOT}
RANDOM_DATA_ROOT=${RANDOM_DATA_ROOT}
PROXIMAL_ANCHOR_L2=${PROXIMAL_ANCHOR_L2}
Promotion gate: perturb-only anchored BGR must beat perturb-only anchored random and official by >=10/400 non-identity episodes and >=0.02, with clean identity no worse than -1/100.
EOF
}

submit_or_print_prep() {
  local script_path="$1"
  if [[ "${SUBMIT_PREP}" -eq 1 ]]; then
    sync_code
    local remote_script="/tmp/bgr-perturbonly-prep-${PREP_TAG}.$(date +%s).sh"
    scp -q "${script_path}" "${REMOTE_HOST}:${remote_script}"
    local job_id
    job_id="$(ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_LOG_DIR} && sbatch --parsable ${remote_script}")"
    echo "prep=${job_id}"
    echo "BGR_DATA_ROOT=${BGR_DATA_ROOT}"
    echo "RANDOM_DATA_ROOT=${RANDOM_DATA_ROOT}"
  else
    sed -n '1,260p' "${script_path}"
    echo "Dry-run only. Re-run with --submit-prep to queue."
  fi
}

run_prep() {
  local script_path
  script_path="$(mktemp "${TMPDIR:-/tmp}/perturbonly_prep.XXXXXX.sh")"
  trap 'rm -f "${script_path}"' RETURN

  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bgr-perturbonly-prep-${PREP_TAG}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${PREP_TIME}
$(exclude_directive)
#SBATCH --output=${REMOTE_LOG_DIR}/%x-%j.out

set -euo pipefail
source ~/.bashrc || true

cd "${REMOTE_PROJECT}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${REMOTE_PROJECT}/src:${REMOTE_PROJECT}:${LIBERO_ROOT}:${OPENVLA_OFT_ROOT}:${OPENVLA_OFT_SITE}:\${PYTHONPATH:-}"

echo "Rendering BGR perturb-only boundary examples at \$(date -Is)"
"${OPENVLA_OFT_PY}" scripts/render_openvla_teacher_examples.py \\
  --manifest "${PERTURB_MANIFEST}" \\
  --out "${BGR_RENDER_ROOT}" \\
  --method bgr_boundary \\
  --max-examples "${MAX_PERTURB_EXAMPLES}" \\
  --selection balanced_episodes \\
  --episodes-per-family "${PERTURB_EPISODES_PER_FAMILY}" \\
  --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}"

echo "Rendering random perturb-only boundary examples at \$(date -Is)"
"${OPENVLA_OFT_PY}" scripts/render_openvla_teacher_examples.py \\
  --manifest "${PERTURB_MANIFEST}" \\
  --out "${RANDOM_RENDER_ROOT}" \\
  --method random_balanced \\
  --max-examples "${MAX_PERTURB_EXAMPLES}" \\
  --selection balanced_episodes \\
  --episodes-per-family "${PERTURB_EPISODES_PER_FAMILY}" \\
  --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}"

echo "Exporting perturb-only TFDS datasets at \$(date -Is)"
"${OPENVLA_OFT_PY}" scripts/export_openvla_oft_tfds.py \\
  --examples "${BGR_RENDER_ROOT}/examples.jsonl" \\
  --out "${BGR_DATA_ROOT}" \\
  --dataset-name "${DATASET_NAME}"
"${OPENVLA_OFT_PY}" scripts/export_openvla_oft_tfds.py \\
  --examples "${RANDOM_RENDER_ROOT}/examples.jsonl" \\
  --out "${RANDOM_DATA_ROOT}" \\
  --dataset-name "${DATASET_NAME}"

echo "BGR_DATA_ROOT=${BGR_DATA_ROOT}"
echo "RANDOM_DATA_ROOT=${RANDOM_DATA_ROOT}"
echo "Completed perturb-only prep at \$(date -Is)"
EOF

  submit_or_print_prep "${script_path}"
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
if [[ "${RUN_PREP}" -eq 1 ]]; then
  run_prep
fi
if [[ "${RUN_ADAPT}" -eq 1 ]]; then
  run_adapt
fi
if [[ "${RUN_PERTURB}" -eq 1 ]]; then
  run_perturb
fi
