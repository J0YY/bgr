#!/usr/bin/env bash
set -euo pipefail

SUBMIT=0
BGR_TRAIN_JOB_ID="${BGR_TRAIN_JOB_ID:-}"
RANDOM_TRAIN_JOB_ID="${RANDOM_TRAIN_JOB_ID:-}"
TRAIN_DEPENDENCY_TYPE="${TRAIN_DEPENDENCY_TYPE:-afterok}"
TAG="${TAG:-step1000}"
PARTITION="${PARTITION:-low-prio-gpu}"
GRES="${GRES:-gpu:a6000:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-90G}"
MERGE_TIME="${MERGE_TIME:-02:00:00}"
EVAL_TIME="${EVAL_TIME:-06:00:00}"
EXCLUDE="${EXCLUDE:-c2-g4-21}"
REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/work/anonymous/bgr/logs}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft}"
OPENVLA_OFT_PY="${OPENVLA_OFT_PY:-${OPENVLA_OFT_ROOT}/.venv-oft/bin/python}"
OPENVLA_OFT_SITE="${OPENVLA_OFT_SITE:-${OPENVLA_OFT_ROOT}/.venv-oft/lib/python3.10/site-packages}"
LIBERO_ROOT="${LIBERO_ROOT:-/home/anonymous/LIBERO}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-openvla/openvla-7b}"
EVAL_ARTIFACT="${EVAL_ARTIFACT:-openvla_oft_eval_balanced2048_${TAG}_v1}"
EVAL_SUITE="${EVAL_SUITE:-libero_goal}"
EVAL_TASKS="${EVAL_TASKS:-5}"
EVAL_TRIALS="${EVAL_TRIALS:-3}"
EVAL_TASK_OFFSET="${EVAL_TASK_OFFSET:-0}"
EVAL_INIT_STATE_OFFSET="${EVAL_INIT_STATE_OFFSET:-0}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:--1}"
EVAL_SEED="${EVAL_SEED:-7}"
LORA_RANK="${LORA_RANK:-8}"

usage() {
  cat <<USAGE
Usage: BGR_TRAIN_JOB_ID=<id> RANDOM_TRAIN_JOB_ID=<id> scripts/queue_openvla_oft_eval.sh [--submit]

Queues dependent OpenVLA-OFT post-training jobs:
  1. merge BGR LoRA after BGR training succeeds
  2. merge random-balanced LoRA after random training succeeds
  3. evaluate each merged checkpoint after its merge succeeds

Default mode is dry-run. Pass --submit to queue with sbatch.

Environment overrides include TAG=${TAG}, TRAIN_DEPENDENCY_TYPE=${TRAIN_DEPENDENCY_TYPE},
EVAL_TASKS=${EVAL_TASKS}, EVAL_TRIALS=${EVAL_TRIALS}, EVAL_MAX_STEPS=${EVAL_MAX_STEPS},
EXCLUDE=${EXCLUDE}.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${BGR_TRAIN_JOB_ID}" || -z "${RANDOM_TRAIN_JOB_ID}" ]]; then
  echo "BGR_TRAIN_JOB_ID and RANDOM_TRAIN_JOB_ID are required." >&2
  usage
  exit 2
fi
case "${TRAIN_DEPENDENCY_TYPE}" in
  afterok|afterany) ;;
  *) echo "TRAIN_DEPENDENCY_TYPE must be afterok or afterany." >&2; exit 2 ;;
esac

exclude_directive() {
  if [[ -n "${EXCLUDE}" ]]; then
    printf '#SBATCH --exclude=%s\n' "${EXCLUDE}"
  fi
}

submit_script() {
  local name="$1"
  local dependency="$2"
  local script_path="$3"
  local remote_script="/tmp/${name}.$(date +%s).sh"

  if [[ "${SUBMIT}" -eq 1 ]]; then
    scp -q "${script_path}" "${REMOTE_HOST}:${remote_script}"
    ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_LOG_DIR} && sbatch --parsable --dependency=${dependency} ${remote_script}"
  else
    echo "### ${name} dependency=${dependency}"
    sed -n '1,220p' "${script_path}"
  fi
}

write_merge_script() {
  local method="$1"
  local checkpoint_dir="$2"
  local script_path="$3"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bgr-merge-${method}-${TAG}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${MERGE_TIME}
$(exclude_directive)
#SBATCH --output=${REMOTE_LOG_DIR}/%x-%j.out

set -euo pipefail
source ~/.bashrc || true
cd "${OPENVLA_OFT_ROOT}"
echo "Merging ${method} checkpoint on \$(hostname) at \$(date -Is)"
env WANDB_MODE=disabled \\
  HF_HOME=/work/anonymous/cache_home/huggingface \\
  TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub \\
  PYTHONPATH="${OPENVLA_OFT_ROOT}" \\
  "${OPENVLA_OFT_PY}" vla-scripts/merge_lora_weights_and_save.py \\
    --base_checkpoint "${BASE_CHECKPOINT}" \\
    --lora_finetuned_checkpoint_dir "${checkpoint_dir}"
EOF
}

write_eval_script() {
  local method="$1"
  local checkpoint_dir="$2"
  local script_path="$3"
  local local_log_dir="/work/anonymous/bgr/runs/${EVAL_ARTIFACT}/logs/${method}"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bgr-eval-${method}-${TAG}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${EVAL_TIME}
$(exclude_directive)
#SBATCH --output=${REMOTE_LOG_DIR}/%x-%j.out

set -euo pipefail
source ~/.bashrc || true
mkdir -p "${local_log_dir}"
cd "${OPENVLA_OFT_ROOT}"
echo "Evaluating ${method} checkpoint on \$(hostname) at \$(date -Is)"
env WANDB_MODE=disabled \\
  HF_HOME=/work/anonymous/cache_home/huggingface \\
  TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub \\
  MUJOCO_GL=egl \\
  PYOPENGL_PLATFORM=egl \\
  PYTHONPATH="${OPENVLA_OFT_ROOT}:${LIBERO_ROOT}:${OPENVLA_OFT_SITE}" \\
  "${OPENVLA_OFT_PY}" experiments/robot/libero/run_libero_eval.py \\
    --model_family openvla \\
    --pretrained_checkpoint "${checkpoint_dir}" \\
    --use_l1_regression True \\
    --use_diffusion False \\
    --use_film False \\
    --num_images_in_input 2 \\
    --use_proprio True \\
    --lora_rank "${LORA_RANK}" \\
    --task_suite_name "${EVAL_SUITE}" \\
    --num_tasks "${EVAL_TASKS}" \\
    --task_offset "${EVAL_TASK_OFFSET}" \\
    --num_trials_per_task "${EVAL_TRIALS}" \\
    --init_state_offset "${EVAL_INIT_STATE_OFFSET}" \\
    --max_steps_override "${EVAL_MAX_STEPS}" \\
    --num_steps_wait 10 \\
    --env_img_res 256 \\
    --seed "${EVAL_SEED}" \\
    --local_log_dir "${local_log_dir}" \\
    --run_id_note "${method}-${TAG}"
EOF
}

run_dir_for() {
  local method="$1"
  local note="$2"
  printf '/work/anonymous/bgr/runs/openvla_oft_finetune_%s_balanced2048_%s_v1/openvla-7b+libero_goal_no_noops+b1+lr-0.0005+lora-r8+dropout-0.0--%s\n' \
    "${method}" "${TAG}" "${note}"
}

BGR_CKPT="$(run_dir_for bgr "${BGR_NOTE:-bgr-balanced2048-${TAG}}")"
RANDOM_CKPT="$(run_dir_for random "${RANDOM_NOTE:-random-balanced2048-${TAG}}")"

tmp_files=()
cleanup() {
  rm -f "${tmp_files[@]:-}"
}
trap cleanup EXIT

bgr_merge_script="$(mktemp "${TMPDIR:-/tmp}/bgr_merge.XXXXXX.sh")"
random_merge_script="$(mktemp "${TMPDIR:-/tmp}/random_merge.XXXXXX.sh")"
bgr_eval_script="$(mktemp "${TMPDIR:-/tmp}/bgr_eval.XXXXXX.sh")"
random_eval_script="$(mktemp "${TMPDIR:-/tmp}/random_eval.XXXXXX.sh")"
tmp_files+=("${bgr_merge_script}" "${random_merge_script}" "${bgr_eval_script}" "${random_eval_script}")

write_merge_script "bgr" "${BGR_CKPT}" "${bgr_merge_script}"
write_merge_script "random" "${RANDOM_CKPT}" "${random_merge_script}"
write_eval_script "bgr" "${BGR_CKPT}" "${bgr_eval_script}"
write_eval_script "random" "${RANDOM_CKPT}" "${random_eval_script}"

if [[ "${SUBMIT}" -eq 1 ]]; then
  bgr_merge_job="$(submit_script "bgr-merge-${TAG}" "${TRAIN_DEPENDENCY_TYPE}:${BGR_TRAIN_JOB_ID}" "${bgr_merge_script}")"
  random_merge_job="$(submit_script "random-merge-${TAG}" "${TRAIN_DEPENDENCY_TYPE}:${RANDOM_TRAIN_JOB_ID}" "${random_merge_script}")"
  bgr_eval_job="$(submit_script "bgr-eval-${TAG}" "afterok:${bgr_merge_job}" "${bgr_eval_script}")"
  random_eval_job="$(submit_script "random-eval-${TAG}" "afterok:${random_merge_job}" "${random_eval_script}")"
  echo "Queued merge/eval jobs:"
  echo "  bgr_merge=${bgr_merge_job}"
  echo "  random_merge=${random_merge_job}"
  echo "  bgr_eval=${bgr_eval_job}"
  echo "  random_eval=${random_eval_job}"
else
  submit_script "bgr-merge-${TAG}" "${TRAIN_DEPENDENCY_TYPE}:${BGR_TRAIN_JOB_ID}" "${bgr_merge_script}"
  submit_script "random-merge-${TAG}" "${TRAIN_DEPENDENCY_TYPE}:${RANDOM_TRAIN_JOB_ID}" "${random_merge_script}"
  submit_script "bgr-eval-${TAG}" "afterok:<bgr_merge_job>" "${bgr_eval_script}"
  submit_script "random-eval-${TAG}" "afterok:<random_merge_job>" "${random_eval_script}"
  echo "Dry-run only. Re-run with --submit to queue."
fi
