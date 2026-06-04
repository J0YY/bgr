#!/usr/bin/env bash
set -euo pipefail

SUBMIT=0
TAG="${TAG:-sanity_v1}"
PARTITION="${PARTITION:-low-prio-gpu}"
GRES="${GRES:-gpu:a6000:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-90G}"
TIME="${TIME:-06:00:00}"
EXCLUDE="${EXCLUDE:-c2-g4-21}"
REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/work/anonymous/bgr/logs}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft}"
OPENVLA_OFT_PY="${OPENVLA_OFT_PY:-${OPENVLA_OFT_ROOT}/.venv-oft/bin/python}"
OPENVLA_OFT_SITE="${OPENVLA_OFT_SITE:-${OPENVLA_OFT_ROOT}/.venv-oft/lib/python3.10/site-packages}"
LIBERO_ROOT="${LIBERO_ROOT:-/home/anonymous/LIBERO}"
EVAL_ARTIFACT="${EVAL_ARTIFACT:-openvla_oft_sanity_eval_${TAG}}"
EVAL_SUITE="${EVAL_SUITE:-libero_goal}"
EVAL_TASKS="${EVAL_TASKS:-5}"
EVAL_TRIALS="${EVAL_TRIALS:-3}"
EVAL_TASK_OFFSET="${EVAL_TASK_OFFSET:-0}"
EVAL_INIT_STATE_OFFSET="${EVAL_INIT_STATE_OFFSET:-0}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:--1}"
EVAL_SEED="${EVAL_SEED:-7}"
METHODS="${METHODS:-base,oft-goal}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/work/anonymous/cache_home/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f}"
OFT_GOAL_CHECKPOINT="${OFT_GOAL_CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-goal}"

usage() {
  cat <<USAGE
Usage: scripts/queue_openvla_sanity_eval.sh [--submit]

Queues or prints same-protocol LIBERO sanity evals:
  1. base OpenVLA with native discrete action decoding
  2. official OpenVLA-OFT LIBERO-Goal checkpoint with L1/proprio heads

Default mode is dry-run. Pass --submit to queue asynchronous Slurm jobs with sbatch.

Environment overrides include TAG=${TAG}, EVAL_TASKS=${EVAL_TASKS},
EVAL_TRIALS=${EVAL_TRIALS}, EVAL_MAX_STEPS=${EVAL_MAX_STEPS}, METHODS=${METHODS},
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

exclude_directive() {
  if [[ -n "${EXCLUDE}" ]]; then
    printf '#SBATCH --exclude=%s\n' "${EXCLUDE}"
  fi
}

write_eval_script() {
  local method="$1"
  local checkpoint="$2"
  local use_l1="$3"
  local use_proprio="$4"
  local num_images="$5"
  local lora_rank="$6"
  local script_path="$7"
  local local_log_dir="/work/anonymous/bgr/runs/${EVAL_ARTIFACT}/logs/${method}"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bgr-sanity-${method}-${TAG}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
$(exclude_directive)
#SBATCH --output=${REMOTE_LOG_DIR}/%x-%j.out

set -euo pipefail
source ~/.bashrc || true
mkdir -p "${local_log_dir}"
cd "${OPENVLA_OFT_ROOT}"
echo "Evaluating ${method} checkpoint ${checkpoint} on \$(hostname) at \$(date -Is)"
env WANDB_MODE=disabled \\
  HF_HOME=/work/anonymous/cache_home/huggingface \\
  TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub \\
  MUJOCO_GL=egl \\
  PYOPENGL_PLATFORM=egl \\
  PYTHONPATH="${OPENVLA_OFT_ROOT}:${LIBERO_ROOT}:${OPENVLA_OFT_SITE}" \\
  "${OPENVLA_OFT_PY}" experiments/robot/libero/run_libero_eval.py \\
    --model_family openvla \\
    --pretrained_checkpoint "${checkpoint}" \\
    --use_l1_regression "${use_l1}" \\
    --use_diffusion False \\
    --use_film False \\
    --num_images_in_input "${num_images}" \\
    --use_proprio "${use_proprio}" \\
    --lora_rank "${lora_rank}" \\
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

submit_script() {
  local name="$1"
  local script_path="$2"
  local remote_script="/tmp/${name}.$(date +%s).sh"

  if [[ "${SUBMIT}" -eq 1 ]]; then
    scp -q "${script_path}" "${REMOTE_HOST}:${remote_script}"
    ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_LOG_DIR} && sbatch --parsable ${remote_script}"
  else
    echo "### ${name}"
    sed -n '1,220p' "${script_path}"
  fi
}

has_method() {
  local needle="$1"
  case ",${METHODS}," in
    *",${needle},"*) return 0 ;;
    *) return 1 ;;
  esac
}

tmp_files=()
cleanup() {
  rm -f "${tmp_files[@]:-}"
}
trap cleanup EXIT

base_script=""
oft_goal_script=""
if has_method "base"; then
  base_script="$(mktemp "${TMPDIR:-/tmp}/openvla_base_eval.XXXXXX.sh")"
  tmp_files+=("${base_script}")
  write_eval_script "base" "${BASE_CHECKPOINT}" "False" "False" "1" "8" "${base_script}"
fi
if has_method "oft-goal"; then
  oft_goal_script="$(mktemp "${TMPDIR:-/tmp}/openvla_oft_goal_eval.XXXXXX.sh")"
  tmp_files+=("${oft_goal_script}")
  write_eval_script "oft-goal" "${OFT_GOAL_CHECKPOINT}" "True" "True" "2" "32" "${oft_goal_script}"
fi
if [[ -z "${base_script}" && -z "${oft_goal_script}" ]]; then
  echo "No methods selected. Set METHODS to a comma-separated subset of base,oft-goal." >&2
  exit 2
fi

if [[ "${SUBMIT}" -eq 1 ]]; then
  base_job=""
  oft_goal_job=""
  if [[ -n "${base_script}" ]]; then
    base_job="$(submit_script "base-sanity-${TAG}" "${base_script}")"
  fi
  if [[ -n "${oft_goal_script}" ]]; then
    oft_goal_job="$(submit_script "oft-goal-sanity-${TAG}" "${oft_goal_script}")"
  fi
  echo "Queued OpenVLA sanity eval jobs:"
  if [[ -n "${base_job}" ]]; then
    echo "  base=${base_job}"
  fi
  if [[ -n "${oft_goal_job}" ]]; then
    echo "  oft_goal=${oft_goal_job}"
  fi
else
  if [[ -n "${base_script}" ]]; then
    submit_script "base-sanity-${TAG}" "${base_script}"
  fi
  if [[ -n "${oft_goal_script}" ]]; then
    submit_script "oft-goal-sanity-${TAG}" "${oft_goal_script}"
  fi
  echo "Dry-run only. Re-run with --submit to queue."
fi
