#!/usr/bin/env bash
set -euo pipefail

SUBMIT=0
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_FREQ="${SAVE_FREQ:-500}"
LR="${LR:-5e-4}"
LORA_RANK="${LORA_RANK:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-2048}"
PARTITION="${PARTITION:-low-prio-gpu}"
GRES="${GRES:-gpu:a6000:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-80G}"
TIME="${TIME:-08:00:00}"
EXCLUDE="${EXCLUDE:-c2-g4-21}"
GIT_PULL="${GIT_PULL:-0}"
GIT_SSH_PREFLIGHT="${GIT_SSH_PREFLIGHT:-}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/anonymous/bgr}"
REMOTE_HOST="${REMOTE_HOST:-athena}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft}"
OPENVLA_OFT_PY="${OPENVLA_OFT_PY:-${OPENVLA_OFT_ROOT}/.venv-oft/bin/torchrun}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-openvla/openvla-7b}"
DATASET_NAME="${DATASET_NAME:-libero_goal_no_noops}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/work/anonymous/bgr/logs}"

usage() {
  cat <<USAGE
Usage: scripts/queue_openvla_oft_long_run.sh [--submit]

Queues or prints the next matched OpenVLA-OFT fine-tuning pair:
  - BGR-boundary balanced 2048
  - random-balanced 2048

Environment overrides:
  MAX_STEPS=${MAX_STEPS}
  SAVE_FREQ=${SAVE_FREQ}
  LR=${LR}
  LORA_RANK=${LORA_RANK}
  BATCH_SIZE=${BATCH_SIZE}
  SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE}
  PARTITION=${PARTITION}
  GRES=${GRES}
  EXCLUDE=${EXCLUDE}
  GIT_PULL=${GIT_PULL}
  GIT_SSH_PREFLIGHT=${GIT_SSH_PREFLIGHT}
  TIME=${TIME}

Default mode is dry-run. Pass --submit to queue asynchronous Slurm jobs with sbatch.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

run_one() {
  local method="$1"
  local data_root="$2"
  local note="$3"
  local run_root="$4"
  local job_name="bgr-oft-${method}-${TAG}"
  local local_script
  local remote_script
  local exclude_line=""
  local_script="$(mktemp "${TMPDIR:-/tmp}/bgr_oft_${method}.XXXXXX.sh")"
  remote_script="/tmp/${job_name}.$(date +%s).sh"
  if [[ -n "${EXCLUDE}" ]]; then
    exclude_line="#SBATCH --exclude=${EXCLUDE}"
  fi

  cat > "${local_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
${exclude_line}
#SBATCH --output=${REMOTE_LOG_DIR}/%x-%j.out

set -euo pipefail
source ~/.bashrc || true

if [[ -n "${GIT_SSH_PREFLIGHT}" ]]; then
  echo "[preflight] git ssh test..."
  ssh -o BatchMode=yes -o ConnectTimeout=8 -T "${GIT_SSH_PREFLIGHT}" || true
fi

cd "${REMOTE_PROJECT}"
if [[ "${GIT_PULL}" == "1" ]]; then
  echo "[preflight] git pull --ff-only in ${REMOTE_PROJECT}"
  git pull --ff-only
else
  echo "[preflight] skipping git pull; using existing remote workspace"
fi

echo "================================="
echo "Running on: \$(hostname)"
echo "Working directory: ${OPENVLA_OFT_ROOT}"
echo "Method: ${method}"
echo "Run root: ${run_root}"
echo "Started at: \$(date -Is)"
echo "================================="

cd "${OPENVLA_OFT_ROOT}"
env WANDB_MODE=disabled \\
  HF_HOME=/work/anonymous/cache_home/huggingface \\
  TRANSFORMERS_CACHE=/work/anonymous/cache_home/huggingface/hub \\
  PYTHONPATH="${OPENVLA_OFT_ROOT}" \\
  "${OPENVLA_OFT_PY}" --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \\
    --vla_path "${BASE_CHECKPOINT}" \\
    --data_root_dir "${data_root}" \\
    --dataset_name "${DATASET_NAME}" \\
    --run_root_dir "${run_root}" \\
    --use_l1_regression True \\
    --use_diffusion False \\
    --use_film False \\
    --num_images_in_input 2 \\
    --use_proprio True \\
    --batch_size "${BATCH_SIZE}" \\
    --learning_rate "${LR}" \\
    --num_steps_before_decay 100000 \\
    --max_steps "${MAX_STEPS}" \\
    --save_freq "${SAVE_FREQ}" \\
    --save_latest_checkpoint_only True \\
    --image_aug False \\
    --lora_rank "${LORA_RANK}" \\
    --merge_lora_during_training False \\
    --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}" \\
    --wandb_entity disabled \\
    --wandb_project bgr \\
    --run_id_note "${note}" \\
    --wandb_log_freq 10
EOF

  if [[ "${SUBMIT}" -eq 1 ]]; then
    scp -q "${local_script}" "${REMOTE_HOST}:${remote_script}"
    ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_LOG_DIR} && sbatch ${remote_script}"
  else
    echo "### ${method}: ${job_name}"
    sed -n '1,160p' "${local_script}"
  fi

  rm -f "${local_script}"
  echo "[${method}] run root: ${run_root}"
}

TAG="step${MAX_STEPS}"
run_one \
  "bgr_boundary" \
  "/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_balanced2048_v1" \
  "bgr-balanced2048-${TAG}" \
  "/work/anonymous/bgr/runs/openvla_oft_finetune_bgr_balanced2048_${TAG}_v1"

run_one \
  "random_balanced" \
  "/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_balanced2048_v1" \
  "random-balanced2048-${TAG}" \
  "/work/anonymous/bgr/runs/openvla_oft_finetune_random_balanced2048_${TAG}_v1"

if [[ "${SUBMIT}" -eq 1 ]]; then
  echo "Submitted matched OpenVLA-OFT jobs for ${MAX_STEPS} steps."
else
  echo "Dry-run only. Re-run with --submit to queue."
fi
