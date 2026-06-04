#!/usr/bin/env bash
set -euo pipefail

SUBMIT=0
RESUME_STEP="${RESUME_STEP:-50000}"
ADAPT_STEPS="${ADAPT_STEPS:-100}"
MAX_STEPS="${MAX_STEPS:-$((RESUME_STEP + ADAPT_STEPS))}"
SAVE_FREQ="${SAVE_FREQ:-${ADAPT_STEPS}}"
LR="${LR:-1e-5}"
LORA_RANK="${LORA_RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
IMAGE_AUG="${IMAGE_AUG:-False}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-2048}"
PARTITION="${PARTITION:-low-prio-gpu}"
GRES="${GRES:-gpu:a6000:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-90G}"
TRAIN_TIME="${TRAIN_TIME:-04:00:00}"
MERGE_TIME="${MERGE_TIME:-02:00:00}"
EVAL_TIME="${EVAL_TIME:-06:00:00}"
EXCLUDE="${EXCLUDE:-c2-g4-21}"
REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/anonymous/bgr}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/work/anonymous/bgr/logs}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/work/anonymous/bgr/runs}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/work/anonymous/cache_home/huggingface}"
REMOTE_TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE:-${REMOTE_HF_HOME}/hub}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft}"
OPENVLA_OFT_TORCHRUN="${OPENVLA_OFT_TORCHRUN:-${OPENVLA_OFT_ROOT}/.venv-oft/bin/torchrun}"
OPENVLA_OFT_PY="${OPENVLA_OFT_PY:-${OPENVLA_OFT_ROOT}/.venv-oft/bin/python}"
OPENVLA_OFT_SITE="${OPENVLA_OFT_SITE:-${OPENVLA_OFT_ROOT}/.venv-oft/lib/python3.10/site-packages}"
FINETUNE_SCRIPT="${FINETUNE_SCRIPT:-vla-scripts/finetune.py}"
TRAIN_DATASET_STATISTICS_SOURCE="${TRAIN_DATASET_STATISTICS_SOURCE:-}"
DATASET_STATISTICS_SOURCE="${DATASET_STATISTICS_SOURCE:-}"
LIBERO_ROOT="${LIBERO_ROOT:-/home/anonymous/LIBERO}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-goal}"
DATASET_NAME="${DATASET_NAME:-libero_goal_no_noops}"
TAG="${TAG:-step${MAX_STEPS}}"
EVAL_ARTIFACT="${EVAL_ARTIFACT:-openvla_oft_goal_adapt_eval_${TAG}_v1}"
BGR_DATA_ROOT="${BGR_DATA_ROOT:-/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_balanced2048_v1}"
RANDOM_DATA_ROOT="${RANDOM_DATA_ROOT:-/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_balanced2048_v1}"
BGR_RUN_ROOT="${BGR_RUN_ROOT:-/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_balanced2048_${TAG}_v1}"
RANDOM_RUN_ROOT="${RANDOM_RUN_ROOT:-/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_balanced2048_${TAG}_v1}"
TRAIN_DEPENDENCY="${TRAIN_DEPENDENCY:-}"
SERIAL_TRAIN="${SERIAL_TRAIN:-1}"
METHODS="${METHODS:-bgr,random}"
EVAL_SUITE="${EVAL_SUITE:-libero_goal}"
EVAL_TASKS="${EVAL_TASKS:-5}"
EVAL_TRIALS="${EVAL_TRIALS:-3}"
EVAL_TASK_OFFSET="${EVAL_TASK_OFFSET:-0}"
EVAL_INIT_STATE_OFFSET="${EVAL_INIT_STATE_OFFSET:-0}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:--1}"
EVAL_SEED="${EVAL_SEED:-7}"
GIT_PULL="${GIT_PULL:-0}"

usage() {
  cat <<USAGE
Usage: scripts/queue_openvla_oft_goal_adapt.sh [--submit]

Queues a matched BGR-boundary vs random-balanced OpenVLA-OFT adaptation smoke:
  - starts from ${BASE_CHECKPOINT}
  - resumes the official L1/proprio heads from RESUME_STEP=${RESUME_STEP}
  - trains ADAPT_STEPS=${ADAPT_STEPS} additional optimizer steps to MAX_STEPS=${MAX_STEPS}
  - merges LoRA and evaluates on LIBERO-Goal with five tasks and three init states

Environment overrides:
  METHODS=bgr|random|bgr,random selects which branches to queue
  SERIAL_TRAIN=1 serializes paired train jobs to avoid shared HF config races
  IMAGE_AUG=True|False forwards OpenVLA-OFT image augmentation to finetune.py
  REMOTE_RUN_ROOT sets the writable root for eval logs
  REMOTE_HF_HOME/REMOTE_TRANSFORMERS_CACHE set writable Hugging Face cache roots
  FINETUNE_SCRIPT selects an alternate OpenVLA-OFT fine-tuning entry point
  TRAIN_DATASET_STATISTICS_SOURCE forces official action/proprio stats during RLDS training normalization
  DATASET_STATISTICS_SOURCE copies known-good action stats into the checkpoint after merge

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
    if [[ -n "${dependency}" ]]; then
      ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_LOG_DIR} && sbatch --parsable --dependency=${dependency} ${remote_script}"
    else
      ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_LOG_DIR} && sbatch --parsable ${remote_script}"
    fi
  else
    echo "### ${name}${dependency:+ dependency=${dependency}}"
    sed -n '1,220p' "${script_path}"
  fi
}

append_afterok_dependency() {
  local base="$1"
  local job_id="$2"
  if [[ -z "${job_id}" ]]; then
    printf '%s\n' "${base}"
  elif [[ -z "${base}" ]]; then
    printf 'afterok:%s\n' "${job_id}"
  elif [[ "${base}" == afterok:* ]]; then
    printf '%s:%s\n' "${base}" "${job_id}"
  else
    printf '%s,afterok:%s\n' "${base}" "${job_id}"
  fi
}

write_train_script() {
  local method="$1"
  local data_root="$2"
  local run_root="$3"
  local script_path="$4"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bgr-goal-adapt-${method}-${TAG}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TRAIN_TIME}
$(exclude_directive)
#SBATCH --output=${REMOTE_LOG_DIR}/%x-%j.out

set -euo pipefail
source ~/.bashrc || true

cd "${REMOTE_PROJECT}"
if [[ "${GIT_PULL}" == "1" ]]; then
  git pull --ff-only
fi

cd "${OPENVLA_OFT_ROOT}"
echo "Adapting official LIBERO-Goal checkpoint with ${method} data on \$(hostname) at \$(date -Is)"

FINETUNE_ENTRYPOINT="${FINETUNE_SCRIPT}"
if [[ -n "${TRAIN_DATASET_STATISTICS_SOURCE}" ]]; then
  mkdir -p "${REMOTE_PROJECT}/runs/openvla_oft_wrappers"
  FINETUNE_ENTRYPOINT="${REMOTE_PROJECT}/runs/openvla_oft_wrappers/finetune_${method}_${TAG}.py"
  cat > "\${FINETUNE_ENTRYPOINT}" <<'PY'
#!/usr/bin/env python
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy

from prismatic.vla.datasets.rlds import dataset as rlds_dataset


_ORIGINAL_MAKE_DATASET_FROM_RLDS = rlds_dataset.make_dataset_from_rlds
_MATERIALIZED_STATS: dict[str, str] = {}


def _stats_path_for_dataset(dataset_name: str) -> str:
    if dataset_name in _MATERIALIZED_STATS:
        return _MATERIALIZED_STATS[dataset_name]

    source_path = Path(os.environ["BGR_TRAIN_DATASET_STATISTICS_SOURCE"])
    raw_stats = json.loads(source_path.read_text())
    if "action" in raw_stats:
        dataset_stats = raw_stats
    elif dataset_name in raw_stats:
        dataset_stats = raw_stats[dataset_name]
    elif len(raw_stats) == 1:
        dataset_stats = next(iter(raw_stats.values()))
    else:
        raise KeyError(f"Cannot find stats for dataset {dataset_name!r} in {source_path}; available keys={sorted(raw_stats)}")

    stats_dir = Path(os.environ.get("BGR_TRAIN_DATASET_STATISTICS_DIR", "/tmp"))
    stats_dir.mkdir(parents=True, exist_ok=True)
    out_path = stats_dir / f"dataset_statistics_{dataset_name}.json"
    out_path.write_text(json.dumps(dataset_stats, indent=2, sort_keys=True) + "\n")
    _MATERIALIZED_STATS[dataset_name] = str(out_path)
    return str(out_path)


def _make_dataset_from_rlds_with_official_stats(*args, **kwargs):
    dataset_name = kwargs.get("name")
    if dataset_name is None and args:
        dataset_name = args[0]
    if dataset_name is not None:
        stats_path = _stats_path_for_dataset(str(dataset_name))
        kwargs["dataset_statistics"] = stats_path
        print(f"[BGR] Forcing RLDS dataset_statistics={stats_path} for dataset={dataset_name}", flush=True)
    return _ORIGINAL_MAKE_DATASET_FROM_RLDS(*args, **kwargs)


rlds_dataset.make_dataset_from_rlds = _make_dataset_from_rlds_with_official_stats
runpy.run_path(os.environ["BGR_ORIGINAL_FINETUNE_SCRIPT"], run_name="__main__")
PY
  chmod +x "\${FINETUNE_ENTRYPOINT}"
  export BGR_ORIGINAL_FINETUNE_SCRIPT="${OPENVLA_OFT_ROOT}/${FINETUNE_SCRIPT}"
  export BGR_TRAIN_DATASET_STATISTICS_SOURCE="${TRAIN_DATASET_STATISTICS_SOURCE}"
  export BGR_TRAIN_DATASET_STATISTICS_DIR="${REMOTE_PROJECT}/runs/openvla_oft_wrappers"
fi

env WANDB_MODE=disabled \\
  HF_HOME="${REMOTE_HF_HOME}" \\
  TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE}" \\
  PYTHONPATH="${OPENVLA_OFT_ROOT}" \\
  "${OPENVLA_OFT_TORCHRUN}" --standalone --nnodes 1 --nproc-per-node 1 "\${FINETUNE_ENTRYPOINT}" \\
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
    --resume True \\
    --resume_step "${RESUME_STEP}" \\
    --image_aug "${IMAGE_AUG}" \\
    --lora_rank "${LORA_RANK}" \\
    --merge_lora_during_training False \\
    --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}" \\
    --wandb_entity disabled \\
    --wandb_project bgr \\
    --run_id_note "${method}-${TAG}" \\
    --wandb_log_freq 10
EOF
}

write_merge_script() {
  local method="$1"
  local checkpoint_dir="$2"
  local script_path="$3"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bgr-goal-merge-${method}-${TAG}
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
echo "Merging ${method} official-goal adaptation checkpoint on \$(hostname) at \$(date -Is)"
env WANDB_MODE=disabled \\
  HF_HOME="${REMOTE_HF_HOME}" \\
  TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE}" \\
  PYTHONPATH="${OPENVLA_OFT_ROOT}" \\
  "${OPENVLA_OFT_PY}" vla-scripts/merge_lora_weights_and_save.py \\
    --base_checkpoint "${BASE_CHECKPOINT}" \\
    --lora_finetuned_checkpoint_dir "${checkpoint_dir}"

if [[ -n "${DATASET_STATISTICS_SOURCE}" ]]; then
  echo "Replacing dataset_statistics.json with ${DATASET_STATISTICS_SOURCE}"
  cp "${DATASET_STATISTICS_SOURCE}" "${checkpoint_dir}/dataset_statistics.json"
fi
EOF
}

write_eval_script() {
  local method="$1"
  local checkpoint_dir="$2"
  local script_path="$3"
  local local_log_dir="${REMOTE_RUN_ROOT}/${EVAL_ARTIFACT}/logs/${method}"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bgr-goal-eval-${method}-${TAG}
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
echo "Evaluating ${method} official-goal adaptation checkpoint on \$(hostname) at \$(date -Is)"
env WANDB_MODE=disabled \\
  HF_HOME="${REMOTE_HF_HOME}" \\
  TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE}" \\
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

tmp_files=()
cleanup() {
  rm -f "${tmp_files[@]:-}"
}
trap cleanup EXIT

queue_method() {
  local method="$1"
  local data_root="$2"
  local run_root="$3"
  local train_dependency="${4:-${TRAIN_DEPENDENCY}}"
  local checkpoint_dir="${run_root}/openvla-7b-oft-finetuned-libero-goal"
  local train_script merge_script eval_script
  train_script="$(mktemp "${TMPDIR:-/tmp}/goal_adapt_train_${method}.XXXXXX.sh")"
  merge_script="$(mktemp "${TMPDIR:-/tmp}/goal_adapt_merge_${method}.XXXXXX.sh")"
  eval_script="$(mktemp "${TMPDIR:-/tmp}/goal_adapt_eval_${method}.XXXXXX.sh")"
  tmp_files+=("${train_script}" "${merge_script}" "${eval_script}")

  write_train_script "${method}" "${data_root}" "${run_root}" "${train_script}"
  write_merge_script "${method}" "${checkpoint_dir}" "${merge_script}"
  write_eval_script "${method}" "${checkpoint_dir}" "${eval_script}"

  if [[ "${SUBMIT}" -eq 1 ]]; then
    train_job="$(submit_script "goal-adapt-train-${method}-${TAG}" "${train_dependency}" "${train_script}")"
    merge_job="$(submit_script "goal-adapt-merge-${method}-${TAG}" "afterok:${train_job}" "${merge_script}")"
    eval_job="$(submit_script "goal-adapt-eval-${method}-${TAG}" "afterok:${merge_job}" "${eval_script}")"
    QUEUED_TRAIN_JOB="${train_job}"
    echo "${method}: train=${train_job} merge=${merge_job} eval=${eval_job}"
  else
    submit_script "goal-adapt-train-${method}-${TAG}" "${train_dependency}" "${train_script}"
    submit_script "goal-adapt-merge-${method}-${TAG}" "afterok:<${method}_train_job>" "${merge_script}"
    submit_script "goal-adapt-eval-${method}-${TAG}" "afterok:<${method}_merge_job>" "${eval_script}"
  fi
}

QUEUED_TRAIN_JOB=""
LAST_TRAIN_JOB=""
IFS=',' read -r -a SELECTED_METHODS <<< "${METHODS}"
for selected_method in "${SELECTED_METHODS[@]}"; do
  selected_method="$(printf '%s' "${selected_method}" | tr -d '[:space:]')"
  train_dependency="${TRAIN_DEPENDENCY}"
  if [[ "${SUBMIT}" -eq 1 && "${SERIAL_TRAIN}" == "1" && -n "${LAST_TRAIN_JOB}" ]]; then
    train_dependency="$(append_afterok_dependency "${train_dependency}" "${LAST_TRAIN_JOB}")"
  fi
  case "${selected_method}" in
    bgr)
      queue_method "bgr" "${BGR_DATA_ROOT}" "${BGR_RUN_ROOT}" "${train_dependency}"
      ;;
    random)
      queue_method "random" "${RANDOM_DATA_ROOT}" "${RANDOM_RUN_ROOT}" "${train_dependency}"
      ;;
    "")
      ;;
    *)
      echo "Unknown METHODS entry: ${selected_method}" >&2
      exit 2
      ;;
  esac
  if [[ "${SUBMIT}" -eq 1 && -n "${QUEUED_TRAIN_JOB}" ]]; then
    LAST_TRAIN_JOB="${QUEUED_TRAIN_JOB}"
  fi
done

if [[ "${SUBMIT}" -eq 0 ]]; then
  echo "Dry-run only. Re-run with --submit to queue."
fi
