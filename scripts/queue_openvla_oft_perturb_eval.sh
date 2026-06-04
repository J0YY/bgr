#!/usr/bin/env bash
set -euo pipefail

SUBMIT=0
TAG="${TAG:-cleanmix_p256_step50100_lr1em6_identitylora_officialstats}"
EVAL_ARTIFACT="${EVAL_ARTIFACT:-openvla_oft_perturb_eval_${TAG}_v1}"
PARTITION="${PARTITION:-low-prio-gpu}"
GRES="${GRES:-gpu:a6000:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-90G}"
EVAL_TIME="${EVAL_TIME:-06:00:00}"
EXCLUDE="${EXCLUDE:-c2-g4-21}"
REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/work/anonymous/bgr/logs}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/work/anonymous/bgr/runs}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/work/anonymous/cache_home/huggingface}"
REMOTE_TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE:-${REMOTE_HF_HOME}/hub}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft}"
OPENVLA_OFT_PY="${OPENVLA_OFT_PY:-${OPENVLA_OFT_ROOT}/.venv-oft/bin/python}"
OPENVLA_OFT_SITE="${OPENVLA_OFT_SITE:-${OPENVLA_OFT_ROOT}/.venv-oft/lib/python3.10/site-packages}"
PATCHED_EVAL_SCRIPT="${PATCHED_EVAL_SCRIPT:-experiments/robot/libero/run_libero_eval_perturb.py}"
LIBERO_ROOT="${LIBERO_ROOT:-/home/anonymous/LIBERO}"
OFFICIAL_CKPT="${OFFICIAL_CKPT:-moojink/openvla-7b-oft-finetuned-libero-goal}"
BGR_CKPT="${BGR_CKPT:-/work/anonymous/bgr/runs/openvla_oft_goal_adapt_bgr_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/openvla-7b-oft-finetuned-libero-goal}"
RANDOM_CKPT="${RANDOM_CKPT:-/work/anonymous/bgr/runs/openvla_oft_goal_adapt_random_cleanmix_p256_step50100_lr1em6_identitylora_officialstats_v1/openvla-7b-oft-finetuned-libero-goal}"
METHODS="${METHODS:-official,bgr,random}"
OFFICIAL_DEPENDENCY="${OFFICIAL_DEPENDENCY:-}"
BGR_DEPENDENCY="${BGR_DEPENDENCY:-}"
RANDOM_DEPENDENCY="${RANDOM_DEPENDENCY:-}"
SERIAL_PERTURB_PER_METHOD="${SERIAL_PERTURB_PER_METHOD:-1}"
if [[ -z "${PERTURBATIONS:-}" ]]; then
  PERTURBATIONS='identity={};blur={"radius":2.5};brightness={"factor":0.5};occlusion={"fraction":0.5};shift={"dx_fraction":0.15,"dy_fraction":0.0}'
fi
EVAL_SUITE="${EVAL_SUITE:-libero_goal}"
EVAL_TASKS="${EVAL_TASKS:-5}"
EVAL_TRIALS="${EVAL_TRIALS:-3}"
EVAL_TASK_OFFSET="${EVAL_TASK_OFFSET:-0}"
EVAL_INIT_STATE_OFFSET="${EVAL_INIT_STATE_OFFSET:-0}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:--1}"
EVAL_SEED="${EVAL_SEED:-7}"
LORA_RANK="${LORA_RANK:-32}"

usage() {
  cat <<USAGE
Usage: scripts/queue_openvla_oft_perturb_eval.sh [--submit]

Queues perturbation-conditioned LIBERO-Goal evals for official, BGR, and random
OpenVLA-OFT checkpoints. The patched eval perturbs only the primary camera image
before policy preprocessing; wrist images and simulator state stay unchanged.

Environment overrides:
  METHODS=official,bgr,random
  PERTURBATIONS='identity={};blur={"radius":2.5};occlusion={"fraction":0.5}'
  OFFICIAL_CKPT=${OFFICIAL_CKPT}
  BGR_CKPT=${BGR_CKPT}
  RANDOM_CKPT=${RANDOM_CKPT}
  REMOTE_RUN_ROOT=${REMOTE_RUN_ROOT}
  REMOTE_HF_HOME=${REMOTE_HF_HOME}
  REMOTE_TRANSFORMERS_CACHE=${REMOTE_TRANSFORMERS_CACHE}
  OFFICIAL_DEPENDENCY/BGR_DEPENDENCY/RANDOM_DEPENDENCY optional sbatch dependencies
  SERIAL_PERTURB_PER_METHOD=1 serializes perturbations per checkpoint to avoid
    shared checkpoint config mutation races during OpenVLA-OFT eval startup

Default mode is dry-run. Pass --submit to queue asynchronous Slurm jobs.
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

ensure_remote_perturb_eval() {
  if [[ "${SUBMIT}" -ne 1 ]]; then
    echo "Would create patched eval script ${OPENVLA_OFT_ROOT}/${PATCHED_EVAL_SCRIPT}"
    return
  fi
  ssh "${REMOTE_HOST}" "cd ${OPENVLA_OFT_ROOT} && PATCHED_EVAL_SCRIPT='${PATCHED_EVAL_SCRIPT}' ${OPENVLA_OFT_PY} - <<'PY'
from pathlib import Path
import os

src_path = Path('experiments/robot/libero/run_libero_eval.py')
dst_path = Path(os.environ['PATCHED_EVAL_SCRIPT'])
src = src_path.read_text()
src = src.replace('import tqdm\\n', 'import tqdm\\nfrom PIL import Image, ImageEnhance, ImageFilter\\n')
helper = r'''

def _apply_primary_perturbation(image):
    perturbation_type = os.environ.get('BGR_EVAL_PERTURBATION_TYPE', 'identity')
    raw_params = os.environ.get('BGR_EVAL_PERTURBATION_PARAMS', '{}')
    try:
        params = json.loads(raw_params)
    except json.JSONDecodeError:
        params = {}
    if perturbation_type in {'', 'identity', 'none'}:
        return image
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert('RGB')
    if perturbation_type == 'blur':
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(params.get('radius', 0.0))))
        return np.asarray(pil, dtype=np.uint8)
    if perturbation_type == 'brightness':
        pil = ImageEnhance.Brightness(pil).enhance(float(params.get('factor', 1.0)))
        return np.asarray(pil, dtype=np.uint8)
    if perturbation_type == 'occlusion':
        arr = np.asarray(pil, dtype=np.uint8).copy()
        fraction = max(0.0, min(1.0, float(params.get('fraction', 0.0))))
        side = max(1, int(round(min(arr.shape[0], arr.shape[1]) * fraction)))
        y0 = (arr.shape[0] - side) // 2
        x0 = (arr.shape[1] - side) // 2
        arr[y0 : y0 + side, x0 : x0 + side, :] = 0
        return arr
    if perturbation_type == 'shift':
        arr = np.asarray(pil, dtype=np.uint8)
        dx = int(round(float(params.get('dx_fraction', 0.0)) * arr.shape[1]))
        dy = int(round(float(params.get('dy_fraction', 0.0)) * arr.shape[0]))
        shifted = np.zeros_like(arr)
        src_x0 = max(0, -dx)
        src_y0 = max(0, -dy)
        dst_x0 = max(0, dx)
        dst_y0 = max(0, dy)
        width = arr.shape[1] - abs(dx)
        height = arr.shape[0] - abs(dy)
        if width > 0 and height > 0:
            shifted[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = arr[src_y0 : src_y0 + height, src_x0 : src_x0 + width]
        return shifted
    raise ValueError(f'Unsupported BGR_EVAL_PERTURBATION_TYPE={perturbation_type!r}')
'''
src = src.replace('\\ndef prepare_observation(obs, resize_size):', helper + '\\ndef prepare_observation(obs, resize_size):')
src = src.replace('    img = get_libero_image(obs)\\n    wrist_img = get_libero_wrist_image(obs)', '    img = _apply_primary_perturbation(get_libero_image(obs))\\n    wrist_img = get_libero_wrist_image(obs)')
dst_path.write_text(src)
print(f'Wrote {dst_path}')
PY"
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

checkpoint_for_method() {
  case "$1" in
    official) printf '%s\n' "${OFFICIAL_CKPT}" ;;
    bgr) printf '%s\n' "${BGR_CKPT}" ;;
    random) printf '%s\n' "${RANDOM_CKPT}" ;;
    *) echo "Unknown method: $1" >&2; exit 2 ;;
  esac
}

dependency_for_method() {
  case "$1" in
    official) printf '%s\n' "${OFFICIAL_DEPENDENCY}" ;;
    bgr) printf '%s\n' "${BGR_DEPENDENCY}" ;;
    random) printf '%s\n' "${RANDOM_DEPENDENCY}" ;;
    *) echo "Unknown method: $1" >&2; exit 2 ;;
  esac
}

write_eval_script() {
  local method="$1"
  local checkpoint="$2"
  local perturbation_name="$3"
  local perturbation_type="$4"
  local perturbation_params="$5"
  local script_path="$6"
  local local_log_dir="${REMOTE_RUN_ROOT}/${EVAL_ARTIFACT}/logs/${method}/${perturbation_name}"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bgr-pert-eval-${method}-${perturbation_name}-${TAG}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${EVAL_TIME}
$(exclude_directive)
#SBATCH --output=${REMOTE_LOG_DIR}/%x-%j.out

set -euo pipefail
source ~/.bashrc || true
mkdir -p "${local_log_dir}" "${local_log_dir}/rollouts"
cd "${OPENVLA_OFT_ROOT}"
echo "Routing rollout videos to ${local_log_dir}/rollouts"
"${OPENVLA_OFT_PY}" - <<'PY'
from pathlib import Path

path = Path("experiments/robot/libero/libero_utils.py")
src = path.read_text()
old = '    rollout_dir = f"./rollouts/{DATE}"'
new = '    rollout_dir = os.environ.get("BGR_EVAL_ROLLOUT_DIR", f"./rollouts/{DATE}")'
if old in src:
    path.write_text(src.replace(old, new))
elif new not in src:
    raise RuntimeError("Could not patch LIBERO rollout directory")
PY
echo "Evaluating ${method} perturbation=${perturbation_name} type=${perturbation_type} params=${perturbation_params} on \$(hostname) at \$(date -Is)"
env WANDB_MODE=disabled \\
  HF_HOME="${REMOTE_HF_HOME}" \\
  TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE}" \\
  MUJOCO_GL=egl \\
  PYOPENGL_PLATFORM=egl \\
  BGR_EVAL_ROLLOUT_DIR="${local_log_dir}/rollouts" \\
  BGR_EVAL_PERTURBATION_TYPE='${perturbation_type}' \\
  BGR_EVAL_PERTURBATION_PARAMS='${perturbation_params}' \\
  PYTHONPATH="${OPENVLA_OFT_ROOT}:${LIBERO_ROOT}:${OPENVLA_OFT_SITE}" \\
  "${OPENVLA_OFT_PY}" "${PATCHED_EVAL_SCRIPT}" \\
    --model_family openvla \\
    --pretrained_checkpoint "${checkpoint}" \\
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
    --run_id_note "${method}-${perturbation_name}-${TAG}"
EOF
}

ensure_remote_perturb_eval

tmp_files=()
cleanup() {
  rm -f "${tmp_files[@]:-}"
}
trap cleanup EXIT

IFS=',' read -r -a SELECTED_METHODS <<< "${METHODS}"
IFS=';' read -r -a SELECTED_PERTURBATIONS <<< "${PERTURBATIONS}"
for method in "${SELECTED_METHODS[@]}"; do
  method="$(printf '%s' "${method}" | tr -d '[:space:]')"
  [[ -z "${method}" ]] && continue
  checkpoint="$(checkpoint_for_method "${method}")"
  method_dependency="$(dependency_for_method "${method}")"
  last_perturb_job=""
  for spec in "${SELECTED_PERTURBATIONS[@]}"; do
    [[ -z "${spec}" ]] && continue
    dependency="${method_dependency}"
    if [[ "${SERIAL_PERTURB_PER_METHOD}" == "1" && -n "${last_perturb_job}" ]]; then
      dependency="$(append_afterok_dependency "${dependency}" "${last_perturb_job}")"
    fi
    perturbation_name="${spec%%=*}"
    perturbation_params="${spec#*=}"
    perturbation_type="${perturbation_name}"
    if [[ "${perturbation_name}" == "identity" ]]; then
      perturbation_type="identity"
    fi
    eval_script="$(mktemp "${TMPDIR:-/tmp}/perturb_eval_${method}_${perturbation_name}.XXXXXX.sh")"
    tmp_files+=("${eval_script}")
    write_eval_script "${method}" "${checkpoint}" "${perturbation_name}" "${perturbation_type}" "${perturbation_params}" "${eval_script}"
    if [[ "${SUBMIT}" -eq 1 ]]; then
      job_id="$(submit_script "perturb-eval-${method}-${perturbation_name}-${TAG}" "${dependency}" "${eval_script}")"
      echo "${method}/${perturbation_name}: ${job_id}"
      last_perturb_job="${job_id}"
    else
      submit_script "perturb-eval-${method}-${perturbation_name}-${TAG}" "${dependency}" "${eval_script}"
      echo "${method}/${perturbation_name}: dry-run"
      last_perturb_job="<${method}_${perturbation_name}_job>"
    fi
  done
done

if [[ "${SUBMIT}" -eq 0 ]]; then
  echo "Dry-run only. Re-run with --submit to queue."
fi
