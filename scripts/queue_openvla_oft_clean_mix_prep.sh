#!/usr/bin/env bash
set -euo pipefail

SUBMIT=0
TAG="${TAG:-p256}"
MAX_CLEAN_EXAMPLES="${MAX_CLEAN_EXAMPLES:-2048}"
MAX_PERTURB_EXAMPLES="${MAX_PERTURB_EXAMPLES:-256}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-64}"
CLEAN_EPISODES_PER_FAMILY="${CLEAN_EPISODES_PER_FAMILY:-64}"
PERTURB_EPISODES_PER_FAMILY="${PERTURB_EPISODES_PER_FAMILY:-1}"
PARTITION="${PARTITION:-low-prio-gpu}"
GRES="${GRES:-gpu:a6000:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-90G}"
TIME="${TIME:-08:00:00}"
EXCLUDE="${EXCLUDE:-c2-g4-21}"
REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/anonymous/bgr}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/work/anonymous/bgr/logs}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft}"
OPENVLA_OFT_PY="${OPENVLA_OFT_PY:-${OPENVLA_OFT_ROOT}/.venv-oft/bin/python}"
OPENVLA_OFT_SITE="${OPENVLA_OFT_SITE:-${OPENVLA_OFT_ROOT}/.venv-oft/lib/python3.10/site-packages}"
LIBERO_ROOT="${LIBERO_ROOT:-/home/anonymous/LIBERO}"
DATASET_NAME="${DATASET_NAME:-libero_goal_no_noops}"
SYNC_CODE="${SYNC_CODE:-1}"

BGR_CLEAN_RENDER="/work/anonymous/bgr/runs/openvla_teacher_oft_bgr_clean_${TAG}_v1"
RANDOM_CLEAN_RENDER="/work/anonymous/bgr/runs/openvla_teacher_oft_random_clean_${TAG}_v1"
BGR_PERTURB_RENDER="/work/anonymous/bgr/runs/openvla_teacher_oft_bgr_perturb_${TAG}_v1"
RANDOM_PERTURB_RENDER="/work/anonymous/bgr/runs/openvla_teacher_oft_random_perturb_${TAG}_v1"
BGR_MIX_RENDER="/work/anonymous/bgr/runs/openvla_teacher_oft_bgr_cleanmix_${TAG}_v1"
RANDOM_MIX_RENDER="/work/anonymous/bgr/runs/openvla_teacher_oft_random_cleanmix_${TAG}_v1"
BGR_TFDS_ROOT="/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_${TAG}_v1"
RANDOM_TFDS_ROOT="/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_cleanmix_${TAG}_v1"
CLEAN_MANIFEST_DIR="/work/anonymous/bgr/runs/openvla_teacher_replay_manifest_clean_anchors_${TAG}_v1"
PERTURB_MANIFEST="/work/anonymous/bgr/results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl"

usage() {
  cat <<USAGE
Usage: scripts/queue_openvla_oft_clean_mix_prep.sh [--submit]

Prepares clean-mix OpenVLA-OFT TFDS datasets:
  BGR:    ${BGR_TFDS_ROOT}
  Random: ${RANDOM_TFDS_ROOT}

Default mode is dry-run. Pass --submit to queue the preparation job.
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

sync_code() {
  if [[ "${SYNC_CODE}" != "1" ]]; then
    return
  fi
  rsync -a \
    scripts/export_openvla_teacher_replay_manifest.py \
    scripts/render_openvla_teacher_examples.py \
    scripts/export_openvla_oft_tfds.py \
    scripts/pack_openvla_oft_examples.py \
    scripts/combine_openvla_oft_examples.py \
    "${REMOTE_HOST}:${REMOTE_PROJECT}/scripts/"
}

script_path="$(mktemp "${TMPDIR:-/tmp}/clean_mix_prep.XXXXXX.sh")"
trap 'rm -f "${script_path}"' EXIT

cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bgr-cleanmix-prep-${TAG}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
$(exclude_directive)
#SBATCH --output=${REMOTE_LOG_DIR}/%x-%j.out

set -euo pipefail
source ~/.bashrc || true

BGR_SOURCE_DIRS=(
  /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed1_lp2_h160
  /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed2_lp2_h160
  /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed3_lp2_h160
  /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed4_lp2_h160
  /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_proposal_balanced_expfit_seed5_lp2_h160
)
RANDOM_SOURCE_DIRS=(
  /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed1b_skip_lp2_h160
  /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed2b_skip_lp2_h160
  /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed3b_skip_lp2_h160
  /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed4b_skip_lp2_h160
  /work/anonymous/dreamaudit_jobs/artifacts/libero_openvla_observation_random_balanced_seed5b_skip_lp2_h160
)

cd "${REMOTE_PROJECT}"
mkdir -p "${CLEAN_MANIFEST_DIR}"

BGR_DIR_ARGS=()
for dir in "\${BGR_SOURCE_DIRS[@]}"; do
  BGR_DIR_ARGS+=(--bgr-dir "\${dir}")
done
RANDOM_DIR_ARGS=()
for dir in "\${RANDOM_SOURCE_DIRS[@]}"; do
  RANDOM_DIR_ARGS+=(--random-dir "\${dir}")
done

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${REMOTE_PROJECT}/src:${REMOTE_PROJECT}:${LIBERO_ROOT}:${OPENVLA_OFT_ROOT}:${OPENVLA_OFT_SITE}:\${PYTHONPATH:-}"

echo "Exporting clean-anchor manifest at \$(date -Is)"
"${OPENVLA_OFT_PY}" scripts/export_openvla_teacher_replay_manifest.py \\
  "\${BGR_DIR_ARGS[@]}" \\
  "\${RANDOM_DIR_ARGS[@]}" \\
  --native-anchors-only \\
  --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}" \\
  --out "${CLEAN_MANIFEST_DIR}"

echo "Rendering clean BGR anchors at \$(date -Is)"
"${OPENVLA_OFT_PY}" scripts/render_openvla_teacher_examples.py \\
  --manifest "${CLEAN_MANIFEST_DIR}/teacher_replay_manifest.jsonl" \\
  --out "${BGR_CLEAN_RENDER}" \\
  --method bgr_boundary \\
  --max-examples "${MAX_CLEAN_EXAMPLES}" \\
  --selection balanced_episodes \\
  --episodes-per-family "${CLEAN_EPISODES_PER_FAMILY}" \\
  --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}"

echo "Rendering clean random anchors at \$(date -Is)"
"${OPENVLA_OFT_PY}" scripts/render_openvla_teacher_examples.py \\
  --manifest "${CLEAN_MANIFEST_DIR}/teacher_replay_manifest.jsonl" \\
  --out "${RANDOM_CLEAN_RENDER}" \\
  --method random_balanced \\
  --max-examples "${MAX_CLEAN_EXAMPLES}" \\
  --selection balanced_episodes \\
  --episodes-per-family "${CLEAN_EPISODES_PER_FAMILY}" \\
  --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}"

echo "Rendering BGR perturbation subset at \$(date -Is)"
"${OPENVLA_OFT_PY}" scripts/render_openvla_teacher_examples.py \\
  --manifest "${PERTURB_MANIFEST}" \\
  --out "${BGR_PERTURB_RENDER}" \\
  --method bgr_boundary \\
  --max-examples "${MAX_PERTURB_EXAMPLES}" \\
  --selection balanced_episodes \\
  --episodes-per-family "${PERTURB_EPISODES_PER_FAMILY}" \\
  --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}"

echo "Rendering random perturbation subset at \$(date -Is)"
"${OPENVLA_OFT_PY}" scripts/render_openvla_teacher_examples.py \\
  --manifest "${PERTURB_MANIFEST}" \\
  --out "${RANDOM_PERTURB_RENDER}" \\
  --method random_balanced \\
  --max-examples "${MAX_PERTURB_EXAMPLES}" \\
  --selection balanced_episodes \\
  --episodes-per-family "${PERTURB_EPISODES_PER_FAMILY}" \\
  --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}"

echo "Combining rendered examples at \$(date -Is)"
"${OPENVLA_OFT_PY}" scripts/combine_openvla_oft_examples.py \\
  --source clean="${BGR_CLEAN_RENDER}" \\
  --source perturb="${BGR_PERTURB_RENDER}" \\
  --out "${BGR_MIX_RENDER}"
"${OPENVLA_OFT_PY}" scripts/combine_openvla_oft_examples.py \\
  --source clean="${RANDOM_CLEAN_RENDER}" \\
  --source perturb="${RANDOM_PERTURB_RENDER}" \\
  --out "${RANDOM_MIX_RENDER}"

echo "Exporting TFDS datasets at \$(date -Is)"
"${OPENVLA_OFT_PY}" scripts/export_openvla_oft_tfds.py \\
  --examples "${BGR_MIX_RENDER}/examples.jsonl" \\
  --out "${BGR_TFDS_ROOT}" \\
  --dataset-name "${DATASET_NAME}"
"${OPENVLA_OFT_PY}" scripts/export_openvla_oft_tfds.py \\
  --examples "${RANDOM_MIX_RENDER}/examples.jsonl" \\
  --out "${RANDOM_TFDS_ROOT}" \\
  --dataset-name "${DATASET_NAME}"

echo "BGR_DATA_ROOT=${BGR_TFDS_ROOT}"
echo "RANDOM_DATA_ROOT=${RANDOM_TFDS_ROOT}"
echo "Completed clean-mix prep at \$(date -Is)"
EOF

if [[ "${SUBMIT}" -eq 1 ]]; then
  sync_code
  remote_script="/tmp/bgr-cleanmix-prep-${TAG}.$(date +%s).sh"
  scp -q "${script_path}" "${REMOTE_HOST}:${remote_script}"
  job_id="$(ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_LOG_DIR} && sbatch --parsable ${remote_script}")"
  echo "prep=${job_id}"
  echo "BGR_DATA_ROOT=${BGR_TFDS_ROOT}"
  echo "RANDOM_DATA_ROOT=${RANDOM_TFDS_ROOT}"
else
  sed -n '1,260p' "${script_path}"
  echo "Dry-run only. Re-run with --submit to queue."
fi
