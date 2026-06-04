#!/usr/bin/env bash
set -euo pipefail

SUBMIT=0
TAG="${TAG:-p512}"
PREP_DEPENDENCY="${PREP_DEPENDENCY:-}"
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
DATASET_NAME="${DATASET_NAME:-libero_goal_no_noops}"
SYNC_CODE="${SYNC_CODE:-1}"

BGR_MIX_RENDER="/work/anonymous/bgr/runs/openvla_teacher_oft_bgr_cleanmix_${TAG}_v1"
RANDOM_MIX_RENDER="/work/anonymous/bgr/runs/openvla_teacher_oft_random_cleanmix_${TAG}_v1"
BGR_TFDS_ROOT="/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_${TAG}_v1"
RANDOM_TFDS_ROOT="/work/anonymous/bgr/runs/openvla_oft_tfds_libero_goal_random_cleanmix_${TAG}_v1"

usage() {
  cat <<USAGE
Usage: scripts/queue_openvla_oft_clean_mix_tfds_export.sh [--submit]

Queues a TFDS-export continuation for already rendered clean-mix examples:
  BGR examples:    ${BGR_MIX_RENDER}/examples.jsonl
  Random examples: ${RANDOM_MIX_RENDER}/examples.jsonl
  BGR TFDS:        ${BGR_TFDS_ROOT}
  Random TFDS:     ${RANDOM_TFDS_ROOT}

Set PREP_DEPENDENCY=afterany:<job_id> to wait for an upstream render/prep job.
The job validates each TFDS root and only rebuilds incomplete exports.

Default mode is dry-run. Pass --submit to queue the continuation job.
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
  rsync -a scripts/export_openvla_oft_tfds.py "${REMOTE_HOST}:${REMOTE_PROJECT}/scripts/"
}

script_path="$(mktemp "${TMPDIR:-/tmp}/clean_mix_tfds_export.XXXXXX.sh")"
trap 'rm -f "${script_path}"' EXIT

cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bgr-cleanmix-tfds-${TAG}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
$(exclude_directive)
#SBATCH --output=${REMOTE_LOG_DIR}/%x-%j.out

set -euo pipefail
source ~/.bashrc || true

cd "${REMOTE_PROJECT}"
export PYTHONPATH="${REMOTE_PROJECT}/src:${REMOTE_PROJECT}:${OPENVLA_OFT_ROOT}:${OPENVLA_OFT_SITE}:\${PYTHONPATH:-}"

tfds_complete() {
  local root="\$1"
  local version_dir="\${root}/${DATASET_NAME}/1.0.0"
  [[ -f "\${version_dir}/dataset_info.json" ]] || return 1
  [[ -f "\${version_dir}/features.json" ]] || return 1
  [[ -f "\${version_dir}/bgr_export_summary.json" ]] || return 1
  compgen -G "\${version_dir}/${DATASET_NAME}-train.tfrecord-*" >/dev/null || return 1
}

export_one() {
  local label="\$1"
  local examples="\$2"
  local root="\$3"
  if [[ ! -f "\${examples}" ]]; then
    echo "Missing rendered examples for \${label}: \${examples}" >&2
    exit 1
  fi
  local count
  count="\$(wc -l < "\${examples}")"
  echo "\${label}: rendered_examples=\${count} examples=\${examples}"
  if tfds_complete "\${root}"; then
    echo "\${label}: TFDS root already complete at \${root}; skipping rebuild"
    return
  fi
  echo "\${label}: rebuilding incomplete TFDS root at \${root}"
  rm -rf "\${root}"
  "${OPENVLA_OFT_PY}" scripts/export_openvla_oft_tfds.py \\
    --examples "\${examples}" \\
    --out "\${root}" \\
    --dataset-name "${DATASET_NAME}"
}

echo "Starting clean-mix TFDS export continuation at \$(date -Is)"
export_one bgr "${BGR_MIX_RENDER}/examples.jsonl" "${BGR_TFDS_ROOT}"
export_one random "${RANDOM_MIX_RENDER}/examples.jsonl" "${RANDOM_TFDS_ROOT}"
echo "BGR_DATA_ROOT=${BGR_TFDS_ROOT}"
echo "RANDOM_DATA_ROOT=${RANDOM_TFDS_ROOT}"
echo "Completed clean-mix TFDS export continuation at \$(date -Is)"
EOF

if [[ "${SUBMIT}" -eq 1 ]]; then
  sync_code
  remote_script="/tmp/bgr-cleanmix-tfds-${TAG}.$(date +%s).sh"
  scp -q "${script_path}" "${REMOTE_HOST}:${remote_script}"
  if [[ -n "${PREP_DEPENDENCY}" ]]; then
    job_id="$(ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_LOG_DIR} && sbatch --parsable --dependency=${PREP_DEPENDENCY} ${remote_script}")"
  else
    job_id="$(ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_LOG_DIR} && sbatch --parsable ${remote_script}")"
  fi
  echo "tfds=${job_id}"
  echo "BGR_DATA_ROOT=${BGR_TFDS_ROOT}"
  echo "RANDOM_DATA_ROOT=${RANDOM_TFDS_ROOT}"
else
  sed -n '1,260p' "${script_path}"
  echo "Dry-run only. Re-run with --submit to queue."
fi
