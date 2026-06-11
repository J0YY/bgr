#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_PROJECT}/.venv-fetchpush}"
OUT_PREFIX="${OUT_PREFIX:-halfcheetah_recovery_calibration_12seed_v1}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MEMORY="${MEMORY:-12G}"
CPUS="${CPUS:-2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SBATCH_PARTITION_ARG=""
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_PARTITION_ARG="--partition='${SLURM_PARTITION}'"
fi

rsync -az tools/halfcheetah_recovery_calibration.py "${REMOTE_HOST}:${REMOTE_PROJECT}/tools/"
rsync -az src/ "${REMOTE_HOST}:${REMOTE_PROJECT}/src/"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_LOG_ROOT}' '${REMOTE_RUN_ROOT}' '${REMOTE_LOG_ROOT}/sbatch'"

remote_sbatch="${REMOTE_LOG_ROOT}/sbatch/${OUT_PREFIX}_$(date +%Y%m%d_%H%M%S)_$$.sbatch"
ssh "${REMOTE_HOST}" "cat > '${remote_sbatch}'" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_PROJECT}'
echo "[start] \$(date -Is) host=\$(hostname) job=\${SLURM_JOB_ID}"
PYTHONPATH='${REMOTE_PROJECT}/src:${REMOTE_PROJECT}' \
  '${REMOTE_VENV}/bin/python' \
  '${REMOTE_PROJECT}/tools/halfcheetah_recovery_calibration.py' \
  --out '${REMOTE_RUN_ROOT}/${OUT_PREFIX}_'\${SLURM_JOB_ID} \
  ${EXTRA_ARGS}
echo "[done] \$(date -Is) job=\${SLURM_JOB_ID}"
EOF

job_id="$(
  ssh "${REMOTE_HOST}" \
    "cd '${REMOTE_PROJECT}' && sbatch --parsable \
      --job-name='bgr-halfcheetah-calib' \
      ${SBATCH_PARTITION_ARG} \
      --cpus-per-task='${CPUS}' \
      --mem='${MEMORY}' \
      --time='${TIME_LIMIT}' \
      --output='${REMOTE_LOG_ROOT}/%x-%j.out' \
      '${remote_sbatch}'"
)"

printf 'submitted HalfCheetah calibration job: %s\n' "${job_id}"
printf 'remote output: %s/%s_%s\n' "${REMOTE_RUN_ROOT}" "${OUT_PREFIX}" "${job_id}"
printf 'remote log: %s/bgr-halfcheetah-calib-%s.out\n' "${REMOTE_LOG_ROOT}" "${job_id}"
