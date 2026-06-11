#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_PROJECT}/.venv-bipedalwalker}"
OUT_PREFIX="${OUT_PREFIX:-bipedalwalker_recovery_calibration_12seed_v1}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MEMORY="${MEMORY:-16G}"
CPUS="${CPUS:-2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SBATCH_PARTITION_ARG=""
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_PARTITION_ARG="--partition='${SLURM_PARTITION}'"
fi

rsync -az tools/bipedalwalker_recovery_calibration.py "${REMOTE_HOST}:${REMOTE_PROJECT}/tools/"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_LOG_ROOT}' '${REMOTE_RUN_ROOT}' '${REMOTE_LOG_ROOT}/sbatch'"

ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT}' && \
  if [[ ! -x '${REMOTE_VENV}/bin/python' ]]; then python3 -m venv '${REMOTE_VENV}'; fi && \
  '${REMOTE_VENV}/bin/python' -m pip install --upgrade pip && \
  '${REMOTE_VENV}/bin/python' -m pip install 'numpy==2.2.6' 'gymnasium==1.3.0' 'box2d==2.3.10' 'pygame-ce==2.5.7' 'swig==4.4.1'"

remote_sbatch="${REMOTE_LOG_ROOT}/sbatch/${OUT_PREFIX}_$(date +%Y%m%d_%H%M%S)_$$.sbatch"
ssh "${REMOTE_HOST}" "cat > '${remote_sbatch}'" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_PROJECT}'
echo "[start] \$(date -Is) host=\$(hostname) job=\${SLURM_JOB_ID}"
PYTHONPATH='${REMOTE_PROJECT}/src:${REMOTE_PROJECT}:${REMOTE_PROJECT}/tools' \
  '${REMOTE_VENV}/bin/python' \
  '${REMOTE_PROJECT}/tools/bipedalwalker_recovery_calibration.py' \
  --out '${REMOTE_RUN_ROOT}/${OUT_PREFIX}_'\${SLURM_JOB_ID} \
  ${EXTRA_ARGS}
echo "[done] \$(date -Is) job=\${SLURM_JOB_ID}"
EOF

job_id="$(
  ssh "${REMOTE_HOST}" \
    "cd '${REMOTE_PROJECT}' && sbatch --parsable \
      --job-name='bgr-bipedal-calib' \
      ${SBATCH_PARTITION_ARG} \
      --cpus-per-task='${CPUS}' \
      --mem='${MEMORY}' \
      --time='${TIME_LIMIT}' \
      --output='${REMOTE_LOG_ROOT}/%x-%j.out' \
      '${remote_sbatch}'"
)"

printf 'submitted BipedalWalker calibration job: %s\n' "${job_id}"
printf 'remote output: %s/%s_%s\n' "${REMOTE_RUN_ROOT}" "${OUT_PREFIX}" "${job_id}"
printf 'remote log: %s/bgr-bipedal-calib-%s.out\n' "${REMOTE_LOG_ROOT}" "${job_id}"
