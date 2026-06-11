#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_PROJECT}/.venv-minigrid-dynamic}"
SETUP_REMOTE="${SETUP_REMOTE:-0}"
OUT_PREFIX="${OUT_PREFIX:-lunarlander_continuous_recovery_calibration_12seed_v1}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MEMORY="${MEMORY:-12G}"
CPUS="${CPUS:-2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SBATCH_PARTITION_ARG=""
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_PARTITION_ARG="--partition='${SLURM_PARTITION}'"
fi

rsync -az tools/lunarlander_recovery_calibration.py "${REMOTE_HOST}:${REMOTE_PROJECT}/tools/"
rsync -az src/ "${REMOTE_HOST}:${REMOTE_PROJECT}/src/"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_LOG_ROOT}' '${REMOTE_RUN_ROOT}' '${REMOTE_LOG_ROOT}/sbatch'"

if [[ "${SETUP_REMOTE}" == "1" ]]; then
  ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT}' && \
    if [[ ! -x '${REMOTE_VENV}/bin/python' ]]; then python3 -m venv '${REMOTE_VENV}'; fi && \
    '${REMOTE_VENV}/bin/python' -m pip install --upgrade pip && \
    '${REMOTE_VENV}/bin/python' -m pip install 'numpy==2.2.6' 'gymnasium==1.3.0' 'box2d==2.3.10' 'pygame-ce==2.5.7' 'swig==4.4.1'"
fi

remote_sbatch="${REMOTE_LOG_ROOT}/sbatch/${OUT_PREFIX}_$(date +%Y%m%d_%H%M%S)_$$.sbatch"
ssh "${REMOTE_HOST}" "cat > '${remote_sbatch}'" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_PROJECT}'
echo "[start] \$(date -Is) host=\$(hostname) job=\${SLURM_JOB_ID}"
PYTHONPATH='${REMOTE_PROJECT}/src:${REMOTE_PROJECT}:${REMOTE_PROJECT}/tools' \
  '${REMOTE_VENV}/bin/python' \
  '${REMOTE_PROJECT}/tools/lunarlander_recovery_calibration.py' \
  --out '${REMOTE_RUN_ROOT}/${OUT_PREFIX}_'\${SLURM_JOB_ID} \
  --env-id LunarLanderContinuous-v3 \
  --continuous \
  --seeds 12 \
  --trials 3 \
  --radii 0,0.5,1.0,1.5,2.0,2.5,3.0 \
  --burn-in 80 \
  --horizon 500 \
  ${EXTRA_ARGS}
echo "[done] \$(date -Is) job=\${SLURM_JOB_ID}"
EOF

job_id="$(
  ssh "${REMOTE_HOST}" \
    "cd '${REMOTE_PROJECT}' && sbatch --parsable \
      --job-name='bgr-lunar-cont-calib' \
      ${SBATCH_PARTITION_ARG} \
      --cpus-per-task='${CPUS}' \
      --mem='${MEMORY}' \
      --time='${TIME_LIMIT}' \
      --output='${REMOTE_LOG_ROOT}/%x-%j.out' \
      '${remote_sbatch}'"
)"

printf 'submitted LunarLanderContinuous calibration job: %s\n' "${job_id}"
printf 'remote output: %s/%s_%s\n' "${REMOTE_RUN_ROOT}" "${OUT_PREFIX}" "${job_id}"
printf 'remote log: %s/bgr-lunar-cont-calib-%s.out\n' "${REMOTE_LOG_ROOT}" "${job_id}"
