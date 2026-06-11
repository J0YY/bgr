#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_PROJECT}/.venv-gymnasium-classic}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
MEMORY="${MEMORY:-12G}"
CPUS="${CPUS:-2}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-acrobot_package_recovery_probe_4seed_v1}"
SEEDS="${SEEDS:-0,1,2,3}"
METHODS="${METHODS:-uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SETUP_REMOTE="${SETUP_REMOTE:-1}"

SBATCH_PARTITION_ARG=""
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_PARTITION_ARG="--partition='${SLURM_PARTITION}'"
fi

rsync -az tools/acrobot_recovery_probe.py "${REMOTE_HOST}:${REMOTE_PROJECT}/tools/acrobot_recovery_probe.py"
rsync -az src/ "${REMOTE_HOST}:${REMOTE_PROJECT}/src/"

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_LOG_ROOT}' '${REMOTE_RUN_ROOT}'"

if [[ "${SETUP_REMOTE}" == "1" ]]; then
  ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT}' && \
    if [[ ! -x '${REMOTE_VENV}/bin/python' ]]; then python3 -m venv '${REMOTE_VENV}'; fi && \
    '${REMOTE_VENV}/bin/python' -m pip install --upgrade pip && \
    '${REMOTE_VENV}/bin/python' -m pip install 'numpy==2.2.6' 'gymnasium==1.3.0'"
fi

job_id="$(
  ssh "${REMOTE_HOST}" \
    "cd '${REMOTE_PROJECT}' && sbatch --parsable \
      --job-name='bgr-acrobot-pkg' \
      ${SBATCH_PARTITION_ARG} \
      --cpus-per-task='${CPUS}' \
      --mem='${MEMORY}' \
      --time='${TIME_LIMIT}' \
      --output='${REMOTE_LOG_ROOT}/%x-%j.out'" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_PROJECT}'
echo "[start] \$(date -Is) host=\$(hostname) job=\${SLURM_JOB_ID}"
'${REMOTE_VENV}/bin/python' - <<'PY'
import gymnasium
import numpy
print('job_acrobot_env_ok', gymnasium.__version__, numpy.__version__)
PY
PYTHONPATH='${REMOTE_PROJECT}/src:${REMOTE_PROJECT}' \
  '${REMOTE_VENV}/bin/python' \
  '${REMOTE_PROJECT}/tools/acrobot_recovery_probe.py' \
  --dynamics-backend gymnasium \
  --env-id Acrobot-v1 \
  --seeds '${SEEDS}' \
  --methods '${METHODS}' \
  --out "${REMOTE_RUN_ROOT}/${ARTIFACT_PREFIX}_\${SLURM_JOB_ID}" \
  ${EXTRA_ARGS}
echo "[done] \$(date -Is) job=\${SLURM_JOB_ID}"
EOF
)"

printf 'submitted Acrobot package-state job: %s\n' "${job_id}"
printf 'remote run: %s/%s_%s\n' "${REMOTE_RUN_ROOT}" "${ARTIFACT_PREFIX}" "${job_id}"
printf 'remote log: %s/bgr-acrobot-pkg-%s.out\n' "${REMOTE_LOG_ROOT}" "${job_id}"
