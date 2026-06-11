#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_PROJECT}/.venv-openml-broad}"
OUT_PREFIX="${OUT_PREFIX:-bsuite_cartpole_swingup_recovery_probe_4seed_v1}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
MEMORY="${MEMORY:-12G}"
CPUS="${CPUS:-2}"
SEEDS="${SEEDS:-0,1,2,3}"
METHODS="${METHODS:-uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SETUP_REMOTE="${SETUP_REMOTE:-1}"

SBATCH_PARTITION_ARG=""
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_PARTITION_ARG="--partition='${SLURM_PARTITION}'"
fi

rsync -az tools/bsuite_cartpole_swingup_recovery_probe.py "${REMOTE_HOST}:${REMOTE_PROJECT}/tools/"
rsync -az src/ "${REMOTE_HOST}:${REMOTE_PROJECT}/src/"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_LOG_ROOT}' '${REMOTE_RUN_ROOT}' '${REMOTE_LOG_ROOT}/sbatch'"

if [[ "${SETUP_REMOTE}" == "1" ]]; then
  ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT}' && \
    if [[ ! -x '${REMOTE_VENV}/bin/python' ]]; then echo 'missing REMOTE_VENV python: ${REMOTE_VENV}' >&2; exit 2; fi && \
    '${REMOTE_VENV}/bin/python' -m pip install --upgrade pip && \
    '${REMOTE_VENV}/bin/python' -m pip install 'numpy==2.2.6' 'dm-env==1.6' 'bsuite==0.3.6'"
fi

remote_sbatch="${REMOTE_LOG_ROOT}/sbatch/${OUT_PREFIX}_$(date +%Y%m%d_%H%M%S)_$$.sbatch"
ssh "${REMOTE_HOST}" "cat > '${remote_sbatch}'" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_PROJECT}'
echo "[start] \$(date -Is) host=\$(hostname) job=\${SLURM_JOB_ID}"
'${REMOTE_VENV}/bin/python' - <<'PY'
import bsuite
import dm_env
import numpy
print('job_bsuite_env_ok', getattr(bsuite, '__version__', 'ok'), dm_env.__version__ if hasattr(dm_env, '__version__') else 'dm-env', numpy.__version__)
PY
PYTHONPATH='${REMOTE_PROJECT}/src:${REMOTE_PROJECT}' \
  '${REMOTE_VENV}/bin/python' \
  '${REMOTE_PROJECT}/tools/bsuite_cartpole_swingup_recovery_probe.py' \
  --out '${REMOTE_RUN_ROOT}/${OUT_PREFIX}_'\${SLURM_JOB_ID} \
  --seeds '${SEEDS}' \
  --methods '${METHODS}' \
  ${EXTRA_ARGS}
echo "[done] \$(date -Is) job=\${SLURM_JOB_ID}"
EOF

job_id="$(
  ssh "${REMOTE_HOST}" \
    "cd '${REMOTE_PROJECT}' && sbatch --parsable \
      --job-name='bgr-bsuite-swingup' \
      ${SBATCH_PARTITION_ARG} \
      --cpus-per-task='${CPUS}' \
      --mem='${MEMORY}' \
      --time='${TIME_LIMIT}' \
      --output='${REMOTE_LOG_ROOT}/%x-%j.out' \
      '${remote_sbatch}'"
)"

printf 'submitted bsuite Cartpole Swingup job: %s\n' "${job_id}"
printf 'remote output: %s/%s_%s\n' "${REMOTE_RUN_ROOT}" "${OUT_PREFIX}" "${job_id}"
printf 'remote log: %s/bgr-bsuite-swingup-%s.out\n' "${REMOTE_LOG_ROOT}" "${job_id}"
printf 'remote sbatch: %s\n' "${remote_sbatch}"
