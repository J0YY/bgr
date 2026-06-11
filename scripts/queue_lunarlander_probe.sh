#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_PROJECT}/.venv-minigrid-dynamic}"
SETUP_REMOTE="${SETUP_REMOTE:-1}"
OUT_PREFIX="${OUT_PREFIX:-lunarlander_recovery_probe_30seed_v1}"
TIME_LIMIT="${TIME_LIMIT:-03:00:00}"
MEMORY="${MEMORY:-16G}"
CPUS="${CPUS:-2}"
SEEDS="${SEEDS:-0,1,2,3}"
METHODS="${METHODS:-uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr}"

SBATCH_PARTITION_ARG=""
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_PARTITION_ARG="--partition='${SLURM_PARTITION}'"
fi

rsync -az \
  tools/lunarlander_recovery_calibration.py \
  tools/lunarlander_recovery_probe.py \
  "${REMOTE_HOST}:${REMOTE_PROJECT}/tools/"
rsync -az src/ "${REMOTE_HOST}:${REMOTE_PROJECT}/src/"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_LOG_ROOT}' '${REMOTE_RUN_ROOT}' '${REMOTE_LOG_ROOT}/sbatch'"

if [[ "${SETUP_REMOTE}" == "1" ]]; then
  ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT}' && \
    if [[ ! -x '${REMOTE_VENV}/bin/python' ]]; then python3 -m venv '${REMOTE_VENV}'; fi && \
    '${REMOTE_VENV}/bin/python' -m pip install --upgrade pip && \
    '${REMOTE_VENV}/bin/python' -m pip install 'numpy==2.2.6' 'gymnasium==1.3.0' 'box2d==2.3.10' 'pygame-ce==2.5.7' 'swig==4.4.1' && \
    '${REMOTE_VENV}/bin/python' - <<'PY'
import numpy
import gymnasium
import Box2D
print('remote_lunar_env_ok', numpy.__version__, gymnasium.__version__)
PY"
fi

remote_sbatch="${REMOTE_LOG_ROOT}/sbatch/${OUT_PREFIX}_$(echo "${METHODS}" | tr ',/' '__').sbatch"
ssh "${REMOTE_HOST}" "cat > '${remote_sbatch}'" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_PROJECT}'
echo "[start] \$(date -Is) host=\$(hostname) job=\${SLURM_JOB_ID}"
'${REMOTE_VENV}/bin/python' - <<'PY'
import numpy
import gymnasium
import Box2D
print('job_lunar_env_ok', numpy.__version__, gymnasium.__version__)
PY
PYTHONPATH='${REMOTE_PROJECT}/src:${REMOTE_PROJECT}:${REMOTE_PROJECT}/tools' \
  '${REMOTE_VENV}/bin/python' \
  '${REMOTE_PROJECT}/tools/lunarlander_recovery_probe.py' \
  --out '${REMOTE_RUN_ROOT}/${OUT_PREFIX}_'\${SLURM_JOB_ID} \
  --seeds '${SEEDS}' \
  --methods '${METHODS}'
echo "[done] \$(date -Is) job=\${SLURM_JOB_ID}"
EOF

job_id="$(
  ssh "${REMOTE_HOST}" \
    "cd '${REMOTE_PROJECT}' && sbatch --parsable \
      --job-name='bgr-lunarlander' \
      ${SBATCH_PARTITION_ARG} \
      --cpus-per-task='${CPUS}' \
      --mem='${MEMORY}' \
      --time='${TIME_LIMIT}' \
      --output='${REMOTE_LOG_ROOT}/%x-%j.out' \
      '${remote_sbatch}'"
)"

printf 'submitted LunarLander job: %s\n' "${job_id}"
printf 'remote output: %s/%s_%s\n' "${REMOTE_RUN_ROOT}" "${OUT_PREFIX}" "${job_id}"
printf 'remote log: %s/bgr-lunarlander-%s.out\n' "${REMOTE_LOG_ROOT}" "${job_id}"
printf 'remote sbatch: %s\n' "${remote_sbatch}"
