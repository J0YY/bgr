#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_PROJECT}/.venv-minigrid-dynamic}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
MEMORY="${MEMORY:-12G}"
CPUS="${CPUS:-2}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-minigrid_dynamic_obstacles_recovery_probe_4seed_v1}"
ENV_ID="${ENV_ID:-MiniGrid-Dynamic-Obstacles-6x6-v0}"
SEEDS="${SEEDS:-0,1,2,3}"

SBATCH_PARTITION_ARG=""
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_PARTITION_ARG="--partition='${SLURM_PARTITION}'"
fi

rsync -az tools/minigrid_dynamic_obstacles_recovery_probe.py "${REMOTE_HOST}:${REMOTE_PROJECT}/tools/minigrid_dynamic_obstacles_recovery_probe.py"
rsync -az src/ "${REMOTE_HOST}:${REMOTE_PROJECT}/src/"

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_LOG_ROOT}' '${REMOTE_RUN_ROOT}'"

job_id="$(
  ssh "${REMOTE_HOST}" \
    "cd '${REMOTE_PROJECT}' && sbatch --parsable \
      --job-name='bgr-minigrid-dynamic' \
      ${SBATCH_PARTITION_ARG} \
      --cpus-per-task='${CPUS}' \
      --mem='${MEMORY}' \
      --time='${TIME_LIMIT}' \
      --output='${REMOTE_LOG_ROOT}/%x-%j.out'" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_PROJECT}'
if [[ ! -x '${REMOTE_VENV}/bin/python' ]]; then
  python3 -m venv '${REMOTE_VENV}'
  '${REMOTE_VENV}/bin/python' -m pip install --upgrade pip
  '${REMOTE_VENV}/bin/python' -m pip install 'gymnasium==1.3.0' 'minigrid==3.0.0'
fi
PYTHONPATH='${REMOTE_PROJECT}/src:${REMOTE_PROJECT}' \
  '${REMOTE_VENV}/bin/python' \
  '${REMOTE_PROJECT}/tools/minigrid_dynamic_obstacles_recovery_probe.py' \
  --env-id '${ENV_ID}' \
  --seeds '${SEEDS}' \
  --out "${REMOTE_RUN_ROOT}/${ARTIFACT_PREFIX}_\${SLURM_JOB_ID}"
EOF
)"

printf 'submitted job: %s\n' "${job_id}"
printf 'remote run: %s/%s_%s\n' "${REMOTE_RUN_ROOT}" "${ARTIFACT_PREFIX}" "${job_id}"
printf 'remote log: %s/bgr-minigrid-dynamic-%s.out\n' "${REMOTE_LOG_ROOT}" "${job_id}"
