#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${REMOTE_PROJECT}/.venv-openml-broad/bin/python}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
MEMORY="${MEMORY:-24G}"
CPUS="${CPUS:-4}"
DATASETS="${DATASETS:-jm1,kc2}"
TARGETS="${TARGETS:-1.0,1.5,2.0}"

SBATCH_PARTITION_ARG=""
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_PARTITION_ARG="--partition='${SLURM_PARTITION}'"
fi

rsync -az tools/openml_margin_scout.py "${REMOTE_HOST}:${REMOTE_PROJECT}/tools/openml_margin_scout.py"

ssh "${REMOTE_HOST}" \
  "mkdir -p '${REMOTE_LOG_ROOT}' '${REMOTE_RUN_ROOT}'"

submit_job() {
  local job_name="$1"
  local seed_args="$2"
  local out_prefix="$3"

  ssh "${REMOTE_HOST}" \
    "cd '${REMOTE_PROJECT}' && sbatch --parsable \
      --job-name='${job_name}' \
      ${SBATCH_PARTITION_ARG} \
      --cpus-per-task='${CPUS}' \
      --mem='${MEMORY}' \
      --time='${TIME_LIMIT}' \
      --output='${REMOTE_LOG_ROOT}/%x-%j.out'" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_PROJECT}'
PYTHONPATH='${REMOTE_PROJECT}/src:${REMOTE_PROJECT}' \
  '${REMOTE_PYTHON}' \
  '${REMOTE_PROJECT}/tools/openml_margin_scout.py' \
  --datasets '${DATASETS}' \
  --targets '${TARGETS}' \
  ${seed_args} \
  --out "${REMOTE_RUN_ROOT}/${out_prefix}_\${SLURM_JOB_ID}"
EOF
}

original_job="$(submit_job \
  bgr-openml-secondary-positive-targets \
  '--seeds 30' \
  openml_secondary_positive_target_sensitivity_30seed)"
replication_job="$(submit_job \
  bgr-openml-secondary-positive-targets-rep \
  '--seed-start 30 --seeds 30' \
  openml_secondary_positive_target_sensitivity_replication_30seed)"

printf 'submitted original sensitivity job: %s\n' "${original_job}"
printf 'submitted held-out sensitivity job: %s\n' "${replication_job}"
