#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${REMOTE_PROJECT}/.venv-openml-broad/bin/python}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
OUT_PREFIX="${OUT_PREFIX:-openml_mixed_binary_scout_v1}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
MEMORY="${MEMORY:-32G}"
CPUS="${CPUS:-4}"
TARGETS="${TARGETS:-0.5,1.0,1.5,2.0}"
SEEDS="${SEEDS:-4}"
SEED_START="${SEED_START:-0}"
STEPS="${STEPS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-128}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-250}"
DATASETS="${DATASETS:-}"
PREPROCESSING="${PREPROCESSING:-mixed}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
RESUME="${RESUME:-1}"

if [[ -n "${DATASETS}" ]]; then
  DATASET_ARGS="--datasets '${DATASETS}' --preprocessing '${PREPROCESSING}'"
else
  DATASET_ARGS="--mixed-binary-suite"
fi

SBATCH_PARTITION_ARG=""
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_PARTITION_ARG="--partition='${SLURM_PARTITION}'"
fi

rsync -az tools/openml_margin_scout.py "${REMOTE_HOST}:${REMOTE_PROJECT}/tools/openml_margin_scout.py"

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_LOG_ROOT}' '${REMOTE_RUN_ROOT}' '${REMOTE_LOG_ROOT}/sbatch'"

remote_script="${REMOTE_LOG_ROOT}/sbatch/${OUT_PREFIX}_$(date +%Y%m%d_%H%M%S)_$$.sbatch"
resume_arg=""
if [[ "${RESUME}" != "0" ]]; then
  resume_arg="--resume"
fi

ssh "${REMOTE_HOST}" "cat > '${remote_script}'" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "[start] \$(date -Is)"
echo "[host] \$(hostname)"
echo "[job] \${SLURM_JOB_ID:-manual}"
cd '${REMOTE_PROJECT}'
PYTHONPATH='${REMOTE_PROJECT}/src:${REMOTE_PROJECT}' \
  '${REMOTE_PYTHON}' \
  '${REMOTE_PROJECT}/tools/openml_margin_scout.py' \
  ${DATASET_ARGS} \
  --targets '${TARGETS}' \
  --seed-start '${SEED_START}' \
  --seeds '${SEEDS}' \
  --steps '${STEPS}' \
  --batch-size '${BATCH_SIZE}' \
  --candidate-count '${CANDIDATE_COUNT}' \
  --eval-examples '${EVAL_EXAMPLES}' \
  --checkpoint-every '${CHECKPOINT_EVERY}' \
  ${resume_arg} \
  --out "${REMOTE_RUN_ROOT}/${OUT_PREFIX}_\${SLURM_JOB_ID}"
echo "[done] \$(date -Is)"
EOF

job_id="$(ssh "${REMOTE_HOST}" \
  "cd '${REMOTE_PROJECT}' && sbatch --parsable \
    --job-name='bgr-openml-mixed-binary' \
    ${SBATCH_PARTITION_ARG} \
    --cpus-per-task='${CPUS}' \
    --mem='${MEMORY}' \
    --time='${TIME_LIMIT}' \
    --output='${REMOTE_LOG_ROOT}/%x-%j.out' \
    '${remote_script}'")"

printf 'submitted mixed OpenML binary scout job: %s\n' "${job_id}"
printf 'remote output: %s/%s_%s\n' "${REMOTE_RUN_ROOT}" "${OUT_PREFIX}" "${job_id}"
printf 'remote log: %s/bgr-openml-mixed-binary-%s.out\n' "${REMOTE_LOG_ROOT}" "${job_id}"
printf 'remote script: %s\n' "${remote_script}"
