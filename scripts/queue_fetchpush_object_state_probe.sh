#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${REMOTE_PROJECT}/.venv-fetchpush/bin/python}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
OUT_PREFIX="${OUT_PREFIX:-fetchpush_object_state_recovery_probe_scout_v1}"
TIME_LIMIT="${TIME_LIMIT:-8:00:00}"
MEMORY="${MEMORY:-16G}"
CPUS="${CPUS:-4}"
SEEDS="${SEEDS:-0,1,2,3}"
METHODS="${METHODS:-uniform,fixed,failure_only,td_loss,bgr_uniform_radius,bgr_coverage,bgr}"
ITERATIONS="${ITERATIONS:-30}"
EVAL_EVERY="${EVAL_EVERY:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_TRIALS="${EVAL_TRIALS:-2}"
RECORD_TRIALS="${RECORD_TRIALS:-1}"
QUICK_TRIALS="${QUICK_TRIALS:-1}"
EVAL_RADII="${EVAL_RADII:-0.00,0.02,0.04,0.06,0.08,0.12,0.16,0.20}"
INITIAL_PROBES="${INITIAL_PROBES:-0.00,0.20}"
MAX_RADIUS="${MAX_RADIUS:-0.20}"
FIXED_RADIUS="${FIXED_RADIUS:-0.02}"
TARGET_RADIUS="${TARGET_RADIUS:-0.014}"
RADIUS_BANDWIDTH="${RADIUS_BANDWIDTH:-0.025}"
RADIUS_UNIFORM_MIX="${RADIUS_UNIFORM_MIX:-0.30}"

SBATCH_PARTITION_ARG=""
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_PARTITION_ARG="--partition='${SLURM_PARTITION}'"
fi

rsync -az \
  tools/fetch_object_goal_recovery_calibration.py \
  tools/fetchpush_object_state_recovery_probe.py \
  "${REMOTE_HOST}:${REMOTE_PROJECT}/tools/"

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_LOG_ROOT}' '${REMOTE_RUN_ROOT}'"

job_id="$(ssh "${REMOTE_HOST}" \
  "cd '${REMOTE_PROJECT}' && sbatch --parsable \
    --job-name='bgr-fetchpush-object-state' \
    ${SBATCH_PARTITION_ARG} \
    --cpus-per-task='${CPUS}' \
    --mem='${MEMORY}' \
    --time='${TIME_LIMIT}' \
    --output='${REMOTE_LOG_ROOT}/%x-%j.out'" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_PROJECT}'
if [[ ! -x '${REMOTE_PYTHON}' ]]; then
  python3 -m venv '${REMOTE_PROJECT}/.venv-fetchpush'
  '${REMOTE_PYTHON}' -m pip install --upgrade pip
  '${REMOTE_PYTHON}' -m pip install 'numpy==2.2.6' 'gymnasium==1.3.0' 'gymnasium-robotics==1.4.2' 'mujoco==3.9.0'
fi
PYTHONPATH='${REMOTE_PROJECT}/src:${REMOTE_PROJECT}:${REMOTE_PROJECT}/tools' \
  '${REMOTE_PYTHON}' \
  '${REMOTE_PROJECT}/tools/fetchpush_object_state_recovery_probe.py' \
  --out '${REMOTE_RUN_ROOT}/${OUT_PREFIX}_'\${SLURM_JOB_ID} \
  --seeds '${SEEDS}' \
  --methods '${METHODS}' \
  --replay-states 4 \
  --iterations '${ITERATIONS}' \
  --eval-every '${EVAL_EVERY}' \
  --train-batch-size '${TRAIN_BATCH_SIZE}' \
  --horizon 250 \
  --eval-trials '${EVAL_TRIALS}' \
  --record-trials '${RECORD_TRIALS}' \
  --quick-trials '${QUICK_TRIALS}' \
  --eval-radii '${EVAL_RADII}' \
  --initial-probes '${INITIAL_PROBES}' \
  --max-radius '${MAX_RADIUS}' \
  --fixed-radius '${FIXED_RADIUS}' \
  --target-radius '${TARGET_RADIUS}' \
  --radius-bandwidth '${RADIUS_BANDWIDTH}' \
  --radius-uniform-mix '${RADIUS_UNIFORM_MIX}' \
  --policy trajectory \
  --max-trajectories 512 \
  --warmstart-policy
EOF
)"

printf 'submitted FetchPush object-state scout job: %s\n' "${job_id}"
printf 'remote output: %s/%s_%s\n' "${REMOTE_RUN_ROOT}" "${OUT_PREFIX}" "${job_id}"
printf 'remote log: %s/bgr-fetchpush-object-state-%s.out\n' "${REMOTE_LOG_ROOT}" "${job_id}"
