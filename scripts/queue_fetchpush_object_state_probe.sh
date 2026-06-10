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
  --iterations 30 \
  --eval-every 10 \
  --train-batch-size 4 \
  --horizon 250 \
  --eval-trials 2 \
  --record-trials 1 \
  --quick-trials 1 \
  --eval-radii 0.00,0.02,0.04,0.06,0.08,0.12,0.16,0.20 \
  --initial-probes 0.00,0.20 \
  --max-radius 0.20 \
  --fixed-radius 0.02 \
  --target-radius 0.014 \
  --radius-bandwidth 0.025 \
  --policy trajectory \
  --max-trajectories 512 \
  --warmstart-policy
EOF
)"

printf 'submitted FetchPush object-state scout job: %s\n' "${job_id}"
printf 'remote output: %s/%s_%s\n' "${REMOTE_RUN_ROOT}" "${OUT_PREFIX}" "${job_id}"
printf 'remote log: %s/bgr-fetchpush-object-state-%s.out\n' "${REMOTE_LOG_ROOT}" "${job_id}"
