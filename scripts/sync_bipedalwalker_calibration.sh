#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
JOB_ID="${JOB_ID:?set JOB_ID}"
OUT_PREFIX="${OUT_PREFIX:-bipedalwalker_recovery_calibration_12seed_v1}"
LOCAL_ROOT="${LOCAL_ROOT:-results}"

remote_out="${REMOTE_RUN_ROOT}/${OUT_PREFIX}_${JOB_ID}"
local_out="${LOCAL_ROOT}/${OUT_PREFIX}_${JOB_ID}"
remote_log="${REMOTE_LOG_ROOT}/bgr-bipedal-calib-${JOB_ID}.out"

echo "### BipedalWalker calibration sync"
ssh "${REMOTE_HOST}" "squeue -j '${JOB_ID}' -o '%i %T %M %R %j' || true"
ssh "${REMOTE_HOST}" "sacct -j '${JOB_ID}' --format=JobID,JobName,State,ExitCode,Elapsed,Start,End -P || true"

mkdir -p "${local_out}"
if ssh "${REMOTE_HOST}" "test -d '${remote_out}'"; then
  rsync -az "${REMOTE_HOST}:${remote_out}/" "${local_out}/"
else
  echo "[missing] ${remote_out}"
fi

if ssh "${REMOTE_HOST}" "test -f '${remote_log}'"; then
  mkdir -p "${local_out}/slurm"
  rsync -az "${REMOTE_HOST}:${remote_log}" "${local_out}/slurm/"
  tail -n 40 "${local_out}/slurm/$(basename "${remote_log}")"
else
  echo "[missing] ${remote_log}"
fi
