#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
OUT_PREFIX="${OUT_PREFIX:-fetchpush_object_state_recovery_probe_scout_v1}"
JOB_ID="${JOB_ID:-777783}"
LOCAL_OUT="${LOCAL_OUT:-results/${OUT_PREFIX}_${JOB_ID}}"

remote_out="${REMOTE_RUN_ROOT}/${OUT_PREFIX}_${JOB_ID}"
remote_log="${REMOTE_LOG_ROOT}/bgr-fetchpush-object-state-${JOB_ID}.out"

ssh "${REMOTE_HOST}" "squeue -j '${JOB_ID}' -o '%i %T %M %R' || true"
ssh "${REMOTE_HOST}" "sacct -j '${JOB_ID}' --format=JobID,JobName,State,ExitCode,Elapsed,Start,End -P || true"

mkdir -p "${LOCAL_OUT}"
if ssh "${REMOTE_HOST}" "test -d '${remote_out}'"; then
  rsync -az "${REMOTE_HOST}:${remote_out}/" "${LOCAL_OUT}/"
else
  echo "[missing] ${remote_out}"
fi

if ssh "${REMOTE_HOST}" "test -f '${remote_log}'"; then
  mkdir -p "${LOCAL_OUT}/slurm"
  rsync -az "${REMOTE_HOST}:${remote_log}" "${LOCAL_OUT}/slurm/"
  tail -n 80 "${LOCAL_OUT}/slurm/$(basename "${remote_log}")"
else
  echo "[missing] ${remote_log}"
fi
