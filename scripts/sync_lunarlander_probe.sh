#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
OUT_PREFIX="${OUT_PREFIX:-lunarlander_recovery_probe_30seed_v1}"
JOB_ID="${JOB_ID:-}"
LOCAL_OUT="${LOCAL_OUT:-results/${OUT_PREFIX}_${JOB_ID}}"

if [[ -z "${JOB_ID}" ]]; then
  echo "JOB_ID is required" >&2
  exit 2
fi

remote_out="${REMOTE_RUN_ROOT}/${OUT_PREFIX}_${JOB_ID}"
remote_log="${REMOTE_LOG_ROOT}/bgr-lunarlander-${JOB_ID}.out"

echo "### LunarLander recovery probe sync"
echo "JOB_ID=${JOB_ID}"
ssh "${REMOTE_HOST}" "squeue -j '${JOB_ID}' -o '%i %T %M %R %j' || true"
ssh "${REMOTE_HOST}" "sacct -j '${JOB_ID}' --format=JobID,JobName,State,ExitCode,Elapsed,Start,End -P 2>/dev/null || true"

mkdir -p "${LOCAL_OUT}"
if ssh "${REMOTE_HOST}" "test -d '${remote_out}'"; then
  rsync -az "${REMOTE_HOST}:${remote_out}/" "${LOCAL_OUT}/"
  echo "[synced] ${remote_out} -> ${LOCAL_OUT}"
else
  echo "[pending] missing remote run ${remote_out}"
fi

if ssh "${REMOTE_HOST}" "test -f '${remote_log}'"; then
  mkdir -p "${LOCAL_OUT}/slurm"
  rsync -az "${REMOTE_HOST}:${remote_log}" "${LOCAL_OUT}/slurm/"
  echo "[synced-log] ${remote_log}"
  tail -n 80 "${LOCAL_OUT}/slurm/$(basename "${remote_log}")"
else
  echo "[pending] missing remote log ${remote_log}"
fi

if [[ -f "${LOCAL_OUT}/summary.csv" ]]; then
  echo "[summary] ${LOCAL_OUT}/summary.csv"
  PYTHONPATH=src:. python3 tools/check_candidate_promotion.py "${LOCAL_OUT}/summary.csv" \
    --treatment bgr_coverage --min-seeds 4 --min-wins 3 --min-delta 0.01 || true
  PYTHONPATH=src:. python3 tools/check_candidate_promotion.py "${LOCAL_OUT}/summary.csv" \
    --treatment bgr --min-seeds 4 --min-wins 3 --min-delta 0.01 || true
fi
