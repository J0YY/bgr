#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
LOCAL_ROOT="${LOCAL_ROOT:-results}"

if [[ -z "${JOB_ID:-}" || -z "${OUT_PREFIX:-}" ]]; then
  echo "usage: JOB_ID=<slurm-id> OUT_PREFIX=<run-prefix> $0" >&2
  exit 2
fi

remote_out="${REMOTE_RUN_ROOT}/${OUT_PREFIX}_${JOB_ID}"
local_out="${LOCAL_ROOT}/${OUT_PREFIX}_${JOB_ID}"
remote_log="${REMOTE_LOG_ROOT}/bgr-openml-mixed-binary-${JOB_ID}.out"

echo "### OpenML single-result sync"
echo "REMOTE_HOST=${REMOTE_HOST}"
echo "JOB_ID=${JOB_ID}"
echo "OUT_PREFIX=${OUT_PREFIX}"

ssh "${REMOTE_HOST}" "squeue -j '${JOB_ID}' -o '%.18i %.10T %.12M %.20R %.80j' || true"
ssh "${REMOTE_HOST}" "sacct -j '${JOB_ID}' --format=JobID,JobName,State,ExitCode,Elapsed,Start,End -P 2>/dev/null || true"

mkdir -p "${local_out}"
if ssh "${REMOTE_HOST}" "test -d '${remote_out}'"; then
  rsync -az "${REMOTE_HOST}:${remote_out}/" "${local_out}/"
  echo "[synced] ${remote_out} -> ${local_out}"
else
  echo "[pending] missing ${remote_out}"
fi

if ssh "${REMOTE_HOST}" "test -f '${remote_log}'"; then
  mkdir -p "${local_out}/slurm"
  rsync -az "${REMOTE_HOST}:${remote_log}" "${local_out}/slurm/"
  echo "--- tail ${remote_log} ---"
  tail -n 80 "${local_out}/slurm/$(basename "${remote_log}")"
else
  echo "[pending] missing ${remote_log}"
fi

if [[ -f "${local_out}/summary.csv" ]]; then
  echo "--- summary.csv ---"
  sed -n '1,120p' "${local_out}/summary.csv"
fi
