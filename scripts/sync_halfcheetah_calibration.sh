#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/work/joy/bgr/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-/work/joy/bgr/logs}"
JOB_ID="${JOB_ID:-}"
OUT_PREFIX="${OUT_PREFIX:-halfcheetah_recovery_calibration_12seed_v1}"

if [[ -z "${JOB_ID}" ]]; then
  echo "JOB_ID is required" >&2
  exit 2
fi

remote_run="${REMOTE_RUN_ROOT}/${OUT_PREFIX}_${JOB_ID}"
local_run="results/${OUT_PREFIX}_${JOB_ID}"
remote_log="${REMOTE_LOG_ROOT}/bgr-halfcheetah-calib-${JOB_ID}.out"

echo "### HalfCheetah recovery calibration sync"
echo "JOB_ID=${JOB_ID}"
ssh "${REMOTE_HOST}" "squeue -j '${JOB_ID}' -o '%.18i %.9T %.10M %.20R %.80j' || true"
ssh "${REMOTE_HOST}" "sacct -j '${JOB_ID}' --format=JobID,JobName%60,State,ExitCode,Elapsed,Start,End -P 2>/dev/null || true"

mkdir -p "${local_run}/slurm"
if ssh "${REMOTE_HOST}" "test -d '${remote_run}'"; then
  rsync -az "${REMOTE_HOST}:${remote_run}/" "${local_run}/"
  echo "[synced] ${remote_run} -> ${local_run}"
else
  echo "[pending] missing remote run ${remote_run}"
fi
if ssh "${REMOTE_HOST}" "test -f '${remote_log}'"; then
  rsync -az "${REMOTE_HOST}:${remote_log}" "${local_run}/slurm/"
  echo "[synced-log] ${remote_log}"
fi
if [[ -f "${local_run}/summary.json" ]]; then
  echo "[summary] ${local_run}/summary.json"
  python3 - <<PY
import json
from pathlib import Path
summary = json.loads(Path("${local_run}/summary.json").read_text())
print(
    "decision={decision} clean={clean_success:.4f} range={min_recovery:.4f}--{max_recovery:.4f} "
    "rauc={rauc:.4f} r80={r80:.4f}".format(**summary)
)
PY
fi
