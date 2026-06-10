#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"

ORIG_JOB_ID="${ORIG_JOB_ID:-778596}"
REP_JOB_ID="${REP_JOB_ID:-778597}"
ORIG_PREFIX="${ORIG_PREFIX:-openml_mixed_binary_target_sensitivity_30seed_v1}"
REP_PREFIX="${REP_PREFIX:-openml_mixed_binary_target_sensitivity_replication_30seed_v1}"
LOCAL_ROOT="${LOCAL_ROOT:-results}"

orig_remote_out="${REMOTE_RUN_ROOT}/${ORIG_PREFIX}_${ORIG_JOB_ID}"
rep_remote_out="${REMOTE_RUN_ROOT}/${REP_PREFIX}_${REP_JOB_ID}"
orig_local_out="${LOCAL_ROOT}/${ORIG_PREFIX}_${ORIG_JOB_ID}"
rep_local_out="${LOCAL_ROOT}/${REP_PREFIX}_${REP_JOB_ID}"

echo "### Mixed OpenML binary suite sync"
echo "REMOTE_HOST=${REMOTE_HOST}"
echo "ORIG_JOB_ID=${ORIG_JOB_ID}"
echo "REP_JOB_ID=${REP_JOB_ID}"

ssh "${REMOTE_HOST}" "squeue -j '${ORIG_JOB_ID},${REP_JOB_ID}' -o '%i %T %M %R %j' || true"
ssh "${REMOTE_HOST}" "sacct -j '${ORIG_JOB_ID},${REP_JOB_ID}' --format=JobID,JobName,State,ExitCode,Elapsed,Start,End -P || true"

sync_one() {
  local job_id="$1"
  local remote_out="$2"
  local local_out="$3"
  local remote_log="${REMOTE_LOG_ROOT}/bgr-openml-mixed-binary-${job_id}.out"

  mkdir -p "${local_out}"
  if ssh "${REMOTE_HOST}" "test -d '${remote_out}'"; then
    rsync -az "${REMOTE_HOST}:${remote_out}/" "${local_out}/"
  else
    echo "[missing] ${remote_out}"
  fi

  if ssh "${REMOTE_HOST}" "test -f '${remote_log}'"; then
    mkdir -p "${local_out}/slurm"
    rsync -az "${REMOTE_HOST}:${remote_log}" "${local_out}/slurm/"
    echo "--- tail ${remote_log} ---"
    tail -n 40 "${local_out}/slurm/$(basename "${remote_log}")"
  else
    echo "[missing] ${remote_log}"
  fi
}

sync_one "${ORIG_JOB_ID}" "${orig_remote_out}" "${orig_local_out}"
sync_one "${REP_JOB_ID}" "${rep_remote_out}" "${rep_local_out}"

if [[ -f "${orig_local_out}/per_seed.csv" && -f "${rep_local_out}/per_seed.csv" ]]; then
  PYTHONPATH=src:. python3 tools/analyze_openml_margin_suite.py \
    --original "${orig_local_out}/per_seed.csv" \
    --replication "${rep_local_out}/per_seed.csv" \
    | tee "${LOCAL_ROOT}/openml_mixed_binary_target_sensitivity_analysis_${ORIG_JOB_ID}_${REP_JOB_ID}.txt"
else
  echo "[pending] missing per_seed.csv for paired analysis"
fi
