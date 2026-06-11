#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_PROJECT}/runs}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_PROJECT}/logs}"
OUT_PREFIX="${OUT_PREFIX:-bsuite_cartpole_swingup_recovery_probe_4seed_v1}"
JOB_IDS="${JOB_IDS:-782844}"

IFS=',' read -r -a JOB_ARRAY <<< "${JOB_IDS}"

printf '### bsuite Cartpole Swingup result sync\n'
printf 'REMOTE_HOST=%s\n' "${REMOTE_HOST}"
printf 'JOB_IDS=%s\n' "${JOB_IDS}"

ssh "${REMOTE_HOST}" "squeue -j '${JOB_IDS}' -o '%.18i %.10T %.12M %.20R %.80j' || true"
ssh "${REMOTE_HOST}" "sacct -j '${JOB_IDS}' --format JobID,JobName,State,ExitCode,Elapsed,Start,End -P 2>/dev/null || true"

for job_id in "${JOB_ARRAY[@]}"; do
  remote_dir="${REMOTE_RUN_ROOT}/${OUT_PREFIX}_${job_id}"
  local_dir="results/${OUT_PREFIX}_${job_id}"
  remote_log="${REMOTE_LOG_ROOT}/bgr-bsuite-swingup-${job_id}.out"
  printf '\n## job %s\n' "${job_id}"
  ssh "${REMOTE_HOST}" "tail -n 80 '${remote_log}' 2>/dev/null || true"
  if ssh "${REMOTE_HOST}" "test -d '${remote_dir}'"; then
    mkdir -p "${local_dir}"
    rsync -az "${REMOTE_HOST}:${remote_dir}/" "${local_dir}/"
    printf '[synced] %s -> %s\n' "${remote_dir}" "${local_dir}"
    if [[ -f "${local_dir}/summary.csv" ]]; then
      sed -n '1,80p' "${local_dir}/summary.csv"
    fi
  else
    printf '[pending] missing remote dir %s\n' "${remote_dir}"
  fi
done
