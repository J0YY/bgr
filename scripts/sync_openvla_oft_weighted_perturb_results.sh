#!/usr/bin/env bash
set -euo pipefail

POLL=0
SYNC=0
CHECK_LOCAL=1

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/work/anonymous/bgr/runs}"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-results}"

TAG="${TAG:-cleanmix_p2048unique_perturbrepeat3_prereg_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1}"
ARTIFACT="${ARTIFACT:-openvla_oft_perturb_eval_${TAG}}"
JOB_IDS="${JOB_IDS:-766822,766823,766824,766825,766826,766827,766828,766829,766830,766831}"
DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-766831}"

REMOTE_SUMMARY="${REMOTE_SUMMARY:-${REMOTE_RUN_ROOT}/${ARTIFACT}/summary.csv}"
LOCAL_SUMMARY="${LOCAL_SUMMARY:-${LOCAL_RESULTS_ROOT}/${ARTIFACT}/summary.csv}"
LOCAL_AVAILABLE_SUMMARY="${LOCAL_AVAILABLE_SUMMARY:-${LOCAL_RESULTS_ROOT}/${ARTIFACT}/summary_available.csv}"

usage() {
  cat <<USAGE
Usage: scripts/sync_openvla_oft_weighted_perturb_results.sh [options]

Dry-run by default. This helper is for the preregistered weighted OpenVLA-OFT
perturbation audit. The audit is already negative against the official
checkpoint before the final matched-random shift row, so this script is for
ledger completion only.

Options:
  --poll          run remote squeue/sacct, selected scontrol details, and summary checks
  --sync          rsync the compact remote summary if present
  --no-check      skip local gate/readiness commands after sync/dry-run output
  -h, --help      show this message

Remote summary:
  ${REMOTE_SUMMARY}

Local summaries:
  complete:   ${LOCAL_SUMMARY}
  incomplete: ${LOCAL_AVAILABLE_SUMMARY}
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --poll) POLL=1; shift ;;
    --sync) SYNC=1; shift ;;
    --no-check) CHECK_LOCAL=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

echo "### Weighted OpenVLA-OFT perturbation result sync"
echo "REMOTE_HOST=${REMOTE_HOST}"
echo "JOB_IDS=${JOB_IDS}"
echo "DETAIL_JOB_IDS=${DETAIL_JOB_IDS}"
echo "REMOTE_SUMMARY=${REMOTE_SUMMARY}"
echo "LOCAL_SUMMARY=${LOCAL_SUMMARY}"
echo "LOCAL_AVAILABLE_SUMMARY=${LOCAL_AVAILABLE_SUMMARY}"

remote_poll_script=$(cat <<'REMOTE'
date
squeue -j "${JOB_IDS}" -o '%i %T %M %R' || true
sacct -j "${JOB_IDS}" --format=JobID,JobName%50,State,ExitCode,Elapsed,Start,End -P || true
if command -v scontrol >/dev/null 2>&1 && [ -n "${DETAIL_JOB_IDS:-}" ]; then
  old_ifs="${IFS}"
  IFS=,
  set -- ${DETAIL_JOB_IDS}
  IFS="${old_ifs}"
  for job_id in "$@"; do
    echo "DETAIL:${job_id}"
    scontrol show job -dd "${job_id}" | \
      egrep 'JobId=|JobState=|Reason=|StartTime=|Partition=|ReqNodeList=|ExcNodeList=|ReqTRES=|TresPerNode=|MinCPUsNode=|MinMemoryNode=' || true
  done
fi
if [ -f "${REMOTE_SUMMARY}" ]; then
  echo "FOUND:${REMOTE_SUMMARY}"
  wc -l "${REMOTE_SUMMARY}"
  tail -8 "${REMOTE_SUMMARY}"
else
  echo "MISSING:${REMOTE_SUMMARY}"
fi
REMOTE
)

if [[ "${POLL}" -eq 1 ]]; then
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "JOB_IDS='${JOB_IDS}' DETAIL_JOB_IDS='${DETAIL_JOB_IDS}' REMOTE_SUMMARY='${REMOTE_SUMMARY}' bash -s" \
    <<<"${remote_poll_script}"
else
  echo "[dry-run] pass --poll to run remote Slurm and summary checks"
fi

sync_summary() {
  mkdir -p "$(dirname "${LOCAL_SUMMARY}")"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "test -f '${REMOTE_SUMMARY}'"; then
    echo "[missing] ${REMOTE_SUMMARY}"
    return
  fi

  local tmp_path
  tmp_path="$(mktemp "$(dirname "${LOCAL_SUMMARY}")/.weighted-summary.XXXXXX.csv")"
  rsync -az "${REMOTE_HOST}:${REMOTE_SUMMARY}" "${tmp_path}"
  if PYTHONPATH=src:. python3 scripts/check_openvla_perturb_gate.py \
      --perturb-summary "${tmp_path}" --require-complete >/dev/null 2>&1; then
    mv "${tmp_path}" "${LOCAL_SUMMARY}"
    echo "[synced-complete] ${REMOTE_SUMMARY} -> ${LOCAL_SUMMARY}"
  else
    mv "${tmp_path}" "${LOCAL_AVAILABLE_SUMMARY}"
    echo "[synced-incomplete] ${REMOTE_SUMMARY} -> ${LOCAL_AVAILABLE_SUMMARY}"
  fi
}

if [[ "${SYNC}" -eq 1 ]]; then
  sync_summary
else
  echo "[dry-run] pass --sync to rsync compact summary if present"
fi

if [[ "${CHECK_LOCAL}" -eq 1 ]]; then
  if [[ -f "${LOCAL_SUMMARY}" ]]; then
    PYTHONPATH=src:. python3 scripts/check_openvla_perturb_gate.py \
      --perturb-summary "${LOCAL_SUMMARY}"
  elif [[ -f "${LOCAL_AVAILABLE_SUMMARY}" ]]; then
    PYTHONPATH=src:. python3 scripts/check_openvla_perturb_gate.py \
      --perturb-summary "${LOCAL_AVAILABLE_SUMMARY}"
  else
    echo "[skip] local weighted perturb summary missing"
  fi
  PYTHONPATH=src:. python3 scripts/check_acceptance_readiness.py --root .
  PYTHONPATH=src:. python3 scripts/acceptance_scorecard.py --root . --out docs/acceptance_scorecard.md
else
  echo "[skip] local gates disabled by --no-check"
fi
