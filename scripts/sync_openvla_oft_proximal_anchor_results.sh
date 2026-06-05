#!/usr/bin/env bash
set -euo pipefail

POLL=0
SYNC=0
CHECK_LOCAL=1

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/work/anonymous/bgr/runs}"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-results}"

DATA_TAG="${DATA_TAG:-p2048unique_perturbrepeat3_prereg}"
PROXIMAL_TAG="${PROXIMAL_TAG:-proxanchor_l2_1em0}"
ADAPT_TAG="${ADAPT_TAG:-cleanmix_${DATA_TAG}_${PROXIMAL_TAG}_step50500_lr5em7_identitylora_imageaug_officialtrainstats_v1}"
PERTURB_TAG="${PERTURB_TAG:-cleanmix_${DATA_TAG}_${PROXIMAL_TAG}_step50500_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1}"
ADAPT_ARTIFACT="${ADAPT_ARTIFACT:-openvla_oft_goal_adapt_eval_${ADAPT_TAG}}"
PERTURB_ARTIFACT="${PERTURB_ARTIFACT:-openvla_oft_perturb_eval_${PERTURB_TAG}}"

JOB_IDS="${JOB_IDS:-767128,767129,767130,767131,767132,767133,767134,767135,767136,767137,767138,767139,767140,767141,767142,767143,767144,767145,767146,767147,767148}"
DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-767128,767134}"

REMOTE_ADAPT_SUMMARY="${REMOTE_ADAPT_SUMMARY:-${REMOTE_RUN_ROOT}/${ADAPT_ARTIFACT}/summary.csv}"
REMOTE_PERTURB_SUMMARY="${REMOTE_PERTURB_SUMMARY:-${REMOTE_RUN_ROOT}/${PERTURB_ARTIFACT}/summary.csv}"
LOCAL_ADAPT_SUMMARY="${LOCAL_ADAPT_SUMMARY:-${LOCAL_RESULTS_ROOT}/${ADAPT_ARTIFACT}/summary.csv}"
LOCAL_PERTURB_SUMMARY="${LOCAL_PERTURB_SUMMARY:-${LOCAL_RESULTS_ROOT}/${PERTURB_ARTIFACT}/summary.csv}"

usage() {
  cat <<USAGE
Usage: scripts/sync_openvla_oft_proximal_anchor_results.sh [options]

Dry-run by default. This helper is for the preregistered proximal-anchor
OpenVLA-OFT route. It polls the fixed Slurm job IDs, syncs compact summaries
when present, and runs the local learned-policy gates.

Options:
  --poll          run remote squeue/sacct and summary-existence checks
  --sync          rsync compact summary.csv files from REMOTE_HOST if present
  --no-check      skip local check commands after sync/dry-run output
  -h, --help      show this message

Remote summaries:
  ${REMOTE_ADAPT_SUMMARY}
  ${REMOTE_PERTURB_SUMMARY}

Local summaries:
  ${LOCAL_ADAPT_SUMMARY}
  ${LOCAL_PERTURB_SUMMARY}
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

echo "### Proximal-anchor OpenVLA-OFT result sync"
echo "REMOTE_HOST=${REMOTE_HOST}"
echo "JOB_IDS=${JOB_IDS}"
echo "DETAIL_JOB_IDS=${DETAIL_JOB_IDS}"
echo "REMOTE_PERTURB_SUMMARY=${REMOTE_PERTURB_SUMMARY}"
echo "LOCAL_PERTURB_SUMMARY=${LOCAL_PERTURB_SUMMARY}"
echo "REMOTE_ADAPT_SUMMARY=${REMOTE_ADAPT_SUMMARY}"
echo "LOCAL_ADAPT_SUMMARY=${LOCAL_ADAPT_SUMMARY}"

remote_poll_script=$(cat <<'REMOTE'
date
squeue -j "${JOB_IDS}" -o '%i %T %M %R' || true
sacct -j "${JOB_IDS}" --format=JobID,JobName%40,State,ExitCode,Elapsed,Start,End -P || true
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
for p in "${REMOTE_PERTURB_SUMMARY}" "${REMOTE_ADAPT_SUMMARY}"; do
  if [ -f "$p" ]; then
    echo "FOUND:$p"
    wc -l "$p"
    head -20 "$p"
  else
    echo "MISSING:$p"
  fi
done
REMOTE
)

if [[ "${POLL}" -eq 1 ]]; then
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "JOB_IDS='${JOB_IDS}' DETAIL_JOB_IDS='${DETAIL_JOB_IDS}' REMOTE_PERTURB_SUMMARY='${REMOTE_PERTURB_SUMMARY}' REMOTE_ADAPT_SUMMARY='${REMOTE_ADAPT_SUMMARY}' bash -s" \
    <<<"${remote_poll_script}"
else
  echo "[dry-run] pass --poll to run remote Slurm and summary checks"
fi

sync_one() {
  local remote_path="$1"
  local local_path="$2"
  mkdir -p "$(dirname "${local_path}")"
  if ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "test -f '${remote_path}'"; then
    rsync -az "${REMOTE_HOST}:${remote_path}" "${local_path}"
    echo "[synced] ${remote_path} -> ${local_path}"
  else
    echo "[missing] ${remote_path}"
  fi
}

if [[ "${SYNC}" -eq 1 ]]; then
  sync_one "${REMOTE_PERTURB_SUMMARY}" "${LOCAL_PERTURB_SUMMARY}"
  sync_one "${REMOTE_ADAPT_SUMMARY}" "${LOCAL_ADAPT_SUMMARY}"
else
  echo "[dry-run] pass --sync to rsync compact summaries if present"
fi

if [[ "${CHECK_LOCAL}" -eq 1 ]]; then
  if [[ -f "${LOCAL_PERTURB_SUMMARY}" ]]; then
    PYTHONPATH=src:. python3 scripts/check_openvla_perturb_gate.py \
      --perturb-summary "${LOCAL_PERTURB_SUMMARY}"
  else
    echo "[skip] local perturb summary missing: ${LOCAL_PERTURB_SUMMARY}"
  fi
  PYTHONPATH=src:. python3 scripts/check_acceptance_readiness.py --root .
  PYTHONPATH=src:. python3 scripts/acceptance_scorecard.py --root . --out docs/acceptance_scorecard.md
else
  echo "[skip] local gates disabled by --no-check"
fi
