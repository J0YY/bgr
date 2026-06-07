#!/usr/bin/env bash
set -euo pipefail

POLL=0
SYNC=0
CHECK_LOCAL=1

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/work/anonymous/bgr/runs}"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-results}"

PREP_TAG="${PREP_TAG:-p2048unique_perturbonly_anchor_prereg}"
ANCHOR_TAG="${ANCHOR_TAG:-perturbonly_proxanchor_l2_5em0}"
ADAPT_TAG="${ADAPT_TAG:-${PREP_TAG}_${ANCHOR_TAG}_step50300_lr2em7_identitylora_imageaug_officialtrainstats_v1}"
PERTURB_TAG="${PERTURB_TAG:-${PREP_TAG}_${ANCHOR_TAG}_step50300_lr2em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1}"
ADAPT_ARTIFACT="${ADAPT_ARTIFACT:-openvla_oft_goal_adapt_eval_${ADAPT_TAG}}"
PERTURB_ARTIFACT="${PERTURB_ARTIFACT:-openvla_oft_perturb_eval_${PERTURB_TAG}}"

JOB_IDS="${JOB_IDS:-767789,767790,767791,767792,767793,767794,767795,767796,767797,767798,767799,767800,767801,767802,767803,767804,767805,767806,767807,767808,767809,767810}"
DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-767789,767790,767793,767796,767801,767806}"

REMOTE_ADAPT_SUMMARY="${REMOTE_ADAPT_SUMMARY:-${REMOTE_RUN_ROOT}/${ADAPT_ARTIFACT}/summary.csv}"
REMOTE_PERTURB_SUMMARY="${REMOTE_PERTURB_SUMMARY:-${REMOTE_RUN_ROOT}/${PERTURB_ARTIFACT}/summary.csv}"
LOCAL_ADAPT_SUMMARY="${LOCAL_ADAPT_SUMMARY:-${LOCAL_RESULTS_ROOT}/${ADAPT_ARTIFACT}/summary.csv}"
LOCAL_PERTURB_SUMMARY="${LOCAL_PERTURB_SUMMARY:-${LOCAL_RESULTS_ROOT}/${PERTURB_ARTIFACT}/summary.csv}"
LOCAL_AVAILABLE_PERTURB_SUMMARY="${LOCAL_AVAILABLE_PERTURB_SUMMARY:-${LOCAL_RESULTS_ROOT}/${PERTURB_ARTIFACT}/summary_available.csv}"
REMOTE_ADAPT_LOGS="${REMOTE_ADAPT_LOGS:-${REMOTE_RUN_ROOT}/${ADAPT_ARTIFACT}/logs}"
REMOTE_PERTURB_LOGS="${REMOTE_PERTURB_LOGS:-${REMOTE_RUN_ROOT}/${PERTURB_ARTIFACT}/logs}"

usage() {
  cat <<USAGE
Usage: scripts/sync_openvla_oft_perturb_only_anchor_results.sh [options]

Dry-run by default. This helper is for the preregistered perturb-only anchored
OpenVLA-OFT route. It polls the fixed Slurm job IDs, syncs compact summaries
when present, and runs the local learned-policy gates.

Options:
  --poll          run remote squeue/sacct, selected scontrol details, and summary checks
  --sync          rsync compact summary.csv files from REMOTE_HOST if present
  --no-check      skip local gate/readiness commands after sync/dry-run output
  -h, --help      show this message

Remote summaries:
  ${REMOTE_ADAPT_SUMMARY}
  ${REMOTE_PERTURB_SUMMARY}

Remote log fallbacks:
  ${REMOTE_ADAPT_LOGS}
  ${REMOTE_PERTURB_LOGS}

Local summaries:
  adapt:                 ${LOCAL_ADAPT_SUMMARY}
  perturb complete:      ${LOCAL_PERTURB_SUMMARY}
  perturb incomplete:    ${LOCAL_AVAILABLE_PERTURB_SUMMARY}
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

echo "### Perturb-only anchored OpenVLA-OFT result sync"
echo "REMOTE_HOST=${REMOTE_HOST}"
echo "JOB_IDS=${JOB_IDS}"
echo "DETAIL_JOB_IDS=${DETAIL_JOB_IDS}"
echo "REMOTE_PERTURB_SUMMARY=${REMOTE_PERTURB_SUMMARY}"
echo "LOCAL_PERTURB_SUMMARY=${LOCAL_PERTURB_SUMMARY}"
echo "LOCAL_AVAILABLE_PERTURB_SUMMARY=${LOCAL_AVAILABLE_PERTURB_SUMMARY}"
echo "REMOTE_ADAPT_SUMMARY=${REMOTE_ADAPT_SUMMARY}"
echo "LOCAL_ADAPT_SUMMARY=${LOCAL_ADAPT_SUMMARY}"
echo "REMOTE_PERTURB_LOGS=${REMOTE_PERTURB_LOGS}"
echo "REMOTE_ADAPT_LOGS=${REMOTE_ADAPT_LOGS}"

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
    scontrol show job -dd "${job_id}" 2>/dev/null | \
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

sync_adapt_summary() {
  mkdir -p "$(dirname "${LOCAL_ADAPT_SUMMARY}")"
  if ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "test -f '${REMOTE_ADAPT_SUMMARY}'"; then
    rsync -az "${REMOTE_HOST}:${REMOTE_ADAPT_SUMMARY}" "${LOCAL_ADAPT_SUMMARY}"
    echo "[synced] ${REMOTE_ADAPT_SUMMARY} -> ${LOCAL_ADAPT_SUMMARY}"
  elif ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "test -d '${REMOTE_ADAPT_LOGS}/bgr' && test -d '${REMOTE_ADAPT_LOGS}/random'"; then
    local tmp_logs
    tmp_logs="$(mktemp -d "${TMPDIR:-/tmp}/perturbonly-adapt-logs.XXXXXX")"
    rsync -az "${REMOTE_HOST}:${REMOTE_ADAPT_LOGS}/bgr/" "${tmp_logs}/bgr/"
    rsync -az "${REMOTE_HOST}:${REMOTE_ADAPT_LOGS}/random/" "${tmp_logs}/random/"
    local tmp_out
    tmp_out="$(mktemp -d "${TMPDIR:-/tmp}/perturbonly-adapt-summary.XXXXXX")"
    PYTHONPATH=src:. python3 scripts/summarize_openvla_oft_eval.py \
      --method-log-dir "bgr=${tmp_logs}/bgr" \
      --method-log-dir "random=${tmp_logs}/random" \
      --out "${tmp_out}"
    python3 - "${tmp_out}/summary.csv" "${LOCAL_ADAPT_SUMMARY}" <<'PY'
import csv
import sys

src, dst = sys.argv[1:3]
with open(src, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
with open(dst, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["method", "episodes", "successes", "success_rate"],
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in writer.fieldnames})
PY
    echo "[summarized-from-logs] ${REMOTE_ADAPT_LOGS} -> ${LOCAL_ADAPT_SUMMARY}"
  else
    echo "[missing] ${REMOTE_ADAPT_SUMMARY}"
  fi
}

sync_perturb_summary() {
  mkdir -p "$(dirname "${LOCAL_PERTURB_SUMMARY}")"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "test -f '${REMOTE_PERTURB_SUMMARY}'"; then
    if ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "test -d '${REMOTE_PERTURB_LOGS}'"; then
      local tmp_logs
      tmp_logs="$(mktemp -d "${TMPDIR:-/tmp}/perturbonly-perturb-logs.XXXXXX")"
      rsync -az "${REMOTE_HOST}:${REMOTE_PERTURB_LOGS}/" "${tmp_logs}/"
      local tmp_out
      tmp_out="$(mktemp -d "${TMPDIR:-/tmp}/perturbonly-perturb-summary.XXXXXX")"
      PYTHONPATH=src:. python3 scripts/summarize_openvla_oft_perturb_eval.py \
        --logs-root "${tmp_logs}" \
        --out "${tmp_out}"
      local tmp_path="${tmp_out}/summary_stripped.csv"
      python3 - "${tmp_out}/summary.csv" "${tmp_path}" <<'PY'
import csv
import sys

src, dst = sys.argv[1:3]
fieldnames = ["method", "perturbation", "episodes", "successes", "success_rate"]
with open(src, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
with open(dst, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in fieldnames})
PY
      if PYTHONPATH=src:. python3 scripts/check_openvla_perturb_gate.py \
          --perturb-summary "${tmp_path}" --require-complete >/dev/null 2>&1; then
        cp "${tmp_path}" "${LOCAL_PERTURB_SUMMARY}"
        echo "[summarized-complete-from-logs] ${REMOTE_PERTURB_LOGS} -> ${LOCAL_PERTURB_SUMMARY}"
      else
        mkdir -p "$(dirname "${LOCAL_AVAILABLE_PERTURB_SUMMARY}")"
        cp "${tmp_path}" "${LOCAL_AVAILABLE_PERTURB_SUMMARY}"
        echo "[summarized-incomplete-from-logs] ${REMOTE_PERTURB_LOGS} -> ${LOCAL_AVAILABLE_PERTURB_SUMMARY}"
      fi
      return
    else
      echo "[missing] ${REMOTE_PERTURB_SUMMARY}"
      return
    fi
  fi

  local tmp_path
  tmp_path="$(mktemp "$(dirname "${LOCAL_PERTURB_SUMMARY}")/.perturbonly-summary.XXXXXX.csv")"
  rsync -az "${REMOTE_HOST}:${REMOTE_PERTURB_SUMMARY}" "${tmp_path}"
  if PYTHONPATH=src:. python3 scripts/check_openvla_perturb_gate.py \
      --perturb-summary "${tmp_path}" --require-complete >/dev/null 2>&1; then
    mv "${tmp_path}" "${LOCAL_PERTURB_SUMMARY}"
    echo "[synced-complete] ${REMOTE_PERTURB_SUMMARY} -> ${LOCAL_PERTURB_SUMMARY}"
  else
    mkdir -p "$(dirname "${LOCAL_AVAILABLE_PERTURB_SUMMARY}")"
    mv "${tmp_path}" "${LOCAL_AVAILABLE_PERTURB_SUMMARY}"
    echo "[synced-incomplete] ${REMOTE_PERTURB_SUMMARY} -> ${LOCAL_AVAILABLE_PERTURB_SUMMARY}"
  fi
}

if [[ "${SYNC}" -eq 1 ]]; then
  sync_perturb_summary
  sync_adapt_summary
else
  echo "[dry-run] pass --sync to rsync compact summaries if present"
fi

if [[ "${CHECK_LOCAL}" -eq 1 ]]; then
  if [[ -f "${LOCAL_PERTURB_SUMMARY}" ]]; then
    PYTHONPATH=src:. python3 scripts/check_openvla_perturb_gate.py \
      --perturb-summary "${LOCAL_PERTURB_SUMMARY}"
  elif [[ -f "${LOCAL_AVAILABLE_PERTURB_SUMMARY}" ]]; then
    PYTHONPATH=src:. python3 scripts/check_openvla_perturb_gate.py \
      --perturb-summary "${LOCAL_AVAILABLE_PERTURB_SUMMARY}"
  else
    echo "[skip] local perturb-only perturb summary missing"
  fi
  PYTHONPATH=src:. python3 scripts/check_acceptance_readiness.py --root .
  PYTHONPATH=src:. python3 scripts/acceptance_scorecard.py --root . --out docs/acceptance_scorecard.md
else
  echo "[skip] local gates disabled by --no-check"
fi
