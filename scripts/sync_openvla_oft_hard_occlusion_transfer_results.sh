#!/usr/bin/env bash
set -euo pipefail

POLL=0
SYNC=0
CHECK_LOCAL=1

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/work/joy/bgr/runs}"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-results}"
ROUTE_LABEL="${ROUTE_LABEL:-Hard-occlusion OpenVLA-OFT transfer}"

ARTIFACT="${ARTIFACT:-openvla_oft_perturb_eval_occlusion_bottleneck_hardocc065_transfer_step50400_lr2em7_v1}"
JOB_IDS="${JOB_IDS:-774711,774712,774713,774714,774715,774716}"
DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-774711,774712,774713,774714,774715,774716}"
GATE_PERTURBATIONS="${GATE_PERTURBATIONS:-occlusion}"

REMOTE_LOGS="${REMOTE_LOGS:-${REMOTE_RUN_ROOT}/${ARTIFACT}/logs}"
REMOTE_SUMMARY="${REMOTE_SUMMARY:-${REMOTE_RUN_ROOT}/${ARTIFACT}/summary.csv}"
LOCAL_SUMMARY="${LOCAL_SUMMARY:-${LOCAL_RESULTS_ROOT}/${ARTIFACT}/summary.csv}"
LOCAL_AVAILABLE_SUMMARY="${LOCAL_AVAILABLE_SUMMARY:-${LOCAL_RESULTS_ROOT}/${ARTIFACT}/summary_available.csv}"

usage() {
  cat <<USAGE
Usage: scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh [options]

Polls and syncs the hard-occlusion transfer diagnostic for the completed
OpenVLA-OFT occlusion-bottleneck checkpoints. The fixed gate is occlusion-only:
BGR must beat both official and matched-random on 400 occlusion episodes by at
least 10 episodes and 0.02 success rate while preserving identity competence.

Options:
  --poll      run remote squeue/sacct, selected scontrol details, and summary checks
  --sync      rsync remote logs and build local compact summaries when possible
  --no-check  skip local gate/readiness commands after sync/dry-run output
  -h, --help  show this message

Remote logs:
  ${REMOTE_LOGS}

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

echo "### ${ROUTE_LABEL} sync"
echo "REMOTE_HOST=${REMOTE_HOST}"
echo "JOB_IDS=${JOB_IDS}"
echo "DETAIL_JOB_IDS=${DETAIL_JOB_IDS}"
echo "REMOTE_LOGS=${REMOTE_LOGS}"
echo "REMOTE_SUMMARY=${REMOTE_SUMMARY}"
echo "LOCAL_SUMMARY=${LOCAL_SUMMARY}"
echo "LOCAL_AVAILABLE_SUMMARY=${LOCAL_AVAILABLE_SUMMARY}"
echo "GATE_PERTURBATIONS=${GATE_PERTURBATIONS}"

remote_poll_script=$(cat <<'REMOTE'
date
squeue -j "${JOB_IDS}" -o '%i %T %M %R' || true
sacct -j "${JOB_IDS}" --format=JobID,JobName%80,State,ExitCode,Elapsed,Start,End -P || true
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
if [ -f "${REMOTE_SUMMARY}" ]; then
  echo "FOUND:${REMOTE_SUMMARY}"
  wc -l "${REMOTE_SUMMARY}"
  head -20 "${REMOTE_SUMMARY}"
else
  echo "MISSING:${REMOTE_SUMMARY}"
fi
if [ -d "${REMOTE_LOGS}" ]; then
  echo "FOUND_LOGS:${REMOTE_LOGS}"
  find "${REMOTE_LOGS}" -maxdepth 3 -type f | sed -n '1,30p'
else
  echo "MISSING_LOGS:${REMOTE_LOGS}"
fi
REMOTE
)

if [[ "${POLL}" -eq 1 ]]; then
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "JOB_IDS='${JOB_IDS}' DETAIL_JOB_IDS='${DETAIL_JOB_IDS}' REMOTE_SUMMARY='${REMOTE_SUMMARY}' REMOTE_LOGS='${REMOTE_LOGS}' bash -s" \
    <<<"${remote_poll_script}"
else
  echo "[dry-run] pass --poll to run remote Slurm and summary checks"
fi

strip_summary() {
  local src="$1"
  local dst="$2"
  python3 - "${src}" "${dst}" <<'PY'
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
}

sync_summary() {
  mkdir -p "$(dirname "${LOCAL_SUMMARY}")"

  local tmp_path
  tmp_path="$(mktemp "$(dirname "${LOCAL_SUMMARY}")/.hardocc-summary.XXXXXX.csv")"
  if ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "test -f '${REMOTE_SUMMARY}'"; then
    rsync -az "${REMOTE_HOST}:${REMOTE_SUMMARY}" "${tmp_path}"
  elif ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "test -d '${REMOTE_LOGS}'"; then
    local tmp_logs
    tmp_logs="$(mktemp -d "${TMPDIR:-/tmp}/hardocc-openvla-logs.XXXXXX")"
    rsync -az "${REMOTE_HOST}:${REMOTE_LOGS}/" "${tmp_logs}/"
    local tmp_out
    tmp_out="$(mktemp -d "${TMPDIR:-/tmp}/hardocc-openvla-summary.XXXXXX")"
    if ! PYTHONPATH=src:. python3 scripts/summarize_openvla_oft_perturb_eval.py \
        --logs-root "${tmp_logs}" \
        --out "${tmp_out}"; then
      echo "[pending] ${REMOTE_LOGS} exists but is not summarizable yet"
      rm -f "${tmp_path}"
      return
    fi
    strip_summary "${tmp_out}/summary.csv" "${tmp_path}"
  else
    echo "[missing] ${REMOTE_SUMMARY} and ${REMOTE_LOGS}"
    rm -f "${tmp_path}"
    return
  fi

  if PYTHONPATH=src:. python3 scripts/check_openvla_perturb_gate.py \
      --perturb-summary "${tmp_path}" \
      --non-identity-perturbations "${GATE_PERTURBATIONS}" \
      --require-complete >/dev/null 2>&1; then
    mv "${tmp_path}" "${LOCAL_SUMMARY}"
    echo "[synced-complete] -> ${LOCAL_SUMMARY}"
  else
    mkdir -p "$(dirname "${LOCAL_AVAILABLE_SUMMARY}")"
    mv "${tmp_path}" "${LOCAL_AVAILABLE_SUMMARY}"
    echo "[synced-incomplete] -> ${LOCAL_AVAILABLE_SUMMARY}"
  fi
}

if [[ "${SYNC}" -eq 1 ]]; then
  sync_summary
else
  echo "[dry-run] pass --sync to rsync logs and build compact summaries"
fi

if [[ "${CHECK_LOCAL}" -eq 1 ]]; then
  if [[ -f "${LOCAL_SUMMARY}" ]]; then
    PYTHONPATH=src:. python3 scripts/check_openvla_perturb_gate.py \
      --perturb-summary "${LOCAL_SUMMARY}" \
      --non-identity-perturbations "${GATE_PERTURBATIONS}"
  elif [[ -f "${LOCAL_AVAILABLE_SUMMARY}" ]]; then
    PYTHONPATH=src:. python3 scripts/check_openvla_perturb_gate.py \
      --perturb-summary "${LOCAL_AVAILABLE_SUMMARY}" \
      --non-identity-perturbations "${GATE_PERTURBATIONS}"
  else
    echo "[skip] local hard-occlusion summary missing"
  fi
  PYTHONPATH=src:. python3 scripts/check_acceptance_readiness.py --root .
  PYTHONPATH=src:. python3 scripts/acceptance_scorecard.py --root . --out docs/acceptance_scorecard.md
else
  echo "[skip] local gates disabled by --no-check"
fi
