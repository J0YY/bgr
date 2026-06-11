#!/usr/bin/env bash
set -euo pipefail

OUT_PREFIX_BASE="${OUT_PREFIX_BASE:-lunarlander_recovery_probe_30seed_target070}"
LOCAL_MERGED="${LOCAL_MERGED:-results/${OUT_PREFIX_BASE}_merged}"
METHOD_JOB_IDS="${METHOD_JOB_IDS:-uniform:782561,fixed:782562,failure_only:782563,td_loss:782564,bgr_uniform_radius:782565,bgr_coverage:782566,bgr:782567}"
TREATMENT="${TREATMENT:-bgr_coverage}"

mkdir -p "${LOCAL_MERGED}"

IFS=',' read -r -a pairs <<< "${METHOD_JOB_IDS}"
available=0
expected=0
summary_files=()

for pair in "${pairs[@]}"; do
  method="${pair%%:*}"
  job_id="${pair##*:}"
  if [[ -z "${method}" || -z "${job_id}" || "${method}" == "${job_id}" ]]; then
    echo "[warn] skipping malformed METHOD_JOB_IDS entry: ${pair}" >&2
    continue
  fi
  expected=$((expected + 1))
  out_prefix="${OUT_PREFIX_BASE}_${method}_v1"
  local_out="results/${out_prefix}_${job_id}"
  echo "### sync ${method} ${job_id}"
  if ! JOB_ID="${job_id}" OUT_PREFIX="${out_prefix}" LOCAL_OUT="${local_out}" scripts/sync_lunarlander_probe.sh; then
    echo "[warn] sync failed or incomplete for ${method} ${job_id}; continuing" >&2
  fi
  if [[ -f "${local_out}/summary.csv" ]]; then
    available=$((available + 1))
    summary_files+=("${local_out}/summary.csv")
  else
    echo "[pending] missing ${local_out}/summary.csv"
  fi
done

if [[ "${available}" -eq 0 ]]; then
  echo "[pending] no method summaries available yet"
  exit 0
fi

merged_summary="${LOCAL_MERGED}/summary.csv"
{
  head -n 1 "${summary_files[0]}"
  for summary in "${summary_files[@]}"; do
    tail -n +2 "${summary}"
  done
} > "${merged_summary}"
echo "[merged] ${available}/${expected} summaries -> ${merged_summary}"

if [[ "${available}" -eq "${expected}" ]]; then
  echo "[promotion-check] complete split run"
  PYTHONPATH=src:. python3 tools/check_candidate_promotion.py "${merged_summary}" \
    --treatment "${TREATMENT}" --min-seeds 30 --min-wins 24 --min-delta 0.01 \
    2>&1 | tee "${LOCAL_MERGED}/promotion_check.txt"
else
  echo "[pending] ${available}/${expected} method summaries available; promotion check deferred"
fi
