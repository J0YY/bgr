#!/usr/bin/env bash
set -euo pipefail

export ARTIFACT="${ARTIFACT:-openvla_oft_perturb_eval_occlusion_bottleneck_combo_occ080_shift015_scout_v1}"
export JOB_IDS="${JOB_IDS:-783312,783314,783315}"
export DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-783312,783314,783315}"
export GATE_PERTURBATIONS="${GATE_PERTURBATIONS:-occlusion_shift}"
export ROUTE_LABEL="${ROUTE_LABEL:-OpenVLA occlusion+shift combined perturbation scout}"

RUN_SCOUT_CHECK=1
base_args=()
for arg in "$@"; do
  if [[ "${arg}" == "--no-check" ]]; then
    RUN_SCOUT_CHECK=0
  else
    base_args+=("${arg}")
  fi
done

# The shared sync script's built-in checker is the 400-episode perturbation gate.
# This route is only a 100-episode scout, so suppress that gate and run the
# scout-specific checker below when requested.
scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh "${base_args[@]}" --no-check

if [[ "${RUN_SCOUT_CHECK}" -eq 1 ]]; then
  summary_path=""
  if [[ -f "${LOCAL_SUMMARY:-results/${ARTIFACT}/summary.csv}" ]]; then
    summary_path="${LOCAL_SUMMARY:-results/${ARTIFACT}/summary.csv}"
  elif [[ -f "${LOCAL_AVAILABLE_SUMMARY:-results/${ARTIFACT}/summary_available.csv}" ]]; then
    summary_path="${LOCAL_AVAILABLE_SUMMARY:-results/${ARTIFACT}/summary_available.csv}"
  fi

  if [[ -n "${summary_path}" ]]; then
    PYTHONPATH=src:. python3 scripts/check_openvla_route_scout.py "${summary_path}"
  else
    echo "[skip] local scout summary missing"
  fi
else
  echo "[skip] local scout check disabled by --no-check"
fi
