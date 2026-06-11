#!/usr/bin/env bash
set -euo pipefail

SUBMIT=0
SYNC_ARGS=(--poll --sync)

usage() {
  cat <<'USAGE'
Usage: scripts/advance_openvla_oft_occlusion_shift_combo_scout.sh [options]

Sync the 100-episode OpenVLA occlusion+shift route-selection scout, run the
scout-specific promotion checker, and optionally submit the locked full gate
only when the checker returns PROMOTE_FULL_GATE.

Options:
  --submit    submit scripts/queue_openvla_oft_occlusion_shift_combo_gate.sh --submit
              if and only if the scout checker returns PROMOTE_FULL_GATE
  --no-poll   skip remote Slurm polling and only sync/evaluate available output
  --no-sync   skip rsync/log summarization and only evaluate an existing local summary
  -h, --help  show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --no-poll)
      declare -a next=()
      for arg in "${SYNC_ARGS[@]}"; do
        [[ "${arg}" == "--poll" ]] || next+=("${arg}")
      done
      SYNC_ARGS=()
      if [[ "${#next[@]}" -gt 0 ]]; then
        SYNC_ARGS=("${next[@]}")
      fi
      shift
      ;;
    --no-sync)
      declare -a next=()
      for arg in "${SYNC_ARGS[@]}"; do
        [[ "${arg}" == "--sync" ]] || next+=("${arg}")
      done
      SYNC_ARGS=()
      if [[ "${#next[@]}" -gt 0 ]]; then
        SYNC_ARGS=("${next[@]}")
      fi
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

ARTIFACT="${ARTIFACT:-openvla_oft_perturb_eval_occlusion_bottleneck_combo_occ080_shift015_scout_v1}"
LOCAL_SUMMARY="${LOCAL_SUMMARY:-results/${ARTIFACT}/summary.csv}"
LOCAL_AVAILABLE_SUMMARY="${LOCAL_AVAILABLE_SUMMARY:-results/${ARTIFACT}/summary_available.csv}"

if [[ "${#SYNC_ARGS[@]}" -gt 0 ]]; then
  scripts/sync_openvla_oft_occlusion_shift_combo_scout_results.sh "${SYNC_ARGS[@]}"
fi

summary_path=""
if [[ -f "${LOCAL_SUMMARY}" ]]; then
  summary_path="${LOCAL_SUMMARY}"
elif [[ -f "${LOCAL_AVAILABLE_SUMMARY}" ]]; then
  summary_path="${LOCAL_AVAILABLE_SUMMARY}"
fi

if [[ -z "${summary_path}" ]]; then
  echo "[INCOMPLETE] no local scout summary exists yet for ${ARTIFACT}"
  exit 0
fi

decision_output="$(PYTHONPATH=src:. python3 scripts/check_openvla_route_scout.py "${summary_path}")"
printf '%s\n' "${decision_output}"

if grep -q '^\[PROMOTE_FULL_GATE\]' <<<"${decision_output}"; then
  if [[ "${SUBMIT}" -eq 1 ]]; then
    scripts/queue_openvla_oft_occlusion_shift_combo_gate.sh --submit
  else
    echo "[ready] scout promotes; rerun with --submit to launch the locked full gate"
  fi
elif grep -q '^\[CLOSE_NEGATIVE\]' <<<"${decision_output}"; then
  echo "[closed] scout does not justify the full gate"
else
  echo "[wait] scout summary is incomplete"
fi
