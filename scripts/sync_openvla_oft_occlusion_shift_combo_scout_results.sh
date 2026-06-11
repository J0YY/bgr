#!/usr/bin/env bash
set -euo pipefail

export ARTIFACT="${ARTIFACT:-openvla_oft_perturb_eval_occlusion_bottleneck_combo_occ080_shift015_scout_v1}"
export JOB_IDS="${JOB_IDS:-783312,783314,783315}"
export DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-783312,783314,783315}"
export GATE_PERTURBATIONS="${GATE_PERTURBATIONS:-occlusion_shift}"
export ROUTE_LABEL="${ROUTE_LABEL:-OpenVLA occlusion+shift combined perturbation scout}"

exec scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh "$@"
