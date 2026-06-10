#!/usr/bin/env bash
set -euo pipefail

export ARTIFACT="${ARTIFACT:-openvla_oft_perturb_eval_hardocc065_adapt_step50400_lr2em7_v1}"
export JOB_IDS="${JOB_IDS:-774717,774718,774719,774720,774721,774722,774723,774724,774725,774726,774727,774728,774729}"
export DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-774717,774718,774721,774724,774726,774728}"
export GATE_PERTURBATIONS="${GATE_PERTURBATIONS:-occlusion}"
export ROUTE_LABEL="${ROUTE_LABEL:-Hard-occlusion adapted OpenVLA-OFT}"

exec scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh "$@"
