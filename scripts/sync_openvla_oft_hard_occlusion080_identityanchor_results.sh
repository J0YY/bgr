#!/usr/bin/env bash
set -euo pipefail

export ARTIFACT="${ARTIFACT:-openvla_oft_perturb_eval_cleanmix_p2048unique_hardocc080_identityanchor_prereg_proxanchor_l2_2em1_step50200_lr1em7_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1}"
export JOB_IDS="${JOB_IDS:-776040,776041,776042,776043,776044,776045}"
export DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-776029,776033,776034,776035,776036,776037,776038,776040,776041,776042,776043,776044,776045}"
export GATE_PERTURBATIONS="${GATE_PERTURBATIONS:-occlusion}"
export ROUTE_LABEL="${ROUTE_LABEL:-Hard-occlusion 0.80 identity-anchored OpenVLA-OFT adaptation}"

exec scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh "$@"
