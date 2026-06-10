#!/usr/bin/env bash
set -euo pipefail

export ARTIFACT="${ARTIFACT:-openvla_oft_perturb_eval_cleanmix_p2048unique_hardocc080_identityanchor_a40_prereg_proxanchor_l2_2em1_step50200_lr1em7_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1}"
export JOB_IDS="${JOB_IDS:-776300,776301,776302,776303,776304,776305}"
export DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-776291,776292,776294,776295,776296,776297,776300,776301,776302,776303,776304,776305}"
export GATE_PERTURBATIONS="${GATE_PERTURBATIONS:-occlusion}"
export ROUTE_LABEL="${ROUTE_LABEL:-Hard-occlusion 0.80 identity-anchored OpenVLA-OFT A40 fallback}"

exec scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh "$@"
