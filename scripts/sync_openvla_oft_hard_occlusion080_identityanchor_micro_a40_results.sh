#!/usr/bin/env bash
set -euo pipefail

export ARTIFACT="${ARTIFACT:-openvla_oft_perturb_eval_cleanmix_p2048unique_hardocc080_identityanchor_micro_a40_prereg_proxanchor_l2_1em2_step50050_lr5em8_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1}"
export JOB_IDS="${JOB_IDS:-777264,777265,777266,777268,777269,777270}"
export DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-777254,777255,777256,777257,777259,777261,777264,777265,777266,777268,777269,777270}"
export GATE_PERTURBATIONS="${GATE_PERTURBATIONS:-occlusion}"
export ROUTE_LABEL="${ROUTE_LABEL:-Hard-occlusion 0.80 micro identity-anchored OpenVLA-OFT A40 fallback}"

exec scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh "$@"
