#!/usr/bin/env bash
set -euo pipefail

export ARTIFACT="${ARTIFACT:-openvla_oft_perturb_eval_cleanmix_p2048unique_hardocc080_identityanchor_strict_prereg_proxanchor_l2_5em1_step50100_lr5em8_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1}"
export JOB_IDS="${JOB_IDS:-776548,776549,776550,776551,776553,776554}"
export DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-776541,776542,776543,776544,776545,776546,776548,776549,776550,776551,776553,776554}"
export GATE_PERTURBATIONS="${GATE_PERTURBATIONS:-occlusion}"
export ROUTE_LABEL="${ROUTE_LABEL:-Hard-occlusion 0.80 strict identity-anchored OpenVLA-OFT adaptation}"

exec scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh "$@"
