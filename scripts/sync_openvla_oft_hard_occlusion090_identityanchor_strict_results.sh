#!/usr/bin/env bash
set -euo pipefail

export ARTIFACT="${ARTIFACT:-openvla_oft_perturb_eval_cleanmix_p2048unique_hardocc090_identityanchor_strict_prereg_proxanchor_l2_5em1_step50100_lr5em8_identitylora_imageaug_officialtrainstats_hardocc090_fullgoal10x40_v1}"
export JOB_IDS="${JOB_IDS:-776611,776613,776615,776616,776617,776619}"
export DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-776601,776602,776603,776604,776605,776606,776607,776611,776613,776615,776616,776617,776619}"
export GATE_PERTURBATIONS="${GATE_PERTURBATIONS:-occlusion}"
export ROUTE_LABEL="${ROUTE_LABEL:-Hard-occlusion 0.90 strict identity-anchored OpenVLA-OFT adaptation}"

exec scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh "$@"
