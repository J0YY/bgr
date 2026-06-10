#!/usr/bin/env bash
set -euo pipefail

export ARTIFACT="${ARTIFACT:-openvla_oft_perturb_eval_cleanmix_p2048unique_hardocc080_identityanchor_micro_prereg_proxanchor_l2_1em2_step50050_lr5em8_identitylora_imageaug_officialtrainstats_hardocc080_fullgoal10x40_v1}"
export JOB_IDS="${JOB_IDS:-777037,777039,777040,777041,777042,777043}"
export DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-776998,777000,777001,777003,777004,777006,777037,777039,777040,777041,777042,777043}"
export GATE_PERTURBATIONS="${GATE_PERTURBATIONS:-occlusion}"
export ROUTE_LABEL="${ROUTE_LABEL:-Hard-occlusion 0.80 micro identity-anchored OpenVLA-OFT adaptation}"

exec scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh "$@"
