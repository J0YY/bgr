#!/usr/bin/env bash
set -euo pipefail

export ARTIFACT="${ARTIFACT:-openvla_oft_perturb_eval_hardocc065_a40_adapt_step50400_lr2em7_v1}"
export JOB_IDS="${JOB_IDS:-774816,774817,774818,774819,774820,774821,774822,774846,774847,774848,774849,774850,774851}"
export DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-774816,774817,774820,774846,774848,774850}"
export GATE_PERTURBATIONS="${GATE_PERTURBATIONS:-occlusion}"
export ROUTE_LABEL="${ROUTE_LABEL:-Hard-occlusion adapted OpenVLA-OFT A40 fallback}"

exec scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh "$@"
