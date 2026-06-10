#!/usr/bin/env bash
set -euo pipefail

export ARTIFACT="${ARTIFACT:-openvla_oft_perturb_eval_occlusion_bottleneck_hardocc080_transfer_step50400_lr2em7_v1}"
export JOB_IDS="${JOB_IDS:-774917,774919,774920,774921,774922,774923}"
export DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-774917,774919,774920,774921,774922,774923}"
export GATE_PERTURBATIONS="${GATE_PERTURBATIONS:-occlusion}"
export ROUTE_LABEL="${ROUTE_LABEL:-Hard-occlusion 0.80 OpenVLA-OFT transfer}"

exec scripts/sync_openvla_oft_hard_occlusion_transfer_results.sh "$@"
