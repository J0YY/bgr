#!/usr/bin/env bash
set -euo pipefail

export REMOTE_HOST="${REMOTE_HOST:-athena}"
export REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/work/joy/bgr/runs}"
export LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-results}"

export PREP_TAG="${PREP_TAG:-p2048unique_occlusion_bottleneck_prereg}"
export ANCHOR_TAG="${ANCHOR_TAG:-proxanchor_l2_5em0}"
export ADAPT_TAG="${ADAPT_TAG:-cleanmix_${PREP_TAG}_${ANCHOR_TAG}_step50400_lr2em7_identitylora_imageaug_officialtrainstats_v1}"
export PERTURB_TAG="${PERTURB_TAG:-cleanmix_${PREP_TAG}_${ANCHOR_TAG}_step50400_lr2em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1}"
export JOB_IDS="${JOB_IDS:-767850,767851,767852,767853,767854,767855,767856,767857,767858,767859,767860,767861,767862,767863,767864,767865,767866,767868,767878,767879,767880,767881}"
export DETAIL_JOB_IDS="${DETAIL_JOB_IDS:-767850,767851,767854,767857,767862,767868}"

exec scripts/sync_openvla_oft_perturb_only_anchor_results.sh "$@"
