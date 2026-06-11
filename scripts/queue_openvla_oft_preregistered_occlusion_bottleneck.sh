#!/usr/bin/env bash
set -euo pipefail

SUBMIT_PREP=0
SUBMIT_ADAPT=0
SUBMIT_PERTURB=0
RUN_PREP=1
RUN_ADAPT=1
RUN_PERTURB=1

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/anonymous/bgr}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/work/anonymous/bgr/logs}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/work/anonymous/bgr/runs}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/work/anonymous/cache_home/huggingface}"
REMOTE_TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE:-${REMOTE_HF_HOME}/hub}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-/work/anonymous/external_validation/openvla_oft_smoke_746850/openvla-oft}"
OPENVLA_OFT_PY="${OPENVLA_OFT_PY:-${OPENVLA_OFT_ROOT}/.venv-oft/bin/python}"
OPENVLA_OFT_SITE="${OPENVLA_OFT_SITE:-${OPENVLA_OFT_ROOT}/.venv-oft/lib/python3.10/site-packages}"
LIBERO_ROOT="${LIBERO_ROOT:-/home/anonymous/LIBERO}"
OFFICIAL_STATS="${OFFICIAL_STATS:-${REMOTE_HF_HOME}/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b/dataset_statistics.json}"
SOURCE_ARTIFACT_ROOT="${SOURCE_ARTIFACT_ROOT:-/work/anonymous/dreamaudit_jobs/artifacts}"
PERTURB_MANIFEST="${PERTURB_MANIFEST:-${REMOTE_PROJECT}/results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl}"

PREP_TAG="${PREP_TAG:-p2048unique_occlusion_bottleneck_prereg}"
ANCHOR_TAG="${ANCHOR_TAG:-proxanchor_l2_5em0}"
ADAPT_TAG="${ADAPT_TAG:-cleanmix_${PREP_TAG}_${ANCHOR_TAG}_step50400_lr2em7_identitylora_imageaug_officialtrainstats_v1}"
PERTURB_TAG="${PERTURB_TAG:-cleanmix_${PREP_TAG}_${ANCHOR_TAG}_step50400_lr2em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_perturb_v1}"
ADAPT_ARTIFACT="${ADAPT_ARTIFACT:-openvla_oft_goal_adapt_eval_${ADAPT_TAG}}"
PERTURB_ARTIFACT="${PERTURB_ARTIFACT:-openvla_oft_perturb_eval_${PERTURB_TAG}}"

BGR_CLEAN_RENDER="${BGR_CLEAN_RENDER:-${REMOTE_RUN_ROOT}/openvla_teacher_oft_bgr_clean_${PREP_TAG}_v1}"
RANDOM_CLEAN_RENDER="${RANDOM_CLEAN_RENDER:-${REMOTE_RUN_ROOT}/openvla_teacher_oft_random_clean_${PREP_TAG}_v1}"
BGR_PERTURB_RENDER="${BGR_PERTURB_RENDER:-${REMOTE_RUN_ROOT}/openvla_teacher_oft_bgr_perturb_${PREP_TAG}_v1}"
RANDOM_PERTURB_RENDER="${RANDOM_PERTURB_RENDER:-${REMOTE_RUN_ROOT}/openvla_teacher_oft_random_perturb_${PREP_TAG}_v1}"
BGR_OCCLUSION_RENDER="${BGR_OCCLUSION_RENDER:-${REMOTE_RUN_ROOT}/openvla_teacher_oft_bgr_occlusion_${PREP_TAG}_v1}"
RANDOM_OCCLUSION_RENDER="${RANDOM_OCCLUSION_RENDER:-${REMOTE_RUN_ROOT}/openvla_teacher_oft_random_occlusion_${PREP_TAG}_v1}"
BGR_MIX_RENDER="${BGR_MIX_RENDER:-${REMOTE_RUN_ROOT}/openvla_teacher_oft_bgr_clean_occlusionmix_${PREP_TAG}_v1}"
RANDOM_MIX_RENDER="${RANDOM_MIX_RENDER:-${REMOTE_RUN_ROOT}/openvla_teacher_oft_random_clean_occlusionmix_${PREP_TAG}_v1}"
BGR_DATA_ROOT="${BGR_DATA_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_tfds_libero_goal_bgr_clean_occlusionmix_${PREP_TAG}_v1}"
RANDOM_DATA_ROOT="${RANDOM_DATA_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_tfds_libero_goal_random_clean_occlusionmix_${PREP_TAG}_v1}"
CLEAN_MANIFEST_DIR="${CLEAN_MANIFEST_DIR:-${REMOTE_RUN_ROOT}/openvla_teacher_replay_manifest_clean_anchors_${PREP_TAG}_v1}"
BGR_RUN_ROOT="${BGR_RUN_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_goal_adapt_bgr_${ADAPT_TAG}}"
RANDOM_RUN_ROOT="${RANDOM_RUN_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_goal_adapt_random_${ADAPT_TAG}}"
BGR_CKPT="${BGR_CKPT:-${BGR_RUN_ROOT}/openvla-7b-oft-finetuned-libero-goal}"
RANDOM_CKPT="${RANDOM_CKPT:-${RANDOM_RUN_ROOT}/openvla-7b-oft-finetuned-libero-goal}"

MAX_CLEAN_EXAMPLES="${MAX_CLEAN_EXAMPLES:-2048}"
MAX_PERTURB_EXAMPLES="${MAX_PERTURB_EXAMPLES:-2048}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-64}"
CLEAN_EPISODES_PER_FAMILY="${CLEAN_EPISODES_PER_FAMILY:-64}"
PERTURB_EPISODES_PER_FAMILY="${PERTURB_EPISODES_PER_FAMILY:-8}"
INCLUDE_CLEAN_ANCHORS="${INCLUDE_CLEAN_ANCHORS:-1}"
OCCLUSION_CAP="${OCCLUSION_CAP:-512}"
OCCLUSION_REPEAT="${OCCLUSION_REPEAT:-4}"
OCCLUSION_FRACTION_OVERRIDE="${OCCLUSION_FRACTION_OVERRIDE:-}"
DATASET_NAME="${DATASET_NAME:-libero_goal_no_noops}"
PROXIMAL_ANCHOR_L2="${PROXIMAL_ANCHOR_L2:-5.0}"
ADAPT_STEPS="${ADAPT_STEPS:-400}"
LR="${LR:-2e-7}"
IMAGE_AUG="${IMAGE_AUG:-True}"
PREP_TIME="${PREP_TIME:-16:00:00}"
TRAIN_TIME="${TRAIN_TIME:-12:00:00}"
MERGE_TIME="${MERGE_TIME:-02:00:00}"
EVAL_TIME="${EVAL_TIME:-12:00:00}"
EVAL_TASKS="${EVAL_TASKS:-10}"
EVAL_TRIALS="${EVAL_TRIALS:-10}"
EVAL_SEED="${EVAL_SEED:-37}"
PARTITION="${PARTITION:-low-prio-gpu}"
GRES="${GRES:-gpu:a6000:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-90G}"
EXCLUDE="${EXCLUDE:-c2-g4-21,c2-g4-19}"
SYNC_CODE="${SYNC_CODE:-1}"
GIT_PULL="${GIT_PULL:-1}"
TRAIN_DEPENDENCY="${TRAIN_DEPENDENCY:-}"
ALLOW_IMMEDIATE_PERTURB="${ALLOW_IMMEDIATE_PERTURB:-0}"

usage() {
  cat <<USAGE
Usage: scripts/queue_openvla_oft_preregistered_occlusion_bottleneck.sh [options]

Dry-run by default. This preregisters a learned-policy intervention aimed at
the only non-saturated perturbation family in the fixed LIBERO-Goal visual
audit:
  - derive the bottleneck from the official-checkpoint eval family, where
    occlusion is the weak non-identity family and blur/brightness/shift are
    already near saturation;
  - train on matched clean anchors plus occlusion-only boundary examples for
    BGR-boundary and matched-random data roots;
  - cap occlusion rows at OCCLUSION_CAP=${OCCLUSION_CAP} for both methods and
    duplicate that subset OCCLUSION_REPEAT=${OCCLUSION_REPEAT} times;
  - set INCLUDE_CLEAN_ANCHORS=0 for a router-specific occlusion-only training
    premise where clean identity is handled by the official checkpoint;
  - keep official stats, identity-LoRA, image augmentation, LR=${LR},
    ADAPT_STEPS=${ADAPT_STEPS}, PROXIMAL_ANCHOR_L2=${PROXIMAL_ANCHOR_L2}, and
    fixed 10-task x 10-trial LIBERO-Goal perturbation evaluation.
  - optionally override rendered occlusion strength with
    OCCLUSION_FRACTION_OVERRIDE, e.g. 0.65 for hard-occlusion training.

Options:
  --prep-only        print or submit only clean+occlusion TFDS prep
  --adapt-only       print or submit only adaptation/merge/clean-eval
  --perturb-only     print or submit only perturbation evals
  --submit-prep      submit clean+occlusion TFDS prep
  --submit-adapt     submit adaptation chain; uses TRAIN_DEPENDENCY if set
  --submit-perturb   submit perturbation evals; requires BGR_DEPENDENCY and
                     RANDOM_DEPENDENCY unless ALLOW_IMMEDIATE_PERTURB=1
  -h, --help         show this message

Promotion gate:
  Promote this result only if occlusion-bottleneck BGR beats both matched
  random and the official checkpoint on the fixed non-identity perturbation
  total by at least 10/400 episodes and at least 0.02 absolute success rate,
  while not trailing clean identity by more than 1/100. Family-specific
  occlusion gains alone are not enough for a main claim.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prep-only) RUN_PREP=1; RUN_ADAPT=0; RUN_PERTURB=0; shift ;;
    --adapt-only) RUN_PREP=0; RUN_ADAPT=1; RUN_PERTURB=0; shift ;;
    --perturb-only) RUN_PREP=0; RUN_ADAPT=0; RUN_PERTURB=1; shift ;;
    --submit-prep) SUBMIT_PREP=1; shift ;;
    --submit-adapt) SUBMIT_ADAPT=1; shift ;;
    --submit-perturb) SUBMIT_PERTURB=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

exclude_directive() {
  if [[ -n "${EXCLUDE}" ]]; then
    printf '#SBATCH --exclude=%s\n' "${EXCLUDE}"
  fi
}

sync_code() {
  if [[ "${SYNC_CODE}" != "1" ]]; then
    return
  fi
  rsync -a \
    scripts/export_openvla_teacher_replay_manifest.py \
    scripts/render_openvla_teacher_examples.py \
    scripts/filter_openvla_oft_examples.py \
    scripts/export_openvla_oft_tfds.py \
    scripts/pack_openvla_oft_examples.py \
    scripts/combine_openvla_oft_examples.py \
    "${REMOTE_HOST}:${REMOTE_PROJECT}/scripts/"
}

print_header() {
  cat <<EOF
### Preregistered OpenVLA-OFT occlusion-bottleneck adaptation
PREP_TAG=${PREP_TAG}
ADAPT_TAG=${ADAPT_TAG}
PERTURB_TAG=${PERTURB_TAG}
BGR_DATA_ROOT=${BGR_DATA_ROOT}
RANDOM_DATA_ROOT=${RANDOM_DATA_ROOT}
OCCLUSION_CAP=${OCCLUSION_CAP}
OCCLUSION_REPEAT=${OCCLUSION_REPEAT}
OCCLUSION_FRACTION_OVERRIDE=${OCCLUSION_FRACTION_OVERRIDE}
INCLUDE_CLEAN_ANCHORS=${INCLUDE_CLEAN_ANCHORS}
PROXIMAL_ANCHOR_L2=${PROXIMAL_ANCHOR_L2}
Promotion gate: BGR must beat matched random and official by >=10/400 non-identity episodes and >=0.02, with clean identity no worse than -1/100.
EOF
}

submit_or_print_prep() {
  local script_path="$1"
  if [[ "${SUBMIT_PREP}" -eq 1 ]]; then
    sync_code
    local remote_script="/tmp/bgr-occlusion-bottleneck-prep-${PREP_TAG}.$(date +%s).sh"
    scp -q "${script_path}" "${REMOTE_HOST}:${remote_script}"
    local job_id
    job_id="$(ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_LOG_DIR} && sbatch --parsable ${remote_script}")"
    echo "prep=${job_id}"
    echo "BGR_DATA_ROOT=${BGR_DATA_ROOT}"
    echo "RANDOM_DATA_ROOT=${RANDOM_DATA_ROOT}"
  else
    sed -n '1,320p' "${script_path}"
    echo "Dry-run only. Re-run with --submit-prep to queue."
  fi
}

run_prep() {
  local script_path
  script_path="$(mktemp "${TMPDIR:-/tmp}/occlusion_bottleneck_prep.XXXXXX")"
  trap 'rm -f "${script_path}"' RETURN

  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bgr-occlusion-prep-${PREP_TAG}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${PREP_TIME}
$(exclude_directive)
#SBATCH --output=${REMOTE_LOG_DIR}/%x-%j.out

set -euo pipefail
source ~/.bashrc || true

BGR_SOURCE_DIRS=(
  "${SOURCE_ARTIFACT_ROOT}/libero_openvla_observation_proposal_balanced_expfit_seed1_lp2_h160"
  "${SOURCE_ARTIFACT_ROOT}/libero_openvla_observation_proposal_balanced_expfit_seed2_lp2_h160"
  "${SOURCE_ARTIFACT_ROOT}/libero_openvla_observation_proposal_balanced_expfit_seed3_lp2_h160"
  "${SOURCE_ARTIFACT_ROOT}/libero_openvla_observation_proposal_balanced_expfit_seed4_lp2_h160"
  "${SOURCE_ARTIFACT_ROOT}/libero_openvla_observation_proposal_balanced_expfit_seed5_lp2_h160"
)
RANDOM_SOURCE_DIRS=(
  "${SOURCE_ARTIFACT_ROOT}/libero_openvla_observation_random_balanced_seed1b_skip_lp2_h160"
  "${SOURCE_ARTIFACT_ROOT}/libero_openvla_observation_random_balanced_seed2b_skip_lp2_h160"
  "${SOURCE_ARTIFACT_ROOT}/libero_openvla_observation_random_balanced_seed3b_skip_lp2_h160"
  "${SOURCE_ARTIFACT_ROOT}/libero_openvla_observation_random_balanced_seed4b_skip_lp2_h160"
  "${SOURCE_ARTIFACT_ROOT}/libero_openvla_observation_random_balanced_seed5b_skip_lp2_h160"
)

cd "${REMOTE_PROJECT}"
mkdir -p "${CLEAN_MANIFEST_DIR}"

BGR_DIR_ARGS=()
for dir in "\${BGR_SOURCE_DIRS[@]}"; do
  BGR_DIR_ARGS+=(--bgr-dir "\${dir}")
done
RANDOM_DIR_ARGS=()
for dir in "\${RANDOM_SOURCE_DIRS[@]}"; do
  RANDOM_DIR_ARGS+=(--random-dir "\${dir}")
done

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${REMOTE_PROJECT}/src:${REMOTE_PROJECT}:${LIBERO_ROOT}:${OPENVLA_OFT_ROOT}:${OPENVLA_OFT_SITE}:\${PYTHONPATH:-}"
RENDER_OVERRIDE_ARGS=()
if [[ -n "${OCCLUSION_FRACTION_OVERRIDE}" ]]; then
  RENDER_OVERRIDE_ARGS+=(--override-perturbation-param "occlusion.fraction=${OCCLUSION_FRACTION_OVERRIDE}")
fi

if [[ "${INCLUDE_CLEAN_ANCHORS}" == "1" ]]; then
  echo "Exporting clean-anchor manifest at \$(date -Is)"
  "${OPENVLA_OFT_PY}" scripts/export_openvla_teacher_replay_manifest.py \\
    "\${BGR_DIR_ARGS[@]}" \\
    "\${RANDOM_DIR_ARGS[@]}" \\
    --native-anchors-only \\
    --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}" \\
    --out "${CLEAN_MANIFEST_DIR}"

  echo "Rendering clean BGR anchors at \$(date -Is)"
  "${OPENVLA_OFT_PY}" scripts/render_openvla_teacher_examples.py \\
    --manifest "${CLEAN_MANIFEST_DIR}/teacher_replay_manifest.jsonl" \\
    --out "${BGR_CLEAN_RENDER}" \\
    --method bgr_boundary \\
    --max-examples "${MAX_CLEAN_EXAMPLES}" \\
    --selection balanced_episodes \\
    --episodes-per-family "${CLEAN_EPISODES_PER_FAMILY}" \\
    --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}"

  echo "Rendering clean random anchors at \$(date -Is)"
  "${OPENVLA_OFT_PY}" scripts/render_openvla_teacher_examples.py \\
    --manifest "${CLEAN_MANIFEST_DIR}/teacher_replay_manifest.jsonl" \\
    --out "${RANDOM_CLEAN_RENDER}" \\
    --method random_balanced \\
    --max-examples "${MAX_CLEAN_EXAMPLES}" \\
    --selection balanced_episodes \\
    --episodes-per-family "${CLEAN_EPISODES_PER_FAMILY}" \\
    --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}"
else
  echo "Skipping clean anchors because INCLUDE_CLEAN_ANCHORS=${INCLUDE_CLEAN_ANCHORS} at \$(date -Is)"
fi

echo "Rendering BGR boundary perturbations at \$(date -Is)"
"${OPENVLA_OFT_PY}" scripts/render_openvla_teacher_examples.py \\
  --manifest "${PERTURB_MANIFEST}" \\
  --out "${BGR_PERTURB_RENDER}" \\
  --method bgr_boundary \\
  "\${RENDER_OVERRIDE_ARGS[@]}" \\
  --max-examples "${MAX_PERTURB_EXAMPLES}" \\
  --selection balanced_episodes \\
  --episodes-per-family "${PERTURB_EPISODES_PER_FAMILY}" \\
  --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}"

echo "Rendering matched-random perturbations at \$(date -Is)"
"${OPENVLA_OFT_PY}" scripts/render_openvla_teacher_examples.py \\
  --manifest "${PERTURB_MANIFEST}" \\
  --out "${RANDOM_PERTURB_RENDER}" \\
  --method random_balanced \\
  "\${RENDER_OVERRIDE_ARGS[@]}" \\
  --max-examples "${MAX_PERTURB_EXAMPLES}" \\
  --selection balanced_episodes \\
  --episodes-per-family "${PERTURB_EPISODES_PER_FAMILY}" \\
  --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}"

echo "Filtering to the occlusion bottleneck family at \$(date -Is)"
"${OPENVLA_OFT_PY}" scripts/filter_openvla_oft_examples.py \\
  --source "${BGR_PERTURB_RENDER}" \\
  --out "${BGR_OCCLUSION_RENDER}" \\
  --include-family occlusion \\
  --cap "occlusion=${OCCLUSION_CAP}"
"${OPENVLA_OFT_PY}" scripts/filter_openvla_oft_examples.py \\
  --source "${RANDOM_PERTURB_RENDER}" \\
  --out "${RANDOM_OCCLUSION_RENDER}" \\
  --include-family occlusion \\
  --cap "occlusion=${OCCLUSION_CAP}"

echo "Combining training examples at \$(date -Is)"
BGR_COMBINE_ARGS=()
RANDOM_COMBINE_ARGS=()
if [[ "${INCLUDE_CLEAN_ANCHORS}" == "1" ]]; then
  BGR_COMBINE_ARGS+=(--source clean="${BGR_CLEAN_RENDER}")
  RANDOM_COMBINE_ARGS+=(--source clean="${RANDOM_CLEAN_RENDER}")
fi
for repeat_idx in \$(seq 1 "${OCCLUSION_REPEAT}"); do
  BGR_COMBINE_ARGS+=(--source "occlusion_\${repeat_idx}=${BGR_OCCLUSION_RENDER}")
  RANDOM_COMBINE_ARGS+=(--source "occlusion_\${repeat_idx}=${RANDOM_OCCLUSION_RENDER}")
done
"${OPENVLA_OFT_PY}" scripts/combine_openvla_oft_examples.py \\
  "\${BGR_COMBINE_ARGS[@]}" \\
  --out "${BGR_MIX_RENDER}"
"${OPENVLA_OFT_PY}" scripts/combine_openvla_oft_examples.py \\
  "\${RANDOM_COMBINE_ARGS[@]}" \\
  --out "${RANDOM_MIX_RENDER}"

echo "Exporting TFDS datasets at \$(date -Is)"
"${OPENVLA_OFT_PY}" scripts/export_openvla_oft_tfds.py \\
  --examples "${BGR_MIX_RENDER}/examples.jsonl" \\
  --out "${BGR_DATA_ROOT}" \\
  --dataset-name "${DATASET_NAME}"
"${OPENVLA_OFT_PY}" scripts/export_openvla_oft_tfds.py \\
  --examples "${RANDOM_MIX_RENDER}/examples.jsonl" \\
  --out "${RANDOM_DATA_ROOT}" \\
  --dataset-name "${DATASET_NAME}"

echo "BGR_DATA_ROOT=${BGR_DATA_ROOT}"
echo "RANDOM_DATA_ROOT=${RANDOM_DATA_ROOT}"
echo "Completed occlusion-bottleneck prep at \$(date -Is)"
EOF

  submit_or_print_prep "${script_path}"
}

run_adapt() {
  local command=(scripts/queue_openvla_oft_goal_adapt.sh)
  if [[ "${SUBMIT_ADAPT}" -eq 1 ]]; then
    command+=(--submit)
  fi

  env \
    REMOTE_PROJECT="${REMOTE_PROJECT}" \
    REMOTE_LOG_DIR="${REMOTE_LOG_DIR}" \
    REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT}" \
    REMOTE_HF_HOME="${REMOTE_HF_HOME}" \
    REMOTE_TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE}" \
    OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT}" \
    LIBERO_ROOT="${LIBERO_ROOT}" \
    TRAIN_DATASET_STATISTICS_SOURCE="${OFFICIAL_STATS}" \
    DATASET_STATISTICS_SOURCE="${OFFICIAL_STATS}" \
    FINETUNE_SCRIPT="vla-scripts/finetune_identity_lora.py" \
    PROXIMAL_ANCHOR_L2="${PROXIMAL_ANCHOR_L2}" \
    ADAPT_STEPS="${ADAPT_STEPS}" \
    LR="${LR}" \
    IMAGE_AUG="${IMAGE_AUG}" \
    TRAIN_TIME="${TRAIN_TIME}" \
    MERGE_TIME="${MERGE_TIME}" \
    EVAL_TIME="${EVAL_TIME}" \
    TAG="${ADAPT_TAG}" \
    EVAL_ARTIFACT="${ADAPT_ARTIFACT}" \
    BGR_DATA_ROOT="${BGR_DATA_ROOT}" \
    RANDOM_DATA_ROOT="${RANDOM_DATA_ROOT}" \
    BGR_RUN_ROOT="${BGR_RUN_ROOT}" \
    RANDOM_RUN_ROOT="${RANDOM_RUN_ROOT}" \
    METHODS="bgr,random" \
    EVAL_TASKS="${EVAL_TASKS}" \
    EVAL_TRIALS="${EVAL_TRIALS}" \
    EVAL_SEED="${EVAL_SEED}" \
    EXCLUDE="${EXCLUDE}" \
    GIT_PULL="${GIT_PULL}" \
    TRAIN_DEPENDENCY="${TRAIN_DEPENDENCY}" \
    "${command[@]}"
}

run_perturb() {
  if [[ "${SUBMIT_PERTURB}" -eq 1 && "${ALLOW_IMMEDIATE_PERTURB}" != "1" ]]; then
    if [[ -z "${BGR_DEPENDENCY:-}" || -z "${RANDOM_DEPENDENCY:-}" ]]; then
      echo "Refusing --submit-perturb without BGR_DEPENDENCY and RANDOM_DEPENDENCY; set ALLOW_IMMEDIATE_PERTURB=1 if checkpoints already exist." >&2
      exit 2
    fi
  fi

  local command=(scripts/queue_openvla_oft_perturb_eval.sh)
  if [[ "${SUBMIT_PERTURB}" -eq 1 ]]; then
    command+=(--submit)
  fi

  env \
    REMOTE_LOG_DIR="${REMOTE_LOG_DIR}" \
    REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT}" \
    REMOTE_HF_HOME="${REMOTE_HF_HOME}" \
    REMOTE_TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE}" \
    OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT}" \
    LIBERO_ROOT="${LIBERO_ROOT}" \
    TAG="${PERTURB_TAG}" \
    EVAL_ARTIFACT="${PERTURB_ARTIFACT}" \
    BGR_CKPT="${BGR_CKPT}" \
    RANDOM_CKPT="${RANDOM_CKPT}" \
    METHODS="official,bgr,random" \
    EVAL_TASKS="${EVAL_TASKS}" \
    EVAL_TRIALS="${EVAL_TRIALS}" \
    EVAL_SEED="${EVAL_SEED}" \
    EVAL_TIME="${EVAL_TIME}" \
    EXCLUDE="${EXCLUDE}" \
    BGR_DEPENDENCY="${BGR_DEPENDENCY:-}" \
    RANDOM_DEPENDENCY="${RANDOM_DEPENDENCY:-}" \
    "${command[@]}"
}

print_header
if [[ "${RUN_PREP}" -eq 1 ]]; then
  run_prep
fi
if [[ "${RUN_ADAPT}" -eq 1 ]]; then
  run_adapt
fi
if [[ "${RUN_PERTURB}" -eq 1 ]]; then
  run_perturb
fi
