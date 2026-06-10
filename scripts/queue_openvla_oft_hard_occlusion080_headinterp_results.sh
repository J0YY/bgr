#!/usr/bin/env bash
set -euo pipefail

SUBMIT=0

REMOTE_HOST="${REMOTE_HOST:-athena}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/work/joy/bgr}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/work/joy/bgr/logs}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/work/joy/bgr/runs}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/work/joy/cache_home/huggingface}"
REMOTE_TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE:-${REMOTE_HF_HOME}/hub}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft}"
OPENVLA_OFT_PY="${OPENVLA_OFT_PY:-${OPENVLA_OFT_ROOT}/.venv-oft/bin/python}"
OPENVLA_OFT_SITE="${OPENVLA_OFT_SITE:-${OPENVLA_OFT_ROOT}/.venv-oft/lib/python3.10/site-packages}"
LIBERO_ROOT="${LIBERO_ROOT:-/work/joy/external_validation/openvla_oft_smoke_746850/LIBERO}"
PARTITION="${PARTITION:-low-prio-gpu}"
GRES="${GRES:-gpu:a6000:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-90G}"
PREP_TIME="${PREP_TIME:-04:00:00}"
EVAL_TIME="${EVAL_TIME:-12:00:00}"
EXCLUDE="${EXCLUDE:-c2-g4-21,c2-g4-19}"

ALPHA="${ALPHA:-0.75}"
LORA_B_SCALE="${LORA_B_SCALE:-${ALPHA}}"
TAG="${TAG:-occlusion_bottleneck_hardocc080_transfer_headinterp075_v1}"
EVAL_ARTIFACT="${EVAL_ARTIFACT:-openvla_oft_perturb_eval_${TAG}}"
INTERP_ROOT="${INTERP_ROOT:-${REMOTE_RUN_ROOT}/openvla_oft_headinterp_${TAG}}"
OFFICIAL_CKPT_DIR="${OFFICIAL_CKPT_DIR:-${REMOTE_HF_HOME}/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b}"
BGR_SOURCE_CKPT="${BGR_SOURCE_CKPT:-${REMOTE_RUN_ROOT}/openvla_oft_goal_adapt_bgr_cleanmix_p2048unique_occlusion_bottleneck_prereg_proxanchor_l2_5em0_step50400_lr2em7_identitylora_imageaug_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal}"
RANDOM_SOURCE_CKPT="${RANDOM_SOURCE_CKPT:-${REMOTE_RUN_ROOT}/openvla_oft_goal_adapt_random_cleanmix_p2048unique_occlusion_bottleneck_prereg_proxanchor_l2_5em0_step50400_lr2em7_identitylora_imageaug_officialtrainstats_v1/openvla-7b-oft-finetuned-libero-goal}"
BGR_INTERP_CKPT="${BGR_INTERP_CKPT:-${INTERP_ROOT}/bgr_alpha${ALPHA}/openvla-7b-oft-finetuned-libero-goal}"
RANDOM_INTERP_CKPT="${RANDOM_INTERP_CKPT:-${INTERP_ROOT}/random_alpha${ALPHA}/openvla-7b-oft-finetuned-libero-goal}"

EVAL_TASKS="${EVAL_TASKS:-10}"
EVAL_TRIALS="${EVAL_TRIALS:-40}"
EVAL_SEED="${EVAL_SEED:-37}"
if [[ -z "${PERTURBATIONS:-}" ]]; then
  PERTURBATIONS='identity={};occlusion={"fraction":0.80}'
fi

usage() {
  cat <<USAGE
Usage: scripts/queue_openvla_oft_hard_occlusion080_headinterp_results.sh [--submit]

Queues a fixed hard-occlusion 0.80 checkpoint-interpolation diagnostic.
It copies the completed occlusion-bottleneck BGR and matched-random
checkpoints, moves trainable heads toward the official checkpoint by
alpha=${ALPHA}, scales LoRA B matrices by the same alpha, then evaluates
official, interpolated BGR, and interpolated random on identity plus
occlusion fraction 0.80 over 10 LIBERO-Goal tasks x 40 trials.

Set LORA_B_SCALE separately to preserve or shrink the adapted LoRA-B tensors
independently of the head interpolation alpha.

This is not a relaxed gate. Promotion requires interpolated BGR to beat both
official and interpolated matched-random by >=10/400 occlusion episodes and
>=0.02 absolute success rate while trailing the best identity comparator by no
more than one episode.

Default mode is dry-run. Pass --submit to queue the prep and eval jobs.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

exclude_directive() {
  if [[ -n "${EXCLUDE}" ]]; then
    printf '#SBATCH --exclude=%s\n' "${EXCLUDE}"
  fi
}

submit_script() {
  local name="$1"
  local dependency="$2"
  local script_path="$3"
  local remote_script="/tmp/${name}.$(date +%s).sh"

  if [[ "${SUBMIT}" -eq 1 ]]; then
    scp -q "${script_path}" "${REMOTE_HOST}:${remote_script}"
    if [[ -n "${dependency}" ]]; then
      ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_LOG_DIR} && sbatch --parsable --dependency=${dependency} ${remote_script}"
    else
      ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_LOG_DIR} && sbatch --parsable ${remote_script}"
    fi
  else
    echo "### ${name}${dependency:+ dependency=${dependency}}"
    sed -n '1,260p' "${script_path}"
  fi
}

write_prep_script() {
  local script_path="$1"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bgr-headinterp-${TAG}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${PREP_TIME}
$(exclude_directive)
#SBATCH --output=${REMOTE_LOG_DIR}/%x-%j.out

set -euo pipefail
source ~/.bashrc || true

echo "Preparing alpha=${ALPHA}, lora_b_scale=${LORA_B_SCALE} interpolated hard-occlusion checkpoints on \$(hostname) at \$(date -Is)"
rm -rf "${BGR_INTERP_CKPT}" "${RANDOM_INTERP_CKPT}"
mkdir -p "\$(dirname "${BGR_INTERP_CKPT}")" "\$(dirname "${RANDOM_INTERP_CKPT}")"
cp -al "${BGR_SOURCE_CKPT}" "${BGR_INTERP_CKPT}"
cp -al "${RANDOM_SOURCE_CKPT}" "${RANDOM_INTERP_CKPT}"

env ALPHA="${ALPHA}" \\
  LORA_B_SCALE="${LORA_B_SCALE}" \\
  OFFICIAL_CKPT_DIR="${OFFICIAL_CKPT_DIR}" \\
  BGR_SOURCE_CKPT="${BGR_SOURCE_CKPT}" \\
  RANDOM_SOURCE_CKPT="${RANDOM_SOURCE_CKPT}" \\
  BGR_INTERP_CKPT="${BGR_INTERP_CKPT}" \\
  RANDOM_INTERP_CKPT="${RANDOM_INTERP_CKPT}" \\
  "${OPENVLA_OFT_PY}" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


alpha = float(os.environ["ALPHA"])
lora_b_scale = float(os.environ["LORA_B_SCALE"])
official = Path(os.environ["OFFICIAL_CKPT_DIR"])
pairs = [
    (Path(os.environ["BGR_SOURCE_CKPT"]), Path(os.environ["BGR_INTERP_CKPT"])),
    (Path(os.environ["RANDOM_SOURCE_CKPT"]), Path(os.environ["RANDOM_INTERP_CKPT"])),
]


def interpolate_pt(official_path: Path, adapted_path: Path, out_path: Path) -> None:
    official_state = torch.load(official_path, map_location="cpu")
    adapted_state = torch.load(adapted_path, map_location="cpu")
    out_state = type(adapted_state)()
    for key, adapted_value in adapted_state.items():
        official_value = official_state.get(key)
        if (
            official_value is not None
            and torch.is_tensor(adapted_value)
            and torch.is_tensor(official_value)
            and adapted_value.shape == official_value.shape
            and adapted_value.is_floating_point()
        ):
            mixed = official_value.float() + alpha * (adapted_value.float() - official_value.float())
            out_state[key] = mixed.to(dtype=adapted_value.dtype)
        else:
            out_state[key] = adapted_value
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(out_state, tmp_path)
    tmp_path.replace(out_path)


def scale_adapter(adapter_path: Path) -> None:
    if not adapter_path.exists():
        return
    state = load_file(str(adapter_path))
    out = {}
    lora_b_count = 0
    for key, value in state.items():
        if torch.is_tensor(value) and value.is_floating_point() and "lora_B" in key:
            out[key] = (value.float() * lora_b_scale).to(dtype=value.dtype)
            lora_b_count += 1
        else:
            out[key] = value
    tmp_path = adapter_path.with_suffix(adapter_path.suffix + ".tmp")
    save_file(out, str(tmp_path))
    tmp_path.replace(adapter_path)
    print(f"scaled {lora_b_count} LoRA-B tensors by {lora_b_scale} in {adapter_path}", flush=True)


for source, out_dir in pairs:
    interpolate_pt(
        official / "action_head--50000_checkpoint.pt",
        source / "action_head--latest_checkpoint.pt",
        out_dir / "action_head--latest_checkpoint.pt",
    )
    interpolate_pt(
        official / "proprio_projector--50000_checkpoint.pt",
        source / "proprio_projector--latest_checkpoint.pt",
        out_dir / "proprio_projector--latest_checkpoint.pt",
    )
    scale_adapter(out_dir / "lora_adapter" / "adapter_model.safetensors")
    marker = out_dir / "bgr_headinterp_manifest.json"
    marker.write_text(
        "{\\n"
        f'  "alpha": {alpha},\\n'
        f'  "lora_b_scale": {lora_b_scale},\\n'
        f'  "source_checkpoint": "{source}",\\n'
        f'  "official_checkpoint": "{official}",\\n'
        '  "interpolated": ["action_head", "proprio_projector", "lora_B"]\\n'
        "}\\n",
        encoding="utf-8",
    )
    print(f"prepared {out_dir}", flush=True)
PY
EOF
}

prep_script="$(mktemp "${TMPDIR:-/tmp}/bgr_headinterp_prep.XXXXXX")"
trap 'rm -f "${prep_script}"' EXIT
write_prep_script "${prep_script}"

echo "### Hard-occlusion 0.80 head interpolation"
echo "TAG=${TAG}"
echo "ALPHA=${ALPHA}"
echo "LORA_B_SCALE=${LORA_B_SCALE}"
echo "EVAL_ARTIFACT=${EVAL_ARTIFACT}"
echo "BGR_INTERP_CKPT=${BGR_INTERP_CKPT}"
echo "RANDOM_INTERP_CKPT=${RANDOM_INTERP_CKPT}"

if [[ "${SUBMIT}" -eq 1 ]]; then
  prep_job="$(submit_script "bgr-headinterp-${TAG}" "" "${prep_script}")"
  echo "prep=${prep_job}"
  eval_dependency="afterok:${prep_job}"
else
  submit_script "bgr-headinterp-${TAG}" "" "${prep_script}"
  echo "prep=<prep_job>"
  eval_dependency="afterok:PREP_JOB"
fi

run_eval_queue() {
  OFFICIAL_CKPT="${OFFICIAL_CKPT_DIR}" \
  BGR_CKPT="${BGR_INTERP_CKPT}" \
  RANDOM_CKPT="${RANDOM_INTERP_CKPT}" \
  TAG="${TAG}" \
  EVAL_ARTIFACT="${EVAL_ARTIFACT}" \
  REMOTE_HOST="${REMOTE_HOST}" \
  REMOTE_LOG_DIR="${REMOTE_LOG_DIR}" \
  REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT}" \
  REMOTE_HF_HOME="${REMOTE_HF_HOME}" \
  REMOTE_TRANSFORMERS_CACHE="${REMOTE_TRANSFORMERS_CACHE}" \
  OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT}" \
  OPENVLA_OFT_PY="${OPENVLA_OFT_PY}" \
  OPENVLA_OFT_SITE="${OPENVLA_OFT_SITE}" \
  LIBERO_ROOT="${LIBERO_ROOT}" \
  PARTITION="${PARTITION}" \
  GRES="${GRES}" \
  CPUS="${CPUS}" \
  MEM="${MEM}" \
  EVAL_TIME="${EVAL_TIME}" \
  EXCLUDE="${EXCLUDE}" \
  PERTURBATIONS="${PERTURBATIONS}" \
  EVAL_TASKS="${EVAL_TASKS}" \
  EVAL_TRIALS="${EVAL_TRIALS}" \
  EVAL_SEED="${EVAL_SEED}" \
  OFFICIAL_DEPENDENCY="" \
  BGR_DEPENDENCY="${eval_dependency}" \
  RANDOM_DEPENDENCY="${eval_dependency}" \
  scripts/queue_openvla_oft_perturb_eval.sh "$@"
}

if [[ "${SUBMIT}" -eq 1 ]]; then
  run_eval_queue --submit
else
  run_eval_queue
fi
