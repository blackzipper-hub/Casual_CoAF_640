#!/usr/bin/env bash
# Inference for 640x480 v6 depth+track+RGB checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/common_env.sh
source "${SCRIPT_DIR}/../scripts/common_env.sh"

CHECKPOINT_BASE="${CHECKPOINT_BASE:-${CASUAL_ROOT}/outputs/checkpoints/i2av/v6_depth_track_rgb_640x480_stage3_no_s0}"
CHECKPOINTS="${CHECKPOINTS:-${CHECKPOINT_STEP:-}}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-8}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1}"
TRACK_MAX_TIMESTEP="${TRACK_MAX_TIMESTEP:-499}"
MODEL_NAME="${MODEL_NAME:-v6_depth_track_rgb_640x480_stage3_no_s0}"

if [[ -n "${CHECKPOINTS}" ]]; then
  if [[ "${CHECKPOINTS}" = /* ]]; then
    CHECKPOINT_DIR="${CHECKPOINTS}"
  else
    CHECKPOINT_DIR="${CHECKPOINT_BASE}/checkpoint-${CHECKPOINTS#checkpoint-}"
  fi
else
  LATEST_CHECKPOINT_STEP=-1
  LATEST_CHECKPOINT_DIR=""
  for path in "${CHECKPOINT_BASE}"/checkpoint-*; do
    [[ -d "${path}" ]] || continue
    step="${path##*-}"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( step > LATEST_CHECKPOINT_STEP )); then
      LATEST_CHECKPOINT_STEP="${step}"
      LATEST_CHECKPOINT_DIR="${path}"
    fi
  done
  if [[ -z "${LATEST_CHECKPOINT_DIR}" ]]; then
    echo "No checkpoint-* directories found under ${CHECKPOINT_BASE}" >&2
    exit 1
  fi
  CHECKPOINT_DIR="${LATEST_CHECKPOINT_DIR}"
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Expected checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
CHECKPOINT_TAG="$(basename "${CHECKPOINT_DIR}")"

DATA_ROOT="${DATA_ROOT:-${FULL_640_DATA_ROOT}}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-${DATA_ROOT}}"
INFER_ROOT="${INFER_ROOT:-${CASUAL_ROOT}/outputs/infer/i2av}"
OUTPUT_DIR="${OUTPUT_DIR:-${INFER_ROOT}/${MODEL_NAME}_${CHECKPOINT_TAG}_seed${SEED}_video_640x480_tmax${TRACK_MAX_TIMESTEP}}"
LOG="${CASUAL_ROOT}/logs/infer/i2av/${MODEL_NAME}_${CHECKPOINT_TAG}_seed${SEED}_video_640x480_tmax${TRACK_MAX_TIMESTEP}_$(date +%Y%m%d_%H%M%S).log"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FREE_MODEL_BEFORE_ENCODE="${FREE_MODEL_BEFORE_ENCODE:-1}"
export FREE_MODEL_BEFORE_DECODE="${FREE_MODEL_BEFORE_DECODE:-1}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${INFER_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONPATH="${CASUAL_ROOT}/finetrainers:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}" "${CASUAL_ROOT}/logs/infer/i2av"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" | tee "${LOG}"
echo "DATA_ROOT=${DATA_ROOT}" | tee -a "${LOG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}" | tee -a "${LOG}"
echo "GUIDANCE_SCALE=${GUIDANCE_SCALE}" | tee -a "${LOG}"

python "${CASUAL_ROOT}/scripts/infer_cogvideox_i2av_lora.py" \
  --model_path "${MODEL_PATH}" \
  --data_root "${DATA_ROOT}" \
  --train_data_root "${TRAIN_DATA_ROOT}" \
  --lora_dir "${CHECKPOINT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --track_norm_stats "${TRAIN_DATA_ROOT}/track_norm_stats.pt" \
  --height 480 --width 640 --fps 8 --num_frames 49 \
  --num_samples "${NUM_SAMPLES}" \
  --train_num_samples "${TRAIN_NUM_SAMPLES}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --seed "${SEED}" \
  --device cuda \
  --i2av_layout v6 \
  --infer_stage stage3 \
  --pose_pixel_frames 25 \
  --rgb_pixel_frames 24 \
  --track_max_timestep "${TRACK_MAX_TIMESTEP}" \
  2>&1 | tee -a "${LOG}"

echo "Done. Output: ${OUTPUT_DIR}/eval_dataset" | tee -a "${LOG}"
