#!/usr/bin/env bash
# Local v6 stage3 inference for the previous 256x256 depth+track+RGB checkpoint.
# Uses a dummy zero-normalized S0 condition instead of loading the initial state.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/disk3/sunkai/Casual_CoAF}"
CASUAL_ROOT="${CASUAL_ROOT:-${PROJECT_ROOT}/training/cog_video_training}"
MODEL_NAME="${MODEL_NAME:-v6_depth_track_rgb_256_stage3_dummy_state}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-${CASUAL_ROOT}/outputs/checkpoints/i2av/v6_depth_track_rgb_stage3_joint_load_tensors_2gpu_v3_sync_timestep_from0}"
CHECKPOINTS="${CHECKPOINTS:-${CHECKPOINT_STEP:-}}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
TRACK_MAX_TIMESTEP="${TRACK_MAX_TIMESTEP:-499}"

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

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/coaf_dataset_24_25/composed/v6_depth_track_rgb_local_paths}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-${DATA_ROOT}}"
INFER_ROOT="${INFER_ROOT:-${CASUAL_ROOT}/outputs/infer/i2av}"
OUTPUT_DIR="${INFER_ROOT}/${MODEL_NAME}_${CHECKPOINT_TAG}_seed${SEED}_video_256x256_tmax${TRACK_MAX_TIMESTEP}"

source /mnt/disk1/sunkai/miniconda3/etc/profile.d/conda.sh
conda activate coaf_train

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export FREE_MODEL_BEFORE_DECODE="${FREE_MODEL_BEFORE_DECODE:-1}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${INFER_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONPATH="${CASUAL_ROOT}/finetrainers:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}" "${CASUAL_ROOT}/logs/infer/i2av"
LOG="${CASUAL_ROOT}/logs/infer/i2av/${MODEL_NAME}_${CHECKPOINT_TAG}_seed${SEED}_video_256x256_tmax${TRACK_MAX_TIMESTEP}_$(date +%Y%m%d_%H%M%S).log"

echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" | tee "${LOG}"
echo "DATA_ROOT=${DATA_ROOT}" | tee -a "${LOG}"
echo "TRAIN_DATA_ROOT=${TRAIN_DATA_ROOT}" | tee -a "${LOG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}" | tee -a "${LOG}"
echo "DUMMY_STATE=1" | tee -a "${LOG}"

python "${CASUAL_ROOT}/scripts/infer_cogvideox_i2av_lora.py" \
  --model_path "${CASUAL_ROOT}/models/CogVideoX-5b-I2V" \
  --data_root "${DATA_ROOT}" \
  --train_data_root "${TRAIN_DATA_ROOT}" \
  --lora_dir "${CHECKPOINT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --state_norm_stats "${PROJECT_ROOT}/coaf_dataset_24_25/state_norm_stats.pt" \
  --track_norm_stats "${TRAIN_DATA_ROOT}/track_norm_stats.pt" \
  --height 256 --width 256 --fps 8 --num_frames 49 \
  --num_samples "${NUM_SAMPLES}" \
  --train_num_samples "${TRAIN_NUM_SAMPLES}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --guidance_scale 6 \
  --seed "${SEED}" \
  --device cuda \
  --i2av_layout v6 \
  --infer_stage stage3 \
  --pose_pixel_frames 25 \
  --rgb_pixel_frames 24 \
  --track_max_timestep "${TRACK_MAX_TIMESTEP}" \
  --dummy_state \
  2>&1 | tee -a "${LOG}"

echo "Done. Output: ${OUTPUT_DIR}/eval_dataset" | tee -a "${LOG}"
