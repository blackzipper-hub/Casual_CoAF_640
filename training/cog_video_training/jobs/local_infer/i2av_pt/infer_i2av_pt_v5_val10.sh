#!/usr/bin/env bash
# Local inference on composed val10 split (standard offline eval, no Simpler replay).
#
# Usage:
#   bash jobs/local_infer/i2av_pt/infer_i2av_pt_v5_val10.sh
#   CHECKPOINT_DIR=.../checkpoint-8000 NUM_SAMPLES=10 bash jobs/local_infer/i2av_pt/infer_i2av_pt_v5_val10.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASUAL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="$(cd "${CASUAL_ROOT}/../.." && pwd)"

export PROJECT_ROOT CASUAL_ROOT
export COAF_ROOT="${CASUAL_ROOT}"
export DATASET_ROOT="${PROJECT_ROOT}/coaf_dataset_24_25"
export COAF_SKIP_CONDA_ACTIVATE=1
export PYTHON="${PYTHON:-/mnt/disk1/sunkai/miniconda3/envs/coaf_train/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export SKIP_VIDEO_DECODE="${SKIP_VIDEO_DECODE:-1}"

# shellcheck disable=SC1091
source "${CASUAL_ROOT}/scripts/cluster_env.sh"

CKPT_BASE="${CASUAL_ROOT}/outputs/checkpoints/i2av/v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_fix1_clean_past_chunks_next_window"
if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  LATEST_STEP=-1
  LATEST_DIR=""
  for path in "${CKPT_BASE}"/checkpoint-*; do
    [[ -d "${path}" ]] || continue
    step="${path##*-}"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( step > LATEST_STEP )); then
      LATEST_STEP="${step}"
      LATEST_DIR="${path}"
    fi
  done
  if [[ -z "${LATEST_DIR}" ]]; then
    echo "No checkpoint found under ${CKPT_BASE}" >&2
    exit 1
  fi
  CHECKPOINT_DIR="${LATEST_DIR}"
fi
export CHECKPOINT_DIR

export MODEL_NAME="i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_fix1_next_window_full15k_val10"
export DATA_ROOT="${DATASET_ROOT}/composed/v4_depth_rgb_local_paths"
export TRAIN_DATA_ROOT="${DATA_ROOT}"
export INFER_OUTPUT_DIR="${CASUAL_ROOT}/outputs/infer/i2av_pt/${MODEL_NAME}_$(basename "${CHECKPOINT_DIR}")"
export INFER_OUTPUT_DIR_IS_FINAL=1

export MODEL_PATH="${MODEL_PATH:-${COAF_ROOT}/models/CogVideoX-5b-I2V}"
export STATE_NORM_STATS="${DATASET_ROOT}/state_norm_stats.pt"
export ACTION_NORM_STATS="${DATASET_ROOT}/action_quantile_norm_stats.pt"
export HEIGHT=256 WIDTH=256 FPS=8 MAX_NUM_FRAMES=49
export I2AV_LAYOUT=v5 POSE_PIXEL_FRAMES=25 RGB_PIXEL_FRAMES=24
export NUM_SAMPLES="${NUM_SAMPLES:-10}"
export TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-0}"
export NUM_INFERENCE_STEPS=50
export GUIDANCE_SCALE=6
export SA_GUIDANCE_SCALE=1
export SEED=42
export INFER_DEVICE=cuda
export INFER_STAGE=stage2
export GRIPPER_CONTINUOUS_ACTION=1
export SA_DENOISE_LOSS=1

mkdir -p "${CASUAL_ROOT}/logs/infer/i2av_pt"
LOG="${CASUAL_ROOT}/logs/infer/i2av_pt/local_val10_$(basename "${CHECKPOINT_DIR}")_$(date +%Y%m%d_%H%M%S).log"

echo "Checkpoint: ${CHECKPOINT_DIR}" | tee "${LOG}"
echo "Data: ${DATA_ROOT}" | tee -a "${LOG}"
echo "SKIP_VIDEO_DECODE: ${SKIP_VIDEO_DECODE}" | tee -a "${LOG}"
bash "${CASUAL_ROOT}/scripts/infer_cogvideox_i2av_lora_causal.sh" 2>&1 | tee -a "${LOG}"

echo "Infer log: ${LOG}"
echo "Output: ${INFER_OUTPUT_DIR}"
