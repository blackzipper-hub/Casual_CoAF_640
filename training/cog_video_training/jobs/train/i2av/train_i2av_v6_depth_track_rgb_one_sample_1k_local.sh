#!/usr/bin/env bash
# Local one-sample overfit for CoAF v6 depth+track+RGB stage2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASUAL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="$(cd "${CASUAL_ROOT}/../.." && pwd)"
export PROJECT_ROOT CASUAL_ROOT
export CONDA_SH="${CONDA_SH:-/mnt/disk1/sunkai/miniconda3/etc/profile.d/conda.sh}"
export CONDA_ENV="${CONDA_ENV:-/mnt/disk1/sunkai/miniconda3/envs/coaf_train}"
# shellcheck disable=SC1091
source "${CASUAL_ROOT}/scripts/cluster_env.sh"

cd "${CASUAL_ROOT}"
mkdir -p logs/train/i2av outputs/checkpoints/i2av

export DATA_ROOT="${PROJECT_ROOT}/coaf_dataset_24_25/composed/v6_depth_track_rgb_one_sample_overfit"
export OUTPUT_DIR="${OUTPUT_DIR:-${CASUAL_ROOT}/outputs/checkpoints/i2av/v6_depth_track_rgb_one_sample_1k}"
export MODEL_PATH="${MODEL_PATH:-${CASUAL_ROOT}/models/CogVideoX-5b-I2V}"
export STATE_NORM_STATS="${PROJECT_ROOT}/coaf_dataset_24_25/state_norm_stats.pt"
export TRACK_NORM_STATS="${DATA_ROOT}/track_norm_stats.pt"

export HEIGHT=256 WIDTH=256 FPS=8 MAX_NUM_FRAMES=49 FRAME_BUCKETS=49
export TRAIN_BATCH_SIZE=1 NUM_GPUS=1 GRADIENT_ACCUMULATION_STEPS=1
export TRAIN_STEPS="${TRAIN_STEPS:-1000}" CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-200}" CHECKPOINTS_TOTAL_LIMIT=8
export LR="${LR:-1e-4}" LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
export TRACKER_NAME="${TRACKER_NAME:-casual-coaf-i2av-v6-depth-track-rgb-one-sample-1k}"
export I2AV_LAYOUT=v6 POSE_PIXEL_FRAMES=25 RGB_PIXEL_FRAMES=24
export TRAIN_STAGE=stage2
export STAGE2_TRAIN_TRANSFORMER_LORA=1
export LAMBDA_TRACK="${LAMBDA_TRACK:-1.0}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"
export VALIDATION_STEPS="${VALIDATION_STEPS:-0}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
unset LOAD_TENSORS

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TRAIN_STEPS=${TRAIN_STEPS}"
echo "TRACK_NORM_STATS=${TRACK_NORM_STATS}"
nvidia-smi || true

bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2av_lora_causal.sh" 2>&1 | tee "${CASUAL_ROOT}/logs/train/i2av/local_v6_depth_track_rgb_one_sample_1k_$(date +%Y%m%d_%H%M%S).log"
