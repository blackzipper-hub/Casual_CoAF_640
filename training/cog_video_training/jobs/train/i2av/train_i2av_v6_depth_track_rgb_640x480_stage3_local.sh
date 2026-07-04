#!/usr/bin/env bash
# Local 4090 launch for CoAF v6 depth+track+RGB stage3 at 640x480.
#
# Usage:
#   bash jobs/train/i2av/train_i2av_v6_depth_track_rgb_640x480_stage3_local.sh
#   CUDA_VISIBLE_DEVICES=0 TRAIN_STEPS=10 bash jobs/train/i2av/train_i2av_v6_depth_track_rgb_640x480_stage3_local.sh
#   NUM_GPUS=2 GRADIENT_ACCUMULATION_STEPS=4 bash jobs/train/i2av/train_i2av_v6_depth_track_rgb_640x480_stage3_local.sh

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

export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/coaf_dataset_24_25/composed/v6_depth_track_rgb_640x480_from_640_local_paths}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CASUAL_ROOT}/outputs/checkpoints/i2av/v6_depth_track_rgb_640x480_stage3_joint_load_tensors}"
export MODEL_PATH="${MODEL_PATH:-${CASUAL_ROOT}/models/CogVideoX-5b-I2V}"
export STATE_NORM_STATS="${PROJECT_ROOT}/coaf_dataset_24_25/state_norm_stats.pt"
export ACTION_NORM_STATS="${PROJECT_ROOT}/coaf_dataset_24_25/action_norm_stats.pt"
export TRACK_NORM_STATS="${DATA_ROOT}/track_norm_stats.pt"

export HEIGHT=480
export WIDTH=640
export FPS=8
export MAX_NUM_FRAMES=49
export FRAME_BUCKETS=49
export I2AV_LAYOUT=v6
export POSE_PIXEL_FRAMES=25
export RGB_PIXEL_FRAMES=24

# 640x480 has 1200 visual patches per latent frame (vs 256 at 256x256).
# Start conservatively; override NUM_GPUS/GRADIENT_ACCUMULATION_STEPS after a smoke run if needed.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export NUM_GPUS="${NUM_GPUS:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export TRAIN_STEPS="${TRAIN_STEPS:-60000}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
export CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-40}"
export LR="${LR:-1e-4}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"

export TRACKER_NAME="${TRACKER_NAME:-casual-coaf-i2av-v6-depth-track-rgb-640x480-stage3}"
export TRAIN_STAGE=stage3
export LAMBDA_SA="${LAMBDA_SA:-0.1}"
export LAMBDA_TRACK="${LAMBDA_TRACK:-1.0}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"
export VALIDATION_STEPS="${VALIDATION_STEPS:-0}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export LOAD_TENSORS=1

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "HEIGHT=${HEIGHT} WIDTH=${WIDTH} MAX_NUM_FRAMES=${MAX_NUM_FRAMES}"
echo "NUM_GPUS=${NUM_GPUS} TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} GRAD_ACCUM=${GRADIENT_ACCUMULATION_STEPS}"
echo "TRAIN_STEPS=${TRAIN_STEPS} RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
echo "TRACK_NORM_STATS=${TRACK_NORM_STATS}"
nvidia-smi || true

bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2av_lora_causal.sh" 2>&1 | tee "${CASUAL_ROOT}/logs/train/i2av/local_v6_depth_track_rgb_640x480_stage3_$(date +%Y%m%d_%H%M%S).log"
