#!/usr/bin/env bash
# Full 640x480 stage3 training for the no-state0 v6 depth+track+RGB model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/common_env.sh
source "${SCRIPT_DIR}/../scripts/common_env.sh"

export DATA_ROOT="${DATA_ROOT:-${FULL_640_DATA_ROOT}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CASUAL_ROOT}/outputs/checkpoints/i2av/v6_depth_track_rgb_640x480_stage3_no_s0}"
export TRACK_NORM_STATS="${TRACK_NORM_STATS:-${DATA_ROOT}/track_norm_stats.pt}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export NUM_GPUS="${NUM_GPUS:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export TRAIN_STEPS="${TRAIN_STEPS:-60000}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
export CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-40}"
export LR="${LR:-1e-4}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"

export TRACKER_NAME="${TRACKER_NAME:-casual-coaf-i2av-v6-640x480-stage3-no-s0}"
export TRAIN_STAGE=stage3
export LAMBDA_SA="${LAMBDA_SA:-0.1}"
export LAMBDA_TRACK="${LAMBDA_TRACK:-1.0}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"
export VALIDATION_STEPS="${VALIDATION_STEPS:-0}"
export REPORT_TO="${REPORT_TO:-tensorboard}"

cd "${CASUAL_ROOT}"
bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2av_lora_causal.sh" \
  2>&1 | tee "${CASUAL_ROOT}/logs/train/i2av/local_v6_640x480_stage3_no_s0_$(date +%Y%m%d_%H%M%S).log"
