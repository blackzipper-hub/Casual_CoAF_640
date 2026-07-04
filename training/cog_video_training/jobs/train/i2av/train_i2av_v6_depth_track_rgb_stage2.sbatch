#!/bin/bash
#SBATCH --job-name=tdtr2cv1
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --output=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/train/i2av/%x-%j.out
#SBATCH --error=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/train/i2av/%x-%j.err

set -euo pipefail
module purge
source /project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/scripts/cluster_env.sh
cd "${CASUAL_ROOT}"
mkdir -p logs/train/i2av outputs/checkpoints

export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/composed/v6_depth_track_rgb
export OUTPUT_DIR="${CASUAL_ROOT}/outputs/checkpoints/i2av/v6_depth_track_rgb_stage2"
export STATE_NORM_STATS=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/state_norm_stats.pt
export TRACK_NORM_STATS="${DATA_ROOT}/track_norm_stats.pt"

export HEIGHT=256 WIDTH=256 FPS=8 MAX_NUM_FRAMES=49 FRAME_BUCKETS=49
export TRAIN_BATCH_SIZE=1 NUM_GPUS=4 GRADIENT_ACCUMULATION_STEPS=8
export TRAIN_STEPS="${TRAIN_STEPS:-60000}" CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-2500}" CHECKPOINTS_TOTAL_LIMIT=12
export LR="${LR:-1e-4}" LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
export TRACKER_NAME=casual-coaf-i2av-v6-depth-track-rgb-stage2
export I2AV_LAYOUT=v6 POSE_PIXEL_FRAMES=25 RGB_PIXEL_FRAMES=24
export TRAIN_STAGE=stage2
export LAMBDA_TRACK="${LAMBDA_TRACK:-1.0}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"

nvidia-smi || true
bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2av_lora_causal.sh"
