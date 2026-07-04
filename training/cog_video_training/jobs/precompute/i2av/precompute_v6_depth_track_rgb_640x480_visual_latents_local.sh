#!/usr/bin/env bash
# Precompute 640x480 visual latents for v6 depth+track+RGB data.
# Computes video_latents and image_latents only; prompt_embeds is reused via symlink.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/disk3/sunkai/Casual_CoAF}"
CASUAL_ROOT="${CASUAL_ROOT:-${PROJECT_ROOT}/training/cog_video_training}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/coaf_dataset_24_25/composed/v6_depth_track_rgb_640x480_from_640_local_paths}"
V4_ROOT="${V4_ROOT:-${PROJECT_ROOT}/coaf_dataset_24_25/composed/v4_depth_rgb}"
MODEL_PATH="${MODEL_PATH:-${CASUAL_ROOT}/models/CogVideoX-5b-I2V}"
LOGDIR="${CASUAL_ROOT}/logs/precompute/i2av"
PYTHON="${PYTHON:-/mnt/disk1/sunkai/miniconda3/envs/coaf_train/bin/python3}"

mkdir -p "${LOGDIR}"
rm -rf "${DATA_ROOT}/prompt_embeds"
ln -s "${V4_ROOT}/prompt_embeds" "${DATA_ROOT}/prompt_embeds"

tmux kill-session -t precompute_640x480_s0 2>/dev/null || true
tmux kill-session -t precompute_640x480_s1 2>/dev/null || true

tmux new-session -d -s precompute_640x480_s0 \
  "CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${PYTHON} ${CASUAL_ROOT}/scripts/precompute_i2av_visual_latents.py --model_path ${MODEL_PATH} --data_root ${DATA_ROOT} --height 480 --width 640 --max_num_frames 49 --dtype bf16 --use_slicing --use_tiling --num_shards 2 --shard_index 0 2>&1 | tee ${LOGDIR}/precompute_640x480_visual_s0_\$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s precompute_640x480_s1 \
  "CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${PYTHON} ${CASUAL_ROOT}/scripts/precompute_i2av_visual_latents.py --model_path ${MODEL_PATH} --data_root ${DATA_ROOT} --height 480 --width 640 --max_num_frames 49 --dtype bf16 --use_slicing --use_tiling --num_shards 2 --shard_index 1 2>&1 | tee ${LOGDIR}/precompute_640x480_visual_s1_\$(date +%Y%m%d_%H%M%S).log"

echo "Started precompute_640x480_s0 and precompute_640x480_s1"
echo "DATA_ROOT=${DATA_ROOT}"
