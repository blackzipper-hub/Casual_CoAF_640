#!/usr/bin/env bash
# Shared environment for the 640x480 CoAF wrappers.

set -euo pipefail

WRAPPER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${WRAPPER_ROOT}/.." && pwd)"
CASUAL_ROOT="${PROJECT_ROOT}/training/cog_video_training"

if [[ ! -d "${CASUAL_ROOT}" ]]; then
  echo "Expected training code at ${CASUAL_ROOT}" >&2
  exit 1
fi

export PROJECT_ROOT
export CASUAL_ROOT
export CONDA_SH="${CONDA_SH:-/mnt/disk1/sunkai/miniconda3/etc/profile.d/conda.sh}"
export CONDA_ENV="${CONDA_ENV:-/mnt/disk1/sunkai/miniconda3/envs/coaf_train}"
export PYTHON="${PYTHON:-/mnt/disk1/sunkai/miniconda3/envs/coaf_train/bin/python3}"

export FULL_640_DATA_ROOT="${FULL_640_DATA_ROOT:-${PROJECT_ROOT}/coaf_dataset_24_25/composed/v6_depth_track_rgb_640x480_from_640_local_paths}"
export ONE_SAMPLE_640_DATA_ROOT="${ONE_SAMPLE_640_DATA_ROOT:-${PROJECT_ROOT}/coaf_dataset_24_25/composed/v6_depth_track_rgb_640x480_one_sample_overfit}"
export MODEL_PATH="${MODEL_PATH:-${CASUAL_ROOT}/models/CogVideoX-5b-I2V}"
export STATE_NORM_STATS="${STATE_NORM_STATS:-${PROJECT_ROOT}/coaf_dataset_24_25/state_norm_stats.pt}"
export ACTION_NORM_STATS="${ACTION_NORM_STATS:-${PROJECT_ROOT}/coaf_dataset_24_25/action_norm_stats.pt}"

export HEIGHT=480
export WIDTH=640
export FPS=8
export MAX_NUM_FRAMES=49
export FRAME_BUCKETS=49
export I2AV_LAYOUT=v6
export POSE_PIXEL_FRAMES=25
export RGB_PIXEL_FRAMES=24
export S0_COND_TOKENS="${S0_COND_TOKENS:-0}"
export LOAD_TENSORS="${LOAD_TENSORS:-1}"
