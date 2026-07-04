#!/usr/bin/env bash
# Local 4090 launch (no Slurm) for one-sample overfit:
# stage2 SA denoise + quantile action norm + clean past-SA teacher forcing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASUAL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck disable=SC1091
source "${CASUAL_ROOT}/scripts/cluster_env.sh"

cd "${CASUAL_ROOT}"
mkdir -p logs/train/i2av_pt outputs/checkpoints/i2av

export MODEL_PATH="${MODEL_PATH:-${COAF_ROOT}/models/CogVideoX-5b-I2V}"
export DATA_ROOT="${DATASET_ROOT}/composed/v4_depth_rgb"
export DATASET_FILE="${DATASET_FILE:-${DATA_ROOT}/one_sample_episode_000000.csv}"
export OUTPUT_DIR="${CASUAL_ROOT}/outputs/checkpoints/i2av/v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_clean_past_one_sample_2k"
export STATE_NORM_STATS="${DATASET_ROOT}/state_norm_stats.pt"
export ACTION_NORM_STATS="${DATASET_ROOT}/action_quantile_norm_stats.pt"

if [[ ! -f "${ACTION_NORM_STATS}" ]]; then
  echo "Building missing action quantile stats: ${ACTION_NORM_STATS}"
  python "${CASUAL_ROOT}/scripts/build_action_quantile_norm_stats.py" \
    --action_paths "${DATA_ROOT}/action_paths.txt" \
    --output "${ACTION_NORM_STATS}"
fi

export HEIGHT=256 WIDTH=256 FPS=8 MAX_NUM_FRAMES=49 FRAME_BUCKETS=49
export TRAIN_BATCH_SIZE=1 NUM_GPUS=1 GRADIENT_ACCUMULATION_STEPS=4
export TRAIN_STEPS="${TRAIN_STEPS:-2000}" CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}" CHECKPOINTS_TOTAL_LIMIT=6
export LR="${LR:-5e-5}" LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
export TRACKER_NAME=casual-coaf-i2av-pt-v5-depth-rgb-2524-stage2-sa-denoise-d6cont-qnt-clean-past-one-sample
export I2AV_LAYOUT=v5 POSE_PIXEL_FRAMES=25 RGB_PIXEL_FRAMES=24
export TRAIN_STAGE=stage2
export STAGE2_TRAIN_TRANSFORMER_LORA=1
export GRIPPER_CONTINUOUS_ACTION=1
export SA_DENOISE_LOSS=1
export STAGE2_CLEAN_PAST_SA=1
export LAMBDA_S="${LAMBDA_S:-1.0}"
export LAMBDA_A="${LAMBDA_A:-2.0}"
export LAMBDA_G="${LAMBDA_G:-1.0}"
export LAMBDA_C="${LAMBDA_C:-0.1}"
export LOAD_TENSORS=1
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"
export VALIDATION_STEPS="${VALIDATION_STEPS:-0}"
export NUM_VALIDATION_VIDEOS="${NUM_VALIDATION_VIDEOS:-0}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export LOGGING_DIR="${LOGGING_DIR:-logs}"
if [[ "${REPORT_TO}" == "wandb" && "${WANDB_MODE:-disabled}" == "disabled" ]]; then
  export WANDB_MODE=online
fi

echo "COAF_ENV=${COAF_ENV}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "DATASET_FILE=${DATASET_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ACTION_NORM_STATS=${ACTION_NORM_STATS}"

nvidia-smi || true
bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2av_lora_causal.sh" 2>&1 | tee "${CASUAL_ROOT}/logs/train/i2av_pt/local_qnt_clean_past_one_sample_$(date +%Y%m%d_%H%M%S).log"
