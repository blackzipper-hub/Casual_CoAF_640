#!/usr/bin/env bash
# Local inference on v4_depth_rgb_simpler_sim + SimplerEnv action replay.
#
# Usage:
#   bash jobs/local_infer/i2av_pt/infer_i2av_pt_v5_simpler_sim.sh
#   CHECKPOINT_DIR=.../checkpoint-10500 bash jobs/local_infer/i2av_pt/infer_i2av_pt_v5_simpler_sim.sh
#
# Defaults: GPU 1, latest checkpoint under next_window run, skip VAE decode (24GB-safe).

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

export MODEL_NAME="i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_fix1_next_window_full15k_simpler_sim"
export DATA_ROOT="${DATASET_ROOT}/composed/v4_depth_rgb_simpler_sim"
export TRAIN_DATA_ROOT="${DATA_ROOT}"
export INFER_OUTPUT_DIR="${CASUAL_ROOT}/outputs/infer/i2av_pt/${MODEL_NAME}_$(basename "${CHECKPOINT_DIR}")"
export INFER_OUTPUT_DIR_IS_FINAL=1

export MODEL_PATH="${MODEL_PATH:-${COAF_ROOT}/models/CogVideoX-5b-I2V}"
export STATE_NORM_STATS="${DATASET_ROOT}/state_norm_stats.pt"
export ACTION_NORM_STATS="${DATASET_ROOT}/action_quantile_norm_stats.pt"
export HEIGHT=256 WIDTH=256 FPS=8 MAX_NUM_FRAMES=49
export I2AV_LAYOUT=v5 POSE_PIXEL_FRAMES=25 RGB_PIXEL_FRAMES=24
export NUM_SAMPLES="${NUM_SAMPLES:-3}"
export TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-1}"
export NUM_INFERENCE_STEPS=50
export GUIDANCE_SCALE=6
export SA_GUIDANCE_SCALE=1
export SEED=42
export INFER_DEVICE=cuda
export INFER_STAGE=stage2
export GRIPPER_CONTINUOUS_ACTION=1
export SA_DENOISE_LOSS=1
export SIMPLER_SIM_REPLAY=0
export SIMPLER_ROOT="${SIMPLER_ROOT:-/mnt/disk1/sunkai/SimplerEnv}"

mkdir -p "${CASUAL_ROOT}/logs/infer/i2av_pt"
LOG="${CASUAL_ROOT}/logs/infer/i2av_pt/local_simpler_sim_$(basename "${CHECKPOINT_DIR}")_$(date +%Y%m%d_%H%M%S).log"

echo "Checkpoint: ${CHECKPOINT_DIR}" | tee "${LOG}"
echo "Data: ${DATA_ROOT}" | tee -a "${LOG}"
echo "SKIP_VIDEO_DECODE: ${SKIP_VIDEO_DECODE}" | tee -a "${LOG}"
bash "${CASUAL_ROOT}/scripts/infer_cogvideox_i2av_lora_causal.sh" 2>&1 | tee -a "${LOG}"

SIMPLER_PYTHON="${SIMPLER_PYTHON:-/mnt/disk1/sunkai/miniconda3/envs/simpler_env/bin/python}"
REPLAY_LOG="${CASUAL_ROOT}/logs/infer/i2av_pt/local_simpler_sim_replay_$(basename "${CHECKPOINT_DIR}")_$(date +%Y%m%d_%H%M%S).log"
"${SIMPLER_PYTHON}" - <<PY 2>&1 | tee -a "${REPLAY_LOG}"
import json
import sys
from pathlib import Path

scripts = Path("${PROJECT_ROOT}") / "coaf_dataset_24_25" / "scripts"
sys.path.insert(0, str(scripts))
from simpler_sim.replay import replay_eval_split

data_root = Path("${DATA_ROOT}")
eval_root = Path("${INFER_OUTPUT_DIR}") / "eval_dataset" / "validation"
payload = json.loads((data_root / "validation.json").read_text(encoding="utf-8"))
items = payload.get("data", payload)
for idx, item in enumerate(items, start=1):
    item.setdefault("sample_index", idx - 1)

results = replay_eval_split(
    eval_split_root=eval_root,
    items=items,
    output_root=eval_root / "simpler_replay",
    action_format="bridge_v2",
    simpler_root=Path("${SIMPLER_ROOT}"),
)
success = sum(1 for row in results if row.get("success"))
print(f"Simpler replay done: {success}/{len(results)} success")
PY

echo "Infer log: ${LOG}"
echo "Replay log: ${REPLAY_LOG}"
echo "Output: ${INFER_OUTPUT_DIR}"
