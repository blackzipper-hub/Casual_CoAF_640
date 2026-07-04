#!/usr/bin/env bash
# Local 4090 launch (no Slurm) for full-data training:
# fix1 SA denoise + quantile action norm + clean-past chunk teacher forcing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASUAL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="$(cd "${CASUAL_ROOT}/../.." && pwd)"
export PROJECT_ROOT CASUAL_ROOT
export COAF_PATH_REMAP_TO="${COAF_PATH_REMAP_TO:-${PROJECT_ROOT}}"
export COAF_PATH_REMAP_FROM="${COAF_PATH_REMAP_FROM:-/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF}"
export CONDA_SH="${CONDA_SH:-/mnt/disk1/sunkai/miniconda3/etc/profile.d/conda.sh}"
export CONDA_ENV="${CONDA_ENV:-/mnt/disk1/sunkai/miniconda3/envs/coaf_train}"
# shellcheck disable=SC1091
source "${CASUAL_ROOT}/scripts/cluster_env.sh"

cd "${CASUAL_ROOT}"
mkdir -p logs/train/i2av_pt outputs/checkpoints/i2av

export SRC_DATA_ROOT="${DATASET_ROOT}/composed/v4_depth_rgb"
export LOCAL_DATA_ROOT="${DATASET_ROOT}/composed/v4_depth_rgb_local_paths"
python - <<'PYLOCAL'
import json
import os
from pathlib import Path

src = Path(os.environ["SRC_DATA_ROOT"])
dst = Path(os.environ["LOCAL_DATA_ROOT"])
project_root = Path(os.environ["PROJECT_ROOT"])
dst.mkdir(parents=True, exist_ok=True)

prefixes = (
    "/project/mscaisuperpod/sunkai/Casual_CoAF",
    "/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF",
)

def localize(value: str) -> str:
    value = value.strip()
    for prefix in prefixes:
        if value.startswith(prefix):
            return str(project_root) + value[len(prefix):]
    return value

for name in ("videos.txt", "images.txt", "state_paths.txt", "action_paths.txt"):
    values = [localize(line) for line in (src / name).read_text(encoding="utf-8").splitlines() if line.strip()]
    (dst / name).write_text("\n".join(values) + "\n", encoding="utf-8")

(dst / "prompt.txt").write_text((src / "prompt.txt").read_text(encoding="utf-8"), encoding="utf-8")

payload = json.loads((src / "validation.json").read_text(encoding="utf-8"))
items = payload.get("data", payload if isinstance(payload, list) else [])
for item in items:
    if isinstance(item, dict):
        if "image_path" in item:
            item["image_path"] = localize(item["image_path"])
        if "video_path" in item:
            item["video_path"] = localize(item["video_path"])
(dst / "validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

for dirname in ("video_latents", "image_latents", "prompt_embeds"):
    target = dst / dirname
    source = src / dirname
    if target.is_symlink():
        target.unlink()
    if not target.exists():
        target.symlink_to(source, target_is_directory=True)

print(f"Prepared local-path full-data manifest at {dst}")
PYLOCAL

export MODEL_PATH="${MODEL_PATH:-${COAF_ROOT}/models/CogVideoX-5b-I2V}"
export DATA_ROOT="${LOCAL_DATA_ROOT}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CASUAL_ROOT}/outputs/checkpoints/i2av/v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_fix1_clean_past_chunks}"
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
export TRAIN_STEPS="${TRAIN_STEPS:-15000}" CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}" CHECKPOINTS_TOTAL_LIMIT=14
export LR="${LR:-5e-5}" LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
export TRACKER_NAME="${TRACKER_NAME:-casual-coaf-i2av-pt-v5-depth-rgb-2524-stage2-sa-denoise-d6cont-qnt-fix1-clean-past-chunks-full}"
export I2AV_LAYOUT=v5 POSE_PIXEL_FRAMES=25 RGB_PIXEL_FRAMES=24
export ACTION_CHUNK_ALIGNMENT="${ACTION_CHUNK_ALIGNMENT:-current}"
export TRAIN_STAGE=stage2
export STAGE2_TRAIN_TRANSFORMER_LORA=1
export GRIPPER_CONTINUOUS_ACTION=1
export SA_DENOISE_LOSS=1
export STAGE2_CLEAN_PAST_SA=1
export LAMBDA_S="${LAMBDA_S:-1.0}"
export LAMBDA_A="${LAMBDA_A:-2.0}"
export LAMBDA_DECODED_STATE="${LAMBDA_DECODED_STATE:-0.1}"
export LAMBDA_DECODED_ACTION="${LAMBDA_DECODED_ACTION:-1.0}"
export LAMBDA_G="${LAMBDA_G:-1.0}"
export LAMBDA_C="${LAMBDA_C:-0.1}"
export LOAD_TENSORS=1
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"
export VALIDATION_STEPS="${VALIDATION_STEPS:-0}"
export NUM_VALIDATION_VIDEOS="${NUM_VALIDATION_VIDEOS:-0}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export LOGGING_DIR="${LOGGING_DIR:-logs}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ACTION_NORM_STATS=${ACTION_NORM_STATS}"
echo "TRAIN_STEPS=${TRAIN_STEPS}"
echo "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
echo "STAGE2_CLEAN_PAST_SA=${STAGE2_CLEAN_PAST_SA}"
echo "ACTION_CHUNK_ALIGNMENT=${ACTION_CHUNK_ALIGNMENT}"
echo "LAMBDA_DECODED_STATE=${LAMBDA_DECODED_STATE}"
echo "LAMBDA_DECODED_ACTION=${LAMBDA_DECODED_ACTION}"

nvidia-smi || true
bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2av_lora_causal.sh" 2>&1 | tee "${CASUAL_ROOT}/logs/train/i2av_pt/local_qnt_fix1_clean_past_chunks_full_15k_$(date +%Y%m%d_%H%M%S).log"
