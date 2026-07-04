#!/usr/bin/env bash
# Local one-sample overfit for CoAF v5 direct continuous action prediction.
# This keeps qnt_fix1 + next_window labels and does not enable clean-past SA.

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
export TINY_DATA_ROOT="${DATASET_ROOT}/composed/v4_depth_rgb_one_sample_direct_action_next_window_qnt_fix1"
python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["SRC_DATA_ROOT"])
dst = Path(os.environ["TINY_DATA_ROOT"])
dst.mkdir(parents=True, exist_ok=True)

def first_nonempty(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            return line
    raise RuntimeError(f"No non-empty lines in {path}")

project_root = Path(os.environ["PROJECT_ROOT"])

def localize(value: str) -> str:
    for prefix in (
        "/project/mscaisuperpod/sunkai/Casual_CoAF",
        "/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF",
    ):
        if value.startswith(prefix):
            return str(project_root) + value[len(prefix) :]
    return value

first_video = localize(first_nonempty(src / "videos.txt"))
first_image = localize(first_nonempty(src / "images.txt"))
first_prompt = first_nonempty(src / "prompt.txt")
first_state = localize(first_nonempty(src / "state_paths.txt"))
first_action = localize(first_nonempty(src / "action_paths.txt"))

for name, value in {
    "videos.txt": first_video,
    "images.txt": first_image,
    "prompt.txt": first_prompt,
    "state_paths.txt": first_state,
    "action_paths.txt": first_action,
}.items():
    (dst / name).write_text(value + "\n", encoding="utf-8")

payload = json.loads((src / "validation.json").read_text(encoding="utf-8"))
payload["data"] = payload.get("data", [])[:1]
if payload["data"]:
    payload["data"][0]["video_path"] = first_video
    payload["data"][0]["image_path"] = first_image
    payload["data"][0]["caption"] = first_prompt
(dst / "validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

stem = Path(first_video).stem
for dirname in ("video_latents", "image_latents", "prompt_embeds"):
    (dst / dirname).mkdir(exist_ok=True)
    source = src / dirname / f"{stem}.pt"
    target = dst / dirname / f"{stem}.pt"
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.is_symlink() or target.is_file():
        target.unlink()
    target.symlink_to(source)

print(f"Prepared one-sample direct-action data at {dst}")
print(f"Sample stem: {stem}")
PY

if [[ -z "${MODEL_PATH:-}" || ! -f "${MODEL_PATH}/model_index.json" ]]; then
  export MODEL_PATH="${CASUAL_ROOT}/models/CogVideoX-5b-I2V"
else
  export MODEL_PATH
fi
export DATA_ROOT="${TINY_DATA_ROOT}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CASUAL_ROOT}/outputs/checkpoints/i2av/v5_direct_action_qnt_fix1_next_window_one_sample}"
export STATE_NORM_STATS="${DATASET_ROOT}/state_norm_stats.pt"
export ACTION_NORM_STATS="${DATASET_ROOT}/action_quantile_norm_stats.pt"

export HEIGHT=256 WIDTH=256 FPS=8 MAX_NUM_FRAMES=49 FRAME_BUCKETS=49
export TRAIN_BATCH_SIZE=1 NUM_GPUS=1 GRADIENT_ACCUMULATION_STEPS=1
export TRAIN_STEPS="${TRAIN_STEPS:-200}" CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-50}" CHECKPOINTS_TOTAL_LIMIT=8
export LR="${LR:-1e-4}" LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
export TRACKER_NAME="${TRACKER_NAME:-casual-coaf-i2av-pt-v5-direct-action-qnt-fix1-next-window-one-sample}"
export I2AV_LAYOUT=v5 POSE_PIXEL_FRAMES=25 RGB_PIXEL_FRAMES=24
export ACTION_CHUNK_ALIGNMENT=next_window
export TRAIN_STAGE=stage2
export STAGE2_TRAIN_TRANSFORMER_LORA=1
export GRIPPER_CONTINUOUS_ACTION=1
export DIRECT_ACTION_HEAD=1
export DIRECT_ACTION_USE_PAST_ACTION_COND=1
export DIRECT_ACTION_HORIZON=25
export SA_DENOISE_LOSS=0
unset STAGE2_CLEAN_PAST_SA
export LAMBDA_DIRECT_ACTION="${LAMBDA_DIRECT_ACTION:-1.0}"
export LAMBDA_G="${LAMBDA_G:-1.0}"
export LAMBDA_SA="${LAMBDA_SA:-1.0}"
export LOAD_TENSORS=1
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"
export VALIDATION_STEPS="${VALIDATION_STEPS:-0}"
export NUM_VALIDATION_VIDEOS="${NUM_VALIDATION_VIDEOS:-0}"
export REPORT_TO="${REPORT_TO:-tensorboard}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TRAIN_STEPS=${TRAIN_STEPS}"
echo "DIRECT_ACTION_HEAD=${DIRECT_ACTION_HEAD}"
echo "DIRECT_ACTION_USE_PAST_ACTION_COND=${DIRECT_ACTION_USE_PAST_ACTION_COND}"
echo "ACTION_CHUNK_ALIGNMENT=${ACTION_CHUNK_ALIGNMENT}"
echo "STAGE2_CLEAN_PAST_SA=${STAGE2_CLEAN_PAST_SA:-0}"
echo "SA_DENOISE_LOSS=${SA_DENOISE_LOSS}"

nvidia-smi || true
bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2av_lora_causal.sh" 2>&1 | tee "${CASUAL_ROOT}/logs/train/i2av_pt/local_direct_action_qnt_fix1_next_window_one_sample_$(date +%Y%m%d_%H%M%S).log"

