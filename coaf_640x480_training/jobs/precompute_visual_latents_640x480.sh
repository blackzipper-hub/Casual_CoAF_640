#!/usr/bin/env bash
# Precompute video/image VAE latents for 640x480 v6 data.
# Prompt embeddings are linked from an existing dataset unless PROMPT_EMBEDS_ROOT is empty.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/common_env.sh
source "${SCRIPT_DIR}/../scripts/common_env.sh"

DATA_ROOT="${DATA_ROOT:-${FULL_640_DATA_ROOT}}"
PROMPT_EMBEDS_ROOT="${PROMPT_EMBEDS_ROOT:-${PROJECT_ROOT}/coaf_dataset_24_25/composed/v4_depth_rgb/prompt_embeds}"
NUM_SHARDS="${NUM_SHARDS:-2}"
LOGDIR="${CASUAL_ROOT}/logs/precompute/i2av"
mkdir -p "${LOGDIR}"

if [[ -n "${PROMPT_EMBEDS_ROOT}" ]]; then
  rm -rf "${DATA_ROOT}/prompt_embeds"
  ln -s "${PROMPT_EMBEDS_ROOT}" "${DATA_ROOT}/prompt_embeds"
fi

for ((shard=0; shard<NUM_SHARDS; shard++)); do
  session="precompute_640x480_s${shard}"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "CUDA_VISIBLE_DEVICES=${shard} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${PYTHON} ${CASUAL_ROOT}/scripts/precompute_i2av_visual_latents.py --model_path ${MODEL_PATH} --data_root ${DATA_ROOT} --height 480 --width 640 --max_num_frames 49 --dtype bf16 --use_slicing --use_tiling --num_shards ${NUM_SHARDS} --shard_index ${shard} 2>&1 | tee ${LOGDIR}/precompute_640x480_visual_s${shard}_\$(date +%Y%m%d_%H%M%S).log"
done

echo "Started ${NUM_SHARDS} precompute shard(s) for ${DATA_ROOT}"
