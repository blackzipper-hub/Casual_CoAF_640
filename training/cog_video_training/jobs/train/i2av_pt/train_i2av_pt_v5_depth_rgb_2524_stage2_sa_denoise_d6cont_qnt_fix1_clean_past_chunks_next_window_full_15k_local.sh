#!/usr/bin/env bash
# Local 4090 launch for the next-window action chunk alignment ablation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASUAL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export ACTION_CHUNK_ALIGNMENT=next_window
export OUTPUT_DIR="${OUTPUT_DIR:-${CASUAL_ROOT}/outputs/checkpoints/i2av/v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_fix1_clean_past_chunks_next_window}"
export TRACKER_NAME="${TRACKER_NAME:-casual-coaf-i2av-pt-v5-depth-rgb-2524-stage2-sa-denoise-d6cont-qnt-fix1-clean-past-chunks-next-window-full}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"

bash "${SCRIPT_DIR}/train_i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_fix1_clean_past_chunks_full_15k_local.sh"
