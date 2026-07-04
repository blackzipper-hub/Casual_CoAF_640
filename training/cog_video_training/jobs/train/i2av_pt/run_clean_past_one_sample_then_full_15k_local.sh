#!/usr/bin/env bash
# Queue clean-past experiments locally:
# 1) one-sample overfit for 2k steps
# 2) full-data training to 15k steps

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASUAL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${CASUAL_ROOT}"

echo "[$(date)] start one-sample clean-past 2k"
bash "jobs/train/i2av_pt/train_i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_clean_past_one_sample_2k_local.sh"

echo "[$(date)] one-sample done; start full clean-past 15k"
bash "jobs/train/i2av_pt/train_i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_clean_past_full_15k_local.sh"

echo "[$(date)] clean-past training queue finished"
