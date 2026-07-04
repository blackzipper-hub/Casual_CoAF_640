# CoAF 640x480 Training Wrappers

This directory contains the lightweight entry points needed to run the current
640x480 CoAF v6 depth+track+RGB training flow.

It intentionally does not copy model weights, datasets, precomputed latents, or
the finetrainers implementation. All model/training code is used from:

```text
../training/cog_video_training/
```

## What This Directory Owns

```text
coaf_640x480_training/
├── README.md
├── scripts/
│   └── common_env.sh
└── jobs/
    ├── precompute_visual_latents_640x480.sh
    ├── train_one_sample_no_s0_1k.sh
    ├── train_stage3_no_s0_640x480.sh
    └── infer_stage3_640x480.sh
```

## Code Dependency Boundary

These wrappers depend on the existing training implementation under
`../training/cog_video_training`:

- Training launcher: `scripts/train_cogvideox_i2av_lora_causal.sh`
- Main training script:
  `finetrainers/examples/_legacy/training/cogvideox/cogvideox_image_to_video_lora_i2av.py`
- v6 model patches:
  `finetrainers/finetrainers/patches/models/cogvideox/`
- Inference script: `scripts/infer_cogvideox_i2av_lora.py`
- Visual latent precompute script: `scripts/precompute_i2av_visual_latents.py`

No code in this directory imports or vendors a second copy of those modules.

## Data And Weight Inputs

The wrappers expect these paths to exist, but do not copy them:

```text
coaf_dataset_24_25/composed/v6_depth_track_rgb_640x480_from_640_local_paths
coaf_dataset_24_25/composed/v6_depth_track_rgb_640x480_one_sample_overfit
training/cog_video_training/models/CogVideoX-5b-I2V
```

Override paths with environment variables if needed:

```bash
FULL_640_DATA_ROOT=/path/to/full_640_data
ONE_SAMPLE_640_DATA_ROOT=/path/to/one_sample_data
MODEL_PATH=/path/to/CogVideoX-5b-I2V
```

## Current Model Input Contract

The current no-state0 version uses:

```text
condition input = image + text prompt
```

The wrappers set:

```bash
S0_COND_TOKENS=0
I2AV_LAYOUT=v6
HEIGHT=480
WIDTH=640
MAX_NUM_FRAMES=49
POSE_PIXEL_FRAMES=25
RGB_PIXEL_FRAMES=24
LOAD_TENSORS=1
```

State files may still appear in dataset manifests and output metadata for
compatibility, but state0 is not appended to the model prompt condition.
`state_norm_stats.pt` and `action_norm_stats.pt` are not required by this
no-state0 v6 track path. They are only needed for legacy S0-conditioned or
state/action loss branches.

## Jobs

### 1. Precompute Visual Latents

```bash
bash coaf_640x480_training/jobs/precompute_visual_latents_640x480.sh
```

This calls `../training/cog_video_training/scripts/precompute_i2av_visual_latents.py`
and writes `video_latents/` and `image_latents/` into the dataset directory.
By default it symlinks prompt embeddings from `v4_depth_rgb/prompt_embeds`.

Useful overrides:

```bash
DATA_ROOT=/path/to/640_data NUM_SHARDS=2 bash coaf_640x480_training/jobs/precompute_visual_latents_640x480.sh
```

### 2. One-Sample Smoke Overfit

```bash
bash coaf_640x480_training/jobs/train_one_sample_no_s0_1k.sh
```

Defaults:

- dataset: `v6_depth_track_rgb_640x480_one_sample_overfit`
- output: `training/cog_video_training/outputs/checkpoints/i2av/v6_depth_track_rgb_640x480_one_sample_no_s0_1k`
- steps: `1000`
- checkpoint interval: `500`
- batch: `1`
- gradient accumulation: `1`

Example:

```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_STEPS=1000 CHECKPOINTING_STEPS=500 \
  bash coaf_640x480_training/jobs/train_one_sample_no_s0_1k.sh
```

### 3. Full Stage3 Training

```bash
bash coaf_640x480_training/jobs/train_stage3_no_s0_640x480.sh
```

Defaults:

- dataset: `v6_depth_track_rgb_640x480_from_640_local_paths`
- output: `training/cog_video_training/outputs/checkpoints/i2av/v6_depth_track_rgb_640x480_stage3_no_s0`
- steps: `60000`
- checkpoint interval: `1000`
- batch: `1`
- gradient accumulation: `8`

Example:

```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_STEPS=20000 GRADIENT_ACCUMULATION_STEPS=2 \
  bash coaf_640x480_training/jobs/train_stage3_no_s0_640x480.sh
```

### 4. Inference

```bash
bash coaf_640x480_training/jobs/infer_stage3_640x480.sh
```

Defaults:

- checkpoint base: `training/cog_video_training/outputs/checkpoints/i2av/v6_depth_track_rgb_640x480_stage3_no_s0`
- latest checkpoint if `CHECKPOINTS` is not set
- `GUIDANCE_SCALE=1` for 24GB 4090 compatibility
- writes to `training/cog_video_training/outputs/infer/i2av/`

Examples:

```bash
CHECKPOINTS=checkpoint-1000 NUM_SAMPLES=1 TRAIN_NUM_SAMPLES=1 \
  bash coaf_640x480_training/jobs/infer_stage3_640x480.sh

CHECKPOINTS=/abs/path/to/checkpoint-3000 GUIDANCE_SCALE=1 \
  bash coaf_640x480_training/jobs/infer_stage3_640x480.sh
```

## Notes

- 640x480 inference with `GUIDANCE_SCALE=6` can OOM on 24GB 4090 cards.
  Use `GUIDANCE_SCALE=1` unless running on a larger GPU.
- The wrappers write logs and checkpoints under `../training/cog_video_training`
  so existing monitoring and output conventions continue to work.
- If a future experiment needs state0 conditioning again, set
  `S0_COND_TOKENS=4`, but the current intended 640x480 path keeps it at `0`.
