# Casual CoAF 640x480

Lightweight export of the 640x480 CoAF v6 depth+track+RGB training code.

This repository contains code and placeholders only. It intentionally excludes datasets, model weights, checkpoints, logs, and generated outputs.

See `coaf_640x480_training/README.md` for the 640x480 training, precompute, and inference entry points.

## Required External Assets

Place these assets locally before running jobs:

- `training/cog_video_training/models/CogVideoX-5b-I2V/`
- `coaf_dataset_24_25/state_norm_stats.pt`
- `coaf_dataset_24_25/action_norm_stats.pt`
- `coaf_dataset_24_25/composed/v6_depth_track_rgb_640x480_from_640_local_paths/`
- `coaf_dataset_24_25/composed/v6_depth_track_rgb_640x480_one_sample_overfit/`

The committed files under those paths are placeholders only.
