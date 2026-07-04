from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


TRAIN_SESSION = "coaf_v6_stage3_one_sample_2k_tfix2"
CHECKPOINT = Path(
    "/mnt/disk3/sunkai/Casual_CoAF/training/cog_video_training/outputs/checkpoints/i2av/"
    "v6_depth_track_rgb_stage3_one_sample_2k_track_timestep_fix2/checkpoint-2000"
)
OUT_ROOT = Path(
    "/mnt/disk3/sunkai/Casual_CoAF/training/cog_video_training/outputs/infer/i2av/"
    "overfit_stage3_tfix2_ckpt2000_seed_sweep"
)
MONITOR_LOG = Path(
    "/mnt/disk3/sunkai/Casual_CoAF/training/cog_video_training/logs/train/i2av/"
    "monitor_stage3_tfix2_overfit_infer.log"
)


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MONITOR_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def session_exists(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def run_inference(seed: int) -> None:
    env = os.environ.copy()
    env["SKIP_VIDEO_DECODE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = (
        "/mnt/disk3/sunkai/Casual_CoAF/training/cog_video_training/finetrainers:"
        + env.get("PYTHONPATH", "")
    )
    output_dir = OUT_ROOT / f"seed_{seed}"
    cmd = [
        "/mnt/disk1/sunkai/miniconda3/envs/coaf_train/bin/python",
        "/mnt/disk3/sunkai/Casual_CoAF/training/cog_video_training/scripts/infer_cogvideox_i2av_lora.py",
        "--model_path",
        "/mnt/disk3/sunkai/Casual_CoAF/training/cog_video_training/models/CogVideoX-5b-I2V",
        "--data_root",
        "/mnt/disk3/sunkai/Casual_CoAF/coaf_dataset_24_25/composed/v6_depth_track_rgb_one_sample_overfit",
        "--train_data_root",
        "/mnt/disk3/sunkai/Casual_CoAF/coaf_dataset_24_25/composed/v6_depth_track_rgb_one_sample_overfit",
        "--lora_dir",
        str(CHECKPOINT),
        "--output_dir",
        str(output_dir),
        "--state_norm_stats",
        "/mnt/disk3/sunkai/Casual_CoAF/coaf_dataset_24_25/state_norm_stats.pt",
        "--track_norm_stats",
        "/mnt/disk3/sunkai/Casual_CoAF/coaf_dataset_24_25/composed/v6_depth_track_rgb_one_sample_overfit/track_norm_stats.pt",
        "--i2av_layout",
        "v6",
        "--infer_stage",
        "stage3",
        "--height",
        "256",
        "--width",
        "256",
        "--num_frames",
        "49",
        "--num_samples",
        "1",
        "--train_num_samples",
        "1",
        "--num_inference_steps",
        "50",
        "--guidance_scale",
        "1.0",
        "--seed",
        str(seed),
        "--device",
        "cuda",
    ]
    log(f"INFER_START seed={seed} output={output_dir}")
    subprocess.run(cmd, check=True, env=env)
    log(f"INFER_DONE seed={seed}")


def summarize(seeds: list[int]) -> None:
    rows: list[dict[str, float | int | str | bool]] = []
    for seed in seeds:
        seed_dir = OUT_ROOT / f"seed_{seed}"
        for split in ["validation", "train"]:
            metrics_path = seed_dir / "eval_dataset" / split / "track_metrics.json"
            if not metrics_path.exists():
                rows.append({"seed": seed, "split": split, "missing": True})
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "ade": float(metrics["track_ade"]),
                    "fde": float(metrics["track_fde"]),
                }
            )

    for row in rows:
        if row.get("missing"):
            log(f"RESULT seed={row['seed']} split={row['split']} MISSING")
        else:
            log(
                f"RESULT seed={row['seed']} split={row['split']} "
                f"ADE={row['ade']:.4f} FDE={row['fde']:.4f}"
            )

    values = [float(row["ade"]) for row in rows if not row.get("missing")]
    if not values:
        log("SUMMARY no metrics produced")
        return
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    median = sorted_values[mid] if len(sorted_values) % 2 else 0.5 * (sorted_values[mid - 1] + sorted_values[mid])
    log(
        "SUMMARY "
        f"count={len(values)} "
        f"ADE_mean={sum(values) / len(values):.4f} "
        f"ADE_median={median:.4f} "
        f"ADE_min={min(values):.4f} "
        f"ADE_max={max(values):.4f} "
        f"bad_over_50={sum(value > 50 for value in values)} "
        f"bad_over_100={sum(value > 100 for value in values)}"
    )


def main() -> None:
    seeds = [0, 1, 2, 3, 4, 42]
    log("MONITOR_START waiting for checkpoint-2000 and training completion")
    while not CHECKPOINT.is_dir():
        time.sleep(20)
    log(f"CHECKPOINT_READY {CHECKPOINT}")
    while session_exists(TRAIN_SESSION):
        time.sleep(5)
    log("TRAINING_SESSION_ENDED starting inference validation")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        run_inference(seed)
    summarize(seeds)
    log("OVERFIT_INFER_DONE")


if __name__ == "__main__":
    main()
