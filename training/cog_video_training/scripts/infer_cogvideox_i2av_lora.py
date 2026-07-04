#!/usr/bin/env python3
"""Run CogVideoX I2AV LoRA inference and save predicted state/action."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import inspect
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import retrieve_timesteps
from diffusers.utils import export_to_video, load_image, load_video
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, ImageDraw

from finetrainers.patches.models.cogvideox.causal_attention import install_temporal_causal_attention
from finetrainers.patches.models.cogvideox.direct_action import DirectActionModules, flatten_action_chunks
from finetrainers.patches.models.cogvideox.i2av_direct_action_forward import forward_i2av_v5_direct_action_transformer
from finetrainers.patches.models.cogvideox.i2av_forward import forward_i2av_transformer, forward_i2av_v5_transformer
from finetrainers.patches.models.cogvideox.i2av_layout import compute_i2av_v5_layout, compute_i2av_v6_layout
from finetrainers.patches.models.cogvideox.i2av_sequence import (
    expand_rope_for_direct_action_i2av_timeids,
    expand_rope_for_i2av,
    expand_rope_for_v6_i2av_timeids,
)
from finetrainers.patches.models.cogvideox.i2av_v6_forward import forward_i2av_v6_transformer
from finetrainers.patches.models.cogvideox.state_action import (
    ChunkedStateActionTokenizer,
    S0Encoder,
    StateActionTokenizer,
    get_action_norm_method,
    load_state_action_modules,
    prepare_gt,
    prepare_gt_chunked,
    prepare_raw_action_gt_chunked,
)
from finetrainers.patches.models.cogvideox.track_tokenizer import (
    TrackModules,
    denormalize_track_pixels,
    normalize_track_pixels,
)


def latest_checkpoint(root: Path) -> Path | None:
    checkpoints = []
    for path in root.glob("checkpoint-*"):
        if path.is_dir():
            try:
                checkpoints.append((int(path.name.split("-")[-1]), path))
            except ValueError:
                continue
    if not checkpoints:
        return None
    return sorted(checkpoints)[-1][1]


def resolve_lora_dir(path: Path) -> Path:
    if (path / "pytorch_lora_weights.safetensors").is_file() or (path / "pytorch_lora_weights.bin").is_file():
        return path
    ckpt = latest_checkpoint(path)
    if ckpt is not None:
        return ckpt
    raise FileNotFoundError(f"No LoRA weights found in {path} or checkpoint-* children")


def resolve_paths(data_root: Path, values: list[str]) -> list[str]:
    return [str(Path(value) if Path(value).is_absolute() else data_root / value) for value in values]


def load_manifest_paths(data_root: Path, name: str, required: bool = True) -> list[str]:
    path = data_root / name
    if not path.is_file():
        if not required:
            return []
        raise FileNotFoundError(f"Missing I2AV manifest: {path}")
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return resolve_paths(data_root, values)


def first_rgb_frame(episode_dir: Path) -> Path:
    frames = sorted((episode_dir / "rgb").glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"Missing RGB frames under {episode_dir / 'rgb'}")
    return frames[0]


def load_test_items(data_root: Path, max_samples: int) -> list[dict[str, Any]]:
    metadata_path = data_root / "splits" / "test_1k_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing test metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    items = []
    for row_idx, item in enumerate(metadata[:max_samples]):
        dataset_idx = int(item.get("dataset_idx", row_idx))
        episode_dir = data_root / "raw" / f"episode_{dataset_idx:06d}"
        instruction_path = episode_dir / "instruction" / "instruction.txt"
        prompt = item.get("instruction")
        if prompt is None and instruction_path.is_file():
            prompt = instruction_path.read_text(encoding="utf-8").strip()
        if prompt is None:
            raise ValueError(f"test item missing instruction: {item}")

        state_path = episode_dir / "state" / "state.npy"
        action_path = episode_dir / "action" / "action.npy"
        video_path = episode_dir / "video.mp4"
        if not state_path.is_file():
            raise FileNotFoundError(f"Missing test state file: {state_path}")
        if not action_path.is_file():
            raise FileNotFoundError(f"Missing test action file: {action_path}")
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing test video file: {video_path}")

        entry = {
            "sample_index": dataset_idx,
            "episode_idx": item.get("episode_idx", item.get("original_episode_idx")),
            "image_path": str(first_rgb_frame(episode_dir)),
            "prompt": prompt,
            "state_path": str(state_path),
            "action_path": str(action_path),
            "video_path": str(video_path),
        }
        for track_name, key in (("track.npy", "track_path"), ("track_visible.npy", "track_visible_path")):
            episode_track_path = episode_dir / track_name
            if episode_track_path.is_file():
                entry[key] = str(episode_track_path)
        if "track_path" not in entry:
            track_root = Path(
                os.environ.get(
                    "COAF_TRACK_ROOT",
                    "/mnt/disk3/sunkai/Casual_CoAF/coaf_dataset_24_25/modalities/track_2d/right_finger_256",
                )
            )
            track_path = track_root / f"episode_{dataset_idx:06d}" / "track.npy"
            track_visible_path = track_root / f"episode_{dataset_idx:06d}" / "track_visible.npy"
            if track_path.is_file():
                entry["track_path"] = str(track_path)
            if track_visible_path.is_file():
                entry["track_visible_path"] = str(track_visible_path)
        items.append(entry)
    return items


def load_validation_items(data_root: Path, max_samples: int) -> list[dict[str, Any]]:
    validation_path = data_root / "validation.json"
    if not validation_path.is_file():
        return load_test_items(data_root, max_samples)

    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    data = payload.get("data", payload)
    video_paths = load_manifest_paths(data_root, "videos.txt")
    state_paths = load_manifest_paths(data_root, "state_paths.txt")
    action_paths = load_manifest_paths(data_root, "action_paths.txt")
    track_paths = load_manifest_paths(data_root, "track_paths.txt", required=False)
    track_visible_paths = load_manifest_paths(data_root, "track_visible_paths.txt", required=False)

    items = []
    for row_idx, item in enumerate(data[:max_samples]):
        sample_index = int(item.get("sample_index", row_idx))
        image_path = item.get("image_path") or item.get("image")
        prompt = item.get("caption") or item.get("prompt") or item.get("text")
        if image_path is None or prompt is None:
            raise ValueError(f"validation item missing image/prompt fields: {item}")
        entry = {
            "sample_index": sample_index,
            "image_path": image_path,
            "prompt": prompt,
            "video_path": video_paths[sample_index],
            "state_path": state_paths[sample_index],
            "action_path": action_paths[sample_index],
        }
        if track_paths and sample_index < len(track_paths):
            entry["track_path"] = track_paths[sample_index]
        if track_visible_paths and sample_index < len(track_visible_paths):
            entry["track_visible_path"] = track_visible_paths[sample_index]
        for key in (
            "simpler_task",
            "simpler_episode_id",
            "simpler_seed",
            "simpler_meta_path",
        ):
            if key in item:
                entry[key] = item[key]
        if "simpler_meta_path" not in entry:
            simpler_meta_paths = load_manifest_paths(data_root, "simpler_meta_paths.txt", required=False)
            if simpler_meta_paths and sample_index < len(simpler_meta_paths):
                entry["simpler_meta_path"] = simpler_meta_paths[sample_index]
        items.append(entry)
    return items


def disable_learned_positional_embeddings(pipe: CogVideoXImageToVideoPipeline) -> None:
    patch_embed = pipe.transformer.patch_embed
    if hasattr(patch_embed, "pos_embedding"):
        del patch_embed.pos_embedding
    patch_embed.use_learned_positional_embeddings = False
    pipe.transformer.config.use_learned_positional_embeddings = False


def get_transformer_hidden_dim(transformer) -> int:
    config = transformer.config
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "num_attention_heads") and hasattr(config, "attention_head_dim"):
        return int(config.num_attention_heads * config.attention_head_dim)
    if hasattr(transformer, "norm_final") and hasattr(transformer.norm_final, "normalized_shape"):
        return int(transformer.norm_final.normalized_shape[0])
    raise ValueError("Cannot infer transformer hidden dimension from CogVideoX config/modules.")


def get_text_embed_dim(transformer) -> int:
    config = transformer.config
    if hasattr(config, "text_embed_dim"):
        return int(config.text_embed_dim)
    text_proj = getattr(transformer.patch_embed, "text_proj", None)
    if text_proj is not None and hasattr(text_proj, "in_features"):
        return int(text_proj.in_features)
    raise ValueError("Cannot infer text embedding dimension from CogVideoX config/modules.")


def prepare_i2av_rotary_emb(
    pipe: CogVideoXImageToVideoPipeline,
    height: int,
    width: int,
    latent_frames: int,
    device: torch.device,
    sa_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not pipe.transformer.config.use_rotary_positional_embeddings:
        return None

    freqs_cos, freqs_sin = pipe._prepare_rotary_positional_embeddings(height, width, latent_frames, device)
    patch = pipe.transformer.config.patch_size
    grid_h = height // (pipe.vae_scale_factor_spatial * patch)
    grid_w = width // (pipe.vae_scale_factor_spatial * patch)
    patches_per_frame = grid_h * grid_w
    return expand_rope_for_i2av(freqs_cos, freqs_sin, latent_frames, patches_per_frame, sa_per_frame)


def prepare_i2av_v5_rotary_emb(
    pipe: CogVideoXImageToVideoPipeline,
    height: int,
    width: int,
    layout,
    device: torch.device,
    *,
    direct_action_head: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not pipe.transformer.config.use_rotary_positional_embeddings:
        return None
    rope_frames = 2 * layout.num_pose_latent_frames + layout.num_rgb_latent_frames
    freqs_cos, freqs_sin = pipe._prepare_rotary_positional_embeddings(height, width, rope_frames, device)
    if direct_action_head:
        return expand_rope_for_direct_action_i2av_timeids(
            freqs_cos,
            freqs_sin,
            layout.num_pose_latent_frames,
            layout.num_rgb_latent_frames,
            layout.patches_per_frame,
        )
    from finetrainers.patches.models.cogvideox.i2av_sequence import expand_rope_for_chunked_i2av_timeids

    return expand_rope_for_chunked_i2av_timeids(
        freqs_cos,
        freqs_sin,
        layout.num_pose_latent_frames,
        layout.num_rgb_latent_frames,
        layout.patches_per_frame,
        layout.chunk_token_count,
    )


def prepare_i2av_v6_rotary_emb(
    pipe: CogVideoXImageToVideoPipeline,
    height: int,
    width: int,
    layout,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not pipe.transformer.config.use_rotary_positional_embeddings:
        return None
    rope_frames = max(layout.num_latent_frames, layout.track_steps)
    freqs_cos, freqs_sin = pipe._prepare_rotary_positional_embeddings(height, width, rope_frames, device)
    return expand_rope_for_v6_i2av_timeids(
        freqs_cos,
        freqs_sin,
        num_pose_latent_frames=layout.num_pose_latent_frames,
        num_rgb_latent_frames=layout.num_rgb_latent_frames,
        patches_per_frame=layout.patches_per_frame,
        track_steps=layout.track_steps,
        track_tokens_per_step=layout.track_tokens_per_step,
    )


def prepare_extra_step_kwargs(scheduler, generator: torch.Generator | None, eta: float) -> dict[str, Any]:
    extra_step_kwargs = {}
    if "eta" in set(inspect.signature(scheduler.step).parameters.keys()):
        extra_step_kwargs["eta"] = eta
    if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def prepare_action_epsilon_scheduler(
    scheduler,
    num_inference_steps: int,
    device: torch.device,
    reference_timesteps: torch.Tensor,
):
    """Use the video timestep schedule, but interpret direct-action outputs as epsilon noise."""
    action_scheduler = scheduler.__class__.from_config(scheduler.config, prediction_type="epsilon")
    action_timesteps, _ = retrieve_timesteps(action_scheduler, num_inference_steps, device, None)
    if action_timesteps.shape != reference_timesteps.shape or not torch.equal(action_timesteps, reference_timesteps):
        raise RuntimeError("Action scheduler timesteps diverged from the video scheduler timesteps.")
    return action_scheduler


def make_track_timesteps(
    video_timesteps: torch.Tensor,
    *,
    max_timestep: int | None,
) -> torch.Tensor:
    """Optionally run track denoising from a lower-noise timestep than video."""
    if max_timestep is None:
        return video_timesteps
    if max_timestep <= 0:
        raise ValueError("--track_max_timestep must be positive.")

    start = min(int(video_timesteps[0].item()), int(max_timestep))
    end = int(video_timesteps[-1].item())
    if start <= end:
        return video_timesteps.clamp(max=start)

    track_timesteps = torch.linspace(
        start,
        end,
        steps=video_timesteps.numel(),
        device=video_timesteps.device,
        dtype=torch.float32,
    ).round()
    return track_timesteps.to(dtype=video_timesteps.dtype)


def load_ground_truth_video_frames(item: dict[str, Any], num_frames: int) -> list[Any]:
    video_path = item.get("video_path")
    if video_path is None:
        raise ValueError("Stage2 inference requires item['video_path'] for clean video injection.")
    frames = list(load_video(str(video_path)))
    if not frames:
        raise ValueError(f"Ground-truth video has no frames: {video_path}")
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames[:num_frames]


def encode_clean_video_latents(
    pipe: CogVideoXImageToVideoPipeline,
    frames: list[Any],
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    video = pipe.video_processor.preprocess_video(frames, height=height, width=width)
    video = video.to(device=device, dtype=dtype)
    latent_dist = pipe.vae.encode(video).latent_dist
    latents = latent_dist.sample() * pipe.vae.config.scaling_factor
    latents = latents.permute(0, 2, 1, 3, 4)
    return latents.to(memory_format=torch.contiguous_format, dtype=dtype)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def draw_track_overlay(
    *,
    image_path: str | Path,
    pred_track: np.ndarray,
    output_path: Path,
    gt_track: np.ndarray | None = None,
    gt_visible: np.ndarray | None = None,
) -> None:
    base = Image.open(image_path).convert("RGB")
    width, height = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def points(track: np.ndarray, visible: np.ndarray | None = None) -> list[tuple[float, float]]:
        arr = np.asarray(track, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[-1] != 2:
            return []
        valid = np.isfinite(arr).all(axis=-1)
        if visible is not None:
            valid &= np.asarray(visible, dtype=bool)[: len(arr)]
        arr = arr[valid]
        if len(arr) == 0:
            return []
        arr[:, 0] = np.clip(arr[:, 0], 0, width - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, height - 1)
        return [(float(x), float(y)) for x, y in arr]

    def draw_path(track_points: list[tuple[float, float]], color: tuple[int, int, int, int]) -> None:
        if len(track_points) >= 2:
            draw.line(track_points, fill=color, width=3, joint="curve")
        radius = 3
        for x, y in track_points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        if track_points:
            x0, y0 = track_points[0]
            x1, y1 = track_points[-1]
            draw.ellipse((x0 - 5, y0 - 5, x0 + 5, y0 + 5), outline=(255, 255, 255, 230), width=2)
            draw.rectangle((x1 - 5, y1 - 5, x1 + 5, y1 + 5), outline=(255, 255, 255, 230), width=2)

    if gt_track is not None:
        draw_path(points(gt_track, gt_visible), (0, 255, 0, 210))
    draw_path(points(pred_track), (255, 0, 0, 220))
    draw.rectangle((6, 6, 142, 46), fill=(0, 0, 0, 130))
    draw.text((12, 10), "GT track", fill=(0, 255, 0, 255))
    draw.text((12, 26), "Pred track", fill=(255, 80, 80, 255))
    Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB").save(output_path)


def draw_track_overlay_video(
    *,
    frames: list[Any] | None = None,
    video_path: str | Path | None = None,
    pred_track: np.ndarray,
    output_path: Path,
    fps: int,
    gt_track: np.ndarray | None = None,
    gt_visible: np.ndarray | None = None,
    pose_pixel_frames: int = 25,
) -> None:
    if video_path is not None:
        frames = load_video(str(video_path))
    if not frames:
        return

    pred = np.asarray(pred_track, dtype=np.float32)
    gt = np.asarray(gt_track, dtype=np.float32) if gt_track is not None else None
    visible = np.asarray(gt_visible, dtype=bool) if gt_visible is not None else None
    track_steps = len(pred)
    rendered = []

    def to_image(frame: Any) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3, 4}:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.dtype != np.uint8:
            if float(np.nanmax(arr)) > 1.0:
                arr = np.clip(arr, 0.0, 255.0)
            else:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            arr = arr.round().astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    def frame_points(track: np.ndarray, end_idx: int, width: int, height: int, mask: np.ndarray | None = None):
        if track.ndim != 2 or track.shape[-1] != 2:
            return []
        end_idx = min(end_idx, len(track) - 1)
        arr = track[: end_idx + 1].copy()
        valid = np.isfinite(arr).all(axis=-1)
        if mask is not None:
            valid &= mask[: len(arr)]
        arr = arr[valid]
        if len(arr) == 0:
            return []
        arr[:, 0] = np.clip(arr[:, 0], 0, width - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, height - 1)
        return [(float(x), float(y)) for x, y in arr]

    def draw_path(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], color: tuple[int, int, int, int]) -> None:
        if len(pts) >= 2:
            draw.line(pts, fill=color[:3], width=3, joint="curve")
        radius = 3
        for x, y in pts:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color[:3])
        if pts:
            x, y = pts[-1]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 255, 255), width=2)

    for frame_idx, frame in enumerate(frames):
        image = to_image(frame)
        if frame_idx < pose_pixel_frames:
            width, height = image.size
            track_idx = min(frame_idx, track_steps - 1)
            draw = ImageDraw.Draw(image)
            if gt is not None:
                draw_path(draw, frame_points(gt, track_idx, width, height, visible), (0, 255, 0, 210))
            draw_path(draw, frame_points(pred, track_idx, width, height), (255, 0, 0, 220))
            draw.rectangle((6, 6, 156, 50), fill=(0, 0, 0))
            draw.text((12, 10), "GT track", fill=(0, 255, 0))
            draw.text((12, 28), "Pred track", fill=(255, 80, 80))
        rendered.append(np.asarray(image.convert("RGB"), dtype=np.uint8))

    import imageio

    with imageio.get_writer(str(output_path), fps=fps, quality=8, macro_block_size=16) as writer:
        for frame in rendered:
            writer.append_data(frame)


def save_eval_sample(
    *,
    split_root: Path,
    episode_name: str,
    item: dict[str, Any],
    pred_state: np.ndarray,
    pred_action: np.ndarray,
    pred_video: list[Any],
    fps: int,
    lora_dir: Path,
    infer_stage: str,
    action_has_gripper_prob: bool,
) -> None:
    gt_dir = split_root / "gt" / episode_name
    pred_dir = split_root / "pred" / episode_name
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(item["state_path"], gt_dir / "state.npy")
    shutil.copy2(item["action_path"], gt_dir / "action.npy")
    if item.get("video_path"):
        shutil.copy2(item["video_path"], gt_dir / "video.mp4")
    elif pred_video is not None:
        export_to_video(pred_video, str(gt_dir / "video.mp4"), fps=fps)
    write_text(gt_dir / "prompt.txt", item["prompt"])

    np.save(pred_dir / "state.npy", pred_state)
    np.save(pred_dir / "action.npy", pred_action)
    if action_has_gripper_prob:
        np.save(pred_dir / "action_gripper_binary.npy", (pred_action[..., 6] >= 0.5).astype(np.float32))
    if pred_video is not None:
        export_to_video(pred_video, str(pred_dir / "video.mp4"), fps=fps)
    if (gt_dir / "video.mp4").is_file():
        shutil.copy2(gt_dir / "video.mp4", pred_dir / "gt_video.mp4")
    write_text(pred_dir / "prompt.txt", item["prompt"])
    metadata = {
        "episode": episode_name,
        "prompt": item["prompt"],
        "image_path": item["image_path"],
        "video_path": item.get("video_path"),
        "state_path": item["state_path"],
        "action_path": item["action_path"],
        "lora_dir": str(lora_dir),
        "infer_stage": infer_stage,
        "action_has_gripper_prob": action_has_gripper_prob,
        "pred_state_path": str(pred_dir / "state.npy"),
        "pred_action_path": str(pred_dir / "action.npy"),
    }
    write_text(pred_dir / "metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))


def save_eval_track_sample(
    *,
    split_root: Path,
    episode_name: str,
    item: dict[str, Any],
    pred_track: np.ndarray,
    pred_video: list[Any],
    fps: int,
    lora_dir: Path,
    infer_stage: str,
    metrics: dict[str, float],
    pose_pixel_frames: int = 25,
) -> None:
    gt_dir = split_root / "gt" / episode_name
    pred_dir = split_root / "pred" / episode_name
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(item["state_path"], gt_dir / "state.npy")
    shutil.copy2(item["action_path"], gt_dir / "action.npy")
    if item.get("track_path"):
        shutil.copy2(item["track_path"], gt_dir / "track.npy")
    if item.get("track_visible_path"):
        shutil.copy2(item["track_visible_path"], gt_dir / "track_visible.npy")
    if item.get("video_path"):
        shutil.copy2(item["video_path"], gt_dir / "video.mp4")
    write_text(gt_dir / "prompt.txt", item["prompt"])

    np.save(pred_dir / "track.npy", pred_track.astype(np.float32))
    track_overlay_path = pred_dir / "track_overlay.png"
    gt_track = None
    gt_visible = None
    if item.get("track_path"):
        gt_track = np.load(item["track_path"]).astype(np.float32)[: pred_track.shape[0]]
    if item.get("track_visible_path"):
        gt_visible = np.load(item["track_visible_path"]).astype(bool)[: pred_track.shape[0]]
    draw_track_overlay(
        image_path=item["image_path"],
        pred_track=pred_track,
        gt_track=gt_track,
        gt_visible=gt_visible,
        output_path=track_overlay_path,
    )
    track_overlay_video_path = pred_dir / "track_overlay_video.mp4"
    if pred_video is not None:
        pred_video_path = pred_dir / "video.mp4"
        export_to_video(pred_video, str(pred_video_path), fps=fps)
        draw_track_overlay_video(
            video_path=pred_video_path,
            pred_track=pred_track,
            gt_track=gt_track,
            gt_visible=gt_visible,
            output_path=track_overlay_video_path,
            fps=fps,
            pose_pixel_frames=pose_pixel_frames,
        )
    if (gt_dir / "video.mp4").is_file():
        shutil.copy2(gt_dir / "video.mp4", pred_dir / "gt_video.mp4")
    write_text(pred_dir / "prompt.txt", item["prompt"])
    metadata = {
        "episode": episode_name,
        "prompt": item["prompt"],
        "image_path": item["image_path"],
        "video_path": item.get("video_path"),
        "state_path": item["state_path"],
        "action_path": item["action_path"],
        "track_path": item.get("track_path"),
        "lora_dir": str(lora_dir),
        "infer_stage": infer_stage,
        "pred_track_path": str(pred_dir / "track.npy"),
        "track_overlay_path": str(track_overlay_path),
        "track_overlay_video_path": str(track_overlay_video_path) if track_overlay_video_path.is_file() else None,
        "metrics": metrics,
    }
    write_text(pred_dir / "metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))


def run_eval_split(
    *,
    split_name: str,
    items: list[dict[str, Any]],
    eval_root: Path,
    pipe: CogVideoXImageToVideoPipeline,
    sa_tokenizer: StateActionTokenizer,
    s0_encoder: S0Encoder,
    norm_stats: dict[str, torch.Tensor],
    args: argparse.Namespace,
    generator: torch.Generator,
    lora_dir: Path,
    direct_action_modules: DirectActionModules | None = None,
) -> None:
    split_root = eval_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, item in enumerate(items, start=1):
        episode_name = f"episode_{idx:04d}"
        pred_state, pred_action, video = run_i2av_sample(
            pipe, sa_tokenizer, s0_encoder, norm_stats, item, args, generator, direct_action_modules
        )
        save_eval_sample(
            split_root=split_root,
            episode_name=episode_name,
            item=item,
            pred_state=pred_state,
            pred_action=pred_action,
            pred_video=video,
            fps=args.fps,
            lora_dir=lora_dir,
            infer_stage=args.infer_stage,
        action_has_gripper_prob=args.action_norm_stats_payload is not None and not args.gripper_continuous_action,
        )
        manifest.append(
            {
                "episode": episode_name,
                "prompt": item["prompt"],
                "image_path": item["image_path"],
                "video_path": item.get("video_path"),
                "state_path": item["state_path"],
                "action_path": item["action_path"],
            }
        )
        print(f"Wrote {split_name}/{episode_name}")
    write_text(split_root / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    if getattr(args, "simpler_sim_replay", False) and _items_have_simpler_meta(items):
        run_simpler_sim_replay(split_root=split_root, items=items, args=args)


def run_eval_split_v6(
    *,
    split_name: str,
    items: list[dict[str, Any]],
    eval_root: Path,
    pipe: CogVideoXImageToVideoPipeline,
    s0_encoder: S0Encoder,
    norm_stats: dict[str, torch.Tensor],
    track_norm_stats: dict[str, Any],
    track_modules: TrackModules,
    args: argparse.Namespace,
    generator: torch.Generator,
    lora_dir: Path,
) -> None:
    split_root = eval_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    metrics_rows = []
    for idx, item in enumerate(items, start=1):
        episode_name = f"episode_{idx:04d}"
        pred_track, video, metrics = run_i2av_v6_sample(
            pipe, s0_encoder, norm_stats, track_norm_stats, track_modules, item, args, generator
        )
        save_eval_track_sample(
            split_root=split_root,
            episode_name=episode_name,
            item=item,
            pred_track=pred_track,
            pred_video=video,
            fps=args.fps,
            lora_dir=lora_dir,
            infer_stage=args.infer_stage,
            metrics=metrics,
            pose_pixel_frames=args.pose_pixel_frames,
        )
        manifest.append(
            {
                "episode": episode_name,
                "prompt": item["prompt"],
                "image_path": item["image_path"],
                "video_path": item.get("video_path"),
                "state_path": item["state_path"],
                "action_path": item["action_path"],
                "track_path": item.get("track_path"),
                "pred_track_path": str(split_root / "pred" / episode_name / "track.npy"),
            }
        )
        metrics_rows.append(metrics)
        print(f"Wrote {split_name}/{episode_name} track_ADE={metrics.get('track_ade', float('nan')):.4f}")
        del pred_track, video, metrics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    write_text(split_root / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    if metrics_rows:
        summary = {
            key: float(np.mean([row[key] for row in metrics_rows if key in row]))
            for key in sorted(metrics_rows[0].keys())
        }
        write_text(split_root / "track_metrics.json", json.dumps(summary, indent=2, ensure_ascii=False))


def _items_have_simpler_meta(items: list[dict[str, Any]]) -> bool:
    for item in items:
        if item.get("simpler_task") or item.get("simpler_meta_path"):
            return True
    return False


def prepare_s0_norm(
    item: dict[str, Any],
    norm_stats: dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    dummy_state: bool = False,
) -> torch.Tensor:
    mean = norm_stats["mean"].to(device=device, dtype=torch.float32)
    std = norm_stats["std"].to(device=device, dtype=torch.float32)
    if dummy_state:
        return torch.zeros((1, mean.shape[-1]), device=device, dtype=dtype)
    state_seq = torch.from_numpy(np.load(item["state_path"]).astype(np.float32)).unsqueeze(0).to(device=device)
    return ((state_seq[:, 0] - mean) / std).to(dtype=dtype)


@torch.no_grad()
def run_i2av_v6_sample(
    pipe: CogVideoXImageToVideoPipeline,
    s0_encoder: S0Encoder,
    norm_stats: dict[str, torch.Tensor],
    track_norm_stats: dict[str, Any],
    track_modules: TrackModules,
    item: dict[str, Any],
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[np.ndarray, list[np.ndarray] | None, dict[str, float]]:
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    do_cfg = args.guidance_scale > 1.0
    layout = compute_i2av_v6_layout(
        pipe.transformer.config,
        pixel_height=args.height,
        pixel_width=args.width,
        pose_pixel_frames=args.pose_pixel_frames,
        rgb_pixel_frames=args.rgb_pixel_frames,
        text_seq_length=pipe.transformer.config.max_text_seq_length,
        s0_cond_tokens=s0_encoder.num_tokens,
        vae_scale_factor_spatial=pipe.vae_scale_factor_spatial,
    )
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=item["prompt"],
        negative_prompt=None,
        do_classifier_free_guidance=do_cfg,
        num_videos_per_prompt=1,
        max_sequence_length=pipe.transformer.config.max_text_seq_length,
        device=device,
        dtype=dtype,
    )
    if s0_encoder.num_tokens > 0:
        s0_norm = prepare_s0_norm(item, norm_stats, device=device, dtype=dtype, dummy_state=args.dummy_state)
        s0_cond = s0_encoder(s0_norm)
        prompt_embeds = torch.cat([prompt_embeds, s0_cond], dim=1)
        if do_cfg:
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, s0_cond], dim=1)
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, args.num_inference_steps, device, None)
    track_scheduler = prepare_action_epsilon_scheduler(pipe.scheduler, args.num_inference_steps, device, timesteps)
    track_timesteps = make_track_timesteps(timesteps, max_timestep=args.track_max_timestep)
    track_extra_step_kwargs = prepare_extra_step_kwargs(track_scheduler, generator, eta=0.0)

    latent_frames = (args.num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    num_frames = args.num_frames
    image = load_image(item["image_path"])
    image = pipe.video_processor.preprocess(image, height=args.height, width=args.width).to(device, dtype=dtype)
    latent_channels = pipe.transformer.config.in_channels // 2
    gt_video_frames = None
    if os.environ.get("V6_ZERO_VIDEO_COND", "0") == "1":
        latent_shape = (
            1,
            layout.num_latent_frames,
            latent_channels,
            args.height // pipe.vae_scale_factor_spatial,
            args.width // pipe.vae_scale_factor_spatial,
        )
        latents = torch.zeros(latent_shape, device=device, dtype=dtype)
        image_latents = torch.zeros_like(latents)
    else:
        freed_model_for_encode = os.environ.get("FREE_MODEL_BEFORE_ENCODE", "0") == "1" and torch.cuda.is_available()
        if freed_model_for_encode:
            pipe.transformer.to("cpu")
            if getattr(pipe, "text_encoder", None) is not None:
                pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        latents, image_latents = pipe.prepare_latents(
            image,
            1,
            latent_channels,
            num_frames,
            args.height,
            args.width,
            dtype,
            device,
            generator,
            None,
        )
        if args.infer_stage == "stage2":
            gt_video_frames = load_ground_truth_video_frames(item, num_frames)
            latents = encode_clean_video_latents(
                pipe,
                gt_video_frames,
                height=args.height,
                width=args.width,
                device=device,
                dtype=dtype,
            )
        if freed_model_for_encode:
            pipe.transformer.to(device=device, dtype=dtype)
            if getattr(pipe, "text_encoder", None) is not None:
                pipe.text_encoder.to(device=device)
    if latents.shape[1] != layout.num_latent_frames:
        raise ValueError(f"v6 expects {layout.num_latent_frames} latent frames, got {latents.shape[1]}.")

    track = randn_tensor((1, layout.track_steps, 2), generator=generator, device=device, dtype=dtype)
    track = track * track_scheduler.init_noise_sigma
    image_rotary_emb = prepare_i2av_v6_rotary_emb(pipe, args.height, args.width, layout, device)
    ofs_emb = None if pipe.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)
    extra_step_kwargs = prepare_extra_step_kwargs(pipe.scheduler, generator, eta=0.0)
    old_pred_original_sample = None
    old_track_pred_original_sample = None
    final_track = None

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
        for i, t in enumerate(timesteps):
            track_t = track_timesteps[i]
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            if args.infer_stage != "stage2":
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            latent_image_input = torch.cat([image_latents] * 2) if do_cfg else image_latents
            latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
            track_model_input = torch.cat([track] * 2) if do_cfg else track
            timestep = t.expand(latent_model_input.shape[0])
            track_timestep = track_t.expand(latent_model_input.shape[0])
            tokenizer_timesteps = track_timestep if getattr(args, "track_has_timestep_embedding", True) else None
            track_tokens = track_modules.track_tokenizer(track_model_input, timesteps=tokenizer_timesteps)
            noise_pred, track_hidden = forward_i2av_v6_transformer(
                pipe.transformer,
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                track_tokens=track_tokens,
                timestep=timestep,
                timestep_track=track_timestep,
                ofs=ofs_emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=None,
                layout=layout,
                return_dict=False,
            )
            track_noise_pred = track_modules.track_tokenizer.decode_noise(track_hidden).float()
            noise_pred = noise_pred.float()
            if do_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + args.guidance_scale * (noise_text - noise_uncond)
                track_uncond, track_text = track_noise_pred.chunk(2)
                track_noise_pred = track_uncond + args.guidance_scale * (track_text - track_uncond)
            if args.infer_stage != "stage2":
                if not isinstance(pipe.scheduler, CogVideoXDPMScheduler):
                    latents = pipe.scheduler.step(noise_pred.to(latents.device), t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = pipe.scheduler.step(
                        noise_pred.to(latents.device),
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(dtype)
            alpha_prod_t = track_scheduler.alphas_cumprod[track_t].to(device=track.device)
            if alpha_prod_t > 0:
                if not isinstance(track_scheduler, CogVideoXDPMScheduler):
                    track = track_scheduler.step(
                        track_noise_pred.to(track.device),
                        track_t,
                        track.float(),
                        **track_extra_step_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    track, old_track_pred_original_sample = track_scheduler.step(
                        track_noise_pred.to(track.device),
                        old_track_pred_original_sample,
                        track_t,
                        track_timesteps[i - 1] if i > 0 else None,
                        track.float(),
                        **track_extra_step_kwargs,
                        return_dict=False,
                    )
                track = track.to(dtype)
                final_track = track
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    if final_track is None:
        final_track = track
    pred_track = denormalize_track_pixels(final_track.float(), track_norm_stats).squeeze(0).cpu().numpy()
    frames = None
    if os.environ.get("SKIP_VIDEO_DECODE", "0") != "1":
        freed_model_for_decode = os.environ.get("FREE_MODEL_BEFORE_DECODE", "0") == "1" and torch.cuda.is_available()
        if freed_model_for_decode:
            pipe.transformer.to("cpu")
            if getattr(pipe, "text_encoder", None) is not None:
                pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        video = pipe.decode_latents(latents)
        frames = pipe.video_processor.postprocess_video(video=video, output_type="np")[0]
        if freed_model_for_decode:
            pipe.transformer.to(device=device, dtype=dtype)
            if getattr(pipe, "text_encoder", None) is not None:
                pipe.text_encoder.to(device=device)
    metrics: dict[str, float] = {}
    if item.get("track_path"):
        gt_track = np.load(item["track_path"]).astype(np.float32)[: pred_track.shape[0]]
        valid = np.ones(gt_track.shape[0], dtype=bool)
        if item.get("track_visible_path"):
            valid = np.load(item["track_visible_path"]).astype(bool)[: pred_track.shape[0]]
        dist = np.linalg.norm(pred_track[: gt_track.shape[0]] - gt_track, axis=-1)
        if valid.any():
            metrics["track_ade"] = float(dist[valid].mean())
            metrics["track_fde"] = float(dist[valid][-1])
        metrics["track_valid_fraction"] = float(valid.mean())
    return pred_track, frames if gt_video_frames is None else gt_video_frames, metrics


def run_simpler_sim_replay(
    *,
    split_root: Path,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    scripts_root = Path(__file__).resolve().parents[3] / "coaf_dataset_24_25" / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    from simpler_sim.replay import replay_eval_split

    output_root = split_root / "simpler_replay"
    results = replay_eval_split(
        eval_split_root=split_root,
        items=items,
        output_root=output_root,
        action_format=getattr(args, "simpler_action_format", "bridge_v2"),
        simpler_root=getattr(args, "simpler_root", None),
    )
    success = sum(1 for row in results if row.get("success"))
    print(f"Simpler replay: {success}/{len(results)} success -> {output_root}")


@torch.no_grad()
def run_i2av_sample(
    pipe: CogVideoXImageToVideoPipeline,
    sa_tokenizer: StateActionTokenizer,
    s0_encoder: S0Encoder,
    norm_stats: dict[str, torch.Tensor],
    item: dict[str, Any],
    args: argparse.Namespace,
    generator: torch.Generator,
    direct_action_modules: DirectActionModules | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    sa_guidance_scale = args.sa_guidance_scale if args.sa_guidance_scale is not None else args.guidance_scale
    do_cfg = args.guidance_scale > 1.0 or sa_guidance_scale > 1.0
    layout = None
    if args.i2av_layout == "v5":
        layout = compute_i2av_v5_layout(
            pipe.transformer.config,
            pixel_height=args.height,
            pixel_width=args.width,
            pose_pixel_frames=args.pose_pixel_frames,
            rgb_pixel_frames=args.rgb_pixel_frames,
            text_seq_length=pipe.transformer.config.max_text_seq_length,
            s0_cond_tokens=s0_encoder.num_tokens,
            vae_scale_factor_spatial=pipe.vae_scale_factor_spatial,
        )
        sa_per_frame = layout.chunk_token_count
    else:
        sa_per_frame = sa_tokenizer.num_tokens

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=item["prompt"],
        negative_prompt=None,
        do_classifier_free_guidance=do_cfg,
        num_videos_per_prompt=1,
        max_sequence_length=pipe.transformer.config.max_text_seq_length,
        device=device,
        dtype=dtype,
    )

    if s0_encoder.num_tokens > 0:
        s0_norm = prepare_s0_norm(item, norm_stats, device=device, dtype=dtype, dummy_state=args.dummy_state)
        s0_cond = s0_encoder(s0_norm)
        prompt_embeds = torch.cat([prompt_embeds, s0_cond], dim=1)
        if do_cfg:
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, s0_cond], dim=1)
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, args.num_inference_steps, device, None)
    pipe._guidance_scale = args.guidance_scale
    pipe._current_timestep = None
    pipe._attention_kwargs = None
    pipe._interrupt = False
    pipe._num_timesteps = len(timesteps)
    action_scheduler = None
    action_extra_step_kwargs = None
    if args.direct_action_head:
        action_scheduler = prepare_action_epsilon_scheduler(pipe.scheduler, args.num_inference_steps, device, timesteps)
        action_extra_step_kwargs = prepare_extra_step_kwargs(action_scheduler, generator, eta=0.0)

    latent_frames = (args.num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    patch_size_t = pipe.transformer.config.patch_size_t
    additional_frames = 0
    num_frames = args.num_frames
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = patch_size_t - latent_frames % patch_size_t
        num_frames += additional_frames * pipe.vae_scale_factor_temporal

    image = load_image(item["image_path"])
    image = pipe.video_processor.preprocess(image, height=args.height, width=args.width).to(device, dtype=dtype)
    latent_channels = pipe.transformer.config.in_channels // 2
    latents, image_latents = pipe.prepare_latents(
        image,
        1,
        latent_channels,
        num_frames,
        args.height,
        args.width,
        dtype,
        device,
        generator,
        None,
    )
    gt_video_frames = None
    if args.infer_stage == "stage2":
        gt_video_frames = load_ground_truth_video_frames(item, num_frames)
        latents = encode_clean_video_latents(
            pipe,
            gt_video_frames,
            height=args.height,
            width=args.width,
            device=device,
            dtype=dtype,
        )

    latent_frames = latents.shape[1]
    if layout is not None and latent_frames != layout.num_latent_frames:
        raise ValueError(
            f"Stage {args.infer_stage} produced {latent_frames} latent frames, "
            f"but v5 layout expects {layout.num_latent_frames}. "
            f"Check GT video frame count and MAX_NUM_FRAMES."
        )
    grid_h = args.height // (pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size)
    grid_w = args.width // (pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size)
    patches_per_frame = grid_h * grid_w
    action_norm_stats = getattr(args, "action_norm_stats_payload", None)
    if args.direct_action_head:
        if layout is None:
            raise RuntimeError("Direct-action inference requires i2av_layout=v5.")
        if direct_action_modules is None:
            raise RuntimeError("Direct-action inference requires loaded direct_action_modules.")
        action_chunks = randn_tensor(
            (1, layout.num_pose_latent_frames, layout.steps_per_chunk, 7),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        action_chunks = action_chunks * action_scheduler.init_noise_sigma
    elif args.infer_stage == "stage1":
        if args.i2av_layout == "v5" and action_norm_stats is not None:
            action_seq = torch.from_numpy(np.load(item["action_path"]).astype(np.float32)).unsqueeze(0).to(device=device)
            state_gt, action_gt, _, _, _ = prepare_raw_action_gt_chunked(
                state_seq,
                action_seq,
                norm_stats,
                action_norm_stats,
                pose_pixel_frames=args.pose_pixel_frames,
                steps_per_chunk=layout.steps_per_chunk,
                gripper_continuous=args.gripper_continuous_action,
            )
        elif args.i2av_layout == "v5":
            state_gt, action_gt, _, _ = prepare_gt_chunked(
                state_seq,
                norm_stats,
                pose_pixel_frames=args.pose_pixel_frames,
                steps_per_chunk=layout.steps_per_chunk,
            )
        else:
            state_gt, action_gt, _ = prepare_gt(state_seq, norm_stats, num_latent_frames=latent_frames)
        sa_tokens = sa_tokenizer.encode(state_gt.to(dtype=dtype), action_gt.to(dtype=dtype))
    else:
        sa_frames = layout.num_pose_latent_frames if layout is not None else latent_frames
        sa_tokens = randn_tensor(
            (1, sa_frames, sa_per_frame, sa_tokenizer.hidden_dim),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        sa_tokens = sa_tokens * pipe.scheduler.init_noise_sigma

    if args.i2av_layout == "v5":
        image_rotary_emb = prepare_i2av_v5_rotary_emb(
            pipe,
            args.height,
            args.width,
            layout,
            device,
            direct_action_head=args.direct_action_head,
        )
    else:
        image_rotary_emb = prepare_i2av_rotary_emb(pipe, args.height, args.width, latent_frames, device, sa_per_frame)
    ofs_emb = None if pipe.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)
    extra_step_kwargs = prepare_extra_step_kwargs(pipe.scheduler, generator, eta=0.0)
    old_pred_original_sample = None
    old_sa_pred_original_sample = None
    old_action_pred_original_sample = None
    final_sa_pred = None
    final_action_chunks = None

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
        for i, t in enumerate(timesteps):
            pipe._current_timestep = t
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            if args.infer_stage != "stage2":
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            latent_image_input = torch.cat([image_latents] * 2) if do_cfg else image_latents
            latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
            if args.direct_action_head:
                action_model_input = torch.cat([action_chunks] * 2) if do_cfg else action_chunks
                action_tokens = direct_action_modules.action_tokenizer(action_model_input)
            else:
                sa_model_input = torch.cat([sa_tokens] * 2) if do_cfg else sa_tokens
            timestep = t.expand(latent_model_input.shape[0])

            cache_context = getattr(pipe.transformer, "cache_context", None)
            transformer_cache_context = cache_context("cond_uncond") if callable(cache_context) else nullcontext()
            with transformer_cache_context:
                if args.direct_action_head:
                    noise_pred, action_token_hidden = forward_i2av_v5_direct_action_transformer(
                        pipe.transformer,
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        action_tokens=action_tokens,
                        timestep=timestep,
                        ofs=ofs_emb,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=None,
                        layout=layout,
                        return_dict=False,
                    )
                    _, action_pred_chunks = direct_action_modules.action_head(action_token_hidden)
                elif args.i2av_layout == "v5":
                    noise_pred, sa_pred = forward_i2av_v5_transformer(
                        pipe.transformer,
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        noisy_chunk_tokens=sa_model_input,
                        timestep=timestep,
                        ofs=ofs_emb,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=None,
                        layout=layout,
                        return_dict=False,
                    )
                else:
                    noise_pred, sa_pred = forward_i2av_transformer(
                        pipe.transformer,
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        noisy_sa_tokens=sa_model_input,
                        timestep=timestep,
                        ofs=ofs_emb,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=None,
                        patches_per_frame=patches_per_frame,
                        sa_per_frame=sa_per_frame,
                        return_dict=False,
                    )
            noise_pred = noise_pred.float()
            if args.direct_action_head:
                action_pred_chunks = action_pred_chunks.float()
            else:
                sa_pred = sa_pred.float()

            if do_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + args.guidance_scale * (noise_text - noise_uncond)
                if args.direct_action_head:
                    action_uncond, action_text = action_pred_chunks.chunk(2)
                    action_pred_chunks = action_uncond + sa_guidance_scale * (action_text - action_uncond)
                else:
                    sa_uncond, sa_text = sa_pred.chunk(2)
                    sa_pred = sa_uncond + sa_guidance_scale * (sa_text - sa_uncond)

            if args.infer_stage != "stage2":
                scheduler_noise_pred = noise_pred.to(device=latents.device)
                if not isinstance(pipe.scheduler, CogVideoXDPMScheduler):
                    latents = pipe.scheduler.step(scheduler_noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = pipe.scheduler.step(
                        scheduler_noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(dtype)
            if args.direct_action_head:
                action_noise_pred = action_pred_chunks.to(device=action_chunks.device)
                alpha_prod_t = action_scheduler.alphas_cumprod[t].to(device=action_chunks.device)
                if alpha_prod_t <= 0:
                    # CogVideoX DDIM uses zero terminal SNR, so the first trailing
                    # timestep has alpha=0. Epsilon prediction cannot reconstruct
                    # x0 from pure noise at that step; skip to the next finite-alpha step.
                    final_action_chunks = action_chunks
                elif not isinstance(action_scheduler, CogVideoXDPMScheduler):
                    action_chunks = action_scheduler.step(
                        action_noise_pred,
                        t,
                        action_chunks.float(),
                        **action_extra_step_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    action_chunks, old_action_pred_original_sample = action_scheduler.step(
                        action_noise_pred,
                        old_action_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        action_chunks.float(),
                        **action_extra_step_kwargs,
                        return_dict=False,
                    )
                if alpha_prod_t > 0:
                    action_chunks = action_chunks.to(dtype)
                    final_action_chunks = action_chunks
            elif args.infer_stage == "stage1":
                final_sa_pred = sa_tokens
            elif args.sa_denoise_loss:
                sa_noise_pred = sa_pred.to(device=sa_tokens.device)
                if not isinstance(pipe.scheduler, CogVideoXDPMScheduler):
                    sa_tokens = pipe.scheduler.step(
                        sa_noise_pred,
                        t,
                        sa_tokens.float(),
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    sa_tokens, old_sa_pred_original_sample = pipe.scheduler.step(
                        sa_noise_pred,
                        old_sa_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        sa_tokens.float(),
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                sa_tokens = sa_tokens.to(dtype)
                final_sa_pred = sa_tokens
            else:
                # SA/chunk tokens are trained with direct decoded state/action loss,
                # not a scheduler velocity/noise loss. Feed the predicted clean token
                # estimate into the next denoising step.
                sa_tokens = sa_pred.to(dtype)
                final_sa_pred = sa_tokens

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    pipe._current_timestep = None
    if additional_frames:
        latents = latents[:, additional_frames:]

    frames = None
    if os.environ.get("SKIP_VIDEO_DECODE", "0") != "1":
        freed_model_for_decode = os.environ.get("FREE_MODEL_BEFORE_DECODE", "0") == "1" and torch.cuda.is_available()
        if freed_model_for_decode:
            pipe.transformer.to("cpu")
            if getattr(pipe, "text_encoder", None) is not None:
                pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        video = pipe.decode_latents(latents)
        frames = pipe.video_processor.postprocess_video(video=video, output_type="np")[0]
        if freed_model_for_decode:
            pipe.transformer.to(device=device, dtype=dtype)
            if getattr(pipe, "text_encoder", None) is not None:
                pipe.text_encoder.to(device=device)

    if args.direct_action_head:
        if final_action_chunks is None:
            raise RuntimeError("Direct-action denoising produced no action prediction.")
        pred_action_norm = flatten_action_chunks(final_action_chunks.float(), real_steps=args.direct_action_horizon)
        pred_state = state_seq[:, : pred_action_norm.shape[1]].float()
    elif final_sa_pred is None:
        raise RuntimeError("I2AV denoising produced no state/action prediction.")
    else:
        pred_state_norm, pred_action_norm = sa_tokenizer.decode(final_sa_pred)
        pred_state = pred_state_norm.float() * std + mean
    action_norm_stats = getattr(args, "action_norm_stats_payload", None)
    if action_norm_stats is not None:
        pred_action = torch.empty_like(pred_action_norm.float())
        action_norm_method = get_action_norm_method(action_norm_stats)
        if action_norm_method == "quantile":
            action_q01 = action_norm_stats["q01"].to(device=device, dtype=torch.float32)
            action_q99 = action_norm_stats["q99"].to(device=device, dtype=torch.float32)
            action_span = (action_q99 - action_q01).clamp_min(1e-6)
            if args.gripper_continuous_action:
                pred_action = (pred_action_norm.float() + 1.0) * 0.5 * action_span + action_q01
            else:
                pred_action[..., :6] = pred_action_norm.float()[..., :6] * 0.5 * action_span[:6] + (
                    action_q01[:6] + action_span[:6] * 0.5
                )
                pred_action[..., 6] = torch.sigmoid(pred_action_norm.float()[..., 6])
        elif action_norm_method == "mean_std":
            action_mean = action_norm_stats["mean"].to(device=device, dtype=torch.float32)
            action_std = action_norm_stats["std"].to(device=device, dtype=torch.float32).clamp_min(1e-6)
            pred_action[..., :6] = pred_action_norm.float()[..., :6] * action_std[:6] + action_mean[:6]
            if args.gripper_continuous_action:
                pred_action[..., 6] = pred_action_norm.float()[..., 6] * action_std[6] + action_mean[6]
            else:
                pred_action[..., 6] = torch.sigmoid(pred_action_norm.float()[..., 6])
        else:
            raise ValueError(f"Unsupported action norm method: {action_norm_method}")
    else:
        pred_action = pred_action_norm.float() * std
    if gt_video_frames is not None:
        return pred_state.squeeze(0).cpu().numpy(), pred_action.squeeze(0).cpu().numpy(), gt_video_frames
    return pred_state.squeeze(0).cpu().numpy(), pred_action.squeeze(0).cpu().numpy(), frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--train_data_root", type=Path)
    parser.add_argument("--lora_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--state_norm_stats", type=Path)
    parser.add_argument("--action_norm_stats", type=Path)
    parser.add_argument("--gripper_continuous_action", action="store_true")
    parser.add_argument("--sa_denoise_loss", action="store_true")
    parser.add_argument("--direct_action_head", action="store_true")
    parser.add_argument("--direct_action_horizon", type=int, default=25)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument(
        "--sa_guidance_scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale for state/action tokens. Defaults to guidance_scale.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--train_num_samples", type=int, default=0)
    parser.add_argument("--simpler_data_root", type=Path)
    parser.add_argument("--simpler_num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--i2av_layout", choices=["legacy", "v5", "v6"], default=None)
    parser.add_argument(
        "--action_chunk_alignment",
        choices=["current", "next_window"],
        default=None,
        help="Override checkpoint action chunk alignment; defaults to state_action.pt metadata.",
    )
    parser.add_argument("--pose_pixel_frames", type=int, default=25)
    parser.add_argument("--rgb_pixel_frames", type=int, default=24)
    parser.add_argument("--track_norm_stats", type=Path)
    parser.add_argument(
        "--dummy_state",
        action="store_true",
        help="Use a zero normalized S0 condition instead of loading state_path during inference.",
    )
    parser.add_argument(
        "--track_max_timestep",
        type=int,
        default=None,
        help="For v6 track diffusion, start track denoising from at most this timestep while video uses the normal schedule.",
    )
    parser.add_argument("--infer_stage", choices=["stage1", "stage2", "stage3", "joint"], default="joint")
    parser.add_argument("--device", default="cuda", help="Inference device. Use cuda for fast custom I2AV forward.")
    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    parser.add_argument(
        "--simpler_sim_replay",
        action="store_true",
        help="After eval_dataset inference, replay predicted actions in SimplerEnv (BridgeV2 camera).",
    )
    parser.add_argument("--simpler_root", type=Path, default=Path("/mnt/disk1/sunkai/SimplerEnv"))
    parser.add_argument(
        "--simpler_action_format",
        choices=["bridge_v2", "env"],
        default="bridge_v2",
        help="Action format for Simpler replay (default: bridge_v2).",
    )
    args = parser.parse_args()

    if args.infer_stage == "stage1" and "clean_sa" not in args.output_dir.name:
        args.output_dir = args.output_dir.with_name(f"{args.output_dir.name}_clean_sa")

    lora_dir = resolve_lora_dir(args.lora_dir)
    state_action_path = lora_dir / "state_action.pt"
    if not state_action_path.is_file():
        raise FileNotFoundError(f"I2AV checkpoint is missing state_action.pt under {lora_dir}")
    direct_action_path = lora_dir / "direct_action.pt"
    if direct_action_path.is_file():
        args.direct_action_head = True
    if args.direct_action_head and not direct_action_path.is_file():
        raise FileNotFoundError(f"Direct-action checkpoint is missing direct_action.pt under {lora_dir}")
    track_path = lora_dir / "track.pt"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    disable_learned_positional_embeddings(pipe)
    pipe.load_lora_weights(str(lora_dir), adapter_name="cogvideox-lora")
    pipe.set_adapters(["cogvideox-lora"], [1.0])
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
        inference_device = pipe._execution_device
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        inference_device = torch.device(args.device)
        pipe.to(inference_device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    print(f"Inference device: {inference_device}")
    print(f"Model CPU offload: {args.enable_model_cpu_offload}")

    hidden_dim = get_transformer_hidden_dim(pipe.transformer)
    text_embed_dim = get_text_embed_dim(pipe.transformer)
    state_action_payload = torch.load(state_action_path, map_location="cpu", weights_only=False)
    checkpoint_layout = state_action_payload.get("tokenizer_type", "legacy")
    args.i2av_layout = args.i2av_layout or ("v5" if checkpoint_layout == "v5" else checkpoint_layout)
    s0_cond_tokens = int(state_action_payload.get("s0_cond_tokens", 4))
    action_chunk_alignment = args.action_chunk_alignment or str(
        state_action_payload.get("action_chunk_alignment", "current")
    )
    if args.i2av_layout == "v5":
        layout = compute_i2av_v5_layout(
            pipe.transformer.config,
            pixel_height=args.height,
            pixel_width=args.width,
            pose_pixel_frames=args.pose_pixel_frames,
            rgb_pixel_frames=args.rgb_pixel_frames,
            text_seq_length=pipe.transformer.config.max_text_seq_length,
            s0_cond_tokens=s0_cond_tokens,
            vae_scale_factor_spatial=pipe.vae_scale_factor_spatial,
        )
        sa_tokenizer = ChunkedStateActionTokenizer(
            hidden_dim=hidden_dim,
            steps_per_chunk=int(state_action_payload.get("steps_per_chunk", layout.steps_per_chunk)),
            first_chunk_pad_steps=layout.first_chunk_pad_steps,
            real_trajectory_steps=layout.real_trajectory_steps,
            action_chunk_alignment=action_chunk_alignment,
        )
    elif args.i2av_layout == "v6":
        layout = compute_i2av_v6_layout(
            pipe.transformer.config,
            pixel_height=args.height,
            pixel_width=args.width,
            pose_pixel_frames=args.pose_pixel_frames,
            rgb_pixel_frames=args.rgb_pixel_frames,
            text_seq_length=pipe.transformer.config.max_text_seq_length,
            s0_cond_tokens=s0_cond_tokens,
            vae_scale_factor_spatial=pipe.vae_scale_factor_spatial,
        )
        sa_tokenizer = StateActionTokenizer(hidden_dim=hidden_dim, num_state_tokens=4, num_action_tokens=4)
    else:
        sa_tokenizer = StateActionTokenizer(hidden_dim=hidden_dim, num_state_tokens=4, num_action_tokens=4)
    s0_encoder = S0Encoder(hidden_dim=text_embed_dim, num_tokens=s0_cond_tokens)
    load_state_action_modules(str(state_action_path), sa_tokenizer, s0_encoder, device=inference_device)
    sa_tokenizer.to(device=inference_device, dtype=torch.bfloat16).eval()
    s0_encoder.to(device=inference_device, dtype=torch.bfloat16).eval()
    direct_action_modules = None
    if args.direct_action_head:
        if args.i2av_layout != "v5":
            raise RuntimeError("Direct-action inference requires i2av_layout=v5.")
        direct_action_modules = DirectActionModules(
            hidden_dim=hidden_dim,
            action_dim=7,
            steps_per_chunk=layout.steps_per_chunk,
        )
        direct_payload = torch.load(direct_action_path, map_location=inference_device, weights_only=False)
        direct_action_modules.load_state_dict(direct_payload["direct_action_modules"])
        direct_action_modules.to(device=inference_device, dtype=torch.bfloat16).eval()
    track_modules = None
    track_norm_stats = None
    if args.i2av_layout == "v6":
        if not track_path.is_file():
            raise FileNotFoundError(f"I2AV v6 checkpoint is missing track.pt under {lora_dir}")
        track_payload = torch.load(track_path, map_location=inference_device, weights_only=False)
        track_modules = TrackModules(hidden_dim=hidden_dim, num_steps=args.pose_pixel_frames)
        track_state = track_payload["track_modules"]
        args.track_has_timestep_embedding = any("timestep_embedding" in key for key in track_state)
        missing, unexpected = track_modules.load_state_dict(track_state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading track.pt: {unexpected}")
        if missing and args.track_has_timestep_embedding:
            raise RuntimeError(f"Missing keys when loading track.pt: {missing}")
        if missing:
            print(f"Loaded legacy track checkpoint without keys: {missing}")
        track_modules.to(device=inference_device, dtype=torch.bfloat16).eval()
        track_stats_path = args.track_norm_stats or Path(track_payload.get("track_norm_stats", ""))
        if not track_stats_path.is_file():
            track_stats_path = args.data_root / "track_norm_stats.pt"
        if not track_stats_path.is_file():
            raise FileNotFoundError(f"Missing track norm stats: {track_stats_path}")
        track_norm_stats = torch.load(track_stats_path, map_location="cpu", weights_only=False)

    install_temporal_causal_attention(
        pipe.transformer,
        num_pixel_frames=args.num_frames,
        pixel_height=args.height,
        pixel_width=args.width,
        text_seq_length=pipe.transformer.config.max_text_seq_length + s0_encoder.num_tokens,
        vae_scale_factor_spatial=pipe.vae_scale_factor_spatial,
        device=inference_device,
        dtype=torch.float32,
        enable_state_action=True,
        sa_per_frame=getattr(sa_tokenizer, "num_tokens", getattr(sa_tokenizer, "chunk_token_count", 8)),
        s0_cond_tokens=s0_encoder.num_tokens,
        i2av_layout=args.i2av_layout,
        pose_pixel_frames=args.pose_pixel_frames,
        rgb_pixel_frames=args.rgb_pixel_frames,
        direct_action_head=args.direct_action_head,
    )

    if s0_encoder.num_tokens > 0:
        if args.state_norm_stats is None:
            raise ValueError("This checkpoint uses S0 condition tokens and requires --state_norm_stats.")
        norm_stats = torch.load(args.state_norm_stats, map_location="cpu")
    else:
        norm_stats = {}
    needs_action_norm_stats = args.action_norm_stats is not None and (
        args.direct_action_head or args.i2av_layout != "v6"
    )
    args.action_norm_stats_payload = (
        torch.load(args.action_norm_stats, map_location="cpu", weights_only=False)
        if needs_action_norm_stats
        else None
    )
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed)
    if args.train_data_root is not None:
        eval_root = args.output_dir / "eval_dataset"
        validation_items = load_validation_items(args.data_root, args.num_samples)
        if args.i2av_layout == "v6":
            run_eval_split_v6(
                split_name="validation",
                items=validation_items,
                eval_root=eval_root,
                pipe=pipe,
                s0_encoder=s0_encoder,
                norm_stats=norm_stats,
                track_norm_stats=track_norm_stats,
                track_modules=track_modules,
                args=args,
                generator=generator,
                lora_dir=lora_dir,
            )
        else:
            run_eval_split(
                split_name="validation",
                items=validation_items,
                eval_root=eval_root,
                pipe=pipe,
                sa_tokenizer=sa_tokenizer,
                s0_encoder=s0_encoder,
                norm_stats=norm_stats,
                args=args,
                generator=generator,
                lora_dir=lora_dir,
                direct_action_modules=direct_action_modules,
            )
        train_num_samples = args.train_num_samples if args.train_num_samples > 0 else args.num_samples
        train_items = load_validation_items(args.train_data_root, train_num_samples)
        if args.i2av_layout == "v6":
            run_eval_split_v6(
                split_name="train",
                items=train_items,
                eval_root=eval_root,
                pipe=pipe,
                s0_encoder=s0_encoder,
                norm_stats=norm_stats,
                track_norm_stats=track_norm_stats,
                track_modules=track_modules,
                args=args,
                generator=generator,
                lora_dir=lora_dir,
            )
        else:
            run_eval_split(
                split_name="train",
                items=train_items,
                eval_root=eval_root,
                pipe=pipe,
                sa_tokenizer=sa_tokenizer,
                s0_encoder=s0_encoder,
                norm_stats=norm_stats,
                args=args,
                generator=generator,
                lora_dir=lora_dir,
                direct_action_modules=direct_action_modules,
            )
        if args.simpler_data_root is not None:
            simpler_items = load_validation_items(args.simpler_data_root, args.simpler_num_samples)
            if args.i2av_layout == "v6":
                run_eval_split_v6(
                    split_name="simpler",
                    items=simpler_items,
                    eval_root=eval_root,
                    pipe=pipe,
                    s0_encoder=s0_encoder,
                    norm_stats=norm_stats,
                    track_norm_stats=track_norm_stats,
                    track_modules=track_modules,
                    args=args,
                    generator=generator,
                    lora_dir=lora_dir,
                )
            else:
                run_eval_split(
                    split_name="simpler",
                    items=simpler_items,
                    eval_root=eval_root,
                    pipe=pipe,
                    sa_tokenizer=sa_tokenizer,
                    s0_encoder=s0_encoder,
                    norm_stats=norm_stats,
                    args=args,
                    generator=generator,
                    lora_dir=lora_dir,
                    direct_action_modules=direct_action_modules,
                )
        return

    for idx, item in enumerate(load_validation_items(args.data_root, args.num_samples)):
        if args.i2av_layout == "v6":
            pred_track, video, metrics = run_i2av_v6_sample(
                pipe,
                s0_encoder,
                norm_stats,
                track_norm_stats,
                track_modules,
                item,
                args,
                generator,
            )
            stem = f"sample_{idx:03d}"
            video_path = args.output_dir / f"{stem}.mp4"
            pred_track_path = args.output_dir / f"{stem}_pred_track.npy"
            np.save(pred_track_path, pred_track)
            if video is not None:
                export_to_video(video, str(video_path), fps=args.fps)
            print(json.dumps({"sample": stem, "pred_track_path": str(pred_track_path), **metrics}, indent=2))
            continue
        pred_state, pred_action, video = run_i2av_sample(
            pipe,
            sa_tokenizer,
            s0_encoder,
            norm_stats,
            item,
            args,
            generator,
            direct_action_modules,
        )

        stem = f"sample_{idx:03d}"
        video_path = args.output_dir / f"{stem}.mp4"
        pred_state_path = args.output_dir / f"{stem}_pred_state.npy"
        pred_action_path = args.output_dir / f"{stem}_pred_action.npy"
        np.save(pred_state_path, pred_state)
        np.save(pred_action_path, pred_action)
        action_gripper_binary_path = None
        if args.action_norm_stats_payload is not None and not args.gripper_continuous_action:
            action_gripper_binary_path = args.output_dir / f"{stem}_pred_action_gripper_binary.npy"
            np.save(action_gripper_binary_path, (pred_action[..., 6] >= 0.5).astype(np.float32))
        export_to_video(video, str(video_path), fps=args.fps)

        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "prompt": item["prompt"],
                    "image_path": item["image_path"],
                    "video_path": item.get("video_path"),
                    "state_path": item["state_path"],
                    "action_path": item["action_path"],
                    "lora_dir": str(lora_dir),
                    "infer_stage": args.infer_stage,
                    "action_has_gripper_prob": args.action_norm_stats_payload is not None and not args.gripper_continuous_action,
                    "gripper_continuous_action": args.gripper_continuous_action,
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.num_frames,
                    "pred_state_path": str(pred_state_path),
                    "pred_action_path": str(pred_action_path),
                    "pred_action_gripper_binary_path": (
                        str(action_gripper_binary_path) if action_gripper_binary_path is not None else None
                    ),
                    "pred_state": pred_state.tolist(),
                    "pred_action": pred_action.tolist(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {video_path}")
        print(f"Wrote {pred_action_path}")


if __name__ == "__main__":
    main()
