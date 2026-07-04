#!/usr/bin/env python3
"""Evaluate v6 stage3 validation loss for a list of checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXTransformer3DModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm


CASUAL_ROOT = Path(__file__).resolve().parents[1]
COGVIDEOX_ROOT = CASUAL_ROOT / "finetrainers" / "examples" / "_legacy" / "training" / "cogvideox"
FINETRAINERS_ROOT = CASUAL_ROOT / "finetrainers"
for path in (str(FINETRAINERS_ROOT), str(COGVIDEOX_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset_i2av import I2AVCollateFunction, I2AVVideoDataset  # noqa: E402
from finetrainers.patches.models.cogvideox.causal_attention import install_temporal_causal_attention  # noqa: E402
from finetrainers.patches.models.cogvideox.i2av_layout import compute_i2av_v6_layout  # noqa: E402
from finetrainers.patches.models.cogvideox.i2av_v6_forward import forward_i2av_v6_transformer  # noqa: E402
from finetrainers.patches.models.cogvideox.state_action import S0Encoder, StateActionTokenizer, load_state_action_modules  # noqa: E402
from finetrainers.patches.models.cogvideox.track_tokenizer import (  # noqa: E402
    TrackModules,
    compute_track_noise_loss,
    noise_track,
    normalize_track_pixels,
)
from utils import prepare_i2av_v6_rotary_positional_embeddings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--state_norm_stats", type=Path)
    parser.add_argument("--track_norm_stats", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--pose_pixel_frames", type=int, default=25)
    parser.add_argument("--rgb_pixel_frames", type=int, default=24)
    parser.add_argument("--s0_cond_tokens", type=int, default=0)
    parser.add_argument("--sa_per_frame", type=int, default=8)
    parser.add_argument("--lambda_sa", type=float, default=0.1)
    parser.add_argument("--lambda_track", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def get_hidden_dim(transformer: CogVideoXTransformer3DModel) -> int:
    cfg = transformer.config
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    return int(cfg.num_attention_heads * cfg.attention_head_dim)


def get_text_embed_dim(transformer: CogVideoXTransformer3DModel) -> int:
    cfg = transformer.config
    if hasattr(cfg, "text_embed_dim"):
        return int(cfg.text_embed_dim)
    return int(transformer.patch_embed.text_proj.in_features)


def validation_indices(data_root: Path, max_samples: int | None = None) -> list[int]:
    payload = json.loads((data_root / "validation.json").read_text(encoding="utf-8"))
    rows = payload.get("data", payload)
    indices = [int(item.get("sample_index", i)) for i, item in enumerate(rows)]
    if max_samples is not None and max_samples > 0:
        indices = indices[:max_samples]
    return indices


def load_checkpoint(transformer: CogVideoXTransformer3DModel, s0_encoder: S0Encoder, track_modules: TrackModules, checkpoint: Path, device: torch.device) -> None:
    load_state_action_modules(checkpoint / "state_action.pt", StateActionTokenizer(hidden_dim=get_hidden_dim(transformer)), s0_encoder, device=device)
    # load_state_action_modules above only needs the S0 weights for v6; keep the temporary tokenizer discarded.
    track_payload = torch.load(checkpoint / "track.pt", map_location=device, weights_only=False)
    track_modules.load_state_dict(track_payload["track_modules"])
    lora_state_dict = {}
    from diffusers import CogVideoXImageToVideoPipeline

    raw_lora = CogVideoXImageToVideoPipeline.lora_state_dict(str(checkpoint))
    transformer_state_dict = {
        k.replace("transformer.", ""): v for k, v in raw_lora.items() if k.startswith("transformer.")
    }
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")


def main() -> int:
    args = parse_args()
    if args.s0_cond_tokens > 0 and args.state_norm_stats is None:
        raise ValueError("--state_norm_stats is required when --s0_cond_tokens > 0.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    if hasattr(transformer.patch_embed, "pos_embedding"):
        del transformer.patch_embed.pos_embedding
    transformer.patch_embed.use_learned_positional_embeddings = False
    transformer.config.use_learned_positional_embeddings = False

    vae = AutoencoderKLCogVideoX.from_pretrained(args.model_path, subfolder="vae")
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    vae.enable_slicing()
    vae.enable_tiling()
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    hidden_dim = get_hidden_dim(transformer)
    text_embed_dim = get_text_embed_dim(transformer)
    s0_encoder = S0Encoder(hidden_dim=text_embed_dim, num_tokens=args.s0_cond_tokens)
    track_modules = TrackModules(hidden_dim=hidden_dim, num_steps=args.pose_pixel_frames)
    transformer.add_adapter(LoraConfig(r=128, lora_alpha=128, init_lora_weights=True, target_modules=["to_k", "to_q", "to_v", "to_out.0"]))

    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    layout = compute_i2av_v6_layout(
        transformer.config,
        pixel_height=args.height,
        pixel_width=args.width,
        pose_pixel_frames=args.pose_pixel_frames,
        rgb_pixel_frames=args.rgb_pixel_frames,
        text_seq_length=transformer.config.max_text_seq_length,
        s0_cond_tokens=args.s0_cond_tokens,
        vae_scale_factor_spatial=vae_scale_factor_spatial,
        track_steps=args.pose_pixel_frames,
    )
    install_temporal_causal_attention(
        transformer,
        num_pixel_frames=args.num_frames,
        pixel_height=args.height,
        pixel_width=args.width,
        text_seq_length=transformer.config.max_text_seq_length + args.s0_cond_tokens,
        vae_scale_factor_spatial=vae_scale_factor_spatial,
        device=device,
        dtype=torch.float32,
        enable_state_action=True,
        sa_per_frame=args.sa_per_frame,
        s0_cond_tokens=args.s0_cond_tokens,
        i2av_layout="v6",
        pose_pixel_frames=args.pose_pixel_frames,
        rgb_pixel_frames=args.rgb_pixel_frames,
    )

    transformer.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    s0_encoder.to(device, dtype=weight_dtype)
    track_modules.to(device, dtype=weight_dtype)
    transformer.eval()
    vae.eval()
    s0_encoder.eval()
    track_modules.eval()

    dataset = I2AVVideoDataset(
        data_root=str(args.data_root),
        dataset_file=None,
        caption_column="prompt.txt",
        video_column="videos.txt",
        image_column="images.txt",
        max_num_frames=args.num_frames,
        id_token="COAF",
        height_buckets=[args.height],
        width_buckets=[args.width],
        frame_buckets=[args.num_frames],
        load_tensors=True,
        image_to_video=True,
        track_column="track_paths.txt",
        track_visible_column="track_visible_paths.txt",
    )
    indices = validation_indices(args.data_root, None if args.max_eval_batches == 0 else args.max_eval_batches * args.batch_size)
    collate = I2AVCollateFunction(weight_dtype, load_tensors=True)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        collate_fn=lambda samples: collate([samples]),
        shuffle=False,
        num_workers=0,
    )

    vae_scaling_factor = vae.config.scaling_factor
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    track_norm_stats = torch.load(args.track_norm_stats, map_location="cpu", weights_only=False)
    results = []

    image_rotary_emb = prepare_i2av_v6_rotary_positional_embeddings(
        height=args.height,
        width=args.width,
        layout=layout,
        vae_scale_factor_spatial=vae_scale_factor_spatial,
        patch_size=transformer.config.patch_size,
        patch_size_t=getattr(transformer.config, "patch_size_t", None),
        attention_head_dim=transformer.config.attention_head_dim,
        device=device,
    )

    for checkpoint in args.checkpoint:
        checkpoint = checkpoint.resolve()
        load_checkpoint(transformer, s0_encoder, track_modules, checkpoint, device)
        totals: list[float] = []
        video_losses: list[float] = []
        tracks: list[float] = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=checkpoint.name):
                video_batch = batch["videos"].to(device=device, dtype=weight_dtype)
                images = batch["images"].to(device=device, dtype=weight_dtype)
                video_latents = DiagonalGaussianDistribution(video_batch).sample() * vae_scaling_factor
                video_latents = video_latents.permute(0, 2, 1, 3, 4).contiguous().to(dtype=weight_dtype)
                image_latents = DiagonalGaussianDistribution(images).sample() * vae_scaling_factor
                image_latents = image_latents.permute(0, 2, 1, 3, 4).contiguous().to(dtype=weight_dtype)
                padding_shape = (video_latents.shape[0], video_latents.shape[1] - 1, *video_latents.shape[2:])
                image_latents = torch.cat([image_latents, image_latents.new_zeros(padding_shape)], dim=1)
                prompt_embeds = batch["prompts"].to(device=device, dtype=weight_dtype)
                bsz = video_latents.shape[0]

                if s0_encoder.num_tokens > 0:
                    state0 = batch["state"][:, 0].to(device=device, dtype=weight_dtype)
                    s0_tokens = s0_encoder(state0)
                    prompt_embeds = torch.cat([prompt_embeds, s0_tokens], dim=1)

                noise = torch.randn(
                    video_latents.shape,
                    device=device,
                    dtype=weight_dtype,
                    generator=generator,
                )
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device, generator=generator).long()
                noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
                noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)

                track_gt = normalize_track_pixels(batch["track"].to(device=device), track_norm_stats).to(dtype=weight_dtype)
                track_valid_mask = batch["track_valid_mask"].to(device=device)
                noisy_track, track_noise = noise_track(track_gt, alphas_cumprod, timesteps)
                track_tokens = track_modules.track_tokenizer(noisy_track, timesteps=timesteps)

                ofs_emb = None
                if getattr(transformer.config, "ofs_embed_dim", None):
                    ofs_emb = torch.full((bsz,), 2.0, device=device, dtype=weight_dtype)

                model_output, track_hidden = forward_i2av_v6_transformer(
                    transformer,
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    track_tokens=track_tokens,
                    timestep=timesteps,
                    timestep_track=timesteps,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    layout=layout,
                    return_dict=False,
                )
                pred_track_noise = track_modules.track_tokenizer.decode_noise(track_hidden)
                model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)
                weights = 1 / (1 - alphas_cumprod[timesteps])
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)
                loss_video = torch.mean((weights * (model_pred - video_latents) ** 2).reshape(bsz, -1), dim=1).mean()
                track_loss = compute_track_noise_loss(
                    pred_track_noise,
                    track_noise.to(dtype=weight_dtype),
                    valid_mask=track_valid_mask,
                    lambda_track=args.lambda_track,
                )
                total = loss_video + args.lambda_sa * track_loss.loss
                totals.append(float(total.detach().cpu()))
                video_losses.append(float(loss_video.detach().cpu()))
                tracks.append(float(track_loss.track_loss.detach().cpu()))
        row = {
            "checkpoint": str(checkpoint),
            "num_samples": len(totals),
            "loss": sum(totals) / len(totals),
            "loss_video": sum(video_losses) / len(video_losses),
            "L_track": sum(tracks) / len(tracks),
        }
        print(json.dumps(row, indent=2), flush=True)
        results.append(row)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
