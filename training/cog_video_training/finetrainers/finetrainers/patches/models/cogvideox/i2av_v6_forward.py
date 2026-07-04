"""CogVideoX v6 forward path for depth + track + RGB block denoising."""

from __future__ import annotations

from typing import Any

import torch

from .i2av_layout import I2AVV6Layout
from .i2av_sequence import build_v6_depth_track_rgb_tokens, deinterleave_v6_depth_track_rgb_tokens


def _module_device(module) -> torch.device:
    return next(module.parameters()).device


def _to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def forward_i2av_v6_transformer(
    transformer,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    track_tokens: torch.Tensor,
    *,
    timestep,
    layout: I2AVV6Layout,
    timestep_track=None,
    timestep_cond=None,
    ofs=None,
    image_rotary_emb=None,
    attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = False,
):
    """CogVideoX forward for v6 [Depth, Track, RGB] block layout."""
    batch_size, num_frames, channels, height, width = hidden_states.shape
    attention_kwargs = attention_kwargs or {}
    device = _module_device(transformer.time_embedding)
    hidden_states = hidden_states.to(device=device)
    encoder_hidden_states = encoder_hidden_states.to(device=device)
    track_tokens = track_tokens.to(device=device)
    timestep = _to_device(timestep, device)
    timestep_track = _to_device(timestep_track, device)
    timestep_cond = _to_device(timestep_cond, device)
    ofs = _to_device(ofs, device)
    image_rotary_emb = _to_device(image_rotary_emb, device)
    if num_frames != layout.num_latent_frames:
        raise ValueError(f"v6 layout expects {layout.num_latent_frames} latent frames, got {num_frames}.")

    t_emb = transformer.time_proj(timestep)
    t_emb = t_emb.to(device=device, dtype=hidden_states.dtype)
    emb = transformer.time_embedding(t_emb, timestep_cond)

    if transformer.ofs_embedding is not None and ofs is not None:
        ofs_emb = transformer.ofs_proj(ofs)
        ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
        ofs_emb = transformer.ofs_embedding(ofs_emb)
        emb = emb + ofs_emb

    embeds = transformer.patch_embed(encoder_hidden_states, hidden_states)
    embeds = transformer.embedding_dropout(embeds)

    text_seq_length = encoder_hidden_states.shape[1]
    encoder_hidden_states = embeds[:, :text_seq_length]
    video_tokens = embeds[:, text_seq_length:]

    expected_visual = layout.num_latent_frames * layout.patches_per_frame
    if video_tokens.shape[1] != expected_visual:
        raise ValueError(f"Unexpected visual token count: got {video_tokens.shape[1]}, expected {expected_visual}.")
    if track_tokens.shape[1] != layout.track_tokens:
        raise ValueError(f"Unexpected track token count: got {track_tokens.shape[1]}, expected {layout.track_tokens}.")

    pose_len = layout.num_pose_latent_frames * layout.patches_per_frame
    pose_tokens = video_tokens[:, :pose_len]
    rgb_tokens = video_tokens[:, pose_len:]
    hidden_states = build_v6_depth_track_rgb_tokens(pose_tokens, track_tokens, rgb_tokens)

    for block in transformer.transformer_blocks:
        if torch.is_grad_enabled() and transformer.gradient_checkpointing:
            hidden_states, encoder_hidden_states = transformer._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                emb,
                image_rotary_emb,
                attention_kwargs,
            )
        else:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )

    hidden_states = transformer.norm_final(hidden_states)
    pose_visual, track_hidden, rgb_visual = deinterleave_v6_depth_track_rgb_tokens(
        hidden_states,
        num_pose_latent_frames=layout.num_pose_latent_frames,
        num_rgb_latent_frames=layout.num_rgb_latent_frames,
        patches_per_frame=layout.patches_per_frame,
        track_tokens=layout.track_tokens,
    )

    visual_tokens = torch.cat([pose_visual, rgb_visual], dim=1)
    visual_tokens = transformer.norm_out(visual_tokens, temb=emb)
    visual_tokens = transformer.proj_out(visual_tokens)

    p = transformer.config.patch_size
    p_t = transformer.config.patch_size_t
    if p_t is not None:
        raise NotImplementedError("I2AV v6 forward does not support patch_size_t layouts yet.")
    output = visual_tokens.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
    output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

    if not return_dict:
        return output, track_hidden
    return output, track_hidden
