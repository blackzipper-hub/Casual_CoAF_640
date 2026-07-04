"""CogVideoX v5 forward path for direct chunk action prediction."""

from __future__ import annotations

from typing import Any

import torch

from .i2av_layout import I2AVV5Layout


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


def build_direct_action_pose_rgb_tokens(
    pose_visual_tokens: torch.Tensor,
    action_tokens: torch.Tensor,
    rgb_visual_tokens: torch.Tensor,
    *,
    patches_per_frame: int,
    num_pose_latent_frames: int,
) -> torch.Tensor:
    """Build [P0,A0_t,P1,A1_t,...,P6,A6_t,RGB] tokens."""
    b, _, d = pose_visual_tokens.shape
    pose = pose_visual_tokens.reshape(b, num_pose_latent_frames, patches_per_frame, d)
    if action_tokens.shape != (b, num_pose_latent_frames, d):
        raise ValueError(
            "action_tokens shape mismatch: "
            f"got {tuple(action_tokens.shape)}, expected {(b, num_pose_latent_frames, d)}."
        )
    parts: list[torch.Tensor] = []
    for chunk_idx in range(num_pose_latent_frames):
        parts.append(pose[:, chunk_idx])
        parts.append(action_tokens[:, chunk_idx : chunk_idx + 1])
    parts.append(rgb_visual_tokens)
    return torch.cat(parts, dim=1)


def deinterleave_direct_action_pose_rgb_tokens(
    tokens: torch.Tensor,
    *,
    num_pose_latent_frames: int,
    num_rgb_latent_frames: int,
    patches_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split direct-action v5 tokens into pose visual, action diffusion, rgb visual."""
    b, _, d = tokens.shape
    pose_parts: list[torch.Tensor] = []
    action_cond_parts: list[torch.Tensor] = []
    cursor = 0
    for chunk_idx in range(num_pose_latent_frames):
        pose_parts.append(tokens[:, cursor : cursor + patches_per_frame])
        cursor += patches_per_frame
        action_cond_parts.append(tokens[:, cursor : cursor + 1])
        cursor += 1
    rgb_len = num_rgb_latent_frames * patches_per_frame
    rgb_visual = tokens[:, cursor : cursor + rgb_len]
    if rgb_visual.shape[1] != rgb_len:
        raise ValueError(f"Unexpected RGB token count: got {rgb_visual.shape[1]}, expected {rgb_len}.")
    pose_visual = torch.cat(pose_parts, dim=1)
    action_cond = torch.cat(action_cond_parts, dim=1)
    return pose_visual, action_cond, rgb_visual


def forward_i2av_v5_direct_action_transformer(
    transformer,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    action_tokens: torch.Tensor,
    *,
    timestep,
    layout: I2AVV5Layout,
    timestep_cond=None,
    ofs=None,
    image_rotary_emb=None,
    attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = False,
):
    """CogVideoX forward for [P0,A0_t,P1,A1_t,...,P6,A6_t,RGB] direct action layout."""
    batch_size, num_frames, channels, height, width = hidden_states.shape
    attention_kwargs = attention_kwargs or {}
    device = _module_device(transformer.time_embedding)
    hidden_states = hidden_states.to(device=device)
    encoder_hidden_states = encoder_hidden_states.to(device=device)
    action_tokens = action_tokens.to(device=device)
    timestep = _to_device(timestep, device)
    timestep_cond = _to_device(timestep_cond, device)
    ofs = _to_device(ofs, device)
    image_rotary_emb = _to_device(image_rotary_emb, device)
    if num_frames != layout.num_latent_frames:
        raise ValueError(f"v5 direct action layout expects {layout.num_latent_frames} latent frames, got {num_frames}.")

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

    pose_len = layout.num_pose_latent_frames * layout.patches_per_frame
    pose_tokens = video_tokens[:, :pose_len]
    rgb_tokens = video_tokens[:, pose_len:]
    hidden_states = build_direct_action_pose_rgb_tokens(
        pose_tokens,
        action_tokens,
        rgb_tokens,
        patches_per_frame=layout.patches_per_frame,
        num_pose_latent_frames=layout.num_pose_latent_frames,
    )

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
    pose_visual, action_token_hidden, rgb_visual = deinterleave_direct_action_pose_rgb_tokens(
        hidden_states,
        num_pose_latent_frames=layout.num_pose_latent_frames,
        num_rgb_latent_frames=layout.num_rgb_latent_frames,
        patches_per_frame=layout.patches_per_frame,
    )

    visual_tokens = torch.cat([pose_visual, rgb_visual], dim=1)
    visual_tokens = transformer.norm_out(visual_tokens, temb=emb)
    visual_tokens = transformer.proj_out(visual_tokens)

    p = transformer.config.patch_size
    p_t = transformer.config.patch_size_t
    if p_t is not None:
        raise NotImplementedError("I2AV v5 direct action forward does not support patch_size_t layouts yet.")
    output = visual_tokens.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
    output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

    if not return_dict:
        return output, action_token_hidden
    return output, action_token_hidden

