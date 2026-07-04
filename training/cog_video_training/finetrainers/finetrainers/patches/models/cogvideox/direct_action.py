"""Direct continuous action heads for CoAF I2AV v5.

This path keeps the existing visual causal transformer, but predicts normalized
raw actions directly from causal pose chunk hidden states instead of denoising
state/action tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .state_action import align_trajectory_steps, normalize_raw_action


@dataclass(frozen=True)
class DirectActionLossOutput:
    loss: torch.Tensor
    action_loss: torch.Tensor
    gripper_loss: torch.Tensor
    pred_action: torch.Tensor
    valid_mask: torch.Tensor


class PastActionConditioner(nn.Module):
    """Project previous action chunks into transformer condition tokens.

    For v5 with 25 pose steps and 7 pose latent chunks, chunks are padded to
    7x4. We provide only the first 6 chunks as A_cond tokens:
    A_0 conditions P_1, ..., A_5 conditions P_6. P_k must not attend A_k.
    """

    def __init__(self, action_dim: int, hidden_dim: int, steps_per_chunk: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.steps_per_chunk = int(steps_per_chunk)
        self.proj = nn.Sequential(
            nn.Linear(self.action_dim * self.steps_per_chunk, 512),
            nn.SiLU(),
            nn.Linear(512, self.hidden_dim),
        )
        self.modality = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)

    def forward(self, action_chunks: torch.Tensor) -> torch.Tensor:
        if action_chunks.ndim != 4:
            raise ValueError(f"Expected action chunks shape (B,C,S,A), got {tuple(action_chunks.shape)}")
        b, c, s, a = action_chunks.shape
        if s != self.steps_per_chunk or a != self.action_dim:
            raise ValueError(
                "PastActionConditioner shape mismatch: "
                f"got steps/action=({s},{a}), expected ({self.steps_per_chunk},{self.action_dim})."
            )
        if c <= 1:
            return action_chunks.new_zeros(b, 0, self.hidden_dim)
        prev_chunks = action_chunks[:, :-1].reshape(b, c - 1, s * a)
        return self.proj(prev_chunks) + self.modality


class ActionDiffusionTokenizer(nn.Module):
    """Project noised action chunks into DiT action diffusion tokens."""

    def __init__(self, action_dim: int, hidden_dim: int, steps_per_chunk: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.steps_per_chunk = int(steps_per_chunk)
        self.proj = nn.Sequential(
            nn.Linear(self.action_dim * self.steps_per_chunk, 512),
            nn.SiLU(),
            nn.Linear(512, self.hidden_dim),
        )
        self.modality = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)

    def forward(self, noisy_action_chunks: torch.Tensor) -> torch.Tensor:
        if noisy_action_chunks.ndim != 4:
            raise ValueError(f"Expected noisy action chunks shape (B,C,S,A), got {tuple(noisy_action_chunks.shape)}")
        b, c, s, a = noisy_action_chunks.shape
        if s != self.steps_per_chunk or a != self.action_dim:
            raise ValueError(
                "ActionDiffusionTokenizer shape mismatch: "
                f"got steps/action=({s},{a}), expected ({self.steps_per_chunk},{self.action_dim})."
            )
        flat = noisy_action_chunks.reshape(b, c, s * a)
        return self.proj(flat) + self.modality


class DirectChunkActionHead(nn.Module):
    """Predict one action-noise chunk from each action diffusion token hidden."""

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int = 7,
        steps_per_chunk: int = 4,
        mlp_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.steps_per_chunk = int(steps_per_chunk)
        self.noisy_action_proj = nn.Sequential(
            nn.Linear(self.action_dim * self.steps_per_chunk, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, self.steps_per_chunk * self.action_dim),
        )

    def forward(
        self,
        pose_chunk_hidden: torch.Tensor,
        noisy_action_chunks: torch.Tensor | None = None,
        *,
        real_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pose_chunk_hidden.ndim != 3:
            raise ValueError(f"Expected pose hidden shape (B,C,D), got {tuple(pose_chunk_hidden.shape)}")
        hidden = pose_chunk_hidden
        b, chunks, _ = pose_chunk_hidden.shape
        if noisy_action_chunks is not None:
            expected = (b, chunks, self.steps_per_chunk, self.action_dim)
            if noisy_action_chunks.shape != expected:
                raise ValueError(f"Expected noisy action chunks shape {expected}, got {tuple(noisy_action_chunks.shape)}")
            noisy_flat = noisy_action_chunks.reshape(b, chunks, self.steps_per_chunk * self.action_dim)
            hidden = hidden + self.noisy_action_proj(noisy_flat.to(dtype=hidden.dtype))
        pred_chunks = self.net(hidden).reshape(b, chunks, self.steps_per_chunk, self.action_dim)
        pred_steps = pred_chunks.reshape(b, chunks * self.steps_per_chunk, self.action_dim)
        if real_steps is not None:
            pred_steps = pred_steps[:, : int(real_steps)]
        return pred_steps, pred_chunks


class DirectActionModules(nn.Module):
    """Auxiliary modules for the direct action branch."""

    def __init__(self, hidden_dim: int, action_dim: int = 7, steps_per_chunk: int = 4) -> None:
        super().__init__()
        self.past_action_conditioner = PastActionConditioner(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            steps_per_chunk=steps_per_chunk,
        )
        self.action_tokenizer = ActionDiffusionTokenizer(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            steps_per_chunk=steps_per_chunk,
        )
        self.action_head = DirectChunkActionHead(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            steps_per_chunk=steps_per_chunk,
        )


def prepare_direct_action_gt(
    state_seq: torch.Tensor,
    action_seq: torch.Tensor,
    state_norm_stats: dict[str, torch.Tensor],
    action_norm_stats: dict[str, torch.Tensor],
    *,
    pose_pixel_frames: int,
    gripper_continuous: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare only the direct-action target and s0 condition.

    Direct-action training does not build or supervise state chunks. The state
    sequence is used only to produce the normalized initial-state condition.
    """
    state_aligned = align_trajectory_steps(state_seq, pose_pixel_frames)
    action_aligned = align_trajectory_steps(action_seq, pose_pixel_frames)

    state_mean = state_norm_stats["mean"].to(state_seq.device, dtype=state_seq.dtype)
    state_std = state_norm_stats["std"].to(state_seq.device, dtype=state_seq.dtype)
    s0_norm = (state_aligned[:, 0] - state_mean) / state_std
    action_gt = normalize_raw_action(
        action_aligned,
        action_norm_stats,
        gripper_continuous=gripper_continuous,
    )
    return action_gt, s0_norm


def pad_actions_to_chunks(
    action_gt: torch.Tensor,
    *,
    num_chunks: int,
    steps_per_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded action chunks and a valid-step mask.

    The first v5 pose chunk contains only one real step, but this direct branch
    intentionally uses uniform 4-step chunks and masks padded tail positions.
    """
    if action_gt.ndim != 3:
        raise ValueError(f"Expected action_gt shape (B,T,A), got {tuple(action_gt.shape)}")
    b, steps, action_dim = action_gt.shape
    target_steps = int(num_chunks) * int(steps_per_chunk)
    if steps < target_steps:
        pad = action_gt[:, -1:].expand(-1, target_steps - steps, -1)
        padded = torch.cat([action_gt, pad], dim=1)
    else:
        padded = action_gt[:, :target_steps]
    valid = torch.arange(target_steps, device=action_gt.device).view(1, target_steps) < steps
    chunks = padded.reshape(b, int(num_chunks), int(steps_per_chunk), action_dim)
    return chunks, valid.expand(b, -1)


def noise_action_chunks(
    action_chunks: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diffuse normalized action chunks and return (x_t, epsilon)."""
    noise = torch.randn_like(action_chunks)
    alpha = alphas_cumprod[timesteps].to(device=action_chunks.device, dtype=action_chunks.dtype)
    while alpha.ndim < action_chunks.ndim:
        alpha = alpha.unsqueeze(-1)
    sqrt_alpha = alpha.sqrt()
    sqrt_one_minus_alpha = (1.0 - alpha).clamp_min(0.0).sqrt()
    noisy = sqrt_alpha * action_chunks + sqrt_one_minus_alpha * noise
    return noisy, noise


def flatten_action_chunks(action_chunks: torch.Tensor, *, real_steps: int | None = None) -> torch.Tensor:
    if action_chunks.ndim != 4:
        raise ValueError(f"Expected action chunks shape (B,C,S,A), got {tuple(action_chunks.shape)}")
    b, chunks, steps, action_dim = action_chunks.shape
    steps_flat = action_chunks.reshape(b, chunks * steps, action_dim)
    if real_steps is not None:
        steps_flat = steps_flat[:, : int(real_steps)]
    return steps_flat


def compute_direct_action_loss(
    pred_action: torch.Tensor,
    action_gt: torch.Tensor,
    *,
    action_norm_stats: dict[str, torch.Tensor] | None = None,
    valid_mask: torch.Tensor | None = None,
    gripper_continuous: bool = True,
    lambda_a: float = 1.0,
    lambda_g: float = 1.0,
) -> DirectActionLossOutput:
    if pred_action.shape != action_gt.shape:
        raise ValueError(f"pred_action shape {tuple(pred_action.shape)} != action_gt shape {tuple(action_gt.shape)}")
    if valid_mask is None:
        valid_mask = torch.ones(pred_action.shape[:2], device=pred_action.device, dtype=pred_action.dtype)
    else:
        valid_mask = valid_mask.to(device=pred_action.device, dtype=pred_action.dtype)

    if action_norm_stats is not None and "valid_action_mask" in action_norm_stats:
        action_mask = action_norm_stats["valid_action_mask"].to(device=pred_action.device, dtype=pred_action.dtype)
    else:
        action_mask = torch.ones(pred_action.shape[-1], device=pred_action.device, dtype=pred_action.dtype)

    action_dims = 7 if gripper_continuous else 6
    dim_mask = action_mask[:action_dims].clamp_min(0.0)
    raw_loss = F.smooth_l1_loss(
        pred_action[..., :action_dims],
        action_gt[..., :action_dims].to(dtype=pred_action.dtype),
        reduction="none",
    )
    raw_loss = raw_loss * dim_mask.view(1, 1, action_dims)
    step_loss = raw_loss.sum(dim=-1) / dim_mask.sum().clamp_min(1.0)
    action_loss = (step_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

    gripper_loss = pred_action.new_zeros(())
    if not gripper_continuous:
        positive_rate = 0.5 if action_norm_stats is None else float(action_norm_stats.get("gripper_positive_rate", 0.5))
        pos_weight = torch.tensor(
            max((1.0 - positive_rate) / max(positive_rate, 1e-6), 1e-3),
            device=pred_action.device,
            dtype=pred_action.dtype,
        )
        gripper_raw = F.binary_cross_entropy_with_logits(
            pred_action[..., 6],
            action_gt[..., 6].to(dtype=pred_action.dtype),
            pos_weight=pos_weight,
            reduction="none",
        )
        gripper_loss = (gripper_raw * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        gripper_loss = gripper_loss * action_mask[6].clamp_min(0.0)

    loss = lambda_a * action_loss + lambda_g * gripper_loss
    return DirectActionLossOutput(
        loss=loss,
        action_loss=action_loss,
        gripper_loss=gripper_loss,
        pred_action=pred_action,
        valid_mask=valid_mask,
    )


def compute_direct_action_noise_loss(
    pred_noise: torch.Tensor,
    target_noise: torch.Tensor,
    *,
    action_norm_stats: dict[str, torch.Tensor] | None = None,
    valid_mask: torch.Tensor | None = None,
    gripper_continuous: bool = True,
    lambda_a: float = 1.0,
    lambda_g: float = 1.0,
) -> DirectActionLossOutput:
    """MSE action diffusion loss where the head predicts epsilon noise."""
    if pred_noise.shape != target_noise.shape:
        raise ValueError(f"pred_noise shape {tuple(pred_noise.shape)} != target_noise shape {tuple(target_noise.shape)}")
    if valid_mask is None:
        valid_mask = torch.ones(pred_noise.shape[:2], device=pred_noise.device, dtype=pred_noise.dtype)
    else:
        valid_mask = valid_mask.to(device=pred_noise.device, dtype=pred_noise.dtype)

    if action_norm_stats is not None and "valid_action_mask" in action_norm_stats:
        action_mask = action_norm_stats["valid_action_mask"].to(device=pred_noise.device, dtype=pred_noise.dtype)
    else:
        action_mask = torch.ones(pred_noise.shape[-1], device=pred_noise.device, dtype=pred_noise.dtype)

    action_dims = 7 if gripper_continuous else 6
    dim_mask = action_mask[:action_dims].clamp_min(0.0)
    raw_loss = F.mse_loss(
        pred_noise[..., :action_dims],
        target_noise[..., :action_dims].to(dtype=pred_noise.dtype),
        reduction="none",
    )
    raw_loss = raw_loss * dim_mask.view(1, 1, action_dims)
    step_loss = raw_loss.sum(dim=-1) / dim_mask.sum().clamp_min(1.0)
    action_loss = (step_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

    gripper_loss = pred_noise.new_zeros(())
    if not gripper_continuous:
        gripper_raw = F.mse_loss(
            pred_noise[..., 6],
            target_noise[..., 6].to(dtype=pred_noise.dtype),
            reduction="none",
        )
        gripper_loss = (gripper_raw * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        gripper_loss = gripper_loss * action_mask[6].clamp_min(0.0)

    loss = lambda_a * action_loss + lambda_g * gripper_loss
    return DirectActionLossOutput(
        loss=loss,
        action_loss=action_loss,
        gripper_loss=gripper_loss,
        pred_action=pred_noise,
        valid_mask=valid_mask,
    )

