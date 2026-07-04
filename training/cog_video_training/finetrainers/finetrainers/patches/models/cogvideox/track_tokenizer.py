"""Track-token diffusion modules for CoAF I2AV v6."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TrackLossOutput:
    loss: torch.Tensor
    track_loss: torch.Tensor
    pred_noise: torch.Tensor
    valid_mask: torch.Tensor


class TrackTokenizer(nn.Module):
    """Embed noised 2D track coordinates as DiT tokens and decode epsilon."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        coord_dim: int = 2,
        num_steps: int = 25,
        tokens_per_step: int = 1,
        mlp_dim: int = 1024,
    ) -> None:
        super().__init__()
        if tokens_per_step != 1:
            raise NotImplementedError("v6 MVP supports one track token per step.")
        self.hidden_dim = int(hidden_dim)
        self.coord_dim = int(coord_dim)
        self.num_steps = int(num_steps)
        self.tokens_per_step = int(tokens_per_step)
        self.input_proj = nn.Sequential(
            nn.Linear(self.coord_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.hidden_dim),
        )
        self.step_embedding = nn.Embedding(self.num_steps, self.hidden_dim)
        self.timestep_embedding = nn.Embedding(1000, self.hidden_dim)
        self.modality = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.noise_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, self.coord_dim),
        )

    def forward(self, noisy_track: torch.Tensor, timesteps: torch.Tensor | None = None) -> torch.Tensor:
        if noisy_track.ndim != 3 or noisy_track.shape[-1] != self.coord_dim:
            raise ValueError(f"Expected track shape (B,T,{self.coord_dim}), got {tuple(noisy_track.shape)}")
        b, steps, _ = noisy_track.shape
        if steps > self.num_steps:
            raise ValueError(f"Track has {steps} steps, tokenizer supports at most {self.num_steps}.")
        step_ids = torch.arange(steps, device=noisy_track.device)
        tokens = self.input_proj(noisy_track)
        tokens = tokens + self.step_embedding(step_ids).view(1, steps, self.hidden_dim)
        if timesteps is not None:
            if timesteps.ndim == 0:
                timesteps = timesteps.expand(b)
            timesteps = timesteps.to(device=noisy_track.device, dtype=torch.long).clamp(0, self.timestep_embedding.num_embeddings - 1)
            tokens = tokens + self.timestep_embedding(timesteps).view(b, 1, self.hidden_dim)
        return tokens + self.modality

    def decode_noise(self, track_hidden: torch.Tensor) -> torch.Tensor:
        if track_hidden.ndim != 3:
            raise ValueError(f"Expected track hidden shape (B,T,D), got {tuple(track_hidden.shape)}")
        return self.noise_head(track_hidden)



class TrackModules(nn.Module):
    def __init__(self, hidden_dim: int, *, coord_dim: int = 2, num_steps: int = 25) -> None:
        super().__init__()
        self.track_tokenizer = TrackTokenizer(hidden_dim=hidden_dim, coord_dim=coord_dim, num_steps=num_steps)


def normalize_track_pixels(
    track_pixels: torch.Tensor,
    track_norm_stats: dict[str, torch.Tensor | int | float | str],
) -> torch.Tensor:
    """Map pixel coordinates to [-1, 1] using the image size stored in stats."""
    width = float(track_norm_stats.get("width", 256))
    height = float(track_norm_stats.get("height", 256))
    out = track_pixels.to(dtype=torch.float32).clone()
    out[..., 0] = out[..., 0] / width * 2.0 - 1.0
    out[..., 1] = out[..., 1] / height * 2.0 - 1.0
    return out


def denormalize_track_pixels(
    track_norm: torch.Tensor,
    track_norm_stats: dict[str, torch.Tensor | int | float | str],
) -> torch.Tensor:
    width = float(track_norm_stats.get("width", 256))
    height = float(track_norm_stats.get("height", 256))
    out = track_norm.to(dtype=torch.float32).clone()
    out[..., 0] = (out[..., 0] + 1.0) * 0.5 * width
    out[..., 1] = (out[..., 1] + 1.0) * 0.5 * height
    return out


def noise_track(
    track_norm: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diffuse normalized track coordinates and return (x_t, epsilon)."""
    noise = torch.randn_like(track_norm)
    alpha = alphas_cumprod[timesteps].to(device=track_norm.device, dtype=track_norm.dtype)
    while alpha.ndim < track_norm.ndim:
        alpha = alpha.unsqueeze(-1)
    noisy = alpha.sqrt() * track_norm + (1.0 - alpha).clamp_min(0.0).sqrt() * noise
    return noisy, noise


def compute_track_noise_loss(
    pred_noise: torch.Tensor,
    target_noise: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    lambda_track: float = 1.0,
) -> TrackLossOutput:
    if pred_noise.shape != target_noise.shape:
        raise ValueError(f"pred_noise shape {tuple(pred_noise.shape)} != target {tuple(target_noise.shape)}")
    if valid_mask is None:
        valid_mask = torch.ones(pred_noise.shape[:2], device=pred_noise.device, dtype=pred_noise.dtype)
    else:
        valid_mask = valid_mask.to(device=pred_noise.device, dtype=pred_noise.dtype)
    raw = F.mse_loss(pred_noise, target_noise.to(dtype=pred_noise.dtype), reduction="none").mean(dim=-1)
    track_loss = (raw * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    loss = float(lambda_track) * track_loss
    return TrackLossOutput(loss=loss, track_loss=track_loss, pred_noise=pred_noise, valid_mask=valid_mask)
