"""State/action tokenizers and losses for I2AV joint training."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateActionModule(Protocol):
    hidden_dim: int

    def encode(self, state_norm: torch.Tensor, action_norm: torch.Tensor) -> torch.Tensor: ...

    def decode(self, token_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class ChunkedStateActionTokenizer(nn.Module):
    """Encode/decode full-resolution state/action into per-pose-latent chunks.

    Each chunk covers ``steps_per_chunk`` trajectory steps with interleaved
    ``[s0, a0, s1, a1, ...]`` tokens (one token per state and per action step).
    ``action_chunk_alignment="next_window"`` keeps state aligned to the current
    latent window but moves each chunk's actions to the following latent window.
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 7,
        steps_per_chunk: int = 4,
        first_chunk_pad_steps: int | None = None,
        real_trajectory_steps: int | None = None,
        action_chunk_alignment: str = "current",
    ) -> None:
        super().__init__()
        if action_chunk_alignment not in {"current", "next_window"}:
            raise ValueError(
                "action_chunk_alignment must be either 'current' or 'next_window', "
                f"got {action_chunk_alignment!r}."
            )
        self.steps_per_chunk = steps_per_chunk
        self.chunk_token_count = 2 * steps_per_chunk
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.first_chunk_pad_steps = steps_per_chunk - 1 if first_chunk_pad_steps is None else first_chunk_pad_steps
        self.real_trajectory_steps = real_trajectory_steps
        self.action_chunk_alignment = action_chunk_alignment

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, hidden_dim),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, hidden_dim),
        )
        self.state_output = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, state_dim),
        )
        self.action_output = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, state_dim),
        )
        self.state_modality = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.action_modality = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def _front_pad_to_chunks(self, seq: torch.Tensor) -> torch.Tensor:
        """Pad the first VAE chunk so 25 steps become 7 uniform 4-step chunks."""
        if seq.shape[1] == 0:
            raise ValueError("Cannot tokenize an empty state/action sequence.")
        if self.first_chunk_pad_steps <= 0:
            return seq
        first_pad = seq[:, :1].expand(-1, self.first_chunk_pad_steps, -1)
        return torch.cat([first_pad, seq], dim=1)

    def _tail_pad_to_length(self, seq: torch.Tensor, target_steps: int) -> torch.Tensor:
        if seq.shape[1] == 0:
            raise ValueError("Cannot tokenize an empty state/action sequence.")
        if seq.shape[1] > target_steps:
            return seq[:, :target_steps]
        if seq.shape[1] == target_steps:
            return seq
        tail_pad = seq[:, -1:].expand(-1, target_steps - seq.shape[1], -1)
        return torch.cat([seq, tail_pad], dim=1)

    def encode(self, state_norm: torch.Tensor, action_norm: torch.Tensor) -> torch.Tensor:
        """Return chunk tokens with shape ``(B, num_chunks, chunk_token_count, D)``."""
        b, _, _ = state_norm.shape
        if state_norm.shape != action_norm.shape:
            raise ValueError(f"State/action shapes must match, got {state_norm.shape} and {action_norm.shape}.")
        state_norm = self._front_pad_to_chunks(state_norm)
        if self.action_chunk_alignment == "next_window":
            action_norm = self._tail_pad_to_length(action_norm, state_norm.shape[1])
        else:
            action_norm = self._front_pad_to_chunks(action_norm)
        t = state_norm.shape[1]
        s_tok = self.state_proj(state_norm) + self.state_modality
        a_tok = self.action_proj(action_norm) + self.action_modality
        interleaved = torch.stack([s_tok, a_tok], dim=2).reshape(b, t * 2, self.hidden_dim)
        chunk_size = self.chunk_token_count
        if interleaved.shape[1] % chunk_size != 0:
            raise ValueError(
                f"Trajectory length {t} yields {interleaved.shape[1]} interleaved tokens, "
                f"not divisible by chunk size {chunk_size}."
            )
        num_chunks = interleaved.shape[1] // chunk_size
        return interleaved.reshape(b, num_chunks, chunk_size, self.hidden_dim)

    def decode(self, chunk_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, num_chunks, chunk_size, dim = chunk_outputs.shape
        flat = chunk_outputs.reshape(b, num_chunks * chunk_size, dim)
        t = flat.shape[1] // 2
        flat = flat.reshape(b, t, 2, dim)
        pred_state = self.state_output(flat[:, :, 0])
        pred_action = self.action_output(flat[:, :, 1])
        if self.first_chunk_pad_steps > 0:
            pred_state = pred_state[:, self.first_chunk_pad_steps :]
            if self.action_chunk_alignment == "current":
                pred_action = pred_action[:, self.first_chunk_pad_steps :]
        if self.real_trajectory_steps is not None:
            pred_state = pred_state[:, : self.real_trajectory_steps]
            pred_action = pred_action[:, : self.real_trajectory_steps]
        return pred_state, pred_action

    def chunk_token_valid_mask(
        self,
        num_chunks: int,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return valid-token mask for padded chunk positions, shape ``(C, 2 * steps)``."""
        mask = torch.ones(num_chunks, self.chunk_token_count, dtype=torch.bool, device=device)
        if self.first_chunk_pad_steps > 0:
            for step_idx in range(self.first_chunk_pad_steps):
                mask[0, step_idx * 2] = False
                if self.action_chunk_alignment == "current":
                    mask[0, step_idx * 2 + 1] = False

        if self.action_chunk_alignment == "next_window" and self.real_trajectory_steps is not None:
            total_steps = num_chunks * self.steps_per_chunk
            for step_idx in range(self.real_trajectory_steps, total_steps):
                chunk_idx = step_idx // self.steps_per_chunk
                in_chunk_idx = step_idx % self.steps_per_chunk
                if chunk_idx < num_chunks:
                    mask[chunk_idx, in_chunk_idx * 2 + 1] = False
        return mask

    def decoded_chunk_ids(
        self,
        num_steps: int,
        *,
        kind: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Map decoded real trajectory steps back to their supervising chunk ids."""
        if kind not in {"state", "action"}:
            raise ValueError(f"kind must be 'state' or 'action', got {kind!r}.")
        step_ids = torch.arange(num_steps, device=device)
        if kind == "action" and self.action_chunk_alignment == "next_window":
            return step_ids // self.steps_per_chunk
        return (step_ids + self.first_chunk_pad_steps) // self.steps_per_chunk


class StateActionTokenizer(nn.Module):
    """Legacy per-latent-frame tokenizer (v3 layout)."""

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 7,
        num_state_tokens: int = 4,
        num_action_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.num_state_tokens = num_state_tokens
        self.num_action_tokens = num_action_tokens
        self.num_tokens = num_state_tokens + num_action_tokens
        self.hidden_dim = hidden_dim
        self.steps_per_chunk = 0
        self.chunk_token_count = self.num_tokens

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_state_tokens * hidden_dim),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_action_tokens * hidden_dim),
        )
        self.state_output = nn.Sequential(
            nn.Linear(num_state_tokens * hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, state_dim),
        )
        self.action_output = nn.Sequential(
            nn.Linear(num_action_tokens * hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, state_dim),
        )
        self.state_modality_emb = nn.Parameter(torch.randn(1, 1, num_state_tokens, hidden_dim) * 0.02)
        self.action_modality_emb = nn.Parameter(torch.randn(1, 1, num_action_tokens, hidden_dim) * 0.02)

    def encode(self, state_norm: torch.Tensor, action_norm: torch.Tensor) -> torch.Tensor:
        b, t, _ = state_norm.shape
        s_tok = self.state_proj(state_norm).reshape(b, t, self.num_state_tokens, self.hidden_dim)
        s_tok = s_tok + self.state_modality_emb
        a_tok = self.action_proj(action_norm).reshape(b, t, self.num_action_tokens, self.hidden_dim)
        a_tok = a_tok + self.action_modality_emb
        return torch.cat([s_tok, a_tok], dim=2)

    def decode(self, token_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _, d = token_output.shape
        s_tok = token_output[:, :, : self.num_state_tokens]
        a_tok = token_output[:, :, self.num_state_tokens :]
        pred_state = self.state_output(s_tok.reshape(b, t, self.num_state_tokens * d))
        pred_action = self.action_output(a_tok.reshape(b, t, self.num_action_tokens * d))
        return pred_state, pred_action


class S0Encoder(nn.Module):
    """Initial joint configuration as global condition tokens."""

    def __init__(self, hidden_dim: int, state_dim: int = 7, num_tokens: int = 4) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        if num_tokens > 0:
            self.proj = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.SiLU(),
                nn.Linear(256, num_tokens * hidden_dim),
            )
        else:
            self.proj = None

    def forward(self, s0_norm: torch.Tensor) -> torch.Tensor:
        b = s0_norm.shape[0]
        if self.num_tokens == 0:
            return s0_norm.new_empty((b, 0, self.hidden_dim))
        return self.proj(s0_norm).reshape(b, self.num_tokens, -1)


def prepare_gt_chunked(
    state_seq: torch.Tensor,
    norm_stats: dict[str, torch.Tensor],
    *,
    num_pose_latent_frames: int | None = None,
    steps_per_chunk: int | None = None,
    pose_pixel_frames: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Full-resolution state/action GT aligned to pose latent chunks."""
    mean = norm_stats["mean"].to(state_seq.device, dtype=state_seq.dtype)
    std = norm_stats["std"].to(state_seq.device, dtype=state_seq.dtype)

    if pose_pixel_frames is not None:
        if state_seq.shape[1] < pose_pixel_frames:
            pad = state_seq[:, -1:].expand(-1, pose_pixel_frames - state_seq.shape[1], -1)
            state_aligned = torch.cat([state_seq, pad], dim=1)
        else:
            state_aligned = state_seq[:, :pose_pixel_frames]
    else:
        if num_pose_latent_frames is None:
            raise ValueError("prepare_gt_chunked requires pose_pixel_frames or num_pose_latent_frames.")
        if steps_per_chunk is None:
            steps_per_chunk = max(state_seq.shape[1] // num_pose_latent_frames, 1)
        t_aligned = steps_per_chunk * num_pose_latent_frames
        if state_seq.shape[1] < t_aligned:
            pad = state_seq[:, -1:].expand(-1, t_aligned - state_seq.shape[1], -1)
            state_aligned = torch.cat([state_seq, pad], dim=1)
        else:
            state_aligned = state_seq[:, :t_aligned]
    if steps_per_chunk is None:
        steps_per_chunk = 4

    state_gt = (state_aligned - mean) / std
    delta = state_aligned[:, 1:] - state_aligned[:, :-1]
    delta_norm = delta / std
    action_gt = F.pad(delta_norm, (0, 0, 0, 1), value=0.0)
    s0_norm = state_gt[:, 0]
    return state_gt, action_gt, s0_norm, steps_per_chunk


def align_trajectory_steps(seq: torch.Tensor, target_steps: int) -> torch.Tensor:
    if seq.shape[1] < target_steps:
        pad = seq[:, -1:].expand(-1, target_steps - seq.shape[1], -1)
        return torch.cat([seq, pad], dim=1)
    return seq[:, :target_steps]


def get_action_norm_method(action_norm_stats: dict[str, torch.Tensor]) -> str:
    method = action_norm_stats.get("norm_method", "mean_std")
    if isinstance(method, torch.Tensor):
        method = method.item()
    if isinstance(method, bytes):
        method = method.decode("utf-8")
    return str(method)


def normalize_raw_action(
    action: torch.Tensor,
    action_norm_stats: dict[str, torch.Tensor],
    *,
    gripper_continuous: bool = False,
) -> torch.Tensor:
    method = get_action_norm_method(action_norm_stats)
    gripper_threshold = float(action_norm_stats.get("gripper_threshold", 0.5))
    action_gt = torch.empty_like(action)

    if method == "quantile":
        q01 = action_norm_stats["q01"].to(action.device, dtype=action.dtype)
        q99 = action_norm_stats["q99"].to(action.device, dtype=action.dtype)
        denom = (q99 - q01).clamp_min(1e-6)
        action_dims = 7 if gripper_continuous else 6
        action_gt[..., :action_dims] = (action[..., :action_dims] - q01[:action_dims]) / denom[:action_dims] * 2.0 - 1.0
        clip_value = float(action_norm_stats.get("clip", 1.5))
        action_gt[..., :action_dims] = action_gt[..., :action_dims].clamp(-clip_value, clip_value)
    elif method == "mean_std":
        action_mean = action_norm_stats["mean"].to(action.device, dtype=action.dtype)
        action_std = action_norm_stats["std"].to(action.device, dtype=action.dtype).clamp_min(1e-6)
        action_dims = 7 if gripper_continuous else 6
        action_gt[..., :action_dims] = (action[..., :action_dims] - action_mean[:action_dims]) / action_std[:action_dims]
    else:
        raise ValueError(f"Unsupported action norm method: {method}")

    if not gripper_continuous:
        action_gt[..., 6] = (action[..., 6] >= gripper_threshold).to(action_gt.dtype)
    return action_gt


def prepare_raw_action_gt_chunked(
    state_seq: torch.Tensor,
    action_seq: torch.Tensor,
    state_norm_stats: dict[str, torch.Tensor],
    action_norm_stats: dict[str, torch.Tensor],
    *,
    pose_pixel_frames: int,
    steps_per_chunk: int | None = None,
    gripper_continuous: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Prepare normalized state and raw-action targets for v5 chunked training.

    The returned action target uses normalized raw action for d0-d5. When
    ``gripper_continuous`` is enabled, d6 is normalized with the same action
    stats; otherwise d6 remains a binary gripper label. State deltas are
    returned separately for auxiliary consistency loss and must not be treated
    as action labels.
    """
    if steps_per_chunk is None:
        steps_per_chunk = 4
    state_aligned = align_trajectory_steps(state_seq, pose_pixel_frames)
    action_aligned = align_trajectory_steps(action_seq, pose_pixel_frames)

    state_mean = state_norm_stats["mean"].to(state_seq.device, dtype=state_seq.dtype)
    state_std = state_norm_stats["std"].to(state_seq.device, dtype=state_seq.dtype)
    state_gt = (state_aligned - state_mean) / state_std
    action_gt = normalize_raw_action(
        action_aligned,
        action_norm_stats,
        gripper_continuous=gripper_continuous,
    )

    delta = state_aligned[:, 1:] - state_aligned[:, :-1]
    delta = F.pad(delta, (0, 0, 0, 1), value=0.0)
    s0_norm = state_gt[:, 0]
    return state_gt, action_gt, delta, s0_norm, steps_per_chunk


def prepare_gt(
    state_seq: torch.Tensor,
    norm_stats: dict[str, torch.Tensor],
    num_latent_frames: int = 13,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy downsampled GT for v3 per-latent layout."""
    mean = norm_stats["mean"].to(state_seq.device, dtype=state_seq.dtype)
    std = norm_stats["std"].to(state_seq.device, dtype=state_seq.dtype)

    t = state_seq.shape[1]
    indices = torch.linspace(0, t - 1, num_latent_frames, device=state_seq.device).long()
    state_13 = state_seq[:, indices]
    state_gt_13 = (state_13 - mean) / std

    delta = state_13[:, 1:] - state_13[:, :-1]
    delta_norm = delta / std
    action_gt_13 = F.pad(delta_norm, (0, 0, 0, 1), value=0.0)
    s0_norm = state_gt_13[:, 0]
    return state_gt_13, action_gt_13, s0_norm


def split_pose_rgb_video(
    videos: torch.Tensor,
    *,
    pose_pixel_frames: int,
    rgb_pixel_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split concatenated ``[pose | rgb]`` video into separate tensors.

    ``videos`` is ``(B, F, C, H, W)``. Pose is padded to ``pose_pixel_frames`` by
    repeating the last pose frame when the pose segment is shorter.
    """
    pose_end = pose_pixel_frames
    if videos.shape[1] < pose_end + rgb_pixel_frames:
        raise ValueError(
            f"Video has {videos.shape[1]} frames, need at least "
            f"{pose_end + rgb_pixel_frames} for pose({pose_pixel_frames})+rgb({rgb_pixel_frames})."
        )
    pose_frames = videos[:, :pose_end]
    if pose_frames.shape[1] < pose_pixel_frames:
        pad_count = pose_pixel_frames - pose_frames.shape[1]
        pad = pose_frames[:, -1:].expand(-1, pad_count, -1, -1, -1)
        pose_frames = torch.cat([pose_frames, pad], dim=1)
    rgb_frames = videos[:, pose_end : pose_end + rgb_pixel_frames]
    return pose_frames, rgb_frames


def relayout_v5_video(
    videos: torch.Tensor,
    *,
    source_reason_frames: int = 24,
    source_rgb_frames: int = 25,
    target_reason_frames: int = 25,
    target_rgb_frames: int = 24,
) -> torch.Tensor:
    """Runtime compatibility path for old [24 reason | 25 RGB] clips."""
    required = source_reason_frames + source_rgb_frames
    if videos.shape[1] < required:
        raise ValueError(f"Video has {videos.shape[1]} frames, need at least {required} for v5 relayout.")
    reason = videos[:, :source_reason_frames]
    rgb = videos[:, source_reason_frames : source_reason_frames + source_rgb_frames]
    if reason.shape[1] < target_reason_frames:
        reason = torch.cat(
            [reason, reason[:, -1:].expand(-1, target_reason_frames - reason.shape[1], -1, -1, -1)],
            dim=1,
        )
    else:
        reason = reason[:, :target_reason_frames]
    rgb = rgb[:, :target_rgb_frames]
    return torch.cat([reason, rgb], dim=1)


def compute_sa_loss(
    sa_output: torch.Tensor,
    state_tokenizer: StateActionModule,
    state_gt: torch.Tensor,
    action_gt: torch.Tensor,
    lambda_s: float = 1.0,
    lambda_a: float = 1.0,
    lambda_c: float = 0.5,
) -> dict[str, torch.Tensor]:
    pred_state, pred_action = state_tokenizer.decode(sa_output)
    l_state = F.mse_loss(pred_state, state_gt)
    l_action = F.mse_loss(pred_action, action_gt)
    implied_delta = pred_state[:, 1:] - pred_state[:, :-1]
    predicted_delta = pred_action[:, :-1]
    l_consistency = F.mse_loss(implied_delta, predicted_delta)
    l_sa = lambda_s * l_state + lambda_a * l_action + lambda_c * l_consistency
    return {
        "L_state": l_state,
        "L_action": l_action,
        "L_consistency": l_consistency,
        "L_sa": l_sa,
    }


def compute_sa_raw_action_loss(
    sa_output: torch.Tensor,
    state_tokenizer: StateActionModule,
    state_gt: torch.Tensor,
    action_gt: torch.Tensor,
    state_delta_gt: torch.Tensor,
    state_norm_stats: dict[str, torch.Tensor],
    action_norm_stats: dict[str, torch.Tensor],
    lambda_s: float = 1.0,
    lambda_a: float = 1.0,
    lambda_g: float = 1.0,
    lambda_c: float = 0.1,
    gripper_continuous: bool = False,
) -> dict[str, torch.Tensor]:
    pred_state, pred_action = state_tokenizer.decode(sa_output)
    state_dim_loss = F.mse_loss(pred_state, state_gt, reduction="none").mean(dim=(0, 1))
    l_state = state_dim_loss.mean()

    action_mask = action_norm_stats.get("valid_action_mask")
    if action_mask is None:
        action_mask = torch.ones(7, device=pred_action.device, dtype=pred_action.dtype)
    else:
        action_mask = action_mask.to(device=pred_action.device, dtype=pred_action.dtype)
    action_dims = 7 if gripper_continuous else 6
    action_loss_mask = action_mask[:action_dims].clamp_min(0.0)
    pose_mask = action_mask[:6].clamp_min(0.0)
    cont_loss = F.smooth_l1_loss(
        pred_action[..., :action_dims],
        action_gt[..., :action_dims],
        reduction="none",
    ).mean(dim=(0, 1))
    l_action = (cont_loss * action_loss_mask).sum() / action_loss_mask.sum().clamp_min(1.0)

    if gripper_continuous:
        l_gripper = pred_action.new_zeros(())
    else:
        positive_rate = float(action_norm_stats.get("gripper_positive_rate", 0.5))
        pos_weight = torch.tensor(
            max((1.0 - positive_rate) / max(positive_rate, 1e-6), 1e-3),
            device=pred_action.device,
            dtype=pred_action.dtype,
        )
        gripper_loss = F.binary_cross_entropy_with_logits(
            pred_action[..., 6],
            action_gt[..., 6],
            pos_weight=pos_weight,
            reduction="none",
        )
        l_gripper = gripper_loss.mean() * action_mask[6].clamp_min(0.0)

    state_mean = state_norm_stats["mean"].to(pred_state.device, dtype=pred_state.dtype)
    state_std = state_norm_stats["std"].to(pred_state.device, dtype=pred_state.dtype)

    pred_state_real = pred_state * state_std + state_mean
    implied_delta = pred_state_real[:, 1:, :6] - pred_state_real[:, :-1, :6]
    delta_loss = F.smooth_l1_loss(
        implied_delta,
        state_delta_gt[:, :-1, :6].to(dtype=implied_delta.dtype),
        reduction="none",
    ).mean(dim=(0, 1))
    l_consistency = (delta_loss * pose_mask).sum() / pose_mask.sum().clamp_min(1.0)
    l_delta_gt = l_consistency
    l_sa = lambda_s * l_state + lambda_a * l_action + lambda_g * l_gripper + lambda_c * l_consistency
    return {
        "L_state": l_state,
        "L_action": l_action,
        "L_gripper": l_gripper,
        "L_consistency": l_consistency,
        "L_delta_gt": l_delta_gt,
        "L_sa": l_sa,
    }


def compute_sa_denoise_loss(
    sa_output: torch.Tensor,
    clean_sa: torch.Tensor,
    noise_sa: torch.Tensor,
    state_tokenizer: StateActionModule,
    *,
    state_gt: torch.Tensor | None = None,
    action_gt: torch.Tensor | None = None,
    lambda_s: float = 1.0,
    lambda_a: float = 1.0,
    lambda_decoded_state: float = 0.0,
    lambda_decoded_action: float = 0.0,
    loss_mask: torch.Tensor | None = None,
    decoded_loss_mask: torch.Tensor | None = None,
    decoded_state_loss_mask: torch.Tensor | None = None,
    decoded_action_loss_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Train SA tokens as a diffusion denoising target in token space.

    ``sa_output`` is expected to be a clean-token estimate. For CogVideoX
    v-prediction training this should be produced from the model velocity with
    ``scheduler.get_velocity(model_output, noisy_sa, timesteps)`` before calling
    this function. The optional decoded losses directly supervise the final 7D
    state/action heads used by inference.
    """
    del noise_sa  # Kept in the signature for compatibility with older callers.
    target = clean_sa.to(device=sa_output.device, dtype=sa_output.dtype)
    token_loss = F.mse_loss(sa_output, target, reduction="none").mean(dim=-1)

    mask = None
    if loss_mask is not None:
        mask = loss_mask.to(device=token_loss.device, dtype=token_loss.dtype)
        if mask.shape != token_loss.shape:
            raise ValueError(f"SA denoise loss mask shape {mask.shape} does not match token loss shape {token_loss.shape}.")

    def masked_mean(values: torch.Tensor, value_mask: torch.Tensor | None) -> torch.Tensor:
        if value_mask is None:
            return values.mean()
        return (values * value_mask).sum() / value_mask.sum().clamp_min(1.0)

    if hasattr(state_tokenizer, "chunk_token_count"):
        state_mask = None if mask is None else mask[..., 0::2]
        action_mask = None if mask is None else mask[..., 1::2]
        state_token_loss = masked_mean(token_loss[..., 0::2], state_mask)
        action_token_loss = masked_mean(token_loss[..., 1::2], action_mask)
    elif hasattr(state_tokenizer, "num_state_tokens"):
        num_state_tokens = int(getattr(state_tokenizer, "num_state_tokens"))
        state_mask = None if mask is None else mask[..., :num_state_tokens]
        action_mask = None if mask is None else mask[..., num_state_tokens:]
        state_token_loss = masked_mean(token_loss[..., :num_state_tokens], state_mask)
        action_token_loss = masked_mean(token_loss[..., num_state_tokens:], action_mask)
    else:
        state_token_loss = masked_mean(token_loss, mask)
        action_token_loss = masked_mean(token_loss, mask)

    l_consistency = sa_output.new_zeros(())
    l_gripper = sa_output.new_zeros(())
    l_sa_denoise = lambda_s * state_token_loss + lambda_a * action_token_loss

    l_decoded_state = sa_output.new_zeros(())
    l_decoded_action = sa_output.new_zeros(())
    if lambda_decoded_state > 0.0 or lambda_decoded_action > 0.0:
        if state_gt is None or action_gt is None:
            raise ValueError("Decoded SA losses require state_gt and action_gt.")
        pred_state, pred_action = state_tokenizer.decode(sa_output)
        decoded_state_mask = decoded_loss_mask if decoded_state_loss_mask is None else decoded_state_loss_mask
        decoded_action_mask = decoded_loss_mask if decoded_action_loss_mask is None else decoded_action_loss_mask
        decoded_state_mask_t = None
        decoded_action_mask_t = None
        decoded_shape = pred_state.shape[:2]
        if decoded_state_mask is not None:
            decoded_state_mask_t = decoded_state_mask.to(device=pred_state.device, dtype=pred_state.dtype)
            if decoded_state_mask_t.shape != decoded_shape:
                raise ValueError(
                    f"Decoded state loss mask shape {decoded_state_mask_t.shape} does not match "
                    f"decoded sequence shape {decoded_shape}."
                )
        if decoded_loss_mask is not None:
            decoded_loss_mask_t = decoded_loss_mask.to(device=pred_state.device, dtype=pred_state.dtype)
            if decoded_loss_mask_t.shape != decoded_shape:
                raise ValueError(
                    f"Decoded SA loss mask shape {decoded_loss_mask_t.shape} does not match decoded sequence shape {decoded_shape}."
                )
        if decoded_action_mask is not None:
            decoded_action_mask_t = decoded_action_mask.to(device=pred_action.device, dtype=pred_action.dtype)
            if decoded_action_mask_t.shape != decoded_shape:
                raise ValueError(
                    f"Decoded action loss mask shape {decoded_action_mask_t.shape} does not match "
                    f"decoded sequence shape {decoded_shape}."
                )
        if lambda_decoded_state > 0.0:
            state_loss = F.mse_loss(
                pred_state,
                state_gt.to(device=pred_state.device, dtype=pred_state.dtype),
                reduction="none",
            ).mean(dim=-1)
            l_decoded_state = masked_mean(state_loss, decoded_state_mask_t)
        if lambda_decoded_action > 0.0:
            action_loss = F.smooth_l1_loss(
                pred_action,
                action_gt.to(device=pred_action.device, dtype=pred_action.dtype),
                reduction="none",
            ).mean(dim=-1)
            l_decoded_action = masked_mean(action_loss, decoded_action_mask_t)

    l_sa = l_sa_denoise + lambda_decoded_state * l_decoded_state + lambda_decoded_action * l_decoded_action
    return {
        "L_state": state_token_loss,
        "L_action": action_token_loss,
        "L_gripper": l_gripper,
        "L_consistency": l_consistency,
        "L_delta_gt": l_consistency,
        "L_sa_denoise": l_sa_denoise,
        "L_decoded_state": l_decoded_state,
        "L_decoded_action": l_decoded_action,
        "L_sa": l_sa,
    }


def save_state_action_modules(
    path: str,
    sa_tokenizer: nn.Module,
    s0_encoder: S0Encoder,
    *,
    tokenizer_type: str = "legacy",
    steps_per_chunk: int | None = None,
    action_chunk_alignment: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "tokenizer_type": tokenizer_type,
        "sa_tokenizer": sa_tokenizer.state_dict(),
        "s0_encoder": s0_encoder.state_dict(),
        "s0_cond_tokens": s0_encoder.num_tokens,
    }
    if steps_per_chunk is not None:
        payload["steps_per_chunk"] = steps_per_chunk
    if action_chunk_alignment is not None:
        payload["action_chunk_alignment"] = action_chunk_alignment
    torch.save(payload, path)


def load_state_action_modules(
    path: str,
    sa_tokenizer: nn.Module,
    s0_encoder: S0Encoder,
    device: torch.device | None = None,
) -> dict[str, object]:
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    sa_tokenizer.load_state_dict(payload["sa_tokenizer"])
    s0_encoder.load_state_dict(payload["s0_encoder"])
    return payload
