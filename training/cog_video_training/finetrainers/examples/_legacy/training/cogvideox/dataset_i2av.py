"""Dataset helpers for I2AV training with state/action manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dataset import VideoDatasetWithResizing


class I2AVVideoDataset(VideoDatasetWithResizing):
    def __init__(
        self,
        *args,
        state_column: str = "state_paths.txt",
        action_column: str = "action_paths.txt",
        track_column: str | None = None,
        track_visible_column: str | None = None,
        relayout_v5: bool = False,
        v5_source_reason_frames: int = 24,
        v5_source_rgb_frames: int = 25,
        v5_reason_frames: int = 25,
        v5_rgb_frames: int = 24,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.relayout_v5 = relayout_v5
        self.v5_source_reason_frames = v5_source_reason_frames
        self.v5_source_rgb_frames = v5_source_rgb_frames
        self.v5_reason_frames = v5_reason_frames
        self.v5_rgb_frames = v5_rgb_frames
        self.state_paths = self._load_paths(state_column, "state_paths.txt")
        self.action_paths = self._load_paths(action_column, "action_paths.txt")
        self.track_paths = self._load_optional_paths(track_column, "track_paths.txt")
        self.track_visible_paths = self._load_optional_paths(track_visible_column, "track_visible_paths.txt")
        if len(self.state_paths) != len(self.video_paths):
            raise ValueError(
                f"state/video count mismatch: {len(self.state_paths)} vs {len(self.video_paths)}"
            )
        if len(self.action_paths) != len(self.video_paths):
            raise ValueError(
                f"action/video count mismatch: {len(self.action_paths)} vs {len(self.video_paths)}"
            )
        if self.track_paths and len(self.track_paths) != len(self.video_paths):
            raise ValueError(
                f"track/video count mismatch: {len(self.track_paths)} vs {len(self.video_paths)}"
            )
        if self.track_visible_paths and len(self.track_visible_paths) != len(self.video_paths):
            raise ValueError(
                f"track_visible/video count mismatch: {len(self.track_visible_paths)} vs {len(self.video_paths)}"
            )

    def _load_paths(self, column: str, fallback_name: str) -> list[Path]:
        if self.dataset_file is not None:
            df = pd.read_csv(self.dataset_file)
            if column not in df.columns:
                raise KeyError(f"Missing column {column!r} in {self.dataset_file}")
            paths = [Path(str(value)) for value in df[column].tolist()]
        else:
            path_file = self.data_root / column
            if not path_file.is_file():
                path_file = self.data_root / fallback_name
            if not path_file.is_file():
                raise FileNotFoundError(f"Missing manifest {path_file}")
            paths = [Path(line.strip()) for line in path_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [path if path.is_absolute() else self.data_root / path for path in paths]

    def _load_optional_paths(self, column: str | None, fallback_name: str) -> list[Path]:
        if column is None:
            path_file = self.data_root / fallback_name
            if not path_file.is_file():
                return []
            column = fallback_name
        return self._load_paths(column, fallback_name)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, list):
            return index
        sample = super().__getitem__(index)
        if self.relayout_v5 and not self.load_tensors:
            sample["video"] = self._relayout_video_v5(sample["video"])
            if self.image_column is None:
                sample["image"] = sample["video"][self.v5_reason_frames : self.v5_reason_frames + 1].clone()
        state = np.load(self.state_paths[index]).astype(np.float32)
        action = np.load(self.action_paths[index]).astype(np.float32)
        if state.ndim != 2 or state.shape[-1] != 7:
            raise ValueError(f"Bad state shape {state.shape} from {self.state_paths[index]}")
        if action.ndim != 2 or action.shape[-1] != 7:
            raise ValueError(f"Bad action shape {action.shape} from {self.action_paths[index]}")
        if self.relayout_v5:
            state = self._resize_state_v5(state)
            action = self._resize_state_v5(action)
        sample["state"] = torch.from_numpy(state)
        sample["action"] = torch.from_numpy(action)
        if self.track_paths:
            track = np.load(self.track_paths[index]).astype(np.float32)
            if track.ndim != 2 or track.shape[-1] != 2:
                raise ValueError(f"Bad track shape {track.shape} from {self.track_paths[index]}")
            sample["track"] = torch.from_numpy(self._resize_track(track))
            if self.track_visible_paths:
                visible = np.load(self.track_visible_paths[index]).astype(bool)
                if visible.ndim != 1:
                    raise ValueError(f"Bad track visible shape {visible.shape} from {self.track_visible_paths[index]}")
                sample["track_valid_mask"] = torch.from_numpy(self._resize_visible(visible))
            else:
                metadata = sample.get("video_metadata", {})
                track_width = float(metadata.get("width", 256))
                track_height = float(metadata.get("height", 256))
                valid = (
                    (track[:, 0] >= 0)
                    & (track[:, 0] < track_width)
                    & (track[:, 1] >= 0)
                    & (track[:, 1] < track_height)
                )
                sample["track_valid_mask"] = torch.from_numpy(self._resize_visible(valid))
        return sample

    def _relayout_video_v5(self, video: torch.Tensor) -> torch.Tensor:
        required = self.v5_source_reason_frames + self.v5_source_rgb_frames
        if video.shape[0] < required:
            raise ValueError(f"Video has {video.shape[0]} frames, need at least {required} for v5 relayout.")
        reason = video[: self.v5_source_reason_frames]
        rgb = video[self.v5_source_reason_frames : self.v5_source_reason_frames + self.v5_source_rgb_frames]
        if reason.shape[0] < self.v5_reason_frames:
            pad = reason[-1:].expand(self.v5_reason_frames - reason.shape[0], -1, -1, -1)
            reason = torch.cat([reason, pad], dim=0)
        else:
            reason = reason[: self.v5_reason_frames]
        rgb = rgb[: self.v5_rgb_frames]
        return torch.cat([reason, rgb], dim=0)

    def _resize_state_v5(self, seq: np.ndarray) -> np.ndarray:
        if seq.shape[0] == self.v5_reason_frames:
            return seq
        if seq.shape[0] > self.v5_reason_frames:
            return seq[: self.v5_reason_frames]
        pad = np.repeat(seq[-1:], self.v5_reason_frames - seq.shape[0], axis=0)
        return np.concatenate([seq, pad], axis=0)

    def _resize_track(self, seq: np.ndarray) -> np.ndarray:
        if seq.shape[0] == self.v5_reason_frames:
            return seq
        if seq.shape[0] > self.v5_reason_frames:
            return seq[: self.v5_reason_frames]
        pad = np.repeat(seq[-1:], self.v5_reason_frames - seq.shape[0], axis=0)
        return np.concatenate([seq, pad], axis=0)

    def _resize_visible(self, seq: np.ndarray) -> np.ndarray:
        seq = seq.astype(bool)
        if seq.shape[0] == self.v5_reason_frames:
            return seq
        if seq.shape[0] > self.v5_reason_frames:
            return seq[: self.v5_reason_frames]
        pad = np.repeat(seq[-1:], self.v5_reason_frames - seq.shape[0], axis=0)
        return np.concatenate([seq, pad], axis=0)


class I2AVCollateFunction:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]
        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        images = torch.stack([x["image"] for x in data[0]]).to(dtype=self.weight_dtype, non_blocking=True)
        videos = torch.stack([x["video"] for x in data[0]]).to(dtype=self.weight_dtype, non_blocking=True)
        states = torch.stack([x["state"] for x in data[0]]).to(dtype=torch.float32, non_blocking=True)
        actions = torch.stack([x["action"] for x in data[0]]).to(dtype=torch.float32, non_blocking=True)
        tracks = None
        track_valid_masks = None
        if "track" in data[0][0]:
            tracks = torch.stack([x["track"] for x in data[0]]).to(dtype=torch.float32, non_blocking=True)
            track_valid_masks = torch.stack([x["track_valid_mask"] for x in data[0]]).to(
                dtype=torch.bool, non_blocking=True
            )

        batch = {
            "images": images,
            "videos": videos,
            "prompts": prompts,
            "state": states,
            "action": actions,
        }
        if tracks is not None:
            batch["track"] = tracks
            batch["track_valid_mask"] = track_valid_masks
        return batch
