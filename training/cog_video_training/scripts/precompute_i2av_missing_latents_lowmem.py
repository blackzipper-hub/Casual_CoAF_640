#!/usr/bin/env python3
"""Precompute only missing I2AV tensors with low peak GPU memory."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLCogVideoX
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


LEGACY_COGVIDEOX_ROOT = Path(__file__).resolve().parents[1] / "finetrainers" / "examples" / "_legacy" / "training" / "cogvideox"
sys.path.insert(0, str(LEGACY_COGVIDEOX_ROOT))

from dataset import VideoDatasetWithResizing  # noqa: E402
from prepare_dataset import compute_prompt_embeddings  # noqa: E402


DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--max_num_frames", type=int, default=49)
    parser.add_argument("--max_sequence_length", type=int, default=226)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--vae_dtype", choices=sorted(DTYPE_MAP), default="bf16")
    parser.add_argument("--prompt_dtype", choices=sorted(DTYPE_MAP), default="fp32")
    parser.add_argument("--vae_device", default="cuda")
    parser.add_argument("--prompt_device", default="cpu")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--stem_prefix", default=None, help="Only process videos whose filename stem starts with this prefix.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_slicing", action="store_true")
    parser.add_argument("--use_tiling", action="store_true")
    return parser.parse_args()


def batched(items: list[int], batch_size: int):
    for offset in range(0, len(items), batch_size):
        yield items[offset : offset + batch_size]


def tensor_paths(video_path: Path) -> tuple[Path, Path, Path]:
    stem = video_path.stem
    root = video_path.parent.parent
    return (
        root / "video_latents" / f"{stem}.pt",
        root / "image_latents" / f"{stem}.pt",
        root / "prompt_embeds" / f"{stem}.pt",
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    data_root = args.data_root
    for name in ("video_latents", "image_latents", "prompt_embeds"):
        (data_root / name).mkdir(parents=True, exist_ok=True)

    dataset = VideoDatasetWithResizing(
        data_root=str(data_root),
        caption_column="prompt.txt",
        video_column="videos.txt",
        image_column="images.txt",
        max_num_frames=args.max_num_frames,
        id_token="COAF",
        height_buckets=[args.height],
        width_buckets=[args.width],
        frame_buckets=[args.max_num_frames],
        load_tensors=False,
        random_flip=None,
        image_to_video=True,
    )

    missing = []
    for idx, video_path in enumerate(dataset.video_paths):
        if args.stem_prefix is not None and not video_path.stem.startswith(args.stem_prefix):
            continue
        outputs = tensor_paths(video_path)
        if args.overwrite or not all(path.is_file() for path in outputs):
            missing.append(idx)
    if args.max_samples is not None:
        missing = missing[: args.max_samples]
    print(f"Missing tensor samples: {len(missing)} / {len(dataset)}")
    if not missing:
        return

    prompt_device = torch.device(args.prompt_device)
    prompt_dtype = DTYPE_MAP[args.prompt_dtype]
    tokenizer = T5Tokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.model_path,
        subfolder="text_encoder",
        torch_dtype=prompt_dtype,
    ).to(prompt_device)
    text_encoder.eval()

    for batch_indices in tqdm(list(batched(missing, args.batch_size)), desc="prompt embeds"):
        prompts = [dataset.id_token + dataset.prompts[idx] for idx in batch_indices]
        prompt_embeds = compute_prompt_embeddings(
            tokenizer,
            text_encoder,
            prompts,
            args.max_sequence_length,
            prompt_device,
            prompt_dtype,
            requires_grad=False,
        )
        for item_idx, idx in enumerate(batch_indices):
            _, _, prompt_path = tensor_paths(dataset.video_paths[idx])
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(prompt_embeds[item_idx].detach().cpu(), prompt_path)

    del text_encoder, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vae_device = torch.device(args.vae_device if torch.cuda.is_available() or args.vae_device == "cpu" else "cpu")
    vae_dtype = DTYPE_MAP[args.vae_dtype]
    vae = AutoencoderKLCogVideoX.from_pretrained(args.model_path, subfolder="vae", torch_dtype=vae_dtype).to(vae_device)
    if args.use_slicing:
        vae.enable_slicing()
    if args.use_tiling:
        vae.enable_tiling()
    vae.eval()

    for batch_indices in tqdm(list(batched(missing, args.batch_size)), desc="vae latents"):
        samples = [dataset[idx] for idx in batch_indices]
        images = torch.stack([sample["image"] for sample in samples]).to(device=vae_device, dtype=vae_dtype, non_blocking=True)
        videos = torch.stack([sample["video"] for sample in samples]).to(device=vae_device, dtype=vae_dtype, non_blocking=True)

        image_latents = vae._encode(images.permute(0, 2, 1, 3, 4)).to(memory_format=torch.contiguous_format, dtype=vae_dtype)
        video_latents = vae._encode(videos.permute(0, 2, 1, 3, 4)).to(memory_format=torch.contiguous_format, dtype=vae_dtype)

        for item_idx, idx in enumerate(batch_indices):
            video_path, image_path, _ = tensor_paths(dataset.video_paths[idx])
            video_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(video_latents[item_idx].detach().cpu(), video_path)
            torch.save(image_latents[item_idx].detach().cpu(), image_path)

    print(f"Completed missing tensor precompute under {data_root}")


if __name__ == "__main__":
    main()
