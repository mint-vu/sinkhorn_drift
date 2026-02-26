"""
Encode ImageNet RGB images to RAE latents (ImageNet256 -> C x H x W, e.g. 768x16x16).

RAE (Representation Autoencoder) uses a frozen pretrained encoder (DINOv2, SigLIP2, MAE)
to produce semantically rich latents, replacing the SD-VAE tokenizer.

Usage:
  torchrun --nproc_per_node=8 -m imagenet.encode_latents_rae \
    --rae-config rae/configs/stage1/pretrained/DINOv2-B.yaml \
    --imagenet-root /path/to/imagenet \
    --split train \
    --out-dir data/imagenet256_rae_latents
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

# Add RAE source to path so its internal imports (stage1, utils.*) resolve.
_RAE_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rae", "src")
if _RAE_SRC not in sys.path:
    sys.path.insert(0, _RAE_SRC)

from imagenet.data.imagenet_folders import build_imagenet_dataset
from imagenet.data.latents_memmap import (
    final_paths,
    merge_shards_to_final,
    open_latents_memmap,
    shard_paths,
    write_meta,
)
from imagenet.utils.dist import DistInfo, barrier, init_distributed, is_main_process
from imagenet.utils.misc import seed_all


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode ImageNet images to RAE latents.")
    p.add_argument("--rae-config", type=str, required=True, help="Path to RAE yaml config (e.g. rae/configs/stage1/pretrained/DINOv2-B.yaml).")
    p.add_argument("--imagenet-root", type=str, default="/home/public/imagenet")
    p.add_argument("--split", type=str, choices=["train", "val"], required=True)
    p.add_argument("--out-dir", type=str, default="data/imagenet256_rae_latents")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--image-size", type=int, default=256, help="Resize/crop target size before feeding to RAE.")
    p.add_argument("--save-dtype", type=str, choices=["fp16", "fp32"], default="fp16", help="Dtype for saved latents.")
    p.add_argument("--merge", action="store_true", help="If DDP, merge shards into a single file on rank0.")
    p.add_argument("--max-items", type=int, default=0, help="Debug: encode only first N items (0=all).")
    return p.parse_args()


def _load_rae_model(config_path: str, device: torch.device):
    """Load RAE model from yaml config."""
    from omegaconf import OmegaConf
    from utils.model_utils import instantiate_from_config

    cfg = OmegaConf.load(config_path)
    stage1_cfg = cfg.get("stage_1")
    if stage1_cfg is None:
        raise ValueError(f"RAE config {config_path} must have a 'stage_1' section.")

    # Resolve relative paths in config against the config file's directory.
    config_dir = os.path.dirname(os.path.abspath(config_path))
    rae_root = os.path.dirname(config_dir)  # go up from configs/stage1/pretrained/ to rae/
    # Walk up to find the rae root (directory containing 'src/')
    while rae_root and not os.path.isdir(os.path.join(rae_root, "src")):
        parent = os.path.dirname(rae_root)
        if parent == rae_root:
            break
        rae_root = parent

    params = dict(stage1_cfg.get("params", {}))
    for key in ("decoder_config_path", "pretrained_decoder_path", "normalization_stat_path"):
        val = params.get(key)
        if val is not None and not os.path.isabs(str(val)):
            # Try config dir, then rae root
            for base in (config_dir, rae_root, os.path.join(rae_root, "src")):
                candidate = os.path.join(base, str(val))
                if os.path.exists(candidate):
                    params[key] = candidate
                    break

    stage1_dict = dict(stage1_cfg)
    stage1_dict["params"] = params

    rae = instantiate_from_config(stage1_dict).to(device)
    rae.eval()
    return rae


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    dist_info: DistInfo = init_distributed(device=args.device)
    seed_all(args.seed + dist_info.rank)

    # RAE expects [0, 1] input (it does internal normalization).
    transform = transforms.Compose([
        transforms.Resize(int(args.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(int(args.image_size)),
        transforms.ToTensor(),  # [0, 1]
    ])
    ds, _ = build_imagenet_dataset(args.imagenet_root, args.split, transform=transform)
    total_n = len(ds)
    if args.max_items and args.max_items > 0:
        total_n = min(total_n, int(args.max_items))
        ds.samples = ds.samples[:total_n]
        ds.targets = ds.targets[:total_n]

    shard_indices = list(range(dist_info.rank, total_n, dist_info.world_size))
    ds_shard = Subset(ds, shard_indices)

    g = torch.Generator()
    g.manual_seed(int(args.seed) + int(dist_info.rank))

    def _worker_init_fn(worker_id: int) -> None:
        seed = int(torch.initial_seed()) % (2**32)
        np.random.seed(seed)
        random.seed(seed)

    dl = DataLoader(
        ds_shard,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
        generator=g,
    )

    rae = _load_rae_model(args.rae_config, device=dist_info.device)

    # Probe latent shape from a single dummy image.
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, int(args.image_size), int(args.image_size), device=dist_info.device)
        dummy_lat = rae.encode(dummy)
        latent_shape = tuple(dummy_lat.shape[1:])  # e.g. (768, 16, 16)
    if is_main_process():
        print(f"RAE latent shape: {latent_shape}")

    save_np_dtype = np.float16 if args.save_dtype == "fp16" else np.float32

    # Create shard memmaps.
    shard_n = len(shard_indices)
    if dist_info.world_size == 1:
        lat_path, lab_path = final_paths(args.out_dir, args.split)
        lat_mm = open_latents_memmap(lat_path, shape=(shard_n, *latent_shape), dtype=save_np_dtype, mode="w+")
        lab_mm = open_latents_memmap(lab_path, shape=(shard_n,), dtype=np.int64, mode="w+")
        idx_mm = None
    else:
        sp = shard_paths(args.out_dir, args.split, dist_info.rank)
        lat_mm = open_latents_memmap(sp.latents_path, shape=(shard_n, *latent_shape), dtype=save_np_dtype, mode="w+")
        lab_mm = open_latents_memmap(sp.labels_path, shape=(shard_n,), dtype=np.int64, mode="w+")
        idx_mm = open_latents_memmap(sp.indices_path, shape=(shard_n,), dtype=np.int64, mode="w+")

    pbar = tqdm(dl, disable=not is_main_process(), desc=f"rae-encode[{args.split}]")
    write_ptr = 0
    for images, labels, indices in pbar:
        images = images.to(dist_info.device, non_blocking=True)
        with torch.inference_mode():
            latents = rae.encode(images)

        lat_np = latents.detach().cpu().to(torch.float16 if args.save_dtype == "fp16" else torch.float32).numpy()
        lab_np = labels.numpy().astype(np.int64, copy=False)
        idx_np = indices.numpy().astype(np.int64, copy=False)

        bsz = lat_np.shape[0]
        lat_mm[write_ptr : write_ptr + bsz] = lat_np
        lab_mm[write_ptr : write_ptr + bsz] = lab_np
        if idx_mm is not None:
            idx_mm[write_ptr : write_ptr + bsz] = idx_np
        write_ptr += bsz

    if write_ptr != shard_n:
        raise RuntimeError(f"Shard write count mismatch: wrote {write_ptr} but expected {shard_n}")
    lat_mm.flush()
    lab_mm.flush()
    if idx_mm is not None:
        idx_mm.flush()

    barrier()

    if dist_info.distributed and args.merge and is_main_process():
        merge_shards_to_final(args.out_dir, args.split, dist_info.world_size, total_n=total_n)

    if is_main_process():
        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "imagenet_root": args.imagenet_root,
            "split": args.split,
            "total_n": total_n,
            "seed": int(args.seed),
            "world_size": int(dist_info.world_size),
            "latent_shape": list(latent_shape),
            "save_dtype": args.save_dtype,
            "rae_config": os.path.abspath(args.rae_config),
            "image_size": int(args.image_size),
        }
        meta_path = write_meta(args.out_dir, args.split, meta)
        print(f"Wrote meta: {meta_path}")


if __name__ == "__main__":
    main()
