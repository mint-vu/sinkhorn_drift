from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


LatentsDType = Literal["float16", "float32"]


@dataclass(frozen=True)
class LatentsShardPaths:
    latents_path: str
    labels_path: str
    indices_path: str


def _np_dtype(dtype: LatentsDType) -> np.dtype:
    if dtype == "float16":
        return np.float16
    if dtype == "float32":
        return np.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def open_latents_memmap(path: str, shape: Tuple[int, ...], dtype: np.dtype, mode: str):
    """Create or open a `.npy` memmap with numpy header."""
    path_p = Path(path)
    path_p.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(str(path_p), mode=mode, dtype=dtype, shape=shape)


def shard_paths(out_dir: str, split: str, rank: int) -> LatentsShardPaths:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return LatentsShardPaths(
        latents_path=str(out / f"{split}_latents.rank{rank}.npy"),
        labels_path=str(out / f"{split}_labels.rank{rank}.npy"),
        indices_path=str(out / f"{split}_indices.rank{rank}.npy"),
    )


def final_paths(out_dir: str, split: str) -> tuple[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return str(out / f"{split}_latents.npy"), str(out / f"{split}_labels.npy")


def write_meta(out_dir: str, split: str, meta: dict[str, Any]) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{split}_meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return str(path)


def merge_shards_to_final(
    out_dir: str,
    split: str,
    world_size: int,
    total_n: int,
    *,
    latent_shape: Tuple[int, ...] | None = None,
    latent_dtype: np.dtype = np.float16,
) -> tuple[str, str]:
    """Merge rank shards into a single memmap in dataset order.

    If *latent_shape* is not given it is auto-detected from the first shard.
    """
    latents_path, labels_path = final_paths(out_dir, split)

    if latent_shape is None:
        shard0 = np.load(shard_paths(out_dir, split, 0).latents_path, mmap_mode="r")
        latent_shape = shard0.shape[1:]
        latent_dtype = shard0.dtype

    latents_mm = open_latents_memmap(latents_path, shape=(total_n, *latent_shape), dtype=latent_dtype, mode="w+")
    labels_mm = open_latents_memmap(labels_path, shape=(total_n,), dtype=np.int64, mode="w+")

    for r in range(world_size):
        sp = shard_paths(out_dir, split, r)
        idx = np.load(sp.indices_path, mmap_mode="r")
        lat = np.load(sp.latents_path, mmap_mode="r")
        lab = np.load(sp.labels_path, mmap_mode="r")
        if idx.shape[0] != lat.shape[0] or idx.shape[0] != lab.shape[0]:
            raise ValueError(f"Shard {r} has inconsistent lengths: idx={idx.shape}, lat={lat.shape}, lab={lab.shape}")
        latents_mm[idx] = lat
        labels_mm[idx] = lab

    latents_mm.flush()
    labels_mm.flush()
    return latents_path, labels_path


class LatentsDataset(Dataset):
    """Memory-mapped dataset of pre-encoded latents."""

    def __init__(self, latents_path: str, labels_path: str):
        self.latents = np.load(latents_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        if self.latents.shape[0] != self.labels.shape[0]:
            raise ValueError(f"Latents/labels length mismatch: {self.latents.shape} vs {self.labels.shape}")
        if self.latents.ndim != 4:
            raise ValueError(f"Expected 4D latent array [N,C,H,W], got shape {self.latents.shape}")

    def __len__(self) -> int:  # type: ignore[override]
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):  # type: ignore[override]
        lat = torch.from_numpy(self.latents[idx])
        lab = int(self.labels[idx])
        return lat, lab
