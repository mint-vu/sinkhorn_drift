# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Runs distributed stage-1 encoding with a pre-trained model.
Inputs are loaded from an ImageFolder dataset, processed with center crops,
and the encoded latents are saved as per-sample .npy files.
Also reports PSNR on the first batch (rank 0 only) using decoded reconstructions.
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import List, Set

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np

from stage1 import RAE
from utils.model_utils import instantiate_from_config
from utils.torch_utils import safe_torch_load
from utils.train_utils import parse_configs


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class IndexedImageFolder(ImageFolder):
    """ImageFolder that also returns the dataset index."""

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image, index


def sanitize_component(component: str) -> str:
    """Replace OS separators to keep path components valid."""
    return component.replace(os.sep, "-")


def compute_batch_psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Mean PSNR (dB) over a batch, assuming inputs are in [0, 1].
    """
    x = x.float()
    y = y.float()
    mse_per_sample = torch.mean((x - y) ** 2, dim=(1, 2, 3))
    inf_tensor = torch.full_like(mse_per_sample, float("inf"))
    psnr_per_sample = torch.where(
        mse_per_sample > 0,
        10.0 * torch.log10(1.0 / mse_per_sample),
        inf_tensor,
    )
    return psnr_per_sample.mean().item()


def _sample_output_path(base_dir: str, idx: int, ext: str, shard_size: int = 0) -> str:
    if shard_size > 0:
        shard = int(idx) // int(shard_size)
        shard_dir = os.path.join(base_dir, f"{shard:05d}")
        os.makedirs(shard_dir, exist_ok=True)
        return os.path.join(shard_dir, f"{idx:06d}{ext}")
    return os.path.join(base_dir, f"{idx:06d}{ext}")


def save_sample_batch_npy(base_dir: str, indices: List[int], latents: np.ndarray, shard_size: int = 0) -> None:
    for latent, idx in zip(latents, indices):
        path = _sample_output_path(base_dir, int(idx), ".npy", shard_size=shard_size)
        np.save(path, latent, allow_pickle=False)


def save_sample_batch_torch(base_dir: str, indices: List[int], latents: torch.Tensor, shard_size: int = 0) -> None:
    for latent, idx in zip(latents, indices):
        path = _sample_output_path(base_dir, int(idx), ".pt", shard_size=shard_size)
        # clone() ensures each file owns only the per-sample tensor storage.
        torch.save(latent.clone().contiguous(), path)


def save_sample_batch_torch_records(
    base_dir: str,
    indices: List[int],
    latents: torch.Tensor,
    shard_size: int = 0,
) -> None:
    """
    B/per-prompt-style save: one .pt file per sample with a record dict.
    """
    for latent, idx in zip(latents, indices):
        idx = int(idx)
        path = _sample_output_path(base_dir, idx, ".pt", shard_size=shard_size)
        rec = {
            "sample_index": idx,
            # clone() ensures each file stores only this sample's tensor storage.
            "latent": latent.clone().contiguous(),
        }
        torch.save(rec, path)


def save_batch_npz(path: str, latents: np.ndarray, indices: np.ndarray) -> None:
    np.savez(path, latents=latents, indices=indices)


def _resolve_existing_path(path: str, config_path: str | None = None) -> str:
    """Resolve a file path from cwd/config/project root and ensure it exists."""
    normalized = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(normalized):
        if os.path.isfile(normalized):
            return normalized
        raise FileNotFoundError(f"Stats file does not exist: {normalized}")

    roots = [os.getcwd()]
    if config_path is not None:
        roots.append(os.path.dirname(os.path.abspath(config_path)))
    roots.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    checked = []
    seen = set()
    for root in roots:
        root = os.path.abspath(root)
        if root in seen:
            continue
        seen.add(root)
        candidate = os.path.abspath(os.path.join(root, normalized))
        checked.append(candidate)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Stats file does not exist. Checked: " + ", ".join(checked)
    )


def _to_positive_scalar(value, source_name: str) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar in `{source_name}`, got tensor shape {tuple(value.shape)}")
        value = value.item()
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"`{source_name}` must be a finite positive scalar, got {scalar}")
    return scalar


def _derive_scale_factor_from_stats(stats: dict) -> tuple[float, str]:
    """
    Match train/sample convention:
    - Prefer explicit scalar keys if present in stats payload.
    - Otherwise, if stats include normalization tensors (`mean`/`var`), use unit scale.
      Stage-2 train/sample paths use RAE's built-in normalization directly and do not
      apply an additional global scalar multiplier to latents.
    """
    for key in ("latent_scaling_factor", "latent_scale_factor", "latent_scale", "scale_factor", "scaling_factor"):
        if key in stats and stats[key] is not None:
            return _to_positive_scalar(stats[key], f"stats.{key}"), f"stat.{key}"

    mean = stats.get("mean", None)
    var = stats.get("var", None)
    if mean is None and var is None:
        raise KeyError(
            "Stats file does not contain an explicit scalar scale key "
            "(`latent_scaling_factor`/`latent_scale_factor`/...) "
            "or normalization tensors (`mean`/`var`)."
        )

    if mean is not None:
        mean_t = torch.as_tensor(mean)
        if mean_t.numel() == 0:
            raise ValueError("`stats.mean` is empty.")
    if var is not None:
        var_t = torch.as_tensor(var)
        if var_t.numel() == 0:
            raise ValueError("`stats.var` is empty.")
        if bool((var_t < 0).any()):
            raise ValueError("`stats.var` contains negative entries.")
        if mean is not None and mean_t.shape != var_t.shape:
            raise ValueError(
                f"`stats.mean` shape {tuple(mean_t.shape)} does not match `stats.var` shape {tuple(var_t.shape)}"
            )

    return 1.0, "train/sample-convention"


def resolve_latent_scale_factor(args, rae_config) -> tuple[float, str]:
    """Resolve latent scaling factor from CLI, or derive it from Stage-1 stats."""
    if args.latent_scale_factor is not None:
        return _to_positive_scalar(args.latent_scale_factor, "args.latent_scale_factor"), "cli"

    stat_path = args.latent_scale_stat_path
    stat_path_source = "cli.latent_scale_stat_path"
    if stat_path is None and rae_config is not None:
        params = rae_config.get("params", None)
        if params is not None:
            stat_path = params.get("normalization_stat_path", None)
            if stat_path is not None:
                stat_path_source = "stage_1.params.normalization_stat_path"

    if stat_path is None:
        raise ValueError(
            "No latent scaling factor found. Set --latent-scale-factor, or provide a stats file via "
            "--latent-scale-stat-path / stage_1.params.normalization_stat_path."
        )

    resolved_stat_path = _resolve_existing_path(str(stat_path), config_path=args.config)
    stats = safe_torch_load(resolved_stat_path, map_location="cpu")
    if not isinstance(stats, dict):
        raise TypeError(f"Expected stats payload to be a dict, got {type(stats).__name__}")
    scale, derived_source = _derive_scale_factor_from_stats(stats)
    return scale, f"{derived_source} ({stat_path_source}: {resolved_stat_path})"


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling with DDP requires at least one GPU.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    rae_config, _, _, _, _, _, _, _ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError("Config must provide a stage_1 section.")
    latent_scale_factor, latent_scale_source = resolve_latent_scale_factor(args, rae_config)

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
    ])
    dataset = IndexedImageFolder(args.data_path, transform=transform)
    total_available = len(dataset)
    if total_available == 0:
        raise ValueError(f"No images found at {args.data_path}.")

    requested = total_available if args.num_samples is None else min(args.num_samples, total_available)
    if requested <= 0:
        raise ValueError("Number of samples to process must be positive.")

    selected_indices = list(range(requested))
    rank_indices = selected_indices[rank::world_size]
    subset = Subset(dataset, rank_indices)

    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)

    model_target = rae_config.get("target", "stage1")
    ckpt_path = rae_config.get("ckpt")
    ckpt_name = "pretrained" if not ckpt_path else os.path.splitext(os.path.basename(str(ckpt_path)))[0]
    folder_components: List[str] = [
        sanitize_component(str(model_target).split(".")[-1]),
        sanitize_component(ckpt_name),
        f"bs{args.per_proc_batch_size}",
        args.precision,
    ]
    if latent_scale_factor != 1.0:
        folder_components.append(f"zsf{latent_scale_factor:g}")
    folder_name = "-".join(folder_components)
    possible_folder_name = os.environ.get('SAVE_FOLDER', None)
    if possible_folder_name:
        folder_name = possible_folder_name
    sample_folder_dir = os.path.join(args.sample_dir, folder_name)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving encoded latents at {sample_folder_dir}")
        print(
            f"Save config: format={args.save_format}, dtype={args.save_dtype}, "
            f"workers={args.save_workers}, max_inflight={args.max_inflight_saves}, "
            f"sample_backend={args.sample_backend}, sample_shard_size={args.sample_shard_size}"
        )
        print(f"Latent scaling factor: {latent_scale_factor:g} (source: {latent_scale_source})")
        metadata = {
            "save_format": args.save_format,
            "sample_backend": args.sample_backend,
            "save_dtype": args.save_dtype,
            "latent_scale_factor": latent_scale_factor,
            "latent_scale_source": latent_scale_source,
        }
        with open(os.path.join(sample_folder_dir, "_save_meta.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    dist.barrier()

    loader_kwargs = dict(
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers
    loader = DataLoader(subset, **loader_kwargs)
    local_total = len(rank_indices)
    iterator = tqdm(loader, desc="Stage1 encode", total=math.ceil(local_total / args.per_proc_batch_size)) if rank == 0 else loader

    psnr_reported = False
    save_torch_dtype = torch.float16 if args.save_dtype == "fp16" else torch.float32
    max_inflight_saves = max(1, args.max_inflight_saves)
    pending_saves: Set[Future] = set()
    batch_id = 0

    def drain_pending(wait_for_all: bool = False) -> None:
        nonlocal pending_saves
        if not pending_saves:
            return
        if wait_for_all:
            done, pending = wait(pending_saves)
        else:
            done, pending = wait(pending_saves, return_when=FIRST_COMPLETED)
        for fut in done:
            fut.result()
        pending_saves = set(pending)

    with ThreadPoolExecutor(max_workers=max(1, args.save_workers), thread_name_prefix="latent-save") as executor:
        with torch.inference_mode():
            for images, indices in iterator:
                if images.numel() == 0:
                    continue
                images = images.to(device, non_blocking=True)

                with autocast(**autocast_kwargs):
                    latents = rae.encode(images)

                    # Report PSNR only on the first batch from rank 0.
                    if rank == 0 and not psnr_reported:
                        recon = rae.decode(latents).clamp(0, 1)
                        psnr = compute_batch_psnr(images, recon)
                        print(f"First-batch PSNR (rank 0): {psnr:.4f} dB")
                        psnr_reported = True

                latents_to_save = latents if latent_scale_factor == 1.0 else (latents * latent_scale_factor)
                latents_cpu = latents_to_save.detach().to(dtype=save_torch_dtype).cpu().contiguous()
                indices_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)

                if args.save_format == "batch":
                    latents_np = latents_cpu.numpy()
                    batch_path = os.path.join(sample_folder_dir, f"rank{rank:03d}-batch{batch_id:07d}.npz")
                    batch_indices = np.asarray(indices_list, dtype=np.int64)
                    pending_saves.add(executor.submit(save_batch_npz, batch_path, latents_np, batch_indices))
                    batch_id += 1

                    while len(pending_saves) >= max_inflight_saves:
                        drain_pending(wait_for_all=False)
                else:
                    # Script-B per-prompt-style streaming save:
                    #   compute one batch -> save each sample immediately -> release batch tensors
                    # This avoids CPU RAM growth from queued async sample saves.
                    if args.sample_backend == "torch":
                        if args.sample_torch_record:
                            save_sample_batch_torch_records(
                                sample_folder_dir,
                                indices_list,
                                latents_cpu,
                                args.sample_shard_size,
                            )
                        else:
                            save_sample_batch_torch(
                                sample_folder_dir,
                                indices_list,
                                latents_cpu,
                                args.sample_shard_size,
                            )
                    else:
                        save_sample_batch_npy(
                            sample_folder_dir,
                            indices_list,
                            latents_cpu.numpy(),
                            args.sample_shard_size,
                        )

                    # Explicit cleanup to mirror script B's stream-save behavior.
                    del latents_cpu
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        drain_pending(wait_for_all=True)

    dist.barrier()
    if rank == 0:
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to an ImageFolder directory with input images.")
    parser.add_argument("--sample-dir", type=str, default="samples", help="Directory to store encoded latents.")
    parser.add_argument("--per-proc-batch-size", type=int, default=8, help="Number of images processed per GPU step.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to encode (defaults to full dataset).")
    parser.add_argument("--image-size", type=int, default=256, help="Target crop size before feeding images to the model.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers per process.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Dataloader prefetch factor (when num_workers > 0).")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep dataloader workers alive across iterations (when num_workers > 0).")
    parser.add_argument("--global-seed", type=int, default=0, help="Base seed for RNG (adjusted per rank).")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Autocast precision mode.")
    parser.add_argument("--save-format", type=str, choices=["sample", "batch"], default="sample",
                        help="`sample`: one .npy per sample. `batch`: one .npz per processed batch (usually faster).")
    parser.add_argument("--sample-backend", type=str, choices=["npy", "torch"], default="npy",
                        help="Serializer for `--save-format sample`: NumPy `.npy` or PyTorch `.pt`.")
    parser.add_argument("--sample-torch-record", action=argparse.BooleanOptionalAction, default=True,
                        help="When `--save-format sample --sample-backend torch`, save one record dict per sample "
                             "(B/per-prompt style) instead of a raw tensor.")
    parser.add_argument("--sample-shard-size", type=int, default=0,
                        help="For `--save-format sample`, write into sharded subfolders every N samples (0 disables).")
    parser.add_argument("--latent-scale-stat-path", type=str, default=None,
                        help="Path to a stats `.pt` file used to derive the latent scaling factor when "
                             "--latent-scale-factor is omitted. Defaults to "
                             "stage_1.params.normalization_stat_path from --config.")
    parser.add_argument("--latent-scale-factor", type=float, default=None,
                        help="Multiply encoded latents by this factor before saving. "
                             "If omitted, derives it from --latent-scale-stat-path or "
                             "stage_1.params.normalization_stat_path. Raises an error if unresolved.")
    parser.add_argument("--save-dtype", type=str, choices=["fp32", "fp16"], default="fp32",
                        help="Data type used for serialized latents.")
    parser.add_argument("--save-workers", type=int, default=8, help="Thread workers used for asynchronous disk writes.")
    parser.add_argument("--max-inflight-saves", type=int, default=16,
                        help="Maximum number of queued save tasks before backpressure (higher uses more CPU RAM).")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable TF32 matmuls (Ampere+). Disable if deterministic results are required.")
    args = parser.parse_args()
    main(args)
