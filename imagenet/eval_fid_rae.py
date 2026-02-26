"""
Generate samples from a drifting generator checkpoint trained on RAE latents and compute FID.

This script mirrors imagenet/eval_fid.py, but:
- samples latent noise with checkpoint-native shape (C,H,W) from dit_cfg
- decodes generated latents with an RAE stage-1 model
- optionally copies the first N generated images to a visualization folder
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from imagenet.models.dit_b2 import DiTB2Config, DiTLatentB2
from imagenet.models.ema import EMA
from imagenet.utils.dist import DistInfo, barrier, init_distributed, is_main_process


def _parse_floats(csv: str) -> list[float]:
    out: list[float] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ImageNet256 samples from RAE-latent drifting checkpoint and compute FID.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to drifting generator checkpoint (.pt).")
    p.add_argument("--rae-config", type=str, required=True, help="Path to RAE config yaml (must contain stage_1).")
    p.add_argument(
        "--latent-scale-factor",
        type=float,
        default=1.0,
        help="If training latents were scaled before saving, divide generated latents by this factor before RAE decode.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-ema", action="store_true", help="Use EMA weights from checkpoint.")

    p.add_argument("--num-gen", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--omegas", type=str, default="1.0", help="Comma-separated omega values to evaluate (ignored if --omega-num > 0).")
    p.add_argument("--omega-min", type=float, default=1.0, help="Omega sweep min (used when --omega-num > 0).")
    p.add_argument("--omega-max", type=float, default=4.0, help="Omega sweep max (used when --omega-num > 0).")
    p.add_argument("--omega-num", type=int, default=0, help="If > 0, evaluate a linspace of this many omegas on [omega-min, omega-max].")

    p.add_argument("--out-dir", type=str, default="", help="Output directory (default: alongside ckpt).")
    p.add_argument("--real-dir", type=str, default="/home/public/imagenet/val", help="Real images root for FID.")
    p.add_argument("--real-stats", type=str, default="", help="Cache file for real mu/sigma (.npz).")

    p.add_argument("--viz-first-n", type=int, default=0, help="If > 0, copy first N generated images for each omega to a viz folder.")
    p.add_argument("--viz-dir", type=str, default="", help="Optional root folder for visualization images (default: under each omega output dir).")

    # clean-fid
    p.add_argument("--fid-mode", type=str, choices=["clean", "legacy_pytorch", "legacy_tensorflow"], default="clean")
    p.add_argument("--fid-batch-size", type=int, default=128)
    p.add_argument("--fid-workers", type=int, default=12)
    return p.parse_args()


def _rank_range(total: int, rank: int, world: int) -> tuple[int, int]:
    base = total // world
    rem = total % world
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


@torch.no_grad()
def _save_images_01(images: torch.Tensor, out_dir: str, start_idx: int) -> None:
    # images: [B,3,H,W] in [0,1]
    img = (images.clamp(0, 1) * 255.0).to(torch.uint8)
    img = img.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    for i in range(img.shape[0]):
        path = os.path.join(out_dir, f"{start_idx + i:06d}.png")
        Image.fromarray(img[i]).save(path)


def _compute_folder_stats(fdir: str, *, mode: str, batch_size: int, num_workers: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    from cleanfid.fid import get_folder_features
    from cleanfid.features import build_feature_extractor

    feat_model = build_feature_extractor(mode, device=device, use_dataparallel=True)
    feats = get_folder_features(
        fdir,
        model=feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        verbose=True,
    )
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def _fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    from cleanfid.fid import frechet_distance

    return float(frechet_distance(mu1, sigma1, mu2, sigma2))


def _load_generator(ckpt_path: str, device: torch.device, *, use_ema: bool) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt.get("dit_cfg"), dict):
        cfg_dict = dict(ckpt["dit_cfg"])
        # Backward compat: older checkpoints used constant "register" tokens and did not store ctx_mode.
        cfg_dict.setdefault("ctx_mode", "register")
        dit_cfg = DiTB2Config(**cfg_dict)
    else:
        dit_cfg = DiTB2Config()
    gen = DiTLatentB2(dit_cfg)
    gen.load_state_dict(ckpt["gen"], strict=True)
    if use_ema and "ema" in ckpt:
        ema_tmp = EMA(gen, decay=float(ckpt["ema"].get("decay", 0.999)))
        ema_tmp.load_state_dict(ckpt["ema"])
        ema_tmp.copy_to(gen)
    gen.to(device)
    gen.eval()
    for p in gen.parameters():
        p.requires_grad_(False)
    meta = {"dit_cfg": asdict(dit_cfg)}
    return gen, meta


def _resolve_existing_path(path: str, *, config_dir: str, rae_root: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(expanded):
        return expanded

    candidates = [
        expanded,
        os.path.join(config_dir, expanded),
        os.path.join(rae_root, expanded),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return path


def _load_rae_model(rae_config_path: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[1]
    rae_root = repo_root / "rae"
    rae_src = rae_root / "src"
    if not rae_src.is_dir():
        raise FileNotFoundError(f"RAE source directory not found: {rae_src}")

    rae_src_str = str(rae_src.resolve())
    if rae_src_str not in sys.path:
        sys.path.insert(0, rae_src_str)

    from omegaconf import OmegaConf
    from utils.model_utils import instantiate_from_config

    cfg = OmegaConf.load(rae_config_path)
    stage1_cfg = cfg.get("stage_1")
    if stage1_cfg is None:
        raise ValueError(f"Config has no stage_1 section: {rae_config_path}")

    stage1_dict = OmegaConf.to_container(stage1_cfg, resolve=True)
    if not isinstance(stage1_dict, dict):
        raise TypeError(f"stage_1 config must resolve to a dict, got: {type(stage1_dict).__name__}")

    config_dir = str(Path(rae_config_path).resolve().parent)
    params = dict(stage1_dict.get("params", {}))
    for key in ("decoder_config_path", "pretrained_decoder_path", "normalization_stat_path"):
        val = params.get(key)
        if isinstance(val, str) and val:
            params[key] = _resolve_existing_path(val, config_dir=config_dir, rae_root=str(rae_root))
    stage1_dict["params"] = params

    ckpt_path = stage1_dict.get("ckpt")
    if isinstance(ckpt_path, str) and ckpt_path:
        stage1_dict["ckpt"] = _resolve_existing_path(ckpt_path, config_dir=config_dir, rae_root=str(rae_root))

    rae = instantiate_from_config(stage1_dict).to(device)
    rae.eval()
    for p in rae.parameters():
        p.requires_grad_(False)

    meta = {
        "rae_config_path": str(Path(rae_config_path).resolve()),
        "stage1_target": stage1_dict.get("target"),
        "stage1_params": params,
        "stage1_ckpt": stage1_dict.get("ckpt", None),
    }
    return rae, meta


def _copy_first_n_images(src_dir: str, dst_dir: str, n: int, num_gen: int) -> int:
    keep = max(0, min(int(n), int(num_gen)))
    if keep == 0:
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    for idx in range(keep):
        src = os.path.join(src_dir, f"{idx:06d}.png")
        if not os.path.exists(src):
            continue
        dst = os.path.join(dst_dir, f"{idx:06d}.png")
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main() -> None:
    args = _parse_args()
    if float(args.latent_scale_factor) <= 0:
        raise ValueError("--latent-scale-factor must be > 0")

    dist_info: DistInfo = init_distributed(device=args.device)
    torch.manual_seed(args.seed + dist_info.rank)

    if int(args.omega_num) > 0:
        omegas = torch.linspace(float(args.omega_min), float(args.omega_max), int(args.omega_num)).tolist()
    else:
        omegas = _parse_floats(args.omegas)
    if not omegas:
        raise ValueError("--omegas must be non-empty")

    ckpt_path = args.ckpt
    out_root = args.out_dir or str(Path(ckpt_path).resolve().parent / "eval_fid_rae")
    if is_main_process():
        os.makedirs(out_root, exist_ok=True)

    gen, gen_meta = _load_generator(ckpt_path, dist_info.device, use_ema=bool(args.use_ema))
    rae, rae_meta = _load_rae_model(args.rae_config, dist_info.device)

    # Real stats cache.
    real_stats_path = args.real_stats or os.path.join(out_root, "real_stats_imagenet_val_256.npz")
    real_mu: np.ndarray
    real_sigma: np.ndarray

    if is_main_process():
        if os.path.exists(real_stats_path):
            stats = np.load(real_stats_path)
            real_mu = stats["mu"]
            real_sigma = stats["sigma"]
        else:
            print(f"Computing real stats for: {args.real_dir}")
            real_mu, real_sigma = _compute_folder_stats(
                args.real_dir,
                mode=str(args.fid_mode),
                batch_size=int(args.fid_batch_size),
                num_workers=int(args.fid_workers),
                device=dist_info.device,
            )
            Path(real_stats_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(real_stats_path, mu=real_mu, sigma=real_sigma)
            print(f"Saved real stats: {real_stats_path}")
    barrier()

    # Broadcast real stats by reloading (simple and robust).
    if not is_main_process():
        stats = np.load(real_stats_path)
        real_mu = stats["mu"]
        real_sigma = stats["sigma"]

    start, end = _rank_range(int(args.num_gen), dist_info.rank, dist_info.world_size)
    n_rank = end - start
    if is_main_process():
        print(f"Generating {args.num_gen} images total; rank0 range [{start},{end})")
        print(f"Generator latent shape: [{gen.cfg.in_ch},{gen.cfg.input_size},{gen.cfg.input_size}]")

    sweep: list[dict[str, Any]] = []
    for omega in omegas:
        tag = str(omega).replace(".", "p")
        out_dir = os.path.join(out_root, f"gen_omega_{tag}")
        if is_main_process():
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ckpt": ckpt_path,
                        "use_ema": bool(args.use_ema),
                        "omega": float(omega),
                        "num_gen": int(args.num_gen),
                        "batch_size": int(args.batch_size),
                        "real_dir": args.real_dir,
                        "real_stats": real_stats_path,
                        "fid": {"mode": args.fid_mode, "batch_size": int(args.fid_batch_size), "workers": int(args.fid_workers)},
                        "latent_scale_factor": float(args.latent_scale_factor),
                        "viz_first_n": int(args.viz_first_n),
                        "viz_dir_arg": args.viz_dir,
                        **gen_meta,
                        **rae_meta,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
        barrier()

        # Generate and save images for this rank.
        bs = int(args.batch_size)
        pbar = tqdm(range(0, n_rank, bs), disable=not is_main_process(), desc=f"gen ω={omega}")
        for off in pbar:
            b = min(bs, n_rank - off)
            cls = torch.randint(0, gen.cfg.num_classes, (b,), device=dist_info.device, dtype=torch.long)
            z = torch.randn(b, int(gen.cfg.in_ch), int(gen.cfg.input_size), int(gen.cfg.input_size), device=dist_info.device)
            omega_t = torch.full((b,), float(omega), device=dist_info.device, dtype=torch.float32)

            lat = gen(z, cls, omega_t)
            if float(args.latent_scale_factor) != 1.0:
                lat = lat / float(args.latent_scale_factor)
            imgs = rae.decode(lat).clamp(0, 1)
            _save_images_01(imgs, out_dir=out_dir, start_idx=start + off)
        barrier()

        viz_copied = 0
        viz_dir_used = ""
        if is_main_process() and int(args.viz_first_n) > 0:
            if args.viz_dir:
                viz_dir_used = os.path.join(args.viz_dir, f"omega_{tag}")
            else:
                viz_dir_used = os.path.join(out_dir, f"viz_first_{int(args.viz_first_n)}")
            viz_copied = _copy_first_n_images(out_dir, viz_dir_used, int(args.viz_first_n), int(args.num_gen))

        # FID on rank0.
        if is_main_process():
            mu_g, sig_g = _compute_folder_stats(
                out_dir,
                mode=str(args.fid_mode),
                batch_size=int(args.fid_batch_size),
                num_workers=int(args.fid_workers),
                device=dist_info.device,
            )
            fid = _fid(mu_g, sig_g, real_mu, real_sigma)
            print(f"FID (ω={omega}): {fid:.4f}")
            result = {"omega": float(omega), "fid": fid}
            if viz_dir_used:
                result["viz_dir"] = viz_dir_used
                result["viz_copied"] = int(viz_copied)
            with open(os.path.join(out_dir, "fid.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, sort_keys=True)
            sweep_row = {"omega": float(omega), "fid": float(fid), "dir": out_dir}
            if viz_dir_used:
                sweep_row["viz_dir"] = viz_dir_used
                sweep_row["viz_copied"] = int(viz_copied)
            sweep.append(sweep_row)
        barrier()

    if is_main_process() and sweep:
        best = min(sweep, key=lambda r: r["fid"])
        summary = {
            "ckpt": ckpt_path,
            "use_ema": bool(args.use_ema),
            "num_gen": int(args.num_gen),
            "batch_size": int(args.batch_size),
            "fid": {"mode": str(args.fid_mode), "batch_size": int(args.fid_batch_size), "workers": int(args.fid_workers)},
            "real_dir": str(args.real_dir),
            "real_stats": str(real_stats_path),
            "latent_scale_factor": float(args.latent_scale_factor),
            "omegas": [r["omega"] for r in sweep],
            "results": sweep,
            "best": best,
        }
        with open(os.path.join(out_root, "fid_sweep.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        with open(os.path.join(out_root, "fid_sweep.csv"), "w", encoding="utf-8") as f:
            f.write("omega,fid,dir\n")
            for r in sweep:
                f.write(f"{r['omega']},{r['fid']},{r['dir']}\n")


if __name__ == "__main__":
    main()
