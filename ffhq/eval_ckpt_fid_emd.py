#!/usr/bin/env python3
"""Evaluate FFHQ latent checkpoints with per-class latent EMD and image FID.

Workflow:
1) Load a trained FFHQ checkpoint.
2) Sample N generated latents per class from the checkpoint model.
3) Sample N real latents per class from a provided NPZ split.
4) Compute per-class latent OT distance (EMD or Sinkhorn).
5) Decode real/fake latents with a frozen ALAE decoder and compute per-class FID.

Typical usage:
    python -m ffhq.eval_ckpt_fid_emd \
        --ckpt-path runs/ffhq/eps_1p0_sinkhorn/drift_ffhq_model.pt \
        --real-npz data/ffhq_latents_6class/train_latents_by_class.npz \
        --n-per-class 1000 \
        --alae-root /path/to/ALAE \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from .drift_ffhq import CLASS_NAMES, ConditionalDriftMLP, load_latents
    from .fid_score import calculate_fid_given_paths
except ImportError:  # pragma: no cover - direct script execution fallback
    from drift_ffhq import CLASS_NAMES, ConditionalDriftMLP, load_latents  # type: ignore
    from fid_score import calculate_fid_given_paths  # type: ignore


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Drift_FFHQ checkpoint with EMD+FID.")
    p.add_argument("--ckpt-path", required=True, help="Path to drift_ffhq_model*.pt")
    p.add_argument(
        "--real-npz",
        default="ffhq_latents_6class/train_latents_by_class.npz",
        help="NPZ that contains per-class real latents.",
    )
    p.add_argument("--n-per-class", type=int, default=1000, help="Samples per class for both real and fake.")
    p.add_argument(
        "--replacement",
        choices=["auto", "true", "false"],
        default="auto",
        help=(
            "Real-latent sampling with replacement. "
            "'auto': use replacement only when pool < n-per-class."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gen-batch", type=int, default=512, help="Batch size for fake latent generation.")
    p.add_argument("--decode-batch", type=int, default=8, help="Batch size for ALAE decoding.")
    p.add_argument("--decode-impl", choices=["batch", "loop"], default="batch")
    p.add_argument("--fid-batch", type=int, default=64, help="FID feature batch size.")
    p.add_argument("--save-size", type=int, default=1024, help="Saved image size for FID.")
    p.add_argument("--solver", choices=["emd", "sinkhorn"], default="emd", help="OT solver for latent distance.")
    p.add_argument("--metric", choices=["l2", "l2_sq"], default="l2_sq")
    p.add_argument("--sinkhorn-reg", type=float, default=0.05)
    p.add_argument("--ot-iters", type=int, default=200000)
    p.add_argument(
        "--alae-root",
        default=os.environ.get("ALAE_ROOT", ""),
        help=(
            "Path to the external ALAE repository containing "
            "alae_ffhq_inference.py, configs/ffhq.yaml, and training_artifacts/ffhq. "
            "Can also be set via ALAE_ROOT."
        ),
    )
    p.add_argument(
        "--alae-config",
        default=None,
        help="Optional override for the ALAE config path. Defaults to <alae-root>/configs/ffhq.yaml.",
    )
    p.add_argument(
        "--alae-artifacts",
        default=None,
        help="Optional override for the ALAE artifacts directory. Defaults to <alae-root>/training_artifacts/ffhq.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Default: <ckpt_dir>/eval_ckpt_fid_emd_n{N}_seed{seed}_"
            "<ckpt_stem>"
        ),
    )
    p.add_argument("--skip-fid", action="store_true", help="Skip image decode + FID.")
    p.add_argument("--skip-emd", action="store_true", help="Skip latent OT distance.")
    p.add_argument(
        "--reuse-images",
        action="store_true",
        help="If set, reuse existing decoded images when counts already match.",
    )
    return p.parse_args()


def _resolve_alae_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if not args.alae_root:
        raise ValueError(
            "ALAE decoder path is required for FID evaluation. "
            "Pass --alae-root /path/to/ALAE or set ALAE_ROOT."
        )
    alae_root = Path(args.alae_root).expanduser().resolve()
    alae_config = (
        Path(args.alae_config).expanduser().resolve()
        if args.alae_config
        else (alae_root / "configs" / "ffhq.yaml").resolve()
    )
    alae_artifacts = (
        Path(args.alae_artifacts).expanduser().resolve()
        if args.alae_artifacts
        else (alae_root / "training_artifacts" / "ffhq").resolve()
    )
    if not alae_root.exists():
        raise FileNotFoundError(f"Missing --alae-root: {alae_root}")
    if not alae_config.exists():
        raise FileNotFoundError(f"Missing ALAE config: {alae_config}")
    if not alae_artifacts.exists():
        raise FileNotFoundError(f"Missing ALAE artifacts dir: {alae_artifacts}")
    return alae_root, alae_config, alae_artifacts


def _count_images(d: Path) -> int:
    if not d.is_dir():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sum(1 for p in d.iterdir() if p.suffix.lower() in exts)


def _prepare_image_dir(d: Path, n_expected: int, *, reuse: bool) -> bool:
    """Return True if caller should decode; False when exact-cache-hit and reuse=True."""
    if d.exists():
        n_have = _count_images(d)
        if reuse and n_have == int(n_expected):
            return False
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return True


def _load_ckpt_model(ckpt_path: Path, device: torch.device) -> tuple[ConditionalDriftMLP, dict]:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = ConditionalDriftMLP(
        d_z=int(cfg["d_z"]),
        d_e=int(cfg["d_e"]),
        d_out=int(cfg["d_out"]),
        hidden=int(cfg["hidden"]),
        n_hidden=int(cfg["n_hidden"]),
        n_classes=int(cfg["n_classes"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _replacement_flag(mode: str, pool_size: int, n: int) -> bool:
    if mode == "true":
        return True
    if mode == "false":
        return False
    return int(pool_size) < int(n)


def _sample_real_latents(
    real_by_class: dict[str, torch.Tensor],
    *,
    n_per_class: int,
    seed: int,
    replacement_mode: str,
) -> tuple[dict[str, torch.Tensor], dict[str, bool]]:
    out: dict[str, torch.Tensor] = {}
    used_replacement: dict[str, bool] = {}
    for cid, cname in enumerate(CLASS_NAMES):
        pool = real_by_class[cname].float()
        n_pool = int(pool.shape[0])
        if n_pool <= 0:
            raise ValueError(f"Empty real pool for class={cname}")

        repl = _replacement_flag(replacement_mode, n_pool, int(n_per_class))
        if (not repl) and n_pool < int(n_per_class):
            raise ValueError(
                f"class={cname} has only {n_pool} samples, but n_per_class={n_per_class} and replacement=False."
            )
        rng = np.random.default_rng(int(seed) + cid)
        idx = rng.choice(n_pool, size=int(n_per_class), replace=repl)
        idx_t = torch.from_numpy(np.asarray(idx, dtype=np.int64))
        out[cname] = pool[idx_t].contiguous()
        used_replacement[cname] = bool(repl)
    return out, used_replacement


def _sample_fake_latents(
    model: ConditionalDriftMLP,
    *,
    n_per_class: int,
    d_z: int,
    gen_batch: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for cid, cname in enumerate(CLASS_NAMES):
        torch.manual_seed(int(seed) + 10_000 + cid)
        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, int(n_per_class), int(gen_batch)):
                bs = min(int(gen_batch), int(n_per_class) - i)
                noise = torch.randn(bs, int(d_z), device=device)
                labels = torch.full((bs,), int(cid), dtype=torch.long, device=device)
                parts.append(model(noise, labels).cpu())
        out[cname] = torch.cat(parts, dim=0).contiguous()
    return out


def _load_alae_model(
    device: torch.device,
    *,
    alae_root: Path,
    alae_config: Path,
    alae_artifacts: Path,
) -> torch.nn.Module:
    import sys

    if str(alae_root) not in sys.path:
        sys.path.insert(0, str(alae_root))
    from alae_ffhq_inference import load_model  # type: ignore

    if device.type == "cuda":
        # ALAE decoder samples noise tensors without explicit device.
        torch.set_default_device(device)
    cwd = os.getcwd()
    try:
        os.chdir(str(alae_root))
        model = load_model(
            str(alae_config),
            training_artifacts_dir=str(alae_artifacts),
        )
    finally:
        os.chdir(cwd)
    model = model.to(device)
    model.eval()
    return model


def _decode_latents(alae_model, latents: torch.Tensor, *, device: torch.device, impl: str) -> torch.Tensor:
    """Decode [B,512] latents -> [B,3,1024,1024] on CPU."""
    x = latents.to(device=device, dtype=torch.float32, non_blocking=True)
    x = x[:, None, :].repeat(1, alae_model.mapping_f.num_layers, 1)
    layer_count = 9

    with torch.no_grad():
        if impl == "batch":
            try:
                out = alae_model.decoder(x, layer_count - 1, 1, noise=True)
                return out.detach().cpu()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        decoded = []
        for i in range(x.shape[0]):
            decoded.append(alae_model.decoder(x[i : i + 1], layer_count - 1, 1, noise=True))
        return torch.cat(decoded, dim=0).detach().cpu()


def _decode_and_save(
    alae_model,
    latents: torch.Tensor,
    out_dir: Path,
    *,
    batch_size: int,
    save_size: int,
    label: str,
    device: torch.device,
    decode_impl: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(latents.shape[0])
    for i in range(0, n, int(batch_size)):
        b = latents[i : i + int(batch_size)].cpu()
        imgs = _decode_latents(alae_model, b, device=device, impl=decode_impl)
        arr = ((imgs.clamp(-1, 1) + 1.0) / 2.0 * 255.0).byte().permute(0, 2, 3, 1).numpy()
        for j, im in enumerate(arr):
            pil = Image.fromarray(im)
            if int(save_size) != 1024:
                pil = pil.resize((int(save_size), int(save_size)), Image.LANCZOS)
            pil.save(out_dir / f"{i + j:06d}.png", format="PNG")
        if (i // int(batch_size)) % 10 == 0:
            print(f"[{label}] {min(i + int(batch_size), n)}/{n}", flush=True)


def _ot_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    solver: str,
    metric: str,
    sinkhorn_reg: float,
    ot_iters: int,
) -> float:
    try:
        import ot
    except ImportError as exc:
        raise ImportError("POT is required. Install with `pip install pot`.") from exc

    if metric == "l2":
        ot_metric = "euclidean"
    elif metric == "l2_sq":
        ot_metric = "sqeuclidean"
    else:
        raise ValueError(f"Unsupported metric={metric}")

    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n_x = int(x.shape[0])
    n_y = int(y.shape[0])
    a = np.full(n_x, 1.0 / n_x, dtype=np.float64)
    b = np.full(n_y, 1.0 / n_y, dtype=np.float64)
    cost = ot.dist(x, y, metric=ot_metric)

    if solver == "emd":
        return float(ot.emd2(a, b, cost, numItermax=int(ot_iters)))
    if solver == "sinkhorn":
        return float(ot.sinkhorn2(a, b, cost, reg=float(sinkhorn_reg), numItermax=int(ot_iters), method="sinkhorn_log"))
    raise ValueError(f"Unsupported solver={solver}")


def main() -> None:
    args = _parse_args()
    ckpt_path = Path(args.ckpt_path).resolve()
    real_npz = Path(args.real_npz).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing --ckpt-path: {ckpt_path}")
    if not real_npz.exists():
        raise FileNotFoundError(f"Missing --real-npz: {real_npz}")
    if int(args.n_per_class) <= 0:
        raise ValueError("--n-per-class must be > 0")
    if int(args.gen_batch) <= 0 or int(args.decode_batch) <= 0 or int(args.fid_batch) <= 0:
        raise ValueError("--gen-batch/--decode-batch/--fid-batch must be > 0")
    if int(args.ot_iters) <= 0:
        raise ValueError("--ot-iters must be > 0")
    if args.solver == "sinkhorn" and float(args.sinkhorn_reg) <= 0:
        raise ValueError("--sinkhorn-reg must be > 0 for sinkhorn solver")

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    ckpt_stem = ckpt_path.stem
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = ckpt_path.parent / f"eval_ckpt_fid_emd_n{int(args.n_per_class)}_seed{int(args.seed)}_{ckpt_stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    real_img_base = out_dir / "real_images"
    fake_img_base = out_dir / "fake_images"

    print(f"[info] ckpt={ckpt_path}", flush=True)
    print(f"[info] real_npz={real_npz}", flush=True)
    print(f"[info] output_dir={out_dir}", flush=True)
    print(f"[info] n_per_class={int(args.n_per_class)}", flush=True)

    # Load model and data.
    model, ckpt = _load_ckpt_model(ckpt_path, device)
    real_by_class = load_latents(str(real_npz))

    # Sample latents.
    real_lat, real_repl = _sample_real_latents(
        real_by_class,
        n_per_class=int(args.n_per_class),
        seed=int(args.seed),
        replacement_mode=str(args.replacement),
    )
    fake_lat = _sample_fake_latents(
        model,
        n_per_class=int(args.n_per_class),
        d_z=int(ckpt["model_config"]["d_z"]),
        gen_batch=int(args.gen_batch),
        seed=int(args.seed),
        device=device,
    )

    # Save sampled latents for collaborator debugging.
    latent_npz = out_dir / "sampled_latents_by_class.npz"
    payload = {}
    for cname in CLASS_NAMES:
        payload[f"real_{cname}"] = real_lat[cname].cpu().numpy().astype(np.float32)
        payload[f"fake_{cname}"] = fake_lat[cname].cpu().numpy().astype(np.float32)
    np.savez_compressed(latent_npz, **payload)
    print(f"[save] sampled latents -> {latent_npz}", flush=True)

    # EMD/Sinkhorn in latent space.
    per_class = []
    for cid, cname in enumerate(CLASS_NAMES):
        rec = {
            "class": int(cid),
            "class_name": cname,
            "n_real": int(args.n_per_class),
            "n_fake": int(args.n_per_class),
            "real_sampling_with_replacement": bool(real_repl[cname]),
        }
        if not args.skip_emd:
            val = _ot_distance(
                fake_lat[cname].cpu().numpy(),
                real_lat[cname].cpu().numpy(),
                solver=str(args.solver),
                metric=str(args.metric),
                sinkhorn_reg=float(args.sinkhorn_reg),
                ot_iters=int(args.ot_iters),
            )
            rec["emd"] = float(val)
            print(f"[emd] {cname:<16s} {val:.6f}", flush=True)
        per_class.append(rec)

    # Decode + FID.
    if not args.skip_fid:
        alae_root, alae_config, alae_artifacts = _resolve_alae_paths(args)
        alae_model = _load_alae_model(
            device,
            alae_root=alae_root,
            alae_config=alae_config,
            alae_artifacts=alae_artifacts,
        )
        for cname in CLASS_NAMES:
            need_real = _prepare_image_dir(real_img_base / cname, int(args.n_per_class), reuse=bool(args.reuse_images))
            need_fake = _prepare_image_dir(fake_img_base / cname, int(args.n_per_class), reuse=bool(args.reuse_images))
            if need_real:
                _decode_and_save(
                    alae_model,
                    real_lat[cname],
                    real_img_base / cname,
                    batch_size=int(args.decode_batch),
                    save_size=int(args.save_size),
                    label=f"real:{cname}",
                    device=device,
                    decode_impl=str(args.decode_impl),
                )
            else:
                print(f"[cache] real:{cname} reuse", flush=True)
            if need_fake:
                _decode_and_save(
                    alae_model,
                    fake_lat[cname],
                    fake_img_base / cname,
                    batch_size=int(args.decode_batch),
                    save_size=int(args.save_size),
                    label=f"fake:{cname}",
                    device=device,
                    decode_impl=str(args.decode_impl),
                )
            else:
                print(f"[cache] fake:{cname} reuse", flush=True)

        use_cuda = device.type == "cuda"
        for rec in per_class:
            cname = rec["class_name"]
            fid = float(
                calculate_fid_given_paths(
                    [str(real_img_base / cname), str(fake_img_base / cname)],
                    batch_size=int(args.fid_batch),
                    cuda=bool(use_cuda),
                    dims=2048,
                )
            )
            rec["fid"] = fid
            print(f"[fid] {cname:<16s} {fid:.4f}", flush=True)

    emd_vals = [r["emd"] for r in per_class if "emd" in r]
    fid_vals = [r["fid"] for r in per_class if "fid" in r]
    out = {
        "ckpt_path": str(ckpt_path),
        "real_npz": str(real_npz),
        "class_names": CLASS_NAMES,
        "n_per_class": int(args.n_per_class),
        "replacement_mode": str(args.replacement),
        "seed": int(args.seed),
        "device": str(device),
        "solver": str(args.solver),
        "metric": str(args.metric),
        "sinkhorn_reg": float(args.sinkhorn_reg) if args.solver == "sinkhorn" else None,
        "ot_iters": int(args.ot_iters),
        "save_size": int(args.save_size),
        "fid_impl": "ffhq.fid_score.calculate_fid_given_paths",
        "per_class": per_class,
        "mean_emd": float(np.mean(emd_vals)) if emd_vals else None,
        "mean_fid": float(np.mean(fid_vals)) if fid_vals else None,
        "sampled_latents_npz": str(latent_npz),
        "real_images_dir": str(real_img_base) if not args.skip_fid else None,
        "fake_images_dir": str(fake_img_base) if not args.skip_fid else None,
    }
    out_path = out_dir / "metrics_fid_emd.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[save] metrics -> {out_path}", flush=True)
    if out["mean_emd"] is not None:
        print(f"[summary] mean EMD: {out['mean_emd']:.6f}", flush=True)
    if out["mean_fid"] is not None:
        print(f"[summary] mean FID: {out['mean_fid']:.4f}", flush=True)


if __name__ == "__main__":
    main()
