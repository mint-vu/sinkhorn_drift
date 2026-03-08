"""Train MLP drifting generator in MNIST 6-dim latent space.

Reuses the core drifting loss from core.drifting_loss.
Simplified vs ImageNet: no DDP, no AMP, no sample queue, nc=10 (all classes).

Usage:
    python -m mnist.train_drifting \
        --coupling sinkhorn --sinkhorn-marginal weighted_cols \
        --drift-form split --steps 5000 \
        --ae-ckpt runs/mnist_ae/<run>/ae_final.pt \
        --device cuda:2 --run-name mnist_sinkhorn
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from core.drifting_loss import (
    FeatureSet,
    drifting_loss_for_feature_set,
    sample_power_law_omega,
)
from core.models.ema import EMA
from mnist.models import ConvAE, MLPGenerator


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MNIST MLP drifting generator.")

    # Data / model paths.
    p.add_argument("--ae-ckpt", type=str, required=True, help="Path to ae_final.pt")
    p.add_argument("--data-root", type=str, default="./data")

    # Generator architecture.
    p.add_argument("--noise-dim", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=3)

    # Drifting loss (paper Table 8 defaults).
    p.add_argument("--coupling", type=str, default="sinkhorn",
                    choices=["row", "partial_two_sided", "sinkhorn"])
    p.add_argument("--drift-form", type=str, default="split",
                    choices=["alg2_joint", "split"])
    p.add_argument("--sinkhorn-iters", type=int, default=20)
    p.add_argument("--sinkhorn-marginal", type=str, default="weighted_cols",
                    choices=["none", "weighted_cols", "post_guidance"])
    p.add_argument("--sinkhorn-agg-kernel", action="store_true",
                    help="Aggregate Gibbs kernels across rho values before running Sinkhorn once.")
    p.add_argument("--drift-tau-scale", action="store_true",
                    help="Multiply drift by 2/tau (L2sq) or 1/tau (L2), matching exact kernel gradient coefficient. "
                         "With theta normalization enabled this is a scalar and has no effect on training.")
    p.add_argument("--drift-unit-vec", action="store_true",
                    help="For L2 distance: use unit displacement vectors (y-x)/||y-x|| instead of (y-x). "
                         "Matches exact L2-kernel gradient direction.")
    p.add_argument("--temps", type=float, nargs="+", default=[0.02, 0.05, 0.2])
    p.add_argument("--omega-min", type=float, default=1.0)
    p.add_argument("--omega-max", type=float, default=4.0)
    p.add_argument("--omega-exponent", type=float, default=3.0)
    p.add_argument("--dist-metric", type=str, default="l2",
                    choices=["l2", "l2_sq"])

    # Batch sizes per class (paper: nneg=npos=64, nuncond=16).
    p.add_argument("--nneg", type=int, default=64)
    p.add_argument("--npos", type=int, default=64)
    p.add_argument("--nuncond", type=int, default=16)

    # Training (paper Table 8).
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=750)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--ema-decay", type=float, default=0.999)

    # Logging / saving.
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--sample-omega", type=float, default=2.0)
    p.add_argument("--sample-nrow", type=int, default=10)

    # Misc.
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-root", type=str, default="runs/mnist_drift")
    p.add_argument("--run-name", type=str, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data: load precomputed latents (from encode_latents.py)
# ---------------------------------------------------------------------------

class LatentDataset:
    """In-memory dataset of latents grouped by class."""

    def __init__(self, latents: np.ndarray, labels: np.ndarray, device: torch.device):
        self.device = device
        self.num_classes = int(labels.max()) + 1
        self.class_indices: list[torch.Tensor] = []
        self.latents = torch.from_numpy(latents).float().to(device)
        for c in range(self.num_classes):
            idx = torch.from_numpy(np.where(labels == c)[0]).long().to(device)
            self.class_indices.append(idx)

    def sample_class(self, c: int, n: int) -> torch.Tensor:
        """Sample n latents from class c (with replacement)."""
        idx = self.class_indices[c]
        sel = idx[torch.randint(len(idx), (n,), device=self.device)]
        return self.latents[sel]

    def sample_random(self, n: int) -> torch.Tensor:
        """Sample n latents from any class (for unconditional)."""
        sel = torch.randint(len(self.latents), (n,), device=self.device)
        return self.latents[sel]


# ---------------------------------------------------------------------------
# LR schedule: linear warmup, constant after.
# ---------------------------------------------------------------------------

def set_lr(optimizer: torch.optim.Optimizer, step: int, warmup: int, base_lr: float) -> float:
    if warmup > 0 and step < warmup:
        lr = base_lr * (step + 1) / warmup
    else:
        lr = base_lr
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Online sampling for visualization.
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_grid(
    gen: MLPGenerator,
    ae: ConvAE,
    omega: float,
    nrow: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate 10-class grid using the generator + AE decoder."""
    all_imgs = []
    for c in range(gen.num_classes):
        noise = torch.randn(nrow, gen.noise_dim, device=device)
        labels = torch.full((nrow,), c, dtype=torch.long, device=device)
        omegas = torch.full((nrow,), omega, device=device)
        z = gen(noise, labels, omegas)
        imgs = ae.decode(z)
        all_imgs.append(imgs)
    return torch.cat(all_imgs, dim=0)


# ---------------------------------------------------------------------------
# Main training loop.
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # --- Load AE ---
    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu", weights_only=True)
    latent_dim = ae_ckpt["latent_dim"]
    ae = ConvAE(latent_dim=latent_dim).to(device)
    ae.load_state_dict(ae_ckpt["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    print(f"AE: latent_dim={latent_dim}, epoch={ae_ckpt['epoch']}, test_loss={ae_ckpt['test_loss']:.6f}")

    # --- Load precomputed latents ---
    ae_dir = os.path.dirname(args.ae_ckpt)
    train_z = np.load(os.path.join(ae_dir, "train_latents.npy"))
    train_y = np.load(os.path.join(ae_dir, "train_labels.npy"))
    dataset = LatentDataset(train_z, train_y, device)
    print(f"Loaded {len(train_z)} train latents, {dataset.num_classes} classes")

    # --- Generator ---
    gen = MLPGenerator(
        num_classes=dataset.num_classes,
        noise_dim=args.noise_dim,
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    ema = EMA(gen, decay=args.ema_decay)
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"MLPGenerator: params={n_params:,}")

    # --- Optimizer ---
    opt = torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # --- Run directory ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    coupling_tag = args.coupling
    if coupling_tag == "sinkhorn" and args.sinkhorn_marginal != "none":
        coupling_tag += f"_{args.sinkhorn_marginal}"
    name = args.run_name or f"{ts}_{coupling_tag}_{args.drift_form}"
    run_dir = os.path.join(args.run_root, name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "logs.jsonl")

    # Save config.
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Run dir: {run_dir}")

    # --- Training ---
    nc = dataset.num_classes  # process all 10 classes per step
    gen.train()

    for step in range(1, args.steps + 1):
        lr = set_lr(opt, step - 1, args.warmup_steps, args.lr)
        opt.zero_grad()

        total_loss = 0.0
        stats_accum: dict[str, float] = {}

        # Sample omega (shared across classes in this step).
        omega = sample_power_law_omega(
            1, omega_min=args.omega_min, omega_max=args.omega_max,
            exponent=args.omega_exponent, device=device,
        ).squeeze(0)

        for c in range(nc):
            # Noise -> generator -> latent.
            noise = torch.randn(args.nneg, args.noise_dim, device=device)
            c_labels = torch.full((args.nneg,), c, dtype=torch.long, device=device)
            omega_rep = omega.expand(args.nneg)
            x_lat = gen(noise, c_labels, omega_rep)  # [nneg, latent_dim]

            # Positive: real latents from class c.
            pos_lat = dataset.sample_class(c, args.npos)  # [npos, latent_dim]

            # Unconditional: random latents from any class.
            if args.nuncond > 0:
                unc_lat = dataset.sample_random(args.nuncond)  # [nuncond, latent_dim]
            else:
                unc_lat = torch.empty(0, latent_dim, device=device)

            # Shape for drifting_loss: [N, L=1, C=latent_dim].
            x_feat = x_lat.unsqueeze(1)
            y_pos_feat = pos_lat.unsqueeze(1)
            y_unc_feat = unc_lat.unsqueeze(1)

            stats: dict[str, float] = {}
            # mask_self_neg: False for sinkhorn (matches ALAE), True for others.
            mask_self_neg = (args.coupling != "sinkhorn")
            loss = drifting_loss_for_feature_set(
                x_feat, y_pos_feat, y_unc_feat,
                omega=omega,
                temps=args.temps,
                vanilla=False,  # use S_j normalization so kernel values are non-trivial
                drift_form=args.drift_form,
                coupling=args.coupling,
                sinkhorn_iters=args.sinkhorn_iters,
                sinkhorn_marginal=args.sinkhorn_marginal,
                sinkhorn_agg_kernel=bool(args.sinkhorn_agg_kernel),
                mask_self_neg=mask_self_neg,
                dist_metric=args.dist_metric,
                normalize_drift_theta=True,
                drift_tau_scale=bool(args.drift_tau_scale),
                drift_unit_vec=bool(args.drift_unit_vec),
                stats=stats,
            )

            # Accumulate (average over classes).
            loss_scaled = loss / nc
            loss_scaled.backward()
            total_loss += loss.item()

            for k, v in stats.items():
                stats_accum[k] = stats_accum.get(k, 0.0) + v / nc

        # Gradient clipping + optimizer step.
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(gen.parameters(), args.grad_clip)
        opt.step()
        ema.update(gen)

        # --- Logging ---
        if step % args.log_every == 0 or step == 1:
            avg_loss = total_loss / nc
            rec = {"step": step, "loss": round(avg_loss, 6), "lr": round(lr, 8),
                   "omega": round(omega.item(), 4)}
            rec.update({k: round(v, 6) for k, v in stats_accum.items()})
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{step}/{args.steps}] loss={avg_loss:.6f} lr={lr:.2e} omega={omega.item():.2f}")

        # --- Online sampling ---
        if step % args.sample_every == 0 or step == args.steps:
            # Use EMA weights for sampling.
            gen.eval()
            orig_state = {n: p.clone() for n, p in gen.named_parameters() if p.requires_grad}
            ema.copy_to(gen)

            grid = sample_grid(gen, ae, omega=args.sample_omega, nrow=args.sample_nrow, device=device)
            fig_path = os.path.join(run_dir, f"sample_step{step}.png")
            save_image(grid, fig_path, nrow=args.sample_nrow, padding=1)
            print(f"  -> saved sample grid: {fig_path}")

            # Restore training weights.
            for n, p in gen.named_parameters():
                if n in orig_state:
                    p.data.copy_(orig_state[n])
            gen.train()

        # --- Checkpoint ---
        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                "step": step,
                "model": gen.state_dict(),
                "ema_state": {n: ema.shadow[n].clone() for n in ema.shadow},
                "optimizer": opt.state_dict(),
                "gen_config": {
                    "num_classes": dataset.num_classes,
                    "noise_dim": args.noise_dim,
                    "latent_dim": latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                },
                "args": vars(args),
            }
            tag = f"ckpt_step{step}.pt" if step < args.steps else "ckpt_final.pt"
            ckpt_path = os.path.join(run_dir, tag)
            torch.save(ckpt, ckpt_path)
            if step == args.steps:
                print(f"Saved final: {ckpt_path}")

    print("Done.")


if __name__ == "__main__":
    main()
