"""Train ConvAE on MNIST: 28x28 -> latent_dim -> 28x28.

Usage:
    python -m mnist.train_ae --epochs 50 --latent-dim 6 --device cuda:2
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from mnist.models import ConvAE


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MNIST ConvAE.")
    p.add_argument("--latent-dim", type=int, default=6)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--run-root", type=str, default="runs/mnist_ae")
    p.add_argument("--save-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Create run directory.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.run_root, f"{ts}_latent{args.latent_dim}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "logs.jsonl")

    # Data.
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=args.data_root, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(root=args.data_root, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model + optimizer.
    ae = ConvAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"ConvAE: latent_dim={args.latent_dim}, params={n_params:,}")
    print(f"Run dir: {run_dir}")

    for epoch in range(1, args.epochs + 1):
        # Train.
        ae.train()
        train_loss_sum = 0.0
        train_count = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            recon, _ = ae(imgs)
            loss = F.mse_loss(recon, imgs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * imgs.shape[0]
            train_count += imgs.shape[0]
        train_loss = train_loss_sum / train_count

        # Eval.
        ae.eval()
        test_loss_sum = 0.0
        test_count = 0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                recon, _ = ae(imgs)
                loss = F.mse_loss(recon, imgs)
                test_loss_sum += loss.item() * imgs.shape[0]
                test_count += imgs.shape[0]
        test_loss = test_loss_sum / test_count

        rec = {"epoch": epoch, "train_loss": round(train_loss, 6), "test_loss": round(test_loss, 6)}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.6f}  test_loss={test_loss:.6f}")

        # Save checkpoint + reconstruction comparison.
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "model": ae.state_dict(),
                "latent_dim": args.latent_dim,
                "epoch": epoch,
                "test_loss": test_loss,
            }
            ckpt_path = os.path.join(run_dir, f"ae_epoch{epoch}.pt")
            torch.save(ckpt, ckpt_path)
            # Save final as ae_final.pt for convenience.
            if epoch == args.epochs:
                final_path = os.path.join(run_dir, "ae_final.pt")
                torch.save(ckpt, final_path)
                print(f"Saved: {final_path}")

            # Reconstruction figure: top row = original, bottom row = reconstructed.
            with torch.no_grad():
                sample_imgs = next(iter(test_loader))[0][:16].to(device)
                sample_recon, _ = ae(sample_imgs)
                comparison = torch.cat([sample_imgs, sample_recon], dim=0)
                fig_path = os.path.join(run_dir, f"recon_epoch{epoch}.png")
                save_image(comparison, fig_path, nrow=16)

    print("Done.")


if __name__ == "__main__":
    main()
