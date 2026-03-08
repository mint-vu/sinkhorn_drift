"""Encode MNIST train/test sets into 6-dim latents using a trained ConvAE.

Usage:
    python -m mnist.encode_latents \
        --ae-ckpt runs/mnist_ae/<run>/ae_final.pt --device cuda:2
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist.models import ConvAE


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode MNIST -> latents.")
    p.add_argument("--ae-ckpt", type=str, required=True, help="Path to ae_final.pt")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data-root", type=str, default="./data")
    return p.parse_args()


@torch.no_grad()
def encode_split(ae: ConvAE, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Encode a full split and return (latents [N, D], labels [N])."""
    all_z, all_y = [], []
    for imgs, labels in loader:
        z = ae.encode(imgs.to(device))
        all_z.append(z.cpu().numpy())
        all_y.append(labels.numpy())
    return np.concatenate(all_z, axis=0), np.concatenate(all_y, axis=0)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    # Load AE.
    ckpt = torch.load(args.ae_ckpt, map_location="cpu", weights_only=True)
    latent_dim = ckpt["latent_dim"]
    ae = ConvAE(latent_dim=latent_dim).to(device)
    ae.load_state_dict(ckpt["model"])
    ae.eval()
    print(f"Loaded AE: latent_dim={latent_dim}, epoch={ckpt['epoch']}, test_loss={ckpt['test_loss']:.6f}")

    # Data.
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=args.data_root, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(root=args.data_root, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Encode.
    train_z, train_y = encode_split(ae, train_loader, device)
    test_z, test_y = encode_split(ae, test_loader, device)
    print(f"Train: {train_z.shape}, Test: {test_z.shape}")
    print(f"Latent stats: mean={train_z.mean():.4f}, std={train_z.std():.4f}, "
          f"min={train_z.min():.4f}, max={train_z.max():.4f}")

    # Save next to the AE checkpoint.
    out_dir = os.path.dirname(args.ae_ckpt)
    for name, z, y in [("train", train_z, train_y), ("test", test_z, test_y)]:
        np.save(os.path.join(out_dir, f"{name}_latents.npy"), z)
        np.save(os.path.join(out_dir, f"{name}_labels.npy"), y)
    print(f"Saved latents to {out_dir}/")


if __name__ == "__main__":
    main()
