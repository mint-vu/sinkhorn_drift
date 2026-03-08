"""Evaluate MNIST drifting generators: class-conditional generation accuracy.

For each class c in {0,...,9}:
    1. Generate N samples using MLPGenerator (EMA weights)
    2. Decode to image space via ConvAE
    3. Classify with a pretrained MNIST CNN classifier
    4. Accuracy = fraction of samples predicted as class c

The classifier is trained on the MNIST train set on first run and cached at
runs/mnist_clf/mnist_clf.pt for reuse.

Usage:
    python -m mnist.eval_acc \\
        --gen-ckpt runs/mnist_drift/mnist_sinkhorn/ckpt_final.pt \\
        --ae-ckpt  runs/mnist_ae/<run>/ae_final.pt \\
        --device cuda:0

    # Multiple runs at once:
    python -m mnist.eval_acc \\
        --gen-ckpt runs/mnist_drift/mnist_sinkhorn/ckpt_final.pt \\
                   runs/mnist_drift/mnist_baseline/ckpt_final.pt \\
        --ae-ckpt  runs/mnist_ae/<run>/ae_final.pt \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist.models import ConvAE, MLPGenerator


# ---------------------------------------------------------------------------
# Simple CNN classifier (trains in ~30s on a GPU)
# ---------------------------------------------------------------------------

class MNISTClassifier(nn.Module):
    """Small CNN: ~99% accuracy on MNIST test set."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 14x14
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 7x7
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_classifier(
    save_path: str,
    *,
    data_root: str = "./data",
    device: torch.device,
    epochs: int = 5,
) -> MNISTClassifier:
    """Train a small CNN on MNIST and save to save_path."""
    print(f"Training MNIST classifier (epochs={epochs})...")
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=data_root, train=True, download=False, transform=tf)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)

    clf = MNISTClassifier().to(device)
    opt = optim.Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    clf.train()
    for epoch in range(epochs):
        correct, total = 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = clf(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        print(f"  Epoch {epoch+1}/{epochs}  train_acc={correct/total:.4f}")

    # Quick test accuracy check.
    test_ds = datasets.MNIST(root=data_root, train=False, download=False, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)
    clf.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (clf(imgs).argmax(1) == labels).sum().item()
            total += len(labels)
    print(f"  Test accuracy: {correct/total:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(clf.state_dict(), save_path)
    print(f"  Saved classifier to {save_path}")
    return clf


def load_or_train_classifier(
    clf_path: str,
    *,
    data_root: str,
    device: torch.device,
) -> MNISTClassifier:
    clf = MNISTClassifier().to(device)
    if os.path.exists(clf_path):
        clf.load_state_dict(torch.load(clf_path, map_location=device, weights_only=True))
        print(f"Loaded classifier from {clf_path}")
    else:
        clf = train_classifier(clf_path, data_root=data_root, device=device)
    clf.eval()
    return clf


# ---------------------------------------------------------------------------
# Generator loading (shared with eval_emd)
# ---------------------------------------------------------------------------

def load_generator(ckpt_path: str, device: torch.device) -> MLPGenerator:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt["gen_config"]
    gen = MLPGenerator(
        num_classes=cfg["num_classes"],
        noise_dim=cfg["noise_dim"],
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    ).to(device)
    gen.load_state_dict(ckpt["ema_state"] if "ema_state" in ckpt else ckpt["model"])
    gen.eval()
    return gen


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_one(
    gen_ckpt: str,
    ae: ConvAE,
    clf: MNISTClassifier,
    *,
    omega: float,
    n_per_class: int,
    device: torch.device,
) -> dict:
    gen = load_generator(gen_ckpt, device)
    run_name = os.path.basename(os.path.dirname(gen_ckpt))

    print(f"\n{'='*60}")
    print(f"Evaluating: {run_name}")
    print(f"  omega={omega}, n_per_class={n_per_class}")
    print(f"{'='*60}")

    per_class: dict[int, float] = {}
    all_correct, all_total = 0, 0

    for c in range(gen.num_classes):
        noise  = torch.randn(n_per_class, gen.noise_dim, device=device)
        labels = torch.full((n_per_class,), c, dtype=torch.long, device=device)
        omegas = torch.full((n_per_class,), omega, device=device)

        z    = gen(noise, labels, omegas)       # [N, latent_dim]
        imgs = ae.decode(z)                     # [N, 1, 28, 28]
        pred = clf(imgs).argmax(dim=1)          # [N]

        correct = int((pred == c).sum().item())
        acc = correct / n_per_class
        per_class[c] = acc
        all_correct += correct
        all_total   += n_per_class
        print(f"  Class {c}: accuracy={acc:.4f}  ({correct}/{n_per_class})")

    overall = all_correct / all_total
    print(f"  ---")
    print(f"  Overall accuracy: {overall:.4f}")

    result = {
        "run_name":        run_name,
        "gen_ckpt":        gen_ckpt,
        "omega":           omega,
        "n_per_class":     n_per_class,
        "per_class_acc":   per_class,
        "overall_acc":     overall,
    }

    out_path = os.path.join(
        os.path.dirname(gen_ckpt), f"acc_results_omega{omega}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    return result


def print_comparison(all_results: list[dict]) -> None:
    if len(all_results) < 2:
        return
    print(f"\n{'='*60}")
    print("COMPARISON TABLE  (class accuracy)")
    print(f"{'='*60}")
    names = [r["run_name"][:28] for r in all_results]
    print(f"{'Class':>6}", end="")
    for n in names:
        print(f"  {n:>28}", end="")
    print()
    print("-" * (6 + 30 * len(all_results)))
    for c in range(10):
        print(f"{c:>6}", end="")
        for r in all_results:
            print(f"  {r['per_class_acc'][c]:>28.4f}", end="")
        print()
    print("-" * (6 + 30 * len(all_results)))
    print(f"{'Avg':>6}", end="")
    for r in all_results:
        print(f"  {r['overall_acc']:>28.4f}", end="")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST class-conditional accuracy evaluation.")
    p.add_argument("--gen-ckpt", type=str, nargs="+", required=True)
    p.add_argument("--ae-ckpt",  type=str, required=True)
    p.add_argument("--clf-ckpt", type=str, default="runs/mnist_clf/mnist_clf.pt",
                   help="Path to cached classifier (trained & saved if missing)")
    p.add_argument("--device",      type=str,   default="cuda")
    p.add_argument("--omega",       type=float, default=1.0)
    p.add_argument("--n-per-class", type=int,   default=1000,
                   help="Generated samples per class")
    p.add_argument("--data-root",   type=str,   default="./data")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Load AE.
    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu", weights_only=True)
    ae = ConvAE(latent_dim=ae_ckpt["latent_dim"]).to(device)
    ae.load_state_dict(ae_ckpt["model"])
    ae.eval()
    print(f"AE: latent_dim={ae_ckpt['latent_dim']}")

    # Load or train classifier.
    clf = load_or_train_classifier(
        args.clf_ckpt, data_root=args.data_root, device=device
    )

    # Evaluate each generator.
    all_results = []
    for ckpt_path in args.gen_ckpt:
        r = evaluate_one(
            ckpt_path, ae, clf,
            omega=args.omega,
            n_per_class=args.n_per_class,
            device=device,
        )
        all_results.append(r)

    if len(all_results) >= 2:
        print_comparison(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
