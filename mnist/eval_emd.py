"""Evaluate MNIST drifting generators: class-level EMD in latent & image space.

For each class c ∈ {0,...,9}:
    1. Generate N samples using MLPGenerator (EMA weights)
    2. Get N real samples from the test set
    3. Compute EMD (Wasserstein-1) between generated and real distributions:
       - Latent space:  W₁(z_gen, z_real)   in R^6
       - Image space:   W₁(x_gen, x_real)   in R^784

EMD is computed via POT (Python Optimal Transport):
    cost matrix C[i,j] = ||a_i - b_j||_2
    EMD = min_π ⟨C, π⟩  s.t. π 1 = μ, πᵀ 1 = ν   (uniform marginals)

Usage:
    python -m mnist.eval_emd \
        --gen-ckpt runs/mnist_drift/mnist_sinkhorn/ckpt_final.pt \
        --ae-ckpt runs/mnist_ae/<run>/ae_final.pt \
        --device cuda:2

    # Compare multiple runs:
    python -m mnist.eval_emd \
        --gen-ckpt runs/mnist_drift/mnist_sinkhorn/ckpt_final.pt \
                   runs/mnist_drift/mnist_baseline/ckpt_final.pt \
        --ae-ckpt runs/mnist_ae/<run>/ae_final.pt \
        --device cuda:2
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import ot
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist.models import ConvAE, MLPGenerator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST class-level EMD evaluation.")
    p.add_argument("--gen-ckpt", type=str, nargs="+", required=True,
                    help="One or more generator checkpoint paths")
    p.add_argument("--ae-ckpt", type=str, required=True, help="Path to ae_final.pt")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--omega", type=float, default=2.0, help="CFG omega for generation")
    p.add_argument("--n-samples", type=int, default=0,
                    help="Samples per class (0 = use all test data per class)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default="./data")
    return p.parse_args()


def compute_emd(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute squared 2-Wasserstein distance W_2^2 between two point clouds.

    X: [N, D], Y: [M, D]
    Returns: W_2^2 value (scalar).
    """
    n, m = len(X), len(Y)
    # Uniform marginals.
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(m, dtype=np.float64) / m
    # Cost matrix: pairwise squared L2 distances.
    C = ot.dist(X, Y, metric="sqeuclidean")  # [N, M]
    emd_val = ot.emd2(a, b, C, numItermax=500000)
    return float(emd_val)  # = W_2^2


@torch.no_grad()
def generate_samples(
    gen: MLPGenerator,
    ae: ConvAE,
    *,
    num_classes: int,
    counts_per_class: dict[int, int],
    omega: float,
    device: torch.device,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Generate samples matching real per-class counts.

    Returns:
        latents_by_class: {c: [N_c, latent_dim]}
        images_by_class:  {c: [N_c, 784]}
    """
    latents_by_class: dict[int, np.ndarray] = {}
    images_by_class: dict[int, np.ndarray] = {}

    for c in range(num_classes):
        n_c = counts_per_class[c]
        noise = torch.randn(n_c, gen.noise_dim, device=device)
        labels = torch.full((n_c,), c, dtype=torch.long, device=device)
        omegas = torch.full((n_c,), omega, device=device)
        z = gen(noise, labels, omegas)  # [N_c, latent_dim]
        imgs = ae.decode(z)  # [N_c, 1, 28, 28]

        latents_by_class[c] = z.cpu().numpy()
        images_by_class[c] = imgs.cpu().view(n_c, -1).numpy()  # [N_c, 784]

    return latents_by_class, images_by_class


@torch.no_grad()
def get_real_samples(
    ae: ConvAE,
    *,
    n_per_class: int,
    device: torch.device,
    data_root: str,
    precomputed_latents: np.ndarray | None = None,
    precomputed_labels: np.ndarray | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int]]:
    """Get real test samples: per-class latents and images.

    Args:
        n_per_class: Max samples per class. 0 means use ALL test data.

    Returns:
        latents_by_class: {c: [N_c, latent_dim]}
        images_by_class:  {c: [N_c, 784]}
        counts_by_class:  {c: N_c}
    """
    # Load test set images.
    tf = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root=data_root, train=False, download=False, transform=tf)
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

    all_imgs, all_labels, all_latents = [], [], []
    for imgs, labels in loader:
        imgs_gpu = imgs.to(device)
        z = ae.encode(imgs_gpu)
        all_imgs.append(imgs.view(imgs.shape[0], -1).numpy())  # [B, 784]
        all_labels.append(labels.numpy())
        all_latents.append(z.cpu().numpy())

    all_imgs_np = np.concatenate(all_imgs, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    all_latents_np = np.concatenate(all_latents, axis=0)

    # If precomputed latents available, use those instead (consistent with training).
    if precomputed_latents is not None and precomputed_labels is not None:
        all_latents_np = precomputed_latents
        all_labels_np_lat = precomputed_labels
    else:
        all_labels_np_lat = all_labels_np

    num_classes = int(all_labels_np.max()) + 1
    latents_by_class: dict[int, np.ndarray] = {}
    images_by_class: dict[int, np.ndarray] = {}
    counts_by_class: dict[int, int] = {}

    for c in range(num_classes):
        idx_lat = np.where(all_labels_np_lat == c)[0]
        idx_img = np.where(all_labels_np == c)[0]

        if n_per_class > 0:
            # Subsample if requested.
            rng = np.random.RandomState(123 + c)
            sel_lat = rng.choice(idx_lat, size=min(n_per_class, len(idx_lat)), replace=False)
            sel_img = rng.choice(idx_img, size=min(n_per_class, len(idx_img)), replace=False)
        else:
            # Use ALL test data for this class.
            sel_lat = idx_lat
            sel_img = idx_img

        latents_by_class[c] = all_latents_np[sel_lat]
        images_by_class[c] = all_imgs_np[sel_img]
        counts_by_class[c] = len(sel_lat)

    return latents_by_class, images_by_class, counts_by_class


def load_generator(ckpt_path: str, device: torch.device) -> MLPGenerator:
    """Load MLPGenerator with EMA weights from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt["gen_config"]
    gen = MLPGenerator(
        num_classes=cfg["num_classes"],
        noise_dim=cfg["noise_dim"],
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    ).to(device)
    if "ema_state" in ckpt:
        gen.load_state_dict(ckpt["ema_state"])
    else:
        gen.load_state_dict(ckpt["model"])
    gen.eval()
    return gen


def evaluate_one(
    gen_ckpt: str,
    ae: ConvAE,
    real_latents: dict[int, np.ndarray],
    real_images: dict[int, np.ndarray],
    counts_per_class: dict[int, int],
    *,
    omega: float,
    device: torch.device,
) -> dict:
    """Evaluate a single generator checkpoint. Returns results dict."""
    gen = load_generator(gen_ckpt, device)
    run_name = os.path.basename(os.path.dirname(gen_ckpt))
    total_n = sum(counts_per_class.values())
    print(f"\n{'='*60}")
    print(f"Evaluating: {run_name}")
    print(f"  Checkpoint: {gen_ckpt}")
    print(f"  omega={omega}, total_samples={total_n} (per-class: {min(counts_per_class.values())}-{max(counts_per_class.values())})")
    print(f"{'='*60}")

    gen_latents, gen_images = generate_samples(
        gen, ae,
        num_classes=gen.num_classes,
        counts_per_class=counts_per_class,
        omega=omega,
        device=device,
    )

    results_per_class = {}
    emd_latent_all = []
    emd_image_all = []

    for c in range(gen.num_classes):
        emd_lat = compute_emd(gen_latents[c], real_latents[c])
        emd_img = compute_emd(gen_images[c], real_images[c])
        emd_latent_all.append(emd_lat)
        emd_image_all.append(emd_img)
        results_per_class[c] = {"emd_latent": emd_lat, "emd_image": emd_img}
        print(f"  Class {c}: EMD_latent={emd_lat:.6f}  EMD_image={emd_img:.6f}")

    avg_latent = float(np.mean(emd_latent_all))
    avg_image = float(np.mean(emd_image_all))
    print(f"  ---")
    print(f"  Average:  EMD_latent={avg_latent:.6f}  EMD_image={avg_image:.6f}")

    result = {
        "run_name": run_name,
        "gen_ckpt": gen_ckpt,
        "omega": omega,
        "n_per_class": dict(counts_per_class),
        "per_class": results_per_class,
        "avg_emd_latent": avg_latent,
        "avg_emd_image": avg_image,
    }

    # Save results next to checkpoint.
    out_path = os.path.join(os.path.dirname(gen_ckpt), f"emd_results_omega{omega}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")

    return result


def print_comparison(all_results: list[dict]) -> None:
    """Print a comparison table across multiple runs."""
    if len(all_results) < 2:
        return

    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")

    # Header.
    header = f"{'Class':>8}"
    for r in all_results:
        name = r["run_name"]
        header += f"  {name:>20}(lat)  {name:>20}(img)"
    # Simplified header.
    print(f"\n{'':>8}", end="")
    for r in all_results:
        name = r["run_name"][:25]
        print(f"  | {name:^30}", end="")
    print()
    print(f"{'Class':>8}", end="")
    for _ in all_results:
        print(f"  | {'EMD_latent':>13} {'EMD_image':>13}", end="")
    print()
    print("-" * (8 + 32 * len(all_results)))

    for c in range(10):
        print(f"{c:>8}", end="")
        for r in all_results:
            lat = r["per_class"][c]["emd_latent"]
            img = r["per_class"][c]["emd_image"]
            print(f"  | {lat:>13.6f} {img:>13.6f}", end="")
        print()

    print("-" * (8 + 32 * len(all_results)))
    print(f"{'Average':>8}", end="")
    for r in all_results:
        print(f"  | {r['avg_emd_latent']:>13.6f} {r['avg_emd_image']:>13.6f}", end="")
    print()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Load AE.
    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu", weights_only=True)
    latent_dim = ae_ckpt["latent_dim"]
    ae = ConvAE(latent_dim=latent_dim).to(device)
    ae.load_state_dict(ae_ckpt["model"])
    ae.eval()
    print(f"AE: latent_dim={latent_dim}")

    # Load precomputed test latents if available.
    ae_dir = os.path.dirname(args.ae_ckpt)
    test_lat_path = os.path.join(ae_dir, "test_latents.npy")
    test_lbl_path = os.path.join(ae_dir, "test_labels.npy")
    precomputed_lat, precomputed_lbl = None, None
    if os.path.exists(test_lat_path):
        precomputed_lat = np.load(test_lat_path)
        precomputed_lbl = np.load(test_lbl_path)
        print(f"Using precomputed test latents: {precomputed_lat.shape}")

    # Get real samples (computed once, shared across all generators).
    print("Loading real test samples...")
    real_latents, real_images, counts_per_class = get_real_samples(
        ae,
        n_per_class=args.n_samples,
        device=device,
        data_root=args.data_root,
        precomputed_latents=precomputed_lat,
        precomputed_labels=precomputed_lbl,
    )
    for c in sorted(counts_per_class):
        print(f"  Class {c}: {counts_per_class[c]} samples")

    # Evaluate each generator.
    all_results = []
    for ckpt_path in args.gen_ckpt:
        result = evaluate_one(
            ckpt_path, ae, real_latents, real_images, counts_per_class,
            omega=args.omega,
            device=device,
        )
        # Convert int keys to str for JSON serialization.
        result["per_class"] = {str(k): v for k, v in result["per_class"].items()}
        all_results.append(result)

    # Print comparison if multiple runs.
    if len(all_results) >= 2:
        # Convert keys back to int for display.
        for r in all_results:
            r["per_class"] = {int(k): v for k, v in r["per_class"].items()}
        print_comparison(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
