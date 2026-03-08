"""Generate high-quality PDF figures for paper (4x1 horizontal layout).

Main figure (Gaussian kernel):
    (a) Baseline rho=0.01  (b) Sinkhorn rho=0.01  (c) Baseline rho=0.1  (d) Sinkhorn rho=0.1

Appendix figure (Laplacian kernel):
    (a) Baseline rho=0.01  (b) Sinkhorn rho=0.01  (c) Baseline rho=0.05  (d) Sinkhorn rho=0.05

Usage:
    # Main figure (Gaussian)
    python -m mnist.make_figure \
        --ae-ckpt runs/mnist_ae/20260226_223742_latent6/ae_final.pt \
        --out paper/Figures/mnist/mnist_gaussian.pdf \
        --device cuda:0

    # Appendix figure (Laplacian)
    python -m mnist.make_figure \
        --ae-ckpt runs/mnist_ae/20260226_223742_latent6/ae_final.pt \
        --out paper/Figures/mnist/mnist_laplacian.pdf \
        --kernel laplacian --device cuda:0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.compression"] = 0   # disable zlib compression → larger but print-ready
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mnist.models import ConvAE, MLPGenerator


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


@torch.no_grad()
def generate_grid_np(
    gen: MLPGenerator,
    ae: ConvAE,
    *,
    omega: float,
    n_per_class: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Returns [10*n_per_class, 28, 28] float32 array in [0,1]."""
    torch.manual_seed(seed)
    imgs_all = []
    for c in range(gen.num_classes):
        noise  = torch.randn(n_per_class, gen.noise_dim, device=device)
        labels = torch.full((n_per_class,), c, dtype=torch.long, device=device)
        omegas = torch.full((n_per_class,), omega, device=device)
        z    = gen(noise, labels, omegas)
        imgs = ae.decode(z).cpu().squeeze(1).numpy()   # [N, 28, 28]
        imgs_all.append(imgs)
    return np.concatenate(imgs_all, axis=0)             # [10*N, 28, 28]


def draw_sample_panel(
    ax: plt.Axes,
    images: np.ndarray,
    *,
    n_per_class: int,
    num_classes: int = 10,
    gap: int = 1,
) -> None:
    """Draw a (num_classes x n_per_class) grid of images onto ax."""
    h, w = images.shape[1], images.shape[2]
    grid_h = num_classes * h + (num_classes - 1) * gap
    grid_w = n_per_class  * w + (n_per_class  - 1) * gap
    canvas = np.ones((grid_h, grid_w), dtype=np.float32)   # white gaps

    for c in range(num_classes):
        for j in range(n_per_class):
            r0 = c * (h + gap)
            c0 = j * (w + gap)
            canvas[r0:r0+h, c0:c0+w] = images[c * n_per_class + j]

    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")


def make_figure(
    panels: list[dict],          # list of {ckpt, label, omega}
    ae: ConvAE,
    *,
    n_per_class: int = 8,
    seed: int = 42,
    device: torch.device,
    out_path: str,
    ncols: int = 2,
) -> None:
    nrows = (len(panels) + ncols - 1) // ncols
    # Each panel is 10 rows x n_per_class cols of 28x28 images.
    # aspect ratio of one panel: (10*28) / (n_per_class*28) = 10/n_per_class
    panel_h = 10 * 28 + 9          # pixels (with 1px gap)
    panel_w = n_per_class * 28 + (n_per_class - 1)
    aspect  = panel_h / panel_w

    # ECCV double-column full width ≈ 6.8 in; single column ≈ 3.3 in.
    # For 4×1, span full text width.
    fig_w   = 6.8 if ncols >= 3 else 3.3
    cell_w  = fig_w / ncols
    cell_h  = cell_w * aspect
    fig_h   = cell_h * nrows + 0.30 * nrows   # extra for subcaption

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=600)
    gs  = gridspec.GridSpec(
        nrows, ncols,
        figure=fig,
        wspace=0.03, hspace=0.15,
    )

    for idx, panel in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])

        gen    = load_generator(panel["ckpt"], device)
        images = generate_grid_np(gen, ae, omega=panel["omega"],
                                  n_per_class=n_per_class, seed=seed,
                                  device=device)
        draw_sample_panel(ax, images, n_per_class=n_per_class)
        ax.set_title(panel["label"], fontsize=8, pad=3)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ae-ckpt",     type=str, required=True)
    p.add_argument("--run-root",    type=str, default="runs/mnist_drift")
    p.add_argument("--out",         type=str, default="paper/Figures/mnist/mnist_gaussian.pdf")
    p.add_argument("--kernel",      type=str, default="gaussian",
                   choices=["gaussian", "laplacian"],
                   help="Which kernel variant to plot")
    p.add_argument("--omega",       type=float, default=1.0)
    p.add_argument("--n-per-class", type=int,   default=8)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      type=str,   default="cuda:0")
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    device = torch.device(args.device)

    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu", weights_only=True)
    ae = ConvAE(latent_dim=ae_ckpt["latent_dim"]).to(device)
    ae.load_state_dict(ae_ckpt["model"])
    ae.eval()

    rr = args.run_root

    if args.kernel == "gaussian":
        panels = [
            {"ckpt": f"{rr}/mnist_baseline_tau0p01_l2sq/ckpt_final.pt",
             "label": "(a) Baseline, $\\tau=0.01$",  "omega": args.omega},
            {"ckpt": f"{rr}/mnist_sinkhorn_tau0p01_l2sq/ckpt_final.pt",
             "label": "(b) Sinkhorn, $\\tau=0.01$",  "omega": args.omega},
            {"ckpt": f"{rr}/mnist_baseline_tau0p1_l2sq/ckpt_final.pt",
             "label": "(c) Baseline, $\\tau=0.1$",   "omega": args.omega},
            {"ckpt": f"{rr}/mnist_sinkhorn_tau0p1_l2sq/ckpt_final.pt",
             "label": "(d) Sinkhorn, $\\tau=0.1$",   "omega": args.omega},
        ]
    else:  # laplacian
        panels = [
            {"ckpt": f"{rr}/mnist_baseline_tau0p01/ckpt_final.pt",
             "label": "(a) Baseline, $\\tau=0.01$",  "omega": args.omega},
            {"ckpt": f"{rr}/mnist_sinkhorn_tau0p01/ckpt_final.pt",
             "label": "(b) Sinkhorn, $\\tau=0.01$",  "omega": args.omega},
            {"ckpt": f"{rr}/mnist_baseline_tau0p05/ckpt_final.pt",
             "label": "(c) Baseline, $\\tau=0.05$",  "omega": args.omega},
            {"ckpt": f"{rr}/mnist_sinkhorn_tau0p05/ckpt_final.pt",
             "label": "(d) Sinkhorn, $\\tau=0.05$",  "omega": args.omega},
        ]

    make_figure(
        panels, ae,
        n_per_class=args.n_per_class,
        seed=args.seed,
        device=device,
        out_path=args.out,
        ncols=4,          # 4x1 horizontal for double-column ECCV
    )


if __name__ == "__main__":
    main()
