"""
drift_ffhq.py
-------------
Conditional generative model  f : R^{d_z + d_e} → R^{512}
trained via barycentric drift in FFHQ ALAE's 512-dimensional latent space.

Mathematical setup
------------------
Let  Z_c = {z ∈ R^{512} : label(z) = c}  be the empirical target for class c.
Let  ε_c ∈ R^{d_e}  be a *learned* class embedding  (c = 0..5).
Let  noise  n ~ N(0, I_{d_z}).

The generator is:
    f(n, c; θ, E) = MLP( concat(n, ε_c) )   ∈ R^{512}

Drift objective (conditional, per iteration):
    L(θ, E) = (1/N) Σ_i  ‖ f(nᵢ, cᵢ) − sg[ f(nᵢ, cᵢ) + V(f(nᵢ,cᵢ); Z_{cᵢ}) ] ‖²

where V(x; Y_c) is the barycentric drift field using **only** class-c target points,
keeping the transport semantically class-consistent.

Three plan estimators (--plan): one-sided | two-sided | sinkhorn.
EMD is tracked per-class and globally (using POT, optional).

Usage
-----
    pip install torch numpy matplotlib pot scikit-learn
    python drift_ffhq.py --train-npz train_latents_by_class.npz \\
                         --test-npz  test_latents_by_class.npz  \\
                         --meta      meta.json

Outputs
-------
  drift_ffhq_model.pt         checkpoint: MLP + embedding table + metadata
  drift_ffhq_emd.png          global EMD² vs iteration
  drift_ffhq_emd_perclass.png per-class EMD² vs iteration
  drift_ffhq_pca.png          PCA scatter: source → final vs target (coloured by class)
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import ot
except ImportError:
    ot = None

try:
    from sklearn.decomposition import PCA as SklearnPCA
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditional latent-space drift – FFHQ ALAE")

    # I/O
    p.add_argument("--train-npz", type=str,
                   default="ffhq_latents_6class/train_latents_by_class.npz")
    p.add_argument("--test-npz",  type=str,
                   default="ffhq_latents_6class/test_latents_by_class.npz")
    p.add_argument("--meta",      type=str, default="meta.json")
    p.add_argument("--save-path", type=str, default="drift_ffhq_model.pt")
    p.add_argument("--emd-plot",          type=str, default="drift_ffhq_emd.png")
    p.add_argument("--emd-perclass-plot", type=str, default="drift_ffhq_emd_perclass.png")
    p.add_argument("--pca-plot",          type=str, default="drift_ffhq_pca.png")

    # Architecture
    p.add_argument("--d-z",    type=int, default=512,
                   help="Noise dimensionality (default: 512 = latent dim).")
    p.add_argument("--d-e",    type=int, default=64,
                   help="Learned class-embedding dimension.")
    p.add_argument("--hidden", type=int, default=1024,
                   help="Hidden width (repeated for --n-hidden layers).")
    p.add_argument("--n-hidden", type=int, default=3,
                   help="Number of hidden layers.")

    # Training
    p.add_argument("--iters",      type=int,   default=2000)
    p.add_argument("--batch-size", type=int,   default=512,
                   help="Total batch size; split equally across 6 classes.")
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--emb-lr",     type=float, default=1e-3,
                   help="Separate (higher) LR for embedding table.")

    # Drift
    p.add_argument("--plan",           type=str,   default="one-sided",
                   choices=["one-sided", "two-sided", "sinkhorn"])
    p.add_argument("--eps",            type=float, default=1.0,
                   help="Entropic regularisation ε for transport plan.")
    p.add_argument("--sinkhorn-iters", type=int,   default=30)
    p.add_argument("--dist",           type=str,   default="l2_sq",
                   choices=["l2_sq", "l2", "cosine"])

    # Evaluation
    p.add_argument("--emd-every",   type=int, default=100)
    p.add_argument("--emd-samples", type=int, default=512,
                   help="Samples per class for EMD evaluation.")
    p.add_argument("--log-every",   type=int, default=50)

    # Misc
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "male_children", "male_adult",   "male_old",
    "female_children","female_adult","female_old",
]

def load_latents(npz_path: str) -> Dict[str, torch.Tensor]:
    """
    Load all class latents from .npz into a dict  class_name → (N, 512) float32 tensor.
    """
    data  = np.load(npz_path, allow_pickle=True)
    out   = {}
    for name in CLASS_NAMES:
        arr = data[name].astype(np.float32)   # (N, 512)
        out[name] = torch.from_numpy(arr)
    return out


class ConditionalLatentDataset:
    """
    Stateful sampler for balanced class mini-batches.

    draw(n_per_class, device) → (latents, labels)
      latents : (6*n_per_class, 512)
      labels  : (6*n_per_class,)  int in {0..5}

    For each class a random draw of n_per_class points is returned (with
    replacement if the class pool is smaller than n_per_class, though that is
    never the case here for reasonable batch sizes).
    """

    def __init__(self, npz_path: str):
        self.latents = load_latents(npz_path)
        self.class_ids = {name: i for i, name in enumerate(CLASS_NAMES)}

    def draw(
        self,
        n_per_class: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_list, label_list = [], []
        for cname, cid in self.class_ids.items():
            pool = self.latents[cname]          # (N_c, 512)
            idx  = torch.randint(0, pool.shape[0], (n_per_class,))
            latent_list.append(pool[idx])
            label_list.append(torch.full((n_per_class,), cid, dtype=torch.long))
        return (
            torch.cat(latent_list, dim=0).to(device),
            torch.cat(label_list,  dim=0).to(device),
        )

    def class_pool(self, cname: str) -> torch.Tensor:
        """Return the full (N_c, 512) tensor for class cname."""
        return self.latents[cname]


# ─────────────────────────────────────────────────────────────────────────────
# Model: conditional MLP + learned class embeddings
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalDriftMLP(nn.Module):
    """
    f(n, c) = MLP( concat(n, E[c]) )

    Parameters
    ----------
    d_z      : noise / source dimension
    d_e      : embedding dimension
    d_out    : output dimension (= latent dim = 512)
    hidden   : hidden layer width
    n_hidden : number of hidden layers
    n_classes: number of classes
    """

    def __init__(
        self,
        d_z:      int = 512,
        d_e:      int = 64,
        d_out:    int = 512,
        hidden:   int = 1024,
        n_hidden: int = 3,
        n_classes:int = 6,
    ) -> None:
        super().__init__()

        # Learned embedding table  E ∈ R^{n_classes × d_e}
        # Initialised N(0, 0.01) so embeddings start near origin and
        # are differentiated from each other only by gradient pressure.
        self.embedding = nn.Embedding(n_classes, d_e)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

        # Build MLP:  (d_z + d_e) → hidden → … → d_out  (linear output)
        d_in = d_z + d_e
        layers: List[nn.Module] = []
        prev = d_in
        for _ in range(n_hidden):
            lin = nn.Linear(prev, hidden)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.extend([lin, nn.ReLU(inplace=True)])
            prev = hidden

        out_lin = nn.Linear(prev, d_out)
        nn.init.xavier_uniform_(out_lin.weight)
        nn.init.zeros_(out_lin.bias)
        layers.append(out_lin)

        self.mlp = nn.Sequential(*layers)

        self.d_z      = d_z
        self.d_e      = d_e
        self.d_out    = d_out

    def forward(
        self,
        noise: torch.Tensor,   # (B, d_z)
        labels: torch.Tensor,  # (B,)  int64
    ) -> torch.Tensor:         # (B, d_out)
        e  = self.embedding(labels)           # (B, d_e)
        x  = torch.cat([noise, e], dim=1)     # (B, d_z + d_e)
        return self.mlp(x)


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise distances
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_dists(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: str = "l2_sq",
) -> torch.Tensor:
    """
    (N, M) distance matrix between rows of x ∈ R^{N×d} and y ∈ R^{M×d}.

    l2_sq  : ‖xᵢ − yⱼ‖²  (numerically stable via ||a-b||² = ||a||²+||b||²-2<a,b>)
    l2     : ‖xᵢ − yⱼ‖
    cosine : 1 − cos(xᵢ, yⱼ) ∈ [0, 2]
    """
    if metric == "l2_sq":
        x_sq  = (x * x).sum(1, keepdim=True)       # (N, 1)
        y_sq  = (y * y).sum(1, keepdim=True).T      # (1, M)
        cross = x @ y.T                             # (N, M)
        return (x_sq + y_sq - 2.0 * cross).clamp(min=0.0)
    elif metric == "l2":
        return pairwise_dists(x, y, "l2_sq").sqrt()
    elif metric == "cosine":
        return 1.0 - F.normalize(x, 2, 1) @ F.normalize(y, 2, 1).T
    else:
        raise ValueError(f"Unknown metric: {metric!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Transport plans  (all in log-space)
# ─────────────────────────────────────────────────────────────────────────────

def _gibbs_logits(x, y, eps, metric):
    return -pairwise_dists(x, y, metric) / eps


def plan_one_sided(x, y, eps, metric="l2_sq") -> torch.Tensor:
    """Row-normalised Gibbs kernel → row-stochastic (N, M)."""
    log_t = _gibbs_logits(x, y, eps, metric)
    log_t = log_t - torch.logsumexp(log_t, 1, keepdim=True)
    t     = torch.exp(log_t)
    return t / (t.sum(1, keepdim=True) + 1e-12)


def plan_two_sided(x, y, eps, metric="l2_sq") -> torch.Tensor:
    """Geometric mean of row & col normalisations, then row-renormalised."""
    log_t   = _gibbs_logits(x, y, eps, metric)
    log_row = log_t - torch.logsumexp(log_t, 1, keepdim=True)
    log_col = log_t - torch.logsumexp(log_t, 0, keepdim=True)
    t       = torch.exp(0.5 * (log_row + log_col))
    return t / (t.sum(1, keepdim=True) + 1e-12)


def plan_sinkhorn(x, y, eps, iters=30, metric="l2_sq") -> torch.Tensor:
    """Balanced Sinkhorn in log-domain → row-stochastic (N, M)."""
    log_t = _gibbs_logits(x, y, eps, metric)
    for _ in range(iters):
        log_t = log_t - torch.logsumexp(log_t, 1, keepdim=True)
        log_t = log_t - torch.logsumexp(log_t, 0, keepdim=True)
    log_t = log_t - torch.logsumexp(log_t, 1, keepdim=True)
    t     = torch.exp(log_t)
    return t / (t.sum(1, keepdim=True) + 1e-12)


def get_plan(plan_type: str):
    return {"one-sided": plan_one_sided,
            "two-sided": plan_two_sided,
            "sinkhorn":  plan_sinkhorn}[plan_type]


# ─────────────────────────────────────────────────────────────────────────────
# Barycentric drift  V(x ; y)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def barycentric_drift(
    x:            torch.Tensor,    # (N, d)  source points
    y:            torch.Tensor,    # (M, d)  target points  (same class)
    eps:          float,
    plan_type:    str   = "one-sided",
    dist_metric:  str   = "l2_sq",
    sinkhorn_iters: int = 30,
) -> torch.Tensor:                 # (N, d)
    """
    V(x; y) = v_attr(x,y) − v_rep(x,x)

    v_attr = P_{xy} · y − x    (pull toward class-y barycentres)
    v_rep  = P_{xx} · x − x    (push away from own cluster centre)

    Both plans are row-stochastic; the result is the *net* displacement that
    transports source mass toward target mass of the same class.
    """
    x = x.float();  y = y.float()
    plan_fn = get_plan(plan_type)
    kwargs  = dict(eps=eps, metric=dist_metric)
    if plan_type == "sinkhorn":
        kwargs["iters"] = sinkhorn_iters

    pxy     = plan_fn(x, y, **kwargs)     # (N, M)
    pxx     = plan_fn(x, x, **kwargs)     # (N, N)

    v_attr  = pxy @ y - x                 # (N, d)
    v_rep   = pxx @ x - x                 # (N, d)
    return v_attr - v_rep


# ─────────────────────────────────────────────────────────────────────────────
# Conditional drift loss
# ─────────────────────────────────────────────────────────────────────────────

def conditional_drift_loss(
    f_x:          torch.Tensor,             # (B, 512)  generator output
    labels:       torch.Tensor,             # (B,)      class ids 0..5
    target_by_class: Dict[int, torch.Tensor],  # cid → (N_c, 512) on device
    eps:          float,
    plan_type:    str,
    dist_metric:  str,
    sinkhorn_iters: int,
) -> torch.Tensor:
    """
    Conditional drift loss.

    For each class c present in the batch:
      1.  Extract f_x[c] = f_x[labels == c]              shape (n_c, 512)
      2.  Draw target cloud Y_c = target_by_class[c]      shape (N_c, 512)
      3.  Compute V(f_x[c]; Y_c)                          shape (n_c, 512)
      4.  pseudo-target  z*[c] = sg(f_x[c] + V)
      5.  loss += ‖ f_x[c] − z*[c] ‖²_F

    Sum over classes, mean over samples.

    Gradient flows only through the left f_x[c] term (right is detached).
    """
    total_loss = torch.zeros(1, device=f_x.device, dtype=f_x.dtype)
    n_total    = 0

    for cid, Y_c in target_by_class.items():
        mask   = (labels == cid)
        n_c    = mask.sum().item()
        if n_c == 0:
            continue

        fx_c   = f_x[mask]                # (n_c, 512),  grad attached

        with torch.no_grad():
            v = barycentric_drift(
                fx_c.detach(), Y_c,
                eps=eps,
                plan_type=plan_type,
                dist_metric=dist_metric,
                sinkhorn_iters=sinkhorn_iters,
            )
            target_c = fx_c.detach() + v  # sg( f_x[c] + V )

        diff       = fx_c - target_c      # grad flows through fx_c
        total_loss = total_loss + (diff * diff).sum()
        n_total   += n_c

    return total_loss / max(n_total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# EMD via POT
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def emd_pot(x: torch.Tensor, y: torch.Tensor) -> float:
    """EMD² with squared-Euclidean ground metric (uniform marginals)."""
    if ot is None:
        raise RuntimeError("POT not installed.  pip install pot")
    x_np = x.cpu().float().numpy().astype(np.float64)
    y_np = y.cpu().float().numpy().astype(np.float64)
    n, m = len(x_np), len(y_np)
    a    = np.full(n, 1.0 / n)
    b    = np.full(m, 1.0 / m)
    C    = ot.dist(x_np, y_np, metric="euclidean") ** 2
    return float(ot.emd2(a, b, C))


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

# Consistent class colour palette
CLASS_COLORS = [
    "#3B82F6",  # blue        – male_children
    "#1D4ED8",  # dark-blue   – male_adult
    "#1E3A5F",  # navy        – male_old
    "#F472B6",  # pink        – female_children
    "#EC4899",  # hot-pink    – female_adult
    "#9D174D",  # dark-pink   – female_old
]


def plot_emd_curve(
    iters_axis: List[int],
    emd_vals:   List[float],
    save_path:  str,
    title:      str = "Global EMD² – f(z,c) vs target",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters_axis, emd_vals, color="#2563EB", linewidth=1.8,
            marker="o", markersize=3)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("EMD²", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {save_path}")


def plot_emd_perclass(
    iters_axis:        List[int],
    emd_by_class:      Dict[int, List[float]],
    save_path:         str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for cid, name in enumerate(CLASS_NAMES):
        vals = emd_by_class.get(cid, [])
        if not vals:
            continue
        ax.plot(iters_axis[:len(vals)], vals,
                color=CLASS_COLORS[cid], linewidth=1.6,
                marker="o", markersize=3, label=name)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("EMD²", fontsize=11)
    ax.set_title("Per-class EMD² over training", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {save_path}")


def plot_pca_trajectory(
    snapshots_by_class: Dict[str, Dict[int, torch.Tensor]],
    # "source"/"final" → {cid → (N_eval, 512)}
    target_latents: Dict[int, torch.Tensor],
    save_path: str,
) -> None:
    """
    PCA scatter coloured by class showing:
      • target distribution (solid markers)
      • source f(z, c) at iter 1 (faint)
      • final f(z, c) at last iter (vivid)
    """
    if not _HAS_SKLEARN:
        print("[plot] scikit-learn not found; skipping PCA trajectory.")
        return

    # Collect all points for PCA fitting
    all_pts = []
    for cid in range(len(CLASS_NAMES)):
        for snap_dict in snapshots_by_class.values():
            if cid in snap_dict:
                all_pts.append(snap_dict[cid])
        if cid in target_latents:
            all_pts.append(target_latents[cid])
    all_pts_np = torch.cat(all_pts, dim=0).numpy()

    pca = SklearnPCA(n_components=2)
    pca.fit(all_pts_np)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["Source (f at iter 1)", "Final (f at last iter)"]
    keys   = ["source", "final"]
    alphas = [0.3, 0.65]

    for ax, key, alpha, title in zip(axes, keys, alphas, titles):
        snap = snapshots_by_class.get(key, {})
        for cid, name in enumerate(CLASS_NAMES):
            color = CLASS_COLORS[cid]
            # Generated
            if cid in snap:
                pts = pca.transform(snap[cid].numpy())
                ax.scatter(pts[:, 0], pts[:, 1],
                           s=4, alpha=alpha, color=color,
                           label=f"gen {name}" if key == "final" else None,
                           rasterized=True)
            # Target
            if cid in target_latents:
                tgt = pca.transform(target_latents[cid].numpy())
                ax.scatter(tgt[:, 0], tgt[:, 1],
                           s=4, alpha=0.2, color=color,
                           marker="^", rasterized=True,
                           label=f"tgt {name}" if key == "final" else None)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"PC 1  ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC 2  ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=9)
        ax.grid(alpha=0.2)

    # Single legend from the "final" panel
    axes[1].legend(fontsize=6, markerscale=2, ncol=2,
                   bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.suptitle("Conditional drift: PCA trajectory (source → target)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()
    seed_all(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print(f"[device]  {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"[data]    loading training latents from {args.train_npz!r} …")
    train_ds = ConditionalLatentDataset(args.train_npz)

    print(f"[data]    loading test latents from {args.test_npz!r} …")
    test_ds  = ConditionalLatentDataset(args.test_npz)

    # Put full class pools on device for target clouds during drift
    train_pool_device: Dict[int, torch.Tensor] = {
        i: train_ds.class_pool(name).to(device)
        for i, name in enumerate(CLASS_NAMES)
    }

    # Fixed eval pool (from test set, subsampled)
    seed_all(args.seed + 777)
    eval_target: Dict[int, torch.Tensor] = {}
    for cid, name in enumerate(CLASS_NAMES):
        pool = test_ds.class_pool(name)
        n    = min(args.emd_samples, pool.shape[0])
        idx  = torch.randperm(pool.shape[0])[:n]
        eval_target[cid] = pool[idx].to(device)
    seed_all(args.seed)

    d_latent  = 512
    n_classes = len(CLASS_NAMES)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConditionalDriftMLP(
        d_z      = args.d_z,
        d_e      = args.d_e,
        d_out    = d_latent,
        hidden   = args.hidden,
        n_hidden = args.n_hidden,
        n_classes= n_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arch_str  = (f"[{args.d_z}+{args.d_e}]"
                 f"→{'→'.join([str(args.hidden)]*args.n_hidden)}"
                 f"→{d_latent}")
    print(f"[model]   ConditionalDriftMLP  {arch_str}  ({n_params:,} params)")
    print(f"[drift]   plan={args.plan}  ε={args.eps}  dist={args.dist}\n")

    # ── Optimiser: separate LRs for embeddings vs MLP ─────────────────────────
    optimizer = torch.optim.Adam([
        {"params": model.mlp.parameters(),       "lr": args.lr},
        {"params": model.embedding.parameters(), "lr": args.emb_lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iters, eta_min=1e-5
    )

    # ── Fixed noise for consistent evaluation ─────────────────────────────────
    seed_all(args.seed + 999)
    n_eval       = args.emd_samples
    # n_eval samples per class
    eval_noise   = torch.randn(n_eval * n_classes, args.d_z, device=device)
    eval_labels  = torch.repeat_interleave(
        torch.arange(n_classes, device=device), n_eval
    )  # 0,0,…,1,1,…,5,5,…
    seed_all(args.seed)

    # ── PCA snapshot storage ──────────────────────────────────────────────────
    pca_snapshots: Dict[str, Dict[int, torch.Tensor]] = {}

    def take_snapshot(tag: str) -> None:
        model.eval()
        with torch.no_grad():
            gen = model(eval_noise, eval_labels)   # (n_eval*6, 512)
        snap = {}
        for cid in range(n_classes):
            mask = (eval_labels == cid)
            snap[cid] = gen[mask].cpu()
        pca_snapshots[tag] = snap

    # ── Training loop ─────────────────────────────────────────────────────────
    n_per_class = max(1, args.batch_size // n_classes)

    emd_iters:    List[int]              = []
    emd_global:   List[float]            = []
    emd_perclass: Dict[int, List[float]] = {i: [] for i in range(n_classes)}

    print(f"{'Iter':>6}  {'Loss':>12}  {'EMD²(global)':>14}  {'LR':>10}")
    print("─" * 52)

    for it in range(1, args.iters + 1):

        # ── Snapshot at iteration 1 ───────────────────────────────────────────
        if it == 1:
            take_snapshot("source")

        model.train()

        # Sample batch: n_per_class samples per class (balanced)
        Y_batch, labels = train_ds.draw(n_per_class, device)
        # Y_batch : (6*n_per_class, 512) — real latents (targets)
        # labels  : (6*n_per_class,)     — class ids

        # Noise input
        noise = torch.randn(labels.shape[0], args.d_z, device=device)

        # Generator output  f(z, c)
        f_x = model(noise, labels)             # (B, 512)

        # Build per-class target dict for this batch's classes
        # We use the *full* training pool for each class (stable target cloud)
        target_by_class = {
            cid: train_pool_device[cid]
            for cid in labels.unique().tolist()
        }

        # Conditional drift loss
        loss = conditional_drift_loss(
            f_x             = f_x,
            labels          = labels,
            target_by_class = target_by_class,
            eps             = args.eps,
            plan_type       = args.plan,
            dist_metric     = args.dist,
            sinkhorn_iters  = args.sinkhorn_iters,
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        # ── EMD evaluation ────────────────────────────────────────────────────
        emd_str = "             —"
        if it % args.emd_every == 0 or it == 1:
            model.eval()
            with torch.no_grad():
                gen_eval = model(eval_noise, eval_labels)   # (n_eval*6, 512)

            if ot is not None:
                # Global EMD²: all generated vs all targets
                all_gen = gen_eval
                all_tgt = torch.cat([eval_target[c] for c in range(n_classes)], 0)
                try:
                    g_emd = emd_pot(all_gen, all_tgt)
                    emd_iters.append(it)
                    emd_global.append(g_emd)
                    emd_str = f"{g_emd:>14.5f}"

                    # Per-class EMD²
                    for cid in range(n_classes):
                        mask = (eval_labels == cid)
                        c_emd = emd_pot(gen_eval[mask], eval_target[cid])
                        emd_perclass[cid].append(c_emd)
                except Exception as e:
                    emd_str = "    POT error"
                    print(f"  [emd warning] {e}")

        # ── Logging ───────────────────────────────────────────────────────────
        if it % args.log_every == 0 or it == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"{it:>6}  {loss.item():>12.6f}  {emd_str}  {lr_now:>10.2e}")

    # ── Final snapshot ────────────────────────────────────────────────────────
    take_snapshot("final")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    checkpoint = {
        "state_dict":  model.state_dict(),
        "model_config": dict(
            d_z       = args.d_z,
            d_e       = args.d_e,
            d_out     = d_latent,
            hidden    = args.hidden,
            n_hidden  = args.n_hidden,
            n_classes = n_classes,
        ),
        "class_names": CLASS_NAMES,
        "hyperparams": vars(args),
        "emd_history": {
            "iters":   emd_iters,
            "global":  emd_global,
            "perclass": {CLASS_NAMES[i]: v for i, v in emd_perclass.items()},
        },
    }
    torch.save(checkpoint, args.save_path)
    print(f"\n[save]   model → {args.save_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if emd_iters:
        plot_emd_curve(emd_iters, emd_global, args.emd_plot)
        plot_emd_perclass(emd_iters, emd_perclass, args.emd_perclass_plot)

    pca_target_cpu = {cid: t.cpu() for cid, t in eval_target.items()}
    plot_pca_trajectory(pca_snapshots, pca_target_cpu, args.pca_plot)

    # ── Final summary ─────────────────────────────────────────────────────────
    if emd_global:
        pct = (1 - emd_global[-1] / emd_global[0]) * 100
        print(f"\n[summary]  Initial EMD²  (global): {emd_global[0]:.5f}")
        print(f"[summary]  Final   EMD²  (global): {emd_global[-1]:.5f}")
        print(f"[summary]  Reduction:              {pct:.1f} %")
        print("\n[summary]  Per-class final EMD²:")
        for cid, name in enumerate(CLASS_NAMES):
            vals = emd_perclass[cid]
            if vals:
                print(f"             {name:<20s} {vals[-1]:.5f}")

    print("\n[done]")


if __name__ == "__main__":
    main()
