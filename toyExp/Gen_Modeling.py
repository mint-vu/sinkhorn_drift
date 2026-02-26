"""
Drift-based generative modeling comparison.

What this script provides:
1) Three drift methods: one-sided / two-sided / sinkhorn
2) Two drift implementations for comparison:
   - "plain": professor version in normal space
   - "log":   log-space stable implementation
3) Three plot families:
   - generated source vs target
   - EMD curve (1x4 style, POT emd2)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as _font_manager
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
try:
    import ot  # POT
except ModuleNotFoundError:  # pragma: no cover
    ot = None
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use Gill Sans from a local OTF (no fallback).
_GILL_SANS_PATH = os.path.join(os.path.dirname(__file__), "Gill Sans Medium.otf")
if not os.path.exists(_GILL_SANS_PATH):
    raise FileNotFoundError(f"Gill Sans font file not found: {_GILL_SANS_PATH}")
_font_manager.fontManager.addfont(_GILL_SANS_PATH)
_GILL_SANS_NAME = _font_manager.FontProperties(fname=_GILL_SANS_PATH).get_name()
# Ensure matplotlib can resolve this font without falling back to defaults.
_font_manager.findfont(_font_manager.FontProperties(family=_GILL_SANS_NAME), fallback_to_default=False)
matplotlib.rcParams.update(
    {
        "font.family": _GILL_SANS_NAME,
        "mathtext.fontset": "custom",
        "mathtext.rm": _GILL_SANS_NAME,
        "mathtext.it": _GILL_SANS_NAME,
        "mathtext.bf": _GILL_SANS_NAME,
        "mathtext.cal": _GILL_SANS_NAME,
        "mathtext.sf": _GILL_SANS_NAME,
        "mathtext.tt": _GILL_SANS_NAME,
    }
)

# ----------------------------
# Reproducibility
# ----------------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Utilities
# ----------------------------
def _as_tuple(x):
    if isinstance(x, str):
        return (x,)
    if np.isscalar(x):
        return (x,)
    return tuple(x)


def _eps_tag(eps: float) -> str:
    return str(eps).replace(".", "p")


def _sanitize_tag(tag: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"-", "_"} ) else "-" for ch in tag).strip("-")


def _float_tag(x: float) -> str:
    return str(x).replace(".", "p")


def _make_run_tag(
    drift_impl: str,
    plan_float64: bool,
    sinkhorn_iters: int,
    eval_n: int,
    hidden: int,
    blocks: int,
    dim_in: int,
    batch_size: int,
    res_scale: float,
    out_init_std: float | None,
    lr_schedule: str,
) -> str:
    parts: list[str] = []
    if drift_impl.lower() == "plain":
        parts.append("planf64" if plan_float64 else "planf32")
    parts.append(f"sinkhorn{sinkhorn_iters}")
    parts.append(f"evaln{eval_n}")
    parts.append(f"h{hidden}")
    parts.append(f"b{blocks}")
    parts.append(f"din{dim_in}")
    parts.append(f"bs{batch_size}")
    if res_scale != 1.0:
        parts.append(f"resscale{_float_tag(res_scale)}")
    if out_init_std is not None:
        parts.append(f"outstd{_float_tag(out_init_std)}")
    if lr_schedule and lr_schedule != "none":
        parts.append(f"lrs_{lr_schedule}")
    return _sanitize_tag("_".join(parts))


def _init_run_dir(
    out_root: str,
    run_name: str | None,
    args: argparse.Namespace,
    run_tag: str,
    extra_config: dict,
) -> str:
    os.makedirs(out_root, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]

    if run_name:
        run_slug = _sanitize_tag(run_name)
    else:
        desc_parts = [
            f"impl_{args.drift_impl}",
            f"{run_tag}",
            f"steps_{args.steps}",
            f"lr_{args.lr}",
        ]
        if args.device:
            desc_parts.append(f"dev_{args.device}")
        run_slug = _sanitize_tag("_".join(desc_parts))

    dir_name = f"{ts}_{run_slug}_{uid}"
    run_dir = os.path.join(out_root, dir_name)
    os.makedirs(run_dir, exist_ok=False)

    cmd_path = os.path.join(run_dir, "cmd.txt")
    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write(" ".join(shlex.quote(a) for a in sys.argv) + "\n")

    cfg = {
        "created_at": ts,
        "run_dir": run_dir,
        "run_tag": run_tag,
        "args": vars(args),
        "env": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        **extra_config,
    }
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    print(f"Run dir: {run_dir}")
    return run_dir


def _save_logs_json(run_dir: str, logs: Dict[Tuple[str, str, float], Dict]) -> None:
    records: list[dict] = []
    for (target, method, eps), payload in logs.items():
        records.append(
            {
                "target": target,
                "method": method,
                "eps": float(eps),
                "log": payload,
            }
        )
    out_path = os.path.join(run_dir, "logs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved: {out_path}")


# ----------------------------
# Toy 2D target distributions
# ----------------------------
def make_moons(n: int, noise: float = 0.08, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.random(n) * math.pi
    x1 = np.stack([np.cos(t), np.sin(t)], axis=1)
    x2 = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], axis=1)
    x = np.concatenate([x1, x2], axis=0)
    x = x[rng.permutation(len(x))][:n]
    x += rng.normal(scale=noise, size=x.shape)
    return x.astype(np.float32)


def make_spiral(n: int, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.random(n) * 4 * math.pi
    r = t / (4 * math.pi)
    x = r * np.cos(t)
    y = r * np.sin(t)
    pts = np.stack([x, y], axis=1)
    pts = 2.5 * pts
    pts += rng.normal(scale=noise, size=pts.shape)
    return pts.astype(np.float32)


def make_8gaussians(n: int, std: float = 0.08, radius: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    idx = rng.integers(0, 8, size=n)
    x = centers[idx] + rng.normal(scale=std, size=(n, 2))
    return x.astype(np.float32)


def make_checkerboard(n: int, n_tiles: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.random(n) * n_tiles
    y = rng.random(n) * n_tiles
    keep = ((np.floor(x) + np.floor(y)) % 2) == 0
    x, y = x[keep], y[keep]
    while len(x) < n:
        x2 = rng.random(n) * n_tiles
        y2 = rng.random(n) * n_tiles
        keep2 = ((np.floor(x2) + np.floor(y2)) % 2) == 0
        x = np.concatenate([x, x2[keep2]])
        y = np.concatenate([y, y2[keep2]])
    x, y = x[:n], y[:n]
    pts = np.stack([x, y], axis=1)
    pts = (pts / n_tiles) * 4.0 - 2.0
    return pts.astype(np.float32)


def sample_target(name: str, n: int, seed: int) -> np.ndarray:
    lname = name.lower()
    if lname == "moons":
        return make_moons(n, seed=seed)
    if lname == "spiral":
        return make_spiral(n, seed=seed)
    if lname in ("8gaussians", "8-gaussians", "eightgaussians"):
        return make_8gaussians(n, seed=seed)
    if lname == "checkerboard":
        return make_checkerboard(n, seed=seed)
    raise ValueError(f"Unknown target: {name}")


# ----------------------------
# Residual MLP with optional LayerNorm
# ----------------------------
'''
What I changed :

Model / MLP:
- Increased capacity via 'hidden' and 'blocks' (may help multimodal targets like 8-Gaussians / Checkerboard).
- Decoupled latent noise dimension from the 2D output: 'dim_in' can be 16/32/... and the network maps 'dim_in -> 2'.
- Added residual scaling to improve stability and reduce late-stage divergence.
    x + h --> x + res_scale * h

Training:
- Exposed 'lr', 'batch_size', and 'lr_schedule'
'''
class ResBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, normalize: bool = True, res_scale: float = 1.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln1 = nn.LayerNorm(hidden) if normalize else nn.Identity()
        self.ln2 = nn.LayerNorm(dim) if normalize else nn.Identity()
        self.res_scale = float(res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.ln1(h)
        h = F.silu(h)
        h = self.fc2(h)
        h = self.ln2(h)
        return x + self.res_scale * h


class ResidualMLP(nn.Module):
    def __init__(
        self,
        dim_in: int = 2,
        dim_h: int = 128,
        n_blocks: int = 4,
        normalize: bool = True,
        dim_out: int = 2,
        res_scale: float = 1.0,
        out_init_std: float | None = None,
    ):
        super().__init__()
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.inp = nn.Linear(dim_in, dim_h)
        self.in_ln = nn.LayerNorm(dim_h) if normalize else nn.Identity()
        self.blocks = nn.ModuleList(
            [ResBlock(dim_h, dim_h * 2, normalize=normalize, res_scale=res_scale) for _ in range(n_blocks)]
        )
        self.out = nn.Linear(dim_h, dim_out)
        if out_init_std is not None:
            nn.init.normal_(self.out.weight, mean=0.0, std=float(out_init_std))
            nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.inp(z)
        h = self.in_ln(h)
        h = F.silu(h)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


# ----------------------------
# Drift mechanisms
# ----------------------------
def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x**2).sum(dim=1, keepdim=True)
    y2 = (y**2).sum(dim=1, keepdim=True).t()
    return x2 - 2 * x @ y.t() + y2


def pairwise_dists(x: torch.Tensor, y: torch.Tensor, metric: str = "l2_sq") -> torch.Tensor:
    """
    Pairwise distance: 'l2_sq' for ||x-y||^2, 'l2' for ||x-y||.
    """
    d2 = pairwise_sq_dists(x, y)
    if metric == "l2_sq":
        return d2
    if metric == "l2":
        return torch.sqrt(d2.clamp_min(1e-24))
    raise ValueError(f"Unknown dist_metric: {metric}")


# Plain-space implementation (professor version)
def plan_one_sided_plain(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    use_float64: bool = False,
    dist_metric: str = "l2_sq",
) -> torch.Tensor:
    out_dtype = x.dtype
    if use_float64:
        x = x.to(torch.float64)
        y = y.to(torch.float64)
    d = pairwise_dists(x, y, metric=dist_metric)
    p = torch.softmax(-d / eps, dim=1)
    return p.to(out_dtype)


def plan_two_sided_plain(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    use_float64: bool = False,
    dist_metric: str = "l2_sq",
) -> torch.Tensor:
    out_dtype = x.dtype
    if use_float64:
        x = x.to(torch.float64)
        y = y.to(torch.float64)

    # Keep regularization identical across dtypes so "float64" only changes
    # arithmetic precision, not the effective algorithm smoothness.
    tiny = 1e-12
    tiny_sqrt = 1e-24

    d = pairwise_dists(x, y, metric=dist_metric)
    k = torch.exp(-d / eps)
    a_row = k / (k.sum(dim=1, keepdim=True) + tiny)
    a_col = k / (k.sum(dim=0, keepdim=True) + tiny)
    a = torch.sqrt(a_row * a_col + tiny_sqrt)
    a = a / (a.sum(dim=1, keepdim=True) + tiny)
    return a.to(out_dtype)


def plan_sinkhorn_plain(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    iters: int = 30,
    use_float64: bool = False,
    dist_metric: str = "l2_sq",
) -> torch.Tensor:
    out_dtype = x.dtype
    if use_float64:
        x = x.to(torch.float64)
        y = y.to(torch.float64)

    # Same epsilon floor across dtypes; float64 is precision-only.
    tiny = 1e-12

    n, m = x.shape[0], y.shape[0]
    d = pairwise_dists(x, y, metric=dist_metric)
    k = torch.exp(-d / eps).clamp_min(tiny)

    r = torch.full((n,), 1.0 / n, device=x.device, dtype=x.dtype)
    c = torch.full((m,), 1.0 / m, device=x.device, dtype=x.dtype)

    u = torch.ones_like(r)
    v = torch.ones_like(c)
    for _ in range(iters):
        u = r / (k @ v + tiny)
        v = c / (k.t() @ u + tiny)

    p = (u[:, None] * k) * v[None, :]
    p = p / (p.sum(dim=1, keepdim=True) + tiny)
    return p.to(out_dtype)


# Log-space implementation (stable version)
def plan_one_sided_log(x: torch.Tensor, y: torch.Tensor, eps: float, dist_metric: str = "l2_sq") -> torch.Tensor:
    d = pairwise_dists(x, y, metric=dist_metric)
    logits = -d / eps
    log_t = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return torch.exp(log_t)


def plan_two_sided_log(x: torch.Tensor, y: torch.Tensor, eps: float, dist_metric: str = "l2_sq") -> torch.Tensor:
    d = pairwise_dists(x, y, metric=dist_metric)
    logits = -d / eps
    log_row = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    log_col = logits - torch.logsumexp(logits, dim=0, keepdim=True)
    log_t = 0.5 * (log_row + log_col)
    # Match plain-space behavior:
    # 1) row-normalize in log-space
    # 2) exponentiate
    # 3) add small floor and row-renormalize in normal space
    log_t = log_t - torch.logsumexp(log_t, dim=1, keepdim=True)
    t = torch.exp(log_t)
    t = t + 1e-24
    t = t / (t.sum(dim=1, keepdim=True) + 1e-12)
    return t


def plan_sinkhorn_log(x: torch.Tensor, y: torch.Tensor, eps: float, iters: int = 30, dist_metric: str = "l2_sq") -> torch.Tensor:
    d = pairwise_dists(x, y, metric=dist_metric)
    log_t = -d / eps
    for _ in range(iters):
        log_t = log_t - torch.logsumexp(log_t, dim=1, keepdim=True)
        log_t = log_t - torch.logsumexp(log_t, dim=0, keepdim=True)
    # Important for barycentric drift usage: enforce row-stochastic plan.
    t = torch.exp(log_t)
    t = t / (t.sum(dim=1, keepdim=True) + 1e-12)
    return t


def compute_drift(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    drift_type: str,
    sinkhorn_iters: int = 30,
    repulsion_mask_diag: bool = True,
    drift_impl: str = "plain",
    plan_float64: bool = False,
    dist_metric: str = "l2_sq",
) -> torch.Tensor:
    """
    Drift: V(x) = V_attr(x,y) - V_rep(x,x)
      V_attr = bary(P_xy) - x
      V_rep  = bary(P_xx) - x  (with diagonal masking for one/two-sided)

    drift_impl:
      - "plain": professor normal-space implementation
      - "log":   log-space implementation
    """
    x_dtype = x.dtype
    dtype = drift_type.lower()
    impl = drift_impl.lower()

    if impl not in {"plain", "log"}:
        raise ValueError(f"Unknown drift_impl: {drift_impl}")

    if impl == "plain":
        if dtype == "one-sided":
            pxy = plan_one_sided_plain(x, y, eps, use_float64=plan_float64, dist_metric=dist_metric)
            pxx = plan_one_sided_plain(x, x, eps, use_float64=plan_float64, dist_metric=dist_metric)
        elif dtype == "two-sided":
            pxy = plan_two_sided_plain(x, y, eps, use_float64=plan_float64, dist_metric=dist_metric)
            pxx = plan_two_sided_plain(x, x, eps, use_float64=plan_float64, dist_metric=dist_metric)
        elif dtype == "sinkhorn":
            pxy = plan_sinkhorn_plain(x, y, eps, iters=sinkhorn_iters, use_float64=plan_float64, dist_metric=dist_metric)
            pxx = plan_sinkhorn_plain(x, x, eps, iters=sinkhorn_iters, use_float64=plan_float64, dist_metric=dist_metric)
        else:
            raise ValueError(f"Unknown drift_type: {drift_type}")
    else:
        if dtype == "one-sided":
            pxy = plan_one_sided_log(x, y, eps, dist_metric=dist_metric)
            pxx = plan_one_sided_log(x, x, eps, dist_metric=dist_metric)
        elif dtype == "two-sided":
            pxy = plan_two_sided_log(x, y, eps, dist_metric=dist_metric)
            pxx = plan_two_sided_log(x, x, eps, dist_metric=dist_metric)
        elif dtype == "sinkhorn":
            pxy = plan_sinkhorn_log(x, y, eps, iters=sinkhorn_iters, dist_metric=dist_metric)
            pxx = plan_sinkhorn_log(x, x, eps, iters=sinkhorn_iters, dist_metric=dist_metric)
        else:
            raise ValueError(f"Unknown drift_type: {drift_type}")

    # Keep the professor mask rule: one/two-sided mask, sinkhorn no mask.
    tiny = 1e-12
    if repulsion_mask_diag and dtype != "sinkhorn":
        n = x.shape[0]
        mask = torch.ones((n, n), device=x.device, dtype=pxx.dtype)
        mask.fill_diagonal_(0.0)
        pxx = pxx * mask
        pxx = pxx / (pxx.sum(dim=1, keepdim=True) + tiny)

    # Keep barycentric computation in plan dtype for plain+float64 path.
    x_bary = x.to(pxy.dtype)
    y_bary = y.to(pxy.dtype)

    bary_xy = pxy @ y_bary
    bary_xx = pxx @ x_bary
    v_attr = bary_xy - x_bary
    v_rep = bary_xx - x_bary
    return (v_attr - v_rep).to(x_dtype)


# ----------------------------
# EMD via POT (emd2)
# ----------------------------
@torch.no_grad()
def emd_pot(x: torch.Tensor, y: torch.Tensor) -> float:
    if ot is None:
        raise RuntimeError('POT (package "pot", import name "ot") is not installed; cannot compute EMD.')
    x_np = x.detach().cpu().numpy().astype(np.float64)
    y_np = y.detach().cpu().numpy().astype(np.float64)
    n = x_np.shape[0]
    m = y_np.shape[0]
    a = np.full((n,), 1.0 / n, dtype=np.float64)
    b = np.full((m,), 1.0 / m, dtype=np.float64)
    cost = ot.dist(x_np, y_np, metric="euclidean") ** 2
    return float(ot.emd2(a, b, cost))


# ----------------------------
# Training
# ----------------------------
@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    dim_in: int = 2   # noise dimension
    dim_out: int = 2  # data dimension (toy targets are 2D)
    hidden: int = 128
    blocks: int = 4
    normalize: bool = True
    res_scale: float = 1.0
    out_init_std: float | None = None

    # training
    steps: int = 2000
    batch_size: int = 512
    lr: float = 2e-4
    lr_schedule: str = "none"  # none | cosine | step
    min_lr: float = 0.0
    lr_step_size: int = 500
    lr_gamma: float = 0.5

    # drift
    eps: float = 0.1
    drift_type: str = "one-sided"  # one-sided | two-sided | sinkhorn
    sinkhorn_iters: int = 30
    drift_impl: str = "plain"      # plain | log
    plan_float64: bool = False     # apply float64 inside plain-space plan computations
    dist_metric: str = "l2_sq"    # l2_sq | l2

    # eval
    eval_every: int = 100
    eval_n: int = 192
    eval_warmup_steps: int = 0
    eval_warmup_every: int = 1


def train_one_return_model(target_name: str, cfg: TrainConfig) -> Tuple[Dict, nn.Module]:
    seed_all(cfg.seed)
    device = torch.device(cfg.device)

    model = ResidualMLP(
        dim_in=cfg.dim_in,
        dim_out=cfg.dim_out,
        dim_h=cfg.hidden,
        n_blocks=cfg.blocks,
        normalize=cfg.normalize,
        res_scale=cfg.res_scale,
        out_init_std=cfg.out_init_std,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=cfg.min_lr)
    elif cfg.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)
    else:
        scheduler = None

    logs = {
        "config": asdict(cfg),
        "target": target_name,
        "emd2": [],  # list[(step, emd2)]
        "loss": [],  # list[(step, loss)]
    }
    warned_no_ot = False
    g_train = torch.Generator(device=device)
    g_train.manual_seed(cfg.seed)
    g_eval = torch.Generator(device=device)
    g_eval.manual_seed(cfg.seed + 999)

    for step in range(1, cfg.steps + 1):
        y_np = sample_target(target_name, cfg.batch_size, seed=cfg.seed + step)
        y = torch.from_numpy(y_np).to(device)

        z = torch.randn(cfg.batch_size, cfg.dim_in, device=device, generator=g_train)
        x = model(z)

        with torch.no_grad():
            v = compute_drift(
                x=x,
                y=y,
                eps=cfg.eps,
                drift_type=cfg.drift_type,
                sinkhorn_iters=cfg.sinkhorn_iters,
                repulsion_mask_diag=True,
                drift_impl=cfg.drift_impl,
                plan_float64=cfg.plan_float64,
                dist_metric=cfg.dist_metric,
            )
            x_target = x + v

        loss = F.mse_loss(x, x_target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        if step % 50 == 0:
            logs["loss"].append((step, float(loss.item())))

        if cfg.eval_warmup_steps > 0 and step <= cfg.eval_warmup_steps:
            eval_period = cfg.eval_warmup_every
        else:
            eval_period = cfg.eval_every

        if eval_period > 0 and step % eval_period == 0:
            if ot is None:
                if not warned_no_ot:
                    print('Warning: skipping EMD eval because POT (import name "ot") is not installed.')
                    warned_no_ot = True
                continue
            z_eval = torch.randn(cfg.eval_n, cfg.dim_in, device=device, generator=g_eval)
            x_eval = model(z_eval)
            y_eval_np = sample_target(target_name, cfg.eval_n, seed=cfg.seed + 10_000 + step)
            y_eval = torch.from_numpy(y_eval_np).to(device)
            emd2_val = emd_pot(x_eval, y_eval)
            logs["emd2"].append((step, emd2_val))
            print(
                f"[{target_name:12s}] {cfg.drift_type:9s} eps={cfg.eps:<6g} "
                f"impl={cfg.drift_impl:5s} step={step:5d} "
                f"loss={loss.item():.4f} emd2={emd2_val:.4f}"
            )

    return logs, model


def train_one(target_name: str, cfg: TrainConfig) -> Dict:
    logs, _ = train_one_return_model(target_name, cfg)
    return logs


def compare_all_and_return_models(
    targets=("Moons", "Spiral", "8-Gaussians", "Checkerboard"),
    eps_list=(1.0, 0.1, 0.01),
    methods=("one-sided", "two-sided", "sinkhorn"),
    steps: int = 2000,
    batch_size: int = 512,
    lr: float = 2e-4,
    lr_schedule: str = "none",
    min_lr: float = 0.0,
    lr_step_size: int = 500,
    lr_gamma: float = 0.5,
    eval_every: int = 100,
    eval_n: int = 192,
    eval_warmup_steps: int = 0,
    eval_warmup_every: int = 1,
    sinkhorn_iters: int = 30,
    normalize: bool = True,
    seed: int = 42,
    device: str | None = None,
    hidden: int = 128,
    blocks: int = 4,
    dim_in: int = 2,
    res_scale: float = 1.0,
    out_init_std: float | None = None,
    drift_impl: str = "plain",
    plan_float64: bool = False,
    dist_metric: str = "l2_sq",
    return_logs: bool = False,
):
    targets = _as_tuple(targets)
    eps_list = tuple(float(e) for e in _as_tuple(eps_list))
    methods = _as_tuple(methods)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    trained_models: Dict[Tuple[str, str, float], nn.Module] = {}
    trained_logs: Dict[Tuple[str, str, float], Dict] = {}

    for target in targets:
        for eps in eps_list:
            for method in methods:
                cfg = TrainConfig(
                    seed=seed,
                    device=device,
                    dim_in=dim_in,
                    dim_out=2,
                    hidden=hidden,
                    blocks=blocks,
                    normalize=normalize,
                    res_scale=res_scale,
                    out_init_std=out_init_std,
                    steps=steps,
                    batch_size=batch_size,
                    lr=lr,
                    lr_schedule=lr_schedule,
                    min_lr=min_lr,
                    lr_step_size=lr_step_size,
                    lr_gamma=lr_gamma,
                    eps=eps,
                    drift_type=method,
                    sinkhorn_iters=sinkhorn_iters,
                    drift_impl=drift_impl,
                    plan_float64=plan_float64,
                    dist_metric=dist_metric,
                    eval_every=eval_every,
                    eval_n=eval_n,
                    eval_warmup_steps=eval_warmup_steps,
                    eval_warmup_every=eval_warmup_every,
                )
                logs, model = train_one_return_model(target, cfg)
                key = (target, method, eps)
                trained_models[key] = model
                trained_logs[key] = logs

    if return_logs:
        return trained_models, trained_logs
    return trained_models


# ----------------------------
# Plotting
# ----------------------------
@torch.no_grad()
def _plot_generated_source_grids(
    models: Dict[Tuple[str, str, float], nn.Module],
    eps_list=(1.0, 0.1, 0.01),
    targets=("Moons", "Spiral", "8-Gaussians", "Checkerboard"),
    methods=("one-sided", "two-sided", "sinkhorn"),
    n_viz: int = 1500,
    seed: int = 123,
    drift_impl: str = "plain",
    steps: int = 2000,
    save_dir: str | None = None,
    tag: str | None = None,
):
    eps_list = tuple(float(e) for e in _as_tuple(eps_list))
    targets = _as_tuple(targets)
    methods = _as_tuple(methods)

    title_fs = 18
    ylabel_fs = 16
    legend_fs = 13
    target_color = "#4F70BE"
    gen_color = "#CAAF78"

    first_model = next(iter(models.values()))
    device = next(first_model.parameters()).device
    z_dim = int(getattr(first_model, "dim_in", first_model.inp.in_features))

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    z = torch.randn(n_viz, z_dim, generator=g).to(device)

    target_cache = {}
    for eps in eps_list:
        for tname in targets:
            y_np = sample_target(tname, n_viz, seed=seed + int(1000 * eps) + hash(tname) % 1000)
            target_cache[(tname, eps)] = torch.from_numpy(y_np).to(device)

    for eps in eps_list:
        fig, axes = plt.subplots(
            len(methods),
            len(targets),
            figsize=(4.5 * len(targets), 3.9 * len(methods)),
            squeeze=False,
        )

        # Precompute generated points so we can:
        # 1) keep consistent view limits within each column
        # 2) keep the subplot boxes aligned (no per-axes box shrinking)
        gen_cache: dict[tuple[str, str], torch.Tensor] = {}
        col_limits: dict[str, tuple[float, float, float, float]] = {}
        for tname in targets:
            y = target_cache[(tname, eps)]
            pts = [y]
            for method in methods:
                key = (tname, method, eps)
                if key not in models:
                    raise KeyError(f"Model not found for key {key}.")
                model = models[key]
                model.eval()
                x_plot = model(z)
                gen_cache[(method, tname)] = x_plot
                pts.append(x_plot)

            all_pts = torch.cat(pts, dim=0)
            x0 = float(all_pts[:, 0].min().item())
            x1 = float(all_pts[:, 0].max().item())
            y0 = float(all_pts[:, 1].min().item())
            y1 = float(all_pts[:, 1].max().item())
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            half = 0.5 * max(x1 - x0, y1 - y0)
            half = max(half * 1.08, 1e-6)
            col_limits[tname] = (cx - half, cx + half, cy - half, cy + half)

        for i, method in enumerate(methods):
            for j, tname in enumerate(targets):
                ax = axes[i, j]
                y = target_cache[(tname, eps)]
                x_plot = gen_cache[(method, tname)]

                ax.scatter(
                    y[:, 0].detach().cpu().numpy(),
                    y[:, 1].detach().cpu().numpy(),
                    s=7,
                    alpha=0.9,
                    color=target_color,
                    label="Target",
                )
                ax.scatter(
                    x_plot[:, 0].detach().cpu().numpy(),
                    x_plot[:, 1].detach().cpu().numpy(),
                    s=7,
                    alpha=0.9,
                    color=gen_color,
                    label="Generated",
                )

                if i == 0:
                    ax.set_title(tname, fontsize=title_fs)
                if j == 0:
                    ax.set_ylabel(method, fontsize=ylabel_fs)

                xl0, xl1, yl0, yl1 = col_limits[tname]
                ax.set_xlim(xl0, xl1)
                ax.set_ylim(yl0, yl1)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([])
                ax.set_yticks([])

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            frameon=False,
            fontsize=legend_fs,
        )
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.03, top=0.97, wspace=0.10, hspace=0.10)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            tag_str = f"_tag_{_sanitize_tag(tag)}" if tag else ""
            fname = f"generated_grid_eps_{_eps_tag(eps)}_impl_{drift_impl}_steps_{steps}{tag_str}.png"
            out_path = os.path.join(save_dir, fname)
            fig.savefig(out_path, dpi=200)
            print(f"Saved: {out_path}")
        plt.show()


def plot_generated_source_grids(
    models: Dict[Tuple[str, str, float], nn.Module],
    eps_list=(1.0, 0.1, 0.01),
    targets=("Moons", "Spiral", "8-Gaussians", "Checkerboard"),
    methods=("one-sided", "two-sided", "sinkhorn"),
    n_viz=1500,
    seed=123,
    save_dir=None,
    steps: int = 2000,
    drift_impl: str = "plain",
    tag: str | None = None,
):
    _plot_generated_source_grids(
        models=models,
        eps_list=eps_list,
        targets=targets,
        methods=methods,
        n_viz=n_viz,
        seed=seed,
        drift_impl=drift_impl,
        steps=steps,
        save_dir=save_dir,
        tag=tag,
    )


def plot_emd_1x4(
    logs: Dict[Tuple[str, str, float], Dict],
    eps_list=(1.0, 0.1, 0.01),
    targets=("Moons", "Spiral", "8-Gaussians", "Checkerboard"),
    methods=("one-sided", "two-sided", "sinkhorn"),
    save_dir: str | None = None,
    steps: int = 2000,
    drift_impl: str = "plain",
    tag: str | None = None,
):
    eps_list = tuple(float(e) for e in _as_tuple(eps_list))
    targets = _as_tuple(targets)
    methods = _as_tuple(methods)

    method_colors = {
        "one-sided": "#83C4CE",
        "two-sided": "#F3CE7F",
        "sinkhorn": "#B56C73",
    }
    method_styles = {
        "one-sided": ":",
        "two-sided": "--",
        "sinkhorn": "-",
    }

    title_fs = 20
    label_fs = 18
    tick_fs = 14
    legend_fs = 13

    for eps in eps_list:
        fig, axes = plt.subplots(1, len(targets), figsize=(4.6 * len(targets), 3.8), sharey=False)
        if len(targets) == 1:
            axes = [axes]

        for ci, tname in enumerate(targets):
            ax = axes[ci]
            ymax = 0.0
            for method in methods:
                key = (tname, method, eps)
                if key not in logs:
                    raise KeyError(f"Logs not found for key {key}.")
                emd_pairs = logs[key]["emd2"]
                if not emd_pairs:
                    continue
                x = [p[0] for p in emd_pairs]
                y = [p[1] for p in emd_pairs]
                ymax = max(ymax, float(max(y)))
                ax.plot(
                    x,
                    y,
                    linewidth=2.4,
                    alpha=1.0,
                    color=method_colors.get(method, None),
                    linestyle=method_styles.get(method, "-"),
                    label=method,
                )

            ax.set_title(tname, fontsize=title_fs)
            ax.set_xlabel("Iteration", fontsize=label_fs)
            if ci == 0:
                ax.set_ylabel(r"$W_2^2$", fontsize=label_fs)

            ax.margins(x=0.0, y=0.0)
            ax.set_ylim(bottom=0.0, top=max(ymax * 1.05, 0.1))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=tick_fs)
            legend_loc = "lower right" if ("gauss" in tname.lower() and "8" in tname) else "upper right"
            ax.legend(loc=legend_loc, frameon=False, fontsize=legend_fs, handlelength=2.4)

        fig.suptitle(f"EMD vs Iteration (1x{len(targets)}) — eps={eps}, impl={drift_impl}, steps={steps}", y=1.04)
        fig.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            tag_str = f"_tag_{_sanitize_tag(tag)}" if tag else ""
            fname = f"emd_1x{len(targets)}_eps_{_eps_tag(eps)}_impl_{drift_impl}_steps_{steps}{tag_str}.png"
            out_path = os.path.join(save_dir, fname)
            fig.savefig(out_path, dpi=220)
            print(f"Saved: {out_path}")
        plt.show()


def plot_generated_and_emd(
    models: Dict[Tuple[str, str, float], nn.Module],
    logs: Dict[Tuple[str, str, float], Dict],
    eps_list=(1.0, 0.1, 0.01),
    targets=("Moons", "Spiral", "8-Gaussians", "Checkerboard"),
    methods=("one-sided", "two-sided", "sinkhorn"),
    n_viz: int = 1500,
    seed: int = 123,
    save_dir: str | None = None,
    steps: int = 2000,
    drift_impl: str = "plain",
    tag: str | None = None,
):
    """
    Composite figure (per-eps):
      - Top: generated-vs-target grid (methods x targets)
      - Bottom: EMD curves (1 row x targets)
    """
    eps_list = tuple(float(e) for e in _as_tuple(eps_list))
    targets = _as_tuple(targets)
    methods = _as_tuple(methods)

    title_fs = 20
    ylabel_fs = 16
    legend_fs = 13
    tick_fs = 14
    label_fs = 18
    target_color = "#4F70BE"
    gen_color = "#CAAF78"

    method_colors = {
        "one-sided": "#83C4CE",
        "two-sided": "#F3CE7F",
        "sinkhorn": "#B56C73",
    }
    method_styles = {
        "one-sided": ":",
        "two-sided": "--",
        "sinkhorn": "-",
    }

    first_model = next(iter(models.values()))
    device = next(first_model.parameters()).device
    z_dim = int(getattr(first_model, "dim_in", first_model.inp.in_features))

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    z = torch.randn(n_viz, z_dim, generator=g).to(device)

    target_cache = {}
    for eps in eps_list:
        for tname in targets:
            y_np = sample_target(tname, n_viz, seed=seed + int(1000 * eps) + hash(tname) % 1000)
            target_cache[(tname, eps)] = torch.from_numpy(y_np).to(device)

    for eps in eps_list:
        nrows = len(methods) + 1
        fig_h = 3.8 * len(methods) + 3.1
        fig, axes = plt.subplots(
            nrows,
            len(targets),
            figsize=(4.5 * len(targets), fig_h),
            squeeze=False,
            gridspec_kw={"height_ratios": [1.0] * len(methods) + [0.9], "hspace": 0.25, "wspace": 0.16},
        )

        # Precompute generated points and limits per column.
        gen_cache: dict[tuple[str, str], torch.Tensor] = {}
        col_limits: dict[str, tuple[float, float, float, float]] = {}
        for tname in targets:
            y = target_cache[(tname, eps)]
            pts = [y]
            for method in methods:
                key = (tname, method, eps)
                if key not in models:
                    raise KeyError(f"Model not found for key {key}.")
                model = models[key]
                model.eval()
                x_plot = model(z)
                gen_cache[(method, tname)] = x_plot
                pts.append(x_plot)

            all_pts = torch.cat(pts, dim=0)
            x0 = float(all_pts[:, 0].min().item())
            x1 = float(all_pts[:, 0].max().item())
            y0 = float(all_pts[:, 1].min().item())
            y1 = float(all_pts[:, 1].max().item())
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            half = 0.5 * max(x1 - x0, y1 - y0)
            half = max(half * 1.08, 1e-6)
            col_limits[tname] = (cx - half, cx + half, cy - half, cy + half)

        # Top grid: scatter plots
        for i, method in enumerate(methods):
            for j, tname in enumerate(targets):
                ax = axes[i, j]
                y = target_cache[(tname, eps)]
                x_plot = gen_cache[(method, tname)]

                ax.scatter(
                    y[:, 0].detach().cpu().numpy(),
                    y[:, 1].detach().cpu().numpy(),
                    s=7,
                    alpha=0.9,
                    color=target_color,
                    label="Target",
                )
                ax.scatter(
                    x_plot[:, 0].detach().cpu().numpy(),
                    x_plot[:, 1].detach().cpu().numpy(),
                    s=7,
                    alpha=0.9,
                    color=gen_color,
                    label="Generated",
                )

                if i == 0:
                    ax.set_title(tname, fontsize=title_fs)
                if j == 0:
                    ax.set_ylabel(method, fontsize=ylabel_fs)

                xl0, xl1, yl0, yl1 = col_limits[tname]
                ax.set_xlim(xl0, xl1)
                ax.set_ylim(yl0, yl1)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([])
                ax.set_yticks([])

        # Bottom row: EMD curves
        for j, tname in enumerate(targets):
            ax = axes[-1, j]
            ymax = 0.0
            for method in methods:
                key = (tname, method, eps)
                if key not in logs:
                    raise KeyError(f"Logs not found for key {key}.")
                emd_pairs = logs[key]["emd2"]
                if not emd_pairs:
                    continue
                x = [p[0] for p in emd_pairs]
                y = [p[1] for p in emd_pairs]
                ymax = max(ymax, float(max(y)))
                ax.plot(
                    x,
                    y,
                    linewidth=2.4,
                    alpha=1.0,
                    color=method_colors.get(method, None),
                    linestyle=method_styles.get(method, "-"),
                    label=method,
                )

            ax.margins(x=0.0, y=0.0)
            ax.set_ylim(bottom=0.0, top=max(ymax * 1.05, 0.1))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=tick_fs)
            ax.set_xlabel("Iteration", fontsize=label_fs)
            if j == 0:
                ax.set_ylabel(r"$W_2^2$", fontsize=label_fs)
            legend_loc = "lower right" if ("gauss" in tname.lower() and "8" in tname) else "upper right"
            ax.legend(loc=legend_loc, frameon=False, fontsize=legend_fs, handlelength=2.4)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            frameon=False,
            fontsize=legend_fs,
        )
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.05, top=0.97, wspace=0.10, hspace=0.12)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            tag_str = f"_tag_{_sanitize_tag(tag)}" if tag else ""
            fname = f"combined_grid_emd_eps_{_eps_tag(eps)}_impl_{drift_impl}_steps_{steps}{tag_str}.png"
            out_path = os.path.join(save_dir, fname)
            fig.savefig(out_path, dpi=220)
            print(f"Saved: {out_path}")
        plt.show()


# ----------------------------
# Main
# ----------------------------
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drift-based generative modeling comparison.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Torch device string, e.g. "cpu", "cuda", "cuda:0", "cuda:1". Default: auto.',
    )
    parser.add_argument(
        "--drift-impl",
        "--drift_impl",
        dest="drift_impl",
        type=str,
        choices=("plain", "log"),
        default="log",
        help='Drift implementation: "plain" or "log".',
    )
    parser.add_argument(
        "--sinkhorn-iters",
        "--sinkhorn_iters",
        dest="sinkhorn_iters",
        type=int,
        default=1000,
        help="Sinkhorn iterations used during training when drift_type='sinkhorn'.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="Moons,Spiral,8-Gaussians,Checkerboard",
        help='Comma-separated targets (default: "Moons,Spiral,8-Gaussians,Checkerboard").',
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="one-sided,two-sided,sinkhorn",
        help='Comma-separated drift methods (default: "one-sided,two-sided,sinkhorn").',
    )
    parser.add_argument(
        "--eps-list",
        "--eps_list",
        dest="eps_list",
        type=str,
        default="0.01",
        help='Comma-separated eps values (default: "0.01").',
    )
    parser.add_argument("--steps", type=int, default=20000, help="Training steps.")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=512, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument(
        "--lr-schedule",
        "--lr_schedule",
        dest="lr_schedule",
        choices=("none", "cosine", "step"),
        default="none",
        help="Learning rate schedule (default: none).",
    )
    parser.add_argument("--min-lr", "--min_lr", dest="min_lr", type=float, default=0.0, help="Cosine schedule minimum LR.")
    parser.add_argument(
        "--lr-step-size",
        "--lr_step_size",
        dest="lr_step_size",
        type=int,
        default=500,
        help="StepLR step size (in iterations).",
    )
    parser.add_argument("--lr-gamma", "--lr_gamma", dest="lr_gamma", type=float, default=0.5, help="StepLR gamma.")
    parser.add_argument("--hidden", type=int, default=128, help="MLP hidden width.")
    parser.add_argument("--blocks", type=int, default=4, help="Number of residual blocks.")
    parser.add_argument(
        "--dim-in",
        "--dim_in",
        dest="dim_in",
        type=int,
        default=2,
        help="Noise input dimension (model output stays 2D).",
    )
    parser.add_argument(
        "--res-scale",
        "--res_scale",
        dest="res_scale",
        type=float,
        default=1.0,
        help="Residual scaling applied in each ResBlock (default: 1.0).",
    )
    parser.add_argument(
        "--out-init-std",
        "--out_init_std",
        dest="out_init_std",
        type=float,
        default=None,
        help="If set, init output layer weight with N(0,std) and zero bias.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=100, help="Eval period.")
    parser.add_argument(
        "--eval-warmup-steps",
        "--eval_warmup_steps",
        dest="eval_warmup_steps",
        type=int,
        default=0,
        help="If >0, run EMD eval more frequently for the first N steps (default: 0).",
    )
    parser.add_argument(
        "--eval-warmup-every",
        "--eval_warmup_every",
        dest="eval_warmup_every",
        type=int,
        default=1,
        help="Warmup eval period when step<=eval_warmup_steps (default: 1, i.e. every step).",
    )
    parser.add_argument("--eval-n", "--eval_n", dest="eval_n", type=int, default=1000, help="EMD eval sample count.")
    parser.add_argument(
        "--plan-float64",
        "--plan_float64",
        dest="plan_float64",
        action="store_true",
        help="Use float64 for plain-space plan computations (one-/two-sided and sinkhorn).",
    )
    parser.add_argument(
        "--dist-metric",
        "--dist_metric",
        dest="dist_metric",
        type=str,
        choices=("l2_sq", "l2"),
        default="l2_sq",
        help='Distance metric: "l2_sq" for ||x-y||^2 (Gaussian kernel), "l2" for ||x-y|| (Laplacian kernel). Default: l2_sq.',
    )
    parser.add_argument(
        "--out-root",
        dest="out_root",
        type=str,
        default="runs",
        help="Root directory to create a unique per-run output folder (default: runs).",
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        type=str,
        default=None,
        help="Optional name to include in the per-run output folder name.",
    )
    parser.add_argument(
        "--no-run-dir",
        dest="no_run_dir",
        action="store_true",
        help='Disable per-run output folders (writes to "viz_grids" in the current directory).',
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    eps_list = tuple(float(e.strip()) for e in args.eps_list.split(",") if e.strip())
    batch_size = args.batch_size
    eval_every = args.eval_every
    eval_warmup_steps = args.eval_warmup_steps
    eval_warmup_every = args.eval_warmup_every

    run_tag = _make_run_tag(
        drift_impl=args.drift_impl,
        plan_float64=args.plan_float64,
        sinkhorn_iters=args.sinkhorn_iters,
        eval_n=args.eval_n,
        hidden=args.hidden,
        blocks=args.blocks,
        dim_in=args.dim_in,
        batch_size=batch_size,
        res_scale=args.res_scale,
        out_init_std=args.out_init_std,
        lr_schedule=args.lr_schedule,
    )
    if args.no_run_dir:
        run_dir = None
        save_dir = "viz_grids"
    else:
        run_dir = _init_run_dir(
            out_root=args.out_root,
            run_name=args.run_name,
            args=args,
            run_tag=run_tag,
            extra_config={
                "targets": targets,
                "methods": methods,
                "eps_list": list(eps_list),
                "batch_size": batch_size,
                "eval_every": eval_every,
                "eval_warmup_steps": eval_warmup_steps,
                "eval_warmup_every": eval_warmup_every,
            },
        )
        save_dir = os.path.join(run_dir, "viz")

    models, logs = compare_all_and_return_models(
        targets=targets,
        eps_list=eps_list,
        methods=methods,
        steps=args.steps,
        batch_size=batch_size,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        min_lr=args.min_lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        eval_every=eval_every,
        eval_n=args.eval_n,
        eval_warmup_steps=eval_warmup_steps,
        eval_warmup_every=eval_warmup_every,
        sinkhorn_iters=args.sinkhorn_iters,
        normalize=True,
        seed=args.seed,
        device=args.device,
        hidden=args.hidden,
        blocks=args.blocks,
        dim_in=args.dim_in,
        res_scale=args.res_scale,
        out_init_std=args.out_init_std,
        drift_impl=args.drift_impl,
        plan_float64=args.plan_float64,
        dist_metric=args.dist_metric,
        return_logs=True,
    )

    if run_dir is not None:
        _save_logs_json(run_dir, logs)

    # 1) generated source vs target
    plot_generated_source_grids(
        models=models,
        eps_list=eps_list,
        targets=targets,
        methods=methods,
        n_viz=1500,
        seed=123,
        save_dir=save_dir,
        steps=args.steps,
        drift_impl=args.drift_impl,
        tag=run_tag,
    )

    # 2) emd_1x4 style curve figure (POT emd2)
    plot_emd_1x4(
        logs=logs,
        eps_list=eps_list,
        targets=targets,
        methods=methods,
        save_dir=save_dir,
        steps=args.steps,
        drift_impl=args.drift_impl,
        tag=run_tag,
    )

    # 3) combined figure (generated grid + per-target EMD row), saved per-eps
    plot_generated_and_emd(
        models=models,
        logs=logs,
        eps_list=eps_list,
        targets=targets,
        methods=methods,
        n_viz=1500,
        seed=123,
        save_dir=save_dir,
        steps=args.steps,
        drift_impl=args.drift_impl,
        tag=run_tag,
    )


if __name__ == "__main__":
    main()
