"""
Drifting loss + feature extraction for ImageNet experiments.

This module is the math core for the ImageNet implementation of
Generative Modeling via Drifting(Kaiming) and Drifting Sinkhorn(Ours).

Generative Modeling via Drifting alignment
-------------------------
- Appendix Algorithm 2:
  joint drift-field construction implemented in 'alg2_from_distances'.
- Appendix A.5:
  multi-scale feature sets from MAE feature maps and encoder input.
- Appendix A.6:
  feature normalization 'S_j' (Eq. 20-21), drift normalization 'theta_j'
  (Eq. 24-25), and loss Eq. 26.
- Appendix A.7:
  CFG weighting via unconditional negatives and 'w(omega)'.

Drifting Sinkhorn alignment
-------------------
We have:
- 'coupling': 'row', 'partial_two_sided', 'sinkhorn'
- 'drift_form': 'alg2_joint' (Kaiming's paper structure) or 'split' (cross-minus-self, ours)

Important: the Kaiming's paper baseline is:
'drift_form="alg2_joint"' + 'coupling="partial_two_sided"'.

Tensor shapes
-------------
All drifting-loss computations use [N, L, C]:
- N: samples (generated / real)
- L: number of feature vectors per sample (for example spatial positions)
- C: feature dimension
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

import torch
import torch.nn.functional as F
from contextlib import nullcontext


@dataclass(frozen=True)
class FeatureSet:
    """
    A set of per-sample feature vectors.

    Shape convention: [N, L, C]
      - N: number of samples
      - L: number of feature vectors per sample (e.g., spatial locations)
      - C: channel / feature dimension
    """

    name: str
    x: torch.Tensor

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def l(self) -> int:
        return int(self.x.shape[1])

    @property
    def c(self) -> int:
        return int(self.x.shape[2])


def _ensure_nlc(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"{name} must have shape [N,L,C], got {tuple(x.shape)}")
    return x


def feature_sets_from_feature_map(fmap: torch.Tensor, *, prefix: str) -> list[FeatureSet]:
    """
    Multi-scale, multi-location feature vectors (Appendix A.5) from one fmap [N,C,H,W].
    """
    if fmap.ndim != 4:
        raise ValueError(f"Expected fmap [N,C,H,W], got {tuple(fmap.shape)}")
    n, c, h, w = fmap.shape

    # (a) Per-location vectors: H*W vectors of C.
    loc = fmap.permute(0, 2, 3, 1).contiguous().view(n, h * w, c)

    # (b) Global mean/std (C-dim each) => L=2.
    mean = fmap.mean(dim=(2, 3))
    mean_sq = (fmap * fmap).mean(dim=(2, 3))
    # NOTE: we add a small epsilon inside sqrt to avoid NaN gradients when the
    # variance is exactly zero. Without this, d/dx sqrt(x) is infinite at x=0,
    # and autograd can produce 0*inf -> NaN for constant channels.
    std_eps = 1e-6
    std = torch.sqrt((mean_sq - mean * mean).clamp_min(0.0) + std_eps)
    global_ms = torch.stack([mean, std], dim=1)  # [N,2,C]

    out: list[FeatureSet] = [
        FeatureSet(name=f"{prefix}.loc", x=loc),
        FeatureSet(name=f"{prefix}.global_ms", x=global_ms),
    ]

    # (c,d) Patch mean/std over 2x2 and 4x4.
    for p in (2, 4):
        if h % p != 0 or w % p != 0:
            continue
        mean_p = F.avg_pool2d(fmap, kernel_size=p, stride=p)
        mean_sq_p = F.avg_pool2d(fmap * fmap, kernel_size=p, stride=p)
        std_p = torch.sqrt((mean_sq_p - mean_p * mean_p).clamp_min(0.0) + std_eps)
        hp, wp = mean_p.shape[-2:]
        mean_v = mean_p.permute(0, 2, 3, 1).contiguous().view(n, hp * wp, c)
        std_v = std_p.permute(0, 2, 3, 1).contiguous().view(n, hp * wp, c)
        patch_ms = torch.cat([mean_v, std_v], dim=1)  # [N,2*hp*wp,C]
        out.append(FeatureSet(name=f"{prefix}.patch{p}_ms", x=patch_ms))
    return out


def feature_sets_from_encoder_input(x: torch.Tensor) -> list[FeatureSet]:
    """
    Extra input feature: per-channel mean of squared values (Appendix A.5).
    """
    if x.ndim != 4:
        raise ValueError(f"Expected input [N,C,H,W], got {tuple(x.shape)}")
    x2 = (x * x).mean(dim=(2, 3), keepdim=False).unsqueeze(1)  # [N,1,C]
    return [FeatureSet(name="input.x2", x=x2)]


@torch.no_grad()
def extract_feature_sets(
    encoder,
    x: torch.Tensor,
    *,
    every_n_blocks: int = 2,
    include_input_x2: bool = True,
) -> list[FeatureSet]:
    """
    Extract all feature sets used by the drifting loss (Appendix A.5).

    Note: this runs under 'no_grad' by default; for generated samples we should
    compute encoder activations with grad and then call 'feature_sets_from_*'
    directly on those tensors.
    """
    maps = encoder.forward_feature_maps(x, every_n_blocks=every_n_blocks)
    out: list[FeatureSet] = []
    for i, fmap in enumerate(maps):
        out.extend(feature_sets_from_feature_map(fmap, prefix=f"enc{i:02d}"))
    if include_input_x2:
        out.extend(feature_sets_from_encoder_input(x))
    return out


def flatten_latents_as_feature_set(x: torch.Tensor) -> FeatureSet:
    """
    Vanilla drifting loss without a feature encoder: ϖ(x)=x (Appendix A.5).
    """
    if x.ndim != 4:
        raise ValueError(f"Expected latents [N,C,H,W], got {tuple(x.shape)}")
    n = x.shape[0]
    return FeatureSet(name="vanilla.latent", x=x.view(n, 1, -1))


def compute_uncond_weight(omega: torch.Tensor, *, nneg: int, nuncond: int) -> torch.Tensor:
    """
    Compute CFG unconditional weight w(ω) (Appendix A.7).

    Appendix A.7 derives how to weight 'Nuncond' unconditional negatives so that
    the effective negative distribution matches the CFG-strength ω used in the
    main paper (see the comparison to Eq. (15)(16) in Appendix A.7).

    We implement the derived scalar:
      w = (Nneg - 1) * (ω - 1) / Nuncond
    """
    if nneg <= 1:
        raise ValueError("nneg must be > 1 for CFG weighting")
    if nuncond <= 0:
        return torch.zeros_like(omega)
    return (float(nneg - 1) * (omega - 1.0)) / float(nuncond)


DriftForm = Literal["alg2_joint", "split"]
Coupling = Literal["row", "partial_two_sided", "sinkhorn"]
SinkhornMarginal = Literal["none", "weighted_cols"]
DistMetric = Literal["l2", "l2_sq"]


def _pairwise_distance(x: torch.Tensor, y: torch.Tensor, *, metric: DistMetric) -> torch.Tensor:
    """
    Pairwise distance matrix for drift construction.

    - "l2": ||x-y||
    - "l2_sq": ||x-y||^2
    """
    dist = torch.cdist(x, y)
    if metric == "l2":
        return dist
    if metric == "l2_sq":
        return dist * dist
    raise ValueError(f"Unknown dist_metric: {metric}")


def _row_stochastic_from_logits(logits: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Row-stochastic coupling P from logits via softmax.
    """
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    return torch.exp(logits - log_z).clamp_min(0.0)


def _partial_two_sided_from_logits(
    logits: torch.Tensor,
    *,
    eps: float = 1e-12,
    impl: Literal["logspace", "kernel"] = "logspace",
) -> torch.Tensor:
    """
    Partial two-sided normalization used by the paper (Algorithm 2; Appendix A.6).

    With 'K = exp(logits)', this is:
      A = K / sqrt(sum_j K_ij * sum_i K_ij).
    Equivalent view used in the appendix pseudocode:
      A = sqrt(softmax_row(logits) * softmax_col(logits)).

    This is a single-pass approximation to full two-sided balancing (Sinkhorn).
    """
    if impl == "kernel":
        kernel = torch.exp(logits)
        row_sum = kernel.sum(dim=-1, keepdim=True).clamp_min(eps)
        col_sum = kernel.sum(dim=-2, keepdim=True).clamp_min(eps)
        return kernel / (row_sum * col_sum).sqrt()
    if impl == "logspace":
        log_row_sum = torch.logsumexp(logits, dim=-1, keepdim=True)
        log_col_sum = torch.logsumexp(logits, dim=-2, keepdim=True)
        log_a = logits - 0.5 * (log_row_sum + log_col_sum)
        # Important for CFG (Appendix A.7): when w(ω)=0 we set log(w)=-inf,
        # producing all-(-inf) unconditional columns. The log-space formula
        # would otherwise create NaNs via (-inf) - (-inf). Those entries should
        # correspond to exactly zero kernel mass, matching the "kernel" impl.
        log_a = log_a.masked_fill(torch.isneginf(logits), -math.inf)
        return torch.exp(log_a)
    raise ValueError(f"Unknown impl: {impl}")


def _sinkhorn_from_logits(
    logits: torch.Tensor,
    *,
    r: torch.Tensor,
    c: torch.Tensor,
    iters: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Log-space Sinkhorn scaling for a rectangular matrix.

    Returns a nonnegative plan \Pi with row sums r and column sums c (approximately, for finite iters).

    logits: [..., N_x, N_y] where exp(logits) is the Gibbs kernel.
    r: [N_x] positive row marginals (sum == 1 recommended).
    c: [N_y] positive column marginals (sum == 1 recommended).
    """
    if iters <= 0:
        raise ValueError(f"iters must be > 0, got {iters}")

    log_r = torch.log(r.clamp_min(eps)).to(device=logits.device, dtype=logits.dtype)
    log_c = torch.log(c.clamp_min(eps)).to(device=logits.device, dtype=logits.dtype)

    log_u = torch.zeros_like(logits[..., :, 0])  # [..., N_x]
    log_v = torch.zeros_like(logits[..., 0, :])  # [..., N_y]
    for _ in range(int(iters)):
        log_u = log_r - torch.logsumexp(logits + log_v.unsqueeze(-2), dim=-1)
        log_v = log_c - torch.logsumexp(logits + log_u.unsqueeze(-1), dim=-2)
    return torch.exp(logits + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)).clamp_min(0.0)


def _apply_uncond_bias_(
    logit_neg: torch.Tensor,
    *,
    nuncond: int,
    uncond_weight: torch.Tensor | None,
) -> None:
    """
    In-place: add 'log(w)' bias to unconditional negative logits (Appendix A.7).

    In Appendix A.7, unconditional negatives are *weighted by w when computing
    the kernel*. For a Gibbs kernel 'K = exp(-dist / T)', multiplying the
    unconditional columns by 'w' is equivalent to adding 'log(w)' to their
    logits.

    If w <= 0, we set the unconditional-logit bias to -inf, so those columns
    receive zero kernel mass. This is consistent with the Kaiming's paper baseline
    (Algorithm 2 joint normalization), which does not enforce fixed column marginals.
    For full Sinkhorn couplings with explicit positive column marginals, zero-mass
    columns can make the constraints infeasible; see 'sinkhorn_marginal' handling
    in this file and the marginal-balancing discussion in our paper.
    """
    if nuncond <= 0 or uncond_weight is None:
        return
    w = uncond_weight.to(device=logit_neg.device, dtype=logit_neg.dtype)
    while w.ndim < logit_neg.ndim:
        w = w.unsqueeze(-1)
    w = w[..., :1, :1]  # broadcast to [...,1,1]
    bias = torch.where(w > 0, w.log(), torch.tensor(-math.inf, device=logit_neg.device, dtype=logit_neg.dtype))
    logit_neg[..., -nuncond:] = logit_neg[..., -nuncond:] + bias


def _has_nonpositive_uncond_weight(uncond_weight: torch.Tensor | None) -> bool:
    if uncond_weight is None:
        return False
    return bool((uncond_weight <= 0).any().item())


def _mask_self_neg_dist_(dist_neg: torch.Tensor) -> None:
    """
    In-place: mask diagonal self-coupling inside the first N_x negative columns.
    """
    n_x = dist_neg.shape[-2]
    idx = torch.arange(n_x, device=dist_neg.device)
    dist_neg[..., idx, idx] = 1e6


def _alg2_from_distances(
    dist_pos: torch.Tensor,
    dist_neg: torch.Tensor,
    *,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temp: float,
    mask_self_in_y_neg: bool,
    nuncond: int,
    uncond_weight: torch.Tensor | None,
    eps: float = 1e-12,
    impl: Literal["logspace", "kernel"] = "logspace",
) -> torch.Tensor:
    """
    Algorithm-2 drift using precomputed distances.

    dist_pos: [..., N_x, N_pos]
    dist_neg: [..., N_x, N_neg] where first N_x columns correspond to x itself.
    y_pos:    [..., N_pos, C]
    y_neg:    [..., N_neg, C]
    returns:  [..., N_x, C]

    Appendix Algorithm 2 chain:
      dist -> logits (with CFG bias on unconditional negatives) -> joint A
      -> split A_pos/A_neg -> W_pos/W_neg -> V.
    """
    if not (temp > 0):
        raise ValueError(f"temp must be > 0, got {temp}")

    if mask_self_in_y_neg:
        n_x = dist_neg.shape[-2]
        idx = torch.arange(n_x, device=dist_neg.device)
        dist_neg[..., idx, idx] = 1e6

    logit_pos = -dist_pos / float(temp)
    logit_neg = -dist_neg / float(temp)

    if nuncond > 0 and uncond_weight is not None:
        w = uncond_weight.to(device=logit_neg.device, dtype=logit_neg.dtype)
        while w.ndim < logit_neg.ndim:
            w = w.unsqueeze(-1)
        w = w[..., :1, :1]  # broadcast to [...,1,1]
        bias = torch.where(w > 0, w.log(), torch.tensor(-math.inf, device=logit_neg.device, dtype=logit_neg.dtype))
        logit_neg[..., -nuncond:] = logit_neg[..., -nuncond:] + bias

    logits = torch.cat([logit_pos, logit_neg], dim=-1)

    if impl == "kernel":
        kernel = torch.exp(logits)
        row_sum = kernel.sum(dim=-1, keepdim=True).clamp_min(eps)
        col_sum = kernel.sum(dim=-2, keepdim=True).clamp_min(eps)
        A = kernel / (row_sum * col_sum).sqrt()
    elif impl == "logspace":
        log_row_sum = torch.logsumexp(logits, dim=-1, keepdim=True)
        log_col_sum = torch.logsumexp(logits, dim=-2, keepdim=True)
        log_a = logits - 0.5 * (log_row_sum + log_col_sum)
        log_a = log_a.masked_fill(torch.isneginf(logits), -math.inf)
        A = torch.exp(log_a)
    else:
        raise ValueError(f"Unknown impl: {impl}")

    n_pos = dist_pos.shape[-1]
    A_pos = A[..., :n_pos]
    A_neg = A[..., n_pos:]

    W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)

    drift_pos = W_pos @ y_pos
    drift_neg = W_neg @ y_neg
    return drift_pos - drift_neg


def drifting_loss_for_feature_set(
    x_feat: torch.Tensor,
    y_pos_feat: torch.Tensor,
    y_uncond_feat: torch.Tensor,
    *,
    omega: torch.Tensor,
    temps: Iterable[float],
    impl: Literal["logspace", "kernel"] = "logspace",
    vanilla: bool = False,
    drift_form: DriftForm = "alg2_joint",
    coupling: Coupling = "partial_two_sided",
    sinkhorn_iters: int = 20,
    sinkhorn_marginal: SinkhornMarginal = "none",
    mask_self_neg: bool = True,
    dist_metric: DistMetric = "l2",
    normalize_drift_theta: bool = True,
    stats: dict[str, float] | None = None,
) -> torch.Tensor:
    """
    Compute drifting loss for one feature set.

    Kaiming's paper reference:
      - Appendix A.6 (feature & drift normalization): Eq. (20)–(21), Eq. (24)–(25),
        multi-temperature aggregation with ρ̃_j=ρ·C_j, and the loss Eq. (26).
      - Appendix A.7 (CFG): unconditional negatives weighted by w(ω).

    Shapes:
      x_feat:        [Nneg, L, C] generated features (requires grad).
      y_pos_feat:    [Npos, L, C] positive real features (no grad needed).
      y_uncond_feat: [Nuncond, L, C] unconditional real features (CFG negatives).

    Args:
      omega: scalar tensor (CFG strength ω) for this class.
      temps: iterable of ρ values (paper uses {0.02, 0.05, 0.2}).
      impl: numeric implementation for the paper's partial two-sided normalization.
      vanilla: if True, skip feature normalization (S=1).
      drift_form:
        - "alg2_joint": paper Algorithm-2 joint normalization over [pos, neg] (default baseline).
        - "split": follow-up ablation exposing V=Pxy@y_pos - Pxneg@y_neg.
      coupling:
        - "partial_two_sided": paper normalized Gibbs coupling (default baseline).
        - "row": one-sided row-softmax coupling.
        - "sinkhorn": full two-sided Sinkhorn scaling.
      sinkhorn_marginal: how to handle CFG(w) for Sinkhorn; prefer "weighted_cols".
      mask_self_neg: mask diagonal self-coupling inside generated negatives (Alg. 2).
      dist_metric: pairwise distance metric ("l2" for ||x-y||, "l2_sq" for ||x-y||^2).
      normalize_drift_theta: if True, apply Eq. (25) drift normalization V/theta.
      stats: optional dict populated with lightweight debug scalars for logging.
    """
    # Important: this loss involves cdist/logsumexp/Sinkhorn-like scaling and is
    # numerically sensitive in fp16. Even if the caller enables AMP for the
    # generator/encoder forward pass, we force the drifting loss computation to
    # run with autocast disabled to avoid NaN/Inf instabilities.
    #
    # This matches the paper intent: the drift construction is a stop-grad
    # statistic computed from distances and normalizations (Appendix A.6/A.7),
    # where float32 stability is preferred over fp16 throughput.
    amp_off = torch.autocast(device_type="cuda", enabled=False) if x_feat.device.type == "cuda" else nullcontext()
    def _impl(x_feat: torch.Tensor, y_pos_feat: torch.Tensor, y_uncond_feat: torch.Tensor) -> torch.Tensor:
        x_feat = _ensure_nlc(x_feat, "x_feat")
        y_pos_feat = _ensure_nlc(y_pos_feat, "y_pos_feat")
        y_uncond_feat = _ensure_nlc(y_uncond_feat, "y_uncond_feat")

        nneg, l, c = x_feat.shape
        npos = y_pos_feat.shape[0]
        nuncond = y_uncond_feat.shape[0]

        if y_pos_feat.shape[1:] != (l, c) or y_uncond_feat.shape[1:] != (l, c):
            raise ValueError("Feature shapes must match across x/pos/uncond")

        if coupling == "sinkhorn" and impl != "logspace":
            raise ValueError("coupling='sinkhorn' requires impl='logspace'")
        if sinkhorn_marginal != "none" and coupling != "sinkhorn":
            raise ValueError("sinkhorn_marginal is only used when coupling='sinkhorn'")
        if dist_metric not in {"l2", "l2_sq"}:
            raise ValueError(f"Unknown dist_metric: {dist_metric}")

        # Reformat to batch over L (per-location drift).
        x_lnc = x_feat.permute(1, 0, 2).contiguous()  # [L,Nneg,C]
        y_pos_lpc = y_pos_feat.permute(1, 0, 2).contiguous()  # [L,Npos,C]
        y_uncond_luc = y_uncond_feat.permute(1, 0, 2).contiguous() if nuncond > 0 else None  # [L,Nunc,C]

        with torch.no_grad():
            # Drift construction is stop-grad; compute it in float32 for stability.
            x_det = x_lnc.detach().float()
            y_pos_det = y_pos_lpc.detach().float()
            omega_s = omega.detach().to(device=x_feat.device, dtype=torch.float32).reshape(())
            if nuncond > 0:
                assert y_uncond_luc is not None
                y_uncond_det = y_uncond_luc.detach().float()
                y_neg_det = torch.cat([x_det, y_uncond_det], dim=1)  # [L,Nneg+Nuncond,C]
                w = compute_uncond_weight(omega_s, nneg=nneg, nuncond=nuncond).to(dtype=torch.float32)
                if coupling == "sinkhorn" and sinkhorn_marginal == "none" and _has_nonpositive_uncond_weight(w):
                    # Sinkhorn imposes explicit positive column marginals, while Appendix A.7's
                    # log(w) kernel weighting removes unconditional columns when w<=0.
                    # These constraints are incompatible (infeasible) and can produce NaNs.
                    raise ValueError(
                        "Incompatible constraints: coupling='sinkhorn' with sinkhorn_marginal='none' "
                        "and non-positive CFG weight w (e.g., omega<=1) can produce infeasible column marginals. "
                        "Use --sinkhorn-marginal weighted_cols or ensure omega>1."
                    )
            else:
                y_neg_det = x_det
                w = None

            dist_pos_raw = _pairwise_distance(x_det, y_pos_det, metric=dist_metric)  # [L,Nneg,Npos]
            dist_neg_raw = _pairwise_distance(x_det, y_neg_det, metric=dist_metric)  # [L,Nneg,Nneg(+Nuncond)]

            if vanilla:
                s = torch.tensor(1.0, device=x_feat.device, dtype=dist_pos_raw.dtype)
            else:
                # Feature normalization S_j (Eq. 21) computed from all x/y in batch.
                #
                # Important details:
                # - Self distances in the generated negatives are ignored (Alg. 2).
                # - For CFG, unconditional negatives are weighted by 'w' (Appendix A.7).
                sum_pos = dist_pos_raw.sum()
                denom_pos = float(dist_pos_raw.numel())

                # Generated negatives: first Nneg columns in dist_neg_raw (ignore diagonal self-coupling).
                dist_neg_gen = dist_neg_raw[..., :nneg]
                sum_neg_gen = dist_neg_gen.sum()
                denom_neg_gen = (
                    float(dist_neg_gen.numel() - (l * nneg)) if mask_self_neg else float(dist_neg_gen.numel())
                )

                if nuncond > 0:
                    dist_neg_unc = dist_neg_raw[..., nneg:]
                    sum_neg_unc = dist_neg_unc.sum()
                    denom_neg_unc = float(dist_neg_unc.numel())

                    w_eff = torch.clamp(w if w is not None else torch.tensor(0.0, device=x_feat.device), min=0.0).to(
                        device=x_feat.device, dtype=dist_pos_raw.dtype
                    )
                    sum_dist = sum_pos + sum_neg_gen + w_eff * sum_neg_unc
                    denom_t = torch.tensor(
                        denom_pos + denom_neg_gen, device=x_feat.device, dtype=dist_pos_raw.dtype
                    ) + w_eff * denom_neg_unc
                else:
                    sum_dist = sum_pos + sum_neg_gen
                    denom_t = torch.tensor(denom_pos + denom_neg_gen, device=x_feat.device, dtype=dist_pos_raw.dtype)

                mean_dist = sum_dist / denom_t.clamp_min(1e-12)
                # Eq. (20)(21): enforce E[dist_j(x,y)] ≈ sqrt(C_j).
                s = (mean_dist / math.sqrt(float(c))).clamp_min(1e-6)

            if stats is not None:
                stats["drift_s"] = float(s)
                stats["drift_c"] = float(c)

            dist_pos = dist_pos_raw / s
            dist_neg = dist_neg_raw / s

            if mask_self_neg:
                _mask_self_neg_dist_(dist_neg)

            # CFG unconditional weighting.
            omega_s = omega_s.to(device=x_feat.device, dtype=dist_pos.dtype)
            w = w.to(device=x_feat.device, dtype=dist_pos.dtype) if w is not None else None

            # Compute aggregated normalized drift across temperatures (A.6).
            y_pos_norm = y_pos_det / s
            if nuncond > 0:
                drift_y_neg = torch.cat([x_det, y_uncond_det], dim=1)
            else:
                drift_y_neg = x_det
            drift_y_neg_raw = drift_y_neg  # still raw; divide after matmul

            v_agg = torch.zeros((l, nneg, c), device=x_feat.device, dtype=x_det.dtype)
            for rho in temps:
                temp_eff = float(rho) * float(c)

                y_neg_norm = drift_y_neg_raw / s

                # Joint form (paper): V = W_pos @ Y_pos - W_neg @ Y_neg, with A built on [pos, neg].
                if drift_form == "alg2_joint":
                    if coupling == "partial_two_sided":
                        # A = sqrt(softmax_row(logits) * softmax_col(logits)) on concatenated logits.
                        v_raw = _alg2_from_distances(
                            dist_pos,
                            dist_neg,
                            y_pos=y_pos_norm,
                            y_neg=y_neg_norm,
                            temp=temp_eff,
                            mask_self_in_y_neg=False,  # already masked above (if enabled)
                            nuncond=nuncond,
                            uncond_weight=w,
                            impl=impl,
                        )
                    else:
                        logit_pos = -dist_pos / float(temp_eff)
                        logit_neg = -dist_neg / float(temp_eff)
                        if coupling != "sinkhorn" or sinkhorn_marginal == "none":
                            _apply_uncond_bias_(logit_neg, nuncond=nuncond, uncond_weight=w)
                        logits = torch.cat([logit_pos, logit_neg], dim=-1)

                        if coupling == "row":
                            # One-sided joint coupling: A = softmax_row([logit_pos, logit_neg]).
                            A = _row_stochastic_from_logits(logits)
                        elif coupling == "sinkhorn":
                            # Full joint coupling: A = Sinkhorn(exp(logits); row marginal r, col marginal c_m).
                            nx = nneg
                            ny = int(logits.shape[-1])
                            r = torch.full((nx,), 1.0 / float(nx), device=logits.device, dtype=logits.dtype)
                            if sinkhorn_marginal == "weighted_cols" and nuncond > 0:
                                w_eff = torch.clamp(
                                    w if w is not None else torch.tensor(0.0, device=logits.device), min=0.0
                                ).to(device=logits.device, dtype=logits.dtype)
                                col_w = torch.ones((ny,), device=logits.device, dtype=logits.dtype)
                                col_w[-nuncond:] = torch.clamp(w_eff, min=1e-6)
                                c_m = (col_w / col_w.sum()).to(device=logits.device, dtype=logits.dtype)
                            else:
                                c_m = torch.full((ny,), 1.0 / float(ny), device=logits.device, dtype=logits.dtype)
                            A = _sinkhorn_from_logits(logits, r=r, c=c_m, iters=int(sinkhorn_iters))
                            if stats is not None:
                                tag = str(rho).replace(".", "p")
                                stats[f"drift_sinkhorn_joint_row_mae_{tag}"] = float((A.sum(dim=-1) - r).abs().mean())
                                stats[f"drift_sinkhorn_joint_col_mae_{tag}"] = float((A.sum(dim=-2) - c_m).abs().mean())
                        else:
                            raise ValueError(f"Unknown coupling: {coupling}")

                        A_pos = A[..., :npos]
                        A_neg = A[..., npos:]
                        # Alg.2 weighting: W_pos = A_pos * sum(A_neg), W_neg = A_neg * sum(A_pos).
                        W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)
                        W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)
                        v_raw = (W_pos @ y_pos_norm) - (W_neg @ y_neg_norm)
                # Split form (follow-up): V = Pxy @ Y_pos - Px,neg @ Y_neg.
                elif drift_form == "split":
                    logit_pos = -dist_pos / float(temp_eff)
                    if coupling == "row":
                        # Pxy = softmax_row(logit_pos).
                        P_pos = _row_stochastic_from_logits(logit_pos)
                    elif coupling == "partial_two_sided":
                        # Pxy from partial-two-sided A, then row-normalized to barycentric weights.
                        A_pos = _partial_two_sided_from_logits(logit_pos, impl=impl)
                        P_pos = A_pos / A_pos.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    elif coupling == "sinkhorn":
                        # Pxy from Sinkhorn plan on (x, y_pos), then row-normalized.
                        nx = nneg
                        ny_pos = npos
                        r = torch.full((nx,), 1.0 / float(nx), device=logit_pos.device, dtype=logit_pos.dtype)
                        c_m = torch.full((ny_pos,), 1.0 / float(ny_pos), device=logit_pos.device, dtype=logit_pos.dtype)
                        Pi_pos = _sinkhorn_from_logits(logit_pos, r=r, c=c_m, iters=int(sinkhorn_iters))
                        if stats is not None:
                            tag = str(rho).replace(".", "p")
                            stats[f"drift_sinkhorn_pos_row_mae_{tag}"] = float((Pi_pos.sum(dim=-1) - r).abs().mean())
                            stats[f"drift_sinkhorn_pos_col_mae_{tag}"] = float((Pi_pos.sum(dim=-2) - c_m).abs().mean())
                        P_pos = Pi_pos / Pi_pos.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    else:
                        raise ValueError(f"Unknown coupling: {coupling}")

                    drift_pos = P_pos @ y_pos_norm

                    logit_neg = -dist_neg / float(temp_eff)
                    if coupling != "sinkhorn" or sinkhorn_marginal == "none":
                        _apply_uncond_bias_(logit_neg, nuncond=nuncond, uncond_weight=w)

                    if coupling == "row":
                        # Px,neg = softmax_row(logit_neg).
                        P_neg = _row_stochastic_from_logits(logit_neg)
                    elif coupling == "partial_two_sided":
                        # Px,neg from partial-two-sided A, then row-normalized.
                        A_neg = _partial_two_sided_from_logits(logit_neg, impl=impl)
                        P_neg = A_neg / A_neg.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    elif coupling == "sinkhorn":
                        # Px,neg from Sinkhorn plan on (x, y_neg), then row-normalized.
                        nx = nneg
                        ny_neg = int(logit_neg.shape[-1])
                        r = torch.full((nx,), 1.0 / float(nx), device=logit_neg.device, dtype=logit_neg.dtype)
                        if sinkhorn_marginal == "weighted_cols" and nuncond > 0:
                            w_eff = torch.clamp(
                                w if w is not None else torch.tensor(0.0, device=logit_neg.device), min=0.0
                            ).to(device=logit_neg.device, dtype=logit_neg.dtype)
                            col_w = torch.ones((ny_neg,), device=logit_neg.device, dtype=logit_neg.dtype)
                            col_w[-nuncond:] = torch.clamp(w_eff, min=1e-6)
                            c_m = (col_w / col_w.sum()).to(device=logit_neg.device, dtype=logit_neg.dtype)
                        else:
                            c_m = torch.full((ny_neg,), 1.0 / float(ny_neg), device=logit_neg.device, dtype=logit_neg.dtype)
                        Pi_neg = _sinkhorn_from_logits(logit_neg, r=r, c=c_m, iters=int(sinkhorn_iters))
                        if stats is not None:
                            tag = str(rho).replace(".", "p")
                            stats[f"drift_sinkhorn_neg_row_mae_{tag}"] = float((Pi_neg.sum(dim=-1) - r).abs().mean())
                            stats[f"drift_sinkhorn_neg_col_mae_{tag}"] = float((Pi_neg.sum(dim=-2) - c_m).abs().mean())
                        P_neg = Pi_neg / Pi_neg.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    else:
                        raise ValueError(f"Unknown coupling: {coupling}")

                    drift_neg = P_neg @ y_neg_norm
                    v_raw = drift_pos - drift_neg
                else:
                    raise ValueError(f"Unknown drift_form: {drift_form}")

                # Drift normalization (Eq. 25).
                v2 = (v_raw * v_raw).mean()
                # Eq. (24)(25): enforce E[||Ṽ||^2]/C ≈ 1  => mean(Ṽ^2) ≈ 1.
                theta = torch.sqrt(v2.clamp_min(1e-12))
                if stats is not None:
                    tag = str(rho).replace(".", "p")
                    stats[f"drift_theta_{tag}"] = float(theta)
                if normalize_drift_theta:
                    v_agg = v_agg + (v_raw / theta)
                else:
                    v_agg = v_agg + v_raw

            v_agg_nlc = v_agg.permute(1, 0, 2).contiguous()  # [Nneg,L,C]

            if stats is not None:
                v_nonfinite = (~torch.isfinite(v_agg_nlc)).sum()
                v_nan = torch.isnan(v_agg_nlc).sum()
                stats["drift_v_rms"] = float(torch.sqrt((v_agg_nlc * v_agg_nlc).mean()).clamp_min(0.0))
                stats["drift_v_nonfinite_count"] = float(v_nonfinite)
                stats["drift_v_nan_count"] = float(v_nan)

        # Loss (Eq. 26): MSE(ϖ̃(x) - sg(ϖ̃(x) + Ṽ)).
        x_norm = x_feat.float() / s.to(dtype=torch.float32)
        target = (x_norm + v_agg_nlc).detach()
        return F.mse_loss(x_norm, target, reduction="mean")

    with amp_off:
        return _impl(x_feat, y_pos_feat, y_uncond_feat)


def sample_power_law_omega(
    n: int,
    *,
    omega_min: float,
    omega_max: float,
    exponent: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample omega ~ p(omega) ∝ omega^{-exponent} on [omega_min, omega_max].
    """
    if not (omega_min > 0 and omega_max > omega_min):
        raise ValueError("Invalid omega range")
    if exponent == 1.0:
        # p(ω) ∝ 1/ω => log-uniform.
        u = torch.rand(n, device=device, dtype=dtype)
        return omega_min * torch.exp(u * math.log(omega_max / omega_min))
    a = 1.0 - float(exponent)
    lo = omega_min**a
    hi = omega_max**a
    u = torch.rand(n, device=device, dtype=dtype)
    return (lo + u * (hi - lo)).pow(1.0 / a)
