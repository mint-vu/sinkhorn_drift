#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DEFAULT_TARGET_ORDER = ("Moons", "Spiral", "8-Gaussians", "Checkerboard")
DEFAULT_METHOD_ORDER = ("one-sided", "two-sided", "sinkhorn")
DEFAULT_METHOD_COLORS = {
    "one-sided": "#83C4CE",
    "two-sided": "#F3CE7F",
    "sinkhorn": "#B56C73",
}
DEFAULT_METHOD_STYLES = {
    "one-sided": ":",
    "two-sided": "--",
    "sinkhorn": "-",
}


@dataclass
class Curve:
    steps: np.ndarray
    values: np.ndarray


def _csv_tuple(text: str | None) -> tuple[str, ...]:
    if text is None:
        return ()
    return tuple(x.strip() for x in text.split(",") if x.strip())


def _csv_float_tuple(text: str | None) -> tuple[float, ...]:
    if text is None:
        return ()
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _load_run_logs(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Expected list in logs: {path}")
    return obj


def _curve_from_rec(rec: dict) -> Curve:
    log = rec.get("log", {})
    emd = log.get("emd2", [])
    if not emd:
        return Curve(steps=np.empty((0,), dtype=np.int64), values=np.empty((0,), dtype=np.float64))
    steps = np.asarray([int(p[0]) for p in emd], dtype=np.int64)
    vals = np.asarray([float(p[1]) for p in emd], dtype=np.float64)
    return Curve(steps=steps, values=vals)


def _stack_curves(curves: Sequence[Curve]) -> tuple[np.ndarray, np.ndarray]:
    if not curves:
        return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float64)
    all_steps = sorted(set().union(*[set(c.steps.tolist()) for c in curves]))
    x = np.asarray(all_steps, dtype=np.int64)
    y = np.full((len(curves), len(x)), np.nan, dtype=np.float64)
    step_index = {int(s): i for i, s in enumerate(x.tolist())}
    for ri, c in enumerate(curves):
        for s, v in zip(c.steps.tolist(), c.values.tolist()):
            y[ri, step_index[int(s)]] = float(v)
    return x, y


def _ordered_subset(found: Sequence[str], preferred: Sequence[str]) -> list[str]:
    found_set = set(found)
    out = [x for x in preferred if x in found_set]
    out.extend(x for x in found if x not in set(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate multi-seed W2^2 curves (mean ± std) and plot.")
    ap.add_argument("--runs-glob", type=str, required=True, help="Glob pattern for run folders.")
    ap.add_argument("--out-pdf", type=str, required=True)
    ap.add_argument("--out-png", type=str, default=None)
    ap.add_argument("--targets", type=str, default=None, help="Optional CSV target order.")
    ap.add_argument("--methods", type=str, default=None, help="Optional CSV method order.")
    ap.add_argument("--eps-list", type=str, default=None, help="Optional CSV eps order.")
    ap.add_argument("--figscale", type=float, default=1.0, help="Global figure scale.")
    ap.add_argument("--fig-width-in", type=float, default=None, help="Override figure width in inches.")
    ap.add_argument("--fig-height-in", type=float, default=None, help="Override figure height in inches.")
    ap.add_argument(
        "--legend-figure-top-right",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Place legend at figure top-right instead of first subplot.",
    )
    ap.add_argument(
        "--legend-subplot-col",
        type=int,
        default=0,
        help="When not using figure-level legend, place legend in top-row subplot at this column index (default: 0).",
    )
    ap.add_argument(
        "--right-ylabel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show right-side W2^2 ylabel on the last column (default: false).",
    )
    ap.add_argument(
        "--eps-descending",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sort eps rows from large to small (default: false).",
    )
    ap.add_argument(
        "--title",
        type=str,
        default=None,
        help='Title. Default: "Mean ± std over N random seeds".',
    )
    args = ap.parse_args()

    run_dirs = sorted(glob.glob(args.runs_glob))
    if not run_dirs:
        raise FileNotFoundError(f"No runs matched: {args.runs_glob}")

    curves: Dict[Tuple[str, str, float], List[Curve]] = defaultdict(list)
    found_targets: list[str] = []
    found_methods: list[str] = []
    found_eps: list[float] = []

    for rd in run_dirs:
        lp = os.path.join(rd, "logs.json")
        if not os.path.exists(lp):
            print(f"[WARN] skip (missing logs.json): {rd}")
            continue
        recs = _load_run_logs(lp)
        for rec in recs:
            t = str(rec.get("target"))
            m = str(rec.get("method"))
            e = float(rec.get("eps"))
            c = _curve_from_rec(rec)
            if c.steps.size == 0:
                continue
            curves[(t, m, e)].append(c)
            if t not in found_targets:
                found_targets.append(t)
            if m not in found_methods:
                found_methods.append(m)
            if e not in found_eps:
                found_eps.append(e)

    if not curves:
        raise RuntimeError(f"No valid curves found from pattern: {args.runs_glob}")

    targets = list(_csv_tuple(args.targets)) if args.targets else _ordered_subset(found_targets, DEFAULT_TARGET_ORDER)
    methods = list(_csv_tuple(args.methods)) if args.methods else _ordered_subset(found_methods, DEFAULT_METHOD_ORDER)
    eps_vals = list(_csv_float_tuple(args.eps_list)) if args.eps_list else sorted(found_eps)
    if bool(args.eps_descending):
        eps_vals = sorted(eps_vals, reverse=True)

    nrows = len(eps_vals)
    ncols = len(targets)
    if nrows == 0 or ncols == 0:
        raise RuntimeError("Nothing to plot (empty eps or targets).")

    # "fig2"-style defaults: landscape and compact.
    fs_title = 15
    fs_col = 13
    fs_tick = 10.5
    fs_legend = 10.5
    fs_ylabel = 12.5
    fs_row = 12.5

    fig_w = (4.0 + 2.1 * ncols) * float(args.figscale)
    fig_h = (2.0 + 1.6 * nrows) * float(args.figscale)
    if args.fig_width_in is not None:
        fig_w = float(args.fig_width_in)
    if args.fig_height_in is not None:
        fig_h = float(args.fig_height_in)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for ri, eps in enumerate(eps_vals):
        for ci, tgt in enumerate(targets):
            ax = axes[ri, ci]
            ymax = 0.0
            for m in methods:
                k = (tgt, m, float(eps))
                if k not in curves:
                    continue
                x, y = _stack_curves(curves[k])
                if x.size == 0:
                    continue
                mean = np.nanmean(y, axis=0)
                std = np.nanstd(y, axis=0)
                color = DEFAULT_METHOD_COLORS.get(m, None)
                style = DEFAULT_METHOD_STYLES.get(m, "-")
                ax.plot(x, mean, color=color, linestyle=style, linewidth=2.0, label=m)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0.0)
                ymax = max(ymax, float(np.nanmax(mean + std)))

            if ri == 0:
                ax.set_title(tgt, fontsize=fs_col)
            if ci == 0:
                ax.set_ylabel(rf"$\epsilon={eps:g}$", fontsize=fs_row)
            if ri == nrows - 1:
                ax.set_xlabel("Iteration", fontsize=11.5)
            ax.tick_params(axis="both", labelsize=fs_tick, direction="in", top=True, right=True)
            ax.grid(True, alpha=0.18, linewidth=0.7)
            if ymax > 0.0:
                ax.set_ylim(0.0, ymax * 1.07)
            ax.margins(x=0.0)

            if bool(args.right_ylabel) and ci == ncols - 1:
                ax_r = ax.twinx()
                ax_r.set_ylabel(r"$W_2^2$", fontsize=fs_ylabel, labelpad=8)
                ax_r.set_yticks([])

            if (not bool(args.legend_figure_top_right)) and ri == 0 and ci == int(args.legend_subplot_col):
                ax.legend(loc="upper right", frameon=False, fontsize=fs_legend, ncol=1, handlelength=2.5)

    n_runs = len(run_dirs)
    title = args.title or rf"$W_2^2$ (Mean ± std over {n_runs} random seeds)"
    fig.suptitle(title, fontsize=fs_title, y=0.99)

    if bool(args.legend_figure_top_right):
        handles = [
            Line2D([], [], color=DEFAULT_METHOD_COLORS[m], linestyle=DEFAULT_METHOD_STYLES[m], linewidth=2.0, label=m)
            for m in methods
            if m in DEFAULT_METHOD_COLORS
        ]
        fig.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.985),
            frameon=False,
            fontsize=fs_legend,
            ncol=1,
            handlelength=2.5,
        )

    right_margin = 0.94 if bool(args.right_ylabel) else 0.98
    fig.subplots_adjust(left=0.08, right=right_margin, bottom=0.08, top=0.93, wspace=0.22, hspace=0.24)

    os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)
    fig.savefig(args.out_pdf, format="pdf", dpi=300)
    print(f"Saved: {args.out_pdf}")
    if args.out_png:
        os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
        fig.savefig(args.out_png, format="png", dpi=300)
        print(f"Saved: {args.out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
