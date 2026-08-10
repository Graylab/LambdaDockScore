"""Success rate as a continuous function of k (not just k=1,5).

    python analysis/topk_curves.py --dataset capri --panel

Reports, per model, the full success-vs-k curve and the k needed to reach fixed
success levels. Use ``--panel`` to render a 1x3 High/Medium/Acceptable figure.
Post-hoc analysis of the per-pose score CSVs in results/ -- no GPU required.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lds_data as D
import lds_metrics as M

# One fixed hue per model (the Medium shade reads well as a line colour).
LINE_COLOR = {m: c["Medium"] for m, c in M.MODEL_COLORS.items()}


def build(dataset: str, kmax: int, quality: str):
    if dataset == "capri":
        loaders = {m: D.load_capri for m in D.CAPRI_SOURCES}
    else:
        loaders = {m: D.load_db55 for m in ("Baseline DFMDock", "LambdaDockScore")}
    curves = {}
    for m, fn in loaders.items():
        df, hib = fn(m)
        c = M.success_curve(df, hib, k_max=kmax, quality=quality)
        curves[m] = c
    return curves


def summarise(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for m, c in curves.items():
        r = c["success_rate"].to_numpy()
        row = {
            "model": m,
            "top1": r[0],
            "top5": r[4] if len(r) >= 5 else np.nan,
            "top10": r[9] if len(r) >= 10 else np.nan,
            "top50": r[49] if len(r) >= 50 else np.nan,
            "plateau": r[-1],
            # normalised AUC: mean success over k in [1, kmax], scaled by the
            # plateau so it measures *how fast* the curve rises, not how high.
            "auc_norm": float(r.mean() / r[-1]) if r[-1] else np.nan,
        }
        for lvl in (25, 50, 75):
            tgt = lvl / 100 * r[-1]
            hit = np.argmax(r >= tgt) + 1 if (r >= tgt).any() else np.nan
            row[f"k_to_{lvl}pct_of_plateau"] = hit
        rows.append(row)
    return pd.DataFrame(rows)


def _draw(ax, curves, quality, dataset, show_ylabel=True, show_legend=True):
    for m, c in curves.items():
        ax.plot(c["k"], c["success_rate"], label=m,
                color=LINE_COLOR.get(m, "0.4"), linewidth=2)
    ax.set_xlabel("k", fontsize=18)
    if show_ylabel:
        ax.set_ylabel("Success Rate (%)", fontsize=18)
    ax.grid(which="major", axis="both", linestyle="--", color="gray", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=14)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(quality, fontsize=16)
    if show_legend:
        ax.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor="0.8")


def plot(curves, quality, dataset, outpath):
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=200)
    _draw(ax, curves, quality, dataset)
    n = int(next(iter(curves.values()))["n_targets"].iloc[0])
    ax.set_title(f"{dataset.upper()} (n={n}) - {quality}", fontsize=15)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print("wrote", outpath)


# Panel order matches the paper's tiers, strongest first.
PANEL_QUALITIES = ("High", "Medium", "Acceptable")


def plot_panel(dataset, kmax, outpath):
    """A 1x3 horizontal panel of success-vs-k curves (High, Medium, Acceptable)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=200, sharey=True)
    n = None
    for i, (ax, q) in enumerate(zip(axes, PANEL_QUALITIES)):
        curves = build(dataset, kmax, q)
        n = int(next(iter(curves.values()))["n_targets"].iloc[0])
        _draw(ax, curves, q, dataset, show_ylabel=(i == 0), show_legend=(i == 0))
    fig.suptitle(f"{dataset.upper()} (n={n})", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print("wrote", outpath)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["capri", "db55"], default="capri")
    ap.add_argument("--kmax", type=int, default=100)
    ap.add_argument("--quality", default="Acceptable", choices=list(M.THRESHOLDS))
    ap.add_argument("--panel", action="store_true",
                    help="render a 1x3 High/Medium/Acceptable panel instead of a single quality")
    ap.add_argument("--outdir", default=os.path.join(D.REPO, "figures"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    if a.panel:
        plot_panel(a.dataset, a.kmax, os.path.join(a.outdir, f"topk_panel_{a.dataset}.png"))
        raise SystemExit

    curves = build(a.dataset, a.kmax, a.quality)
    tag = f"{a.dataset}_{a.quality.lower()}"
    long = pd.concat([c.assign(model=m) for m, c in curves.items()], ignore_index=True)
    long.to_csv(os.path.join(a.outdir, f"topk_curve_{tag}.csv"), index=False)
    s = summarise(curves)
    s.to_csv(os.path.join(a.outdir, f"topk_summary_{tag}.csv"), index=False)
    print(s.to_string(index=False))
    plot(curves, a.quality, a.dataset, os.path.join(a.outdir, f"topk_curve_{tag}.png"))