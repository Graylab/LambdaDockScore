"""Regenerate the paper's figures from the per-pose result CSVs in results/.

    python analysis/make_paper_figures.py --outdir figures

Produces:
  fig1_capri_topk.png        Figure 1  — CAPRI score set, 3 models, top-1/top-5
  fig2_db55_topk.png         Figure 2  — DB5.5, 253 complexes, top-1/top-5/oracle
  fig3_db55_antibody.png     Figure 3  — DB5.5 antibody-antigen subset (n=55)
  figS2_db55_asa_bins.png    Figure S2 — DB5.5 split into 5 delta-ASA bins

The stacked-bar encoding follows the published figures: one hue per model (fixed,
never reassigned by rank), shades running light->dark across the three CAPRI
quality tiers. Bars are *overlaid*, not summed: the Medium bar is drawn on top of
the Acceptable bar because every Medium pose is also Acceptable.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lds_data as D
import lds_metrics as M

QUALITIES = ["Acceptable", "Medium", "High"]


# Shared figure style, matching the paper's published stacked-bar plots:
# dashed grey y-grid, alpha-blended bars in light->dark tier shades, per-tier
# tinted value labels in fixed rows near the top and a lightly boxed legend.
def _style(ax, ymax, label_fs=18, tick_fs=14):
    ax.set_ylabel("Success Rate (%)", fontsize=label_fs)
    ax.set_ylim(0, ymax)
    ax.grid(which="major", axis="y", linestyle="--", color="gray", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=tick_fs)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _legend(ax, fontsize):
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=fontsize,
              frameon=True, framealpha=0.9, edgecolor="0.8")


def _annotate(ax, cols, values, ymax, fontsize=13):
    """Per-tier value labels in three fixed rows near the top, each tinted with
    its quality shade so the number matches the bar it summarises."""
    rows = {"Acceptable": ymax * 0.97, "Medium": ymax * 0.90, "High": ymax * 0.83}
    n_models = len(values)
    width = 0.8 / n_models
    for ci, col in enumerate(cols):
        for mi, (model, vals) in enumerate(values.items()):
            x = ci + (mi - (n_models - 1) / 2) * width
            for q in QUALITIES:
                ax.text(x, rows[q], f"{vals[col][q]:.1f}",
                        ha="center", va="bottom", fontsize=fontsize,
                        color=M.MODEL_COLORS[model][q])


def _grouped_stacked(ax, cols, values, models, ymax, annot_fontsize=13,
                     label_fs=18, tick_fs=14, xtick_fs=16):
    """cols = x groups (e.g. ['Top-1','Top-5']); values[model][col][quality] = pct."""
    n = len(models)
    width = 0.8 / n
    for mi, model in enumerate(models):
        offs = (mi - (n - 1) / 2) * width
        for q in QUALITIES:  # draw light (widest) first, darker on top
            heights = [values[model][c][q] for c in cols]
            ax.bar(
                np.arange(len(cols)) + offs, heights, width * 0.92,
                color=M.MODEL_COLORS[model][q], alpha=0.8,
                label=f"{model} - {q}",
                zorder=2 + QUALITIES.index(q),
            )
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=xtick_fs)
    _style(ax, ymax=ymax, label_fs=label_fs, tick_fs=tick_fs)
    _annotate(ax, cols, values, ymax, fontsize=annot_fontsize)


def figure1(outdir):
    """CAPRI Figure 1 -- shares the common paper stacked-bar style (see helpers)."""
    models = ["Baseline DFMDock", "LambdaDockScore", "EuDockScore"]
    cols = ["Top-1", "Top-5"]
    values, n = {}, None
    for m in models:
        df, hib = D.load_capri(m)
        r = M.success_at_k(df, hib, ks=(1, 5))
        n = int(r["n_targets"].iloc[0])
        values[m] = {
            "Top-1": {q: r[(r.k == 1) & (r.quality == q)].success_rate.iloc[0] for q in QUALITIES},
            "Top-5": {q: r[(r.k == 5) & (r.quality == q)].success_rate.iloc[0] for q in QUALITIES},
        }
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=200)
    _grouped_stacked(ax, cols, values, models, ymax=70)
    ax.set_title(f"CAPRI Score Set (n={n})", fontsize=15)
    _legend(ax, fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_capri_topk.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)
    return values


COL_TO_K = {"Top-1": 1, "Top-5": 5, "Oracle": "oracle"}


def _db55_boot(subset_ids, n_boot):
    """Pose-level bootstrap for both models -> {model: {col: {quality: (mean,lo,hi)}}}.

    This is the statistic the paper reports for Figures 2/3/S2 (see M.pose_bootstrap).
    """
    models = ["Baseline DFMDock", "LambdaDockScore"]
    out, n = {}, None
    for m in models:
        df, hib = D.load_db55(m)
        if subset_ids is not None:
            df = df[df["id"].isin(subset_ids)]
        per_col = {c: {} for c in COL_TO_K}
        for q in QUALITIES:
            r = M.pose_bootstrap(df, hib, ks=(1, 5), quality=q, n_boot=n_boot)
            n = int(r["n_targets"].iloc[0])
            for col, k in COL_TO_K.items():
                row = r[r["k"] == k].iloc[0]
                per_col[col][q] = (row["mean"], row["lo"], row["hi"])
        out[m] = per_col
    return models, out, n


def _db55_values(subset_ids=None, n_boot=10_000):
    models, boot, n = _db55_boot(subset_ids, n_boot)
    values = {
        m: {col: {q: boot[m][col][q][0] for q in QUALITIES} for col in COL_TO_K}
        for m in models
    }
    return models, values, n, boot


def _db55_errbars(ax, models, cols, boot):
    """95% pose-bootstrap CI, drawn on the Acceptable (outermost) bar."""
    nm = len(models)
    width = 0.8 / nm
    for mi, m in enumerate(models):
        for ci, col in enumerate(cols):
            mean, lo, hi = boot[m][col]["Acceptable"]
            x = ci + (mi - (nm - 1) / 2) * width
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                        ecolor="0.25", capsize=3, linewidth=1.0, zorder=10)


def figure2(outdir, n_boot):
    models, values, n, boot = _db55_values(n_boot=n_boot)
    cols = ["Top-1", "Top-5", "Oracle"]
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=200)
    _grouped_stacked(ax, cols, values, models, ymax=70)
    _db55_errbars(ax, models, cols, boot)
    ax.set_title(f"DB5.5 (n={n})", fontsize=15)
    _legend(ax, fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_db55_topk.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def figure3(outdir, n_boot):
    meta = D.load_db55_metadata()
    ab = set(meta[meta["cat"] == "AA"]["id"])
    models, values, n, boot = _db55_values(subset_ids=ab, n_boot=n_boot)
    cols = ["Top-1", "Top-5", "Oracle"]
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=200)
    _grouped_stacked(ax, cols, values, models, ymax=70)
    _db55_errbars(ax, models, cols, boot)
    ax.set_title(f"Antibody-antigen complexes (n={n})", fontsize=15)
    _legend(ax, fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, "fig3_db55_antibody.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def figureS2(outdir, n_boot):
    meta = D.load_db55_metadata()
    bins = D.asa_bins(meta, n_bins=5)
    labels = list(bins["asa_bin"].cat.categories)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=200)
    cols = ["Top-1", "Top-5", "Oracle"]
    for i, lab in enumerate(labels):
        ax = axes.flat[i]
        ids = set(bins[bins["asa_bin"] == lab]["id"])
        models, values, n, boot = _db55_values(subset_ids=ids, n_boot=n_boot)
        _grouped_stacked(ax, cols, values, models, ymax=70)
        _db55_errbars(ax, models, cols, boot)
        rng = bins[bins["asa_bin"] == lab]["delta_asa"]
        ax.set_title(f"({chr(97+i)}) {lab}\n({rng.min():.0f}-{rng.max():.0f} $\\AA^2$, n={n})", fontsize=13)
    axes.flat[-1].axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for a, b in zip(h, l):
        if b not in seen:
            seen.add(b); hh.append(a); ll.append(b)
    axes.flat[-1].legend(hh, ll, fontsize=13, loc="center",
                         frameon=True, framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()
    p = os.path.join(outdir, "figS2_db55_asa_bins.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(D.REPO, "figures"))
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--only", nargs="*", choices=["1", "2", "3", "S2"])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    want = set(a.only) if a.only else {"1", "2", "3", "S2"}
    if "1" in want: figure1(a.outdir)
    if "2" in want: figure2(a.outdir, a.n_boot)
    if "3" in want: figure3(a.outdir, a.n_boot)
    if "S2" in want: figureS2(a.outdir, a.n_boot)