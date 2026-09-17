"""Regenerate paper figures and bar height tables from the per-pose result CSVs in results/.

    python analysis/make_paper_figures.py --outdir figures

Produces SVG figures:
  fig1_capri_topk.svg        Figure 1  - CAPRI score set, 3 models, top-1/top-5 (50% cutoff, no number labels)
  fig2_db55_topk.svg         Figure 2  - DB5.5, 253 complexes, top-1/top-5/oracle (50% cutoff, 3-tier 95% CIs)
  fig3_db55_antibody.svg     Figure 3  - DB5.5 antibody-antigen subset (n=55) (50% cutoff, 3-tier 95% CIs)
  figS2_db55_asa_bins.svg    Figure S2 - DB5.5 split into 5 delta-ASA bins (50% cutoff, 3-tier 95% CIs)
  topk_panel_capri.svg       Top-k success curves for CAPRI (High, Medium, Acceptable)
  topk_panel_db55.svg        Top-k success curves for DB5.5 (High, Medium, Acceptable)
  bar_heights.md             Markdown table listing exact bar heights and 95% bootstrap CIs
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, "analysis"))

import lds_data as D
import lds_metrics as M
import topk_curves

QUALITIES = ["Acceptable", "Medium", "High"]
COL_TO_K = {"Top-1": 1, "Top-5": 5, "Oracle": "oracle"}


def _style(ax, ymax=50, label_fs=18, tick_fs=14):
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


def _grouped_stacked(ax, cols, values, models, ymax=50,
                     label_fs=18, tick_fs=14, xtick_fs=16):
    n = len(models)
    width = 0.8 / n
    for mi, model in enumerate(models):
        offs = (mi - (n - 1) / 2) * width
        for q in QUALITIES:
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


def _db55_boot(subset_ids, n_boot=10_000):
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
                per_col[col][q] = (float(row["mean"]), float(row["lo"]), float(row["hi"]))
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
    """95% pose-bootstrap CI, drawn for all three quality tiers (Acceptable, Medium, High)."""
    nm = len(models)
    width = 0.8 / nm
    for mi, m in enumerate(models):
        for ci, col in enumerate(cols):
            x = ci + (mi - (nm - 1) / 2) * width
            for q in QUALITIES:
                mean, lo, hi = boot[m][col][q]
                if mean == 0.0 and lo == 0.0 and hi == 0.0:
                    continue
                ax.errorbar(
                    x, mean,
                    yerr=[[max(0.0, mean - lo)], [max(0.0, hi - mean)]],
                    fmt="none", ecolor="0.25", capsize=3, linewidth=1.0, zorder=10
                )


def figure1(outdir):
    models = ["Baseline DFMDock", "LambdaDockScore", "EuDockScore"]
    cols = ["Top-1", "Top-5"]
    values, n = {}, None
    for m in models:
        df, hib = D.load_capri(m)
        r = M.success_at_k(df, hib, ks=(1, 5))
        n = int(r["n_targets"].iloc[0])
        values[m] = {
            "Top-1": {q: float(r[(r.k == 1) & (r.quality == q)].success_rate.iloc[0]) for q in QUALITIES},
            "Top-5": {q: float(r[(r.k == 5) & (r.quality == q)].success_rate.iloc[0]) for q in QUALITIES},
        }
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=200)
    _grouped_stacked(ax, cols, values, models, ymax=50)
    ax.set_title(f"CAPRI Score Set (n={n})", fontsize=15)
    _legend(ax, fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_capri_topk.svg")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)
    return {"name": "Figure 1: CAPRI Score Set (n=69)", "models": models, "cols": cols, "values": values, "n": n}


def figure2(outdir, n_boot=10_000):
    models, values, n, boot = _db55_values(n_boot=n_boot)
    cols = ["Top-1", "Top-5", "Oracle"]
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=200)
    _grouped_stacked(ax, cols, values, models, ymax=50)
    _db55_errbars(ax, models, cols, boot)
    ax.set_title(f"DB5.5 (n={n})", fontsize=15)
    _legend(ax, fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_db55_topk.svg")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)
    return {"name": "Figure 2: DB5.5 (n=253)", "models": models, "cols": cols, "values": values, "boot": boot, "n": n}


def figure3(outdir, n_boot=10_000):
    meta = D.load_db55_metadata()
    ab = set(meta[meta["cat"] == "AA"]["id"])
    models, values, n, boot = _db55_values(subset_ids=ab, n_boot=n_boot)
    cols = ["Top-1", "Top-5", "Oracle"]
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=200)
    _grouped_stacked(ax, cols, values, models, ymax=50)
    _db55_errbars(ax, models, cols, boot)
    ax.set_title(f"Antibody-antigen complexes (n={n})", fontsize=15)
    _legend(ax, fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, "fig3_db55_antibody.svg")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)
    return {"name": "Figure 3: DB5.5 Antibody-Antigen Complexes (n=55)", "models": models, "cols": cols, "values": values, "boot": boot, "n": n}


def figureS2(outdir, n_boot=10_000):
    meta = D.load_db55_metadata()
    bins = D.asa_bins(meta, n_bins=5)
    labels = list(bins["asa_bin"].cat.categories)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=200)
    cols = ["Top-1", "Top-5", "Oracle"]
    bin_results = []
    for i, lab in enumerate(labels):
        ax = axes.flat[i]
        ids = set(bins[bins["asa_bin"] == lab]["id"])
        models, values, n, boot = _db55_values(subset_ids=ids, n_boot=n_boot)
        _grouped_stacked(ax, cols, values, models, ymax=50)
        _db55_errbars(ax, models, cols, boot)
        rng = bins[bins["asa_bin"] == lab]["delta_asa"]
        ax.set_title(f"({chr(97+i)}) {lab}\n({rng.min():.0f}-{rng.max():.0f} $\\AA^2$, n={n})", fontsize=13)
        bin_results.append({
            "panel": f"({chr(97+i)}) {lab}",
            "range": f"{rng.min():.0f}-{rng.max():.0f} Å²",
            "n": n,
            "models": models,
            "cols": cols,
            "values": values,
            "boot": boot,
        })
    axes.flat[-1].axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for a, b in zip(h, l):
        if b not in seen:
            seen.add(b); hh.append(a); ll.append(b)
    axes.flat[-1].legend(hh, ll, fontsize=13, loc="center",
                         frameon=True, framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()
    p = os.path.join(outdir, "figS2_db55_asa_bins.svg")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)
    return {"name": "Figure S2: DB5.5 delta-ASA Bins", "bins": bin_results}


def figure_panels(outdir):
    for ds in ("capri", "db55"):
        p = os.path.join(outdir, f"topk_panel_{ds}.svg")
        topk_curves.plot_panel(ds, 100, p)
        print("wrote", p)


def build_markdown_report(fig1_data, fig2_data, fig3_data, figS2_data) -> str:
    lines = []
    lines.append("# Bar Heights for Figures\n")
    lines.append("This document tabulates the exact bar heights (success rates in %) and bootstrap 95% confidence intervals for all bar plot figures.\n")
    lines.append("**Note on Overlay Stacking:** In all figures, bars are overlaid (Acceptable encompasses Medium, which encompasses High). The bar heights listed below represent the absolute success rate (%) for each quality tier.\n")

    if fig1_data:
        lines.append("## Figure 1: CAPRI Score Set (n=69)\n")
        lines.append("*Point estimates computed via `M.success_at_k`.*\n")
        lines.append("| Evaluation Metric | Model | Acceptable (%) | Medium (%) | High (%) |")
        lines.append("| :--- | :--- | :---: | :---: | :---: |")
        for col in fig1_data["cols"]:
            for m in fig1_data["models"]:
                v = fig1_data["values"][m][col]
                lines.append(f"| **{col}** | {m} | {v['Acceptable']:.1f}% | {v['Medium']:.1f}% | {v['High']:.1f}% |")
        lines.append("\n---\n")

    if fig2_data:
        lines.append("## Figure 2: DB5.5 (n=253)\n")
        lines.append("*Mean performance from 10,000 pose-bootstrap samples. 95% confidence intervals reported for all three quality tiers (represented as error bars on the plots).*\n")
        lines.append("| Evaluation Metric | Model | Acceptable (%) [95% CI] | Medium (%) [95% CI] | High (%) [95% CI] |")
        lines.append("| :--- | :--- | :---: | :---: | :---: |")
        for col in fig2_data["cols"]:
            for m in fig2_data["models"]:
                v = fig2_data["values"][m][col]
                b_acc = fig2_data["boot"][m][col]["Acceptable"]
                b_med = fig2_data["boot"][m][col]["Medium"]
                b_hi = fig2_data["boot"][m][col]["High"]
                acc_str = f"{v['Acceptable']:.1f}% [{b_acc[1]:.1f}, {b_acc[2]:.1f}]"
                med_str = f"{v['Medium']:.1f}% [{b_med[1]:.1f}, {b_med[2]:.1f}]"
                hi_str = f"{v['High']:.1f}% [{b_hi[1]:.1f}, {b_hi[2]:.1f}]"
                lines.append(f"| **{col}** | {m} | {acc_str} | {med_str} | {hi_str} |")
        lines.append("\n---\n")

    if fig3_data:
        lines.append("## Figure 3: DB5.5 Antibody-Antigen Complexes (n=55)\n")
        lines.append("*Mean performance from 10,000 pose-bootstrap samples (subset: `cat == 'AA'`). 95% confidence intervals reported for all three quality tiers.*\n")
        lines.append("| Evaluation Metric | Model | Acceptable (%) [95% CI] | Medium (%) [95% CI] | High (%) [95% CI] |")
        lines.append("| :--- | :--- | :---: | :---: | :---: |")
        for col in fig3_data["cols"]:
            for m in fig3_data["models"]:
                v = fig3_data["values"][m][col]
                b_acc = fig3_data["boot"][m][col]["Acceptable"]
                b_med = fig3_data["boot"][m][col]["Medium"]
                b_hi = fig3_data["boot"][m][col]["High"]
                acc_str = f"{v['Acceptable']:.1f}% [{b_acc[1]:.1f}, {b_acc[2]:.1f}]"
                med_str = f"{v['Medium']:.1f}% [{b_med[1]:.1f}, {b_med[2]:.1f}]"
                hi_str = f"{v['High']:.1f}% [{b_hi[1]:.1f}, {b_hi[2]:.1f}]"
                lines.append(f"| **{col}** | {m} | {acc_str} | {med_str} | {hi_str} |")
        lines.append("\n---\n")

    if figS2_data:
        lines.append("## Figure S2: DB5.5 Split into 5 ΔASA Bins\n")
        lines.append("*Mean performance from 10,000 pose-bootstrap samples across 5 interface area bins. 95% confidence intervals reported for all three quality tiers.*\n")
        for b in figS2_data["bins"]:
            lines.append(f"### {b['panel']} ({b['range']}, n={b['n']})\n")
            lines.append("| Evaluation Metric | Model | Acceptable (%) [95% CI] | Medium (%) [95% CI] | High (%) [95% CI] |")
            lines.append("| :--- | :--- | :---: | :---: | :---: |")
            for col in b["cols"]:
                for m in b["models"]:
                    v = b["values"][m][col]
                    b_acc = b["boot"][m][col]["Acceptable"]
                    b_med = b["boot"][m][col]["Medium"]
                    b_hi = b["boot"][m][col]["High"]
                    acc_str = f"{v['Acceptable']:.1f}% [{b_acc[1]:.1f}, {b_acc[2]:.1f}]"
                    med_str = f"{v['Medium']:.1f}% [{b_med[1]:.1f}, {b_med[2]:.1f}]"
                    hi_str = f"{v['High']:.1f}% [{b_hi[1]:.1f}, {b_hi[2]:.1f}]"
                    lines.append(f"| **{col}** | {m} | {acc_str} | {med_str} | {hi_str} |")
            lines.append("")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regenerate paper figures and bar height tables.")
    ap.add_argument("--outdir", default=os.path.join(D.REPO, "figures"))
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--only", nargs="*", choices=["1", "2", "3", "S2", "panels"])
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    want = set(a.only) if a.only else {"1", "2", "3", "S2", "panels"}

    fig1_data = figure1(a.outdir) if "1" in want else None
    fig2_data = figure2(a.outdir, a.n_boot) if "2" in want else None
    fig3_data = figure3(a.outdir, a.n_boot) if "3" in want else None
    figS2_data = figureS2(a.outdir, a.n_boot) if "S2" in want else None
    if "panels" in want:
        figure_panels(a.outdir)

    if any([fig1_data, fig2_data, fig3_data, figS2_data]):
        md_text = build_markdown_report(fig1_data, fig2_data, fig3_data, figS2_data)
        out_md = os.path.join(a.outdir, "bar_heights.md")
        with open(out_md, "w") as f:
            f.write(md_text)
        print("wrote", out_md)
