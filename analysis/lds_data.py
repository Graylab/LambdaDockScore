"""Loaders that normalise every result source to the same schema.

Every loader returns a long DataFrame with columns ``id``, ``pose_id``,
``score``, ``DockQ`` plus a ``higher_is_better`` flag reported alongside, so
downstream metric code never has to know where the numbers came from.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Which result set to analyse. Defaults to the per-pose CSVs in results/ that
# back the published figures; point LDS_RESULTS at another directory with the
# same layout to re-run every analysis against a different scoring run. Every
# script in analysis/ reads through here, so the env var re-points the figures,
# the top-k curves and the by-type breakdown at once.
RESULTS = os.environ.get("LDS_RESULTS") or os.path.join(REPO, "results")
if not os.path.isabs(RESULTS):
    RESULTS = os.path.join(REPO, RESULTS)

# name -> (relative path, score column, higher_is_better)
CAPRI_SOURCES: dict[str, tuple[str, bool]] = {
    "Baseline DFMDock": ("capri_score_set/dfmdock_baseline.csv", False),
    "LambdaDockScore": ("capri_score_set/lambdadockscore.csv", False),
    "EuDockScore": ("capri_score_set/eudockscore.csv", True),
}


def load_capri(model: str) -> tuple[pd.DataFrame, bool]:
    """Load one model's CAPRI score-set predictions.

    All three CSVs are pre-aligned to the *same* 76,078 (target, decoy) pairs
    over 69 targets, so top-k numbers are directly comparable across models.
    """
    rel, higher = CAPRI_SOURCES[model]
    df = pd.read_csv(os.path.join(RESULTS, rel))
    df = df.rename(columns={"energy": "score", "File_Name": "pose_id"})
    return df[["id", "pose_id", "score", "DockQ"]], higher


def load_all_capri() -> dict[str, tuple[pd.DataFrame, bool]]:
    return {m: load_capri(m) for m in CAPRI_SOURCES}


def load_db55(model: str) -> tuple[pd.DataFrame, bool]:
    """Load DB5.5 poses sampled by baseline DFMDock, scored by ``model``.

    Both models rank the *same* 120 sampled poses per complex, so differences
    are attributable to ranking alone (the oracle is identical by construction).
    """
    sub, col = {
        "Baseline DFMDock": ("baseline", "baseline_energy"),
        "LambdaDockScore": ("lambdadockscore", "ranking_energy"),
    }[model]
    files = sorted(glob.glob(os.path.join(RESULTS, "db55", sub, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no DB5.5 CSVs under results/db55/{sub}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.rename(columns={col: "score"})
    df["pose_id"] = df["id"].astype(str) + "_" + df.index.astype(str)
    return df[["id", "pose_id", "score", "DockQ"]], False


def load_db55_metadata() -> pd.DataFrame:
    """DB5.5 per-complex metadata: category and interface size (delta ASA)."""
    m = pd.read_csv(os.path.join(RESULTS, "metadata", "db55_all_metadata.csv"))
    return m.rename(columns={"complex_id": "id"})


# DB5.5 category codes -> readable names (Vreven et al. 2015)
CATEGORY_NAMES: dict[str, str] = {
    "AA": "Antibody-Antigen",
    "AS": "Antigen-Bound Antibody",
    "EI": "Enzyme-Inhibitor",
    "ES": "Enzyme-Substrate",
    "ER": "Enzyme-Regulator",
    "OG": "Others, G-protein",
    "OR": "Others, Receptor",
    "OX": "Others, Miscellaneous",
}


def attach_db55_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Join category / delta_asa / difficulty onto a DB5.5 pose frame."""
    meta = load_db55_metadata()
    out = df.merge(meta[["id", "cat", "delta_asa", "difficulty"]], on="id", how="left")
    out["category"] = out["cat"].map(CATEGORY_NAMES).fillna(out["cat"])
    return out


def asa_bins(meta: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Split complexes into equal-count delta-ASA bins (Figure S2's scheme)."""
    m = meta.dropna(subset=["delta_asa"]).copy()
    m["asa_bin"] = pd.qcut(
        m["delta_asa"], n_bins, labels=[f"{i*100//n_bins}-{(i+1)*100//n_bins}th pct" for i in range(n_bins)]
    )
    return m[["id", "delta_asa", "asa_bin"]]