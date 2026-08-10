"""Core ranking metrics for LambdaDockScore.

Shared by every figure/analysis script in this directory so that the paper
figures and the follow-up experiments are computed by *identical* code.

Conventions
-----------
* A "pose set" is a long-format DataFrame with columns ``id`` (target),
  ``score`` (model output) and ``DockQ`` (ground truth).
* ``higher_is_better`` says how ``score`` orders poses. DFMDock/LambdaDockScore
  emit an *energy* (lower = better); EuDockScore emits a score (higher = better).
* CAPRI quality thresholds are strict ``>`` comparisons, matching the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# CAPRI quality thresholds (Basu & Wallner 2016; Lensink et al. 2007)
THRESHOLDS: dict[str, float] = {
    "Acceptable": 0.23,
    "Medium": 0.49,
    "High": 0.80,
}

# Fixed entity -> color map. Never reassign by rank; a model keeps its hue
# across every figure in the paper. Shades run light->dark with quality tier.
MODEL_COLORS: dict[str, dict[str, str]] = {
    "Baseline DFMDock": {"Acceptable": "#b8d8ee", "Medium": "#3b7ebf", "High": "#12385f"},
    "LambdaDockScore": {"Acceptable": "#f6b8b4", "Medium": "#e03127", "High": "#7d1410"},
    "EuDockScore":     {"Acceptable": "#d5c4e3", "Medium": "#8a63ab", "High": "#432459"},
    "PIsToN":          {"Acceptable": "#bfe3cf", "Medium": "#3f9e6a", "High": "#17492f"},
}


def top_k_poses(group: pd.DataFrame, higher_is_better: bool, k: int) -> pd.DataFrame:
    """The k best-scoring poses for one target.

    Uses ``nsmallest``/``nlargest`` rather than ``sort_values().head(k)``: the
    DFMDock energy columns contain many exact within-target ties, and the two
    approaches break those ties differently.
    """
    k = min(k, len(group))
    return group.nlargest(k, "score") if higher_is_better else group.nsmallest(k, "score")


def rank_poses(group: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    """Sort one target's poses best-first (full ordering)."""
    return group.sort_values("score", ascending=not higher_is_better, kind="stable")


def success_at_k(
    df: pd.DataFrame,
    higher_is_better: bool,
    ks=(1, 5),
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Top-k success rate (%) per quality threshold.

    A target counts as a success at k if at least one of its k best-ranked
    poses exceeds the DockQ threshold.

    Returns a tidy frame with columns ``k``, ``quality``, ``success_rate``,
    ``n_success``, ``n_targets``.
    """
    thresholds = thresholds or THRESHOLDS
    ordered = {
        tid: rank_poses(g, higher_is_better)["DockQ"].to_numpy()
        for tid, g in df.groupby("id")
    }
    n = len(ordered)
    rows = []
    for k in ks:
        for quality, t in thresholds.items():
            hits = sum(1 for dq in ordered.values() if (dq[:k] > t).any())
            rows.append(
                {
                    "k": k,
                    "quality": quality,
                    "n_success": hits,
                    "n_targets": n,
                    "success_rate": 100.0 * hits / n if n else 0.0,
                }
            )
    return pd.DataFrame(rows)


def oracle(df: pd.DataFrame, thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    """Best achievable success rate given the sampled pose set (perfect ranking)."""
    thresholds = thresholds or THRESHOLDS
    per_target = df.groupby("id")["DockQ"].max()
    n = len(per_target)
    return pd.DataFrame(
        [
            {
                "quality": q,
                "n_success": int((per_target > t).sum()),
                "n_targets": n,
                "success_rate": 100.0 * (per_target > t).sum() / n if n else 0.0,
            }
            for q, t in thresholds.items()
        ]
    )


def bootstrap_ci(
    df: pd.DataFrame,
    higher_is_better: bool,
    k: int,
    quality: str = "Acceptable",
    n_boot: int = 10_000,
    ci: float = 95.0,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap over *targets* for a single (k, quality) cell.

    NOTE: this is **not** the statistic behind Figures 2/3/S2 -- see
    ``pose_bootstrap``. Resampling targets leaves the mean equal to the point
    estimate, so it widens error bars without shifting bar heights. Kept for
    target-level uncertainty questions ("would another 253 complexes agree?").
    """
    t = THRESHOLDS[quality]
    per_target = np.array(
        [
            float((rank_poses(g, higher_is_better)["DockQ"].to_numpy()[:k] > t).any())
            for _, g in df.groupby("id")
        ]
    )
    n = len(per_target)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    draws = per_target[rng.integers(0, n, size=(n_boot, n))].mean(axis=1) * 100.0
    lo, hi = np.percentile(draws, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(per_target.mean() * 100.0), float(lo), float(hi)


def pose_bootstrap(
    df: pd.DataFrame,
    higher_is_better: bool,
    ks=(1, 5),
    quality: str = "Acceptable",
    n_boot: int = 10_000,
    ci: float = 95.0,
    seed: int = 0,
    include_oracle: bool = True,
) -> pd.DataFrame:
    """Bootstrap over *poses within each target* -- the statistic used in the paper.

    This is what Figures 2, 3 and S2 report ("mean performance from 10,000
    bootstrap samples"). Each replicate draws, for every target, ``m`` poses with
    replacement from that target's ``m`` sampled poses, re-ranks them by score,
    and recomputes success. It answers "how much does the result depend on which
    poses the sampler happened to produce?".

    Because a bootstrap resample of ``m`` items contains only ~63% of the distinct
    poses, this **lowers** the mean relative to the point estimate whenever good
    poses are rare -- which is why the paper's oracle (31.1%) sits below the point
    estimate (36.0%).

    Figure 1 (CAPRI) has no error bars and is *not* bootstrapped -- use
    ``success_at_k`` for it.

    Returns a tidy frame with ``k`` ("oracle" for the full set), ``mean``,
    ``lo``, ``hi``, ``n_targets``.
    """
    t = THRESHOLDS[quality]
    targets = [
        (g["score"].to_numpy(), g["DockQ"].to_numpy()) for _, g in df.groupby("id")
    ]
    n = len(targets)
    if n == 0:
        return pd.DataFrame(columns=["k", "mean", "lo", "hi", "n_targets"])

    rng = np.random.default_rng(seed)
    cols = list(ks) + (["oracle"] if include_oracle else [])
    # hit_counts[c][b] = how many targets succeed in replicate b
    hit_counts = {c: np.zeros(n_boot, dtype=np.int32) for c in cols}
    sign = -1.0 if higher_is_better else 1.0  # ascending sort on sign*score
    kmax = max(ks)

    # Vectorised over replicates, looped over targets: for each target draw an
    # (n_boot, m) index matrix at once. Memory stays at one target's worth.
    for score, dockq in targets:
        m = len(score)
        idx = rng.integers(0, m, size=(n_boot, m))
        s = sign * score[idx]                      # (n_boot, m)
        d = dockq[idx]
        # only the top-kmax ranks matter for the top-k columns
        part = np.argpartition(s, min(kmax, m - 1), axis=1)[:, :kmax]
        top = np.take_along_axis(d, part, axis=1)  # (n_boot, kmax), unordered
        top_s = np.take_along_axis(s, part, axis=1)
        order = np.argsort(top_s, axis=1, kind="stable")
        top = np.take_along_axis(top, order, axis=1)
        good = top > t
        for k in ks:
            hit_counts[k] += good[:, :k].any(axis=1)
        if include_oracle:
            hit_counts["oracle"] += (d > t).any(axis=1)

    draws = {c: 100.0 * hit_counts[c] / n for c in cols}
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    return pd.DataFrame(
        [
            {
                "k": c,
                "quality": quality,
                "mean": float(draws[c].mean()),
                "lo": float(np.percentile(draws[c], lo_q)),
                "hi": float(np.percentile(draws[c], hi_q)),
                "n_targets": n,
            }
            for c in cols
        ]
    )


def success_curve(
    df: pd.DataFrame,
    higher_is_better: bool,
    k_max: int | None = None,
    quality: str = "Acceptable",
) -> pd.DataFrame:
    """Top-k success rate as a *function of k* — the success-vs-k curve.

    ``k_max`` defaults to the smallest pose count across targets, so every
    target contributes at every k (no survivorship bias as k grows).
    """
    t = THRESHOLDS[quality]
    ordered = [
        rank_poses(g, higher_is_better)["DockQ"].to_numpy() for _, g in df.groupby("id")
    ]
    n = len(ordered)
    limit = min(len(dq) for dq in ordered) if k_max is None else k_max
    # first_hit[i] = smallest k at which target i succeeds (inf if never)
    first_hit = np.array(
        [
            (np.argmax(dq > t) + 1) if (dq > t).any() else np.inf
            for dq in ordered
        ]
    )
    ks = np.arange(1, limit + 1)
    rates = [(100.0 * (first_hit <= k).sum() / n) if n else 0.0 for k in ks]
    return pd.DataFrame({"k": ks, "success_rate": rates, "n_targets": n, "quality": quality})


def enrichment(df: pd.DataFrame, higher_is_better: bool, quality: str = "Acceptable") -> float:
    """Success rate at k=1 divided by the random-selection baseline.

    Random baseline = overall fraction of acceptable poses in the set. >1 means
    the scoring function concentrates good poses at the top.
    """
    t = THRESHOLDS[quality]
    random_rate = (df["DockQ"] > t).mean() * 100.0
    top1 = success_at_k(df, higher_is_better, ks=(1,), thresholds={quality: t})
    return float(top1["success_rate"].iloc[0] / random_rate) if random_rate else np.nan
