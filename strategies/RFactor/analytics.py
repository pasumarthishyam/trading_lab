"""
R-Factor Backtest — Analytics
=============================

Pure functions that turn the saved substrate (``picks`` + ``daily``) into
the seven reporting outputs (Section 6 of the spec).  No plotting here —
the notebook imports these and renders them.

Conventions
-----------
* Rates are **pooled** over the test period: e.g. the top-N hit rate at a
  checkpoint is ``mean(hit_2pct)`` over every pick at that checkpoint across
  all days (not a per-day average of rates).  The base rate is pooled the
  same way over every eligible (symbol, day), so the two are comparable and
  ``lift = top_n_hit - base_rate``.
* The base rate is a full-day measure and therefore identical across
  checkpoints; it is reported per checkpoint (a flat control line) so lift
  can be read off at each checkpoint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def base_rate(daily: pd.DataFrame) -> float:
    """Pooled ≥2% hit rate across all eligible (symbol, day)."""
    elig = daily[daily["eligible"]]
    return float(elig["hit_2pct"].mean()) if len(elig) else float("nan")


def hit_rate_curve(picks: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Output #1 + #2: hit-rate curves and lift, per checkpoint.

    Columns: checkpoint, n_picks, topn_hit, top5_hit, base_rate,
             lift_topn, lift_top5.
    """
    br = base_rate(daily)
    rows = []
    for cp, g in picks.groupby("checkpoint"):
        topn = g["hit_2pct"].mean()
        top5 = g.loc[g["in_top5"], "hit_2pct"].mean()
        rows.append({
            "checkpoint": cp,
            "n_picks": len(g),
            "topn_hit": topn,
            "top5_hit": top5,
            "base_rate": br,
            "lift_topn": topn - br,
            "lift_top5": top5 - br,
        })
    return pd.DataFrame(rows).sort_values("checkpoint").reset_index(drop=True)


def capturable_curve(picks: pd.DataFrame) -> pd.DataFrame:
    """Output #3: capturable-after-checkpoint, per checkpoint.

    Columns: checkpoint, median_fav_move, mean_fav_move, pct_capturable.
    """
    rows = []
    for cp, g in picks.groupby("checkpoint"):
        fav = g["fav_move_after_cp_pct"].dropna()
        rows.append({
            "checkpoint": cp,
            "median_fav_move": fav.median(),
            "mean_fav_move": fav.mean(),
            "pct_capturable": g["capturable"].mean(),
        })
    return pd.DataFrame(rows).sort_values("checkpoint").reset_index(drop=True)


def churn_curve(picks: pd.DataFrame, checkpoints: list[str]) -> pd.DataFrame:
    """Output #4: persistence of the top-N from one checkpoint to the next.

    For each day, the fraction of checkpoint i's top-N that is still in the
    top-N at checkpoint i+1, averaged over days.  ``persistence`` is that
    fraction; ``churn`` = 1 - persistence.
    """
    sets = {
        (d, cp): set(g["symbol"])
        for (d, cp), g in picks.groupby(["date", "checkpoint"])
    }
    days = picks["date"].unique()
    rows = []
    for i in range(len(checkpoints) - 1):
        cp_i, cp_j = checkpoints[i], checkpoints[i + 1]
        fracs = []
        for d in days:
            a = sets.get((d, cp_i), set())
            b = sets.get((d, cp_j), set())
            if a:
                fracs.append(len(a & b) / len(a))
        persistence = float(np.mean(fracs)) if fracs else float("nan")
        rows.append({
            "checkpoint": cp_i,
            "next_checkpoint": cp_j,
            "persistence": persistence,
            "churn": 1 - persistence,
        })
    return pd.DataFrame(rows)


def picks_per_day_distribution(picks: pd.DataFrame) -> pd.DataFrame:
    """Output #5: per (day, checkpoint), how many of the top-N hit ≥2%.

    Returns one row per (date, checkpoint) with ``n_hits`` and ``n_picks``.
    """
    g = picks.groupby(["date", "checkpoint"])
    out = g.agg(n_hits=("hit_2pct", "sum"),
                n_picks=("hit_2pct", "size")).reset_index()
    return out


def direction_split(picks: pd.DataFrame) -> pd.DataFrame:
    """Output #6: up vs down vs both among hitting picks, per checkpoint."""
    rows = []
    for cp, g in picks.groupby("checkpoint"):
        hits = g[g["hit_2pct"]]
        n = len(hits)
        rows.append({
            "checkpoint": cp,
            "n_hits": n,
            "pct_up": (hits["hit_direction"] == "up").mean() if n else np.nan,
            "pct_down": (hits["hit_direction"] == "down").mean() if n else np.nan,
            "pct_both": (hits["hit_direction"] == "both").mean() if n else np.nan,
        })
    return pd.DataFrame(rows).sort_values("checkpoint").reset_index(drop=True)


def magnitude_distribution(picks: pd.DataFrame) -> pd.DataFrame:
    """Output #7: how far picks travelled (max abs move from 9:15), per cp.

    Columns: checkpoint, pct_ge_2, pct_ge_3, pct_ge_5, median_abs_move.
    """
    rows = []
    p = picks.copy()
    p["abs_move"] = p[["max_up_pct", "max_down_pct"]].max(axis=1)
    for cp, g in p.groupby("checkpoint"):
        rows.append({
            "checkpoint": cp,
            "pct_ge_2": (g["abs_move"] >= 0.02).mean(),
            "pct_ge_3": (g["abs_move"] >= 0.03).mean(),
            "pct_ge_5": (g["abs_move"] >= 0.05).mean(),
            "median_abs_move": g["abs_move"].median(),
        })
    return pd.DataFrame(rows).sort_values("checkpoint").reset_index(drop=True)


def build_summary(picks: pd.DataFrame, daily: pd.DataFrame,
                  checkpoints: list[str]) -> pd.DataFrame:
    """The one-look verdict table: hit / base / lift / capturable per cp."""
    hr = hit_rate_curve(picks, daily)
    cc = capturable_curve(picks)
    ch = churn_curve(picks, checkpoints)[["checkpoint", "churn"]]
    ppd = picks_per_day_distribution(picks).groupby("checkpoint")["n_hits"].mean()

    out = (hr.merge(cc, on="checkpoint", how="left")
             .merge(ch, on="checkpoint", how="left"))
    out["mean_picks_hit"] = out["checkpoint"].map(ppd)
    cols = ["checkpoint", "n_picks", "topn_hit", "top5_hit", "base_rate",
            "lift_topn", "lift_top5", "median_fav_move", "pct_capturable",
            "mean_picks_hit", "churn"]
    return out[cols].sort_values("checkpoint").reset_index(drop=True)
