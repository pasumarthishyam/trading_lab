"""
Pass 1 — Distributions & Audit
==============================

Turns the trigger table into the measurement outputs (Section 9): the
percentile tables and histogram data for the "huge candle" metrics and the
stop-distance candidates, plus the audit sample the user opens on
TradingView.  No thresholds are applied — these are the shapes from which
thresholds get set later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Metrics whose distributions set the thresholds.
HUGE_METRICS = ["atr_ratio", "range_ratio"]
STOP_METRICS = ["stop_dist_swing_wick_pct", "stop_dist_swing_body_pct",
                "stop_dist_thirdback_pct", "stop_dist_bocandle_pct"]
PCTS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def percentile_table(trig: pd.DataFrame, cols=None, pcts=None) -> pd.DataFrame:
    """Percentiles (+ mean/std/n) for each metric column."""
    cols = cols or (HUGE_METRICS + STOP_METRICS)
    pcts = pcts or PCTS
    rows = []
    for c in cols:
        s = trig[c].dropna()
        row = {"metric": c, "n": len(s), "mean": s.mean(), "std": s.std()}
        for p in pcts:
            row[f"p{int(p*100)}"] = s.quantile(p)
        rows.append(row)
    return pd.DataFrame(rows)


def huge_threshold_readoff(trig: pd.DataFrame) -> pd.DataFrame:
    """The explicit median/75/90/95 read-off for the 'huge' metrics (Section 9.2)."""
    out = []
    for c in HUGE_METRICS:
        s = trig[c].dropna()
        out.append({
            "metric": c,
            "median": s.median(), "p75": s.quantile(.75),
            "p90": s.quantile(.90), "p95": s.quantile(.95),
            "pct_over_1x": (s > 1).mean(),
            "pct_over_1_5x": (s > 1.5).mean(),
            "pct_over_2x": (s > 2).mean(),
        })
    return pd.DataFrame(out)


def sanity_stats(trig: pd.DataFrame, n_test_days: int) -> dict:
    per_day = trig.groupby("date").size()
    ls = trig["direction"].value_counts()
    pv = trig.groupby(["direction", "pos_vs_open"]).size()
    return {
        "n_triggers": len(trig),
        "triggers_per_day_mean": float(per_day.mean()),
        "triggers_per_day_median": float(per_day.median()),
        "triggers_per_day_max": int(per_day.max()),
        "days_with_trigger": int(per_day.index.nunique()),
        "n_test_days": n_test_days,
        "pct_days_with_trigger": 100 * per_day.index.nunique() / n_test_days,
        "long": int(ls.get("long", 0)),
        "short": int(ls.get("short", 0)),
        "pos_vs_open": {f"{d}_{p}": int(v) for (d, p), v in pv.items()},
    }


def audit_sample(trig: pd.DataFrame, n_extreme: int = 25, n_random: int = 25,
                 seed: int = 7) -> pd.DataFrame:
    """Human-readable audit list: the most extreme atr_ratio rows (candidate
    'huge' candles) + a random sample of normal ones, stamped with what the
    user needs to find the bar on TradingView."""
    cols = ["date", "trigger_time", "symbol", "direction", "r_rank",
            "pos_vs_open", "swing_candle_time", "level_price", "entry_price",
            "bo_range", "atr_at_entry", "atr_ratio", "range_ratio"]
    t = trig.dropna(subset=["atr_ratio"]).copy()
    extreme = t.nlargest(n_extreme, "atr_ratio").assign(audit_bucket="extreme_huge")
    rest = t.drop(extreme.index)
    rnd = rest.sample(min(n_random, len(rest)), random_state=seed).assign(audit_bucket="random_normal")
    out = pd.concat([extreme, rnd])[["audit_bucket"] + cols]
    return out.sort_values(["audit_bucket", "atr_ratio"], ascending=[True, False]).reset_index(drop=True)


def histogram_data(trig: pd.DataFrame, col: str, clip_q: float = 0.99):
    """Return clipped values for a histogram (clip the long right tail for display)."""
    s = trig[col].dropna()
    hi = s.quantile(clip_q)
    return np.clip(s, 0, hi), hi
