"""
Pass 2 — Trade Metrics
======================

Headline metrics from a per-filter trade log (Section 7).  Expectancy in R
is the verdict; win rate is expected to look low by design (3R target / 1R
stop).  All gross, spot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown_R(realized_R: np.ndarray):
    """Peak-to-trough drawdown of the cumulative-R equity curve.

    Returns (max_dd_R, duration_trades).
    """
    if len(realized_R) == 0:
        return 0.0, 0
    equity = np.cumsum(realized_R)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    trough = int(np.argmax(dd))
    max_dd = float(dd[trough])
    # peak index preceding the trough
    peak_idx = int(np.argmax(equity[:trough + 1])) if trough > 0 else 0
    return max_dd, trough - peak_idx


def _longest_losing_streak(realized_R: np.ndarray) -> int:
    streak = best = 0
    for r in realized_R:
        if r < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def compute(log: pd.DataFrame, n_basket_days: int) -> dict:
    """Headline metrics for one filter's trade log."""
    n = len(log)
    if n == 0:
        return {"n_trades": 0, "no_trade_days": n_basket_days,
                "expectancy_R": float("nan")}

    R = log["realized_R"].to_numpy()
    wins = log[log["realized_R"] > 0]
    losses = log[log["realized_R"] < 0]
    reasons = log["exit_reason"].value_counts()
    time_closed = log[log["exit_reason"] == "time"]
    max_dd, dd_dur = _max_drawdown_R(R)

    return {
        "n_trades": n,
        "no_trade_days": n_basket_days - n,
        "expectancy_R": float(R.mean()),                 # the verdict
        "total_R": float(R.sum()),
        "win_rate": float((R > 0).mean()),
        "avg_win_R": float(wins["realized_R"].mean()) if len(wins) else 0.0,
        "avg_loss_R": float(losses["realized_R"].mean()) if len(losses) else 0.0,
        "pct_target": float(reasons.get("target", 0) / n),
        "pct_stop": float(reasons.get("stop", 0) / n),
        "pct_breakeven": float(reasons.get("breakeven", 0) / n),
        "pct_time": float(reasons.get("time", 0) / n),
        "n_breakeven": int(reasons.get("breakeven", 0)),
        "time_close_mean_R": float(time_closed["realized_R"].mean()) if len(time_closed) else float("nan"),
        "max_drawdown_R": max_dd,
        "max_dd_duration_trades": dd_dur,
        "longest_losing_streak": _longest_losing_streak(R),
        "avg_stop_dist_pct": float(log["stop_dist_pct"].mean()),
        "n_long": int((log["direction"] == "long").sum()),
        "n_short": int((log["direction"] == "short").sum()),
        "n_tie_stopfirst": int(log["tie_stopfirst"].sum()),
    }


def comparison_table(metrics_by_filter: dict[str, dict]) -> pd.DataFrame:
    """Two filters' metrics side by side (rows = metric, cols = filter)."""
    df = pd.DataFrame(metrics_by_filter)
    df.index.name = "metric"
    return df.reset_index()
