"""
Pass 1 — Entry Geometry (swings, levels, ATR, triggers)
=======================================================

Pure replay of the entry geometry on the 5-min series.  Measures and logs;
applies no threshold, stop choice, exit or P&L.

Point-in-time guarantees (the two silent traps, guarded explicitly):
  * A swing pivot at bar ``i`` is only *known* after ``K`` later bars close,
    so it becomes an active level at bar ``i + K`` and can be broken only at
    or after ``i + K`` (`replay_session`).
  * ATR at a breakout uses the RMA of True Range over candles **strictly
    before** the breakout candle — the breakout candle never inflates its own
    ATR denominator (`atr_at_entry = atr[p-1]`).
"""

from __future__ import annotations

from datetime import time as dtime
from typing import Optional

import numpy as np
import pandas as pd


# ── Indicators on the continuous (regular-session) 5-min series ──────

def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Wilder True Range. First bar has no prior close -> high-low.

    Computed on the continuous series, so the bar after an overnight gap
    carries the gap into its TR exactly as TradingView shows it.
    """
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    if n > 1:
        prev_close = close[:-1]
        hl = high[1:] - low[1:]
        hc = np.abs(high[1:] - prev_close)
        lc = np.abs(low[1:] - prev_close)
        tr[1:] = np.maximum.reduce([hl, hc, lc])
    return tr


def rma(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RMA (a.k.a. ta.rma / SMMA), matching TradingView ATR.

    Seed = simple mean of the first ``period`` values; thereafter
    ``rma[i] = (rma[i-1]*(period-1) + values[i]) / period``.
    Leading positions (< period) are NaN.
    """
    n = len(values)
    out = np.full(n, np.nan)
    if n < period:
        return out
    seed = values[:period].mean()
    out[period - 1] = seed
    prev = seed
    for i in range(period, n):
        prev = (prev * (period - 1) + values[i]) / period
        out[i] = prev
    return out


def compute_indicators(df5: pd.DataFrame, atr_period: int) -> pd.DataFrame:
    """Add ``tr``, ``atr`` (RMA) and ``avg_range`` to a symbol's continuous
    regular-session 5-min frame (sorted by time).

    ``atr[i]`` / ``avg_range[i]`` use candles up to and *including* ``i``;
    callers take the value at ``p-1`` to get "strictly before the breakout".
    """
    df = df5.sort_values("ts").reset_index(drop=True)
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    df["tr"] = true_range(h, l, c)
    df["atr"] = rma(df["tr"].to_numpy(), atr_period)
    df["avg_range"] = (df["high"] - df["low"]).rolling(atr_period, min_periods=atr_period).mean()
    return df


# ── Swing detection (strict K-bar fractal, confirmed at i+K) ─────────

def _body_top(o, c):  return max(o, c)
def _body_bottom(o, c): return min(o, c)


def detect_swings(session: pd.DataFrame, K: int) -> list[dict]:
    """Return confirmed swing pivots within one session (session-scoped).

    A swing high at session-index ``j`` requires ``high[j]`` strictly greater
    than the ``K`` highs on each side; a swing low strictly lower than the
    ``K`` lows on each side.  Confirmed at ``j + K``.
    """
    n = len(session)
    o = session["open"].to_numpy(); c = session["close"].to_numpy()
    h = session["high"].to_numpy();  l = session["low"].to_numpy()
    ts = session["ts"].to_numpy()
    pivots: list[dict] = []
    for j in range(K, n - K):
        left_h, right_h = h[j-K:j], h[j+1:j+K+1]
        left_l, right_l = l[j-K:j], l[j+1:j+K+1]
        is_high = h[j] > left_h.max() and h[j] > right_h.max()
        is_low = l[j] < left_l.min() and l[j] < right_l.min()
        if is_high:
            pivots.append({
                "type": "high", "j": j, "confirm_j": j + K, "ts": ts[j],
                "color": "green" if c[j] >= o[j] else "red",
                "body_extreme": _body_top(o[j], c[j]),  # long resistance level
                "wick": h[j],
                "body_stop": _body_bottom(o[j], c[j]),  # body-extreme for stop side
            })
        if is_low:
            pivots.append({
                "type": "low", "j": j, "confirm_j": j + K, "ts": ts[j],
                "color": "green" if c[j] >= o[j] else "red",
                "body_extreme": _body_bottom(o[j], c[j]),  # short support level
                "wick": l[j],
                "body_stop": _body_top(o[j], c[j]),
            })
    return pivots


# ── Session replay -> trigger rows ──────────────────────────────────

def _to_time(s: str) -> dtime:
    h, m = map(int, s.split(":"))
    return dtime(h, m)


def replay_session(
    session: pd.DataFrame,
    full: pd.DataFrame,
    sess_global_pos: np.ndarray,
    cfg: dict,
    *,
    symbol: str,
    date,
    r_rank: int,
) -> list[dict]:
    """Replay one (symbol, day) session and return one row per trigger.

    Parameters
    ----------
    session : per-day regular-session 5-min candles (sorted).
    full : the symbol's continuous indicator frame (has atr/avg_range).
    sess_global_pos : for each session row, its integer position in ``full``.
    """
    K = cfg["swing_K"]
    win_start = _to_time(cfg["entry_window_start"])
    win_end = _to_time(cfg["entry_window_end"])
    third = cfg["third_candle_back"]

    n = len(session)
    o = session["open"].to_numpy(); c = session["close"].to_numpy()
    h = session["high"].to_numpy(); l = session["low"].to_numpy()
    ts = session["ts"].to_numpy()
    open_0915 = o[0]

    atr_arr = full["atr"].to_numpy()
    avgr_arr = full["avg_range"].to_numpy()
    full_low = full["low"].to_numpy()
    full_high = full["high"].to_numpy()

    pivots = detect_swings(session, K)
    confirm_map: dict[int, list[dict]] = {}
    for p in pivots:
        confirm_map.setdefault(p["confirm_j"], []).append(p)

    active_long: Optional[dict] = None    # most-recent confirmed unbroken high-level
    active_short: Optional[dict] = None
    recent_swing_low: Optional[dict] = None   # protective stop reference (longs)
    recent_swing_high: Optional[dict] = None  # protective stop reference (shorts)

    rows: list[dict] = []

    for cbar in range(n):
        # 1) apply confirmations that become known at this bar
        for p in confirm_map.get(cbar, []):
            if p["type"] == "high":
                active_long = {**p, "confirm_bar": cbar}
                recent_swing_high = p
            else:
                active_short = {**p, "confirm_bar": cbar}
                recent_swing_low = p

        # 2) triggers only inside the entry window
        tt = pd.Timestamp(ts[cbar]).time()
        if not (win_start <= tt <= win_end):
            continue
        gp = int(sess_global_pos[cbar])

        for direction, lvl in (("long", active_long), ("short", active_short)):
            if lvl is None:
                continue
            broke = (c[cbar] > lvl["body_extreme"]) if direction == "long" \
                else (c[cbar] < lvl["body_extreme"])
            if not broke:
                continue

            entry = c[cbar]
            bo_range = h[cbar] - l[cbar]
            atr_at_entry = atr_arr[gp - 1] if gp - 1 >= 0 else np.nan
            avg_range_at_entry = avgr_arr[gp - 1] if gp - 1 >= 0 else np.nan
            tb_pos = gp - third
            if direction == "long":
                third_back_stop = full_low[tb_pos] if tb_pos >= 0 else np.nan
                bo_candle_stop = l[cbar]
                sw = recent_swing_low
                swing_wick = sw["wick"] if sw else np.nan
                swing_body = sw["body_stop"] if sw else np.nan
            else:
                third_back_stop = full_high[tb_pos] if tb_pos >= 0 else np.nan
                bo_candle_stop = h[cbar]
                sw = recent_swing_high
                swing_wick = sw["wick"] if sw else np.nan
                swing_body = sw["body_stop"] if sw else np.nan

            def dist(stop):
                return abs(entry - stop) / entry if pd.notna(stop) else np.nan

            rows.append({
                "date": pd.Timestamp(date),
                "trigger_time": pd.Timestamp(ts[cbar]).strftime("%H:%M"),
                "symbol": symbol,
                "direction": direction,
                "r_rank": r_rank,
                "pos_vs_open": "above" if entry > open_0915 else "below",
                "open_0915": open_0915,
                "level_price": lvl["body_extreme"],
                "swing_candle_time": pd.Timestamp(lvl["ts"]).strftime("%H:%M"),
                "swing_candle_color": lvl["color"],
                "bars_since_swing": cbar - lvl["confirm_bar"],
                "bo_open": o[cbar], "bo_high": h[cbar], "bo_low": l[cbar], "bo_close": c[cbar],
                "bo_range": bo_range,
                "entry_price": entry,
                "atr_at_entry": atr_at_entry,
                "avg_range_at_entry": avg_range_at_entry,
                "atr_ratio": bo_range / atr_at_entry if atr_at_entry and pd.notna(atr_at_entry) else np.nan,
                "range_ratio": bo_range / avg_range_at_entry if avg_range_at_entry and pd.notna(avg_range_at_entry) else np.nan,
                # stop candidates (raw — not chosen)
                "swing_stop_price_wick": swing_wick,
                "swing_stop_price_body": swing_body,
                "third_back_stop_price": third_back_stop,
                "bo_candle_stop_price": bo_candle_stop,
                "stop_dist_swing_wick_pct": dist(swing_wick),
                "stop_dist_swing_body_pct": dist(swing_body),
                "stop_dist_thirdback_pct": dist(third_back_stop),
                "stop_dist_bocandle_pct": dist(bo_candle_stop),
            })

            # spend the level: it is broken and never reused
            if direction == "long":
                active_long = None
            else:
                active_short = None

    return rows
