"""
Pass 2 — Full Trade Backtest (spot, gross) — Engine
===================================================

Layers the trade onto the Pass 1 geometry: qualifiers (huge filter, stop
cap, momentum filter), first-qualifying-trigger selection (one trade/day),
and forward resolution (3R target / 1R stop / 3:00 force-close).

Locked rules (this run):
  * Stop = **farther** of {swing-body, 3rd-candle-back} (avoids degenerate
    near-0 stops); no strong-candle exception.
  * Huge filter: skip trigger if breakout candle > 2x ATR.
  * Stop cap: skip trigger if the chosen stop is > 1% from entry.
  * Two momentum filters, run separately (A = 9:15-extreme held, B = side of
    open). A trigger qualifies only if direction is allowed AND not huge AND
    stop <= 1%.
  * Resolution on the 5-min series; if a candle spans both target and stop,
    assume **stop-first** (conservative); count these ties.

Point-in-time: everything derives from Pass 1 triggers (which already honour
the swing-confirmation lag and breakout-excluded ATR); momentum filters use
only candles up to the entry candle; resolution walks strictly forward.
"""

from __future__ import annotations

from datetime import time as dtime

import numpy as np
import pandas as pd

from strategies.EntryGeometry import geometry


def _t(s: str) -> dtime:
    h, m = map(int, s.split(":"))
    return dtime(h, m)


def _choose_stop(direction, entry, swing_body, third_back):
    """Farther of {swing-body, 3rd-candle-back} from entry.

    Returns (stop_price, stop_dist_pct, which). NaN swing_body -> use 3rd-back.
    """
    d_sw = abs(entry - swing_body) / entry if pd.notna(swing_body) else np.nan
    d_th = abs(entry - third_back) / entry if pd.notna(third_back) else np.nan
    if pd.isna(d_sw) and pd.isna(d_th):
        return np.nan, np.nan, "none"
    if pd.isna(d_sw) or (pd.notna(d_th) and d_th > d_sw):
        return third_back, d_th, "third_back"
    return swing_body, d_sw, "swing_body"


def _resolve(direction, entry, stop, target, sess, c_idx, force_close_t):
    """Walk 5-min forward from the candle after entry to force-close.

    Returns (exit_time, exit_price, exit_reason, realized_R, tie).
    """
    h = sess["high"].to_numpy(); l = sess["low"].to_numpy()
    o = sess["open"].to_numpy(); c = sess["close"].to_numpy()
    ts = sess["ts"].to_numpy()
    n = len(sess)
    R = abs(entry - stop)
    sign = 1.0 if direction == "long" else -1.0

    for k in range(c_idx + 1, n):
        if pd.Timestamp(ts[k]).time() >= force_close_t:
            # force-close at the 3:00 candle's open
            exitp = o[k]
            return (pd.Timestamp(ts[k]).strftime("%H:%M"), exitp, "time",
                    sign * (exitp - entry) / R, False)
        if direction == "long":
            hit_stop = l[k] <= stop
            hit_tgt = h[k] >= target
        else:
            hit_stop = h[k] >= stop
            hit_tgt = l[k] <= target
        if hit_stop and hit_tgt:                      # ambiguous -> stop-first
            return (pd.Timestamp(ts[k]).strftime("%H:%M"), stop, "stop", -1.0, True)
        if hit_tgt:
            return (pd.Timestamp(ts[k]).strftime("%H:%M"), target, "target", 3.0, False)
        if hit_stop:
            return (pd.Timestamp(ts[k]).strftime("%H:%M"), stop, "stop", -1.0, False)

    # no force-close candle (partial day) -> exit at last close
    exitp = c[-1]
    return (pd.Timestamp(ts[-1]).strftime("%H:%M"), exitp, "time",
            sign * (exitp - entry) / R, False)


def enrich_session_triggers(base_rows, sess, cfg):
    """Enrich Pass 1 triggers for one (symbol, day) session with the Pass 2
    stop, target, qualifiers, both momentum-filter flags, and the resolved
    outcome (computed once per trigger, independent of filter/selection)."""
    if not base_rows:
        return []
    o = sess["open"].to_numpy(); h = sess["high"].to_numpy(); l = sess["low"].to_numpy()
    ts = sess["ts"].to_numpy()
    open0, low0, high0 = o[0], l[0], h[0]
    cummin_low = np.minimum.accumulate(l)
    cummax_high = np.maximum.accumulate(h)
    time_to_idx = {pd.Timestamp(t).strftime("%H:%M"): i for i, t in enumerate(ts)}

    huge_mult = cfg["huge_atr_mult"]
    stop_max = cfg["stop_max_pct"]
    target_R = cfg["target_R"]
    fc_t = _t(cfg["force_close"])

    out = []
    for r in base_rows:
        c_idx = time_to_idx.get(r["trigger_time"])
        if c_idx is None:
            continue
        direction = r["direction"]
        entry = r["entry_price"]

        # momentum filters (point-in-time, up to entry candle)
        if direction == "long":
            filterA_ok = bool(cummin_low[c_idx] >= low0 - 1e-9)   # 9:15 low still the low
            filterB_ok = bool(entry > open0)
        else:
            filterA_ok = bool(cummax_high[c_idx] <= high0 + 1e-9)  # 9:15 high still the high
            filterB_ok = bool(entry < open0)

        # stop = farther of swing-body / 3rd-back
        stop, stop_dist, which = _choose_stop(
            direction, entry, r["swing_stop_price_body"], r["third_back_stop_price"])
        qual_huge = bool(r["atr_ratio"] <= huge_mult) if pd.notna(r["atr_ratio"]) else False
        qual_stop = bool(pd.notna(stop_dist) and stop_dist <= stop_max)

        row = {
            "date": r["date"], "symbol": r["symbol"], "direction": direction,
            "r_rank": r["r_rank"], "trigger_time": r["trigger_time"],
            "entry_price": entry, "level_price": r["level_price"],
            "atr_ratio": r["atr_ratio"], "pos_vs_open": r["pos_vs_open"],
            "stop_price": stop, "stop_source": which, "stop_dist_pct": stop_dist,
            "qual_huge": qual_huge, "qual_stop": qual_stop,
            "filterA_ok": filterA_ok, "filterB_ok": filterB_ok,
        }
        # resolve the hypothetical trade (only meaningful if stop is valid)
        if qual_stop:
            R = abs(entry - stop)
            target = entry + target_R * R if direction == "long" else entry - target_R * R
            et, ep, reason, rr, tie = _resolve(direction, entry, stop, target, sess, c_idx, fc_t)
            row.update({"R_unit": R, "target_price": target, "exit_time": et,
                        "exit_price": ep, "exit_reason": reason, "realized_R": rr,
                        "tie_stopfirst": tie})
        else:
            row.update({"R_unit": np.nan, "target_price": np.nan, "exit_time": None,
                        "exit_price": np.nan, "exit_reason": None, "realized_R": np.nan,
                        "tie_stopfirst": False})
        out.append(row)
    return out


def select_trades(enriched: pd.DataFrame, filter_name: str, baskets: dict) -> pd.DataFrame:
    """First fully-qualifying trigger per day for one filter -> the trade log.

    Qualify = not huge AND stop <= 1% AND direction allowed by the filter.
    Ties on the same candle -> higher rank (lower r_rank). One trade/day.
    """
    fcol = "filterA_ok" if filter_name == "A" else "filterB_ok"
    q = enriched[enriched["qual_huge"] & enriched["qual_stop"] & enriched[fcol]].copy()

    trades = []
    for day in baskets:                       # iterate all basket days (incl. no-trade)
        cand = q[q["date"] == day]
        if len(cand) == 0:
            continue
        cand = cand.sort_values(["trigger_time", "r_rank"])
        trades.append(cand.iloc[0])
    if not trades:
        return pd.DataFrame()
    log = pd.DataFrame(trades).reset_index(drop=True)
    log.insert(1, "filter", filter_name)
    return log
