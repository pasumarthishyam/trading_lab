"""
Pass 2.2 — Rolling Basket + Breakeven — Engine
==============================================

Two rule changes on top of Pass 2 / 2.1:

1. **Rolling basket.** The top-4 R-rank bucket is re-frozen every 15 min across
   the entry window (09:30 … 12:30).  A breakout qualifies only if its stock is
   in the top-4 as of the **most recent checkpoint <= the trigger time** — so a
   stock that climbs into the bucket at 10:30 becomes tradeable then, and one
   that drops out stops being tradeable.  (Contrast: 2.1 locks the 09:30 bucket
   for the whole day.)

2. **Breakeven at 1:2.** When price first touches **+2R** in favour, the stop
   moves to **entry (0R)** from the next candle onward.  Outcomes: +3R (target),
   0R (breakeven), −1R (stop), or a 3:00 time-close partial.

Same-candle target-and-stop stays **stop-first** (conservative).  Everything
else is inherited from Pass 2 (Filter A, 2× huge, stop = farther of
{swing-body, 3rd-back}, spot, gross).  Swing K is swept by the driver.
"""

from __future__ import annotations

import bisect
from datetime import time as dtime

import numpy as np
import pandas as pd

from strategies.EntryGeometry.pass2_engine import _choose_stop, _t


def _governing_cp(trigger_time: str, rolling_cps: list[str]) -> str | None:
    """Most recent rolling checkpoint at or before the trigger time."""
    i = bisect.bisect_right(rolling_cps, trigger_time)
    return rolling_cps[i - 1] if i > 0 else None


def _resolve_breakeven(direction, entry, stop, target, be_price, sess, c_idx, fc_t):
    """Forward resolution with a breakeven stop armed at +2R.

    Returns (exit_time, exit_price, exit_reason, realized_R, tie).
    exit_reason in {target, breakeven, stop, time}.
    """
    h = sess["high"].to_numpy(); l = sess["low"].to_numpy()
    o = sess["open"].to_numpy(); c = sess["close"].to_numpy()
    ts = sess["ts"].to_numpy()
    n = len(sess)
    R = abs(entry - stop)
    sign = 1.0 if direction == "long" else -1.0
    be_armed = False

    for k in range(c_idx + 1, n):
        if pd.Timestamp(ts[k]).time() >= fc_t:
            exitp = o[k]
            return (pd.Timestamp(ts[k]).strftime("%H:%M"), exitp, "time",
                    sign * (exitp - entry) / R, False)

        stop_level = entry if be_armed else stop
        if direction == "long":
            hit_stop = l[k] <= stop_level
            hit_tgt = h[k] >= target
        else:
            hit_stop = h[k] >= stop_level
            hit_tgt = l[k] <= target

        reason_stop = "breakeven" if be_armed else "stop"
        if hit_stop and hit_tgt:                                  # ambiguous -> stop-first
            return (pd.Timestamp(ts[k]).strftime("%H:%M"), stop_level, reason_stop,
                    sign * (stop_level - entry) / R, True)
        if hit_tgt:
            return (pd.Timestamp(ts[k]).strftime("%H:%M"), target, "target", 3.0, False)
        if hit_stop:
            return (pd.Timestamp(ts[k]).strftime("%H:%M"), stop_level, reason_stop,
                    sign * (stop_level - entry) / R, False)

        # not exited this candle — arm breakeven once +2R is touched in favour
        if not be_armed:
            if direction == "long" and h[k] >= be_price:
                be_armed = True
            elif direction == "short" and l[k] <= be_price:
                be_armed = True

    exitp = c[-1]
    return (pd.Timestamp(ts[-1]).strftime("%H:%M"), exitp, "time",
            sign * (exitp - entry) / R, False)


def enrich_rolling(base_rows, sess, cfg, day_basket, rolling_cps):
    """Enrich Pass 1 triggers with the rolling eligibility + breakeven outcome.

    Parameters
    ----------
    day_basket : dict[str, dict[str, int]]
        For this day: checkpoint -> {symbol: rank} for the top-4 bucket.
    rolling_cps : list[str]
        Sorted rolling checkpoint times covering the entry window.
    """
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
        sym = r["symbol"]

        # rolling eligibility: is this stock in the top-4 at the governing cp?
        gov_cp = _governing_cp(r["trigger_time"], rolling_cps)
        cp_bucket = day_basket.get(gov_cp, {}) if gov_cp else {}
        rank_at_cp = cp_bucket.get(sym)
        rolling_ok = rank_at_cp is not None

        # Filter A (9:15 extreme held, point-in-time to entry candle)
        if direction == "long":
            filterA_ok = bool(cummin_low[c_idx] >= low0 - 1e-9)
        else:
            filterA_ok = bool(cummax_high[c_idx] <= high0 + 1e-9)

        stop, stop_dist, which = _choose_stop(
            direction, entry, r["swing_stop_price_body"], r["third_back_stop_price"])
        qual_huge = bool(r["atr_ratio"] <= huge_mult) if pd.notna(r["atr_ratio"]) else False
        qual_stop = bool(pd.notna(stop_dist) and stop_dist <= stop_max)

        row = {
            "date": r["date"], "symbol": sym, "direction": direction,
            "gov_checkpoint": gov_cp, "rank_at_cp": rank_at_cp if rank_at_cp else 99,
            "trigger_time": r["trigger_time"], "entry_price": entry,
            "level_price": r["level_price"], "atr_ratio": r["atr_ratio"],
            "pos_vs_open": r["pos_vs_open"], "stop_price": stop,
            "stop_source": which, "stop_dist_pct": stop_dist,
            "qual_huge": qual_huge, "qual_stop": qual_stop,
            "filterA_ok": filterA_ok, "rolling_ok": rolling_ok,
        }
        if qual_stop:
            R = abs(entry - stop)
            target = entry + target_R * R if direction == "long" else entry - target_R * R
            be_price = entry + 2.0 * R if direction == "long" else entry - 2.0 * R
            et, ep, reason, rr, tie = _resolve_breakeven(
                direction, entry, stop, target, be_price, sess, c_idx, fc_t)
            row.update({"R_unit": R, "target_price": target, "be_price": be_price,
                        "exit_time": et, "exit_price": ep, "exit_reason": reason,
                        "realized_R": rr, "tie_stopfirst": tie})
        else:
            row.update({"R_unit": np.nan, "target_price": np.nan, "be_price": np.nan,
                        "exit_time": None, "exit_price": np.nan, "exit_reason": None,
                        "realized_R": np.nan, "tie_stopfirst": False})
        out.append(row)
    return out


def select_rolling(enriched: pd.DataFrame, days) -> pd.DataFrame:
    """First fully-qualifying trigger per day (Filter A + huge + stop + rolling).

    Ties on the same candle -> higher rank at the governing checkpoint.
    """
    if enriched.empty:
        return pd.DataFrame()
    q = enriched[enriched["qual_huge"] & enriched["qual_stop"]
                 & enriched["filterA_ok"] & enriched["rolling_ok"]].copy()
    trades = []
    for day in days:
        cand = q[q["date"] == day]
        if len(cand) == 0:
            continue
        cand = cand.sort_values(["trigger_time", "rank_at_cp"])
        trades.append(cand.iloc[0])
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades).reset_index(drop=True)
