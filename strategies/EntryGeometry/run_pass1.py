"""
Pass 1 — Entry Geometry Measurement & Audit — Driver
====================================================

Per test day: freeze the R-factor basket at 10:00 (top-N), replay the entry
geometry on each basket symbol's 5-min series, and log one row per trigger.
Measure-and-audit only — no thresholds, no stop choice, no exits, no P&L.

Outputs
-------
    results/pass1_triggers.parquet   one row per trigger (Section 7)
    results/pass1_run_meta.json      run provenance
    (distributions + audit sample + HTML come from report.py)

Usage
-----
    python strategies/EntryGeometry/run_pass1.py
    python strategies/EntryGeometry/run_pass1.py --test-days 120
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import duckdb
import numpy as np
import pandas as pd

from strategies.EntryGeometry.config import (
    CONFIG, P1_SUBSTRATE, TRIGGERS_PATH, RUN_META_PATH, five_min_path,
)
from strategies.EntryGeometry import geometry
from strategies.RFactor import engine as rf_engine
from strategies.RFactor.config import CONFIG as RF_CONFIG

logger = logging.getLogger(__name__)

MIN_ELIGIBLE_FRACTION = 0.5
_SESS_OPEN = pd.Timestamp(CONFIG["session_open"]).time()
_SESS_END = pd.Timestamp(CONFIG["session_end"]).time()


def build_baskets(con, cfg, test_days_n: int):
    """Return (baskets, meta) where baskets maps date -> [(symbol, r_rank)]."""
    rf = dict(RF_CONFIG)
    rf["rvol_lookback"] = cfg["rvol_lookback"]
    # Ensure the basket-lock time is an R checkpoint (e.g. 09:30 for a variant
    # that locks earlier than the RFactor defaults).
    rf["checkpoints"] = sorted(set(rf["checkpoints"]) | {cfg["basket_lock_time"]})
    lookback = rf["rvol_lookback"]

    cal = rf_engine.trading_calendar(con)
    universe = rf_engine.load_universe()
    n_test = min(test_days_n, len(cal) - lookback - 1)
    test_days = set(pd.Timestamp(d) for d in cal[-n_test:])
    cutoff = cal[-(n_test + lookback + 30)]

    logger.info("Building baskets: R-rank @ %s, top-%d, %d test days",
                cfg["basket_lock_time"], cfg["basket_size"], n_test)
    agg = rf_engine.aggregate_symbol_days(con, rf, cutoff, universe=universe)
    corp = rf_engine.load_corp_action_dates()
    long_df, daily = rf_engine.compute_features(agg, rf, corp, test_days)

    # drop degenerate days (muhurat / partial), same rule as RFactor
    elig_per_day = daily.groupby("date")["eligible"].sum()
    dropped = sorted(d for d, cnt in elig_per_day.items()
                     if cnt < MIN_ELIGIBLE_FRACTION * len(universe))
    lock = cfg["basket_lock_time"]
    sel = long_df[(long_df["checkpoint"] == lock) & (long_df["eligible"])
                  & (long_df["r_rank"] <= cfg["basket_size"])
                  & (~long_df["date"].isin(dropped))]

    baskets: dict[pd.Timestamp, list[tuple[str, int]]] = {}
    for d, g in sel.groupby("date"):
        baskets[d] = list(g.sort_values("r_rank")[["symbol", "r_rank"]]
                          .itertuples(index=False, name=None))

    meta = {
        "universe_size": len(universe),
        "n_test_days": len(baskets),
        "test_start": str(min(baskets).date()),
        "test_end": str(max(baskets).date()),
        "dropped_degenerate_days": [str(pd.Timestamp(d).date()) for d in dropped],
        "corp_actions_excluded": {k: [str(x.date()) for x in v] for k, v in corp.items()},
    }
    return baskets, meta


def _load_symbol_5min(symbol: str, cfg: dict) -> pd.DataFrame | None:
    """Load a symbol's continuous regular-session 5-min frame with indicators."""
    path = five_min_path(symbol)
    if not path.exists():
        logger.warning("No 5-min file for %s", symbol)
        return None
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["date"])
    t = df["ts"].dt.time
    df = df[(t >= _SESS_OPEN) & (t <= _SESS_END)].copy()   # drop muhurat evening etc.
    df = geometry.compute_indicators(df, cfg["atr_period"])
    # tz-naive midnight to match the RFactor basket dates (which are tz-naive)
    df["d"] = df["ts"].dt.tz_localize(None).dt.normalize()
    return df


def run(cfg: dict, test_days_n: int) -> pd.DataFrame:
    con = duckdb.connect()
    baskets, meta = build_baskets(con, cfg, test_days_n)

    # invert: symbol -> [(date, r_rank)]
    by_symbol: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for d, members in baskets.items():
        for sym, rank in members:
            by_symbol.setdefault(sym, []).append((d, int(rank)))

    logger.info("Replaying geometry: %d unique basket symbols across %d days",
                len(by_symbol), len(baskets))

    all_rows: list[dict] = []
    for i, (sym, day_ranks) in enumerate(sorted(by_symbol.items()), 1):
        full = _load_symbol_5min(sym, cfg)
        if full is None:
            continue
        full = full.reset_index(drop=True)
        for day, rank in day_ranks:
            mask = (full["d"] == day).to_numpy()
            if not mask.any():
                continue
            sess_global_pos = np.flatnonzero(mask)
            session = full.iloc[sess_global_pos].reset_index(drop=True)
            rows = geometry.replay_session(
                session, full, sess_global_pos, cfg,
                symbol=sym, date=day, r_rank=rank,
            )
            all_rows.extend(rows)
        if i % 25 == 0:
            logger.info("  ... %d/%d symbols, %d triggers so far",
                        i, len(by_symbol), len(all_rows))

    triggers = pd.DataFrame(all_rows)
    if len(triggers):
        triggers = triggers.sort_values(["date", "trigger_time", "symbol"]).reset_index(drop=True)

    P1_SUBSTRATE.mkdir(parents=True, exist_ok=True)
    triggers.to_parquet(TRIGGERS_PATH, index=False)

    meta.update({
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg,
        "n_triggers": len(triggers),
        "n_basket_symbol_days": int(sum(len(v) for v in baskets.values())),
    })
    RUN_META_PATH.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    _print_sanity(triggers, baskets, meta)
    return triggers


def _print_sanity(triggers: pd.DataFrame, baskets: dict, meta: dict) -> None:
    print("=" * 66)
    print("  PASS 1 — ENTRY GEOMETRY MEASUREMENT")
    print(f"  Window      : {meta['test_start']} -> {meta['test_end']} "
          f"({meta['n_test_days']} days)")
    print(f"  Basket      : top-{CONFIG['basket_size']} R-rank @ {CONFIG['basket_lock_time']}"
          f"  |  swing_K={CONFIG['swing_K']}  ATR={CONFIG['atr_period']}")
    print(f"  Triggers    : {len(triggers):,}")
    print("=" * 66)
    if not len(triggers):
        print("  No triggers recorded."); return

    n_days = meta["n_test_days"]
    per_day = triggers.groupby("date").size()
    days_with = per_day.index.nunique()
    print("\n  SANITY STATS")
    print(f"    Triggers/day      : mean {per_day.mean():.2f}  median {per_day.median():.0f}  "
          f"max {per_day.max()}")
    print(f"    Days with >=1     : {days_with}/{n_days} ({100*days_with/n_days:.0f}%)  "
          f"| no-trigger days: {n_days - days_with}")
    ls = triggers["direction"].value_counts()
    print(f"    Long / Short      : {ls.get('long',0)} / {ls.get('short',0)}")
    pv = triggers.groupby(["direction", "pos_vs_open"]).size()
    print(f"    pos_vs_open split :")
    for (dirn, pos), cnt in pv.items():
        print(f"        {dirn:5s} {pos:5s}: {cnt}")
    print("\n  HUGE-CANDLE METRIC (atr_ratio = bo_range / ATR) percentiles:")
    q = triggers["atr_ratio"].quantile([.5, .75, .9, .95]).round(2)
    print(f"    p50={q[.5]}  p75={q[.75]}  p90={q[.9]}  p95={q[.95]}")
    print(f"\n  Saved: {TRIGGERS_PATH}  ({len(triggers):,} rows)")
    print("=" * 66)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pass 1 entry-geometry measurement")
    ap.add_argument("--test-days", type=int, default=CONFIG["test_period_days"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    run(dict(CONFIG), args.test_days)


if __name__ == "__main__":
    main()
