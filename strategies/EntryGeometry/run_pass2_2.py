"""
Pass 2.2 — Rolling Basket + Breakeven, swept over swing K
=========================================================

Rolling top-4 bucket (re-ranked every 15 min), breakeven-at-2R stop, Filter A,
entry 09:30–12:30 — run for swing K = 2, 3, 4, 5 and compared.

The R-factor bucket is independent of swing K, so the rolling baskets are built
once and each candidate symbol's 5-min series is loaded once; only the swing/
level geometry is recomputed per K.

Outputs
-------
    results/pass2.2_rolling_be/
      run_meta.json
      K2/ {metrics.csv, trades.parquet, trades.csv}   (and K3, K4, K5)
      report.html, trade_log.html   (built by pass2_2_report.py)

Usage
-----
    python strategies/EntryGeometry/run_pass2_2.py
    python strategies/EntryGeometry/run_pass2_2.py --test-days 120 --ks 2 3
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
    CONFIG_P2_2, ROLLING_CHECKPOINTS, p2_paths,
)
from strategies.EntryGeometry import geometry, pass2_rolling, pass2_metrics
from strategies.EntryGeometry.run_pass1 import _load_symbol_5min
from strategies.RFactor import engine as rf_engine
from strategies.RFactor.config import CONFIG as RF_CONFIG

logger = logging.getLogger(__name__)
TEST_NAME = "pass2.2_rolling_be"
BASKET_SIZE = 4


def build_rolling_baskets(con, cfg, test_days_n):
    """day -> {checkpoint -> {symbol: rank}} for the top-4, at every rolling cp."""
    rf = dict(RF_CONFIG)
    rf["rvol_lookback"] = cfg["rvol_lookback"]
    rf["checkpoints"] = sorted(set(rf["checkpoints"]) | set(ROLLING_CHECKPOINTS))
    lookback = rf["rvol_lookback"]

    cal = rf_engine.trading_calendar(con)
    universe = rf_engine.load_universe()
    n_test = min(test_days_n, len(cal) - lookback - 1)
    test_days = set(pd.Timestamp(d) for d in cal[-n_test:])
    cutoff = cal[-(n_test + lookback + 30)]

    logger.info("Building rolling baskets (%d checkpoints) ...", len(ROLLING_CHECKPOINTS))
    agg = rf_engine.aggregate_symbol_days(con, rf, cutoff, universe=universe)
    corp = rf_engine.load_corp_action_dates()
    long_df, daily = rf_engine.compute_features(agg, rf, corp, test_days)

    elig = daily.groupby("date")["eligible"].sum()
    dropped = sorted(d for d, c in elig.items() if c < 0.5 * len(universe))

    lb = long_df[(long_df["eligible"]) & (long_df["r_rank"] <= BASKET_SIZE)
                 & (long_df["checkpoint"].isin(ROLLING_CHECKPOINTS))
                 & (~long_df["date"].isin(dropped))]

    day_basket: dict = {}
    for (day, cp), g in lb.groupby(["date", "checkpoint"]):
        day_basket.setdefault(day, {})[cp] = dict(zip(g["symbol"], g["r_rank"].astype(int)))

    meta = {
        "universe_size": len(universe), "n_test_days": len(day_basket),
        "test_start": str(min(day_basket).date()), "test_end": str(max(day_basket).date()),
        "dropped_degenerate_days": [str(pd.Timestamp(d).date()) for d in dropped],
        "rolling_checkpoints": ROLLING_CHECKPOINTS,
    }
    return day_basket, meta


def run(cfg: dict, test_days_n: int, ks: list[int]):
    con = duckdb.connect()
    day_basket, meta = build_rolling_baskets(con, cfg, test_days_n)
    days = sorted(day_basket.keys())

    # candidate symbols per day = union of the day's rolling top-4; invert.
    sym_days: dict[str, list] = {}
    for day, cps in day_basket.items():
        syms = set().union(*[set(b) for b in cps.values()])
        for s in syms:
            sym_days.setdefault(s, []).append(day)
    logger.info("Rolling candidates: %d unique symbols across %d days", len(sym_days), len(days))

    # load each symbol once; run all K over it (geometry only is K-dependent)
    enriched_by_k = {k: [] for k in ks}
    for i, (sym, sdays) in enumerate(sorted(sym_days.items()), 1):
        full = _load_symbol_5min(sym, cfg)
        if full is None:
            continue
        full = full.reset_index(drop=True)
        day_pos = {d: np.flatnonzero((full["d"] == d).to_numpy()) for d in sdays}
        for k in ks:
            cfg_k = dict(cfg); cfg_k["swing_K"] = k
            for day in sdays:
                gpos = day_pos[day]
                if not len(gpos):
                    continue
                sess = full.iloc[gpos].reset_index(drop=True)
                base = geometry.replay_session(sess, full, gpos, cfg_k,
                                               symbol=sym, date=day, r_rank=0)
                enriched_by_k[k].extend(pass2_rolling.enrich_rolling(
                    base, sess, cfg_k, day_basket[day], ROLLING_CHECKPOINTS))
        if i % 25 == 0:
            logger.info("  ... %d/%d symbols", i, len(sym_days))

    # select + metrics per K
    paths = p2_paths(TEST_NAME)
    paths.dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for k in ks:
        enriched = pd.DataFrame(enriched_by_k[k])
        log = pass2_rolling.select_rolling(enriched, days)
        m = pass2_metrics.compute(log, len(days))
        results[k] = m
        kdir = paths.dir / f"K{k}"
        kdir.mkdir(parents=True, exist_ok=True)
        if len(log):
            log.insert(1, "swing_K", k)
            log.to_parquet(kdir / "trades.parquet", index=False)
            log.to_csv(kdir / "trades.csv", index=False)
        pd.Series(m).to_csv(kdir / "metrics.csv", header=["value"])

    meta.update({
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "test_name": TEST_NAME, "config": cfg, "k_sweep": ks,
        "n_days": len(days), "metrics_by_k": results,
    })
    paths.run_meta.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    _print_sweep(results, meta, ks)
    return results, meta


def _print_sweep(results, meta, ks):
    print("=" * 74)
    print(f"  PASS 2.2 — ROLLING BASKET + BREAKEVEN@2R — swing-K sweep")
    print(f"  {meta['test_start']} -> {meta['test_end']}  ({meta['n_days']} days)  "
          f"Filter A · entry 09:30-12:30 · rolling top-4")
    print("=" * 74)
    rows = [
        ("Trades", "n_trades", "int"), ("No-trade days", "no_trade_days", "int"),
        ("Expectancy (R)", "expectancy_R", "R"), ("Total R", "total_R", "num"),
        ("Win rate", "win_rate", "pct"),
        ("% target", "pct_target", "pct"), ("% breakeven", "pct_breakeven", "pct"),
        ("% stop", "pct_stop", "pct"), ("% time", "pct_time", "pct"),
        ("Max DD (R)", "max_drawdown_R", "num"),
        ("Longest lose streak", "longest_losing_streak", "int"),
        ("Avg stop dist", "avg_stop_dist_pct", "pct"),
    ]
    print(f"  {'Metric':<22}" + "".join(f"{'K='+str(k):>12}" for k in ks))
    print("  " + "-" * (22 + 12 * len(ks)))
    for label, key, kind in rows:
        cells = []
        for k in ks:
            v = results[k].get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("—")
            elif kind == "pct":
                cells.append(f"{v*100:.1f}%")
            elif kind == "R":
                cells.append(f"{v:+.3f}R")
            elif kind == "int":
                cells.append(f"{int(v)}")
            else:
                cells.append(f"{v:.2f}")
        print(f"  {label:<22}" + "".join(f"{c:>12}" for c in cells))
    print("  " + "-" * (22 + 12 * len(ks)))
    best = max(ks, key=lambda k: results[k].get("expectancy_R", -9))
    print(f"  Best expectancy: K={best} ({results[best]['expectancy_R']:+.3f}R)  "
          "— spot & gross, one regime; read expectancy not win rate.")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description="Pass 2.2 rolling + breakeven, K sweep")
    ap.add_argument("--test-days", type=int, default=CONFIG_P2_2["test_period_days"])
    ap.add_argument("--ks", nargs="+", type=int, default=CONFIG_P2_2["k_sweep"])
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

    run(dict(CONFIG_P2_2), args.test_days, args.ks)

    if not args.no_report:
        from strategies.EntryGeometry import pass2_2_report
        rp = pass2_2_report.build_report(TEST_NAME)
        tl = pass2_2_report.build_tradelog(TEST_NAME)
        print(f"  Report:    {rp}")
        print(f"  Trade log: {tl}")


if __name__ == "__main__":
    main()
