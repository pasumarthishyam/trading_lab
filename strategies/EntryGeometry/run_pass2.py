"""
Pass 2 — Full Trade Backtest (spot, gross) — Driver
===================================================

Runs the entire backtest twice — once under each momentum filter (A, B) —
and reports both cleanly side by side.  No winner is chosen.

Outputs (organised per test)
----------------------------
    results/pass2_trade_backtest/
      run_meta.json
      filter_A/{metrics.csv, trades.parquet, trades.csv}
      filter_B/{metrics.csv, trades.parquet, trades.csv}
      report.html   (built by pass2_report.py)

Usage
-----
    python strategies/EntryGeometry/run_pass2.py
    python strategies/EntryGeometry/run_pass2.py --test-days 120
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

from strategies.EntryGeometry.config import CONFIG_P2, p2_paths
from strategies.EntryGeometry import geometry, pass2_engine, pass2_metrics
from strategies.EntryGeometry.run_pass1 import build_baskets, _load_symbol_5min

logger = logging.getLogger(__name__)


def run(cfg: dict, test_days_n: int, test_name: str = "pass2_trade_backtest"):
    paths = p2_paths(test_name)
    con = duckdb.connect()
    baskets, meta = build_baskets(con, cfg, test_days_n)

    by_symbol: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for d, members in baskets.items():
        for sym, rank in members:
            by_symbol.setdefault(sym, []).append((d, int(rank)))

    logger.info("Replaying + enriching: %d symbols across %d basket days",
                len(by_symbol), len(baskets))

    enriched_rows = []
    for i, (sym, day_ranks) in enumerate(sorted(by_symbol.items()), 1):
        full = _load_symbol_5min(sym, cfg)
        if full is None:
            continue
        full = full.reset_index(drop=True)
        for day, rank in day_ranks:
            gpos = np.flatnonzero((full["d"] == day).to_numpy())
            if not len(gpos):
                continue
            sess = full.iloc[gpos].reset_index(drop=True)
            base = geometry.replay_session(sess, full, gpos, cfg,
                                           symbol=sym, date=day, r_rank=rank)
            enriched_rows.extend(pass2_engine.enrich_session_triggers(base, sess, cfg))
        if i % 25 == 0:
            logger.info("  ... %d/%d symbols", i, len(by_symbol))

    enriched = pd.DataFrame(enriched_rows)
    n_basket_days = len(baskets)

    # ── run each filter separately ──────────────────────────────────
    results = {}
    paths.dir.mkdir(parents=True, exist_ok=True)
    for f in cfg["filters"]:
        log = pass2_engine.select_trades(enriched, f, baskets)
        m = pass2_metrics.compute(log, n_basket_days)
        results[f] = {"log": log, "metrics": m}

        fdir = paths.filter_dir(f)
        fdir.mkdir(parents=True, exist_ok=True)
        if len(log):
            log.to_parquet(fdir / "trades.parquet", index=False)
            log.to_csv(fdir / "trades.csv", index=False)
        pd.Series(m).to_csv(fdir / "metrics.csv", header=["value"])

    # ── run meta ────────────────────────────────────────────────────
    meta.update({
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "pass": 2,
        "test_name": test_name,
        "config": cfg,
        "n_basket_days": n_basket_days,
        "n_enriched_triggers": len(enriched),
        "metrics": {f: results[f]["metrics"] for f in cfg["filters"]},
    })
    paths.run_meta.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    _print_comparison(results, meta, cfg, paths)
    return results, meta


def _fmt(m, k, pct=False, r=False):
    v = m.get(k)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if pct:
        return f"{v*100:.1f}%"
    if r:
        return f"{v:+.3f}R" if k in ("expectancy_R",) else f"{v:.2f}"
    return f"{v}"


def _print_comparison(results, meta, cfg, paths):
    fs = cfg["filters"]
    print("=" * 74)
    print(f"  PASS 2 [{meta['test_name']}] — FULL TRADE BACKTEST (spot, gross)  |  3R / 1R")
    print(f"  Window {meta['test_start']} -> {meta['test_end']}  "
          f"({meta['n_basket_days']} basket days)")
    print(f"  Basket lock {cfg['basket_lock_time']} | entry {cfg['entry_window_start']}"
          f"-{cfg['entry_window_end']} | filters {','.join(fs)} | 2x huge, 1% stop cap")
    print("=" * 74)
    rows = [
        ("Trades", "n_trades", ""), ("No-trade days", "no_trade_days", ""),
        ("Expectancy (R/trade)", "expectancy_R", "expR"), ("Total R", "total_R", "num"),
        ("Win rate", "win_rate", "pct"),
        ("Avg winner (R)", "avg_win_R", "num"), ("Avg loser (R)", "avg_loss_R", "num"),
        ("% target", "pct_target", "pct"), ("% stop", "pct_stop", "pct"),
        ("% time-closed", "pct_time", "pct"), ("Time-close mean R", "time_close_mean_R", "num"),
        ("Max drawdown (R)", "max_drawdown_R", "num"),
        ("Longest losing streak", "longest_losing_streak", ""),
        ("Avg stop dist", "avg_stop_dist_pct", "pct"),
        ("Long / Short", None, "ls"), ("Stop-first ties", "n_tie_stopfirst", ""),
    ]
    hdr = f"  {'Metric':<24}" + "".join(f"{'Filter '+f:>14}" for f in fs)
    print(hdr); print("  " + "-" * (24 + 14 * len(fs)))
    for label, key, kind in rows:
        cells = []
        for f in fs:
            m = results[f]["metrics"]
            if kind == "ls":
                cells.append(f"{m.get('n_long',0)}/{m.get('n_short',0)}")
            elif kind == "pct":
                cells.append(_fmt(m, key, pct=True))
            elif kind == "expR":
                v = m.get(key); cells.append("—" if v is None or np.isnan(v) else f"{v:+.3f}R")
            elif kind == "num":
                v = m.get(key); cells.append("—" if v is None or (isinstance(v,float) and np.isnan(v)) else f"{v:.2f}")
            else:
                cells.append(_fmt(m, key))
        print(f"  {label:<24}" + "".join(f"{c:>14}" for c in cells))
    print("  " + "-" * (24 + 14 * len(fs)))
    print(f"  Saved per-filter logs + metrics under {paths.dir}")
    print("  NOTE: spot & gross, single (rising) regime, entry at candle close "
          "(live slightly worse).")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description="Pass 2 full trade backtest")
    ap.add_argument("--test-days", type=int, default=CONFIG_P2["test_period_days"])
    ap.add_argument("--test-name", default="pass2_trade_backtest",
                    help="output folder name under RFactor/results/")
    ap.add_argument("--basket-lock", default=None, help="override basket_lock_time (HH:MM)")
    ap.add_argument("--entry-start", default=None, help="override entry_window_start (HH:MM)")
    ap.add_argument("--entry-end", default=None, help="override entry_window_end (HH:MM)")
    ap.add_argument("--filters", nargs="+", default=None, help="subset of filters, e.g. A")
    ap.add_argument("--no-report", action="store_true", help="skip building the HTML files")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

    cfg = dict(CONFIG_P2)
    if args.basket_lock:
        cfg["basket_lock_time"] = args.basket_lock
    if args.entry_start:
        cfg["entry_window_start"] = args.entry_start
    if args.entry_end:
        cfg["entry_window_end"] = args.entry_end
    if args.filters:
        cfg["filters"] = args.filters

    run(cfg, args.test_days, test_name=args.test_name)

    if not args.no_report:
        from strategies.EntryGeometry import pass2_report, pass2_tradelog
        rp = pass2_report.build_report(args.test_name)
        tl = pass2_tradelog.build_tradelog(args.test_name)
        print(f"  Report:    {rp}")
        print(f"  Trade log: {tl}")


if __name__ == "__main__":
    main()
