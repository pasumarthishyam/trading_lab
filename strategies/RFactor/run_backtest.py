"""
R-Factor Ranking -> Move Validation — Driver
============================================

Runs the full point-in-time backtest over the F&O universe and the most
recent ``test_period_days`` trading days, then writes the reusable
substrate and the verdict summary to ``strategies/RFactor/results/``.

Usage
-----
    python strategies/RFactor/run_backtest.py
    python strategies/RFactor/run_backtest.py --test-days 120
    python strategies/RFactor/run_backtest.py --symbols RELIANCE TCS INFY

Outputs
-------
    results/rfactor_picks.parquet          frozen top-N rows (Section 5)
    results/rfactor_universe_daily.parquet per (symbol, day) outcomes + eligibility
    results/rfactor_summary.csv            per-checkpoint verdict table
    results/rfactor_run_meta.json          run configuration + provenance
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import duckdb
import pandas as pd

from strategies.RFactor.config import (
    CONFIG, SUBSTRATE_DIR, PICKS_PATH, UNIVERSE_DAILY_PATH,
    SUMMARY_PATH, RUN_META_PATH,
)
from strategies.RFactor import engine, analytics

logger = logging.getLogger(__name__)

# A test day is degenerate (e.g. a partial download day) when fewer than
# this fraction of the universe is eligible; such days are dropped.
MIN_ELIGIBLE_FRACTION = 0.5


def _time_to_str(v) -> object:
    """datetime.time -> 'HH:MM:SS' (None/NaT preserved) for parquet safety."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return v.strftime("%H:%M:%S")
    except AttributeError:
        return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the R-Factor backtest")
    parser.add_argument("--test-days", type=int, default=CONFIG["test_period_days"],
                        help="Number of most-recent trading days to test")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Restrict universe (default: full F&O manifest)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = dict(CONFIG)
    cfg["test_period_days"] = args.test_days
    lookback = cfg["rvol_lookback"]

    SUBSTRATE_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # ── calendar, test window, scan cutoff ──────────────────────────
    cal = engine.trading_calendar(con)
    universe = sorted(args.symbols) if args.symbols else engine.load_universe()
    n_test = min(args.test_days, len(cal) - lookback - 1)
    test_days = set(pd.Timestamp(d) for d in cal[-n_test:])
    buffer_days = 30
    cutoff = cal[-(n_test + lookback + buffer_days)]

    print("=" * 66)
    print("  R-FACTOR RANKING -> MOVE VALIDATION BACKTEST")
    print(f"  Universe:     {len(universe)} F&O symbols")
    print(f"  Test window:  {min(test_days).date()} -> {max(test_days).date()}"
          f"  ({n_test} trading days)")
    print(f"  Checkpoints:  {', '.join(cfg['checkpoints'])}")
    print(f"  Top-N:        {cfg['top_n']}  (top-{cfg['top_subset']} subset)")
    print(f"  Baseline:     {lookback} prior data-days, point-in-time")
    print(f"  Scan cutoff:  {cutoff}  (incl. baseline room)")
    print("=" * 66)

    # ── heavy DuckDB pass ───────────────────────────────────────────
    logger.info("Aggregating 1-min Parquet via DuckDB ...")
    agg = engine.aggregate_symbol_days(con, cfg, cutoff, universe=universe)

    # ── features, ranking, measurement ──────────────────────────────
    logger.info("Computing baselines, R, ranks, outcomes ...")
    corp = engine.load_corp_action_dates()
    long_df, daily_df = engine.compute_features(agg, cfg, corp, test_days)

    # ── drop degenerate (partial) test days ─────────────────────────
    elig_per_day = daily_df.groupby("date")["eligible"].sum()
    min_elig = MIN_ELIGIBLE_FRACTION * len(universe)
    dropped = sorted(d for d, c in elig_per_day.items() if c < min_elig)
    if dropped:
        logger.warning("Dropping %d degenerate test day(s) (eligible < %.0f): %s",
                       len(dropped), min_elig, [str(pd.Timestamp(d).date()) for d in dropped])
        keep = ~daily_df["date"].isin(dropped)
        daily_df = daily_df[keep].reset_index(drop=True)
        long_df = long_df[~long_df["date"].isin(dropped)].reset_index(drop=True)

    picks = engine.make_picks(long_df, cfg)

    # ── stringify time columns for parquet portability ──────────────
    for frame in (picks, daily_df):
        if "time_first_hit" in frame.columns:
            frame["time_first_hit"] = frame["time_first_hit"].map(_time_to_str)
    for col in ("first_t", "last_t"):
        if col in daily_df.columns:
            daily_df[col] = daily_df[col].map(_time_to_str)

    # ── analytics + summary ─────────────────────────────────────────
    summary = analytics.build_summary(picks, daily_df, cfg["checkpoints"])
    br = analytics.base_rate(daily_df)

    # ── persist ─────────────────────────────────────────────────────
    picks.to_parquet(PICKS_PATH, index=False)
    daily_df.to_parquet(UNIVERSE_DAILY_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    actual_test_days = sorted(daily_df["date"].unique())
    run_meta = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "config": {k: v for k, v in cfg.items()},
        "universe_size": len(universe),
        "n_test_days": len(actual_test_days),
        "test_start": str(pd.Timestamp(actual_test_days[0]).date()),
        "test_end": str(pd.Timestamp(actual_test_days[-1]).date()),
        "dropped_degenerate_days": [str(pd.Timestamp(d).date()) for d in dropped],
        "scan_cutoff": str(cutoff),
        "pooled_base_rate": br,
        "n_picks_rows": len(picks),
        "n_universe_daily_rows": len(daily_df),
        "corp_actions_excluded": {k: [str(x.date()) for x in v] for k, v in corp.items()},
    }
    RUN_META_PATH.write_text(json.dumps(run_meta, indent=2, default=str), encoding="utf-8")

    # ── report ──────────────────────────────────────────────────────
    print("\n  VERDICT -- per-checkpoint top-N vs base rate")
    print("  " + "-" * 88)
    disp = summary.copy()
    for c in ["topn_hit", "top5_hit", "base_rate", "lift_topn", "lift_top5",
              "median_fav_move", "pct_capturable"]:
        disp[c] = (disp[c] * 100).round(1)
    disp["mean_picks_hit"] = disp["mean_picks_hit"].round(1)
    disp["churn"] = (disp["churn"] * 100).round(0)
    print(disp.to_string(index=False))
    print("  " + "-" * 88)
    print(f"  Pooled base rate (eligible universe): {br*100:.1f}%")
    print(f"\n  Saved:")
    print(f"    picks           -> {PICKS_PATH}   ({len(picks):,} rows)")
    print(f"    universe_daily  -> {UNIVERSE_DAILY_PATH}   ({len(daily_df):,} rows)")
    print(f"    summary         -> {SUMMARY_PATH}")
    print(f"    run_meta        -> {RUN_META_PATH}")
    print("=" * 66)


if __name__ == "__main__":
    main()
