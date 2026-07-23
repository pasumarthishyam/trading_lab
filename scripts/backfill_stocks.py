"""
Stock Data Backfill Script
============================

Fetches historical data for any NSE stock by auto-discovering the
instrument token from Kite API.  Supports multiple stocks, custom
date ranges, and specific timeframes.

Usage:
    python scripts/backfill_stocks.py EICHERMOT
    python scripts/backfill_stocks.py EICHERMOT RELIANCE TCS INFY
    python scripts/backfill_stocks.py EICHERMOT --start 2022-01-01
    python scripts/backfill_stocks.py EICHERMOT --timeframes 1min daily
    python scripts/backfill_stocks.py EICHERMOT --dry-run

Prerequisites:
    - ``ZERODHA_ACCESS_TOKEN`` is set  (run ``generate_token.py`` first)
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


DEFAULT_START = "2020-02-22"
DEFAULT_TIMEFRAMES = ["daily", "15min", "1min"]


def main() -> None:
    """Execute the stock backfill pipeline."""
    parser = argparse.ArgumentParser(
        description="Backfill historical data for NSE stocks",
    )
    parser.add_argument(
        "symbols",
        nargs="+",
        help="Stock symbols to backfill (e.g. EICHERMOT RELIANCE TCS)",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        default=str(date.today()),
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=DEFAULT_TIMEFRAMES,
        choices=["1min", "15min", "60min", "daily"],
        help=f"Timeframes to fetch (default: {' '.join(DEFAULT_TIMEFRAMES)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview chunk plan without making API calls",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from infrastructure.data.ingestion import backfill, lookup_token
    from infrastructure.data.loader import load

    # Uppercase all symbols for consistency.
    symbols = [s.upper() for s in args.symbols]

    print("=" * 60)
    print("  Stock Data Backfill")
    print(f"  Stocks:     {', '.join(symbols)}")
    print(f"  Timeframes: {', '.join(args.timeframes)}")
    print(f"  Range:      {args.start} -> {args.end}")
    if args.dry_run:
        print("  Mode:       DRY-RUN (no API calls)")
    print("=" * 60)

    # ── Resolve instrument tokens ───────────────────────────────
    if not args.dry_run:
        print("\n[TOKEN LOOKUP] Resolving instrument tokens...")
        for symbol in symbols:
            try:
                token = lookup_token(symbol)
                print(f"    {symbol}: token={token}")
            except ValueError as e:
                print(f"    {symbol}: FAILED — {e}")
                print(f"\n[STOP] Cannot proceed without token for {symbol}.")
                sys.exit(1)

    # ── Build job list ──────────────────────────────────────────
    jobs = []
    for symbol in symbols:
        for tf in args.timeframes:
            jobs.append({
                "label": f"{symbol} {tf}",
                "symbol": symbol,
                "asset_type": "stock",
                "timeframe": tf,
                "start_date": args.start,
                "end_date": args.end,
            })

    total = len(jobs)

    # ── Execute backfill ────────────────────────────────────────
    for i, job in enumerate(jobs, start=1):
        label = job["label"]
        print(f"\n[{i}/{total}] Backfilling {label}...")
        print(f"         {job['start_date']} -> {job['end_date']}")

        df = backfill(
            symbol=job["symbol"],
            asset_type=job["asset_type"],
            timeframe=job["timeframe"],
            start_date=job["start_date"],
            end_date=job["end_date"],
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            print(f"    [OK]  {label}: {len(df):,} total rows")
        else:
            print(f"    [OK]  Dry-run complete for {label}")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)

    if args.dry_run:
        print("  [DRY-RUN COMPLETE]  No data was fetched.")
        print("  Run without --dry-run to perform the actual backfill.")
    else:
        print("  [SUCCESS]  Stock backfill complete!")
        print()

        for job in jobs:
            try:
                df = load(job["symbol"], job["asset_type"], job["timeframe"])
                print(
                    f"  {job['label']:>20s}: {len(df):>9,} rows  "
                    f"[{df.index.min().date()} -> {df.index.max().date()}]"
                )
            except FileNotFoundError:
                print(f"  {job['label']:>20s}: FILE NOT FOUND")

        print(f"\n  Files saved under: data/raw/stocks/")
        for symbol in symbols:
            print(f"    - stocks/{symbol}/")

    print("=" * 60)


if __name__ == "__main__":
    main()
