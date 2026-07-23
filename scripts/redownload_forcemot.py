"""
FORCEMOT Gap Repair
===================

The FORCEMOT daily series is missing a contiguous block of trading days
(2023-10-26 -> 2024-02-13). This script re-fetches FORCEMOT across the
full history and merges it back; ``backfill`` deduplicates on the
timestamp, so existing good bars are preserved and only the gap is
filled.

Prerequisites
-------------
A *fresh* Zerodha access token (they expire ~6 AM IST daily). Generate
one first:

    python scripts/generate_token.py

Usage
-----
    python scripts/redownload_forcemot.py
    python scripts/redownload_forcemot.py --timeframes daily
    python scripts/redownload_forcemot.py --dry-run

After it finishes, re-run the validation to confirm the gap is closed.
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

SYMBOL = "FORCEMOT"
DEFAULT_START = "2020-02-22"
DEFAULT_TIMEFRAMES = ["daily", "15min", "5min", "3min", "1min"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair the FORCEMOT data gap")
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})")
    parser.add_argument("--end", default=str(date.today()),
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES,
                        choices=["1min", "3min", "5min", "15min", "60min", "daily"],
                        help="Timeframes to refetch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview the chunk plan without API calls")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from infrastructure.data.ingestion import backfill, lookup_token
    from infrastructure.data.loader import load

    print("=" * 60)
    print(f"  FORCEMOT gap repair")
    print(f"  Timeframes: {', '.join(args.timeframes)}")
    print(f"  Range:      {args.start} -> {args.end}")
    if args.dry_run:
        print("  Mode:       DRY-RUN (no API calls)")
    print("=" * 60)

    if not args.dry_run:
        print("\n[TOKEN LOOKUP] Resolving instrument token...")
        try:
            token = lookup_token(SYMBOL)
            print(f"    {SYMBOL}: token={token}")
        except Exception as e:  # noqa: BLE001
            print(f"    {SYMBOL}: FAILED — {e}")
            print("\n[STOP] Could not resolve token. Is the access token fresh?")
            print("       Run: python scripts/generate_token.py")
            sys.exit(1)

    for tf in args.timeframes:
        print(f"\n[REFETCH] {SYMBOL} {tf}  ({args.start} -> {args.end})")
        backfill(
            symbol=SYMBOL,
            asset_type="stock",
            timeframe=tf,
            start_date=args.start,
            end_date=args.end,
            dry_run=args.dry_run,
        )

    print("\n" + "=" * 60)
    if args.dry_run:
        print("  [DRY-RUN COMPLETE]  No data fetched.")
    else:
        print("  [SUCCESS]  FORCEMOT refetched. Resulting coverage:")
        for tf in args.timeframes:
            try:
                df = load(SYMBOL, "stock", tf)
                print(f"    {tf:>6s}: {len(df):>9,} rows  "
                      f"[{df.index.min().date()} -> {df.index.max().date()}]")
            except FileNotFoundError:
                print(f"    {tf:>6s}: FILE NOT FOUND")
        print("\n  Next: re-run your data validation to confirm the gap is closed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
