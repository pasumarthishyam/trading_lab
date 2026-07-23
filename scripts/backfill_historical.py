"""
Historical Data Backfill Script
================================

Fetches 6 years of historical data for multiple indices from
Zerodha Kite Connect by chunking the date range into API-safe
windows (60 days for 1min, 200 days for 15min, 2000 days for daily).

Supports checkpoint/resume — if the script is interrupted, re-run
it and it will pick up where it left off.

Usage:
    python scripts/backfill_historical.py              # Full backfill
    python scripts/backfill_historical.py --dry-run    # Preview chunks only

Prerequisites:
    - ``ZERODHA_ACCESS_TOKEN`` is set  (run ``generate_token.py`` first)
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# Add project root to sys.path so infrastructure imports work
# regardless of whether pip install -e . has been run.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── Backfill configuration ─────────────────────────────────────────
# Each entry backfills data from a start date up to the given end date.
# Grouped by instrument, daily first (fastest), then 15min, then 1min.

TODAY = str(date.today())

BACKFILL_JOBS: list[dict[str, str]] = [
    # ── SENSEX ──────────────────────────────────────────────────
    {
        "label": "SENSEX daily",
        "symbol": "SENSEX",
        "asset_type": "index",
        "timeframe": "daily",
        "start_date": "2020-02-22",
        "end_date": TODAY,
    },
    {
        "label": "SENSEX 15min",
        "symbol": "SENSEX",
        "asset_type": "index",
        "timeframe": "15min",
        "start_date": "2020-02-22",
        "end_date": TODAY,
    },
    {
        "label": "SENSEX 1min",
        "symbol": "SENSEX",
        "asset_type": "index",
        "timeframe": "1min",
        "start_date": "2020-02-22",
        "end_date": TODAY,
    },
    # ── BANKNIFTY ───────────────────────────────────────────────
    {
        "label": "BANKNIFTY daily",
        "symbol": "BANKNIFTY",
        "asset_type": "index",
        "timeframe": "daily",
        "start_date": "2020-02-22",
        "end_date": TODAY,
    },
    {
        "label": "BANKNIFTY 15min",
        "symbol": "BANKNIFTY",
        "asset_type": "index",
        "timeframe": "15min",
        "start_date": "2020-02-22",
        "end_date": TODAY,
    },
    {
        "label": "BANKNIFTY 1min",
        "symbol": "BANKNIFTY",
        "asset_type": "index",
        "timeframe": "1min",
        "start_date": "2020-02-22",
        "end_date": TODAY,
    },
    # ── FINNIFTY ────────────────────────────────────────────────
    # Launched Jan 2021 — earlier dates will return empty.
    {
        "label": "FINNIFTY daily",
        "symbol": "FINNIFTY",
        "asset_type": "index",
        "timeframe": "daily",
        "start_date": "2021-01-01",
        "end_date": TODAY,
    },
    {
        "label": "FINNIFTY 15min",
        "symbol": "FINNIFTY",
        "asset_type": "index",
        "timeframe": "15min",
        "start_date": "2021-01-01",
        "end_date": TODAY,
    },
    {
        "label": "FINNIFTY 1min",
        "symbol": "FINNIFTY",
        "asset_type": "index",
        "timeframe": "1min",
        "start_date": "2021-01-01",
        "end_date": TODAY,
    },
]


def main() -> None:
    """Execute the backfill pipeline."""
    parser = argparse.ArgumentParser(
        description="Backfill historical data for SENSEX, BANKNIFTY, FINNIFTY",
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

    from infrastructure.data.ingestion import backfill
    from infrastructure.data.loader import load

    print("=" * 60)
    print("  Historical Data Backfill — Multi-Index")
    print(f"  Today: {date.today()}")
    if args.dry_run:
        print("  Mode:  DRY-RUN (no API calls)")
    print("=" * 60)

    total = len(BACKFILL_JOBS)

    for i, job in enumerate(BACKFILL_JOBS, start=1):
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

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)

    if args.dry_run:
        print("  [DRY-RUN COMPLETE]  No data was fetched.")
        print("  Run without --dry-run to perform the actual backfill.")
    else:
        print("  [SUCCESS]  Backfill complete!")
        print()

        # Show final data stats.
        for job in BACKFILL_JOBS:
            try:
                df = load(job["symbol"], job["asset_type"], job["timeframe"])
                print(
                    f"  {job['label']:>18s}: {len(df):>9,} rows  "
                    f"[{df.index.min().date()} -> {df.index.max().date()}]"
                )
            except FileNotFoundError:
                print(f"  {job['label']:>18s}: FILE NOT FOUND")

    print("=" * 60)


if __name__ == "__main__":
    main()
