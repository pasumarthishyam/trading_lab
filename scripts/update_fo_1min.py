"""
F&O Universe Data Update — All Timeframes
==========================================

Updates 1-minute (and optionally daily, 15min, 5min, 3min) data for all
211 F&O stocks from the manifest.  Reads the last available date from
each stock's existing Parquet file and fetches only the missing range.

Usage:
    python scripts/update_fo_1min.py                    # 1min only
    python scripts/update_fo_1min.py --all-timeframes   # all timeframes
    python scripts/update_fo_1min.py --dry-run           # preview only

Prerequisites:
    - ``ZERODHA_ACCESS_TOKEN`` is set  (run ``generate_token.py`` first)
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path(_PROJECT_ROOT) / "data" / "raw" / "stocks" / "_fo_universe_manifest.json"

# Timeframes in order of priority.
ALL_TIMEFRAMES = ["1min", "3min", "5min", "15min", "daily"]


def _latest_date_in_parquet(path: Path) -> date | None:
    """Read the latest date from a Parquet file without loading everything."""
    import pandas as pd
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["date"])
        if df.empty:
            return None
        return pd.Timestamp(df["date"].max()).date()
    except Exception:
        return None


def main() -> None:
    # Fix Windows console encoding — ingestion.py logs contain '→' (U+2192).
    import io, os
    if os.name == "nt":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Update F&O universe stock data (incremental fetch)",
    )
    parser.add_argument(
        "--all-timeframes", action="store_true",
        help="Fetch all timeframes (daily, 15min, 5min, 3min, 1min). Default: 1min only.",
    )
    parser.add_argument(
        "--end", default=str(date.today()),
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview plan without making API calls",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from dotenv import load_dotenv
    load_dotenv()

    from infrastructure.data.ingestion import backfill, lookup_token
    from infrastructure.data.loader import get_data_path

    # Load F&O universe from manifest.
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    symbols = sorted(manifest["symbols"])

    timeframes = ALL_TIMEFRAMES if args.all_timeframes else ["1min"]

    print("=" * 65)
    print("  F&O Universe Data Update")
    print(f"  Symbols:    {len(symbols)}")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"  End date:   {args.end}")
    if args.dry_run:
        print("  Mode:       DRY-RUN")
    print("=" * 65)

    # Resolve tokens first.
    if not args.dry_run:
        print("\n[TOKENS] Resolving instrument tokens...")
        failed_lookups = []
        for i, sym in enumerate(symbols, 1):
            try:
                lookup_token(sym, exchange="NSE")
                if i <= 3 or i % 50 == 0:
                    print(f"  [{i}/{len(symbols)}] {sym}: OK")
            except ValueError as e:
                logger.error("Token lookup failed for %s: %s", sym, e)
                failed_lookups.append(sym)
        if failed_lookups:
            print(f"  [WARN] Failed lookups: {failed_lookups}")
            symbols = [s for s in symbols if s not in failed_lookups]
        print(f"  Token resolution done. Proceeding with {len(symbols)} symbols.")

    # Fetch data.
    total = len(symbols)
    completed = 0
    skipped = 0
    failed = 0
    failed_symbols = []

    for si, symbol in enumerate(symbols, 1):
        symbol_ok = True

        for tf in timeframes:
            parquet_path = get_data_path(symbol, "stock", tf)
            latest = _latest_date_in_parquet(parquet_path)

            end_dt = date.fromisoformat(args.end)
            if latest and latest >= end_dt:
                logger.info("[%d/%d] %s %s: already current (%s)", si, total, symbol, tf, latest)
                skipped += 1
                continue

            # Start from 1 day after the latest existing date, or a reasonable default.
            if latest:
                start_dt = latest + timedelta(days=1)
            else:
                start_dt = date(2026, 6, 23)  # Fallback

            start_str = start_dt.strftime("%Y-%m-%d")

            if args.dry_run:
                print(f"  [{si}/{total}] {symbol} {tf}: would fetch {start_str} → {args.end}")
                continue

            try:
                print(f"  [{si}/{total}] {symbol} {tf}: {start_str} → {args.end} ...", end=" ", flush=True)
                df = backfill(
                    symbol=symbol,
                    asset_type="stock",
                    timeframe=tf,
                    start_date=start_str,
                    end_date=args.end,
                    dry_run=False,
                )
                print(f"{len(df):,} total rows")
            except Exception as exc:
                error_str = str(exc).lower()
                cause_str = str(exc.__cause__).lower() if exc.__cause__ else ""
                is_token_error = any(
                    kw in error_str or kw in cause_str
                    for kw in ["access_token", "api_key", "token"]
                )
                print(f"FAILED — {exc}")
                symbol_ok = False

                if is_token_error:
                    print("\n" + "!" * 65)
                    print("  ACCESS TOKEN EXPIRED — aborting.")
                    print("  Run: python scripts/generate_token.py")
                    print("  Then re-run this script.")
                    print("!" * 65)
                    sys.exit(1)
                break

        if symbol_ok:
            completed += 1
        else:
            failed += 1
            failed_symbols.append(symbol)

        # Inter-symbol delay.
        if si < total and not args.dry_run:
            time.sleep(0.4 + random.uniform(0, 0.1))

    # Summary.
    print("\n" + "=" * 65)
    if args.dry_run:
        print("  [DRY-RUN COMPLETE]")
    else:
        print("  [UPDATE COMPLETE]")
        print(f"  Completed:  {completed}")
        print(f"  Failed:     {failed}")
        if failed_symbols:
            print(f"  Failed:     {failed_symbols}")
    print("=" * 65)


if __name__ == "__main__":
    main()
