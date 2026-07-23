"""
Sectoral Index Backfill
=======================

Fetches the NSE sectoral indices + Nifty 50 into ``data/raw/indices/`` as
siblings of the existing indices, at the same granularity/date range as the
stock data, using the existing chunked/resumable backfill pipeline.

Nifty 50 / Bank / Fin-Services already exist (1min + daily); this refreshes
them to the latest date and adds the missing 5-min, and fetches every new
sectoral index fresh.

Prereq: a fresh Zerodha token (``python scripts/generate_token.py``).

Usage
-----
    python scripts/backfill_indices.py                    # 5min + daily, all indices
    python scripts/backfill_indices.py --timeframes 5min
    python scripts/backfill_indices.py --only NIFTYIT NIFTYPHARMA
    python scripts/backfill_indices.py --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from infrastructure.rs.index_registry import INDEX_REGISTRY, data_folder, token, kite_symbol

DEFAULT_START = "2020-02-22"
DEFAULT_TIMEFRAMES = ["5min", "daily"]
FETCH_LOG_PATH = Path(_ROOT) / "data" / "rs" / "fetch_log.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill NSE sectoral indices")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES,
                    choices=["1min", "5min", "15min", "60min", "daily"])
    ap.add_argument("--only", nargs="+", default=None,
                    help="canonical keys to fetch (default: all in the registry)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

    from infrastructure.data.ingestion import backfill, register_instrument
    from infrastructure.data.loader import load

    keys = args.only or list(INDEX_REGISTRY.keys())

    print("=" * 64)
    print("  SECTORAL INDEX BACKFILL")
    print(f"  Indices:    {', '.join(keys)}")
    print(f"  Timeframes: {', '.join(args.timeframes)}")
    print(f"  Range:      {args.start} -> {args.end}")
    if args.dry_run:
        print("  Mode:       DRY-RUN")
    print("=" * 64)

    # Skip entries with no Kite instrument (token 0, e.g. Nifty Cement).
    skipped = [k for k in keys if token(k) <= 0]
    keys = [k for k in keys if token(k) > 0]
    if skipped:
        print(f"  [SKIP] no Kite instrument (token 0): {', '.join(skipped)}")

    # Register tokens so backfill() can look them up by folder name.
    for k in keys:
        register_instrument(data_folder(k), token(k))

    fetch_log = {"fetched_at": str(date.today()), "start": args.start, "end": args.end,
                 "results": {}, "unavailable": []}

    for k in keys:
        folder = data_folder(k)
        fetch_log["results"][k] = {"kite_symbol": kite_symbol(k), "folder": folder,
                                   "timeframes": {}}
        for tf in args.timeframes:
            label = f"{k} ({folder}) {tf}"
            print(f"\n[FETCH] {label}  {args.start} -> {args.end}")
            try:
                backfill(symbol=folder, asset_type="index", timeframe=tf,
                         start_date=args.start, end_date=args.end, dry_run=args.dry_run)
                if not args.dry_run:
                    df = load(folder, "index", tf)
                    info = {"rows": len(df),
                            "first": str(df.index.min().date()) if len(df) else None,
                            "last": str(df.index.max().date()) if len(df) else None}
                    fetch_log["results"][k]["timeframes"][tf] = info
                    print(f"   OK: {info['rows']:,} rows [{info['first']} -> {info['last']}]")
                else:
                    fetch_log["results"][k]["timeframes"][tf] = {"dry_run": True}
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                fetch_log["results"][k]["timeframes"][tf] = {"error": msg}
                if "no data" in str(e).lower() or "not found" in str(e).lower():
                    fetch_log["unavailable"].append(f"{k}/{tf}")
                print(f"   FAILED: {msg}")

    if not args.dry_run:
        FETCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        FETCH_LOG_PATH.write_text(json.dumps(fetch_log, indent=2), encoding="utf-8")
        print(f"\n[LOG] {FETCH_LOG_PATH}")

    print("\n" + "=" * 64)
    print("  DONE" if not args.dry_run else "  DRY-RUN COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()
