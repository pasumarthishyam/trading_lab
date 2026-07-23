"""
Backfill Data Verification Script
===================================
Runs comprehensive integrity checks on the backfilled data.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd


def verify_file(label: str, path: str) -> bool:
    """Run all integrity checks on a single parquet file."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")

    df = pd.read_parquet(path)

    # ── Basic stats ─────────────────────────────────────────────
    dt_col = "datetime" if "datetime" in df.columns else "date"
    print(f"  Total rows:        {len(df):,}")
    print(f"  Columns:           {list(df.columns)}")
    print(f"  Date dtype:        {df[dt_col].dtype}")
    print(f"  First timestamp:   {df[dt_col].iloc[0]}")
    print(f"  Last timestamp:    {df[dt_col].iloc[-1]}")

    # ── Check 1: Sorted ────────────────────────────────────────
    is_sorted = df[dt_col].is_monotonic_increasing
    status = "PASS" if is_sorted else "FAIL"
    print(f"\n  [CHECK 1] Chronologically sorted:  {status}")

    # ── Check 2: No duplicates ─────────────────────────────────
    dupes = df[dt_col].duplicated().sum()
    status = "PASS" if dupes == 0 else f"FAIL ({dupes:,} duplicates)"
    print(f"  [CHECK 2] No duplicate timestamps: {status}")

    # ── Check 3: No nulls in OHLC ─────────────────────────────
    ohlc = ["open", "high", "low", "close"]
    nulls = df[ohlc].isnull().sum().sum()
    status = "PASS" if nulls == 0 else f"FAIL ({nulls} nulls)"
    print(f"  [CHECK 3] No null OHLC values:     {status}")

    # ── Check 4: Prices are positive ───────────────────────────
    min_price = df[ohlc].min().min()
    status = "PASS" if min_price > 0 else f"FAIL (min={min_price})"
    print(f"  [CHECK 4] All prices positive:     {status}")

    # ── Check 5: High >= Low ───────────────────────────────────
    bad_hl = (df["high"] < df["low"]).sum()
    status = "PASS" if bad_hl == 0 else f"FAIL ({bad_hl} rows)"
    print(f"  [CHECK 5] High >= Low:             {status}")

    # ── Check 6: Unique trading days ───────────────────────────
    dates_utc = pd.to_datetime(df[dt_col], utc=True)
    unique_days = dates_utc.dt.date.nunique()
    print(f"\n  Unique trading days: {unique_days}")

    # ── Check 7: Yearly breakdown ──────────────────────────────
    years = dates_utc.dt.year.value_counts().sort_index()
    print(f"\n  Rows per year:")
    for year, count in years.items():
        days_in_year = dates_utc[dates_utc.dt.year == year].dt.date.nunique()
        print(f"    {year}: {count:>8,} rows  ({days_in_year} trading days)")

    # ── Check 8: Timestamp sanity (IST market hours) ───────────
    hours = dates_utc.dt.tz_convert("Asia/Kolkata").dt.hour
    outside_market = ((hours < 9) | (hours >= 16)).sum()
    status = "PASS" if outside_market == 0 else f"WARN ({outside_market:,} rows outside 9-16 IST)"
    print(f"\n  [CHECK 8] Times within market hours: {status}")

    # ── Sample data at key points ──────────────────────────────
    print(f"\n  Sample data:")
    print(f"    First:  {df.iloc[0].to_dict()}")
    mid = len(df) // 2
    print(f"    Middle: {df.iloc[mid].to_dict()}")
    print(f"    Last:   {df.iloc[-1].to_dict()}")

    all_ok = is_sorted and dupes == 0 and nulls == 0 and min_price > 0 and bad_hl == 0
    return all_ok


def main():
    print("=" * 60)
    print("  BACKFILL DATA INTEGRITY VERIFICATION")
    print("=" * 60)

    files = [
        # NIFTY
        ("NIFTY 1min",      "data/raw/indices/NIFTY/1min.parquet"),
        ("NIFTY 15min",     "data/raw/indices/NIFTY/15min.parquet"),
        ("NIFTY daily",     "data/raw/indices/NIFTY/daily.parquet"),
        # SENSEX
        ("SENSEX 1min",     "data/raw/indices/SENSEX/1min.parquet"),
        ("SENSEX 15min",    "data/raw/indices/SENSEX/15min.parquet"),
        ("SENSEX daily",    "data/raw/indices/SENSEX/daily.parquet"),
        # BANKNIFTY
        ("BANKNIFTY 1min",  "data/raw/indices/BANKNIFTY/1min.parquet"),
        ("BANKNIFTY 15min", "data/raw/indices/BANKNIFTY/15min.parquet"),
        ("BANKNIFTY daily", "data/raw/indices/BANKNIFTY/daily.parquet"),
        # FINNIFTY
        ("FINNIFTY 1min",   "data/raw/indices/FINNIFTY/1min.parquet"),
        ("FINNIFTY 15min",  "data/raw/indices/FINNIFTY/15min.parquet"),
        ("FINNIFTY daily",  "data/raw/indices/FINNIFTY/daily.parquet"),
    ]

    results = []
    for label, path in files:
        if Path(path).exists():
            ok = verify_file(label, path)
            results.append((label, ok))
        else:
            print(f"\n  SKIP: {path} not found")
            results.append((label, None))

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    checked = [(l, r) for l, r in results if r is not None]
    skipped = [(l, r) for l, r in results if r is None]
    all_passed = all(r for _, r in checked) if checked else False
    for label, ok in results:
        icon = "PASS" if ok else ("FAIL" if ok is False else "SKIP")
        print(f"  {label:>18s}: {icon}")
    print(f"\n  VERDICT: {'ALL CHECKS PASSED' if all_passed else 'ISSUES FOUND'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
