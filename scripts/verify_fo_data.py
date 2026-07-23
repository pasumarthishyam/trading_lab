"""
F&O Backfill Data Verification
================================

Comprehensive check of all downloaded F&O stock data.
Produces a detailed report covering completeness, schema,
date ranges, row counts, OHLC consistency, and anomalies.

Usage:
    python scripts/verify_fo_data.py
"""

import json
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from infrastructure.data.loader import get_data_path

# ── Configuration ───────────────────────────────────────────────────

STOCKS_DIR = Path(_PROJECT_ROOT) / "data" / "raw" / "stocks"
MANIFEST_PATH = STOCKS_DIR / "_fo_universe_manifest.json"
REPORT_PATH = STOCKS_DIR / "_verification_report.json"

EXPECTED_TIMEFRAMES = ["daily", "15min", "5min", "3min", "1min"]
EXPECTED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
EXPECTED_START = pd.Timestamp("2020-02-22")

# Approximate expected row counts per symbol (lower bounds).
MIN_ROWS = {
    "daily": 1000,    # ~1250 trading days in 5y, allow some buffer
    "15min": 20000,   # ~25 candles/day * 1250 days
    "5min": 60000,    # ~75 candles/day * 1250 days
    "3min": 100000,   # ~125 candles/day * 1250 days
    "1min": 300000,   # ~375 candles/day * 1250 days
}


def main():
    print("=" * 70)
    print("  F&O BACKFILL DATA VERIFICATION")
    print("=" * 70)

    # ── Load manifest ───────────────────────────────────────────────
    if not MANIFEST_PATH.exists():
        print("[ERROR] Manifest not found:", MANIFEST_PATH)
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    fo_symbols = manifest["symbols"]
    print(f"\n  Manifest: {len(fo_symbols)} symbols")
    print(f"  Resolved: {manifest['resolved_at']}")

    # ── Check 1: File completeness ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  CHECK 1: FILE COMPLETENESS")
    print(f"{'=' * 70}")

    complete = []
    missing_files = []
    empty_files = []

    for symbol in fo_symbols:
        sym_missing = []
        sym_empty = []
        for tf in EXPECTED_TIMEFRAMES:
            path = get_data_path(symbol, "stock", tf)
            if not path.exists():
                sym_missing.append(tf)
            elif path.stat().st_size == 0:
                sym_empty.append(tf)

        if sym_missing:
            missing_files.append((symbol, sym_missing))
        elif sym_empty:
            empty_files.append((symbol, sym_empty))
        else:
            complete.append(symbol)

    print(f"\n  Complete (all 5 files):  {len(complete)} / {len(fo_symbols)}")

    if missing_files:
        print(f"\n  MISSING FILES ({len(missing_files)} symbols):")
        for sym, tfs in missing_files:
            print(f"    {sym}: missing {tfs}")
            # Check for leftover staging/checkpoint files.
            sym_dir = STOCKS_DIR / sym
            if sym_dir.exists():
                leftovers = list(sym_dir.glob("*staging*")) + list(sym_dir.glob("*progress*"))
                for f in leftovers:
                    print(f"      (leftover: {f.name})")

    if empty_files:
        print(f"\n  EMPTY FILES ({len(empty_files)} symbols):")
        for sym, tfs in empty_files:
            print(f"    {sym}: empty {tfs}")

    # ── Check 2: Schema validation ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  CHECK 2: SCHEMA VALIDATION")
    print(f"{'=' * 70}")

    schema_issues = []
    for symbol in complete[:5] + complete[-5:]:  # Spot-check 10 symbols.
        for tf in EXPECTED_TIMEFRAMES:
            path = get_data_path(symbol, "stock", tf)
            df = pd.read_parquet(path)
            cols = set(df.columns)
            if cols != EXPECTED_COLUMNS:
                schema_issues.append((symbol, tf, cols))

    if schema_issues:
        print(f"\n  SCHEMA MISMATCHES ({len(schema_issues)}):")
        for sym, tf, cols in schema_issues:
            print(f"    {sym} {tf}: {cols} (expected: {EXPECTED_COLUMNS})")
    else:
        print(f"\n  Schema OK (spot-checked 10 symbols x 5 timeframes)")

    # ── Check 3: Date ranges + row counts ───────────────────────────
    print(f"\n{'=' * 70}")
    print("  CHECK 3: DATE RANGES & ROW COUNTS")
    print(f"{'=' * 70}")

    range_issues = []
    low_row_issues = []
    all_stats = []

    for symbol in complete:
        for tf in EXPECTED_TIMEFRAMES:
            path = get_data_path(symbol, "stock", tf)
            df = pd.read_parquet(path)
            rows = len(df)

            if rows == 0:
                range_issues.append((symbol, tf, "EMPTY FILE"))
                continue

            dt_col = df["date"]
            dt_min = dt_col.min()
            dt_max = dt_col.max()

            all_stats.append({
                "symbol": symbol,
                "timeframe": tf,
                "rows": rows,
                "start": str(dt_min.date()) if hasattr(dt_min, "date") else str(dt_min),
                "end": str(dt_max.date()) if hasattr(dt_max, "date") else str(dt_max),
            })

            # Check if start date is reasonably close to expected.
            if hasattr(dt_min, "date"):
                start_date = dt_min.date()
            else:
                start_date = pd.Timestamp(dt_min).date()

            # Some stocks may not have existed in 2020 — flag if start > 2022.
            if start_date > date(2022, 1, 1) and tf == "daily":
                range_issues.append((
                    symbol, tf,
                    f"Late start: {start_date} (expected near 2020-02-24)"
                ))

            # Row count check.
            if rows < MIN_ROWS.get(tf, 0):
                low_row_issues.append((symbol, tf, rows, MIN_ROWS[tf]))

    # Print summary stats for daily.
    daily_stats = [s for s in all_stats if s["timeframe"] == "daily"]
    if daily_stats:
        rows_list = [s["rows"] for s in daily_stats]
        print(f"\n  Daily data across {len(daily_stats)} symbols:")
        print(f"    Rows:  min={min(rows_list):,}  max={max(rows_list):,}  avg={sum(rows_list)//len(rows_list):,}")
        starts = sorted(set(s["start"] for s in daily_stats))
        ends = sorted(set(s["end"] for s in daily_stats))
        print(f"    Start dates: {starts[0]} to {starts[-1]}")
        print(f"    End dates:   {ends[0]} to {ends[-1]}")

    for tf in ["15min", "5min", "3min", "1min"]:
        tf_stats = [s for s in all_stats if s["timeframe"] == tf]
        if tf_stats:
            rows_list = [s["rows"] for s in tf_stats]
            print(f"\n  {tf} data across {len(tf_stats)} symbols:")
            print(f"    Rows:  min={min(rows_list):,}  max={max(rows_list):,}  avg={sum(rows_list)//len(rows_list):,}")

    if range_issues:
        print(f"\n  DATE RANGE ISSUES ({len(range_issues)}):")
        for sym, tf, issue in range_issues:
            print(f"    {sym} {tf}: {issue}")

    if low_row_issues:
        print(f"\n  LOW ROW COUNTS ({len(low_row_issues)} — may be newly listed stocks):")
        for sym, tf, actual, expected in low_row_issues[:20]:
            print(f"    {sym} {tf}: {actual:,} rows (expected >= {expected:,})")
        if len(low_row_issues) > 20:
            print(f"    ... and {len(low_row_issues) - 20} more")

    # ── Check 4: OHLC consistency ───────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  CHECK 4: OHLC CONSISTENCY (daily only)")
    print(f"{'=' * 70}")

    ohlc_issues = []
    for symbol in complete:
        path = get_data_path(symbol, "stock", "daily")
        df = pd.read_parquet(path)

        if len(df) == 0:
            continue

        # Check: high >= low (always).
        bad_hl = (df["high"] < df["low"]).sum()

        # Check: high >= open and high >= close.
        bad_ho = (df["high"] < df["open"]).sum()
        bad_hc = (df["high"] < df["close"]).sum()

        # Check: low <= open and low <= close.
        bad_lo = (df["low"] > df["open"]).sum()
        bad_lc = (df["low"] > df["close"]).sum()

        # Check: zero volume days.
        zero_vol = (df["volume"] == 0).sum()

        # Check: negative/zero prices.
        bad_prices = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()

        total_issues = bad_hl + bad_ho + bad_hc + bad_lo + bad_lc + bad_prices
        if total_issues > 0 or zero_vol > 5:
            ohlc_issues.append({
                "symbol": symbol,
                "rows": len(df),
                "high<low": bad_hl,
                "high<open": bad_ho,
                "high<close": bad_hc,
                "low>open": bad_lo,
                "low>close": bad_lc,
                "price<=0": bad_prices,
                "zero_vol": zero_vol,
            })

    if ohlc_issues:
        print(f"\n  OHLC ISSUES ({len(ohlc_issues)} symbols):")
        for issue in ohlc_issues:
            parts = []
            for k in ["high<low", "high<open", "high<close", "low>open", "low>close", "price<=0"]:
                if issue[k] > 0:
                    parts.append(f"{k}={issue[k]}")
            if issue["zero_vol"] > 5:
                parts.append(f"zero_vol={issue['zero_vol']}")
            print(f"    {issue['symbol']}: {', '.join(parts)}")
    else:
        print(f"\n  OHLC consistency OK across all {len(complete)} symbols")

    # ── Check 5: Duplicate timestamps ───────────────────────────────
    print(f"\n{'=' * 70}")
    print("  CHECK 5: DUPLICATE TIMESTAMPS (daily only)")
    print(f"{'=' * 70}")

    dup_issues = []
    for symbol in complete:
        path = get_data_path(symbol, "stock", "daily")
        df = pd.read_parquet(path)
        dups = df["date"].duplicated().sum()
        if dups > 0:
            dup_issues.append((symbol, dups))

    if dup_issues:
        print(f"\n  DUPLICATES FOUND ({len(dup_issues)} symbols):")
        for sym, count in dup_issues:
            print(f"    {sym}: {count} duplicate timestamp(s)")
    else:
        print(f"\n  No duplicate timestamps found across {len(complete)} symbols")

    # ── Check 6: Cross-timeframe consistency ────────────────────────
    print(f"\n{'=' * 70}")
    print("  CHECK 6: CROSS-TIMEFRAME CONSISTENCY (spot-check 10 symbols)")
    print(f"{'=' * 70}")

    xcheck_issues = []
    check_symbols = complete[:5] + complete[-5:]  # First 5 + last 5.
    for symbol in check_symbols:
        daily_path = get_data_path(symbol, "stock", "daily")
        min1_path = get_data_path(symbol, "stock", "1min")

        df_d = pd.read_parquet(daily_path)
        df_1 = pd.read_parquet(min1_path)

        if len(df_d) == 0 or len(df_1) == 0:
            continue

        # Compare trading day counts.
        daily_days = df_d["date"].dt.date.nunique()
        min1_days = df_1["date"].dt.date.nunique()

        # They should be very close (1min might have slightly fewer).
        day_diff = abs(daily_days - min1_days)
        if day_diff > 20:
            xcheck_issues.append(
                f"{symbol}: daily has {daily_days} trading days, 1min has {min1_days} ({day_diff} diff)"
            )

        # Compare last close: daily close vs last 1min close on same day.
        last_daily_date = df_d["date"].max().date()
        last_daily_close = df_d[df_d["date"] == df_d["date"].max()]["close"].iloc[0]

        day_1min = df_1[df_1["date"].dt.date == last_daily_date]
        if len(day_1min) > 0:
            last_1min_close = day_1min.iloc[-1]["close"]
            pct_diff = abs(last_daily_close - last_1min_close) / last_daily_close * 100
            if pct_diff > 1.0:
                xcheck_issues.append(
                    f"{symbol} {last_daily_date}: daily close={last_daily_close:.2f} vs 1min last={last_1min_close:.2f} ({pct_diff:.2f}% diff)"
                )

    if xcheck_issues:
        print(f"\n  CROSS-TIMEFRAME ISSUES:")
        for issue in xcheck_issues:
            print(f"    {issue}")
    else:
        print(f"\n  Cross-timeframe consistency OK (10 symbols checked)")

    # ── Check 7: Corporate actions review ───────────────────────────
    print(f"\n{'=' * 70}")
    print("  CHECK 7: CORPORATE ACTIONS")
    print(f"{'=' * 70}")

    corp_path = STOCKS_DIR / "_corporate_actions.json"
    if corp_path.exists():
        corp_actions = json.loads(corp_path.read_text(encoding="utf-8"))
        print(f"\n  {len(corp_actions)} candidate event(s) detected:")
        for ca in corp_actions:
            print(
                f"    {ca['symbol']:>15s}  {ca['date']}  "
                f"ratio={ca['ratio']:.4f}  "
                f"prev_close={ca['prev_close']:>10.2f}  open={ca['open']:>10.2f}"
            )
    else:
        print("\n  Corporate actions file not found.")

    # ── Save verification report ────────────────────────────────────
    report = {
        "verified_at": str(pd.Timestamp.now()),
        "universe_count": len(fo_symbols),
        "complete": len(complete),
        "missing_files": [(s, t) for s, t in missing_files],
        "empty_files": [(s, t) for s, t in empty_files],
        "range_issues": [(s, t, i) for s, t, i in range_issues],
        "low_row_count": len(low_row_issues),
        "ohlc_issues": len(ohlc_issues),
        "duplicate_issues": len(dup_issues),
        "cross_tf_issues": len(xcheck_issues),
        "corporate_actions": len(json.loads(corp_path.read_text(encoding="utf-8"))) if corp_path.exists() else 0,
        "all_stats": all_stats,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Final verdict ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    total_issues = len(missing_files) + len(empty_files) + len(ohlc_issues) + len(dup_issues)
    if total_issues == 0 and len(complete) == len(fo_symbols):
        print("  VERDICT: ALL CLEAR")
        print(f"  {len(complete)} symbols x 5 timeframes = {len(complete) * 5} files verified")
    else:
        print("  VERDICT: ISSUES FOUND")
        print(f"    Complete:     {len(complete)} / {len(fo_symbols)}")
        print(f"    Missing:      {len(missing_files)}")
        print(f"    Empty:        {len(empty_files)}")
        print(f"    OHLC issues:  {len(ohlc_issues)}")
        print(f"    Duplicates:   {len(dup_issues)}")

    print(f"\n  Full report saved: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
