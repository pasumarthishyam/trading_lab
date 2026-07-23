"""
F&O Universe Historical Data Backfill
=======================================

Resolves the current NSE F&O equity universe from the Kite instruments
dump, then fetches ~6 years of historical OHLCV data (daily, 15min,
5min, 3min, 1min) for every underlying equity — reusing the existing
``infrastructure.data.ingestion.backfill`` pipeline.

Features:
- Programmatic F&O universe resolution (NFO-FUT ∩ NSE EQ)
- Index exclusion (NIFTY, BANKNIFTY, etc. filtered automatically)
- Symbol-by-symbol fetch with idempotent resume
- Corporate actions detection from daily data (split/bonus flagging)
- SQLite instrument registration
- Batch validation via ``infrastructure.data.validation``
- Manifest, run log, and corporate-actions JSON output

Usage:
    python scripts/backfill_fo_universe.py              # Full backfill
    python scripts/backfill_fo_universe.py --dry-run    # Preview only
    python scripts/backfill_fo_universe.py --skip-delete # Resume without deleting existing

Prerequisites:
    - ``ZERODHA_ACCESS_TOKEN`` is set  (run ``generate_token.py`` first)
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from datetime import date, datetime
from pathlib import Path

# ── Project root setup ──────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────

DEFAULT_START_DATE = "2020-02-22"

# Timeframes to fetch, in order (fastest first for early sanity check).
FETCH_TIMEFRAMES = ["daily", "15min", "5min", "3min", "1min"]

# Existing stock symbols whose data will be deleted and re-downloaded.
EXISTING_STOCK_SYMBOLS = ["RELIANCE", "INFY", "EICHERMOT", "TATAELXSI", "MANAPPURAM"]

# Inter-symbol delay: 0.4s base + up to 0.1s jitter.
_INTER_SYMBOL_DELAY_BASE = 0.4
_INTER_SYMBOL_JITTER_MAX = 0.1

# Corporate actions detection threshold.
# Flag when open/prev_close ratio deviates more than this from 1.0.
# 0.25 → flags ratios outside [0.75, 1.25].
_CORP_ACTION_THRESHOLD = 0.25

# ── Path helpers ────────────────────────────────────────────────────

DATA_ROOT = Path(_PROJECT_ROOT)
STOCKS_DIR = DATA_ROOT / "data" / "raw" / "stocks"
MANIFEST_PATH = STOCKS_DIR / "_fo_universe_manifest.json"
BACKFILL_LOG_PATH = STOCKS_DIR / "_fo_backfill_log.json"
CORP_ACTIONS_PATH = STOCKS_DIR / "_corporate_actions.json"


# ── Universe resolution ────────────────────────────────────────────


def resolve_fo_equity_universe(kite) -> tuple[list[str], list[str]]:
    """Resolve F&O equity universe from Kite instruments dump.

    Returns
    -------
    tuple[list[str], list[str]]
        (fo_stock_symbols, excluded_names) — sorted lists.
        ``fo_stock_symbols`` are names present in both NFO-FUT and
        NSE with ``instrument_type == "EQ"``.
        ``excluded_names`` are NFO-FUT names that are NOT NSE equities
        (i.e. indices).
    """
    logger.info("Fetching NSE instrument list...")
    nse_instruments = kite.instruments("NSE")
    nse_equity_symbols = {
        inst["tradingsymbol"]
        for inst in nse_instruments
        if inst["instrument_type"] == "EQ"
    }
    logger.info(
        "Found %d NSE equity symbols.", len(nse_equity_symbols),
    )

    logger.info("Fetching NFO instrument list...")
    nfo_instruments = kite.instruments("NFO")
    fo_all_names = {
        inst["name"]
        for inst in nfo_instruments
        if inst["segment"] == "NFO-FUT"
    }
    logger.info(
        "Found %d unique NFO-FUT underlying names.", len(fo_all_names),
    )

    # Intersect: only equities.
    fo_stock_symbols = sorted(fo_all_names & nse_equity_symbols)
    excluded_names = sorted(fo_all_names - nse_equity_symbols)

    logger.info(
        "F&O equity universe: %d stocks. Excluded (indices): %d → %s",
        len(fo_stock_symbols), len(excluded_names), excluded_names,
    )
    return fo_stock_symbols, excluded_names


def save_manifest(
    symbols: list[str],
    excluded: list[str],
) -> None:
    """Persist the resolved F&O universe as a JSON manifest."""
    manifest = {
        "resolved_at": datetime.now().astimezone().isoformat(),
        "source": "kite.instruments('NFO')[segment=NFO-FUT] ∩ kite.instruments('NSE')[instrument_type=EQ]",
        "count": len(symbols),
        "excluded_indices": excluded,
        "symbols": symbols,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Manifest saved: %s (%d symbols)", MANIFEST_PATH, len(symbols))


# ── Existing data deletion ──────────────────────────────────────────


def delete_existing_stock_data() -> list[str]:
    """Delete data for existing stock symbols to re-download fresh.

    Returns
    -------
    list[str]
        Symbols whose directories were deleted.
    """
    deleted = []
    for symbol in EXISTING_STOCK_SYMBOLS:
        symbol_dir = STOCKS_DIR / symbol
        if symbol_dir.exists():
            shutil.rmtree(symbol_dir)
            logger.info("Deleted existing data: %s", symbol_dir)
            deleted.append(symbol)
        else:
            logger.info("No existing data for %s — nothing to delete.", symbol)
    return deleted


# ── Symbol completeness check ──────────────────────────────────────


def is_symbol_complete(symbol: str) -> bool:
    """Check if all expected Parquet files exist for a symbol."""
    from infrastructure.data.loader import get_data_path

    for tf in FETCH_TIMEFRAMES:
        path = get_data_path(symbol, "stock", tf)
        if not path.exists():
            return False
    return True


# ── Corporate actions detection ─────────────────────────────────────


def detect_corporate_actions(
    symbol: str,
    threshold: float = _CORP_ACTION_THRESHOLD,
) -> list[dict]:
    """Detect candidate split/bonus events from daily OHLCV data.

    Computes the ratio of each day's open to the previous day's close.
    Flags rows where the ratio deviates from 1.0 by more than
    ``threshold``.

    Parameters
    ----------
    symbol : str
        Stock symbol.
    threshold : float
        Deviation threshold from 1.0.  Default 0.25 → flags ratios
        outside [0.75, 1.25].

    Returns
    -------
    list[dict]
        Each dict: {symbol, date, ratio, prev_close, open, close}.
    """
    import pandas as pd
    from infrastructure.data.loader import get_data_path

    daily_path = get_data_path(symbol, "stock", "daily")
    if not daily_path.exists():
        return []

    df = pd.read_parquet(daily_path)
    if len(df) < 2:
        return []

    df = df.sort_values("date").reset_index(drop=True)

    prev_close = df["close"].shift(1)
    ratio = df["open"] / prev_close

    # Flag: ratio outside [1 - threshold, 1 + threshold].
    mask = (ratio < (1 - threshold)) | (ratio > (1 + threshold))
    # Skip the first row (no previous close).
    mask.iloc[0] = False

    flagged = df[mask].copy()
    if flagged.empty:
        return []

    flagged_ratio = ratio[mask]
    flagged_prev_close = prev_close[mask]

    results = []
    for idx in flagged.index:
        row = flagged.loc[idx]
        # Handle timezone-aware datetime — format to date string.
        dt = row["date"]
        if hasattr(dt, "strftime"):
            date_str = dt.strftime("%Y-%m-%d")
        else:
            date_str = str(dt)

        results.append({
            "symbol": symbol,
            "date": date_str,
            "ratio": round(float(flagged_ratio.loc[idx]), 4),
            "prev_close": round(float(flagged_prev_close.loc[idx]), 2),
            "open": round(float(row["open"]), 2),
            "close": round(float(row["close"]), 2),
        })

    return results


# ── SQLite registration ─────────────────────────────────────────────


def register_instruments_in_db(symbols: list[str]) -> int:
    """Register F&O stock symbols in the SQLite instruments table.

    Uses INSERT OR IGNORE for idempotency.

    Returns
    -------
    int
        Number of newly inserted rows.
    """
    from infrastructure.db import get_connection

    conn = get_connection()
    try:
        before = conn.execute("SELECT COUNT(*) as c FROM instruments").fetchone()["c"]

        for symbol in symbols:
            conn.execute(
                "INSERT OR IGNORE INTO instruments (name, symbol, asset_type) "
                "VALUES (?, ?, ?)",
                (symbol, symbol, "stock"),
            )
        conn.commit()

        after = conn.execute("SELECT COUNT(*) as c FROM instruments").fetchone()["c"]
        inserted = after - before
        logger.info(
            "SQLite registration: %d new instruments (total: %d).",
            inserted, after,
        )
        return inserted
    finally:
        conn.close()


# ── Main backfill orchestration ─────────────────────────────────────


def main() -> None:
    """Execute the F&O universe backfill pipeline."""
    parser = argparse.ArgumentParser(
        description="Backfill historical data for all NSE F&O equities",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview universe and chunk plans without making API calls",
    )
    parser.add_argument(
        "--skip-delete",
        action="store_true",
        help="Skip deletion of existing stock data (useful for resume)",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end",
        default=str(date.today()),
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the post-fetch validation pass",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from dotenv import load_dotenv
    load_dotenv()

    from kiteconnect import KiteConnect  # type: ignore[import-untyped]

    from infrastructure.data.ingestion import backfill, lookup_token, register_instrument
    from infrastructure.data.loader import get_data_path

    api_key = os.environ["ZERODHA_API_KEY"]
    access_token = os.environ["ZERODHA_ACCESS_TOKEN"]

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    run_started_at = datetime.now().astimezone().isoformat()

    # ── Banner ──────────────────────────────────────────────────────
    print("=" * 65)
    print("  F&O Universe Historical Data Backfill")
    print(f"  Range:      {args.start} → {args.end}")
    print(f"  Timeframes: {', '.join(FETCH_TIMEFRAMES)}")
    if args.dry_run:
        print("  Mode:       DRY-RUN (no API calls, no file changes)")
    print("=" * 65)

    # ── Step 1: Resolve F&O equity universe ─────────────────────────
    print("\n[STEP 1] Resolving F&O equity universe...")

    fo_symbols, excluded_indices = resolve_fo_equity_universe(kite)

    print(f"    F&O equities:      {len(fo_symbols)}")
    print(f"    Excluded (indices): {len(excluded_indices)} → {excluded_indices}")

    # Save manifest.
    if not args.dry_run:
        save_manifest(fo_symbols, excluded_indices)
        print(f"    Manifest saved:    {MANIFEST_PATH}")

    # ── Step 2: Delete existing stock data ──────────────────────────
    if not args.dry_run and not args.skip_delete:
        print("\n[STEP 2] Deleting existing stock data for re-download...")
        deleted = delete_existing_stock_data()
        print(f"    Deleted: {len(deleted)} symbol(s) → {deleted}")
    else:
        if args.skip_delete:
            print("\n[STEP 2] Skipped deletion (--skip-delete).")
        else:
            print("\n[STEP 2] Skipped deletion (dry-run).")

    # ── Step 3: Resolve tokens ──────────────────────────────────────
    print("\n[STEP 3] Resolving NSE instrument tokens...")

    if not args.dry_run:
        failed_lookups = []
        for i, symbol in enumerate(fo_symbols, 1):
            try:
                token = lookup_token(symbol, exchange="NSE")
                if i <= 5 or i % 50 == 0:
                    print(f"    [{i}/{len(fo_symbols)}] {symbol}: token={token}")
            except ValueError as exc:
                logger.error("Token lookup failed for %s: %s", symbol, exc)
                failed_lookups.append(symbol)

        if failed_lookups:
            print(f"\n    [WARNING] Token lookup failed for {len(failed_lookups)} symbols:")
            for sym in failed_lookups:
                print(f"      - {sym}")
            # Remove failed symbols from the universe.
            fo_symbols = [s for s in fo_symbols if s not in failed_lookups]
            print(f"    Proceeding with {len(fo_symbols)} symbols.")
    else:
        print(f"    [DRY-RUN] Would resolve tokens for {len(fo_symbols)} symbols.")

    # ── Step 4: Fetch data symbol-by-symbol ─────────────────────────
    print(f"\n[STEP 4] Fetching data for {len(fo_symbols)} symbols...")

    total_symbols = len(fo_symbols)
    total_timeframes = len(FETCH_TIMEFRAMES)
    total_jobs = total_symbols * total_timeframes

    symbols_completed = 0
    symbols_skipped = 0
    symbols_failed = 0
    failed_symbols = []
    all_corporate_actions = []
    row_counts = {tf: 0 for tf in FETCH_TIMEFRAMES}
    token_expired = False  # Early abort flag.

    for si, symbol in enumerate(fo_symbols, 1):
        # ── Abort if token has expired ──────────────────────────────
        if token_expired:
            symbols_failed += 1
            failed_symbols.append(symbol)
            continue

        # ── Check if symbol is already complete ─────────────────────
        if not args.dry_run and is_symbol_complete(symbol):
            symbols_skipped += 1
            logger.info(
                "[%d/%d] %s: SKIPPED (all %d files exist)",
                si, total_symbols, symbol, total_timeframes,
            )
            # Still detect corporate actions for skipped symbols.
            corp_actions = detect_corporate_actions(symbol)
            all_corporate_actions.extend(corp_actions)
            continue

        print(f"\n  [{si}/{total_symbols}] {symbol}")

        if args.dry_run:
            for tf in FETCH_TIMEFRAMES:
                backfill(
                    symbol=symbol,
                    asset_type="stock",
                    timeframe=tf,
                    start_date=args.start,
                    end_date=args.end,
                    dry_run=True,
                )
            symbols_completed += 1
            continue

        # ── Fetch each timeframe ────────────────────────────────────
        symbol_ok = True
        for ti, tf in enumerate(FETCH_TIMEFRAMES, 1):
            # Skip if this specific file already exists.
            tf_path = get_data_path(symbol, "stock", tf)
            if tf_path.exists():
                logger.info(
                    "    [%d/%d] %s %s: EXISTS — skipping",
                    ti, total_timeframes, symbol, tf,
                )
                continue

            try:
                print(f"    [{ti}/{total_timeframes}] {tf}...", end=" ", flush=True)
                df = backfill(
                    symbol=symbol,
                    asset_type="stock",
                    timeframe=tf,
                    start_date=args.start,
                    end_date=args.end,
                    dry_run=False,
                )
                rows = len(df)
                row_counts[tf] += rows
                print(f"{rows:,} rows")

            except Exception as exc:
                error_str = str(exc).lower()
                # Check the full exception chain for token errors.
                cause_str = str(exc.__cause__).lower() if exc.__cause__ else ""
                is_token_error = any(
                    keyword in error_str or keyword in cause_str
                    for keyword in ["access_token", "api_key", "token"]
                )

                logger.error(
                    "FAILED: %s %s — %s", symbol, tf, exc,
                )
                print(f"FAILED — {exc}")
                symbol_ok = False

                if is_token_error:
                    print("\n" + "!" * 65)
                    print("  ACCESS TOKEN EXPIRED — aborting all remaining symbols.")
                    print("  To resume:")
                    print("    1. python scripts/generate_token.py")
                    print("    2. python scripts/backfill_fo_universe.py --skip-delete")
                    print("!" * 65)
                    token_expired = True

                break  # Abort remaining timeframes for this symbol.

        if symbol_ok:
            symbols_completed += 1

            # ── Corporate actions detection ─────────────────────────
            corp_actions = detect_corporate_actions(symbol)
            if corp_actions:
                logger.info(
                    "    Corporate actions detected: %d event(s) for %s",
                    len(corp_actions), symbol,
                )
                for ca in corp_actions:
                    print(
                        f"    ⚡ CORP ACTION: {ca['date']}  "
                        f"ratio={ca['ratio']:.4f}  "
                        f"prev_close={ca['prev_close']:.2f} → open={ca['open']:.2f}"
                    )
            all_corporate_actions.extend(corp_actions)
        else:
            symbols_failed += 1
            failed_symbols.append(symbol)

        # ── Inter-symbol delay with jitter ──────────────────────────
        if si < total_symbols and not token_expired:
            delay = _INTER_SYMBOL_DELAY_BASE + random.uniform(0, _INTER_SYMBOL_JITTER_MAX)
            time.sleep(delay)

    # ── Step 5: Save corporate actions ──────────────────────────────
    if not args.dry_run:
        print(f"\n[STEP 5] Corporate actions: {len(all_corporate_actions)} candidate(s) detected.")

        CORP_ACTIONS_PATH.write_text(
            json.dumps(all_corporate_actions, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"    Saved: {CORP_ACTIONS_PATH}")

    # ── Step 6: Register in SQLite ──────────────────────────────────
    if not args.dry_run:
        print("\n[STEP 6] Registering instruments in SQLite...")
        try:
            newly_inserted = register_instruments_in_db(fo_symbols)
            print(f"    Newly registered: {newly_inserted}")
        except Exception as exc:
            logger.error("SQLite registration failed: %s", exc)
            print(f"    [ERROR] Registration failed: {exc}")

    # ── Step 7: Validation ──────────────────────────────────────────
    validation_summary = {"clean": 0, "issues": 0}

    if not args.dry_run and not args.skip_validation:
        print(f"\n[STEP 7] Validating data for {len(fo_symbols)} symbols...")

        from infrastructure.data.validation import validate

        validation_issues = []
        for si, symbol in enumerate(fo_symbols, 1):
            for tf in FETCH_TIMEFRAMES:
                try:
                    report = validate(symbol, "stock", tf)
                    if report["clean"]:
                        validation_summary["clean"] += 1
                    else:
                        validation_summary["issues"] += 1
                        validation_issues.append({
                            "symbol": symbol,
                            "timeframe": tf,
                            "issues": report["issues"],
                        })
                except Exception as exc:
                    validation_summary["issues"] += 1
                    validation_issues.append({
                        "symbol": symbol,
                        "timeframe": tf,
                        "issues": [f"Validation error: {exc}"],
                    })

            if si % 50 == 0 or si == len(fo_symbols):
                print(
                    f"    Validated {si}/{len(fo_symbols)} symbols — "
                    f"clean: {validation_summary['clean']}, issues: {validation_summary['issues']}"
                )

        if validation_issues:
            print(f"\n    Validation issues ({len(validation_issues)} file(s)):")
            for vi in validation_issues[:20]:  # Show first 20.
                print(f"      {vi['symbol']} {vi['timeframe']}: {'; '.join(vi['issues'])}")
            if len(validation_issues) > 20:
                print(f"      ... and {len(validation_issues) - 20} more.")
    elif args.skip_validation:
        print("\n[STEP 7] Skipped validation (--skip-validation).")

    # ── Step 8: Save run log ────────────────────────────────────────
    if not args.dry_run:
        run_log = {
            "started_at": run_started_at,
            "completed_at": datetime.now().astimezone().isoformat(),
            "start_date": args.start,
            "end_date": args.end,
            "universe_count": len(fo_symbols),
            "excluded_indices": excluded_indices,
            "timeframes": FETCH_TIMEFRAMES,
            "symbols_completed": symbols_completed,
            "symbols_skipped": symbols_skipped,
            "symbols_failed": symbols_failed,
            "failed_symbols": failed_symbols,
            "total_rows": row_counts,
            "corporate_actions_detected": len(all_corporate_actions),
            "validation_summary": validation_summary,
        }

        BACKFILL_LOG_PATH.write_text(
            json.dumps(run_log, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[STEP 8] Run log saved: {BACKFILL_LOG_PATH}")

    # ── Final summary ───────────────────────────────────────────────
    print("\n" + "=" * 65)

    if args.dry_run:
        print("  [DRY-RUN COMPLETE]")
        print(f"  Universe:     {len(fo_symbols)} F&O equities")
        print(f"  Timeframes:   {', '.join(FETCH_TIMEFRAMES)}")
        print(f"  Per symbol:   {sum(_chunks_per_tf())} API chunks")
        est_requests = len(fo_symbols) * sum(_chunks_per_tf())
        print(f"  Total calls:  ~{est_requests:,}")
        est_hours = est_requests * 0.8 / 3600
        print(f"  Est. runtime: ~{est_hours:.1f} hours")
        print("  Run without --dry-run to perform the actual backfill.")
    else:
        print("  [BACKFILL COMPLETE]")
        print(f"  Symbols completed:  {symbols_completed}")
        print(f"  Symbols skipped:    {symbols_skipped}")
        print(f"  Symbols failed:     {symbols_failed}")
        print(f"  Corporate actions:  {len(all_corporate_actions)} candidate(s)")
        print(f"  Validation:         {validation_summary['clean']} clean, {validation_summary['issues']} with issues")
        print()
        for tf in FETCH_TIMEFRAMES:
            print(f"    {tf:>8s}: {row_counts[tf]:>12,} rows")
        if failed_symbols:
            print(f"\n  FAILED SYMBOLS: {failed_symbols}")

    print("=" * 65)


def _chunks_per_tf() -> list[int]:
    """Estimate chunks per symbol for each timeframe (for dry-run summary)."""
    from math import ceil
    # Approximate span: 2020-02-22 to today ≈ 2310 days (as of Jun 2026).
    span_days = (date.today() - date(2020, 2, 22)).days
    max_days_map = {"daily": 2000, "15min": 200, "5min": 100, "3min": 100, "1min": 60}
    return [
        ceil(span_days / max_days_map[tf])
        for tf in FETCH_TIMEFRAMES
    ]


if __name__ == "__main__":
    main()
