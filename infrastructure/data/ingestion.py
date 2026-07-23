"""
Data Ingestion Module
=====================

Provides functions to fetch, store, and update market data from
Zerodha KiteConnect API.  Supports incremental updates, batch
processing with exponential backoff on errors, historical backfill
with date-range chunking, and runtime instrument registration.

All paths are constructed dynamically via ``loader.get_data_path``.
No hardcoded paths, no print statements — only Python logging.
"""

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from infrastructure.data.loader import get_data_path

logger = logging.getLogger(__name__)

# ── Module-level constants ──────────────────────────────────────────

TIMEFRAME_MAP: dict[str, dict[str, object]] = {
    "1min":  {"kite_interval": "minute",    "max_days": 60},
    "3min":  {"kite_interval": "3minute",   "max_days": 100},
    "5min":  {"kite_interval": "5minute",   "max_days": 100},
    "15min": {"kite_interval": "15minute",  "max_days": 200},
    "60min": {"kite_interval": "60minute",  "max_days": 400},
    "daily": {"kite_interval": "day",       "max_days": 2000},
}

INSTRUMENT_TOKENS: dict[str, int] = {
    "NIFTY":     256265,
    "BANKNIFTY": 260105,
    "INDIAVIX":  264969,
    "SENSEX":    265,
    "FINNIFTY":  257801,
}

# Rate-limiting settings for Zerodha API.
_NORMAL_DELAY_SECONDS: float = 0.5
_BACKOFF_BASE_SECONDS: float = 2.0
_MAX_RETRIES: int = 3


# ── Public API ──────────────────────────────────────────────────────


def fetch_and_store(
    symbol: str,
    asset_type: str,
    timeframe: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """Fetch historical data from Zerodha and save as a Parquet file.

    Loads ``ZERODHA_API_KEY`` and ``ZERODHA_ACCESS_TOKEN`` from the
    environment (via ``.env``).  Creates the instrument folder
    automatically if it does not exist.

    Parameters
    ----------
    symbol : str
        Instrument symbol, e.g. ``"NIFTY"``.
    asset_type : str
        One of ``"index"``, ``"stock"``, ``"volatility"``, ``"options"``.
    timeframe : str
        One of ``"1min"``, ``"15min"``, ``"60min"``, ``"daily"``.
    from_date : str
        Start date in ``"YYYY-MM-DD"`` format.
    to_date : str
        End date in ``"YYYY-MM-DD"`` format.

    Returns
    -------
    pd.DataFrame
        The fetched data as a DataFrame.

    Raises
    ------
    KeyError
        If *symbol* is not found in ``INSTRUMENT_TOKENS``.
        Call ``register_instrument()`` first to add it.

    Examples
    --------
    >>> df = fetch_and_store("NIFTY", "index", "daily", "2024-01-01",
    ...                      "2024-06-30")
    >>> len(df) > 0
    True
    """
    import os

    load_dotenv()

    api_key = os.environ["ZERODHA_API_KEY"]
    access_token = os.environ["ZERODHA_ACCESS_TOKEN"]

    from kiteconnect import KiteConnect  # type: ignore[import-untyped]

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    # Look up instrument token.
    if symbol not in INSTRUMENT_TOKENS:
        raise KeyError(
            f"Symbol '{symbol}' not found in INSTRUMENT_TOKENS. "
            f"Register it first with: register_instrument('{symbol}', <token>)\n"
            f"Available symbols: {sorted(INSTRUMENT_TOKENS.keys())}"
        )
    token = INSTRUMENT_TOKENS[symbol]

    # Resolve Kite interval string.
    tf_config = TIMEFRAME_MAP[timeframe]
    kite_interval = tf_config["kite_interval"]

    logger.info(
        "Fetching %s %s from %s to %s (interval=%s, token=%d)",
        symbol, timeframe, from_date, to_date, kite_interval, token,
    )

    raw_data = kite.historical_data(
        instrument_token=token,
        from_date=from_date,
        to_date=to_date,
        interval=kite_interval,
    )

    df = pd.DataFrame(raw_data)

    # Normalise column names to lowercase.
    df.columns = [col.lower() for col in df.columns]

    # Build save path and create folders automatically.
    save_path = get_data_path(symbol, asset_type, timeframe)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(save_path, index=False)

    logger.info(
        "Saved %s %s — %d rows → %s",
        symbol, timeframe, len(df), save_path,
    )
    return df


def update(
    symbol: str,
    asset_type: str,
    timeframe: str,
) -> pd.DataFrame:
    """Incrementally update an existing Parquet file with new data.

    Loads the existing file, finds the latest date, fetches data from
    the next day to today, concatenates, deduplicates, and saves back.
    If data is already current (latest date == today), no API call is
    made.

    Parameters
    ----------
    symbol : str
        Instrument symbol, e.g. ``"NIFTY"``.
    asset_type : str
        One of ``"index"``, ``"stock"``, ``"volatility"``, ``"options"``.
    timeframe : str
        One of ``"1min"``, ``"15min"``, ``"60min"``, ``"daily"``.

    Returns
    -------
    pd.DataFrame
        The full updated DataFrame.

    Examples
    --------
    >>> df = update("NIFTY", "index", "daily")
    >>> df.index.name
    'datetime'
    """
    from infrastructure.data.loader import load as load_data

    existing_df = load_data(symbol, asset_type, timeframe)

    latest_date = existing_df.index.max().date()
    today = date.today()

    if latest_date >= today:
        logger.info(
            "%s %s is already current (latest=%s). No update needed.",
            symbol, timeframe, latest_date,
        )
        return existing_df

    next_day = latest_date + timedelta(days=1)
    from_date_str = next_day.strftime("%Y-%m-%d")
    to_date_str = today.strftime("%Y-%m-%d")

    logger.info(
        "Updating %s %s: fetching %s → %s",
        symbol, timeframe, from_date_str, to_date_str,
    )

    new_df = fetch_and_store(symbol, asset_type, timeframe, from_date_str, to_date_str)

    # Reload existing, concatenate, deduplicate.
    existing_df = load_data(symbol, asset_type, timeframe)

    # Ensure new_df has a datetime column for proper concat.
    if "datetime" in new_df.columns:
        new_df["datetime"] = pd.to_datetime(new_df["datetime"])
        new_df = new_df.set_index("datetime")
    elif "date" in new_df.columns:
        new_df["date"] = pd.to_datetime(new_df["date"])
        new_df = new_df.set_index("date")
        new_df.index.name = "datetime"

    combined = pd.concat([existing_df, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    new_rows = len(combined) - len(existing_df)
    logger.info(
        "Update complete for %s %s — %d new rows added (total: %d).",
        symbol, timeframe, new_rows, len(combined),
    )

    # Save back to parquet.
    save_path = get_data_path(symbol, asset_type, timeframe)
    combined.to_parquet(save_path)

    return combined


def batch_update(instruments: list[tuple[str, str, str]]) -> None:
    """Update multiple instruments with rate limiting and exponential backoff.

    Calls :func:`update` for each instrument.  On failure, retries with
    exponential backoff (2 s → 4 s → skip).  A 0.5 s delay is inserted
    between successful calls to respect Zerodha rate limits.

    Parameters
    ----------
    instruments : list[tuple[str, str, str]]
        List of ``(symbol, asset_type, timeframe)`` tuples.

    Examples
    --------
    >>> batch_update([
    ...     ("NIFTY", "index", "daily"),
    ...     ("BANKNIFTY", "index", "daily"),
    ...     ("INDIAVIX", "volatility", "daily"),
    ... ])
    """
    succeeded = 0
    failed = 0

    for symbol, asset_type, timeframe in instruments:
        attempt = 0
        success = False

        while attempt < _MAX_RETRIES and not success:
            try:
                update(symbol, asset_type, timeframe)
                success = True
                succeeded += 1

                # Normal rate-limit delay between successful calls.
                time.sleep(_NORMAL_DELAY_SECONDS)

            except Exception as exc:
                attempt += 1
                if attempt < _MAX_RETRIES:
                    backoff = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                    logger.warning(
                        "Attempt %d/%d failed for %s %s %s: %s — "
                        "retrying in %.1f s",
                        attempt, _MAX_RETRIES, symbol, asset_type,
                        timeframe, exc, backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        "All %d attempts failed for %s %s %s: %s — skipping.",
                        _MAX_RETRIES, symbol, asset_type, timeframe, exc,
                    )
                    failed += 1

    logger.info(
        "Batch update complete: %d succeeded, %d failed out of %d total.",
        succeeded, failed, len(instruments),
    )


def register_instrument(symbol: str, token: int) -> None:
    """Register a new instrument token at runtime.

    Adds the symbol–token pair to the module-level
    ``INSTRUMENT_TOKENS`` dictionary so that subsequent calls to
    :func:`fetch_and_store` can look it up without code changes.

    Parameters
    ----------
    symbol : str
        Instrument symbol, e.g. ``"RELIANCE"``.
    token : int
        Zerodha instrument token, e.g. ``738561``.

    Examples
    --------
    >>> register_instrument("RELIANCE", 738561)
    >>> INSTRUMENT_TOKENS["RELIANCE"]
    738561
    """
    INSTRUMENT_TOKENS[symbol] = token
    logger.info(
        "Registered instrument: %s → token %d. "
        "Total registered: %d.",
        symbol, token, len(INSTRUMENT_TOKENS),
    )


# Module-level cache for the full instrument list from Kite API.
_instrument_cache: Optional[list[dict]] = None


def lookup_token(
    symbol: str,
    exchange: str = "NSE",
) -> int:
    """Auto-discover an instrument token from Kite API and register it.

    Fetches the full instrument list from Kite (cached after
    first call), finds the matching tradingsymbol on the given
    exchange, and registers it via :func:`register_instrument`.

    Parameters
    ----------
    symbol : str
        Trading symbol, e.g. ``"EICHERMOT"``, ``"RELIANCE"``.
    exchange : str, optional
        Exchange to search on, default ``"NSE"``.

    Returns
    -------
    int
        The instrument token.

    Raises
    ------
    ValueError
        If the symbol is not found on the given exchange.

    Examples
    --------
    >>> token = lookup_token("EICHERMOT")
    >>> token > 0
    True
    """
    import os

    global _instrument_cache

    # Return immediately if already registered.
    if symbol in INSTRUMENT_TOKENS:
        return INSTRUMENT_TOKENS[symbol]

    # Fetch instrument list (once per session).
    if _instrument_cache is None:
        load_dotenv()
        api_key = os.environ["ZERODHA_API_KEY"]
        access_token = os.environ["ZERODHA_ACCESS_TOKEN"]

        from kiteconnect import KiteConnect  # type: ignore[import-untyped]

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)

        logger.info("Fetching full instrument list from Kite API...")
        _instrument_cache = kite.instruments(exchange)
        logger.info(
            "Loaded %d instruments from %s.", len(_instrument_cache), exchange,
        )

    # Search for the symbol.
    for inst in _instrument_cache:
        if inst["tradingsymbol"] == symbol:
            token = inst["instrument_token"]
            register_instrument(symbol, token)
            return token

    raise ValueError(
        f"Symbol '{symbol}' not found on {exchange}. "
        f"Check spelling or try a different exchange."
    )


# ── Backfill (date-range chunking) ──────────────────────────────────


def _generate_chunks(
    start: date, end: date, max_days: int,
) -> list[tuple[date, date]]:
    """Split [start, end] into consecutive sub-ranges of *max_days*.

    Returns
    -------
    list[tuple[date, date]]
        Each element is ``(chunk_start, chunk_end)`` with
        ``chunk_end - chunk_start <= timedelta(days=max_days - 1)``.
    """
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=max_days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _staging_path(symbol: str, asset_type: str, timeframe: str) -> Path:
    """Return path for the staging parquet file used during backfill."""
    parquet_path = get_data_path(symbol, asset_type, timeframe)
    return parquet_path.parent / f"{timeframe}_backfill_staging.parquet"


def _checkpoint_path(symbol: str, asset_type: str, timeframe: str) -> Path:
    """Return path for the backfill checkpoint JSON file."""
    parquet_path = get_data_path(symbol, asset_type, timeframe)
    return parquet_path.parent / f"{timeframe}_backfill_progress.json"


def _load_checkpoint(
    symbol: str, asset_type: str, timeframe: str,
) -> Optional[dict]:
    """Load an existing checkpoint, or return None."""
    cp = _checkpoint_path(symbol, asset_type, timeframe)
    if cp.exists():
        data = json.loads(cp.read_text(encoding="utf-8"))
        logger.info("Resuming from checkpoint: %s (completed %d chunks)",
                    cp, data.get("completed_chunks", 0))
        return data
    return None


def _save_checkpoint(
    symbol: str, asset_type: str, timeframe: str,
    completed_chunks: int, total_chunks: int,
    total_rows: int, last_chunk_end: str,
) -> None:
    """Persist progress so the backfill can resume after interruption."""
    cp = _checkpoint_path(symbol, asset_type, timeframe)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(
        json.dumps({
            "completed_chunks": completed_chunks,
            "total_chunks": total_chunks,
            "total_rows": total_rows,
            "last_chunk_end": last_chunk_end,
        }, indent=2),
        encoding="utf-8",
    )


def _cleanup_backfill_files(
    symbol: str, asset_type: str, timeframe: str,
) -> None:
    """Remove checkpoint and staging files after successful completion."""
    for path in (
        _checkpoint_path(symbol, asset_type, timeframe),
        _staging_path(symbol, asset_type, timeframe),
    ):
        if path.exists():
            path.unlink()
            logger.info("Removed backfill file: %s", path)


def _append_to_staging(
    symbol: str, asset_type: str, timeframe: str,
    chunk_df: pd.DataFrame,
) -> None:
    """Append a chunk to the staging parquet file on disk.

    If the staging file doesn't exist yet, creates it.
    If it exists, reads it, concatenates, and overwrites.
    This ensures data survives crashes — nothing is held only in memory.
    """
    staging = _staging_path(symbol, asset_type, timeframe)
    staging.parent.mkdir(parents=True, exist_ok=True)

    if staging.exists():
        existing = pd.read_parquet(staging)
        combined = pd.concat([existing, chunk_df], ignore_index=True)
    else:
        combined = chunk_df

    combined.to_parquet(staging, index=False)


def backfill(
    symbol: str,
    asset_type: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Fetch historical data over a large date range using chunked API calls.

    Splits ``[start_date, end_date]`` into chunks that respect the
    Zerodha API limit (e.g. 60 days for 1-minute data), fetches each
    chunk sequentially with rate limiting and retry logic, and merges
    all results with any existing data on disk.

    Each chunk is saved to a staging parquet file on disk immediately
    after fetching, so data survives crashes.  A checkpoint JSON file
    tracks which chunks are done, so the process resumes from where
    it left off.

    Parameters
    ----------
    symbol : str
        Instrument symbol, e.g. ``"NIFTY"``.
    asset_type : str
        One of ``"index"``, ``"stock"``, ``"volatility"``, ``"options"``.
    timeframe : str
        One of ``"1min"``, ``"15min"``, ``"60min"``, ``"daily"``.
    start_date : str
        Start date in ``"YYYY-MM-DD"`` format.
    end_date : str
        End date in ``"YYYY-MM-DD"`` format.
    dry_run : bool, optional
        If True, only log the chunk plan without making API calls.

    Returns
    -------
    pd.DataFrame
        The full combined DataFrame (backfilled + existing data).

    Examples
    --------
    >>> df = backfill("NIFTY", "index", "1min",
    ...              "2020-02-22", "2025-12-22", dry_run=True)
    """
    import os

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    tf_config = TIMEFRAME_MAP[timeframe]
    max_days = int(tf_config["max_days"])
    kite_interval = tf_config["kite_interval"]

    chunks = _generate_chunks(start, end, max_days)
    total_chunks = len(chunks)

    logger.info(
        "Backfill plan: %s %s %s from %s to %s → %d chunks (max %d days each)",
        symbol, asset_type, timeframe, start_date, end_date,
        total_chunks, max_days,
    )

    # ── Dry-run: just log the chunks and return ─────────────────────
    if dry_run:
        for i, (cs, ce) in enumerate(chunks, 1):
            days = (ce - cs).days + 1
            logger.info(
                "  [DRY-RUN] Chunk %2d/%d: %s → %s (%d days)",
                i, total_chunks, cs, ce, days,
            )
        logger.info("Dry-run complete. No API calls were made.")
        return pd.DataFrame()

    # ── Live: set up Kite client ────────────────────────────────────
    load_dotenv()
    api_key = os.environ["ZERODHA_API_KEY"]
    access_token = os.environ["ZERODHA_ACCESS_TOKEN"]

    from kiteconnect import KiteConnect  # type: ignore[import-untyped]

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    if symbol not in INSTRUMENT_TOKENS:
        raise KeyError(
            f"Symbol '{symbol}' not found in INSTRUMENT_TOKENS. "
            f"Register it first with: register_instrument('{symbol}', <token>)"
        )
    token = INSTRUMENT_TOKENS[symbol]

    # ── Resume from checkpoint if available ─────────────────────────
    checkpoint = _load_checkpoint(symbol, asset_type, timeframe)
    skip_until = 0
    if checkpoint:
        skip_until = checkpoint["completed_chunks"]

    # ── Fetch each chunk (saved to disk immediately) ────────────────
    total_rows_fetched = 0

    for i, (cs, ce) in enumerate(chunks, 1):
        if i <= skip_until:
            logger.info(
                "  Chunk %2d/%d: skipped (already on disk)", i, total_chunks,
            )
            continue

        cs_str = cs.strftime("%Y-%m-%d")
        ce_str = ce.strftime("%Y-%m-%d")
        days = (ce - cs).days + 1

        attempt = 0
        success = False

        while attempt < _MAX_RETRIES and not success:
            try:
                logger.info(
                    "  Chunk %2d/%d: fetching %s → %s (%d days) ...",
                    i, total_chunks, cs_str, ce_str, days,
                )
                raw_data = kite.historical_data(
                    instrument_token=token,
                    from_date=cs_str,
                    to_date=ce_str,
                    interval=kite_interval,
                )
                chunk_df = pd.DataFrame(raw_data)
                chunk_df.columns = [col.lower() for col in chunk_df.columns]

                rows = len(chunk_df)
                total_rows_fetched += rows
                success = True

                # Save chunk to staging parquet on disk immediately.
                _append_to_staging(symbol, asset_type, timeframe, chunk_df)

                logger.info(
                    "  Chunk %2d/%d: OK — %d rows (running total: %d)",
                    i, total_chunks, rows, total_rows_fetched,
                )

                # Save checkpoint after data is safely on disk.
                _save_checkpoint(
                    symbol, asset_type, timeframe,
                    completed_chunks=i,
                    total_chunks=total_chunks,
                    total_rows=total_rows_fetched,
                    last_chunk_end=ce_str,
                )

                # Rate-limit delay (skip after the last chunk).
                if i < total_chunks:
                    time.sleep(_NORMAL_DELAY_SECONDS)

            except Exception as exc:
                attempt += 1
                if attempt < _MAX_RETRIES:
                    backoff = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                    logger.warning(
                        "  Chunk %2d/%d attempt %d/%d failed: %s — "
                        "retrying in %.1f s",
                        i, total_chunks, attempt, _MAX_RETRIES, exc, backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        "  Chunk %2d/%d: ALL %d attempts failed: %s — ABORTING.",
                        i, total_chunks, _MAX_RETRIES, exc,
                    )
                    raise RuntimeError(
                        f"Backfill failed at chunk {i}/{total_chunks} "
                        f"({cs_str} → {ce_str}). "
                        f"Re-run to resume from checkpoint."
                    ) from exc

    # ── Load staging data from disk ─────────────────────────────────
    staging = _staging_path(symbol, asset_type, timeframe)
    if staging.exists():
        backfilled = pd.read_parquet(staging)
        logger.info(
            "Loaded %d rows from staging file.", len(backfilled),
        )
    else:
        logger.warning("No staging file found — nothing was fetched.")
        backfilled = pd.DataFrame()

    # ── Merge with existing data on disk ────────────────────────────
    save_path = get_data_path(symbol, asset_type, timeframe)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and len(backfilled) > 0:
        existing = pd.read_parquet(save_path)
        logger.info(
            "Merging with existing file: %d existing rows.", len(existing),
        )
        combined = pd.concat([backfilled, existing], ignore_index=True)
    elif len(backfilled) > 0:
        combined = backfilled
    else:
        logger.warning("No data to save.")
        return pd.DataFrame()

    # Deduplicate and sort — same approach as fetch_and_store.
    # No datetime conversion needed; values are already correct
    # tz-aware IST datetimes from the Kite API / parquet.
    dt_col = "datetime" if "datetime" in combined.columns else "date"
    combined = combined.drop_duplicates(subset=[dt_col], keep="last")
    combined = combined.sort_values(dt_col).reset_index(drop=True)

    combined.to_parquet(save_path, index=False)

    logger.info(
        "Backfill saved: %s — %d total rows → %s",
        f"{symbol} {timeframe}", len(combined), save_path,
    )

    # Clean up staging + checkpoint on success.
    _cleanup_backfill_files(symbol, asset_type, timeframe)

    return combined
