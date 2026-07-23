"""
Data Validation Module
======================

Provides functions to run quality-assurance checks on market data
stored as Parquet files.  All checks are non-destructive: they
report issues without modifying the underlying data.

No hardcoded paths, no print statements — only Python logging.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from infrastructure.data.loader import load

# Resolved path to data/events/market_holidays.csv.
# Uses the same DATA_ROOT logic as loader.py (3 parents up).
_HOLIDAYS_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "events" / "market_holidays.csv"
)

# Module-level cache for market holidays (loaded once).
_market_holidays_cache: Optional[pd.DatetimeIndex] = None

logger = logging.getLogger(__name__)

# Price columns to inspect when present in the DataFrame.
_PRICE_COLUMNS: list[str] = ["open", "high", "low", "close"]


# ── Public API ──────────────────────────────────────────────────────


def validate(
    symbol: str,
    asset_type: str,
    timeframe: str,
    spike_threshold: float = 0.10,
) -> dict:
    """Run five quality checks on an instrument's data file.

    Parameters
    ----------
    symbol : str
        Instrument symbol, e.g. ``"NIFTY"``.
    asset_type : str
        One of ``"index"``, ``"stock"``, ``"volatility"``, ``"options"``.
    timeframe : str
        One of ``"1min"``, ``"15min"``, ``"60min"``, ``"daily"``.
    spike_threshold : float, optional
        Maximum single-candle percentage change (absolute) in the
        ``close`` column before it is flagged as an extreme spike.
        Default is ``0.10`` (10 %).

    Returns
    -------
    dict
        Validation report with exactly these keys:

        - ``symbol`` (str)
        - ``asset_type`` (str)
        - ``timeframe`` (str)
        - ``rows`` (int)
        - ``date_range`` (str) — ``"<start> → <end>"``
        - ``clean`` (bool) — ``True`` only if zero issues
        - ``issues`` (list[str]) — human-readable issue descriptions

    Examples
    --------
    >>> report = validate("NIFTY", "index", "daily")
    >>> report["clean"]
    True
    """
    df = load(symbol, asset_type, timeframe)
    issues: list[str] = []

    # ── Check 1 — Missing timestamps ────────────────────────────────
    missing_count = _check_missing_timestamps(df, timeframe)
    if missing_count > 0:
        issues.append(f"Missing timestamps: {missing_count} gap(s) detected.")

    # ── Check 2 — Zero or negative prices ───────────────────────────
    bad_price_count = _check_zero_or_negative_prices(df)
    if bad_price_count > 0:
        issues.append(
            f"Zero or negative prices: {bad_price_count} row(s) affected."
        )

    # ── Check 3 — Corrupt candles (high < low) ──────────────────────
    corrupt_count = _check_corrupt_candles(df)
    if corrupt_count > 0:
        issues.append(f"Corrupt candles (high < low): {corrupt_count} row(s).")

    # ── Check 4 — Extreme price spikes ──────────────────────────────
    spike_count, spike_timestamps = _check_extreme_spikes(df, spike_threshold)
    if spike_count > 0:
        ts_list = ", ".join(str(t) for t in spike_timestamps[:10])
        suffix = (
            f" (showing first 10 of {spike_count})"
            if spike_count > 10
            else ""
        )
        issues.append(
            f"Extreme price spikes (>{spike_threshold:.0%}): "
            f"{spike_count} occurrence(s) at [{ts_list}]{suffix}."
        )

    # ── Check 5 — Duplicate timestamps ──────────────────────────────
    dup_count = _check_duplicate_timestamps(df)
    if dup_count > 0:
        issues.append(f"Duplicate timestamps: {dup_count} duplicate(s).")

    # ── Build report ────────────────────────────────────────────────
    date_range = (
        f"{df.index.min()} → {df.index.max()}" if len(df) else "N/A"
    )

    report: dict = {
        "symbol": symbol,
        "asset_type": asset_type,
        "timeframe": timeframe,
        "rows": len(df),
        "date_range": date_range,
        "clean": len(issues) == 0,
        "issues": issues,
    }

    if report["clean"]:
        logger.info(
            "Validation PASSED for %s %s %s — %d rows, no issues.",
            symbol, asset_type, timeframe, len(df),
        )
    else:
        logger.warning(
            "Validation FAILED for %s %s %s — %d issue(s): %s",
            symbol, asset_type, timeframe, len(issues), "; ".join(issues),
        )

    return report


def validate_batch(
    instruments: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """Validate multiple instruments and return a summary DataFrame.

    Parameters
    ----------
    instruments : list[tuple[str, str, str]]
        List of ``(symbol, asset_type, timeframe)`` tuples.

    Returns
    -------
    pd.DataFrame
        One row per instrument with columns matching the keys returned
        by :func:`validate`.  Instruments that raise exceptions are
        included with ``clean=False`` and the error message in
        ``issues``.

    Examples
    --------
    >>> summary = validate_batch([
    ...     ("NIFTY", "index", "daily"),
    ...     ("INDIAVIX", "volatility", "daily"),
    ... ])
    >>> list(summary.columns)
    ['symbol', 'asset_type', 'timeframe', 'rows', 'date_range', 'clean', 'issues']
    """
    results: list[dict] = []

    for symbol, asset_type, timeframe in instruments:
        try:
            report = validate(symbol, asset_type, timeframe)
            results.append(report)
        except Exception as exc:
            logger.error(
                "Validation error for %s %s %s: %s",
                symbol, asset_type, timeframe, exc,
            )
            results.append(
                {
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "timeframe": timeframe,
                    "rows": 0,
                    "date_range": "N/A",
                    "clean": False,
                    "issues": [f"Error: {exc}"],
                }
            )

    return pd.DataFrame(results)


# ── Private helpers ─────────────────────────────────────────────────


def _load_market_holidays() -> pd.DatetimeIndex:
    """Load NSE market holidays from the CSV calendar.

    Returns
    -------
    pd.DatetimeIndex
        Sorted index of all known market holiday dates.
        Returns an empty index if the file is missing or empty.
    """
    global _market_holidays_cache  # noqa: PLW0603

    if _market_holidays_cache is not None:
        return _market_holidays_cache

    if not _HOLIDAYS_PATH.exists():
        logger.warning(
            "Market holidays file not found at %s — "
            "timestamp gap checks will NOT account for holidays.",
            _HOLIDAYS_PATH,
        )
        _market_holidays_cache = pd.DatetimeIndex([])
        return _market_holidays_cache

    holidays_df = pd.read_csv(_HOLIDAYS_PATH)
    if "date" not in holidays_df.columns or holidays_df.empty:
        _market_holidays_cache = pd.DatetimeIndex([])
        return _market_holidays_cache

    _market_holidays_cache = pd.to_datetime(holidays_df["date"])
    logger.info(
        "Loaded %d market holidays from %s",
        len(_market_holidays_cache), _HOLIDAYS_PATH,
    )
    return _market_holidays_cache


def _check_missing_timestamps(df: pd.DataFrame, timeframe: str) -> int:
    """Count gaps in the expected timestamp sequence.

    For daily data, checks against business days minus known NSE
    market holidays (loaded from ``data/events/market_holidays.csv``).
    For intraday timeframes (1min, 15min, 60min), checks for gaps
    during NSE market hours 09:15–15:30 on actual trading days.
    """
    if len(df) < 2:
        return 0

    holidays = _load_market_holidays()

    if timeframe == "daily":
        # Strip timezone from everything — we only care about calendar dates.
        # Zerodha returns tz-aware (IST), holidays CSV is tz-naive,
        # pd.bdate_range inherits tz from input.  Normalise all to naive.
        idx_min = df.index.min().tz_localize(None) if df.index.tz else df.index.min()
        idx_max = df.index.max().tz_localize(None) if df.index.tz else df.index.max()

        expected = pd.bdate_range(start=idx_min, end=idx_max)  # tz-naive
        # Remove known market holidays from expected (both tz-naive now).
        if len(holidays) > 0:
            expected = expected.difference(holidays)

        actual_dates = df.index.normalize()
        if actual_dates.tz is not None:
            actual_dates = actual_dates.tz_localize(None)
        actual_dates = actual_dates.unique()

        missing = expected.difference(actual_dates)
        return len(missing)

    # Intraday: build expected timestamps during market hours.
    freq_map = {"1min": "1min", "15min": "15min", "60min": "60min"}
    freq = freq_map.get(timeframe)
    if freq is None:
        return 0

    # Strip timezone for consistent comparison.
    start_date = df.index.min().tz_localize(None) if df.index.tz else df.index.min()
    end_date = df.index.max().tz_localize(None) if df.index.tz else df.index.max()
    start_date = start_date.normalize()
    end_date = end_date.normalize()

    business_days = pd.bdate_range(start=start_date, end=end_date)
    # Remove known market holidays for intraday too.
    if len(holidays) > 0:
        business_days = business_days.difference(holidays)

    # Make actual index tz-naive for comparison.
    actual_index = df.index
    if actual_index.tz is not None:
        actual_index = actual_index.tz_localize(None)

    missing_count = 0
    for day in business_days:
        market_open = day.replace(hour=9, minute=15)
        market_close = day.replace(hour=15, minute=30)

        # Candle timestamps mark the START of each period.
        # Last candle starts at market_close - interval:
        #   15min → 15:15,  1min → 15:29,  60min → 15:00
        # No candle starts AT 15:30 (there's no 15:30–15:45 session).
        last_candle = market_close - pd.Timedelta(freq)
        expected_range = pd.date_range(
            start=market_open, end=last_candle, freq=freq,
        )
        actual_in_day = actual_index[
            (actual_index >= market_open) & (actual_index <= last_candle)
        ]
        missing = expected_range.difference(actual_in_day)
        missing_count += len(missing)

    return missing_count


def _check_zero_or_negative_prices(df: pd.DataFrame) -> int:
    """Count rows where any price column is zero or negative."""
    cols = [c for c in _PRICE_COLUMNS if c in df.columns]
    if not cols:
        return 0
    mask = (df[cols] <= 0).any(axis=1)
    return int(mask.sum())


def _check_corrupt_candles(df: pd.DataFrame) -> int:
    """Count rows where high < low."""
    if "high" not in df.columns or "low" not in df.columns:
        return 0
    return int((df["high"] < df["low"]).sum())


def _check_extreme_spikes(
    df: pd.DataFrame, threshold: float,
) -> tuple[int, list]:
    """Find single-candle pct changes above *threshold* in close."""
    if "close" not in df.columns or len(df) < 2:
        return 0, []
    pct_change = df["close"].pct_change().abs()
    spike_mask = pct_change > threshold
    spike_timestamps = df.index[spike_mask].tolist()
    return len(spike_timestamps), spike_timestamps


def _check_duplicate_timestamps(df: pd.DataFrame) -> int:
    """Count duplicate index entries."""
    return int(df.index.duplicated().sum())
