"""
Data Loader Module
==================

Provides functions to load, locate, and discover market data stored
as Parquet files in the trading_lab data directory.

All paths are constructed dynamically using pathlib.Path relative to
the repository root.  No hardcoded paths, no os.path, no print
statements — only Python logging at INFO level.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Path constants ──────────────────────────────────────────────────
# DATA_ROOT is resolved as two parents up from this file's location:
#   this file  : trading_lab/infrastructure/data/loader.py
#   parent (1) : trading_lab/infrastructure/data/
#   parent (2) : trading_lab/infrastructure/
#   parent (3) : trading_lab/                       <-- repo root
DATA_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Maps the logical asset_type argument to the folder name on disk.
ASSET_TYPE_FOLDER_MAP: dict[str, str] = {
    "index": "indices",
    "stock": "stocks",
    "volatility": "volatility",
    "options": "options",
}

VALID_TIMEFRAMES: set[str] = {"1min", "3min", "5min", "15min", "60min", "daily"}


# ── Public API ──────────────────────────────────────────────────────


def get_data_path(
    symbol: str,
    asset_type: str,
    timeframe: str,
    data_layer: str = "raw",
) -> Path:
    """Return the absolute path to a parquet file for a given instrument.

    Parameters
    ----------
    symbol : str
        Instrument symbol, e.g. ``"NIFTY"``, ``"RELIANCE"``.
    asset_type : str
        One of ``"index"``, ``"stock"``, ``"volatility"``, ``"options"``.
    timeframe : str
        One of ``"1min"``, ``"15min"``, ``"60min"``, ``"daily"``.
    data_layer : str, optional
        Data layer folder name, by default ``"raw"``.

    Returns
    -------
    Path
        Fully resolved path to the parquet file.

    Raises
    ------
    ValueError
        If *asset_type* or *timeframe* is not in the allowed set.

    Examples
    --------
    >>> get_data_path("NIFTY", "index", "1min")
    PosixPath('.../data/raw/indices/NIFTY/1min.parquet')
    """
    if asset_type not in ASSET_TYPE_FOLDER_MAP:
        raise ValueError(
            f"Invalid asset_type '{asset_type}'. "
            f"Must be one of {sorted(ASSET_TYPE_FOLDER_MAP.keys())}."
        )
    if timeframe not in VALID_TIMEFRAMES:
        raise ValueError(
            f"Invalid timeframe '{timeframe}'. "
            f"Must be one of {sorted(VALID_TIMEFRAMES)}."
        )

    folder_name = ASSET_TYPE_FOLDER_MAP[asset_type]

    # Options always use a fixed filename regardless of timeframe.
    if asset_type == "options":
        filename = "bhav_daily.parquet"
    else:
        filename = f"{timeframe}.parquet"

    path = DATA_ROOT / "data" / data_layer / folder_name / symbol / filename
    return path


def load(
    symbol: str,
    asset_type: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_layer: str = "raw",
    corporate_actions: str = "ignore",
) -> pd.DataFrame:
    """Load a parquet data file and return it as a datetime-indexed DataFrame.

    The function standardises the index by detecting either a ``"datetime"``
    or ``"date"`` column and converting it to a proper ``DatetimeIndex``.

    Parameters
    ----------
    symbol : str
        Instrument symbol, e.g. ``"NIFTY"``.
    asset_type : str
        One of ``"index"``, ``"stock"``, ``"volatility"``, ``"options"``.
    timeframe : str
        One of ``"1min"``, ``"15min"``, ``"60min"``, ``"daily"``.
    start_date : str, optional
        Inclusive start date filter in ``"YYYY-MM-DD"`` format.
    end_date : str, optional
        Inclusive end date filter in ``"YYYY-MM-DD"`` format.
    data_layer : str, optional
        Data layer folder name, by default ``"raw"``.
    corporate_actions : str, optional
        How to handle registered corporate-action ex-dates (demergers etc.)
        that leave a discontinuity in the raw series.  One of:

        - ``"ignore"``  : do nothing (default — backwards compatible).
        - ``"exclude"`` : drop the ex-date bar(s) so strategies never see
          the fake overnight gap.  **Recommended for backtests.**
        - ``"flag"``    : keep all bars but add a boolean ``is_corp_action``
          column.

        See ``data/events/corporate_actions.csv`` for the registry.

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``DatetimeIndex`` and all data columns.

    Raises
    ------
    FileNotFoundError
        If the parquet file does not exist on disk.
    ValueError
        If *corporate_actions* is not one of the allowed values.

    Examples
    --------
    >>> df = load("NIFTY", "index", "daily", start_date="2024-01-01")
    >>> df.index.name
    'datetime'
    >>> clean = load("NMDC", "stock", "daily", corporate_actions="exclude")
    """
    if corporate_actions not in ("ignore", "exclude", "flag"):
        raise ValueError(
            f"corporate_actions must be 'ignore', 'exclude', or 'flag'; "
            f"got {corporate_actions!r}."
        )
    path = get_data_path(symbol, asset_type, timeframe, data_layer)

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Run ingestion first:\n"
            f"  from infrastructure.data.ingestion import fetch_and_store\n"
            f"  fetch_and_store('{symbol}', '{asset_type}', '{timeframe}', "
            f"'<from_date>', '<to_date>')"
        )

    logger.info("Loading %s from %s", symbol, path)
    df = pd.read_parquet(path)

    # Standardise the index to DatetimeIndex.
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.index.name = "datetime"
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Attempt to coerce whatever index exists.
        df.index = pd.to_datetime(df.index)
        df.index.name = "datetime"

    df = df.sort_index()

    # Apply optional date filters.
    if start_date is not None:
        df = df.loc[start_date:]  # type: ignore[misc]
    if end_date is not None:
        df = df.loc[:end_date]  # type: ignore[misc]

    # Handle corporate-action ex-dates (demergers etc.) that create
    # discontinuities in the raw price series.
    if corporate_actions != "ignore":
        from infrastructure.data.corporate_actions import apply_corporate_actions

        mode = "exclude" if corporate_actions == "exclude" else "flag"
        df = apply_corporate_actions(df, symbol, mode=mode)

    logger.info(
        "Loaded %s %s %s — %d rows [%s → %s]",
        symbol,
        asset_type,
        timeframe,
        len(df),
        df.index.min() if len(df) else "N/A",
        df.index.max() if len(df) else "N/A",
    )
    return df


def list_available(asset_type: Optional[str] = None) -> dict[str, dict[str, list[str]]]:
    """Scan the ``data/raw/`` directory and report available instruments.

    Parameters
    ----------
    asset_type : str, optional
        If provided, limits the scan to a single asset category.
        Must be one of ``"index"``, ``"stock"``, ``"volatility"``,
        ``"options"``.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Nested mapping of ``{asset_type: {symbol: [timeframes]}}``.
        Example::

            {
                "index": {
                    "NIFTY": ["1min", "daily"],
                    "BANKNIFTY": ["daily"],
                },
                "stock": {
                    "RELIANCE": ["15min"],
                },
            }

    Examples
    --------
    >>> available = list_available()
    >>> available.get("index", {})
    {'NIFTY': ['1min', 'daily']}
    """
    raw_root = DATA_ROOT / "data" / "raw"
    result: dict[str, dict[str, list[str]]] = {}

    # Determine which asset types to scan.
    if asset_type is not None:
        if asset_type not in ASSET_TYPE_FOLDER_MAP:
            raise ValueError(
                f"Invalid asset_type '{asset_type}'. "
                f"Must be one of {sorted(ASSET_TYPE_FOLDER_MAP.keys())}."
            )
        types_to_scan = {asset_type: ASSET_TYPE_FOLDER_MAP[asset_type]}
    else:
        types_to_scan = ASSET_TYPE_FOLDER_MAP

    for a_type, folder_name in types_to_scan.items():
        folder_path = raw_root / folder_name
        if not folder_path.exists():
            continue

        symbols: dict[str, list[str]] = {}
        for symbol_dir in sorted(folder_path.iterdir()):
            if not symbol_dir.is_dir():
                continue
            timeframes: list[str] = []
            for parquet_file in sorted(symbol_dir.glob("*.parquet")):
                # Derive timeframe from filename (strip .parquet).
                tf = parquet_file.stem  # e.g. "1min", "daily", "bhav_daily"
                timeframes.append(tf)
            if timeframes:
                symbols[symbol_dir.name] = timeframes

        if symbols:
            result[a_type] = symbols

    logger.info("Discovered instruments: %s", result)
    return result
