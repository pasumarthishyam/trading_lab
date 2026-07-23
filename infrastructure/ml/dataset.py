"""
ML Feature Dataset Module
=========================

Builds the two feature matrices consumed by the unsupervised models:

* :func:`build_stock_day_features` — per-``(stock, day)`` matrix over the
  whole F&O universe (~333K rows × 14 features).  A single DuckDB pass
  scans all 211 ``daily.parquet`` files and extracts raw OHLCV + the
  previous close; the rolling / ratio features are then computed in
  Pandas with strict point-in-time (trailing-window) discipline.

* :func:`build_market_day_features` — per-market-day matrix
  (~1,500 rows × 8 features) built from NIFTY + INDIAVIX daily data,
  reusing the shared ``infrastructure.features`` modules.

No hardcoded absolute paths; everything is resolved relative to the
repository root.  Logging only — no ``print``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from infrastructure.data.loader import load
from infrastructure.features.session import build_session_features
from infrastructure.features.vix import build_vix_features

logger = logging.getLogger(__name__)

# ── Path constants ──────────────────────────────────────────────────
#   this file : trading_lab/infrastructure/ml/dataset.py
#   parent(3) : trading_lab/                              <-- repo root
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
STOCKS_ROOT: Path = REPO_ROOT / "data" / "raw" / "stocks"
UNIVERSE_MANIFEST: Path = STOCKS_ROOT / "_fo_universe_manifest.json"

# Rolling-window length for volume / range / return z-scores (trading days).
ROLLING_WINDOW: int = 20
# Minimum observations before a trailing statistic is considered valid.
ROLLING_MIN_PERIODS: int = 10
# DVR divisor for market-day session features (project standard).
DVR_DIVISOR: int = 16

# The 12 scale-free features actually fed to Isolation Forest.  Raw
# ``volume`` and ``prev_close`` are deliberately excluded (absolute,
# not comparable across stocks); ``volume_ratio_20d`` carries the
# volume signal in scale-free form.
STOCK_FEATURE_COLUMNS: list[str] = [
    "return_pct",
    "abs_return_pct",
    "gap_pct",
    "abs_gap_pct",
    "range_pct",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "volume_ratio_20d",
    "volume_zscore_20d",
    "range_zscore_20d",
    "return_zscore_20d",
    "open_close_ratio",
]

MARKET_FEATURE_COLUMNS: list[str] = [
    "session_range_pct",
    "gap_pct",
    "dvr_ratio",
    "vix_level",
    "vix_change_pct",
    "body_pct",
    "upper_wick_ratio",
    "lower_wick_ratio",
]

# VIX regime bands are only needed to satisfy the shared feature
# builder's signature — we consume ``vix_close`` / ``vix_change_pct``,
# not the regime label, so the exact bands are immaterial here.
_VIX_BANDS: dict[str, tuple[float, float]] = {
    "low": (0.0, 13.0),
    "golden": (13.0, 18.0),
    "elevated": (18.0, 25.0),
    "high": (25.0, 1000.0),
}


# ── universe helper ─────────────────────────────────────────────────

def load_universe() -> list[str]:
    """Return the 211-symbol F&O equity universe from the manifest."""
    manifest = json.loads(UNIVERSE_MANIFEST.read_text(encoding="utf-8"))
    symbols = list(manifest["symbols"])
    logger.info("Loaded F&O universe — %d symbols", len(symbols))
    return symbols


# ── stock-day feature matrix ────────────────────────────────────────

def _scan_raw_ohlcv(universe: list[str]) -> pd.DataFrame:
    """Single DuckDB pass → raw OHLCV + prev_close for every symbol/day.

    Reads all ``daily.parquet`` files in one scan, extracts the symbol
    from the file path, and derives ``prev_close`` with a partitioned
    ``LAG`` window.  Only the compact result set is materialised into
    Pandas — the heavy multi-file read stays in DuckDB.
    """
    stocks_root = STOCKS_ROOT.as_posix()
    paths = [f"{stocks_root}/{sym}/daily.parquet" for sym in universe]
    existing = [p for p in paths if Path(p).exists()]
    missing = len(paths) - len(existing)
    if missing:
        logger.warning("%d symbols have no daily.parquet — skipped", missing)
    if not existing:
        raise FileNotFoundError("No daily.parquet files found for the universe.")

    # Bracketed SQL list literal of single-quoted POSIX paths.
    source = "[" + ", ".join("'" + p.replace("'", "''") + "'" for p in existing) + "]"

    sql = f"""
    WITH raw AS (
        SELECT
            regexp_extract(filename, 'stocks/([^/]+)/daily', 1) AS symbol,
            CAST(date AS TIMESTAMP)::DATE                        AS date,
            open, high, low, close, volume
        FROM read_parquet({source}, filename=true)
    )
    SELECT
        symbol, date, open, high, low, close, volume,
        LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS prev_close
    FROM raw
    ORDER BY symbol, date
    """

    con = duckdb.connect()
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()

    df["date"] = pd.to_datetime(df["date"])
    logger.info(
        "DuckDB scan → %d (symbol, day) rows across %d symbols",
        len(df),
        df["symbol"].nunique(),
    )
    return df


def _trailing_zscore(group_value: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Point-in-time z-score against the *trailing* window (excludes today).

    ``shift(1)`` guarantees the statistics use only information available
    before the current bar, so the score for a corporate-action day is
    measured against the calm days that preceded it.
    """
    shifted = group_value.shift(1)
    mean = shifted.rolling(window, min_periods=min_periods).mean()
    std = shifted.rolling(window, min_periods=min_periods).std()
    return (group_value - mean) / std


def build_stock_day_features(universe: Optional[list[str]] = None) -> pd.DataFrame:
    """Build the per-``(stock, day)`` feature matrix.

    Parameters
    ----------
    universe : list[str], optional
        Symbols to include.  Defaults to the full F&O manifest (211).

    Returns
    -------
    pd.DataFrame
        ``symbol``, ``date``, ``volume`` plus the 13 model features
        listed in :data:`STOCK_FEATURE_COLUMNS`.  No NaNs remain —
        information-less warm-up cells are imputed to neutral values
        (0 for returns / z-scores, 1 for ratios) so every trading day,
        including every known corporate-action date, is retained.
    """
    if universe is None:
        universe = load_universe()

    df = _scan_raw_ohlcv(universe)

    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = df["prev_close"]

    # ── price-derived, scale-free features (1–9, 14) ────────────────
    df["return_pct"] = (close - prev_close) / prev_close
    df["abs_return_pct"] = df["return_pct"].abs()
    df["gap_pct"] = (open_ - prev_close) / prev_close
    df["abs_gap_pct"] = df["gap_pct"].abs()
    df["range_pct"] = (high - low) / open_
    df["body_pct"] = (close - open_).abs() / open_
    df["upper_wick_pct"] = (high - np.maximum(open_, close)) / open_
    df["lower_wick_pct"] = (np.minimum(open_, close) - low) / open_
    df["open_close_ratio"] = open_ / prev_close

    # ── trailing-window features (10–13), computed per symbol ───────
    grp = df.groupby("symbol", sort=False)

    vol_mean = grp["volume"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).mean()
    )
    df["volume_ratio_20d"] = df["volume"] / vol_mean
    df["volume_zscore_20d"] = grp["volume"].transform(
        lambda s: _trailing_zscore(s, ROLLING_WINDOW, ROLLING_MIN_PERIODS)
    )
    df["range_zscore_20d"] = grp["range_pct"].transform(
        lambda s: _trailing_zscore(s, ROLLING_WINDOW, ROLLING_MIN_PERIODS)
    )
    df["return_zscore_20d"] = grp["abs_return_pct"].transform(
        lambda s: _trailing_zscore(s, ROLLING_WINDOW, ROLLING_MIN_PERIODS)
    )

    # ── clean up ────────────────────────────────────────────────────
    # Ratios default to 1.0 (parity / no change), everything else to 0.0
    # (no move / not anomalous on that axis).  ``inf`` arises when a
    # trailing std or mean is exactly zero — treat as missing first.
    df = df.replace([np.inf, -np.inf], np.nan)
    ratio_defaults = {"open_close_ratio": 1.0, "volume_ratio_20d": 1.0}
    for col in STOCK_FEATURE_COLUMNS:
        default = ratio_defaults.get(col, 0.0)
        df[col] = df[col].fillna(default)

    out_cols = ["symbol", "date", "volume", *STOCK_FEATURE_COLUMNS]
    result = df[out_cols].reset_index(drop=True)

    logger.info(
        "Built stock-day feature matrix — %d rows × %d features",
        len(result),
        len(STOCK_FEATURE_COLUMNS),
    )
    return result


# ── market-day feature matrix ───────────────────────────────────────

def build_market_day_features() -> pd.DataFrame:
    """Build the per-market-day feature matrix from NIFTY + INDIAVIX.

    Reuses :func:`infrastructure.features.session.build_session_features`
    (gap, DVR ratio) and
    :func:`infrastructure.features.vix.build_vix_features` (VIX level and
    momentum); the candle-shape features are derived from the shared
    session OHLC.

    Returns
    -------
    pd.DataFrame
        ``date`` plus the 8 features in :data:`MARKET_FEATURE_COLUMNS`.
        Rows with any missing feature (warm-up bar, days without VIX)
        are dropped.
    """
    nifty = load("NIFTY", "index", "daily")
    vix = load("INDIAVIX", "volatility", "daily")

    session = build_session_features(nifty, vix, dvr_divisor=DVR_DIVISOR)
    vix_feat = build_vix_features(vix, _VIX_BANDS, downgrade_threshold=0.05)

    high = session["session_high"]
    low = session["session_low"]
    open_ = session["session_open"]
    close = session["session_close"]
    rng = (high - low)

    out = pd.DataFrame(index=session.index)
    out.index.name = "date"

    out["session_range_pct"] = rng / open_
    out["gap_pct"] = session["gap_pct"]
    out["dvr_ratio"] = session["dvr_ratio"]
    out["body_pct"] = (close - open_).abs() / open_

    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    # Guard the degenerate zero-range day (range in the denominator).
    safe_rng = rng.replace(0.0, np.nan)
    out["upper_wick_ratio"] = upper_wick / safe_rng
    out["lower_wick_ratio"] = lower_wick / safe_rng

    # VIX features are indexed on VIX dates; align onto NIFTY calendar.
    out["vix_level"] = vix_feat["vix_close"].reindex(out.index)
    out["vix_change_pct"] = vix_feat["vix_change_pct"].reindex(out.index)

    out = out.replace([np.inf, -np.inf], np.nan)
    before = len(out)
    out = out.dropna(subset=MARKET_FEATURE_COLUMNS)
    dropped = before - len(out)

    result = out[MARKET_FEATURE_COLUMNS].reset_index()
    logger.info(
        "Built market-day feature matrix — %d rows × %d features (%d dropped)",
        len(result),
        len(MARKET_FEATURE_COLUMNS),
        dropped,
    )
    return result
