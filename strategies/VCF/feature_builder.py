"""
VCF Feature Builder
===================

Assembles the master feature DataFrame for the Volatility
Capture Framework strategy by calling all four feature modules
and joining their outputs on date.

Three public functions:

- ``build(force=False)`` — creates ``vcf_master.parquet``
- ``load_master(start_date, end_date)`` — loads and slices
- ``build_summary(df)`` — prints diagnostics

Master parquet is saved to ``data/processed/VCF/``.
Metadata JSON tracks build freshness.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to sys.path for script-style execution.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from infrastructure.data.loader import load
from infrastructure.features.session import (
    build_dvr_consumed,
    build_session_features,
)
from infrastructure.features.vix import build_vix_features
from infrastructure.features.swings import build_swing_features
from infrastructure.features.calendar import build_calendar_features
from strategies.VCF.config import CONFIG

logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────

_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
_OUTPUT_DIR = _DATA_ROOT / "processed" / "VCF"
_MASTER_PATH = _OUTPUT_DIR / "vcf_master.parquet"
_METADATA_PATH = _OUTPUT_DIR / "vcf_master_metadata.json"

_EVENT_CALENDAR_PATH = _DATA_ROOT / "events" / "event_calendar.csv"
_HOLIDAY_PATH = _DATA_ROOT / "events" / "market_holidays.csv"

# Raw data file paths for freshness checks.
_RAW_FILES = [
    _DATA_ROOT / "raw" / "indices" / "NIFTY" / "daily.parquet",
    _DATA_ROOT / "raw" / "indices" / "NIFTY" / "15min.parquet",
    _DATA_ROOT / "raw" / "indices" / "NIFTY" / "1min.parquet",
    _DATA_ROOT / "raw" / "volatility" / "INDIAVIX" / "daily.parquet",
]


# ── Public API ──────────────────────────────────────────────────────


def build(force: bool = False) -> pd.DataFrame:
    """Build the full VCF master feature DataFrame.

    Parameters
    ----------
    force : bool
        If True, always rebuild regardless of freshness.
        If False (default), skip rebuild if master is newer
        than all raw data files.

    Returns
    -------
    pd.DataFrame
        The assembled master DataFrame (also saved to disk).
    """
    # ── Rebuild check ───────────────────────────────────────────
    if not force and _master_is_current():
        logger.info("Master is current, skipping rebuild.")
        print("[OK]  Master is current, skipping rebuild.")
        return pd.read_parquet(_MASTER_PATH)

    logger.info("Building VCF master feature DataFrame...")
    print("=" * 60)
    print("  Building VCF Master Feature DataFrame")
    print("=" * 60)

    # ── Load raw data ───────────────────────────────────────────
    nifty_daily = load("NIFTY", "index", "daily")
    vix_daily = load("INDIAVIX", "volatility", "daily")
    nifty_15min = load("NIFTY", "index", "15min")
    nifty_1min = load("NIFTY", "index", "1min")

    vcf = CONFIG["VCF"]
    market = CONFIG["MARKET"]

    # ── 1. Session features (daily — full history) ──────────────
    print("\n  [1/5] Building session features (daily)...")
    session_df = build_session_features(
        nifty_daily=nifty_daily,
        vix_daily=vix_daily,
        dvr_divisor=vcf["dvr_divisor"],
    )
    master = session_df.copy()
    print(f"        {len(master):,} rows")

    # ── 2. DVR consumed (15-minute — ~195 days) ─────────────────
    print("  [2/5] Building DVR consumed (15min)...")
    dvr_consumed = build_dvr_consumed(
        nifty_15min=nifty_15min,
        dvr_series=master["dvr"],
    )
    master = master.join(dvr_consumed, how="left")
    dvr_coverage = dvr_consumed.dropna(how="all").shape[0]
    print(f"        {dvr_coverage} days with coverage")

    # ── 3. VIX features (daily VIX — full history) ──────────────
    print("  [3/5] Building VIX features (daily)...")
    vix_df = build_vix_features(
        vix_daily=vix_daily,
        vix_bands=vcf["vix_bands"],
        downgrade_threshold=vcf["vix_direction_downgrade_threshold"],
    )

    # Left-join VIX onto Nifty dates.  Flag missing.
    master = master.join(vix_df, how="left")
    master["vix_data_missing"] = master["vix_regime"].isna()
    vix_missing = master["vix_data_missing"].sum()
    if vix_missing > 0:
        logger.warning("%d dates have missing VIX data", vix_missing)
    print(f"        {vix_missing} dates with missing VIX")

    # ── 4. Swing features (1-minute — ~58 days) ─────────────────
    print("  [4/5] Building swing features (1min)...")
    swing_df = build_swing_features(
        nifty_1min=nifty_1min,
        reversal_thresholds=vcf["swing_reversal_thresholds"],
        capture_zone_min=vcf["capture_zone_min"],
        capture_zone_max=vcf["capture_zone_max"],
        swing_reversal_default=vcf["swing_reversal_default"],
    )
    master = master.join(swing_df, how="left")
    swing_coverage = swing_df.dropna(subset=["swing_30_magnitude"]).shape[0]
    print(f"        {swing_coverage} days with coverage")

    # ── 5. Calendar features (derived — full history) ───────────
    print("  [5/5] Building calendar features...")
    calendar_df = build_calendar_features(
        trading_dates=master.index,
        event_calendar_path=_EVENT_CALENDAR_PATH,
        holiday_path=_HOLIDAY_PATH,
        expiry_change_date="2025-09-02",
    )
    master = master.join(calendar_df, how="left")

    # ── Drop first row (NaN prev_close / vix_prev_close) ───────
    master = master.iloc[1:]

    # ── Save ────────────────────────────────────────────────────
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master.to_parquet(_MASTER_PATH)
    logger.info("Saved master to %s", _MASTER_PATH)

    # ── Save metadata ──────────────────────────────────────────
    metadata = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "daily_data_range": (
            f"{nifty_daily.index.min().date()} to "
            f"{nifty_daily.index.max().date()}"
        ),
        "15min_data_range": (
            f"{nifty_15min.index.min().date()} to "
            f"{nifty_15min.index.max().date()}"
        ),
        "1min_data_range": (
            f"{nifty_1min.index.min().date()} to "
            f"{nifty_1min.index.max().date()}"
        ),
        "total_rows": len(master),
        "swing_coverage_days": swing_coverage,
        "dvr_consumed_coverage_days": dvr_coverage,
    }
    _METADATA_PATH.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    logger.info("Saved metadata to %s", _METADATA_PATH)

    # ── Summary ─────────────────────────────────────────────────
    build_summary(master)

    return master


def load_master(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load the saved master parquet, optionally sliced by date.

    Parameters
    ----------
    start_date, end_date : str, optional
        Slice bounds (inclusive).  Format: ``"YYYY-MM-DD"``.

    Returns
    -------
    pd.DataFrame
        The master DataFrame (or a date-sliced view of it).

    Raises
    ------
    FileNotFoundError
        If master has not been built yet.
    """
    if not _MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Master parquet not found at {_MASTER_PATH}. "
            f"Run build() first."
        )

    df = pd.read_parquet(_MASTER_PATH)

    if start_date is not None:
        df = df.loc[start_date:]
    if end_date is not None:
        df = df.loc[:end_date]

    logger.info(
        "Loaded master — %d rows [%s to %s]",
        len(df),
        df.index.min().date() if len(df) > 0 else "N/A",
        df.index.max().date() if len(df) > 0 else "N/A",
    )
    return df


def build_summary(df: pd.DataFrame) -> None:
    """Print a diagnostic summary of the master DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The master feature DataFrame.
    """
    print("\n" + "=" * 60)
    print("  VCF MASTER — BUILD SUMMARY")
    print("=" * 60)

    # ── Metadata ────────────────────────────────────────────────
    if _METADATA_PATH.exists():
        meta = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
        print(f"\n  Built at:      {meta.get('built_at', 'unknown')}")
        print(f"  Daily range:   {meta.get('daily_data_range', 'unknown')}")
        print(f"  15min range:   {meta.get('15min_data_range', 'unknown')}")
        print(f"  1min range:    {meta.get('1min_data_range', 'unknown')}")

    # ── Overview ────────────────────────────────────────────────
    print(f"\n  Total rows:    {len(df):,}")
    print(f"  Date range:    {df.index.min().date()} to {df.index.max().date()}")

    # ── VIX regime distribution ─────────────────────────────────
    if "vix_regime" in df.columns:
        print("\n  VIX Regime Distribution:")
        regime_counts = df["vix_regime"].value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(df) * 100
            print(f"    {regime:15s}  {count:5d}  ({pct:5.1f}%)")

    # ── Capture zone hit rates ──────────────────────────────────
    if "capture_zone_100" in df.columns:
        swing_valid = df["capture_zone_100"].notna()
        if swing_valid.any():
            n_valid = swing_valid.sum()
            cap100 = df.loc[swing_valid, "capture_zone_100"].sum()
            cap150 = df.loc[swing_valid, "capture_zone_150"].sum()
            print(f"\n  Capture Zone Hit Rates (on {n_valid} days with swing data):")
            print(f"    100pt zone:  {cap100:,} hits ({cap100/n_valid*100:.1f}%)")
            print(f"    150pt zone:  {cap150:,} hits ({cap150/n_valid*100:.1f}%)")

    # ── NaN report by column group ──────────────────────────────
    print("\n  NaN Report:")
    column_groups = {
        "session":      [c for c in df.columns if c.startswith("session_")],
        "dvr_base":     [c for c in df.columns if c in ("dvr", "dvr_ratio")],
        "dvr_consumed": [c for c in df.columns if c.startswith("dvr_consumed")],
        "vix":          [c for c in df.columns if c.startswith("vix_")],
        "swing":        [c for c in df.columns if c.startswith("swing_")],
        "capture_zone": [c for c in df.columns if c.startswith("capture_zone")],
        "calendar":     [c for c in df.columns if c in (
            "day_of_week", "week_of_month", "month",
            "is_expiry_day", "is_no_trade_day",
            "is_event_day", "is_pre_event", "is_post_event",
        )],
    }
    for group_name, cols in column_groups.items():
        if cols:
            nan_count = df[cols].isna().sum().sum()
            print(f"    {group_name:15s}  {nan_count:,} NaN values")

    # ── No-trade days ───────────────────────────────────────────
    if "is_no_trade_day" in df.columns:
        no_trade = df["is_no_trade_day"].sum()
        print(f"\n  No-trade days (expiry): {no_trade}")

    print("\n" + "=" * 60)


# ── Private helpers ─────────────────────────────────────────────────


def _master_is_current() -> bool:
    """Check if master parquet is newer than all raw data files.

    Returns
    -------
    bool
        True if master exists and is newer than all raw files.
    """
    if not _MASTER_PATH.exists() or not _METADATA_PATH.exists():
        return False

    try:
        meta = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
        built_at = datetime.fromisoformat(meta["built_at"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return False

    for raw_path in _RAW_FILES:
        if not raw_path.exists():
            continue
        raw_mtime = datetime.fromtimestamp(raw_path.stat().st_mtime)
        if raw_mtime > built_at:
            logger.info(
                "Raw file %s is newer than master (modified %s, built %s)",
                raw_path.name, raw_mtime, built_at,
            )
            return False

    return True


# ── CLI entry point ─────────────────────────────────────────────────


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Build VCF master DataFrame")
    parser.add_argument(
        "--force", action="store_true",
        help="Force rebuild even if master is current",
    )
    args = parser.parse_args()

    build(force=args.force)
