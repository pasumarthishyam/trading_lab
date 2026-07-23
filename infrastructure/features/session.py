"""
Session Feature Module
======================

Computes per-trading-day session features from NIFTY data.

Two calculation paths:
    - ``build_session_features``: daily OHLCV + VIX → full history
    - ``build_dvr_consumed``: 15-minute data → ~195 days only

No hardcoded values — all parameters from function arguments.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Hour-end boundaries for DVR consumed calculation (IST, from 09:15 open).
_HOUR_ENDS: list[str] = [
    "10:15",  # h1: 09:15 – 10:15
    "11:15",  # h2: 09:15 – 11:15
    "12:15",  # h3: 09:15 – 12:15
    "13:15",  # h4: 09:15 – 13:15
    "14:15",  # h5: 09:15 – 14:15
    "15:15",  # h6: 09:15 – 15:15
]


def build_session_features(
    nifty_daily: pd.DataFrame,
    vix_daily: pd.DataFrame,
    dvr_divisor: int,
) -> pd.DataFrame:
    """Build session-level features from daily OHLCV data.

    Parameters
    ----------
    nifty_daily : pd.DataFrame
        NIFTY daily OHLCV with DatetimeIndex.
    vix_daily : pd.DataFrame
        INDIAVIX daily OHLCV with DatetimeIndex.
    dvr_divisor : int
        Divisor for DVR calculation (typically 16).

    Returns
    -------
    pd.DataFrame
        One row per trading day, indexed by date (tz-naive).
    """
    # Normalise indices to tz-naive dates for reliable joins.
    nifty = nifty_daily.copy()
    nifty.index = nifty.index.normalize()
    if nifty.index.tz is not None:
        nifty.index = nifty.index.tz_localize(None)

    vix = vix_daily[["open"]].copy()
    vix.index = vix.index.normalize()
    if vix.index.tz is not None:
        vix.index = vix.index.tz_localize(None)
    vix = vix.rename(columns={"open": "vix_open_for_dvr"})

    # Left-join VIX open onto Nifty dates.
    merged = nifty.join(vix, how="left")

    result = pd.DataFrame(index=merged.index)
    result.index.name = "date"

    result["session_open"] = merged["open"].values
    result["session_high"] = merged["high"].values
    result["session_low"] = merged["low"].values
    result["session_close"] = merged["close"].values
    result["session_range"] = result["session_high"] - result["session_low"]

    result["prev_close"] = result["session_close"].shift(1)
    result["gap"] = result["session_open"] - result["prev_close"]
    result["gap_pct"] = result["gap"] / result["prev_close"]

    # DVR = session_open * (vix_open / 100) / dvr_divisor
    vix_open = merged["vix_open_for_dvr"].values
    result["dvr"] = (
        result["session_open"] * (vix_open / 100.0) / dvr_divisor
    )
    result["dvr_ratio"] = result["session_range"] / result["dvr"]

    logger.info(
        "Built session features — %d rows [%s to %s]",
        len(result),
        result.index.min().date(),
        result.index.max().date(),
    )
    return result


def build_dvr_consumed(
    nifty_15min: pd.DataFrame,
    dvr_series: pd.Series,
) -> pd.DataFrame:
    """Compute cumulative DVR consumption at each hour boundary.

    For each trading day, tracks how the cumulative range
    (running_high - running_low from session open) grows
    relative to DVR by the end of each hour.

    Parameters
    ----------
    nifty_15min : pd.DataFrame
        NIFTY 15-minute OHLCV with DatetimeIndex.
    dvr_series : pd.Series
        DVR values indexed by date (tz-naive), from
        ``build_session_features`` output.

    Returns
    -------
    pd.DataFrame
        Columns ``dvr_consumed_h1`` through ``dvr_consumed_h6``,
        indexed by date. NaN for days where 15min data is missing.
    """
    df = nifty_15min.copy()

    # Strip timezone for consistent grouping.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    records: list[dict] = []

    for trade_date, day_data in df.groupby(df.index.date):
        day_data = day_data.sort_index()

        date_key = pd.Timestamp(trade_date)
        dvr_val = dvr_series.get(date_key, np.nan)

        if np.isnan(dvr_val) or dvr_val <= 0:
            # Cannot compute consumed ratio without valid DVR.
            records.append({
                "date": date_key,
                **{f"dvr_consumed_h{i}": np.nan for i in range(1, 7)},
            })
            continue

        # Running high and low from session open, cumulative.
        running_high = day_data["high"].cummax()
        running_low = day_data["low"].cummin()
        running_range = running_high - running_low

        row: dict = {"date": date_key}

        for i, hour_end_str in enumerate(_HOUR_ENDS, start=1):
            hour_end_h, hour_end_m = map(int, hour_end_str.split(":"))
            cutoff = day_data.index[0].normalize().replace(
                hour=hour_end_h, minute=hour_end_m,
            )

            # All candles up to and including the cutoff.
            mask = day_data.index <= cutoff
            if mask.any():
                cum_range = running_range.loc[mask].iloc[-1]
                row[f"dvr_consumed_h{i}"] = cum_range / dvr_val
            else:
                row[f"dvr_consumed_h{i}"] = np.nan

        records.append(row)

    result = pd.DataFrame(records).set_index("date")

    logger.info(
        "Built DVR consumed — %d days [%s to %s]",
        len(result),
        result.index.min().date() if len(result) > 0 else "N/A",
        result.index.max().date() if len(result) > 0 else "N/A",
    )
    return result
