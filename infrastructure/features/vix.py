"""
VIX Feature Module
==================

Classifies each trading day into a VIX regime and computes
directional features from daily INDIAVIX data.

No hardcoded values — all band boundaries and thresholds
from function arguments.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_vix_features(
    vix_daily: pd.DataFrame,
    vix_bands: dict[str, tuple[float, float]],
    downgrade_threshold: float,
) -> pd.DataFrame:
    """Build VIX regime and directional features.

    Parameters
    ----------
    vix_daily : pd.DataFrame
        INDIAVIX daily OHLCV with DatetimeIndex.
    vix_bands : dict
        Mapping of regime name → (lower, upper) using ``[lower, upper)``
        convention.  Example: ``{"golden": (13, 18)}``.
    downgrade_threshold : float
        Fractional increase in VIX (e.g. 0.05 for 5%) that triggers
        the downgrade flag.

    Returns
    -------
    pd.DataFrame
        One row per trading day, indexed by date (tz-naive).
    """
    df = vix_daily.copy()

    # Normalise index to tz-naive dates.
    df.index = df.index.normalize()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    result = pd.DataFrame(index=df.index)
    result.index.name = "date"

    result["vix_open"] = df["open"].values
    result["vix_close"] = df["close"].values
    result["vix_prev_close"] = result["vix_close"].shift(1)

    # Direction: rising or falling relative to previous close.
    result["vix_direction"] = np.where(
        result["vix_close"] > result["vix_prev_close"],
        "rising",
        "falling",
    )

    # Percentage change.
    result["vix_change_pct"] = (
        (result["vix_close"] - result["vix_prev_close"])
        / result["vix_prev_close"]
    )

    # Downgrade flag: VIX rose more than threshold from previous close.
    result["vix_downgrade"] = result["vix_change_pct"] > downgrade_threshold

    # Regime classification using [lower, upper) convention.
    result["vix_regime"] = _classify_regime(result["vix_close"], vix_bands)

    logger.info(
        "Built VIX features — %d rows [%s to %s]",
        len(result),
        result.index.min().date(),
        result.index.max().date(),
    )
    return result


def _classify_regime(
    vix_close: pd.Series,
    bands: dict[str, tuple[float, float]],
) -> pd.Series:
    """Assign VIX regime label per [lower, upper) convention.

    Parameters
    ----------
    vix_close : pd.Series
        VIX closing values.
    bands : dict
        Regime name → (lower_bound, upper_bound).

    Returns
    -------
    pd.Series
        Regime label for each row.
    """
    regimes = pd.Series("unknown", index=vix_close.index, dtype="object")

    for regime_name, (lower, upper) in bands.items():
        mask = (vix_close >= lower) & (vix_close < upper)
        regimes.loc[mask] = regime_name

    unknown_count = (regimes == "unknown").sum()
    if unknown_count > 0:
        logger.warning(
            "%d rows could not be classified into any VIX regime",
            unknown_count,
        )

    return regimes
