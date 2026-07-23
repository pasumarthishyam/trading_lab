"""
Swing Feature Module
====================

Detects the largest directional swing per trading session
from 1-minute NIFTY data and computes swing metrics.

Runs independently for each reversal threshold (e.g. 20, 30, 40
points).  Produces one row per trading day with swing metrics
per threshold plus capture-zone flags.

No hardcoded values — all parameters from function arguments.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class _Swing:
    """Represents a single detected directional swing."""

    direction: str  # "up" or "down"
    magnitude: float  # absolute points
    start_time: pd.Timestamp
    peak_time: pd.Timestamp
    start_price: float
    peak_price: float


def build_swing_features(
    nifty_1min: pd.DataFrame,
    reversal_thresholds: list[int],
    capture_zone_min: int,
    capture_zone_max: int,
    swing_reversal_default: int,
) -> pd.DataFrame:
    """Build swing features for each trading day.

    Parameters
    ----------
    nifty_1min : pd.DataFrame
        NIFTY 1-minute OHLCV with DatetimeIndex.
    reversal_thresholds : list[int]
        Point thresholds at which a reversal is detected
        (e.g. [20, 30, 40]).
    capture_zone_min : int
        Minimum capture zone target in points (e.g. 100).
    capture_zone_max : int
        Maximum capture zone target in points (e.g. 150).
    swing_reversal_default : int
        Default threshold used for capture zone flags (e.g. 30).

    Returns
    -------
    pd.DataFrame
        One row per trading day, indexed by date (tz-naive).
        Columns: ``swing_{t}_direction``, ``swing_{t}_magnitude``,
        ``swing_{t}_start_time``, ``swing_{t}_peak_time``,
        ``swing_{t}_retracement``, ``swing_{t}_false_starts``
        for each threshold ``t``, plus ``capture_zone_100``
        and ``capture_zone_150``.
    """
    df = nifty_1min.copy()

    # Strip timezone for consistent grouping.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    records: list[dict] = []

    for trade_date, day_data in df.groupby(df.index.date):
        day_data = day_data.sort_index()
        date_key = pd.Timestamp(trade_date)
        row: dict = {"date": date_key}

        for threshold in reversal_thresholds:
            swings = _detect_swings(day_data, threshold)
            largest = _select_largest(swings)
            false_starts = _count_false_starts(
                swings, largest, capture_zone_min,
            )
            retracement = _compute_retracement(
                day_data, largest,
            )

            t = str(threshold)
            if largest is not None:
                row[f"swing_{t}_direction"] = largest.direction
                row[f"swing_{t}_magnitude"] = largest.magnitude
                row[f"swing_{t}_start_time"] = largest.start_time
                row[f"swing_{t}_peak_time"] = largest.peak_time
                row[f"swing_{t}_retracement"] = retracement
                row[f"swing_{t}_false_starts"] = false_starts
            else:
                row[f"swing_{t}_direction"] = np.nan
                row[f"swing_{t}_magnitude"] = np.nan
                row[f"swing_{t}_start_time"] = pd.NaT
                row[f"swing_{t}_peak_time"] = pd.NaT
                row[f"swing_{t}_retracement"] = np.nan
                row[f"swing_{t}_false_starts"] = 0

        # Capture zone flags use the default threshold.
        default_t = str(swing_reversal_default)
        mag_col = f"swing_{default_t}_magnitude"
        mag_val = row.get(mag_col, np.nan)

        if pd.notna(mag_val):
            row["capture_zone_100"] = mag_val >= capture_zone_min
            row["capture_zone_150"] = mag_val >= capture_zone_max
        else:
            row["capture_zone_100"] = False
            row["capture_zone_150"] = False

        records.append(row)

    result = pd.DataFrame(records).set_index("date")

    logger.info(
        "Built swing features — %d days [%s to %s]",
        len(result),
        result.index.min().date() if len(result) > 0 else "N/A",
        result.index.max().date() if len(result) > 0 else "N/A",
    )
    return result


# ── Private helpers ─────────────────────────────────────────────────


def _detect_swings(
    day_data: pd.DataFrame,
    reversal_threshold: int,
) -> list[_Swing]:
    """Detect all swings in a single trading session.

    Algorithm
    ---------
    1. Start at session open.  Set swing_start = open price.
    2. Determine initial direction from first non-zero move.
    3. Track running_high and running_low from swing start.
    4. A swing ends when price moves against current direction
       by more than reversal_threshold points:
       - Up-swing reversal: running_high - current_low >= threshold
       - Down-swing reversal: current_high - running_low >= threshold
    5. Record completed swing, reset, flip direction.
    6. Repeat until session end.  Final unclosed swing is recorded.
    """
    if len(day_data) < 2:
        return []

    closes = day_data["close"].values
    highs = day_data["high"].values
    lows = day_data["low"].values
    timestamps = day_data.index

    # Determine initial direction from first meaningful move.
    direction = _initial_direction(closes)
    if direction is None:
        return []

    swings: list[_Swing] = []

    swing_start_idx = 0
    swing_start_price = closes[0]
    running_high = highs[0]
    running_low = lows[0]
    peak_idx = 0  # index of running extreme

    for i in range(1, len(closes)):
        # Update running extremes.
        if highs[i] > running_high:
            running_high = highs[i]
            if direction == "up":
                peak_idx = i

        if lows[i] < running_low:
            running_low = lows[i]
            if direction == "down":
                peak_idx = i

        # Check for reversal.
        reversed_swing = False

        if direction == "up":
            if running_high - lows[i] >= reversal_threshold:
                # Up-swing completed.  Record it.
                magnitude = running_high - swing_start_price
                swings.append(_Swing(
                    direction="up",
                    magnitude=abs(magnitude),
                    start_time=timestamps[swing_start_idx],
                    peak_time=timestamps[peak_idx],
                    start_price=swing_start_price,
                    peak_price=running_high,
                ))
                reversed_swing = True
                direction = "down"
                swing_start_idx = peak_idx
                swing_start_price = running_high
                running_high = highs[i]
                running_low = lows[i]
                peak_idx = i

        elif direction == "down":
            if highs[i] - running_low >= reversal_threshold:
                # Down-swing completed.  Record it.
                magnitude = swing_start_price - running_low
                swings.append(_Swing(
                    direction="down",
                    magnitude=abs(magnitude),
                    start_time=timestamps[swing_start_idx],
                    peak_time=timestamps[peak_idx],
                    start_price=swing_start_price,
                    peak_price=running_low,
                ))
                reversed_swing = True
                direction = "up"
                swing_start_idx = peak_idx
                swing_start_price = running_low
                running_high = highs[i]
                running_low = lows[i]
                peak_idx = i

    # Record the final unclosed swing.
    if direction == "up":
        magnitude = running_high - swing_start_price
        swings.append(_Swing(
            direction="up",
            magnitude=abs(magnitude),
            start_time=timestamps[swing_start_idx],
            peak_time=timestamps[peak_idx],
            start_price=swing_start_price,
            peak_price=running_high,
        ))
    elif direction == "down":
        magnitude = swing_start_price - running_low
        swings.append(_Swing(
            direction="down",
            magnitude=abs(magnitude),
            start_time=timestamps[swing_start_idx],
            peak_time=timestamps[peak_idx],
            start_price=swing_start_price,
            peak_price=running_low,
        ))

    return swings


def _initial_direction(closes: np.ndarray) -> str | None:
    """Determine initial swing direction from first meaningful move."""
    for i in range(1, len(closes)):
        if closes[i] > closes[0]:
            return "up"
        if closes[i] < closes[0]:
            return "down"
    return None


def _select_largest(swings: list[_Swing]) -> _Swing | None:
    """Select the swing with greatest magnitude."""
    if not swings:
        return None
    return max(swings, key=lambda s: s.magnitude)


def _count_false_starts(
    swings: list[_Swing],
    largest: _Swing | None,
    capture_zone_min: int,
) -> int:
    """Count false starts before the largest swing.

    A false start is a swing that:
    1. Reversed before reaching capture_zone_min (100 points), AND
    2. Was followed by a larger swing in the opposite direction.
    """
    if largest is None or len(swings) <= 1:
        return 0

    largest_idx = swings.index(largest)
    false_starts = 0

    for i, swing in enumerate(swings):
        if i >= largest_idx:
            break
        if swing.magnitude < capture_zone_min:
            # Check if any subsequent swing in opposite direction is larger.
            for j in range(i + 1, len(swings)):
                if (swings[j].direction != swing.direction
                        and swings[j].magnitude > swing.magnitude):
                    false_starts += 1
                    break

    return false_starts


def _compute_retracement(
    day_data: pd.DataFrame,
    largest: _Swing | None,
) -> float:
    """Compute max retracement from largest swing's peak to session end.

    Retracement = how far price moved against the swing direction
    from the peak to the lowest/highest subsequent point before
    session end.
    """
    if largest is None:
        return np.nan

    # Data after the peak.
    after_peak = day_data.loc[largest.peak_time:]
    if len(after_peak) <= 1:
        return 0.0

    if largest.direction == "up":
        # Retracement = peak - lowest low after peak.
        lowest_after = after_peak["low"].min()
        return largest.peak_price - lowest_after
    else:
        # Retracement = highest high after peak - peak.
        highest_after = after_peak["high"].max()
        return highest_after - largest.peak_price
