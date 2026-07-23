"""
Calendar Feature Module
=======================

Adds calendar context to each trading day: expiry flags,
event flags, day-of-week, and no-trade logic.

Handles the NIFTY weekly expiry transition from Thursday
(before 2025-09-02) to Tuesday (on/after 2025-09-02).

No hardcoded values — all parameters from function arguments.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def build_calendar_features(
    trading_dates: pd.DatetimeIndex,
    event_calendar_path: Path,
    holiday_path: Path,
    expiry_change_date: str,
) -> pd.DataFrame:
    """Build calendar features for the given trading dates.

    Parameters
    ----------
    trading_dates : pd.DatetimeIndex
        All trading dates (tz-naive) to generate features for.
    event_calendar_path : Path
        Path to ``event_calendar.csv`` with columns
        ``date, event_type, description, impact_level``.
    holiday_path : Path
        Path to ``market_holidays.csv`` (used for pre/post
        event adjacency — next/previous *trading* day).
    expiry_change_date : str
        Date string (YYYY-MM-DD) when NIFTY expiry moved
        from Thursday to Tuesday.  E.g. ``"2025-09-02"``.

    Returns
    -------
    pd.DataFrame
        One row per trading day, indexed by date.
    """
    dates = trading_dates.copy()
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    dates = dates.unique().sort_values()

    expiry_cutoff = pd.Timestamp(expiry_change_date)

    result = pd.DataFrame(index=dates)
    result.index.name = "date"

    # ── Day / week / month ──────────────────────────────────────────
    result["day_of_week"] = result.index.day_name()
    result["week_of_month"] = (result.index.day - 1) // 7 + 1
    result["month"] = result.index.month

    # ── Expiry day logic ────────────────────────────────────────────
    # Before expiry_change_date: Thursday is expiry.
    # On/after expiry_change_date: Tuesday is expiry.
    result["is_expiry_day"] = False

    before_mask = result.index < expiry_cutoff
    result.loc[before_mask & (result["day_of_week"] == "Thursday"), "is_expiry_day"] = True

    after_mask = result.index >= expiry_cutoff
    result.loc[after_mask & (result["day_of_week"] == "Tuesday"), "is_expiry_day"] = True

    # No-trade day = expiry day.
    result["is_no_trade_day"] = result["is_expiry_day"]

    # ── Event flags ─────────────────────────────────────────────────
    events = _load_events(event_calendar_path)

    result["is_event_day"] = result.index.isin(events.index)
    result["event_type"] = ""
    result["event_impact"] = ""

    if not events.empty:
        overlap = result.index.intersection(events.index)
        if len(overlap) > 0:
            result.loc[overlap, "event_type"] = events.loc[overlap, "event_type"].values
            result.loc[overlap, "event_impact"] = events.loc[overlap, "impact_level"].values

    # ── Pre-event / post-event flags ────────────────────────────────
    # Uses actual trading dates (not calendar days) for adjacency.
    result["is_pre_event"] = False
    result["is_post_event"] = False

    event_dates_set = set(events.index)
    date_list = list(dates)

    for i, dt in enumerate(date_list):
        # Pre-event: next trading day is an event day.
        if i + 1 < len(date_list) and date_list[i + 1] in event_dates_set:
            result.loc[dt, "is_pre_event"] = True
        # Post-event: previous trading day was an event day.
        if i - 1 >= 0 and date_list[i - 1] in event_dates_set:
            result.loc[dt, "is_post_event"] = True

    logger.info(
        "Built calendar features — %d rows, %d event days, "
        "expiry transition at %s",
        len(result),
        result["is_event_day"].sum(),
        expiry_change_date,
    )
    return result


def _load_events(path: Path) -> pd.DataFrame:
    """Load event calendar CSV.  Returns empty DataFrame if missing.

    Parameters
    ----------
    path : Path
        Path to CSV with columns ``date, event_type, description, impact_level``.

    Returns
    -------
    pd.DataFrame
        Indexed by date (tz-naive).  Empty if file missing or empty.
    """
    if not path.exists():
        logger.warning(
            "Event calendar not found at %s — "
            "event flags will all be False.",
            path,
        )
        return pd.DataFrame(
            columns=["event_type", "description", "impact_level"],
        )

    df = pd.read_csv(path, comment="#")

    if df.empty or "date" not in df.columns:
        return pd.DataFrame(
            columns=["event_type", "description", "impact_level"],
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index = df.index.normalize()

    # Fill missing values with empty strings.
    for col in ["event_type", "description", "impact_level"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df
