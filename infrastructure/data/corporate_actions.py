"""
Corporate Actions Module
========================

Maintains a registry of corporate actions (demergers, splits, bonuses,
special dividends) that create discontinuities in the raw OHLCV price
series, and provides helpers so strategies can *exclude* or *flag* the
affected ex-dates during backtesting.

Why this exists
---------------
The raw vendor (Zerodha) series is adjusted for ordinary splits and
bonuses, but **demergers are not cleanly adjusted** — they leave a large
overnight gap on the ex-date.  A strategy that computes overnight or
daily returns across such a bar sees a fake ±30-85% move.  Excluding the
ex-date bar removes that artifact without touching legitimate data.

The registry lives in ``data/events/corporate_actions.csv`` and is the
single source of truth.  Add new rows there as corporate actions occur.

Only verified, discontinuity-causing actions belong here.  Genuine market
moves (circuit days, news gaps) are NOT corporate actions and must stay
in the data.

All paths are resolved dynamically; no print statements — logging only.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# data/events/corporate_actions.csv, resolved relative to repo root.
#   this file : trading_lab/infrastructure/data/corporate_actions.py
#   parents[2]: trading_lab/
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
CORP_ACTIONS_PATH: Path = _REPO_ROOT / "data" / "events" / "corporate_actions.csv"


def load_corporate_actions(symbol: Optional[str] = None) -> pd.DataFrame:
    """Load the corporate-actions registry.

    Parameters
    ----------
    symbol : str, optional
        If given, return only the rows for this symbol.

    Returns
    -------
    pd.DataFrame
        Columns: ``symbol``, ``ex_date`` (datetime64), ``action_type``,
        ``ratio``, ``description``, ``verified``, ``source``.
        Returns an empty (correctly-typed) frame if the registry file is
        missing.
    """
    cols = ["symbol", "ex_date", "action_type", "ratio",
            "description", "verified", "source"]

    if not CORP_ACTIONS_PATH.exists():
        logger.warning("Corporate-actions registry not found: %s",
                       CORP_ACTIONS_PATH)
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(CORP_ACTIONS_PATH)
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    df["symbol"] = df["symbol"].str.upper()

    if symbol is not None:
        df = df[df["symbol"] == symbol.upper()]

    return df.reset_index(drop=True)


def get_corp_action_dates(symbol: str) -> set[date]:
    """Return the set of corporate-action ex-dates for a symbol.

    Parameters
    ----------
    symbol : str
        Instrument symbol, e.g. ``"NMDC"``.

    Returns
    -------
    set[datetime.date]
        Calendar dates (no time component) of corporate actions.
    """
    df = load_corporate_actions(symbol)
    return set(df["ex_date"].dt.date)


def apply_corporate_actions(
    df: pd.DataFrame,
    symbol: str,
    mode: str = "exclude",
) -> pd.DataFrame:
    """Exclude or flag corporate-action ex-dates in a price DataFrame.

    Works for both daily and intraday frames: matching is done on the
    calendar date of the (datetime) index, so an entire intraday ex-date
    session is handled together.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` (as returned by ``loader.load``).
    symbol : str
        Instrument symbol used to look up its corporate actions.
    mode : str
        - ``"exclude"`` : drop all rows that fall on a corporate-action
          ex-date (default).  Best for **intraday / per-session**
          strategies (e.g. VCF): the contaminated session is removed
          entirely.  Note: for strategies that compute *multi-day*
          returns, dropping the single ex-date bar still leaves a return
          that bridges the gap (prev-day close → next-day close); use
          ``"flag"`` mode for those (recipe below).
        - ``"flag"`` : keep all rows but add a boolean column
          ``is_corp_action`` marking the ex-date rows, leaving it to the
          strategy to decide.  Recommended for **return-based** strategies,
          which should invalidate the returns both *into* and *out of* the
          ex-date::

              df = load(sym, "stock", "daily", corporate_actions="flag")
              ret = df["close"].pct_change()
              bad = df["is_corp_action"] | df["is_corp_action"].shift(1, fill_value=False)
              ret = ret.mask(bad)   # NaN across the discontinuity

          (Back-adjusting the raw series is intentionally *not* done
          automatically: these demergers are not cleanly adjusted by the
          vendor, so a reliable adjustment factor cannot be inferred.)

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame.  Returned unchanged (aside from a
        possible all-False ``is_corp_action`` column in flag mode) when
        the symbol has no registered corporate actions.

    Raises
    ------
    ValueError
        If *mode* is not ``"exclude"`` or ``"flag"``, or if the index is
        not a DatetimeIndex.
    """
    if mode not in ("exclude", "flag"):
        raise ValueError(f"mode must be 'exclude' or 'flag', got {mode!r}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "apply_corporate_actions expects a DatetimeIndex "
            "(use loader.load, which sets one)."
        )

    action_dates = get_corp_action_dates(symbol)

    # Match on the calendar date of each (tz-aware) index entry. This
    # avoids tz-naive vs tz-aware comparison pitfalls and handles intraday
    # frames (a whole ex-date session matches together).
    is_action = pd.Series(df.index.date, index=df.index).isin(action_dates)

    if mode == "flag":
        df = df.copy()
        df["is_corp_action"] = is_action.values
        return df

    # mode == "exclude"
    n_dropped = int(is_action.sum())
    if n_dropped:
        logger.info(
            "Excluded %d corporate-action rows for %s (%s)",
            n_dropped, symbol, sorted(action_dates),
        )
    return df.loc[~is_action.values]
