"""
Supervised Feature Builder
==========================

Builds the ``(symbol, date)`` feature panel for the significant-move
classifier.  Five feature families are assembled and joined:

============  ==================================================
Family        Source
============  ==================================================
daily         ``data/raw/stocks/*/daily.parquet`` (one DuckDB pass)
swing         fractal pivot geometry over the daily bars
intraday      ``data/raw/stocks/*/15min.parquet`` (one DuckDB pass)
market        NIFTY + INDIAVIX daily, broadcast to every symbol
calendar      derived from the date index
============  ==================================================

**Point-in-time discipline.**  Every column is computed so that its value
on date ``t`` depends only on information observable at or before the
close of ``t``.  Three rules enforce this:

1. Trailing statistics use ``shift(1)`` before rolling, so the window
   that a value is scored against never contains that value.
2. Fractal pivots are *centred* by construction — a pivot at bar ``i``
   cannot be recognised until bar ``i + k``.  Pivot columns are therefore
   shifted forward by ``k`` bars so the panel only ever sees confirmed
   pivots.
3. Market-context columns are joined on the same date ``t`` and use only
   that day's close, never a forward value.

Logging only — no ``print``.
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
from infrastructure.ml.classifier import config as C

logger = logging.getLogger(__name__)


# ── universe ────────────────────────────────────────────────────────

def load_universe() -> list[str]:
    """Return the F&O equity universe from the on-disk manifest."""
    manifest = json.loads(C.UNIVERSE_MANIFEST.read_text(encoding="utf-8"))
    symbols = sorted(set(manifest["symbols"]))
    logger.info("Loaded F&O universe — %d symbols", len(symbols))
    return symbols


def _existing_paths(universe: list[str], timeframe: str) -> list[str]:
    """POSIX paths of the parquet files that actually exist on disk."""
    root = C.STOCKS_ROOT.as_posix()
    paths = [f"{root}/{sym}/{timeframe}.parquet" for sym in universe]
    existing = [p for p in paths if Path(p).exists()]
    if not existing:
        raise FileNotFoundError(
            f"No {timeframe}.parquet files found under {C.STOCKS_ROOT}"
        )
    if len(existing) < len(paths):
        logger.warning(
            "%d/%d symbols have no %s.parquet — skipped",
            len(paths) - len(existing), len(paths), timeframe,
        )
    return existing


def _sql_path_list(paths: list[str]) -> str:
    """Render paths as a bracketed SQL list literal."""
    return "[" + ", ".join("'" + p.replace("'", "''") + "'" for p in paths) + "]"


# ── rolling helpers (point-in-time) ─────────────────────────────────

def _min_periods(window: int) -> int:
    """Warm-up requirement, never larger than the window itself."""
    return max(2, min(C.ROLLING_MIN_PERIODS, window))


def _trailing_mean(s: pd.Series, window: int) -> pd.Series:
    """Mean of the *window* bars ending at ``t-1`` (excludes today)."""
    return s.shift(1).rolling(window, min_periods=_min_periods(window)).mean()


def _trailing_zscore(s: pd.Series, window: int) -> pd.Series:
    """Z-score of today's value against the trailing window before it."""
    shifted = s.shift(1)
    mp = _min_periods(window)
    mean = shifted.rolling(window, min_periods=mp).mean()
    std = shifted.rolling(window, min_periods=mp).std()
    return (s - mean) / std


def _rolling_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation, computed from rolling moments.

    Expressed via rolling sums rather than ``Series.rolling().corr()`` so
    it can run inside a ``groupby.transform`` without a Python-level loop
    over the 200-odd symbols.
    """
    mp = _min_periods(window)
    ex = x.rolling(window, min_periods=mp).mean()
    ey = y.rolling(window, min_periods=mp).mean()
    exy = (x * y).rolling(window, min_periods=mp).mean()
    sx = x.rolling(window, min_periods=mp).std()
    sy = y.rolling(window, min_periods=mp).std()
    cov = exy - ex * ey
    return cov / (sx * sy)


# ── daily OHLCV scan ────────────────────────────────────────────────

def scan_daily_ohlcv(universe: list[str]) -> pd.DataFrame:
    """One DuckDB pass over every ``daily.parquet`` → raw OHLCV panel.

    The heavy multi-file read stays inside DuckDB; only the compact
    result set is materialised into Pandas.
    """
    source = _sql_path_list(_existing_paths(universe, "daily"))
    sql = f"""
    SELECT
        regexp_extract(filename, 'stocks/([^/]+)/daily', 1)  AS symbol,
        CAST(date AT TIME ZONE '{C.MARKET_TZ}' AS DATE)      AS date,
        open, high, low, close, volume
    FROM read_parquet({source}, filename=true)
    WHERE close > 0 AND open > 0 AND high > 0 AND low > 0
    ORDER BY symbol, date
    """
    con = duckdb.connect()
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()

    df["date"] = pd.to_datetime(df["date"])
    logger.info(
        "Daily scan — %d rows across %d symbols [%s .. %s]",
        len(df), df["symbol"].nunique(),
        df["date"].min().date(), df["date"].max().date(),
    )
    return df


# ── daily feature family ────────────────────────────────────────────

def add_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return / volatility / volume / trend / candle-shape features."""
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("symbol", sort=False)

    open_, high, low, close, volume = (
        df["open"], df["high"], df["low"], df["close"], df["volume"]
    )
    prev_close = g["close"].shift(1)
    df["prev_close"] = prev_close

    # ── returns ─────────────────────────────────────────────────────
    for w in C.RETURN_WINDOWS:
        df[f"ret_{w}d"] = g["close"].transform(lambda s, w=w: s.pct_change(w))
    df["abs_ret_1d"] = df["ret_1d"].abs()
    df["gap_pct"] = open_ / prev_close - 1.0
    df["abs_gap_pct"] = df["gap_pct"].abs()
    df["overnight_vs_intraday"] = df["gap_pct"] - (close / open_ - 1.0)

    # ── candle shape ────────────────────────────────────────────────
    rng = (high - low)
    safe_rng = rng.replace(0.0, np.nan)
    df["range_pct"] = rng / close
    df["body_pct"] = (close - open_).abs() / close
    df["upper_wick_pct"] = (high - np.maximum(open_, close)) / close
    df["lower_wick_pct"] = (np.minimum(open_, close) - low) / close
    df["body_to_range"] = (close - open_).abs() / safe_rng
    df["close_position_in_range"] = (close - low) / safe_rng
    df["signed_body"] = (close - open_) / close

    # ── volatility ──────────────────────────────────────────────────
    true_range = pd.concat(
        [rng, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    df["_true_range"] = true_range
    df["atr_pct"] = (
        df.groupby("symbol", sort=False)["_true_range"]
        .transform(lambda s: s.rolling(C.ATR_WINDOW,
                                       min_periods=C.ROLLING_MIN_PERIODS).mean())
        / close
    )
    g2 = df.groupby("symbol", sort=False)
    df[f"rvol_{C.VOL_SHORT_WINDOW}d"] = g2["ret_1d"].transform(
        lambda s: s.rolling(C.VOL_SHORT_WINDOW, min_periods=3).std()
    )
    df[f"rvol_{C.VOL_LONG_WINDOW}d"] = g2["ret_1d"].transform(
        lambda s: s.rolling(C.VOL_LONG_WINDOW,
                            min_periods=C.ROLLING_MIN_PERIODS).std()
    )
    df["vol_ratio_short_long"] = (
        df[f"rvol_{C.VOL_SHORT_WINDOW}d"] / df[f"rvol_{C.VOL_LONG_WINDOW}d"]
    )
    # Parkinson estimator — uses the high/low range, so it reacts to
    # intraday excursions that close-to-close vol misses entirely.
    log_hl_sq = np.log(high / low) ** 2
    df["_log_hl_sq"] = log_hl_sq
    df["parkinson_vol_20d"] = np.sqrt(
        df.groupby("symbol", sort=False)["_log_hl_sq"].transform(
            lambda s: s.rolling(C.VOL_LONG_WINDOW,
                                min_periods=C.ROLLING_MIN_PERIODS).mean()
        ) / (4.0 * np.log(2.0))
    )
    g3 = df.groupby("symbol", sort=False)
    df["vol_of_vol_20d"] = g3[f"rvol_{C.VOL_SHORT_WINDOW}d"].transform(
        lambda s: s.rolling(C.VOL_LONG_WINDOW,
                            min_periods=C.ROLLING_MIN_PERIODS).std()
    )
    df["range_zscore_20d"] = g3["range_pct"].transform(
        lambda s: _trailing_zscore(s, C.VOL_LONG_WINDOW)
    )
    df["abs_ret_zscore_20d"] = g3["abs_ret_1d"].transform(
        lambda s: _trailing_zscore(s, C.VOL_LONG_WINDOW)
    )

    # Volatility regime: where does today's trailing vol sit inside the
    # symbol's *own* history?  Expanding rank, so it is point-in-time by
    # construction and comparable across symbols.
    df["vol_regime"] = g3[f"rvol_{C.VOL_LONG_WINDOW}d"].transform(
        lambda s: s.shift(1).expanding(min_periods=C.MIN_HISTORY_BARS).rank(pct=True)
    )
    df["vol_regime_bucket"] = np.digitize(
        df["vol_regime"].fillna(0.5), list(C.VOL_REGIME_QUANTILES)
    ).astype(float)
    df["vol_expansion"] = df[f"rvol_{C.VOL_SHORT_WINDOW}d"] / g3[
        f"rvol_{C.VOL_SHORT_WINDOW}d"
    ].transform(lambda s: s.shift(1).rolling(C.VOL_LONG_WINDOW,
                                             min_periods=C.ROLLING_MIN_PERIODS).mean())

    # ── volume ──────────────────────────────────────────────────────
    g4 = df.groupby("symbol", sort=False)
    for w in C.VOLUME_WINDOWS:
        df[f"volume_ratio_{w}d"] = volume / g4["volume"].transform(
            lambda s, w=w: _trailing_mean(s, w)
        )
    df["volume_zscore_20d"] = g4["volume"].transform(
        lambda s: _trailing_zscore(s, C.VOL_LONG_WINDOW)
    )
    df["volume_trend_5_20"] = (
        df[f"volume_ratio_{C.VOLUME_WINDOWS[0]}d"]
        / df[f"volume_ratio_{C.VOLUME_WINDOWS[-1]}d"]
    )
    turnover = close * volume
    df["_turnover"] = turnover
    df["turnover_ratio_20d"] = turnover / df.groupby("symbol", sort=False)[
        "_turnover"
    ].transform(lambda s: _trailing_mean(s, C.VOL_LONG_WINDOW))
    g5 = df.groupby("symbol", sort=False)
    df["_vol_chg"] = g5["volume"].transform(lambda s: s.pct_change())
    df["volume_return_corr_20d"] = (
        df.groupby("symbol", sort=False)
        .apply(
            lambda d: _rolling_corr(d["_vol_chg"], d["abs_ret_1d"], C.VOL_LONG_WINDOW),
            include_groups=False,
        )
        .reset_index(level=0, drop=True)
        .sort_index()
    )

    # ── trend / position ────────────────────────────────────────────
    g6 = df.groupby("symbol", sort=False)
    for w in C.SMA_WINDOWS:
        sma = g6["close"].transform(
            lambda s, w=w: s.rolling(w, min_periods=C.ROLLING_MIN_PERIODS).mean()
        )
        df[f"dist_sma{w}"] = close / sma - 1.0
        if w == C.SMA_WINDOWS[0]:
            df[f"sma{w}_slope_5d"] = sma / sma.groupby(df["symbol"]).shift(5) - 1.0
    g7 = df.groupby("symbol", sort=False)
    df["close_to_high_20d"] = close / g7["high"].transform(
        lambda s: s.rolling(C.VOL_LONG_WINDOW,
                            min_periods=C.ROLLING_MIN_PERIODS).max()
    ) - 1.0
    df["close_to_low_20d"] = close / g7["low"].transform(
        lambda s: s.rolling(C.VOL_LONG_WINDOW,
                            min_periods=C.ROLLING_MIN_PERIODS).min()
    ) - 1.0
    df["_up_day"] = (df["ret_1d"] > 0).astype(float)
    df["pct_up_days_20d"] = df.groupby("symbol", sort=False)["_up_day"].transform(
        lambda s: s.shift(1).rolling(C.VOL_LONG_WINDOW,
                                     min_periods=C.ROLLING_MIN_PERIODS).mean()
    )
    df["consec_direction"] = df.groupby("symbol", sort=False)["ret_1d"].transform(
        lambda s: np.sign(s).rolling(3, min_periods=3).sum()
    )

    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    logger.info("Daily feature family built — %d columns", df.shape[1])
    return df


# ── swing geometry family ───────────────────────────────────────────

def _confirmed_pivots(
    high: pd.Series, low: pd.Series, k: int
) -> tuple[pd.Series, pd.Series]:
    """Fractal pivot prices, shifted so only *confirmed* pivots are seen.

    A bar is a pivot high when its high is the maximum of the ``2k+1``
    window centred on it.  That fact is not knowable until ``k`` bars
    later, so the result is shifted forward by ``k``: the value at index
    ``t`` describes a pivot that sits at ``t - k`` and became visible
    exactly now.
    """
    win = 2 * k + 1
    roll_max = high.rolling(win, center=True, min_periods=win).max()
    roll_min = low.rolling(win, center=True, min_periods=win).min()
    ph = high.where(high >= roll_max)
    pl = low.where(low <= roll_min)
    return ph.shift(k), pl.shift(k)


def _swing_block(d: pd.DataFrame) -> pd.DataFrame:
    """Swing-geometry columns for one symbol's chronological bars."""
    k = C.SWING_FRACTAL_K
    high, low, close = d["high"], d["low"], d["close"]
    ph_at, pl_at = _confirmed_pivots(high, low, k)

    last_ph = ph_at.ffill()
    last_pl = pl_at.ffill()

    pos = pd.Series(np.arange(len(d), dtype=float), index=d.index)
    ph_pos = pos.where(ph_at.notna()).ffill()
    pl_pos = pos.where(pl_at.notna()).ffill()

    out = pd.DataFrame(index=d.index)
    out["dist_from_pivot_high"] = close / last_ph - 1.0
    out["dist_from_pivot_low"] = close / last_pl - 1.0
    # Amplitude of the most recent completed high→low (or low→high) leg.
    out["swing_amplitude_pct"] = (last_ph - last_pl).abs() / close
    out["bars_since_pivot_high"] = pos - ph_pos
    out["bars_since_pivot_low"] = pos - pl_pos
    # Which pivot came last tells you the structural direction in force.
    out["last_pivot_was_high"] = (ph_pos > pl_pos).astype(float)
    # Where does price sit inside the last swing leg? 0 = at the low,
    # 1 = at the high.  Scale-free measure of retracement.
    span = (last_ph - last_pl).replace(0.0, np.nan)
    out["swing_retracement"] = (close - last_pl) / span

    w = C.SWING_STRUCTURE_WINDOW
    out["pivot_count_20d"] = (
        ph_at.notna().rolling(w, min_periods=1).sum()
        + pl_at.notna().rolling(w, min_periods=1).sum()
    )
    # Structure score: are confirmed pivot highs rising and lows rising?
    ph_series = ph_at.ffill()
    pl_series = pl_at.ffill()
    out["higher_highs_20d"] = (ph_series > ph_series.shift(w)).astype(float)
    out["higher_lows_20d"] = (pl_series > pl_series.shift(w)).astype(float)
    out["swing_speed"] = out["swing_amplitude_pct"] / (
        out["bars_since_pivot_high"] + out["bars_since_pivot_low"] + 1.0
    )
    return out


def add_swing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach fractal-pivot swing geometry, computed per symbol."""
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    blocks = df.groupby("symbol", sort=False)[["high", "low", "close"]].apply(
        _swing_block, include_groups=False
    )
    if isinstance(blocks.index, pd.MultiIndex):
        blocks = blocks.reset_index(level=0, drop=True)
    out = pd.concat([df, blocks.sort_index()], axis=1)
    logger.info("Swing feature family built — %d columns", blocks.shape[1])
    return out


# ── intraday family ─────────────────────────────────────────────────

def build_intraday_features(universe: list[str]) -> pd.DataFrame:
    """Aggregate 15-minute bars into per-``(symbol, date)`` session shape.

    All of it happens in a single DuckDB ``GROUP BY`` — roughly six
    million intraday bars collapse to one row per symbol-day without ever
    materialising the raw bars in Pandas.
    """
    source = _sql_path_list(_existing_paths(universe, C.INTRADAY_TIMEFRAME))
    sql = f"""
    WITH raw AS (
        SELECT
            regexp_extract(filename, 'stocks/([^/]+)/{C.INTRADAY_TIMEFRAME}', 1) AS symbol,
            (date AT TIME ZONE '{C.MARKET_TZ}') AS ts,
            open, high, low, close, volume
        FROM read_parquet({source}, filename=true)
        WHERE close > 0 AND open > 0
    ),
    bars AS (
        SELECT
            symbol,
            CAST(ts AS DATE) AS date,
            ts,
            CAST(ts AS TIME) AS tod,
            open, high, low, close, volume,
            LAG(close) OVER (PARTITION BY symbol, CAST(ts AS DATE) ORDER BY ts)
                AS prev_bar_close
        FROM raw
    )
    SELECT
        symbol,
        date,
        count(*)                                        AS intraday_bar_count,
        first(open ORDER BY ts)                         AS day_open,
        last(close ORDER BY ts)                         AS day_close,
        max(high)                                       AS day_high,
        min(low)                                        AS day_low,
        sum(((high + low + close) / 3.0) * volume)
            / NULLIF(sum(volume), 0)                    AS vwap,
        last(close ORDER BY ts) FILTER (WHERE tod <= TIME '{C.FIRST_HOUR_END}')
                                                        AS first_hour_close,
        first(open ORDER BY ts) FILTER (WHERE tod >= TIME '{C.LAST_HOUR_START}')
                                                        AS last_hour_open,
        sum(volume) FILTER (WHERE tod <= TIME '{C.FIRST_HOUR_END}')
                                                        AS first_hour_volume,
        sum(volume) FILTER (WHERE tod >= TIME '{C.LAST_HOUR_START}')
                                                        AS last_hour_volume,
        sum(volume)                                     AS total_volume,
        sum(abs(close - prev_bar_close))                AS path_length,
        stddev_samp((close - prev_bar_close) / NULLIF(prev_bar_close, 0))
                                                        AS intraday_bar_vol,
        max(high) FILTER (WHERE tod <= TIME '{C.FIRST_HOUR_END}')
                                                        AS first_hour_high,
        min(low)  FILTER (WHERE tod <= TIME '{C.FIRST_HOUR_END}')
                                                        AS first_hour_low
    FROM bars
    GROUP BY symbol, date
    ORDER BY symbol, date
    """
    con = duckdb.connect()
    try:
        raw = con.execute(sql).fetchdf()
    finally:
        con.close()

    raw["date"] = pd.to_datetime(raw["date"])
    day_open = raw["day_open"]

    out = pd.DataFrame({"symbol": raw["symbol"], "date": raw["date"]})
    out["intraday_bar_count"] = raw["intraday_bar_count"]
    out["first_hour_ret"] = raw["first_hour_close"] / day_open - 1.0
    out["last_hour_ret"] = raw["day_close"] / raw["last_hour_open"] - 1.0
    out["intraday_range_pct"] = (raw["day_high"] - raw["day_low"]) / day_open
    out["intraday_runup_pct"] = raw["day_high"] / day_open - 1.0
    out["intraday_drawdown_pct"] = raw["day_low"] / day_open - 1.0
    out["close_vs_vwap"] = raw["day_close"] / raw["vwap"] - 1.0
    # Efficiency ratio: net move divided by the distance actually
    # travelled.  Near 1 = a clean directional session; near 0 = chop.
    out["intraday_efficiency"] = (
        (raw["day_close"] - day_open).abs()
        / raw["path_length"].replace(0.0, np.nan)
    )
    out["intraday_bar_vol"] = raw["intraday_bar_vol"]
    total_vol = raw["total_volume"].replace(0.0, np.nan)
    out["first_hour_volume_share"] = raw["first_hour_volume"] / total_vol
    out["last_hour_volume_share"] = raw["last_hour_volume"] / total_vol
    out["first_hour_range_share"] = (
        (raw["first_hour_high"] - raw["first_hour_low"])
        / (raw["day_high"] - raw["day_low"]).replace(0.0, np.nan)
    )

    logger.info(
        "Intraday feature family built — %d symbol-days from %s bars",
        len(out), C.INTRADAY_TIMEFRAME,
    )
    return out


# ── market-context family ───────────────────────────────────────────

def build_market_features() -> pd.DataFrame:
    """NIFTY + INDIAVIX context, one row per market date."""
    nifty = load("NIFTY", "index", "daily")
    vix = load("INDIAVIX", "volatility", "daily")

    idx = pd.to_datetime(nifty.index).tz_localize(None).normalize()
    n = pd.DataFrame(index=idx)
    close = nifty["close"].to_numpy()
    n["nifty_close"] = close
    n["nifty_ret_1d"] = pd.Series(close, index=idx).pct_change()
    n["nifty_ret_5d"] = pd.Series(close, index=idx).pct_change(5)
    n["nifty_rvol_20d"] = n["nifty_ret_1d"].rolling(
        C.VOL_LONG_WINDOW, min_periods=C.ROLLING_MIN_PERIODS
    ).std()
    n["nifty_abs_ret_1d"] = n["nifty_ret_1d"].abs()
    n["nifty_range_pct"] = (
        (nifty["high"].to_numpy() - nifty["low"].to_numpy()) / close
    )

    vidx = pd.to_datetime(vix.index).tz_localize(None).normalize()
    v = pd.DataFrame(index=vidx)
    v["vix_level"] = vix["close"].to_numpy()
    v["vix_change_pct"] = v["vix_level"].pct_change()
    v["vix_zscore_20d"] = _trailing_zscore(v["vix_level"], C.VOL_LONG_WINDOW)
    v["vix_percentile"] = (
        v["vix_level"].shift(1).expanding(min_periods=C.MIN_HISTORY_BARS).rank(pct=True)
    )

    market = n.join(v, how="left")
    market.index.name = "date"
    market = market.reset_index()
    logger.info("Market feature family built — %d market dates", len(market))
    return market


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Beta, correlation and relative strength against NIFTY.

    Computed after the market join so each symbol's returns line up with
    the index returns on the same dates.
    """
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    def _block(d: pd.DataFrame) -> pd.DataFrame:
        r, m = d["ret_1d"], d["nifty_ret_1d"]
        mp = C.ROLLING_MIN_PERIODS
        w = C.BETA_WINDOW
        cov = (r * m).rolling(w, min_periods=mp).mean() - (
            r.rolling(w, min_periods=mp).mean() * m.rolling(w, min_periods=mp).mean()
        )
        var_m = m.rolling(w, min_periods=mp).var()
        res = pd.DataFrame(index=d.index)
        res["beta_60d"] = cov / var_m.replace(0.0, np.nan)
        res["corr_nifty_20d"] = _rolling_corr(r, m, C.VOL_LONG_WINDOW)
        res["rel_strength_20d"] = d["ret_20d"] - d["nifty_ret_1d"].rolling(
            C.VOL_LONG_WINDOW, min_periods=mp
        ).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
        res["idio_vol_ratio"] = (
            d[f"rvol_{C.VOL_LONG_WINDOW}d"] / d["nifty_rvol_20d"].replace(0.0, np.nan)
        )
        return res

    blocks = df.groupby("symbol", sort=False)[
        ["ret_1d", "ret_20d", f"rvol_{C.VOL_LONG_WINDOW}d",
         "nifty_ret_1d", "nifty_rvol_20d"]
    ].apply(_block, include_groups=False)
    if isinstance(blocks.index, pd.MultiIndex):
        blocks = blocks.reset_index(level=0, drop=True)
    out = pd.concat([df, blocks.sort_index()], axis=1)
    logger.info("Relative feature family built — %d columns", blocks.shape[1])
    return out


# ── calendar family ─────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week, month position and F&O expiry proximity."""
    dates = df["date"]
    df["day_of_week"] = dates.dt.dayofweek.astype(float)
    df["day_of_month"] = dates.dt.day.astype(float)
    df["month"] = dates.dt.month.astype(float)
    month_end = dates + pd.offsets.MonthEnd(0)
    df["days_to_month_end"] = (month_end - dates).dt.days.astype(float)

    # NSE monthly F&O expiry is the last Thursday of the month; the days
    # around it carry their own volatility signature.
    last_thu = month_end - pd.to_timedelta(
        (month_end.dt.dayofweek - 3) % 7, unit="D"
    )
    df["days_to_expiry"] = (last_thu - dates).dt.days.astype(float)
    df["is_expiry_week"] = (df["days_to_expiry"].between(0, 4)).astype(float)
    return df


# ── orchestration ───────────────────────────────────────────────────

def build_feature_panel(
    universe: Optional[list[str]] = None,
    include_intraday: bool = True,
) -> pd.DataFrame:
    """Assemble the full ``(symbol, date)`` feature panel.

    Parameters
    ----------
    universe : list[str], optional
        Symbols to include.  Defaults to the full F&O manifest.
    include_intraday : bool, optional
        Join the 15-minute session-shape family.  Disabling it skips the
        heaviest scan, for quick iteration on the daily features.

    Returns
    -------
    pd.DataFrame
        ``symbol``, ``date``, raw OHLCV, and every feature column.  NaNs
        are *retained* here — imputation is a modelling decision made
        downstream, and the tree models handle missing values natively.
    """
    if universe is None:
        universe = load_universe()

    panel = scan_daily_ohlcv(universe)
    panel = add_daily_features(panel)
    panel = add_swing_features(panel)

    if include_intraday:
        intra = build_intraday_features(universe)
        before = len(panel)
        panel = panel.merge(intra, on=["symbol", "date"], how="left")
        matched = panel["intraday_bar_count"].notna().sum()
        logger.info(
            "Intraday join — %d/%d daily rows matched (%.1f%%)",
            matched, before, 100.0 * matched / max(before, 1),
        )

    market = build_market_features()
    panel = panel.merge(market, on="date", how="left")
    panel = add_relative_features(panel)
    panel = add_calendar_features(panel)

    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)

    logger.info(
        "Feature panel assembled — %d rows x %d columns",
        len(panel), panel.shape[1],
    )
    return panel


# Columns that are inputs to the panel or bookkeeping, never model
# features.  Everything else in the panel is offered to the model.
NON_FEATURE_COLUMNS: set[str] = {
    "symbol", "date", "open", "high", "low", "close", "volume",
    "prev_close", "nifty_close", "intraday_bar_count",
}


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Model feature names: every numeric column that is not bookkeeping.

    Label columns (``y``, ``fwd_*``, ``label_*``) are excluded by prefix,
    which is the single guard that stops a target-derived column from
    silently becoming a feature.
    """
    banned_prefixes = ("fwd_", "label_", "target_")
    cols = []
    for c in panel.columns:
        if c in NON_FEATURE_COLUMNS or c == "y":
            continue
        if c.startswith(banned_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(panel[c]):
            continue
        cols.append(c)
    return cols
