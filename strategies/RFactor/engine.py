"""
R-Factor Ranking -> Move Validation — Engine
============================================

Point-in-time backtest engine.  Pipeline:

  1. DuckDB heavy pass: collapse the raw 1-min Parquet of the whole F&O
     universe into one compact row per (symbol, trading-day) holding the
     day-level outcome aggregates and, for each checkpoint, the cumulative
     9:15->T volume, the price at T, and the post-T high/low.
  2. pandas light pass: build the point-in-time 20-day volume baseline
     (strictly prior days), form R(T) = cum_vol(T) / baseline(T), rank the
     eligible universe per (day, checkpoint), freeze the top-N, and measure
     both outcome blocks (from-9:15 and from-checkpoint).

Look-ahead discipline:
  * R(T) uses only candles up to and including T on the test day.
  * The baseline uses only the stock's own data-days *strictly before* the
    test day (``shift(1).rolling(lookback)``).
  * Outcomes are measured forward from the frozen checkpoint; selection is
    never revised after the fact.

Nothing here loads the full universe into pandas — only the collapsed
(symbol, day) aggregates (~tens of thousands of rows) cross the boundary.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from strategies.RFactor.config import (
    CONFIG, STOCKS_GLOB, MANIFEST_PATH, CORP_ACTIONS_PATH, CALENDAR_SYMBOL,
    REPO_ROOT,
)

logger = logging.getLogger(__name__)


# ── Helpers: universe, corp actions, calendar ───────────────────────

def load_universe() -> list[str]:
    """Return the F&O universe symbol list from the manifest."""
    m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return sorted(m["symbols"])


def load_corp_action_dates() -> dict[str, list[pd.Timestamp]]:
    """Return {symbol: [ex_dates]} from the price-gap event table."""
    if not CORP_ACTIONS_PATH.exists():
        logger.warning("Corp-action table not found: %s", CORP_ACTIONS_PATH)
        return {}
    rows = json.loads(CORP_ACTIONS_PATH.read_text(encoding="utf-8"))
    out: dict[str, list[pd.Timestamp]] = {}
    for r in rows:
        out.setdefault(r["symbol"], []).append(pd.Timestamp(r["date"]))
    return out


def trading_calendar(con: duckdb.DuckDBPyConnection) -> list[date]:
    """Derive the trading-day calendar from a liquid reference symbol."""
    path = REPO_ROOT / "data" / "raw" / "stocks" / CALENDAR_SYMBOL / "1min.parquet"
    df = con.execute(
        f"SELECT DISTINCT CAST(date AS TIMESTAMP)::DATE AS d "
        f"FROM read_parquet(?) ORDER BY d",
        [str(path)],
    ).fetchdf()
    return list(df["d"])


# ── DuckDB heavy pass ───────────────────────────────────────────────

def _label(cp: str) -> str:
    """'09:25' -> '0925' (safe column suffix)."""
    return cp.replace(":", "")


def _build_agg_sql(cfg: dict, source: str) -> str:
    """Construct the per-(symbol, day) aggregation SQL.

    ``source`` is a SQL expression for ``read_parquet``'s first argument —
    either a quoted glob or a bracketed list literal of POSIX file paths.
    Paths are POSIX (forward-slash) so the symbol regex is simple.
    """
    checkpoints = cfg["checkpoints"]
    thr = cfg["move_threshold"]
    s_open = cfg["session_open"]
    s_end = cfg["session_end"]

    # Per-checkpoint conditional aggregates.
    cp_cols = []
    for cp in checkpoints:
        lb = _label(cp)
        cp_cols.append(
            f"SUM(volume) FILTER (WHERE t <= TIME '{cp}') AS cv_{lb}"
        )
        cp_cols.append(
            f"arg_max(close, t) FILTER (WHERE t <= TIME '{cp}') AS px_{lb}"
        )
        cp_cols.append(
            f"MAX(high) FILTER (WHERE t > TIME '{cp}') AS hi_after_{lb}"
        )
        cp_cols.append(
            f"MIN(low) FILTER (WHERE t > TIME '{cp}') AS lo_after_{lb}"
        )
    cp_block = ",\n        ".join(cp_cols)

    sql = f"""
    WITH mins AS (
        SELECT
            regexp_extract(filename, 'stocks/([^/]+)/1min', 1) AS symbol,
            CAST(date AS TIMESTAMP)::DATE AS d,
            CAST(date AS TIMESTAMP)::TIME AS t,
            open, high, low, close, volume
        FROM read_parquet({source}, filename=true)
        WHERE date >= ?
          AND CAST(date AS TIMESTAMP)::TIME BETWEEN TIME '{s_open}' AND TIME '{s_end}'
    ),
    day_agg AS (
        SELECT
            symbol, d,
            arg_min(open, t)  AS open_0915,
            MAX(high)         AS day_high,
            MIN(low)          AS day_low,
            arg_max(close, t) AS session_close,
            COUNT(*)          AS n_candles,
            MIN(t)            AS first_t,
            MAX(t)            AS last_t,
        {cp_block}
        FROM mins
        GROUP BY symbol, d
    ),
    first_hit AS (
        SELECT
            m.symbol, m.d,
            MIN(CASE WHEN m.high >= a.open_0915 * (1 + {thr})
                       OR m.low  <= a.open_0915 * (1 - {thr})
                     THEN m.t END) AS time_first_hit
        FROM mins m
        JOIN day_agg a USING (symbol, d)
        GROUP BY m.symbol, m.d
    )
    SELECT a.*, f.time_first_hit
    FROM day_agg a
    JOIN first_hit f USING (symbol, d)
    ORDER BY a.symbol, a.d
    """
    return sql


def aggregate_symbol_days(
    con: duckdb.DuckDBPyConnection,
    cfg: dict,
    cutoff: date,
    universe: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run the DuckDB pass -> compact per-(symbol, day) feature frame.

    Parameters
    ----------
    cutoff : date
        Earliest trading day to scan (covers test window + baseline room).
    universe : list[str], optional
        Symbols to scan.  Defaults to the full F&O manifest.  An explicit
        POSIX file list is passed to DuckDB so only these symbols are read.
    """
    if universe is None:
        universe = load_universe()

    stocks_root = (REPO_ROOT / "data" / "raw" / "stocks").as_posix()
    paths = [f"{stocks_root}/{sym}/1min.parquet" for sym in universe]
    # Bracketed SQL list literal of single-quoted POSIX paths.
    source = "[" + ", ".join("'" + p.replace("'", "''") + "'" for p in paths) + "]"

    sql = _build_agg_sql(cfg, source)
    cutoff_ts = pd.Timestamp(cutoff).tz_localize("Asia/Kolkata")
    df = con.execute(sql, [cutoff_ts.to_pydatetime()]).fetchdf()

    df["d"] = pd.to_datetime(df["d"])
    df = df.sort_values(["symbol", "d"]).reset_index(drop=True)
    logger.info("Aggregated %d (symbol, day) rows for %d symbols",
                len(df), df["symbol"].nunique())
    return df


# ── pandas light pass: baseline, R, eligibility ─────────────────────

def _corp_contaminated_mask(
    sub: pd.DataFrame,
    corp_dates: list[pd.Timestamp],
    lookback: int,
) -> np.ndarray:
    """Boolean mask over a single symbol's sorted day rows.

    A test day is contaminated when a corporate-action ex-date falls within
    its baseline window (the ``lookback`` prior data-days) or on the day
    itself.  For each ex-date C present in the series at position p, that
    means rows [p, p + lookback] (C itself + the next ``lookback`` data-days
    whose baseline reaches back to C).
    """
    n = len(sub)
    mask = np.zeros(n, dtype=bool)
    if not corp_dates:
        return mask
    days = sub["d"].values
    for c in corp_dates:
        pos = np.searchsorted(days, np.datetime64(c))
        # Mark [pos, pos+lookback] if C is an actual data-day at pos.
        if pos < n and days[pos] == np.datetime64(c):
            mask[pos: min(n, pos + lookback + 1)] = True
        elif pos < n:
            # C not a data-day for this symbol (rare): still blackout the
            # following lookback data-days whose window spans C.
            mask[pos: min(n, pos + lookback)] = True
    return mask


def compute_features(
    agg: pd.DataFrame,
    cfg: dict,
    corp_actions: dict[str, list[pd.Timestamp]],
    test_days: set,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the long picks-substrate and the per-day universe table.

    Returns
    -------
    (long_df, daily_df)
        ``long_df``  : one row per (symbol, date, checkpoint) over test days,
                       with R, rank, eligibility and both outcome blocks.
        ``daily_df`` : one row per (symbol, date) over test days, with the
                       from-9:15 outcomes and the eligibility flag — the
                       base-rate / magnitude substrate.
    """
    checkpoints = cfg["checkpoints"]
    lookback = cfg["rvol_lookback"]
    thr = cfg["move_threshold"]
    cap_bar = cfg["capture_bar"]
    min_candles = cfg["min_day_candles"]
    close_after = pd.Timestamp(cfg["session_close_after"]).time()
    exclude_ca = cfg["exclude_corp_action_window"]

    agg = agg.sort_values(["symbol", "d"]).reset_index(drop=True)

    # ── normal-session mask (computed first; gates the baseline) ────
    # Only normal full regular sessions (valid 9:15 open, ~complete day)
    # feed the RVOL baseline.  Special/short sessions — e.g. Diwali
    # Muhurat — have an abnormal volume profile and a NaN early-window
    # cumulative volume; including them would both distort the baseline
    # and (via rolling min_periods) blank out the next `lookback` days.
    close_after_min = close_after.hour * 60 + close_after.minute
    last_min = agg["last_t"].map(
        lambda x: x.hour * 60 + x.minute if x is not None else -1
    )
    has_open = agg["open_0915"].notna()
    normal_session = (
        has_open
        & (agg["n_candles"] >= min_candles)
        & (last_min >= close_after_min)
    )
    agg["normal_session"] = normal_session
    complete_day = normal_session

    # ── point-in-time baseline + R for each checkpoint ──────────────
    # Baseline rolls over normal sessions ONLY (strictly prior days).  On a
    # normal session a NaN cumulative volume means the stock simply had no
    # prints in that early window -> genuine zero participation -> 0.
    norm_idx = agg.index[normal_session.values]
    norm = agg.loc[norm_idx, ["symbol"]].copy()
    for cp in checkpoints:
        lb = _label(cp)
        cvf = agg.loc[norm_idx, f"cv_{lb}"].fillna(0.0)
        agg[f"cv_{lb}"] = agg[f"cv_{lb}"].where(~normal_session, agg[f"cv_{lb}"].fillna(0.0))
        base = (
            cvf.groupby(norm["symbol"].values, sort=False)
            .transform(lambda s: s.shift(1).rolling(lookback, min_periods=lookback).mean())
        )
        agg[f"base_{lb}"] = np.nan
        agg.loc[norm_idx, f"base_{lb}"] = base.values
        agg[f"R_{lb}"] = agg[f"cv_{lb}"] / agg[f"base_{lb}"]

    # ── per-(symbol, day) day-level eligibility (checkpoint-independent) ──
    first_lb = _label(checkpoints[0])
    has_baseline = agg[f"base_{first_lb}"].notna()

    # corp-action window contamination (per symbol)
    contam = np.zeros(len(agg), dtype=bool)
    if exclude_ca:
        for sym, idx in agg.groupby("symbol", sort=False).indices.items():
            sub = agg.iloc[idx]
            m = _corp_contaminated_mask(sub, corp_actions.get(sym, []), lookback)
            contam[idx] = m
    agg["corp_contaminated"] = contam

    agg["eligible"] = complete_day & has_baseline & (~agg["corp_contaminated"])

    # ── from-9:15 outcome block (per symbol, day) ───────────────────
    o = agg["open_0915"]
    agg["max_up_pct"] = (agg["day_high"] - o) / o
    agg["max_down_pct"] = (o - agg["day_low"]) / o
    up_hit = agg["max_up_pct"] >= thr
    dn_hit = agg["max_down_pct"] >= thr
    agg["hit_2pct"] = up_hit | dn_hit
    agg["hit_direction"] = np.select(
        [up_hit & dn_hit, up_hit & ~dn_hit, ~up_hit & dn_hit],
        ["both", "up", "down"], default="none",
    )

    # ── restrict to test days ───────────────────────────────────────
    test_mask = agg["d"].isin(test_days)
    daily_cols = [
        "symbol", "d", "open_0915", "day_high", "day_low", "session_close",
        "n_candles", "first_t", "last_t", "time_first_hit",
        "max_up_pct", "max_down_pct", "hit_2pct", "hit_direction",
        "corp_contaminated", "eligible",
    ]
    daily_df = agg.loc[test_mask, daily_cols].rename(
        columns={"d": "date"}
    ).reset_index(drop=True)

    # ── build long (symbol, day, checkpoint) frame ──────────────────
    test_agg = agg.loc[test_mask].reset_index(drop=True)
    blocks = []
    for cp in checkpoints:
        lb = _label(cp)
        b = pd.DataFrame({
            "date": test_agg["d"].values,
            "symbol": test_agg["symbol"].values,
            "checkpoint": cp,
            "cv": test_agg[f"cv_{lb}"].values,
            "baseline": test_agg[f"base_{lb}"].values,
            "r_factor": test_agg[f"R_{lb}"].values,
            "price_at_checkpoint": test_agg[f"px_{lb}"].values,
            "hi_after": test_agg[f"hi_after_{lb}"].values,
            "lo_after": test_agg[f"lo_after_{lb}"].values,
            "open_0915": test_agg["open_0915"].values,
            "day_high": test_agg["day_high"].values,
            "day_low": test_agg["day_low"].values,
            "session_close": test_agg["session_close"].values,
            "max_up_pct": test_agg["max_up_pct"].values,
            "max_down_pct": test_agg["max_down_pct"].values,
            "hit_2pct": test_agg["hit_2pct"].values,
            "hit_direction": test_agg["hit_direction"].values,
            "time_first_hit": test_agg["time_first_hit"].values,
            "eligible": test_agg["eligible"].values,
        })
        blocks.append(b)
    long_df = pd.concat(blocks, ignore_index=True)

    # ── rank eligible universe per (date, checkpoint) by R desc ──────
    elig = long_df["eligible"] & long_df["r_factor"].notna()
    long_df["r_rank"] = np.nan
    ranked = (
        long_df.loc[elig]
        .sort_values(["date", "checkpoint", "r_factor"],
                     ascending=[True, True, False])
    )
    ranks = ranked.groupby(["date", "checkpoint"], sort=False).cumcount() + 1
    long_df.loc[ranked.index, "r_rank"] = ranks.values

    # ── from-checkpoint capture block ───────────────────────────────
    px = long_df["price_at_checkpoint"]
    direction_up = px >= long_df["open_0915"]
    long_df["direction_at_checkpoint"] = np.where(direction_up, "up", "down")
    fav = np.where(
        direction_up,
        (long_df["hi_after"] - px) / px,
        (px - long_df["lo_after"]) / px,
    )
    long_df["fav_move_after_cp_pct"] = fav
    long_df["capturable"] = long_df["fav_move_after_cp_pct"] >= cap_bar
    long_df["close"] = long_df["session_close"]

    # top-N / top-subset flags
    top_n = cfg["top_n"]
    top_sub = cfg["top_subset"]
    long_df["in_topn"] = long_df["r_rank"] <= top_n
    long_df["in_top5"] = long_df["r_rank"] <= top_sub

    return long_df, daily_df


def make_picks(long_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Filter the long frame to the frozen top-N picks (section-5 schema)."""
    cols = [
        "date", "checkpoint", "symbol", "r_rank", "in_top5",
        "r_factor", "cv", "baseline",
        "open_0915", "price_at_checkpoint", "direction_at_checkpoint",
        "max_up_pct", "max_down_pct", "hit_2pct", "hit_direction",
        "time_first_hit",
        "fav_move_after_cp_pct", "capturable", "close",
    ]
    picks = long_df.loc[long_df["in_topn"], cols].copy()
    picks = picks.sort_values(["date", "checkpoint", "r_rank"]).reset_index(drop=True)
    return picks
