"""
RS Lift Test — Optimised Engine (DuckDB-native)
================================================
Uses DuckDB for the heavy RS join and Filter A computation.
Pulls only the compact result (picks-level data with RS attached) into pandas.

The approach:
1. DuckDB: build rs_at_checkpoints table (last RS at or before each checkpoint)
2. DuckDB: build filter_A table (cummin/cummax from 1min data)
3. DuckDB: join picks + RS + filterA into one compact evaluation frame
4. Pandas light pass: evaluate all selection methods

This avoids loading 24M RS rows into pandas memory.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
RS_5MIN = (REPO / "data/rs/rs_values/timeframe=5min/**/*.parquet").as_posix()
SECTOR_MAP_CSV = REPO / "data/rs/sector_map.csv"
PICKS_PATH = REPO / "strategies/RFactor/results/move_validation/substrate/picks.parquet"
DAILY_PATH  = REPO / "strategies/RFactor/results/move_validation/substrate/universe_daily.parquet"
STOCKS_ROOT = REPO / "data/raw/stocks"

CHECKPOINTS = ["09:25", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30"]
TOP_N  = 10
TOP_SUB = 5


def build_rs_table(con: duckdb.DuckDBPyConnection) -> None:
    """
    Build rs_at_cp table in DuckDB: last RS at or before each checkpoint
    per (symbol, date, checkpoint).
    RS_sector = max across all sector memberships (multi-sector handled).
    RS_market = mean across broad benchmarks.
    """
    print("  [DuckDB] Building RS lookup at each checkpoint ...")
    
    sect_cols = ",\n            ".join(
        f"MAX(CASE WHEN t <= '{cp}' THEN rs_value END) AS rs_sect_{cp.replace(':', '')}"
        for cp in CHECKPOINTS
    )
    mkt_cols = ",\n            ".join(
        f"AVG(CASE WHEN t <= '{cp}' THEN rs_value END) AS rs_mkt_{cp.replace(':', '')}"
        for cp in CHECKPOINTS
    )
    # Columns for final SELECT from joined CTEs
    final_sect = ",\n        ".join(
        f"s.rs_sect_{cp.replace(':', '')}" for cp in CHECKPOINTS
    )
    final_mkt = ",\n        ".join(
        f"m.rs_mkt_{cp.replace(':', '')}" for cp in CHECKPOINTS
    )
    
    sql = f"""
    CREATE OR REPLACE TABLE rs_at_cp AS
    WITH raw AS (
        SELECT
            symbol,
            CAST(timestamp AT TIME ZONE 'Asia/Calcutta' AS DATE) AS d,
            strftime(timestamp AT TIME ZONE 'Asia/Calcutta', '%H:%M') AS t,
            membership_type,
            rs_value
        FROM read_parquet('{RS_5MIN}', hive_partitioning=true)
        WHERE strftime(timestamp AT TIME ZONE 'Asia/Calcutta', '%H:%M') <= '11:30'
    ),
    sector AS (
        SELECT symbol, d,
            {sect_cols}
        FROM raw
        WHERE membership_type = 'sector'
        GROUP BY symbol, d
    ),
    market AS (
        SELECT symbol, d,
            {mkt_cols}
        FROM raw
        WHERE membership_type = 'broad'
        GROUP BY symbol, d
    )
    SELECT
        COALESCE(s.symbol, m.symbol) AS symbol,
        COALESCE(s.d, m.d) AS d,
        {final_sect},
        {final_mkt}
    FROM sector s
    FULL OUTER JOIN market m ON s.symbol = m.symbol AND s.d = m.d
    """
    con.execute(sql)
    n = con.execute("SELECT COUNT(*) FROM rs_at_cp").fetchone()[0]
    print(f"    rs_at_cp: {n:,} rows (symbol, date)")


def build_filterA_table(con: duckdb.DuckDBPyConnection,
                        universe: list[str],
                        min_date: str) -> None:
    """
    Build filterA table: for each (symbol, date, checkpoint),
    was the 9:15 extreme still the session extreme at checkpoint?
    - long filterA: cummin(low) at checkpoint >= low at 09:15
    - short filterA: cummax(high) at checkpoint <= high at 09:15
    """
    print("  [DuckDB] Building Filter A table from 1min data ...")
    paths = [(STOCKS_ROOT / sym / "1min.parquet").as_posix()
             for sym in universe
             if (STOCKS_ROOT / sym / "1min.parquet").exists()]
    path_list = "[" + ", ".join(f"'{p}'" for p in paths) + "]"
    
    cummin_cols = ",\n            ".join(
        f"MIN(CASE WHEN t <= '{cp}' THEN low END) AS cummin_{cp.replace(':', '')}"
        for cp in CHECKPOINTS
    )
    cummax_cols = ",\n            ".join(
        f"MAX(CASE WHEN t <= '{cp}' THEN high END) AS cummax_{cp.replace(':', '')}"
        for cp in CHECKPOINTS
    )
    
    sql = f"""
    CREATE OR REPLACE TABLE filterA AS
    WITH mins AS (
        SELECT
            regexp_extract(filename, 'stocks/([^/]+)/1min', 1) AS symbol,
            CAST(date AS TIMESTAMP)::DATE AS d,
            strftime(CAST(date AS TIMESTAMP), '%H:%M') AS t,
            high, low
        FROM read_parquet({path_list}, filename=true)
        WHERE CAST(date AS TIMESTAMP)::DATE >= DATE '{min_date}'
          AND strftime(CAST(date AS TIMESTAMP), '%H:%M') >= '09:15'
          AND strftime(CAST(date AS TIMESTAMP), '%H:%M') <= '11:30'
    ),
    open_bar AS (
        SELECT symbol, d,
               MIN(low)  AS low0915,
               MAX(high) AS high0915
        FROM mins WHERE t = '09:15'
        GROUP BY symbol, d
    ),
    agg AS (
        SELECT m.symbol, m.d,
            {cummin_cols},
            {cummax_cols}
        FROM mins m
        GROUP BY m.symbol, m.d
    )
    SELECT
        a.symbol, a.d,
        {", ".join(
            f"COALESCE(a.cummin_{cp.replace(':', '')} >= o.low0915 - 1e-9, false) AS fa_long_{cp.replace(':', '')},  "
            f"COALESCE(a.cummax_{cp.replace(':', '')} <= o.high0915 + 1e-9, false) AS fa_short_{cp.replace(':', '')}"
            for cp in CHECKPOINTS
        )}
    FROM agg a
    JOIN open_bar o ON a.symbol = o.symbol AND a.d = o.d
    """
    con.execute(sql)
    n = con.execute("SELECT COUNT(*) FROM filterA").fetchone()[0]
    print(f"    filterA: {n:,} rows (symbol, date)")


def build_eval_frame(con: duckdb.DuckDBPyConnection,
                     picks: pd.DataFrame,
                     daily: pd.DataFrame) -> pd.DataFrame:
    """
    Build the evaluation frame by joining picks with RS and filterA.
    Returns a long DataFrame with one row per (date, checkpoint, symbol)
    with RS values and filterA flags added.
    """
    print("  Building evaluation frame (joining picks + RS + filterA) ...")
    
    # Register picks with DuckDB
    con.register("picks_df", picks)
    
    # For each checkpoint, build a wide row
    blocks = []
    for cp in CHECKPOINTS:
        lb = cp.replace(":", "")
        
        sql = f"""
        SELECT 
            p.date, p.checkpoint, p.symbol, p.r_rank, p.in_top5,
            p.r_factor, p.open_0915, p.price_at_checkpoint,
            p.direction_at_checkpoint, p.max_up_pct, p.max_down_pct,
            p.hit_2pct, p.hit_direction,
            r.rs_sect_{lb} AS RS_sector,
            r.rs_mkt_{lb} AS RS_market,
            CASE WHEN p.direction_at_checkpoint = 'up'   THEN fa.fa_long_{lb}
                 ELSE fa.fa_short_{lb}
            END AS filterA_ok
        FROM picks_df p
        LEFT JOIN rs_at_cp r ON r.symbol = p.symbol 
                              AND r.d = CAST(p.date AS DATE)
        LEFT JOIN filterA fa ON fa.symbol = p.symbol 
                              AND fa.d = CAST(p.date AS DATE)
        WHERE p.checkpoint = '{cp}'
        """
        block = con.execute(sql).fetchdf()
        blocks.append(block)
    
    out = pd.concat(blocks, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    print(f"    Eval frame: {len(out):,} rows | "
          f"RS_sector coverage: {out['RS_sector'].notna().mean()*100:.1f}% | "
          f"RS_market coverage: {out['RS_market'].notna().mean()*100:.1f}%")
    return out


def compute_method_results(
    long: pd.DataFrame,
    base_rate: float,
    method_label: str,
    get_selection_fn,  # (grp) -> selected_df
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a selection method across all (date, checkpoint) groups.
    Returns (curve_df, daily_picks_df).
    """
    curve_rows = []
    daily_rows = []
    
    for (dt, cp), grp in long.groupby(["date", "checkpoint"]):
        selected = get_selection_fn(grp)
        if len(selected) == 0:
            continue
        top_n = selected.iloc[:TOP_N]
        top5  = selected.iloc[:TOP_SUB]
        
        hr_n = top_n["hit_2pct"].mean()
        hr_5 = top5["hit_2pct"].mean()
        
        curve_rows.append({
            "method": method_label, "checkpoint": cp,
            "n_picks": len(top_n),
            "hit_rate": hr_n, "base_rate": base_rate,
            "lift": hr_n - base_rate,
            "n_picks_top5": len(top5),
            "hit_rate_top5": hr_5, "lift_top5": hr_5 - base_rate,
        })
        
        # Best stock of the day: rank #1 of selected set
        best = selected.iloc[0]
        daily_rows.append({
            "method": method_label, "date": dt, "checkpoint": cp,
            "symbol": best["symbol"],
            "r_rank": best["r_rank"],
            "r_factor": best["r_factor"],
            "RS_sector": best.get("RS_sector", float("nan")),
            "RS_market": best.get("RS_market", float("nan")),
            "direction": best["direction_at_checkpoint"],
            "hit_2pct": best["hit_2pct"],
            "max_up_pct": best["max_up_pct"],
            "max_down_pct": best["max_down_pct"],
            "open_0915": best["open_0915"],
        })
    
    return pd.DataFrame(curve_rows), pd.DataFrame(daily_rows)


def run_lift_test() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Main lift test orchestration."""
    print("\n" + "=" * 70)
    print("  RS LIFT TEST (DuckDB-native) — COMPUTING SELECTION METHODS")
    print("=" * 70)
    
    con = duckdb.connect()
    
    picks = pd.read_parquet(PICKS_PATH)
    daily = pd.read_parquet(DAILY_PATH)
    
    print(f"  Substrate: {len(picks):,} picks | {len(daily):,} daily rows")
    
    base_rate = daily[daily["eligible"]]["hit_2pct"].mean()
    print(f"  Base rate: {base_rate*100:.2f}%")
    
    test_days = pd.DatetimeIndex(picks["date"].unique())
    universe = sorted(set(picks["symbol"]) | set(daily["symbol"]))
    min_date = test_days.min().strftime("%Y-%m-%d")
    
    # Heavy DuckDB passes
    build_rs_table(con)
    build_filterA_table(con, universe, min_date)
    
    # Build eval frame (compact, pandas-friendly)
    long = build_eval_frame(con, picks, daily)
    
    all_curves = []
    all_daily  = {}
    
    # ── Selection functions ───────────────────────────────────────────────
    
    def sel_B0(grp):
        """R-alone: top-N by r_rank."""
        return grp.sort_values("r_rank").head(TOP_N)
    
    def sel_B1(grp):
        """R + Filter A (momentum filter)."""
        fa = grp[grp["filterA_ok"] == True].sort_values("r_rank")
        return fa.head(TOP_N) if len(fa) > 0 else grp.sort_values("r_rank").head(TOP_N)
    
    def sel_S1_filter(grp):
        """R + Sector RS > 0 (filter mode)."""
        has = grp["RS_sector"].notna()
        pos = grp["RS_sector"] > 0
        f = grp[has & pos].sort_values("r_rank")
        if len(f) == 0:
            f = grp[has].sort_values("r_rank")
        return f.head(TOP_N) if len(f) > 0 else grp.sort_values("r_rank").head(TOP_N)
    
    def sel_S1_rank(grp):
        """R + Sector RS (rank mode: sort by RS_sector desc)."""
        has = grp["RS_sector"].notna()
        sub = grp[has].sort_values("RS_sector", ascending=False)
        return sub.head(TOP_N) if len(sub) > 0 else grp.sort_values("r_rank").head(TOP_N)
    
    def sel_S1_abs_rank(grp):
        """R + |Sector RS| (rank mode: sort by absolute RS_sector desc — U-shape insight)."""
        has = grp["RS_sector"].notna()
        sub = grp[has].copy()
        sub["abs_RS_sector"] = sub["RS_sector"].abs()
        sub = sub.sort_values("abs_RS_sector", ascending=False)
        return sub.head(TOP_N) if len(sub) > 0 else grp.sort_values("r_rank").head(TOP_N)
    
    def sel_S2_filter(grp):
        """R + Market RS > 0 (filter mode)."""
        has = grp["RS_market"].notna()
        pos = grp["RS_market"] > 0
        f = grp[has & pos].sort_values("r_rank")
        if len(f) == 0:
            f = grp[has].sort_values("r_rank")
        return f.head(TOP_N) if len(f) > 0 else grp.sort_values("r_rank").head(TOP_N)
    
    def sel_S2_rank(grp):
        """R + Market RS (rank mode)."""
        has = grp["RS_market"].notna()
        sub = grp[has].sort_values("RS_market", ascending=False)
        return sub.head(TOP_N) if len(sub) > 0 else grp.sort_values("r_rank").head(TOP_N)
    
    def sel_S2_abs_rank(grp):
        """R + |Market RS| (rank mode — U-shape insight)."""
        has = grp["RS_market"].notna()
        sub = grp[has].copy()
        sub["abs_RS_market"] = sub["RS_market"].abs()
        sub = sub.sort_values("abs_RS_market", ascending=False)
        return sub.head(TOP_N) if len(sub) > 0 else grp.sort_values("r_rank").head(TOP_N)
    
    def sel_S3_filter(grp):
        """R + both RS positive."""
        pos = (grp["RS_sector"] > 0) & (grp["RS_market"] > 0)
        f = grp[pos].sort_values("r_rank")
        if len(f) < 3:
            fallback = grp[(grp["RS_sector"] > 0) | (grp["RS_market"] > 0)].sort_values("r_rank")
            return fallback.head(TOP_N) if len(fallback) > 0 else grp.sort_values("r_rank").head(TOP_N)
        return f.head(TOP_N)
    
    def sel_S3_rank(grp):
        """R + combined RS (sector + market) rank mode."""
        sub = grp.copy()
        sub["RS_combined"] = sub["RS_sector"].fillna(0) + sub["RS_market"].fillna(0)
        has_any = sub["RS_sector"].notna() | sub["RS_market"].notna()
        sub = sub[has_any].sort_values("RS_combined", ascending=False)
        return sub.head(TOP_N) if len(sub) > 0 else grp.sort_values("r_rank").head(TOP_N)
    
    def sel_S4_sector(grp):
        """R + Filter A + Sector RS rank."""
        fa = grp[grp["filterA_ok"] == True]
        base = fa if len(fa) > 0 else grp
        has = base["RS_sector"].notna()
        sub = base[has].sort_values("RS_sector", ascending=False)
        return sub.head(TOP_N) if len(sub) > 0 else base.sort_values("r_rank").head(TOP_N)
    
    def sel_S4_market(grp):
        """R + Filter A + Market RS rank."""
        fa = grp[grp["filterA_ok"] == True]
        base = fa if len(fa) > 0 else grp
        has = base["RS_market"].notna()
        sub = base[has].sort_values("RS_market", ascending=False)
        return sub.head(TOP_N) if len(sub) > 0 else base.sort_values("r_rank").head(TOP_N)
    
    def sel_S5_sector(grp):
        """R + Sector RS rank, no Filter A."""
        return sel_S1_rank(grp)
    
    def sel_S5_market(grp):
        """R + Market RS rank, no Filter A."""
        return sel_S2_rank(grp)
    
    # ── Run all methods ───────────────────────────────────────────────────
    methods = [
        ("B0_R_alone",            sel_B0),
        ("B1_R_filterA",          sel_B1),
        ("S1_sector_filter",      sel_S1_filter),
        ("S1_sector_rank",        sel_S1_rank),
        ("S1_sector_abs_rank",    sel_S1_abs_rank),
        ("S2_market_filter",      sel_S2_filter),
        ("S2_market_rank",        sel_S2_rank),
        ("S2_market_abs_rank",    sel_S2_abs_rank),
        ("S3_both_filter",        sel_S3_filter),
        ("S3_both_rank",          sel_S3_rank),
        ("S4_filterA_sectorRS",   sel_S4_sector),
        ("S4_filterA_marketRS",   sel_S4_market),
        ("S5_sectorRS_nofilter",  sel_S5_sector),
        ("S5_marketRS_nofilter",  sel_S5_market),
    ]
    
    for label, fn in methods:
        print(f"  Computing {label} ...")
        c, d = compute_method_results(long, base_rate, label, fn)
        all_curves.append(c)
        all_daily[label] = d
    
    # ── Pool ──────────────────────────────────────────────────────────────
    print("\n  Pooling results ...")
    curves_df = pd.concat(all_curves, ignore_index=True)
    
    summary = (
        curves_df.groupby(["method", "checkpoint"])
        .agg(
            n_total_picks=("n_picks", "sum"),
            hit_rate=("hit_rate", "mean"),
            base_rate=("base_rate", "first"),
            lift=("lift", "mean"),
            n_total_top5=("n_picks_top5", "sum"),
            hit_rate_top5=("hit_rate_top5", "mean"),
            lift_top5=("lift_top5", "mean"),
        )
        .reset_index()
    )
    
    b0_lift = summary[summary["method"] == "B0_R_alone"].set_index("checkpoint")["lift"]
    b1_lift = summary[summary["method"] == "B1_R_filterA"].set_index("checkpoint")["lift"]
    summary["lift_delta_vs_B0"] = summary.apply(
        lambda r: r["lift"] - b0_lift.get(r["checkpoint"], float("nan")), axis=1
    )
    summary["lift_delta_vs_B1"] = summary.apply(
        lambda r: r["lift"] - b1_lift.get(r["checkpoint"], float("nan")), axis=1
    )
    
    daily_df = pd.concat(all_daily.values(), ignore_index=True)
    
    return summary, curves_df, daily_df, {
        "base_rate": base_rate,
        "long_with_rs": long,
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    summary, curves, daily_picks, extras = run_lift_test()
    print("\n--- POOLED SUMMARY (mean lift across checkpoints) ---")
    pooled = summary.groupby("method")[["lift", "lift_top5", "lift_delta_vs_B0", "lift_delta_vs_B1"]].mean()
    print((pooled * 100).round(2).to_string())
