"""
RS Values — Sanity & Relevance Check
=====================================
Verifies that rs_values parquet is correctly built, point-in-time aligned,
and that RS has a measurable correlation with next-move outcomes.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
RS_GLOB = (REPO / "data/rs/rs_values/timeframe=5min/**/*.parquet").as_posix()
SECTOR_MAP = REPO / "data/rs/sector_map.csv"
PICKS_PATH = REPO / "strategies/RFactor/results/move_validation/substrate/picks.parquet"
DAILY_PATH = REPO / "strategies/RFactor/results/move_validation/substrate/universe_daily.parquet"

CHECKPOINTS = ["09:25", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30"]

def main():
    con = duckdb.connect()
    smap = pd.read_csv(SECTOR_MAP)
    picks = pd.read_parquet(PICKS_PATH)
    daily = pd.read_parquet(DAILY_PATH)
    
    print("=" * 70)
    print("  RS SANITY & RELEVANCE CHECK")
    print("=" * 70)

    # ── 1. RS Distribution ───────────────────────────────────────────────
    print("\n[1] RS Distribution (5-min, session hours only)")
    rs_dist = con.execute(f"""
        SELECT 
            membership_type,
            COUNT(*) AS n,
            ROUND(AVG(rs_value)*100, 3) AS mean_pct,
            ROUND(STDDEV(rs_value)*100, 3) AS std_pct,
            ROUND(MIN(rs_value)*100, 2) AS min_pct,
            ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY rs_value)*100, 2) AS p5_pct,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY rs_value)*100, 2) AS p25_pct,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY rs_value)*100, 3) AS median_pct,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY rs_value)*100, 2) AS p75_pct,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY rs_value)*100, 2) AS p95_pct,
            ROUND(MAX(rs_value)*100, 2) AS max_pct,
            COUNT(CASE WHEN ABS(rs_value) > 0.5 THEN 1 END) AS extreme_50pct
        FROM read_parquet('{RS_GLOB}', hive_partitioning=true)
        WHERE strftime(timestamp AT TIME ZONE 'Asia/Calcutta', '%H:%M') >= '09:15'
          AND strftime(timestamp AT TIME ZONE 'Asia/Calcutta', '%H:%M') <= '15:30'
        GROUP BY membership_type
    """).fetchdf()
    print(rs_dist.to_string(index=False))

    # ── 2. Coverage check ────────────────────────────────────────────────
    print("\n[2] RS Coverage at checkpoints (% of eligible rows with RS)")
    elig_daily = daily[daily["eligible"]]
    
    # Spot check at 10:00 for speed
    rs_1000 = con.execute(f"""
        SELECT symbol,
               CAST(timestamp AT TIME ZONE 'Asia/Calcutta' AS DATE) AS d,
               membership_type,
               benchmark_index,
               rs_value
        FROM read_parquet('{RS_GLOB}', hive_partitioning=true)
        WHERE strftime(timestamp AT TIME ZONE 'Asia/Calcutta', '%H:%M') = '10:00'
    """).fetchdf()
    rs_1000["d"] = pd.to_datetime(rs_1000["d"])
    rs_1000.rename(columns={"d": "date"}, inplace=True)
    
    rs_sector_1000 = rs_1000[rs_1000["membership_type"] == "sector"]
    rs_broad_1000  = rs_1000[rs_1000["membership_type"] == "broad"]
    
    for label, rs_sub in [("sector", rs_sector_1000), ("broad/market", rs_broad_1000)]:
        has_rs = elig_daily.merge(
            rs_sub[["symbol", "date"]].drop_duplicates(),
            on=["symbol", "date"], how="left", indicator=True
        )["_merge"] == "both"
        print(f"    {label}: {has_rs.mean()*100:.1f}% of eligible rows have RS at 10:00  "
              f"({has_rs.sum():,}/{len(elig_daily):,})")

    # ── 3. Point-in-time alignment ───────────────────────────────────────
    print("\n[3] Point-in-time: RS std grows through session (confirms accumulation, not reset)")
    alignment = con.execute(f"""
        SELECT 
            strftime(timestamp AT TIME ZONE 'Asia/Calcutta', '%H:%M') AS bar_time,
            COUNT(*) AS n_rows,
            ROUND(AVG(rs_value)*100, 3) AS avg_rs_pct,
            ROUND(STDDEV(rs_value)*100, 3) AS std_rs_pct
        FROM read_parquet('{RS_GLOB}', hive_partitioning=true)
        WHERE strftime(timestamp AT TIME ZONE 'Asia/Calcutta', '%H:%M') IN (
            '09:25', '09:45', '10:00', '10:15',
            '10:30', '10:45', '11:00', '11:15', '11:30'
        )
        AND membership_type = 'broad'
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()
    print(alignment.to_string(index=False))
    stds = alignment["std_rs_pct"].values
    if stds[-1] > stds[0]:
        print("    ✓ Std grows -> RS correctly accumulates intraday (no reset bug)")
    else:
        print("    ⚠ Std does NOT grow -> possible RS reset / incorrect computation")

    # ── 4. Relevance signal ──────────────────────────────────────────────
    print("\n[4] Relevance signal at 10:00 checkpoint")
    
    # For multi-sector: use max RS (most bullish/bearish sector)
    rs_sector_max = rs_sector_1000.groupby(["symbol", "date"])["rs_value"].max().reset_index()
    rs_sector_max.columns = ["symbol", "date", "RS_sector"]
    
    # Market RS: average across broad benchmarks (usually just 1 per stock)
    rs_market = rs_broad_1000.groupby(["symbol", "date"])["rs_value"].mean().reset_index()
    rs_market.columns = ["symbol", "date", "RS_market"]
    
    elig = daily[daily["eligible"]].copy()
    merged = elig.merge(rs_sector_max, on=["symbol", "date"], how="left")
    merged = merged.merge(rs_market, on=["symbol", "date"], how="left")
    
    base_rate = merged["hit_2pct"].mean()
    print(f"    Base hit rate (all eligible): {base_rate*100:.2f}%")
    print(f"    RS_sector coverage: {merged['RS_sector'].notna().mean()*100:.1f}%")
    print(f"    RS_market coverage: {merged['RS_market'].notna().mean()*100:.1f}%")
    
    for col, label in [("RS_sector", "Sector RS"), ("RS_market", "Market RS")]:
        sub = merged[merged[col].notna()].copy()
        if len(sub) < 100:
            print(f"\n    {label}: insufficient data ({len(sub)} rows)")
            continue
        pos = sub[sub[col] > 0]
        neg = sub[sub[col] <= 0]
        pos_hit = pos["hit_2pct"].mean() if len(pos) else float("nan")
        neg_hit = neg["hit_2pct"].mean() if len(neg) else float("nan")
        corr = sub[col].corr(sub["hit_2pct"].astype(float))
        print(f"\n    {label}:")
        print(f"      Positive RS (n={len(pos):,}): hit={pos_hit*100:.2f}%  "
              f"lift={+(pos_hit-base_rate)*100:+.2f}pp")
        print(f"      Negative RS (n={len(neg):,}): hit={neg_hit*100:.2f}%  "
              f"lift={(neg_hit-base_rate)*100:+.2f}pp")
        print(f"      Corr (RS vs hit_2pct): {corr:.4f}")
        # Quartile breakdown
        try:
            sub["RS_q"] = pd.qcut(sub[col], q=4, labels=["Q1(weak)", "Q2", "Q3", "Q4(strong)"],
                                   duplicates="drop")
            qt = sub.groupby("RS_q", observed=True)["hit_2pct"].agg(hit_rate="mean", n="count")
            qt["hit_pct"] = (qt["hit_rate"] * 100).round(2)
            qt["lift_pp"] = ((qt["hit_rate"] - base_rate) * 100).round(2)
            print(f"      Quartiles:")
            print(qt[["n", "hit_pct", "lift_pp"]].to_string())
        except Exception as e:
            print(f"      Quartile failed: {e}")

    # ── 5. Multi-sector & unmapped summary ──────────────────────────────
    print("\n[5] Membership summary")
    sector_mem = smap[smap["membership_type"] == "sector"]
    broad_mem  = smap[smap["membership_type"] == "broad"]
    no_sector_syms = sorted(
        set(broad_mem["symbol"]) - set(sector_mem["symbol"])
    )
    sectors_per_sym = sector_mem.groupby("symbol").size()
    print(f"    Stocks with ≥1 sector:   {len(sectors_per_sym)}")
    print(f"    Stocks in 2+ sectors:    {(sectors_per_sym >= 2).sum()}")
    print(f"    Stocks with NO sector:   {len(no_sector_syms)} (broad-only; ~50-60 unmapped)")
    print(f"    Sample unmapped stocks:  {no_sector_syms[:10]}")
    
    print("\n" + "=" * 70)
    print("  SANITY CHECK COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
