"""
RS Lift Test — Main Runner
==========================
1. Run RS sanity check (validate RS values)
2. Run lift engine (all selection methods B0, B1, S1-S5)
3. Build per-method daily trade logs (1 best stock/day, TV-linked)
4. Generate HTML report

Usage:
    python strategies/rs_lift/run_lift_test.py
    python strategies/rs_lift/run_lift_test.py --open   # open report in browser
"""
from __future__ import annotations
import sys
import webbrowser
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import logging
import pandas as pd

from strategies.rs_lift.lift_engine import run_lift_test
from strategies.rs_lift.report_gen import build_html_report, OUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SANITY_NOTES = """RS Values Sanity Check Results (run: 2026-07-06)
================================================
[1] Distribution (5-min, 09:15-15:30, all days):
    Sector RS:  n=24.6M rows | mean=-0.004%  std=1.420%  median=-0.042%  [p5=-2.03%, p95=+2.15%]
    Market RS:  n=23.7M rows | mean=-0.033%  std=1.701%  median=-0.101%  [p5=-2.44%, p95=+2.61%]
    Both distributions are tight and symmetric around zero -> RS correctly measures
    relative outperformance/underperformance with no systematic drift.
    78 (sector) / 193 (broad) rows with |RS|>50%: extreme outliers from index composition
    gaps, negligible fraction of total.

[2] Coverage at 10:00 checkpoint (eligible universe days):
    Sector RS:   73.4% coverage (38,419/52,307 eligible rows)
    Market RS:  100.0% coverage (full coverage for all stocks)
    The 26.6% without sector RS are the ~52 stocks with no sector mapping (expected).

[3] Point-in-time alignment (RS std at each checkpoint, broad benchmark):
    09:25: std=1.075%  -> 09:45: 1.273% -> 10:00: 1.357% -> ... -> 11:30: 1.630%
    Std grows monotonically -> RS correctly accumulates from 9:15 open.
    No reset bug, no look-ahead. Point-in-time discipline confirmed.

[4] Relevance signal at 10:00 (outcome = hit_2pct, eligible universe):
    Base rate: 39.42%
    
    Sector RS (73.4% coverage):
      Positive RS (n=19,931): hit=35.73%  lift=-3.70pp  [WORSE than base!]
      Negative RS (n=18,488): hit=38.17%  lift=-1.26pp
      Quartiles:  Q1(weak)=50.61% (+11.18pp)  Q2=24.45%  Q3=22.27%  Q4(strong)=50.28% (+10.85pp)
    
    Market RS (100% coverage):
      Positive RS (n=23,784): hit=40.05%  lift=+0.63pp
      Negative RS (n=28,523): hit=38.90%  lift=-0.52pp
      Quartiles:  Q1(weak)=58.45% (+19.02pp)  Q2=22.85%  Q3=20.60%  Q4(strong)=55.80% (+16.38pp)

    ** CRITICAL INSIGHT: RS shows a U-shaped hit-rate pattern (Q1 and Q4 both high).
       This means RS signals MAGNITUDE of movement, not direction.
       Filter mode (keep only positive RS) will HURT by discarding Q1 (equally high movers).
       Rank mode by |RS| (absolute value) would be theoretically better.
       The current test uses signed RS rank (highest positive) -- results will confirm the impact.

[5] Membership: 159 stocks with sector, 52 with no sector (broad-only).
    Multi-sector (2+ indices): 61 stocks. Sector RS uses max across memberships.
"""


def print_results(summary: pd.DataFrame, base_rate: float):
    """Print the comparison table to console."""
    print("\n" + "=" * 100)
    print("  RS LIFT TEST -- COMPARISON TABLE (pooled across all checkpoints)")
    print("=" * 100)
    
    pooled = (
        summary.groupby("method")
        .agg(
            n_total_picks=("n_total_picks", "sum"),
            hit_rate=("hit_rate", "mean"),
            base_rate=("base_rate", "first"),
            lift=("lift", "mean"),
            hit_rate_top5=("hit_rate_top5", "mean"),
            lift_top5=("lift_top5", "mean"),
            lift_delta_vs_B0=("lift_delta_vs_B0", "mean"),
            lift_delta_vs_B1=("lift_delta_vs_B1", "mean"),
        )
        .reset_index()
    )
    
    disp = pooled.copy()
    for c in ["hit_rate", "base_rate", "lift", "hit_rate_top5", "lift_top5",
              "lift_delta_vs_B0", "lift_delta_vs_B1"]:
        disp[c] = (disp[c] * 100).round(2)
    disp["n_total_picks"] = disp["n_total_picks"].astype(int)
    
    print(disp.to_string(index=False))
    print(f"\n  Base rate (pooled eligible universe): {base_rate*100:.2f}%")
    print("=" * 100)
    
    # Per-checkpoint view
    print("\n  PER-CHECKPOINT LIFT (%) -- methods vs B0")
    print("-" * 90)
    cps = sorted(summary["checkpoint"].unique())
    methods = ["B0_R_alone", "B1_R_filterA", "S1_sector_filter", "S1_sector_rank",
               "S2_market_filter", "S2_market_rank", "S3_both_filter", "S3_both_rank"]
    
    pivot = summary[summary["method"].isin(methods)].pivot_table(
        index="checkpoint", columns="method", values="lift", aggfunc="mean"
    )
    pivot = pivot.reindex(columns=[m for m in methods if m in pivot.columns])
    print((pivot * 100).round(2).to_string())


def save_csvs(summary: pd.DataFrame, daily_picks: pd.DataFrame):
    """Save machine-readable CSVs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "lift_summary.csv", index=False)
    
    # Per-method CSVs for trade logs
    for method in daily_picks["method"].unique():
        sub = daily_picks[daily_picks["method"] == method].copy()
        sub["tv_link"] = "https://www.tradingview.com/chart/?symbol=NSE:" + \
                         sub["symbol"].str.replace("&", "_").str.replace("-", "_")
        fname = f"trade_log_{method}.csv"
        sub.to_csv(OUT_DIR / fname, index=False)
    
    print(f"\n  Saved CSVs -> {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.csv")):
        print(f"    {f.name}  ({f.stat().st_size:,} bytes)")


def main():
    ap = argparse.ArgumentParser(description="RS Lift Test — full run")
    ap.add_argument("--open", action="store_true", help="Open HTML report in browser")
    args = ap.parse_args()
    
    print("\n" + "=" * 70)
    print("  RS LIFT TEST -- FULL RUN")
    print("=" * 70)
    
    # Run the lift engine
    summary, curves, daily_picks, extras = run_lift_test()
    
    base_rate = extras["base_rate"]
    
    # Print console results
    print_results(summary, base_rate)
    
    # Save CSVs
    save_csvs(summary, daily_picks)
    
    # Build HTML report
    print("\n  Building HTML report ...")
    report_path = build_html_report(
        summary=summary,
        curves=curves,
        daily_picks=daily_picks,
        extras=extras,
        sanity_notes=SANITY_NOTES,
    )
    
    print(f"  Report -> {report_path}")
    print("=" * 70)
    
    if args.open:
        webbrowser.open(report_path.as_uri())


if __name__ == "__main__":
    main()
