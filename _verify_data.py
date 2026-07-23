"""
Comprehensive Data Verification
================================
Spot-check fetched data against known market values to confirm accuracy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from infrastructure.data.loader import load, list_available

# ── Part 1: What data do we have? ───────────────────────────────────

print("=" * 70)
print("  DATA INVENTORY")
print("=" * 70)

available = list_available()
for asset_type, symbols in available.items():
    for symbol, timeframes in symbols.items():
        for tf in timeframes:
            df = load(symbol, asset_type, tf)
            date_min = df.index.min()
            date_max = df.index.max()
            days_span = (date_max - date_min).days
            print(f"\n  {symbol} | {asset_type} | {tf}")
            print(f"    Rows:       {len(df):,}")
            print(f"    Columns:    {list(df.columns)}")
            print(f"    From:       {date_min}")
            print(f"    To:         {date_max}")
            print(f"    Span:       {days_span} calendar days")

# ── Part 2: Spot-check NIFTY daily against known events ─────────────

print("\n" + "=" * 70)
print("  SPOT-CHECK: NIFTY Daily vs Known Market Events")
print("=" * 70)

nifty = load("NIFTY", "index", "daily")

# Known NIFTY close values on specific dates (approximate, from public records)
# These are well-documented market events anyone can verify
spot_checks = [
    ("2021-02-01", "Budget 2021 (big rally day)", 14281, 14650),
    ("2022-06-17", "Bear market low zone 2022", 15200, 15400),
    ("2023-12-01", "Dec 2023 (Nifty near 20,000)", 20100, 20300),
    ("2024-06-04", "Election results day 2024 (crash)", 21600, 22100),
    ("2024-09-27", "Nifty ATH zone Sep 2024", 26100, 26300),
    ("2025-04-07", "Tariff shock Apr 2025", 21700, 22200),
]

for date_str, event, low_bound, high_bound in spot_checks:
    if date_str in nifty.index.strftime("%Y-%m-%d"):
        row = nifty.loc[date_str]
        if hasattr(row, "iloc"):
            row = row.iloc[0] if len(row.shape) > 1 else row
        close = row["close"]
        in_range = low_bound <= close <= high_bound
        status = "[OK]" if in_range else "[??]"
        print(f"\n  {status}  {date_str} -- {event}")
        print(f"        Close: {close:,.2f}  (expected range: {low_bound:,}-{high_bound:,})")
    else:
        print(f"\n  [--]  {date_str} -- {event}")
        print(f"        Date not in dataset (holiday or outside range)")

# ── Part 3: Spot-check INDIAVIX daily ───────────────────────────────

print("\n" + "=" * 70)
print("  SPOT-CHECK: INDIAVIX Daily")
print("=" * 70)

vix = load("INDIAVIX", "volatility", "daily")

vix_checks = [
    ("2024-06-04", "Election day VIX spike", 20, 32),
    ("2025-04-07", "Tariff shock VIX spike", 18, 30),
]

for date_str, event, low_bound, high_bound in vix_checks:
    if date_str in vix.index.strftime("%Y-%m-%d"):
        row = vix.loc[date_str]
        if hasattr(row, "iloc"):
            row = row.iloc[0] if len(row.shape) > 1 else row
        close = row["close"]
        in_range = low_bound <= close <= high_bound
        status = "[OK]" if in_range else "[??]"
        print(f"\n  {status}  {date_str} -- {event}")
        print(f"        VIX Close: {close:.2f}  (expected range: {low_bound}-{high_bound})")
    else:
        print(f"\n  [--]  {date_str} -- {event}")
        print(f"        Date not in dataset")

# ── Part 4: NIFTY 15min basic sanity ────────────────────────────────

print("\n" + "=" * 70)
print("  SPOT-CHECK: NIFTY 15min Structure")
print("=" * 70)

nifty_15 = load("NIFTY", "index", "15min")
unique_dates = nifty_15.index.normalize().unique()
candles_per_day = nifty_15.groupby(nifty_15.index.date).size()

print(f"\n  Trading days:        {len(unique_dates)}")
print(f"  Avg candles/day:     {candles_per_day.mean():.1f}")
print(f"  Min candles/day:     {candles_per_day.min()} (on {candles_per_day.idxmin()})")
print(f"  Max candles/day:     {candles_per_day.max()} (on {candles_per_day.idxmax()})")
print(f"  First candle time:   {nifty_15.index[0].strftime('%H:%M')}")
print(f"  Last candle time:    {nifty_15.index[-1].strftime('%H:%M')}")

# ── Part 5: NIFTY 1min basic sanity ────────────────────────────────

print("\n" + "=" * 70)
print("  SPOT-CHECK: NIFTY 1min Structure")
print("=" * 70)

nifty_1 = load("NIFTY", "index", "1min")
unique_dates_1 = nifty_1.index.normalize().unique()
candles_per_day_1 = nifty_1.groupby(nifty_1.index.date).size()

print(f"\n  Trading days:        {len(unique_dates_1)}")
print(f"  Avg candles/day:     {candles_per_day_1.mean():.1f}")
print(f"  Min candles/day:     {candles_per_day_1.min()} (on {candles_per_day_1.idxmin()})")
print(f"  Max candles/day:     {candles_per_day_1.max()} (on {candles_per_day_1.idxmax()})")
print(f"  First candle time:   {nifty_1.index[0].strftime('%H:%M')}")
print(f"  Last candle time:    {nifty_1.index[-1].strftime('%H:%M')}")

# ── Part 6: Cross-check daily vs 15min ──────────────────────────────

print("\n" + "=" * 70)
print("  CROSS-CHECK: Daily close vs 15min last candle close")
print("=" * 70)

# Pick 5 random overlapping dates
overlap_dates = nifty.index.intersection(
    pd.DatetimeIndex(unique_dates)
)
if len(overlap_dates) > 5:
    check_dates = overlap_dates[-5:]  # last 5 dates
else:
    check_dates = overlap_dates

for dt in check_dates:
    date_str = dt.strftime("%Y-%m-%d")
    daily_close = nifty.loc[date_str]["close"]
    if hasattr(daily_close, "iloc"):
        daily_close = daily_close.iloc[0]

    day_15 = nifty_15.loc[date_str]
    last_15_close = day_15.iloc[-1]["close"]

    diff_pct = abs(daily_close - last_15_close) / daily_close * 100
    status = "[OK]" if diff_pct < 0.5 else "[??]"
    print(f"  {status}  {date_str}  daily={daily_close:,.2f}  15min_last={last_15_close:,.2f}  diff={diff_pct:.3f}%")

print("\n" + "=" * 70)
print("  VERIFICATION COMPLETE")
print("=" * 70)
