# Daily R-Factor Rank Leaderboard

For the past ~1-year window, the top-N R-factor-ranked stocks **in the bucket at
every checkpoint, every day** — with each symbol linked to its TradingView chart.

Window **2025-06-18 → 2026-06-19** · 248 trading days · top-10 of 211 F&O ·
checkpoints **09:25, 09:30, 09:45, 10:00, 10:15, 10:30, 10:45, 11:00, 11:15, 11:30**.
Bucket = top-4 (highlighted green). 24,800 rows.

## Files

| File | What it is |
|------|-----------|
| `leaderboard.html` | **Open this.** One collapsible grid per day (rows = checkpoints, columns = rank 1–10). Click any symbol → its TradingView chart opens (set the date there). Hover a cell for its R value. Date-jump box + expand/collapse all. |
| `leaderboard.csv` | Flat table: date, checkpoint, r_rank, symbol, r_factor, in_basket, **tradingview** (URL column). |
| `leaderboard.parquet` | Same, machine-readable. |
| `run_meta.json` | Window, checkpoints, top-N, dropped days. |

## TradingView links

Symbols map to `NSE:<SYMBOL>` with `&`/`-` → `_` (e.g. `M&M` → `NSE:M_M`,
`BAJAJ-AUTO` → `NSE:BAJAJ_AUTO`). The chart opens at the current date; set the
historical date manually. If a rare ticker doesn't resolve, TradingView opens a
search near it.

## Note on churn (relevant to Pass 2 / 2.1)

The bucket **churns** across checkpoints — a stock in the top-4 at 9:30 is often
not in the top-4 by 10:15. Pass 2 / 2.1 **lock the bucket once** (at 10:00 / 9:30)
and watch only those 4 all day, so they never trade a stock that climbs in later.
This view is how you see what that lock costs.

## Regenerate

```bash
python strategies/RFactor/daily_leaderboard.py            # top-10, past year
python strategies/RFactor/daily_leaderboard.py --top-n 4 --open   # bucket only, and open it
```
