# Pass 2.2 · Rolling Basket + Breakeven · swing-K sweep

Two rule changes on top of [2.1](../pass2.1_filterA_0930/), swept over swing K:

1. **Rolling basket** — the top-4 bucket is re-ranked **every 15 min** across
   09:30–12:30 (not locked once). A breakout qualifies only if its stock is in
   the top-4 as of the most recent checkpoint ≤ the trigger. So a stock that
   climbs into the bucket at 10:30 becomes tradeable then; one that drops out
   stops being tradeable.
2. **Breakeven at 1:2** — once price touches **+2R** in favour, the stop moves to
   **entry (0R)**. Outcomes: +3R / 0R (breakeven) / −1R / 3:00 time-close.

Everything else fixed: Filter A, entry 09:30–12:30, 2× huge, stop = farther of
{swing-body, 3rd-back}, spot, gross. Only **swing K ∈ {2,3,4,5}** varies.

## Result — K = 4 is best

Window **2025-06-30 → 2026-07-03** · 249 days · rolling top-4 · Filter A.

| Metric | K=2 | K=3 | **K=4** | K=5 |
|---|---|---|---|---|
| **Expectancy / trade** | +0.018 R | −0.011 R | **+0.152 R** | +0.137 R |
| Trades · no-trade days | 247 · 2 | 243 · 6 | 234 · 15 | 213 · 36 |
| Win rate | 30.8% | 27.2% | 35.5% | 34.3% |
| Target / breakeven / stop / time | 15/7/60/18% | 15/11/58/16% | 17/9/52/22% | 18/8/55/19% |
| Max drawdown | 13.2 R | 26.9 R | 14.7 R | 15.1 R |
| Longest losing streak | 7 | 10 | 7 | 10 |

**Higher K (fewer, more significant swings) works better** — K=4 wins on
expectancy *and* drawdown *and* streak, K=5 close behind. K=2/K=3 are
flat-to-negative. The cost of higher K is fewer trades (K=5: 36 no-trade days).

**Caveats / honest reads:**
- **Breakeven looks like it caps winners.** 7–11% of trades end at 0R (touched
  +2R, came back to entry) — some of those would have reached +3R without the
  breakeven stop. Worth an explicit *breakeven vs no-breakeven* comparison next.
- **Not directly comparable to 2.1's +0.187R** — the data now extends to
  2026-07-03, so this is a *different (249-day) window*, and it changes three
  things at once (rolling, breakeven, K). To isolate each, re-run 2.1 on this
  window and/or toggle breakeven off.
- Spot & gross, single regime — necessary, not sufficient.

## Files

| File | What it is |
|------|-----------|
| `report.html` | **Open first.** K-comparison metrics table (best K highlighted), equity curves, outcome split, expectancy-by-K. |
| `trade_log.html` | All 937 trades across K in one filterable/sortable table (green=win, grey=breakeven, red=loss). |
| `run_meta.json` | Config + rolling checkpoints + per-K metrics. |
| `K2/ … K5/` | Per-K `metrics.csv` + trade log (`trades.parquet` + `trades.csv`). |

## Regenerate

```bash
python strategies/EntryGeometry/run_pass2_2.py               # full sweep K=2..5
python strategies/EntryGeometry/run_pass2_2.py --ks 4 5      # subset
```
