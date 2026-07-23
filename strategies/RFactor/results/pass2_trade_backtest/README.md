# Pass 2 · Full Trade Backtest (spot, gross)

**Question:** does price travel **+3R before −1R** (or force-close at 3:00) often
enough to yield positive **expectancy per trade**? Measured on **spot**, **gross**
of costs — the clean underlying-move edge only.

## Headline result

Window **2025-06-18 → 2026-06-19** · 248 basket days · 1 trade/day · gross, spot.
Fixed: 2× huge filter, 1% stop cap, **stop = farther of {swing-body, 3rd-back}**,
3R target, 3:00 hard force-close. Two momentum filters, run separately:

| | Filter A (9:15-extreme held) | Filter B (side of open) |
|---|---|---|
| **Expectancy / trade** | **+0.146 R** | **+0.136 R** |
| Trades | 245 | 248 |
| Win rate | 34.7% | 34.7% |
| Target / stop / time | 20% / 62% / 18% | 19% / 61% / 19% |
| Max drawdown | 16.8 R | 16.8 R |
| Longest losing streak | 15 | 15 |

**Both filters show a small positive edge, and are nearly identical on this
window** — the momentum-filter choice barely matters here. Read *expectancy*, not
win rate (low by design with 3R:1R). No winner is chosen — see the report.

## Files

| File | What it is |
|------|-----------|
| `report.html` | **Open first.** A vs B side by side: verdict table, equity/drawdown curves, outcome split, R distribution, trade-log preview, honesty notes. |
| `run_meta.json` | Config + window + both filters' full metrics. |
| `filter_A/` · `filter_B/` | Per-filter `metrics.csv` + full trade log (`trades.parquet` + `trades.csv`), one row per taken trade, stamped for TradingView. |

## Honesty notes (also in the report)

- Fixed 3R is a **proxy**, not the real 15-min-level exit — results will shift.
- Entry at candle **close** is slightly optimistic (live = next candle's open).
- **Spot & gross** — no IV/theta/spread/costs; the option + cost layers come only if an edge holds.
- **One (rising) regime** — necessary, not sufficient; walk-forward is later.
- Same-candle target+stop → **stop-first** (pessimistic); occurred 1× each (negligible).

## Regenerate

```bash
python strategies/EntryGeometry/run_pass2.py         # -> filter_*/ + run_meta.json
python strategies/EntryGeometry/pass2_report.py --open
```

Full spec, method, and what a later Pass 3 would add: `strategies/EntryGeometry/README.md`.
