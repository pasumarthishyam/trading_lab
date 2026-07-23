# Pass 2.1 · Filter A, entry from 9:30

Variant of the Pass 2 full trade backtest with **two changes** vs the
[10:00 baseline](../pass2_trade_backtest/):

1. **Only Filter A** (9:15-extreme held) — Filter B is not run.
2. **Basket lock + entry window moved to 9:30** (from 10:00). R-rank now freezes
   at 9:30 using the 9:15–9:30 volume window, and entries are hunted 9:30–12:30.
   *(Entries can only start once a K=2 swing has confirmed — earliest ~9:35.)*

Everything else identical and fixed: 2× huge filter, 1% stop cap, **stop =
farther of {swing-body, 3rd-back}**, 3R target, 3:00 force-close, 1 trade/day,
spot, gross.

## Headline result — earlier entry helped

Window **2025-06-18 → 2026-06-19** · 248 basket days.

| | Pass 2.1 (A @ 9:30) | Baseline (A @ 10:00) |
|---|---|---|
| **Expectancy / trade** | **+0.187 R** | +0.146 R |
| Trades · no-trade days | 248 · 0 | 245 · 3 |
| Win rate | 36.3% | 34.7% |
| Target / stop / time | 21 / 59 / 21% | 20 / 62 / 18% |
| Max drawdown | **13.3 R** | 16.8 R |
| Longest losing streak | **7** | 15 |
| Long / short | 129 / 119 | 143 / 102 |

Moving the decision earlier to 9:30 **raised expectancy and cut both drawdown
and the worst losing streak** — consistent with the RFactor finding that the
capturable slice is largest early and exhausts through the morning. Still a
*single rising regime*, spot & gross, proxy-3R — necessary, not sufficient.

## Files

| File | What it is |
|------|-----------|
| `report.html` | **Charts + metrics** — verdict table, equity/drawdown curves, outcome split, R distribution, honesty notes. |
| `trade_log.html` | **Complete trade log** — all 248 trades in a live-filterable, sortable table (green = win, red = loss). |
| `run_meta.json` | Config (incl. the 9:30 overrides) + full metrics. |
| `filter_A/` | `metrics.csv` + full log (`trades.parquet` + `trades.csv`). |

## Regenerate (one command builds data + both HTML files)

```bash
python strategies/EntryGeometry/run_pass2.py --test-name pass2.1_filterA_0930 \
    --basket-lock 09:30 --entry-start 09:30 --filters A
```
