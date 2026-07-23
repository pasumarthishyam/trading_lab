# RFactor — Test Results

The RFactor strategy is a top-down intraday flow: **R-factor stock selection →
entry geometry → full trade**. Every test below is a stage of that pipeline, and
each gets **its own folder** so nothing clobbers anything else.

## Folder convention

```
<test_name>/
├── README.md          what the test asks, the headline result, a file guide
├── report.html        START HERE — interactive, self-contained; open in any browser
├── *.csv              human-readable tables (summary / metrics / distributions)
├── run_meta.json      config + window + provenance
└── substrate/ | filter_*/   machine-readable outputs reused by later stages
```

**View** = open `report.html` (no Jupyter/Colab). **Understand** = read its `README.md`.
**Build on** = load the `.parquet` files.

## Tests

| Test | Stage / question | Headline |
|------|------------------|----------|
| [`move_validation/`](move_validation/) | **Selection** — does R-rank pick ≥2% movers vs random? | Yes — top-10 lift +34–41pp, top-5 +39–46pp over a 39% base. |
| [`pass1_entry_geometry/`](pass1_entry_geometry/) | **Entry, Pass 1** — measure & audit the breakout geometry (no thresholds). | 3,978 triggers; `atr_ratio` p90≈1.6×; stop distances mapped. |
| [`pass2_trade_backtest/`](pass2_trade_backtest/) | **Trade, Pass 2** — 3R/1R full trade, spot, gross; filters A vs B, basket & entry at **10:00**. | Both positive: A **+0.146R**, B **+0.136R** per trade. |
| [`pass2.1_filterA_0930/`](pass2.1_filterA_0930/) | **Trade, Pass 2.1** — Filter A only, basket lock + entry moved to **9:30**. | **+0.187R** per trade — earlier entry beats 10:00 (lower DD, shorter streak). |
| [`pass2.2_rolling_be/`](pass2.2_rolling_be/) | **Trade, Pass 2.2** — rolling bucket (re-ranked every 15 min) + breakeven@2R; swing-K swept 2–5. | **K=4 best (+0.152R)**; higher K beats lower; breakeven looks like it caps winners. |
| [`daily_rank_leaderboard/`](daily_rank_leaderboard/) | **Reference** — top-10 bucket at every checkpoint every day, symbols linked to TradingView. | Browse/audit which stocks ranked where; shows the bucket churn. |

## Code locations

- Selection / move_validation: `strategies/RFactor/*.py`
- Entry geometry + trade backtests (pass1, pass2, pass2.1): `strategies/EntryGeometry/*.py`
  (results are written here under RFactor because they're the same strategy pipeline).

## Regenerating

```bash
# selection
python strategies/RFactor/run_backtest.py         &&  python strategies/RFactor/report.py
# entry geometry (pass 1)
python strategies/EntryGeometry/run_pass1.py      &&  python strategies/EntryGeometry/report.py
# trade backtest (pass 2, both filters @10:00)
python strategies/EntryGeometry/run_pass2.py
# variant 2.1 (filter A, 9:30) — one command builds data + both HTML files
python strategies/EntryGeometry/run_pass2.py --test-name pass2.1_filterA_0930 \
    --basket-lock 09:30 --entry-start 09:30 --filters A
```
