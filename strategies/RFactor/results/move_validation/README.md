# RFactor · Move Validation

**Question:** When a stock is ranked among the top by **R-factor** (time-of-day-
normalised RVOL) at an intraday checkpoint, how reliably is it a stock that
moved **≥2% from its 9:15 open** that day — and how much better is that than a
random F&O stock? Validates the *signal on spot price*, not P&L.

## Headline result

Window **2025-06-18 → 2026-06-19** · 248 trading days · 211 F&O symbols.

- **Base rate** (any eligible stock moves ≥2% intraday): **39.4%**
- **Top-10 by R** hit rate **73–80%** → **lift +34 to +41 pp**
- **Top-5 by R** hit rate **79–85%** → **lift +39 to +46 pp** (sharper than top-10 at every checkpoint)
- Capturable slice falls ~1.2% → 0.75% and churn 23% → 6% across the morning → **enter earlier**.

**Verdict: the selection signal validates.** (Lift, not raw hit rate, is the verdict — see the report's "read before concluding" note.)

## Files in this folder

| File | What it is |
|------|-----------|
| `report.html` | **Open this first.** Interactive charts of all 7 outputs + a clickable table of contents. Self-contained; no kernel/Colab. |
| `summary.csv` | The verdict table — one row per checkpoint: top-N hit, base rate, lift, capturable, churn. |
| `run_meta.json` | Exactly how this run was produced: config, window, dropped days, base rate, row counts. |
| `substrate/picks.parquet` | One row per (day, checkpoint, top-10 stock) — the Section-5 schema. The reusable substrate downstream tests build on. |
| `substrate/universe_daily.parquet` | One row per (symbol, day): eligibility + from-9:15 outcomes. Base-rate / magnitude substrate. |

Dropped days: `2025-10-21` (Diwali Muhurat — special session) and `2026-06-22`
(partial download). Both excluded automatically; recorded in `run_meta.json`.

## Regenerate

```bash
python strategies/RFactor/run_backtest.py    # -> substrate/, summary.csv, run_meta.json
python strategies/RFactor/report.py --open   # -> report.html (and opens it)
```

Full method, definitions, and interpretation guidance: `strategies/RFactor/README.md`.
