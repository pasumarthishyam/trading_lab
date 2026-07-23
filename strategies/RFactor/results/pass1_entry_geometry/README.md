# Pass 1 · Entry Geometry Measurement & Audit

**Measure and audit only** — replays the entry geometry and logs the raw
candle/swing/stop measurements. No thresholds, no stops chosen, no P&L. The
"huge candle" cutoff, swing `K`, and stop rule are *read off* these
distributions (and were, to configure Pass 2).

Window **2025-06-18 → 2026-06-19** · 248 days · **3,978 triggers** (~16/day).

## Files

| File | What it is |
|------|-----------|
| `report.html` | **Open first.** Huge-candle read-off, stop-distance distributions, direction/momentum context, per-day counts, and the audit sample. |
| `distributions.csv` | Percentiles of `atr_ratio`, `range_ratio`, and the four stop-distance candidates. |
| `audit_sample.csv` | Extreme + random triggers to open on TradingView and confirm the geometry by eye. |
| `run_meta.json` | Config + window + provenance. |
| `substrate/triggers.parquet` | One row per trigger (§7). **Pass 2 builds directly on this.** |

## What Pass 2 read off this

- Huge cutoff **2× ATR** (top decile of `atr_ratio` ≈ the thrusts).
- Stop = **farther of {swing-body, 3rd-back}** (swing-body alone can be ~0% — too tight).
- `swing_K = 2`, ATR = Wilder(14) — validated by the audit.

Regenerate: `python strategies/EntryGeometry/run_pass1.py && python strategies/EntryGeometry/report.py`.
Full method: `strategies/EntryGeometry/README.md`.
