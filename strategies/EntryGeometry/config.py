"""
Pass 1 — Entry Geometry Measurement & Audit — Configuration
===========================================================

Measure-and-audit only.  No thresholds, no stop choice, no exits, no P&L.
The "huge candle" cutoff, swing ``K`` and the stop rule are *outputs* of
this pass, read off the distributions it produces.

Every knob is explicit here so a re-run is a one-line change.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

# Repo root: strategies/EntryGeometry/config.py -> parents[2]
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

CONFIG: dict = {
    "entry_timeframe": "5min",       # swings, triggers, ATR all on 5-min
    "basket_lock_time": "10:00",     # freeze R-rank, take top N
    "basket_size": 4,
    "entry_window_start": "10:00",
    "entry_window_end": "12:30",     # inclusive (the 12:30 candle counts)
    "swing_K": 2,                    # bars each side to confirm a pivot (AUDITED)
    "atr_period": 14,                # Wilder/RMA, matches TradingView default
    "third_candle_back": 3,          # offset for the 3rd-candle-back stop component
    "test_period_days": 250,         # ~1 year of trading days
    "universe": "fo_manifest",       # NSE-EQ only; corp-action lookback excluded
    "rvol_lookback": 20,             # R-factor baseline (reused from RFactor)

    # Session bounds (regular session) used for session-scoped swing levels.
    "session_open": "09:15",
    "session_end": "15:30",
}

# ── Pass 2 — Full Trade Backtest (spot, gross) ──────────────────────
# Everything FIXED this run (no sweeps). Only the momentum filter varies,
# as two separate full runs (A, B) that are compared, never blended.
CONFIG_P2: dict = {
    **CONFIG,                        # basket/entry/swing/atr all inherited, unchanged
    "huge_atr_mult": 2.0,            # skip trigger if breakout candle > 2x ATR
    "stop_max_pct": 0.01,            # skip trade if structural stop > 1% from entry
    "stop_rule": "farther",          # stop = farther of {swing-body, 3rd-candle-back}
    "use_strong_candle_exception": False,   # omitted this clean run
    "target_R": 3.0,                 # fixed 3R target
    "break_even": None,              # none
    "force_close": "15:00",          # hard 3:00 PM force-close (exit at 15:00 open)
    "trades_per_day": 1,             # first fully-qualifying trigger wins
    "resolution_tf": "5min",         # resolve target/stop on 5-min; stop-first ties
    "filters": ["A", "B"],           # A = 9:15-extreme-held; B = side-of-open
}

# ── Pass 2.2 — rolling basket + breakeven + K sweep ─────────────────
# Rolling checkpoints: re-rank the top-4 every 15 min across the entry window.
ROLLING_CHECKPOINTS: list[str] = [
    "09:30", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00",
    "11:15", "11:30", "11:45", "12:00", "12:15", "12:30",
]
CONFIG_P2_2: dict = {
    **CONFIG_P2,
    "basket_mode": "rolling",        # re-rank bucket at each rolling checkpoint
    "breakeven_R": 2.0,              # move stop to entry once +2R touched
    "entry_window_start": "09:30",
    "entry_window_end": "12:30",
    "filters": ["A"],                # Filter A only
    "k_sweep": [2, 3, 4, 5],         # swing_K values to compare
}

# ── Resolved paths (organised one folder per test) ──────────────────
#   results/
#     README.md
#     pass1_entry_geometry/   report.html, distributions.csv, audit_sample.csv,
#                             run_meta.json, substrate/triggers.parquet
#     pass2_trade_backtest/   report.html, run_meta.json,
#                             filter_A/{metrics.csv, trades.parquet, trades.csv}
#                             filter_B/{metrics.csv, trades.parquet, trades.csv}
STOCKS_ROOT: Path = REPO_ROOT / "data" / "raw" / "stocks"
# This entry/trade work is part of the RFactor strategy (R-factor selection ->
# entry geometry -> trades), so all results live under RFactor's results/, one
# folder per test, alongside move_validation.
RESULTS_DIR: Path = REPO_ROOT / "strategies" / "RFactor" / "results"

# Pass 1
P1_DIR: Path = RESULTS_DIR / "pass1_entry_geometry"
P1_SUBSTRATE: Path = P1_DIR / "substrate"
TRIGGERS_PATH: Path = P1_SUBSTRATE / "triggers.parquet"
AUDIT_SAMPLE_PATH: Path = P1_DIR / "audit_sample.csv"
DISTRIB_PATH: Path = P1_DIR / "distributions.csv"
REPORT_PATH: Path = P1_DIR / "report.html"
RUN_META_PATH: Path = P1_DIR / "run_meta.json"


# Pass 2 — parameterised per named variant so each backtest gets its own
# folder (pass2_trade_backtest, pass2.1_filterA_0930, ...).
def p2_paths(test_name: str = "pass2_trade_backtest") -> SimpleNamespace:
    d = RESULTS_DIR / test_name
    return SimpleNamespace(
        dir=d,
        report=d / "report.html",
        tradelog=d / "trade_log.html",
        run_meta=d / "run_meta.json",
        filter_dir=lambda f: d / f"filter_{f}",
    )


def five_min_path(symbol: str) -> Path:
    return STOCKS_ROOT / symbol / "5min.parquet"
