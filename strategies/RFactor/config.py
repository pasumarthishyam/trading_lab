"""
R-Factor Ranking -> Move Validation Backtest — Configuration
============================================================

Every knob lives here so a re-run is a one-line change.  Imported by the
engine, the driver, and the results notebook alike.

See ``strategies/RFactor/README.md`` for the full specification.
"""

from __future__ import annotations

from pathlib import Path

# Repo root: strategies/RFactor/config.py -> parents[2] == trading_lab/
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

CONFIG: dict = {
    # Intraday ranking checkpoints (IST wall-clock, "HH:MM").
    "checkpoints": ["09:25", "09:45", "10:00", "10:15", "10:30",
                    "10:45", "11:00", "11:15", "11:30"],
    "top_n": 10,                  # frozen leaderboard size; top 5 reported as subset
    "top_subset": 5,
    "rvol_lookback": 20,          # prior trading days for the R baseline (point-in-time)
    "move_threshold": 0.02,       # 2% from 09:15 open, either direction
    "move_reference": "09:15_open",
    "capture_bar": 0.0075,        # min capturable slice after a checkpoint (0.75%)
    "test_period_days": 250,      # most-recent N trading days used as test days
    "universe": "fo_manifest",    # F&O stocks from the manifest, NSE-EQ only
    "exclude_corp_action_window": True,  # drop stocks w/ a corp-action date in lookback..test day

    # Eligibility thresholds for "complete 1-min data for the day".
    "min_day_candles": 300,       # of 375 regular-session minutes
    "session_close_after": "15:20",  # last candle must reach at least this time
    "session_open": "09:15",
    "session_end": "15:30",
}

# ── Resolved paths ──────────────────────────────────────────────────
STOCKS_GLOB: str = (REPO_ROOT / "data" / "raw" / "stocks" / "*" / "1min.parquet").as_posix()
MANIFEST_PATH: Path = REPO_ROOT / "data" / "raw" / "stocks" / "_fo_universe_manifest.json"
CORP_ACTIONS_PATH: Path = REPO_ROOT / "data" / "raw" / "stocks" / "_corporate_actions.json"
CALENDAR_SYMBOL: str = "RELIANCE"  # liquid stock used to derive the trading-day calendar

# Results are organised one folder per test, so future RFactor tests slot in
# alongside without clobbering:
#   results/
#     README.md                     <- index of tests + the folder convention
#     move_validation/              <- this test
#       README.md                   <- question, headline, file guide, how to re-run
#       report.html                 <- START HERE (interactive, self-contained)
#       summary.csv                 <- the per-checkpoint verdict table
#       run_meta.json               <- config + window + provenance for the run
#       substrate/                  <- machine-readable outputs for downstream tests
#         picks.parquet
#         universe_daily.parquet
RESULTS_DIR: Path = REPO_ROOT / "strategies" / "RFactor" / "results"
TEST_NAME: str = "move_validation"
TEST_DIR: Path = RESULTS_DIR / TEST_NAME
SUBSTRATE_DIR: Path = TEST_DIR / "substrate"

PICKS_PATH: Path = SUBSTRATE_DIR / "picks.parquet"
UNIVERSE_DAILY_PATH: Path = SUBSTRATE_DIR / "universe_daily.parquet"
SUMMARY_PATH: Path = TEST_DIR / "summary.csv"
RUN_META_PATH: Path = TEST_DIR / "run_meta.json"
REPORT_PATH: Path = TEST_DIR / "report.html"
