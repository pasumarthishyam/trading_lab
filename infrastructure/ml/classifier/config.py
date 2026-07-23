"""
Classifier Configuration
========================

Every tunable constant for the significant-move classification pipeline
lives here — no magic numbers scattered through the modules.

The pipeline predicts, at the close of day ``t``, whether day ``t+1``
will produce an abnormally large absolute move for that symbol, where
"abnormally large" is scaled by the symbol's own trailing volatility so
the label means the same thing for a 2%-a-day stock and a 6%-a-day one.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────
#   this file : trading_lab/infrastructure/ml/classifier/config.py
#   parent(4) : trading_lab/                              <-- repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
STOCKS_ROOT: Path = REPO_ROOT / "data" / "raw" / "stocks"
UNIVERSE_MANIFEST: Path = STOCKS_ROOT / "_fo_universe_manifest.json"
RESULTS_ROOT: Path = REPO_ROOT / "data" / "processed" / "ml_classifier"
CACHE_PATH: Path = RESULTS_ROOT / "panel.parquet"

# Parquet timestamps are TIMESTAMP WITH TIME ZONE.  Convert with an
# explicit zone so the build is identical on any machine, rather than
# relying on DuckDB's session TimeZone setting.
MARKET_TZ: str = "Asia/Kolkata"

# ── Label definition ────────────────────────────────────────────────
# y = 1 when |return over the next LABEL_HORIZON_DAYS| exceeds
# LABEL_VOL_MULTIPLE x a trailing volatility estimate known at the close
# of day t.
LABEL_HORIZON_DAYS: int = 1
LABEL_VOL_MULTIPLE: float = 2.0

# Which trailing volatility sets the per-symbol threshold.
#
#   "rvol_20d"  close-to-close return volatility  (default)
#   "atr_pct"   average true range as % of close
#
# These are NOT interchangeable.  ATR measures the intraday high-low
# excursion, which on this universe averages ~4% a day; a close-to-close
# return is systematically smaller than the range that contains it, so
# thresholding a close-to-close return at 2x ATR demands an ~8.5% single
# session move and yields a ~1.9% base rate — sparse enough that folds
# start containing only a handful of positives.  Scaling by the
# volatility of the same quantity being thresholded gives a ~5% base
# rate, which is the rare-event regime the metrics in `evaluate` are
# designed for.  Switch to "atr_pct" for a deliberately rarer, more
# extreme target.
LABEL_SCALE_FEATURE: str = "rvol_20d"
# Trailing window for the ATR% variant.
LABEL_ATR_WINDOW: int = 14
# A symbol needs this many bars of history before its labels are usable.
MIN_HISTORY_BARS: int = 60

# ── Feature windows ─────────────────────────────────────────────────
RETURN_WINDOWS: tuple[int, ...] = (1, 5, 10, 20)
VOL_SHORT_WINDOW: int = 5
VOL_LONG_WINDOW: int = 20
VOLUME_WINDOWS: tuple[int, ...] = (5, 20)
ATR_WINDOW: int = 14
SMA_WINDOWS: tuple[int, ...] = (20, 50)
BETA_WINDOW: int = 60
# Fractal pivot detection: a pivot high needs SWING_FRACTAL_K bars on
# each side that are strictly lower (and vice versa for pivot lows).
SWING_FRACTAL_K: int = 3
SWING_STRUCTURE_WINDOW: int = 20
# Minimum observations before any trailing statistic is trusted.
ROLLING_MIN_PERIODS: int = 10

# Intraday features are aggregated from this timeframe.  15min gives 25
# bars per session — enough shape for momentum/VWAP features at a
# fraction of the I/O of the 1min files.
INTRADAY_TIMEFRAME: str = "15min"
# Regular NSE equity session, used to bucket open/close activity.
SESSION_OPEN: str = "09:15"
FIRST_HOUR_END: str = "10:15"
LAST_HOUR_START: str = "14:30"
SESSION_CLOSE: str = "15:30"

# Volatility-regime label: trailing realised vol is bucketed against the
# symbol's own expanding history at these quantiles (ordinal encoded).
VOL_REGIME_QUANTILES: tuple[float, ...] = (0.25, 0.50, 0.75)

# ── Walk-forward cross-validation ───────────────────────────────────
# Splits are made on DATES, not row positions: this is a cross-sectional
# panel where ~214 symbols share every date, so a row-based split would
# put the same date on both sides of the boundary.
N_OUTER_FOLDS: int = 6
# Purge removes training rows whose label window overlaps the test block.
# It must be >= LABEL_HORIZON_DAYS.
PURGE_DAYS: int = LABEL_HORIZON_DAYS
# Embargo drops training dates immediately before the test block beyond
# the purge, to blunt the serial correlation that rolling features carry
# across the boundary.
EMBARGO_DAYS: int = 10
# Expanding training window (True) vs fixed-length rolling window.
EXPANDING_TRAIN: bool = True
# Minimum trading dates in a training block for a fold to be usable.
MIN_TRAIN_DATES: int = 250

# ── Tuning / evaluation protocol ────────────────────────────────────
# Hyperparameters are searched on the first DEV_FRACTION of the timeline
# using inner purged folds, then frozen and evaluated on the untouched
# remainder.  This keeps the reported lift free of tuning contamination.
DEV_FRACTION: float = 0.60
N_INNER_FOLDS: int = 3
OPTUNA_TRIALS: int = 40
OPTUNA_TIMEOUT_SEC: int | None = None
RANDOM_SEED: int = 42

# Fraction of the training block held out (chronologically, purged) for
# early stopping inside each model fit.
EARLY_STOPPING_FRACTION: float = 0.15
EARLY_STOPPING_ROUNDS: int = 100
MAX_BOOST_ROUNDS: int = 2000

# Precision@k is reported at these top-k rates of the test block.
PRECISION_AT_K: tuple[float, ...] = (0.01, 0.05, 0.10)

# ── Hyperparameter search space ─────────────────────────────────────
# Consumed by tuning.suggest_params; ranges are deliberately conservative
# (shallow trees, strong regularisation) because the signal-to-noise
# ratio on daily price data is low and deep trees memorise folds.
LGBM_SEARCH_SPACE: dict[str, tuple] = {
    "learning_rate":     ("float", 0.01, 0.20, True),
    "num_leaves":        ("int", 15, 255, False),
    "max_depth":         ("int", 3, 10, False),
    "min_child_samples": ("int", 50, 1000, False),
    "subsample":         ("float", 0.5, 1.0, False),
    "colsample_bytree":  ("float", 0.4, 1.0, False),
    "reg_alpha":         ("float", 1e-4, 10.0, True),
    "reg_lambda":        ("float", 1e-4, 10.0, True),
}

XGB_SEARCH_SPACE: dict[str, tuple] = {
    "learning_rate":     ("float", 0.01, 0.20, True),
    "max_depth":         ("int", 3, 10, False),
    "min_child_weight":  ("float", 1.0, 100.0, True),
    "subsample":         ("float", 0.5, 1.0, False),
    "colsample_bytree":  ("float", 0.4, 1.0, False),
    "gamma":             ("float", 1e-4, 5.0, True),
    "reg_alpha":         ("float", 1e-4, 10.0, True),
    "reg_lambda":        ("float", 1e-4, 10.0, True),
}

# Rows sampled for the SHAP TreeExplainer pass (full panel is too large
# to explain in one go; the sample is drawn from held-out test folds).
SHAP_SAMPLE_SIZE: int = 20_000
