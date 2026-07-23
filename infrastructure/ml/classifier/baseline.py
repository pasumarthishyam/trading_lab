"""
Statistical Baselines
=====================

A gradient-boosted model with 50-odd features will always beat *nothing*.
The question is whether it beats the cheap, well-known statistics that
already predict volatility, because those are what a trading desk would
use if the model did not exist.

Three baselines, in increasing order of strength:

``prior``
    Predict the training base rate for every row.  Establishes the floor:
    PR-AUC equals the base rate, lift equals 1.0 by construction.

``vol_cluster``
    Volatility clusters — a large move today makes a large move tomorrow
    more likely (Engle's ARCH observation, and the single most robust
    fact about daily returns).  Scores each row by today's absolute
    return in units of trailing ATR.  No fitting at all.

``logit``
    L2-regularised logistic regression on a handful of classic volatility
    and volume features.  This is the honest bar: a linear model on the
    obvious inputs.  Beating ``prior`` is trivial, beating ``vol_cluster``
    is expected, beating ``logit`` is the result worth reporting.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from infrastructure.ml.classifier import config as C

logger = logging.getLogger(__name__)

# Features the logistic baseline is allowed to see — the ones a
# volatility analyst would reach for first.
LOGIT_FEATURES: list[str] = [
    "abs_ret_1d",
    "atr_pct",
    "range_pct",
    "vol_ratio_short_long",
    "volume_ratio_20d",
    "abs_ret_zscore_20d",
    "vix_level",
]


def predict_prior(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Constant prediction at the training base rate."""
    rate = float(np.nanmean(y_train)) if len(y_train) else 0.5
    return np.full(n_test, rate, dtype=float)


def predict_vol_cluster(test: pd.DataFrame) -> np.ndarray:
    """Score by today's move size in trailing-ATR units.

    Deliberately unfitted — it is a ranking, not a probability, which is
    fine for PR-AUC and precision@k but makes Brier meaningless.  The
    squashing below maps it to (0, 1) so calibration metrics still run
    without pretending the numbers are calibrated.
    """
    ratio = (test["abs_ret_1d"] / test["atr_pct"].replace(0.0, np.nan)).to_numpy()
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    return 1.0 - np.exp(-ratio / 2.0)


def _logit_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=C.RANDOM_SEED,
                ),
            ),
        ]
    )


def fit_predict_logit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str] | None = None,
) -> np.ndarray:
    """Fit the logistic baseline on *train*, score *test*."""
    features = features or [f for f in LOGIT_FEATURES if f in train.columns]
    missing = set(LOGIT_FEATURES) - set(features)
    if missing:
        logger.warning("Logit baseline missing features: %s", sorted(missing))

    pipe = _logit_pipeline()
    pipe.fit(train[features], y_train)
    return pipe.predict_proba(test[features])[:, 1]


def all_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
) -> dict[str, np.ndarray]:
    """Every baseline's predictions for one fold, keyed by name."""
    return {
        "prior": predict_prior(y_train, len(test)),
        "vol_cluster": predict_vol_cluster(test),
        "logit": fit_predict_logit(train, test, y_train),
    }
