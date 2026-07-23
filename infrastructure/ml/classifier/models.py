"""
Gradient-Boosted Model Wrappers
===============================

Thin, uniform wrappers over LightGBM and XGBoost so the walk-forward
driver can treat them interchangeably.

Two choices worth stating explicitly:

**Missing values are passed through, not imputed.**  Both libraries learn
a default split direction for NaN.  A missing ``beta_60d`` means "this
symbol has under 60 bars of history", which is information; median
imputation would destroy it.

**Class imbalance is handled by weight, not resampling.**  Positives run
5-10% of rows.  Oversampling in a time-series panel duplicates rows that
are already serially correlated and inflates the effective sample size;
``scale_pos_weight`` leaves the data alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from infrastructure.ml.classifier import config as C

logger = logging.getLogger(__name__)

LGBM_STATIC: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "n_jobs": -1,
    "seed": C.RANDOM_SEED,
    "force_row_wise": True,
}

XGB_STATIC: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "nthread": -1,
    "seed": C.RANDOM_SEED,
    "verbosity": 0,
}


@dataclass
class FittedModel:
    """A trained booster plus what it took to train it."""

    kind: str
    booster: Any
    features: list[str]
    best_iteration: int
    params: dict = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probability for *X*."""
        if self.kind == "lgbm":
            return self.booster.predict(
                X[self.features], num_iteration=self.best_iteration
            )
        import xgboost as xgb

        dm = xgb.DMatrix(X[self.features], feature_names=self.features)
        return self.booster.predict(dm, iteration_range=(0, self.best_iteration))


def _scale_pos_weight(y: np.ndarray) -> float:
    """Negative-to-positive ratio, floored so it cannot explode."""
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos <= 0:
        return 1.0
    return float(np.clip(neg / pos, 1.0, 50.0))


def train_lgbm(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    features: list[str],
    params: dict | None = None,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    num_boost_round: int = C.MAX_BOOST_ROUNDS,
) -> FittedModel:
    """Train a LightGBM binary classifier with optional early stopping."""
    import lightgbm as lgb

    p = dict(LGBM_STATIC)
    p.update(params or {})
    p["scale_pos_weight"] = _scale_pos_weight(y_fit)

    dtrain = lgb.Dataset(X_fit[features], label=y_fit, feature_name=features)
    valid_sets, callbacks = [], [lgb.log_evaluation(period=0)]
    if X_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
        valid_sets = [lgb.Dataset(X_val[features], label=y_val, reference=dtrain)]
        callbacks.append(
            lgb.early_stopping(C.EARLY_STOPPING_ROUNDS, verbose=False)
        )

    booster = lgb.train(
        p, dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )
    best = booster.best_iteration or num_boost_round
    return FittedModel("lgbm", booster, features, best, p)


def train_xgb(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    features: list[str],
    params: dict | None = None,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    num_boost_round: int = C.MAX_BOOST_ROUNDS,
) -> FittedModel:
    """Train an XGBoost binary classifier with optional early stopping."""
    import xgboost as xgb

    p = dict(XGB_STATIC)
    p.update(params or {})
    p["scale_pos_weight"] = _scale_pos_weight(y_fit)

    dtrain = xgb.DMatrix(X_fit[features], label=y_fit, feature_names=features)
    evals, early = [], None
    if X_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
        dval = xgb.DMatrix(X_val[features], label=y_val, feature_names=features)
        evals = [(dval, "val")]
        early = C.EARLY_STOPPING_ROUNDS

    booster = xgb.train(
        p, dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early,
        verbose_eval=False,
    )
    best = getattr(booster, "best_iteration", None)
    best = (best + 1) if best is not None else num_boost_round
    return FittedModel("xgb", booster, features, best, p)


TRAINERS = {"lgbm": train_lgbm, "xgb": train_xgb}


def train(
    kind: str,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    features: list[str],
    params: dict | None = None,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
) -> FittedModel:
    """Dispatch to the trainer for *kind* (``"lgbm"`` or ``"xgb"``)."""
    if kind not in TRAINERS:
        raise ValueError(f"Unknown model kind {kind!r}; expected one of {list(TRAINERS)}")
    return TRAINERS[kind](X_fit, y_fit, features, params, X_val, y_val)


def native_importance(model: FittedModel) -> pd.DataFrame:
    """Gain-based importance from the booster itself.

    Kept alongside SHAP as a sanity check: when the two disagree sharply
    on a feature, that feature is usually interacting rather than acting
    on its own.
    """
    if model.kind == "lgbm":
        gain = model.booster.feature_importance(importance_type="gain")
        df = pd.DataFrame({"feature": model.features, "gain": gain})
    else:
        raw = model.booster.get_score(importance_type="total_gain")
        df = pd.DataFrame(
            {"feature": model.features,
             "gain": [raw.get(f, 0.0) for f in model.features]}
        )
    total = df["gain"].sum()
    df["gain_pct"] = df["gain"] / total if total > 0 else 0.0
    return df.sort_values("gain", ascending=False).reset_index(drop=True)
