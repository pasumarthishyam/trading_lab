"""
Optuna Hyperparameter Optimisation
==================================

Searches LightGBM / XGBoost hyperparameters against **mean PR-AUC over
purged inner walk-forward folds**, not a single random holdout.

The protocol that keeps the reported lift honest
------------------------------------------------

Optuna evaluates hundreds of configurations.  If the score guiding that
search comes from the same blocks used to report performance, the best
trial is partly selected on noise in those blocks, and the reported lift
is optimistic — the search itself becomes a leak.

So the timeline is cut once, up front:

* the first :data:`config.DEV_FRACTION` of dates is the **development
  period** — Optuna runs here, over its own purged inner folds;
* the remainder is the **evaluation period**, which the search never
  touches.  Winning parameters are frozen and walk-forward tested there.

Every inner fold still carries the full purge and embargo, so even within
the development period the search is not scoring itself on leaked rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from infrastructure.ml.classifier import config as C
from infrastructure.ml.classifier import cv, evaluate, models

logger = logging.getLogger(__name__)

SEARCH_SPACES = {
    "lgbm": C.LGBM_SEARCH_SPACE,
    "xgb": C.XGB_SEARCH_SPACE,
}


@dataclass
class TuningResult:
    """Outcome of one Optuna study."""

    kind: str
    best_params: dict
    best_value: float
    n_trials: int
    trials: pd.DataFrame

    def summary(self) -> str:
        return (
            f"{self.kind}: best inner PR-AUC {self.best_value:.5f} "
            f"over {self.n_trials} trials"
        )


def suggest_params(trial, kind: str) -> dict[str, Any]:
    """Sample one configuration from the space declared in config."""
    space = SEARCH_SPACES[kind]
    params: dict[str, Any] = {}
    for name, spec in space.items():
        dtype, lo, hi, log = spec
        if dtype == "int":
            params[name] = trial.suggest_int(name, int(lo), int(hi), log=log)
        else:
            params[name] = trial.suggest_float(name, float(lo), float(hi), log=log)
    # LightGBM's num_leaves and max_depth interact; keep leaves within
    # what the depth can actually support so trials are not silently
    # equivalent to one another.
    if kind == "lgbm" and "max_depth" in params and "num_leaves" in params:
        params["num_leaves"] = int(
            min(params["num_leaves"], 2 ** params["max_depth"] - 1)
        )
    return params


def _score_params(
    panel: pd.DataFrame,
    features: list[str],
    folds: list[cv.Fold],
    kind: str,
    params: dict,
) -> float:
    """Mean PR-AUC of *params* across the given inner folds."""
    scores: list[float] = []
    for fold in folds:
        train = panel.iloc[fold.train_idx]
        test = panel.iloc[fold.test_idx]
        y_train = train["y"].to_numpy(dtype=float)
        y_test = test["y"].to_numpy(dtype=float)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        fit_idx, val_idx = cv.chronological_holdout(panel["date"], fold.train_idx)
        fit, val = panel.iloc[fit_idx], panel.iloc[val_idx]
        model = models.train(
            kind,
            fit, fit["y"].to_numpy(dtype=float), features, params,
            X_val=val, y_val=val["y"].to_numpy(dtype=float),
        )
        preds = model.predict(test)
        scores.append(evaluate.score(y_test, preds).pr_auc)

    if not scores:
        return float("-inf")
    return float(np.nanmean(scores))


def tune(
    panel: pd.DataFrame,
    features: list[str],
    kind: str,
    dev_range: tuple[pd.Timestamp, pd.Timestamp],
    n_trials: int = C.OPTUNA_TRIALS,
    timeout: Optional[int] = C.OPTUNA_TIMEOUT_SEC,
    callback: Optional[Callable] = None,
) -> TuningResult:
    """Run an Optuna study for *kind* over the development period.

    Parameters
    ----------
    panel : pd.DataFrame
        Labelled panel, restricted to rows with a valid target.
    features : list[str]
        Model feature names.
    kind : str
        ``"lgbm"`` or ``"xgb"``.
    dev_range : (Timestamp, Timestamp)
        Development window; the study never sees dates outside it.
    n_trials, timeout : optional
        Search budget.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    splitter = cv.PurgedWalkForward(
        n_folds=C.N_INNER_FOLDS,
        min_train_dates=max(60, C.MIN_TRAIN_DATES // 2),
    )
    folds = list(splitter.split(panel["date"], date_range=dev_range))
    for fold in folds:
        cv.assert_no_leakage(fold, panel["date"], C.LABEL_HORIZON_DAYS)
    logger.info(
        "Tuning %s over %d inner folds within %s..%s",
        kind, len(folds), dev_range[0].date(), dev_range[1].date(),
    )

    def objective(trial) -> float:
        params = suggest_params(trial, kind)
        return _score_params(panel, features, folds, kind, params)

    sampler = optuna.samplers.TPESampler(seed=C.RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[callback] if callback else None,
        show_progress_bar=False,
    )

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    result = TuningResult(
        kind=kind,
        best_params=study.best_params,
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        trials=trials_df,
    )
    logger.info(result.summary())
    logger.info("Best %s params: %s", kind, result.best_params)
    return result


def dev_eval_split(dates: pd.Series, fraction: float = C.DEV_FRACTION):
    """Cut the timeline into development and evaluation windows.

    Returns ``(dev_range, eval_range)`` as inclusive ``(lo, hi)`` date
    pairs.  The evaluation window starts one full embargo after the
    development window ends, so the split itself does not leak.
    """
    unique = np.sort(pd.to_datetime(pd.Series(dates)).unique())
    n = len(unique)
    cut = int(n * fraction)
    gap = C.EMBARGO_DAYS + C.LABEL_HORIZON_DAYS
    dev = (pd.Timestamp(unique[0]), pd.Timestamp(unique[cut - 1]))
    eval_lo = min(cut + gap, n - 1)
    ev = (pd.Timestamp(unique[eval_lo]), pd.Timestamp(unique[-1]))
    logger.info(
        "Timeline split — dev %s..%s (%d dates) | eval %s..%s (%d dates)",
        dev[0].date(), dev[1].date(), cut,
        ev[0].date(), ev[1].date(), n - eval_lo,
    )
    return dev, ev
