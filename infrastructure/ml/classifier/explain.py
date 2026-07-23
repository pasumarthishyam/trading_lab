"""
SHAP Feature Attribution
========================

Global feature importance via ``shap.TreeExplainer``, which computes
exact Shapley values for tree ensembles.

Why SHAP rather than the booster's own gain
-------------------------------------------

Split-gain importance is biased toward high-cardinality continuous
features — they simply offer more places to cut, so they accumulate gain
whether or not they carry signal.  It is also computed on the *training*
data, so a feature the model overfits looks important.

SHAP values here are computed on **held-out test rows** and are additive
per prediction, so ``mean |SHAP|`` answers the question actually being
asked: how much did this feature move the predictions the model made on
data it had never seen?

Both are reported.  Sharp disagreement between them is diagnostic — a
feature high on gain and low on SHAP is usually being used to memorise
the training block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from infrastructure.ml.classifier import config as C
from infrastructure.ml.classifier.models import FittedModel

logger = logging.getLogger(__name__)


@dataclass
class ShapResult:
    """Global SHAP attribution for one fitted model."""

    importance: pd.DataFrame          # feature, mean_abs_shap, mean_shap, share
    values: np.ndarray                # (n_sample, n_features), column order = features
    sample: pd.DataFrame              # the rows explained
    features: list[str]               # column order of `values`
    expected_value: float

    def top(self, n: int = 20) -> pd.DataFrame:
        return self.importance.head(n)

    def column(self, feature: str) -> np.ndarray:
        """SHAP values for one feature, by name rather than position."""
        return self.values[:, self.features.index(feature)]


def _sample_rows(
    X: pd.DataFrame, n: int, seed: int = C.RANDOM_SEED
) -> pd.DataFrame:
    """Draw an evaluation sample, preserving chronological spread.

    A uniform random draw over a walk-forward panel would over-weight
    whichever regime happens to have the most rows; sampling within date
    strata keeps the explanation representative of the whole test period.
    """
    if len(X) <= n:
        return X
    if "date" not in X.columns:
        return X.sample(n=n, random_state=seed)

    rng = np.random.default_rng(seed)
    per_date = max(1, n // max(X["date"].nunique(), 1))
    picked = (
        X.groupby("date", sort=False, group_keys=False)
        .apply(
            lambda d: d.sample(min(len(d), per_date),
                               random_state=int(rng.integers(0, 2**31))),
            include_groups=True,
        )
    )
    if len(picked) > n:
        picked = picked.sample(n=n, random_state=seed)
    return picked


def explain(
    model: FittedModel,
    X: pd.DataFrame,
    sample_size: int = C.SHAP_SAMPLE_SIZE,
) -> ShapResult:
    """Compute global SHAP importance for *model* over held-out rows *X*."""
    import shap

    sample = _sample_rows(X, sample_size)
    features = model.features
    Xs = sample[features]

    explainer = shap.TreeExplainer(model.booster)
    values = explainer.shap_values(Xs, check_additivity=False)

    # LightGBM binary returns a list of two arrays on some versions and a
    # single array on others; XGBoost returns one.  Normalise to the
    # positive class.
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = float(np.ravel(expected)[-1])

    mean_abs = np.abs(values).mean(axis=0)
    total = mean_abs.sum()
    importance = (
        pd.DataFrame(
            {
                "feature": features,
                "mean_abs_shap": mean_abs,
                "mean_shap": values.mean(axis=0),
                "share": mean_abs / total if total > 0 else 0.0,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance["cumulative_share"] = importance["share"].cumsum()

    logger.info(
        "SHAP computed on %d held-out rows — top 5: %s",
        len(sample), ", ".join(importance["feature"].head(5)),
    )
    return ShapResult(importance, values, sample, list(features), float(expected))


def direction_table(result: ShapResult, top_n: int = 15) -> pd.DataFrame:
    """Sign of each top feature's relationship with the prediction.

    Correlates a feature's value against its own SHAP value across the
    sample.  Positive means "higher values push the prediction toward a
    significant move"; near zero means the effect is non-monotonic, which
    is exactly what a tree model is for and what a linear baseline
    cannot represent.
    """
    rows = []
    for f in result.importance["feature"].head(top_n):
        x = result.sample[f].to_numpy(dtype=float)
        s = result.column(f)
        finite = np.isfinite(x) & np.isfinite(s)
        if finite.sum() < 10 or np.std(x[finite]) == 0 or np.std(s[finite]) == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(x[finite], s[finite])[0, 1])
        rows.append({
            "feature": f,
            "value_shap_corr": corr,
            "effect": (
                "monotone +" if corr > 0.5
                else "monotone -" if corr < -0.5
                else "non-monotone"
            ),
        })
    return pd.DataFrame(rows)
