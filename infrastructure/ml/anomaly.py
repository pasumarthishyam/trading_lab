"""
Isolation Forest Anomaly Detection
==================================

Flags abnormal ``(stock, day)`` sessions across the full F&O universe
using an unsupervised Isolation Forest over 13 scale-free session
features.

The model is never told about corporate actions.  It is *validated*
against the four registered events in
``data/raw/stocks/_corporate_actions.json`` — a genuine held-out
ground truth, since splits and demergers produce extreme
open/prev-close discontinuities that any competent outlier detector
must isolate.

Interpretability comes from permutation importance computed directly
on the anomaly score: each feature is shuffled and the mean absolute
shift in ``decision_function`` output is measured.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from infrastructure.ml.dataset import (
    REPO_ROOT,
    STOCK_FEATURE_COLUMNS,
    build_stock_day_features,
)

logger = logging.getLogger(__name__)

KNOWN_ACTIONS_PATH: Path = REPO_ROOT / "data" / "raw" / "stocks" / "_corporate_actions.json"

# Default anomaly budget.  0.5% of ~333K sessions ≈ 1,650 flagged days,
# which sits inside the 0.1%–2% sanity band: tight enough that a flag
# means something, loose enough to catch a genuine tail.  Pass
# ``contamination="auto"`` to defer to sklearn's own offset instead.
DEFAULT_CONTAMINATION: float = 0.005

# Permutation importance is estimated on a subsample — 50K rows over
# 3 repeats is ample for a stable ranking and keeps the pass fast.
_PERM_SAMPLE_SIZE: int = 50_000
_PERM_REPEATS: int = 3
# The flagged subset is small (~1.5K rows), so it affords more repeats.
_PERM_REPEATS_FOCUSED: int = 5
_MIN_ROWS_FOR_FOCUSED_IMPORTANCE: int = 100


@dataclass
class AnomalyResult:
    """Everything produced by an Isolation Forest run."""

    labels: pd.DataFrame
    """``symbol, date, anomaly_score, is_anomaly`` for every session."""

    model: IsolationForest
    scaler: StandardScaler
    feature_names: list[str]

    feature_importances: pd.Series
    """Permutation importance measured **on the flagged sessions** —
    i.e. what actually drives the anomaly verdicts.  Normalised to
    sum to 1, descending."""

    feature_importances_global: pd.Series
    """Permutation importance over a random cross-section of all
    sessions — the model's overall sensitivity.  Necessarily flatter,
    since normal rows look normal on every axis at once."""

    known_events: pd.DataFrame
    """Per-event validation detail for the registered corporate actions."""

    known_recall: float
    """Fraction of registered corporate actions flagged as anomalies."""

    stats: dict = field(default_factory=dict)


# ── validation helpers ──────────────────────────────────────────────

def load_known_corporate_actions() -> pd.DataFrame:
    """Load the registered corporate-action ground truth."""
    if not KNOWN_ACTIONS_PATH.exists():
        logger.warning("No known corporate-action file at %s", KNOWN_ACTIONS_PATH)
        return pd.DataFrame(columns=["symbol", "date", "ratio"])

    records = json.loads(KNOWN_ACTIONS_PATH.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded %d known corporate actions for validation", len(df))
    return df


def _permutation_importance(
    model: IsolationForest,
    X: np.ndarray,
    feature_names: list[str],
    random_state: int,
    n_repeats: int = _PERM_REPEATS,
) -> pd.Series:
    """Mean absolute shift in anomaly score when each feature is shuffled.

    This is the unsupervised analogue of permutation importance: there is
    no accuracy metric to degrade, so we measure how much the model's own
    decision surface depends on each feature.  Returned values are
    normalised to sum to 1 and sorted descending.
    """
    rng = np.random.default_rng(random_state)

    n = X.shape[0]
    if n > _PERM_SAMPLE_SIZE:
        idx = rng.choice(n, size=_PERM_SAMPLE_SIZE, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    baseline = model.decision_function(X_sample)

    importances: dict[str, float] = {}
    for j, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_sample.copy()
            rng.shuffle(X_perm[:, j])
            permuted = model.decision_function(X_perm)
            deltas.append(np.abs(permuted - baseline).mean())
        importances[name] = float(np.mean(deltas))

    series = pd.Series(importances, name="importance")
    total = series.sum()
    if total > 0:
        series = series / total
    return series.sort_values(ascending=False)


# ── main entry point ────────────────────────────────────────────────

def detect_anomalies(
    features: Optional[pd.DataFrame] = None,
    contamination: Union[float, str] = DEFAULT_CONTAMINATION,
    n_estimators: int = 200,
    random_state: int = 42,
) -> AnomalyResult:
    """Detect abnormal ``(stock, day)`` sessions using Isolation Forest.

    Parameters
    ----------
    features : DataFrame, optional
        Pre-built matrix from
        :func:`infrastructure.ml.dataset.build_stock_day_features`.
        Built on demand when omitted.
    contamination : float or ``"auto"``
        Expected fraction of anomalies.  A float fixes the anomaly
        budget; ``"auto"`` defers to sklearn's own score offset.
    n_estimators : int
        Number of isolation trees.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    AnomalyResult
    """
    if features is None:
        features = build_stock_day_features()

    missing = [c for c in STOCK_FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"Feature matrix is missing columns: {missing}")

    feature_names = list(STOCK_FEATURE_COLUMNS)
    X_raw = features[feature_names].to_numpy(dtype=np.float64)

    # Isolation Forest is tree-based and scale-tolerant, but the feature
    # ranges here differ by orders of magnitude (z-scores ~N(0,1) vs
    # volume ratios that reach 50+).  Standardising keeps any single
    # wide-range feature from dominating the split geometry.
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    logger.info(
        "Fitting IsolationForest — %d rows × %d features, "
        "n_estimators=%d, contamination=%s",
        X.shape[0], X.shape[1], n_estimators, contamination,
    )
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)

    # Lower decision_function → more anomalous.  predict() returns -1
    # for outliers using the fitted offset.
    scores = model.decision_function(X)
    is_anomaly = model.predict(X) == -1

    labels = pd.DataFrame({
        "symbol": features["symbol"].to_numpy(),
        "date": features["date"].to_numpy(),
        "anomaly_score": scores,
        "is_anomaly": is_anomaly,
    })

    # Two views of feature importance.  The global view answers "what is
    # this model sensitive to overall"; because ~99.5% of rows are
    # unremarkable on every axis simultaneously, it comes out nearly
    # uniform and says little.  The anomaly-focused view permutes only
    # the flagged sessions and answers the question actually worth
    # asking — "what makes the outliers outliers" — so it is the primary
    # interpretation surface.
    importances_global = _permutation_importance(
        model, X, feature_names, random_state,
    )
    X_flagged = X[is_anomaly]
    if len(X_flagged) >= _MIN_ROWS_FOR_FOCUSED_IMPORTANCE:
        importances = _permutation_importance(
            model, X_flagged, feature_names, random_state,
            n_repeats=_PERM_REPEATS_FOCUSED,
        )
    else:
        logger.warning(
            "Only %d flagged sessions — falling back to global importance.",
            len(X_flagged),
        )
        importances = importances_global

    # ── validation against registered corporate actions ─────────────
    known = load_known_corporate_actions()
    if len(known):
        merged = known.merge(labels, on=["symbol", "date"], how="left")
        # Percentile of the score across the whole population — 0 means
        # "most anomalous session in the dataset".
        merged["score_percentile"] = merged["anomaly_score"].apply(
            lambda s: float((scores < s).mean() * 100) if pd.notna(s) else np.nan
        )
        merged["detected"] = merged["is_anomaly"].fillna(False).astype(bool)
        known_recall = float(merged["detected"].mean())
    else:
        merged = pd.DataFrame(columns=["symbol", "date", "detected"])
        known_recall = float("nan")

    n_flagged = int(is_anomaly.sum())
    stats = {
        "n_sessions": int(len(labels)),
        "n_symbols": int(features["symbol"].nunique()),
        "n_anomalies": n_flagged,
        "anomaly_rate": float(n_flagged / len(labels)) if len(labels) else 0.0,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "score_threshold": float(-model.offset_),
        "date_min": str(pd.Timestamp(labels["date"].min()).date()),
        "date_max": str(pd.Timestamp(labels["date"].max()).date()),
        "known_events_total": int(len(known)),
        "known_events_detected": int(merged["detected"].sum()) if len(known) else 0,
        "known_recall": known_recall,
        "top_feature": str(importances.index[0]) if len(importances) else None,
        "top3_features": [str(f) for f in importances.index[:3]],
    }

    logger.info(
        "Flagged %d / %d sessions (%.3f%%) — known-event recall %.0f%%",
        n_flagged, len(labels), stats["anomaly_rate"] * 100, known_recall * 100,
    )

    return AnomalyResult(
        labels=labels,
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        feature_importances=importances,
        feature_importances_global=importances_global,
        known_events=merged,
        known_recall=known_recall,
        stats=stats,
    )


# ── reporting helpers ───────────────────────────────────────────────

def top_anomalies(
    result: AnomalyResult,
    features: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:
    """Return the ``n`` most anomalous sessions with their key features."""
    cols = ["gap_pct", "return_pct", "range_pct", "volume_ratio_20d", "open_close_ratio"]
    merged = result.labels.merge(
        features[["symbol", "date", *cols]], on=["symbol", "date"], how="left",
    )
    return merged.nsmallest(n, "anomaly_score").reset_index(drop=True)


def anomalies_per_symbol(result: AnomalyResult, n: int = 20) -> pd.DataFrame:
    """Anomaly counts for the ``n`` most-flagged symbols."""
    flagged = result.labels[result.labels["is_anomaly"]]
    counts = (
        flagged.groupby("symbol").size()
        .sort_values(ascending=False)
        .head(n)
        .rename("n_anomalies")
        .reset_index()
    )
    return counts
