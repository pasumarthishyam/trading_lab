"""
K-Means Market-Day Clustering
=============================

Segments market-level trading days into behavioural regimes from 8
session/volatility features (NIFTY range and candle shape, gap, DVR
ratio, VIX level and momentum).

Methodological choices that make the result defensible rather than
arbitrary:

* ``K`` is **selected**, not assumed — every ``K`` in the requested
  range is scored by mean silhouette and the best one wins.
* Clusters are **profiled in original units**, so each regime can be
  read as a sentence rather than a centroid vector.
* Assignments are **stability-tested** — the model is refit under
  several seeds and the mean pairwise Adjusted Rand Index reports
  whether the partition is real or an artefact of initialisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from infrastructure.ml.dataset import (
    MARKET_FEATURE_COLUMNS,
    build_market_day_features,
)

logger = logging.getLogger(__name__)

# Seeds used for the stability check (mean pairwise ARI across refits).
_STABILITY_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

# Minimum |z| a centroid coordinate needs before it earns a name token.
_TIER_THRESHOLD: float = 0.3
# Maximum descriptive tokens in a cluster name.  Three, not two: a
# genuine crisis regime is extreme on volatility *and* range *and*
# conviction at once, and truncating that to two tokens makes it
# indistinguishable from an ordinary trending day.
_MAX_NAME_TOKENS: int = 3


@dataclass
class ClusterResult:
    """Everything produced by a K-Means market-regime run."""

    labels: pd.DataFrame
    """``date, cluster_id, cluster_name`` for every clustered day."""

    model: KMeans
    scaler: StandardScaler
    feature_names: list[str]

    profiles: pd.DataFrame
    """Per-cluster feature means and stds in original units, plus size."""

    centroid_z: pd.DataFrame
    """Per-cluster standardised centroids — the behavioural fingerprint."""

    silhouette_scores: dict[int, float]
    best_k: int
    stability_ari: float
    stats: dict = field(default_factory=dict)


# ── cluster naming ──────────────────────────────────────────────────

def _name_clusters(centroid_z: pd.DataFrame) -> dict[int, str]:
    """Derive human-readable regime names from standardised centroids.

    A regime is described along three semantic *slots*.  Each slot draws
    from a group of correlated features and contributes at most one
    token, which keeps names informative without repeating the same idea
    twice (``session_range_pct`` and ``dvr_ratio`` both measure range, so
    only the stronger of the two ever speaks).

    A slot only contributes when its strongest coordinate clears
    :data:`_TIER_THRESHOLD`, and the two loudest qualifying slots are
    emitted in canonical order.  Naming is therefore adaptive: whichever
    axes actually separate this particular partition are the ones that
    end up in the name.
    """
    # slot key -> (features, positive token, negative token)
    slots: list[tuple[str, list[str], str, str]] = [
        ("vol", ["vix_level", "vix_change_pct"], "high_vol", "low_vol"),
        ("range", ["session_range_pct", "dvr_ratio"], "wide_range", "narrow_range"),
        ("character", ["body_pct"], "trending", "choppy"),
    ]

    names: dict[int, str] = {}
    for cluster_id, row in centroid_z.iterrows():
        scored: list[tuple[float, int, str]] = []
        for order, (_key, feats, pos, neg) in enumerate(slots):
            present = [f for f in feats if f in row.index]
            if not present:
                continue
            # The loudest feature in the group speaks for the slot.
            lead = max(present, key=lambda f: abs(row[f]))
            z = float(row[lead])
            if abs(z) < _TIER_THRESHOLD:
                continue
            scored.append((abs(z), order, pos if z > 0 else neg))

        # Keep the two loudest slots, then restore canonical order so
        # names read consistently across clusters.
        chosen = sorted(scored, key=lambda t: t[0], reverse=True)[:_MAX_NAME_TOKENS]
        tokens = [tok for _mag, _order, tok in sorted(chosen, key=lambda t: t[1])]
        names[int(cluster_id)] = "_".join(tokens) if tokens else "baseline_regime"

    return _deduplicate_names(names, centroid_z)


def _deduplicate_names(
    names: dict[int, str],
    centroid_z: pd.DataFrame,
) -> dict[int, str]:
    """Guarantee unique names, disambiguating by gap direction first."""
    grouped: dict[str, list[int]] = {}
    for cluster_id, name in names.items():
        grouped.setdefault(name, []).append(cluster_id)

    for name, ids in grouped.items():
        if len(ids) == 1:
            continue
        for cluster_id in ids:
            gap = float(centroid_z.loc[cluster_id].get("gap_pct", 0.0))
            names[cluster_id] = f"{name}_{'gap_up' if gap > 0 else 'gap_down'}"

    # Final guard — force uniqueness if gap direction was not enough.
    regrouped: dict[str, list[int]] = {}
    for cluster_id, name in names.items():
        regrouped.setdefault(name, []).append(cluster_id)
    for name, ids in regrouped.items():
        if len(ids) > 1:
            for rank, cluster_id in enumerate(sorted(ids)):
                names[cluster_id] = f"{name}_{chr(ord('a') + rank)}"

    return names


# ── main entry point ────────────────────────────────────────────────

def cluster_market_days(
    features: Optional[pd.DataFrame] = None,
    k_range: tuple[int, int] = (3, 8),
    random_state: int = 42,
) -> ClusterResult:
    """Segment trading days into behavioural regimes using K-Means.

    Parameters
    ----------
    features : DataFrame, optional
        Pre-built matrix from
        :func:`infrastructure.ml.dataset.build_market_day_features`.
        Built on demand when omitted.
    k_range : tuple[int, int]
        Inclusive range of ``K`` values scored by silhouette.
    random_state : int
        Reproducibility seed for the final model.

    Returns
    -------
    ClusterResult
    """
    if features is None:
        features = build_market_day_features()

    missing = [c for c in MARKET_FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"Feature matrix is missing columns: {missing}")

    feature_names = list(MARKET_FEATURE_COLUMNS)
    frame = features.dropna(subset=feature_names).reset_index(drop=True)

    X_raw = frame[feature_names].to_numpy(dtype=np.float64)

    # K-Means is distance-based, so standardisation is mandatory here —
    # VIX levels (~10-80) would otherwise swamp fractional range features.
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    k_lo, k_hi = k_range
    if k_lo < 2:
        raise ValueError("k_range lower bound must be >= 2 for silhouette scoring.")
    if k_hi < k_lo:
        raise ValueError("k_range upper bound must be >= lower bound.")

    # ── K selection by silhouette ───────────────────────────────────
    silhouette_scores: dict[int, float] = {}
    for k in range(k_lo, k_hi + 1):
        km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=random_state)
        assign = km.fit_predict(X)
        score = float(silhouette_score(X, assign))
        silhouette_scores[k] = score
        logger.info("K=%d — silhouette %.4f", k, score)

    best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
    logger.info("Selected K=%d (silhouette %.4f)", best_k, silhouette_scores[best_k])

    # ── final model ─────────────────────────────────────────────────
    model = KMeans(n_clusters=best_k, n_init=20, max_iter=500, random_state=random_state)
    assignments = model.fit_predict(X)

    # ── profiling in original units ─────────────────────────────────
    profiled = frame[feature_names].copy()
    profiled["cluster_id"] = assignments

    means = profiled.groupby("cluster_id")[feature_names].mean()
    stds = profiled.groupby("cluster_id")[feature_names].std()
    sizes = profiled.groupby("cluster_id").size().rename("n_days")

    profiles = means.add_suffix("_mean").join(stds.add_suffix("_std")).join(sizes)
    profiles["pct_days"] = profiles["n_days"] / len(profiled)

    # Standardised centroids — directly comparable across features.
    centroid_z = pd.DataFrame(
        model.cluster_centers_, columns=feature_names,
    )
    centroid_z.index.name = "cluster_id"

    names = _name_clusters(centroid_z)
    profiles.insert(0, "cluster_name", [names[int(i)] for i in profiles.index])

    labels = pd.DataFrame({
        "date": frame["date"].to_numpy(),
        "cluster_id": assignments,
        "cluster_name": [names[int(i)] for i in assignments],
    })

    # ── stability: mean pairwise ARI across refits ───────────────────
    seed_assignments: list[np.ndarray] = []
    for seed in _STABILITY_SEEDS:
        km = KMeans(n_clusters=best_k, n_init=20, max_iter=500, random_state=seed)
        seed_assignments.append(km.fit_predict(X))

    pairwise = [
        adjusted_rand_score(seed_assignments[i], seed_assignments[j])
        for i in range(len(seed_assignments))
        for j in range(i + 1, len(seed_assignments))
    ]
    stability_ari = float(np.mean(pairwise)) if pairwise else float("nan")
    logger.info(
        "Stability — mean pairwise ARI %.4f over %d refits",
        stability_ari, len(_STABILITY_SEEDS),
    )

    stats = {
        "n_days": int(len(labels)),
        "best_k": int(best_k),
        "best_silhouette": float(silhouette_scores[best_k]),
        "stability_ari": stability_ari,
        "inertia": float(model.inertia_),
        "random_state": random_state,
        "k_range": [int(k_lo), int(k_hi)],
        "date_min": str(pd.Timestamp(labels["date"].min()).date()),
        "date_max": str(pd.Timestamp(labels["date"].max()).date()),
        "cluster_names": {int(k): v for k, v in names.items()},
        "cluster_sizes": {int(k): int(v) for k, v in sizes.items()},
    }

    return ClusterResult(
        labels=labels,
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        profiles=profiles,
        centroid_z=centroid_z,
        silhouette_scores=silhouette_scores,
        best_k=best_k,
        stability_ari=stability_ari,
        stats=stats,
    )
