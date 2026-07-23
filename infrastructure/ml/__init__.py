"""
Unsupervised ML Module
======================

Two production-quality unsupervised systems over the trading_lab
data lake:

1. **Isolation Forest anomaly detection** — flags abnormal
   ``(stock, day)`` sessions across the full F&O universe
   (~333K rows).  Validated against known corporate-action events.

2. **K-Means day clustering** — segments market-level trading days
   into behavioural regimes, with silhouette-based ``K`` selection
   and a stability check.

Public API
----------
    from infrastructure.ml import detect_anomalies, cluster_market_days
    from infrastructure.ml import (
        build_stock_day_features,
        build_market_day_features,
    )
"""

from infrastructure.ml.anomaly import AnomalyResult, detect_anomalies
from infrastructure.ml.clustering import ClusterResult, cluster_market_days
from infrastructure.ml.dataset import (
    build_market_day_features,
    build_stock_day_features,
)

__all__ = [
    "detect_anomalies",
    "cluster_market_days",
    "build_stock_day_features",
    "build_market_day_features",
    "AnomalyResult",
    "ClusterResult",
]
