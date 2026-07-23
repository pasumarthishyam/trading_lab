"""
Evaluation Metrics
==================

Metrics chosen for a rare-event, cross-sectional problem.

**PR-AUC is the headline, not ROC-AUC.**  With a base rate around 5-10%,
ROC-AUC is dominated by the true-negative mass and looks respectable for
a model that is barely better than a coin flip on the cases that matter.
Average precision tracks the positive class directly.

**Lift over the base rate is the number to quote.**  A PR-AUC of 0.12
means nothing in isolation; against a 0.06 base rate it is 2x lift.  All
comparisons in this pipeline are expressed that way.

**Precision@k mirrors how the signal would actually be used** — you act
on the top few percent of a day's cross-section, not on every row above
0.5.  It is computed per test block, ranking all rows in the block.

**Brier score and calibration** matter because a probability that is
ranked well but scaled badly cannot be sized against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from infrastructure.ml.classifier import config as C

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    """Scores for one set of predictions."""

    n: int
    base_rate: float
    pr_auc: float
    pr_auc_lift: float
    roc_auc: float
    brier: float
    log_loss: float
    precision_at_k: dict[str, float] = field(default_factory=dict)
    lift_at_k: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict:
        row = {
            "n": self.n,
            "base_rate": self.base_rate,
            "pr_auc": self.pr_auc,
            "pr_auc_lift": self.pr_auc_lift,
            "roc_auc": self.roc_auc,
            "brier": self.brier,
            "log_loss": self.log_loss,
        }
        row.update({f"prec@{k}": v for k, v in self.precision_at_k.items()})
        row.update({f"lift@{k}": v for k, v in self.lift_at_k.items()})
        return row


def _precision_at_k(y: np.ndarray, p: np.ndarray, rate: float) -> float:
    """Positive rate among the top *rate* fraction by predicted score."""
    n_top = max(1, int(len(y) * rate))
    order = np.argsort(-p, kind="stable")[:n_top]
    return float(y[order].mean())


def score(y_true, y_pred) -> Metrics:
    """Compute the full metric set for one block of predictions."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y) & np.isfinite(p)
    y, p = y[finite], p[finite]

    base = float(y.mean()) if len(y) else float("nan")
    # A block with only one class present cannot be scored; return NaNs
    # rather than letting sklearn raise mid-walk-forward.
    if len(y) == 0 or len(np.unique(y)) < 2:
        logger.warning("Degenerate block — %d rows, %d classes", len(y), len(np.unique(y)))
        return Metrics(len(y), base, float("nan"), float("nan"), float("nan"),
                       float("nan"), float("nan"))

    pr = float(average_precision_score(y, p))
    prec = {f"{r:.0%}": _precision_at_k(y, p, r) for r in C.PRECISION_AT_K}
    return Metrics(
        n=len(y),
        base_rate=base,
        pr_auc=pr,
        pr_auc_lift=pr / base if base > 0 else float("nan"),
        roc_auc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, np.clip(p, 0.0, 1.0))),
        log_loss=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0.0, 1.0])),
        precision_at_k=prec,
        lift_at_k={k: (v / base if base > 0 else float("nan"))
                   for k, v in prec.items()},
    )


def aggregate(fold_metrics: list[Metrics]) -> dict:
    """Row-weighted mean across folds.

    Weighted by block size so a short final fold cannot swing the
    headline number as much as a full-length one.
    """
    usable = [m for m in fold_metrics if np.isfinite(m.pr_auc)]
    if not usable:
        return {}
    w = np.array([m.n for m in usable], dtype=float)
    w = w / w.sum()

    def wm(attr: str) -> float:
        return float(np.sum(w * np.array([getattr(m, attr) for m in usable])))

    out = {
        "n_folds": len(usable),
        "n_rows": int(sum(m.n for m in usable)),
        "base_rate": wm("base_rate"),
        "pr_auc": wm("pr_auc"),
        "pr_auc_lift": wm("pr_auc_lift"),
        "roc_auc": wm("roc_auc"),
        "brier": wm("brier"),
        "log_loss": wm("log_loss"),
        "pr_auc_std": float(np.std([m.pr_auc for m in usable])),
    }
    for k in usable[0].precision_at_k:
        out[f"prec@{k}"] = float(
            np.sum(w * np.array([m.precision_at_k[k] for m in usable]))
        )
        out[f"lift@{k}"] = float(
            np.sum(w * np.array([m.lift_at_k[k] for m in usable]))
        )
    return out


def calibration_table(y_true, y_pred, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs realised frequency, by decile of predicted probability."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y) & np.isfinite(p)
    y, p = y[finite], p[finite]
    if len(y) == 0:
        return pd.DataFrame()

    # Rank-based bins keep every bucket populated even when the score
    # distribution is heavily skewed toward zero.
    ranks = pd.Series(p).rank(method="first", pct=True)
    bins = np.clip((ranks * n_bins).astype(int), 0, n_bins - 1)
    df = pd.DataFrame({"bin": bins, "y": y, "p": p})
    return (
        df.groupby("bin")
        .agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
        .reset_index()
    )


def fold_table(fold_metrics: list[Metrics], names: list[str] | None = None) -> pd.DataFrame:
    """One row per fold, for the report."""
    rows = []
    for i, m in enumerate(fold_metrics):
        row = {"fold": names[i] if names else i}
        row.update(m.to_row())
        rows.append(row)
    return pd.DataFrame(rows)
