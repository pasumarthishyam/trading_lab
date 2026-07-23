"""
Target Construction
===================

Defines the binary "significant move" label:

.. math::

    y_t = \\mathbb{1}\\left[\\;
        \\left| \\frac{C_{t+h}}{C_t} - 1 \\right|
        \\;>\\; k \\cdot \\mathrm{ATR\\%}_t
    \\;\\right]

where :math:`h` is :data:`config.LABEL_HORIZON_DAYS`, :math:`k` is
:data:`config.LABEL_VOL_MULTIPLE`, and :math:`\\mathrm{ATR\\%}_t` is the
trailing average-true-range percentage **known at the close of day t**.

Scaling the threshold by each symbol's own trailing volatility is what
makes the label portable across the universe: a 3% day is unremarkable
for a high-beta midcap and extraordinary for a large-cap staple, and an
absolute threshold would simply teach the model to recognise which
symbols are volatile — a fact it already has as a feature.

Two exclusions keep the target honest:

* **Corporate actions.**  A split or demerger ex-date shows up in raw
  prices as an enormous overnight gap.  Labelling those as "significant
  moves" would train the model to predict registry events from price
  history.  Both the ex-date and the bar whose forward window covers it
  are masked out.
* **Warm-up bars.**  A symbol needs :data:`config.MIN_HISTORY_BARS` of
  history before its trailing ATR is meaningful.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from infrastructure.data.corporate_actions import load_corporate_actions
from infrastructure.ml.classifier import config as C

logger = logging.getLogger(__name__)


def _corp_action_index(panel: pd.DataFrame) -> pd.Series:
    """Boolean mask: True where ``(symbol, date)`` is a corp-action ex-date."""
    registry = load_corporate_actions()
    if registry.empty:
        logger.warning("Corporate-actions registry empty — no dates masked")
        return pd.Series(False, index=panel.index)

    pairs = set(
        zip(registry["symbol"], pd.to_datetime(registry["ex_date"]).dt.normalize())
    )
    mask = pd.Series(
        [(s, d) in pairs for s, d in zip(panel["symbol"], panel["date"])],
        index=panel.index,
    )
    logger.info(
        "Corporate-action registry — %d entries, %d panel rows matched",
        len(registry), int(mask.sum()),
    )
    return mask


def add_labels(
    panel: pd.DataFrame,
    horizon: int = C.LABEL_HORIZON_DAYS,
    vol_multiple: float = C.LABEL_VOL_MULTIPLE,
    scale_feature: str = C.LABEL_SCALE_FEATURE,
) -> pd.DataFrame:
    """Attach the forward return, the per-row threshold, and the label ``y``.

    Parameters
    ----------
    panel : pd.DataFrame
        Feature panel from :func:`features.build_feature_panel`.  Must
        carry ``symbol``, ``date``, ``close`` and *scale_feature*.
    horizon : int, optional
        Forward window in trading days.
    vol_multiple : float, optional
        Threshold as a multiple of the trailing volatility estimate.
    scale_feature : str, optional
        Column holding that estimate — see
        :data:`config.LABEL_SCALE_FEATURE` for why the default is
        close-to-close volatility rather than ATR.

    Returns
    -------
    pd.DataFrame
        The panel plus ``fwd_ret``, ``fwd_abs_ret``, ``label_threshold``,
        ``y`` and ``label_valid``.  Rows are **not** dropped here — the
        caller decides, using ``label_valid``.
    """
    if scale_feature not in panel.columns:
        raise KeyError(
            f"Label scale feature {scale_feature!r} not in panel; "
            f"expected one of the trailing-volatility columns."
        )

    df = panel.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("symbol", sort=False)

    # Forward return over the next `horizon` bars, per symbol.
    fwd_close = g["close"].shift(-horizon)
    df["fwd_ret"] = fwd_close / df["close"] - 1.0
    df["fwd_abs_ret"] = df["fwd_ret"].abs()

    # Threshold is built from information as of today's close only.  A
    # multi-day horizon scales with sqrt(h) under the usual iid-increment
    # assumption, so the bar does not become trivially easy to clear as
    # the horizon lengthens.
    scale = df[scale_feature] * np.sqrt(horizon)
    df["label_threshold"] = vol_multiple * scale
    df["label_scale_feature"] = scale_feature

    df["y"] = (df["fwd_abs_ret"] > df["label_threshold"]).astype(float)

    # ── validity mask ───────────────────────────────────────────────
    bar_number = g.cumcount()
    has_forward = df["fwd_ret"].notna()
    has_threshold = df["label_threshold"].notna() & (df["label_threshold"] > 0)
    warm = bar_number >= C.MIN_HISTORY_BARS

    corp = _corp_action_index(df)
    # A corp-action ex-date contaminates both its own bar and the bars
    # whose forward window reaches across it.
    corp_forward = (
        corp.groupby(df["symbol"], sort=False)
        .transform(lambda s: s.rolling(horizon + 1, min_periods=1).max().shift(-horizon))
        .fillna(0.0)
        .astype(bool)
    )
    contaminated = corp | corp_forward

    df["label_valid"] = has_forward & has_threshold & warm & ~contaminated
    df.loc[~df["label_valid"], "y"] = np.nan

    valid = df["label_valid"]
    logger.info(
        "Labels built — %d valid rows (%.1f%% of panel), base rate %.4f",
        int(valid.sum()),
        100.0 * valid.mean(),
        float(df.loc[valid, "y"].mean()),
    )
    logger.info(
        "Excluded — %d no-forward, %d no-threshold, %d warm-up, %d corp-action",
        int((~has_forward).sum()),
        int((~has_threshold).sum()),
        int((~warm).sum()),
        int(contaminated.sum()),
    )
    return df


def label_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Base rate by year — a quick check that the target is stationary.

    A label whose base rate drifts hard across the sample would make
    walk-forward folds incomparable, so this is worth eyeballing before
    trusting any lift number.
    """
    valid = df[df["label_valid"]]
    out = (
        valid.assign(year=valid["date"].dt.year)
        .groupby("year")
        .agg(
            rows=("y", "size"),
            base_rate=("y", "mean"),
            mean_threshold=("label_threshold", "mean"),
            mean_abs_fwd_ret=("fwd_abs_ret", "mean"),
        )
        .reset_index()
    )
    return out
