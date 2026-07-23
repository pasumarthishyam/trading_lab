"""
Purged Walk-Forward Cross-Validation
====================================

Time-series splitting for a *cross-sectional panel*, where ~200 symbols
share every trading date.

Why the usual splitters are wrong here
--------------------------------------

``KFold`` shuffles, which puts tomorrow in the training set and yesterday
in the test set — the model learns the future outright.  ``TimeSeriesSplit``
fixes the ordering but splits on **row position**, and in a panel that
cuts a single date in half: 100 symbols from 2023-04-11 land in train and
the other 114 land in test.  Since every symbol on a given day shares the
same market-wide shock, that leaks the answer sideways across the
boundary.

This module splits on **dates**, so a date is wholly in train or wholly
in test, and then applies two further guards from Lopez de Prado's
*Advances in Financial Machine Learning* (ch. 7):

**Purge.**  The label on date ``t`` is realised at ``t + h``.  Training
rows whose label window reaches into the test block therefore know part
of the test period's outcome.  The last ``h`` training dates before the
test block are dropped.

**Embargo.**  Features are built from rolling windows, so the rows
immediately after a test block are still partly composed of test-period
bars.  An additional ``EMBARGO_DAYS`` of dates either side of the test
block is discarded so that serial correlation cannot carry information
across the seam.

Layout of one fold (expanding training window)::

    |<------------- train ------------->|xxxxx|<-- test -->|
                                         purge
                                       + embargo
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from infrastructure.ml.classifier import config as C

logger = logging.getLogger(__name__)


@dataclass
class Fold:
    """One walk-forward split, described by dates and row positions."""

    index: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_purged_dates: int
    meta: dict = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"fold {self.index}: "
            f"train {self.train_start.date()}..{self.train_end.date()} "
            f"({len(self.train_idx):,} rows) | "
            f"gap {self.n_purged_dates}d | "
            f"test {self.test_start.date()}..{self.test_end.date()} "
            f"({len(self.test_idx):,} rows)"
        )


class PurgedWalkForward:
    """Date-based walk-forward splitter with purge and embargo gaps.

    Parameters
    ----------
    n_folds : int
        Number of sequential test blocks.
    horizon : int
        Label horizon in trading days; sets the minimum purge.
    embargo : int
        Extra trading days dropped from the end of each training block.
    expanding : bool
        ``True`` grows the training block from the start of the sample;
        ``False`` slides a fixed-length window.
    min_train_dates : int
        Folds whose training block has fewer dates than this are skipped.
    """

    def __init__(
        self,
        n_folds: int = C.N_OUTER_FOLDS,
        horizon: int = C.LABEL_HORIZON_DAYS,
        embargo: int = C.EMBARGO_DAYS,
        expanding: bool = C.EXPANDING_TRAIN,
        min_train_dates: int = C.MIN_TRAIN_DATES,
    ) -> None:
        if embargo < 0 or horizon < 1:
            raise ValueError("horizon must be >= 1 and embargo >= 0")
        self.n_folds = n_folds
        self.horizon = horizon
        self.embargo = embargo
        self.expanding = expanding
        self.min_train_dates = min_train_dates

    @property
    def gap_days(self) -> int:
        """Total trading days discarded between train and test."""
        return max(self.horizon, C.PURGE_DAYS) + self.embargo

    def split(
        self,
        dates: pd.Series,
        date_range: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
    ) -> Iterator[Fold]:
        """Yield folds for a panel whose row dates are *dates*.

        Parameters
        ----------
        dates : pd.Series
            The ``date`` column of the panel, one entry per row.  Need
            not be sorted; row positions are returned, not labels.
        date_range : (Timestamp, Timestamp), optional
            Restrict splitting to this inclusive date window — used to
            confine hyperparameter search to the development period.
        """
        dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        mask = pd.Series(True, index=dates.index)
        if date_range is not None:
            lo, hi = date_range
            mask = (dates >= lo) & (dates <= hi)

        unique_dates = np.sort(dates[mask].unique())
        n_dates = len(unique_dates)
        if n_dates < self.min_train_dates + self.n_folds * (self.gap_days + 1):
            raise ValueError(
                f"Only {n_dates} dates available — not enough for "
                f"{self.n_folds} folds with a {self.gap_days}-day gap."
            )

        # Reserve the tail of the timeline for the test blocks and carve
        # it into n_folds contiguous, equal-sized pieces.
        first_test_pos = max(
            self.min_train_dates + self.gap_days,
            int(n_dates * 0.35),
        )
        test_span = n_dates - first_test_pos
        block = test_span // self.n_folds
        if block < 1:
            raise ValueError("Test blocks would be empty — reduce n_folds.")

        gap = self.gap_days
        emitted = 0

        for k in range(self.n_folds):
            test_lo_pos = first_test_pos + k * block
            test_hi_pos = (
                n_dates - 1 if k == self.n_folds - 1
                else first_test_pos + (k + 1) * block - 1
            )
            train_hi_pos = test_lo_pos - gap - 1
            if train_hi_pos < 0:
                continue

            train_lo_pos = 0
            if not self.expanding:
                train_lo_pos = max(0, train_hi_pos - self.min_train_dates * 2 + 1)

            train_dates = unique_dates[train_lo_pos: train_hi_pos + 1]
            test_dates = unique_dates[test_lo_pos: test_hi_pos + 1]
            if len(train_dates) < self.min_train_dates or len(test_dates) == 0:
                continue

            train_set = set(train_dates)
            test_set = set(test_dates)
            train_idx = np.flatnonzero(mask.to_numpy() & dates.isin(train_set).to_numpy())
            test_idx = np.flatnonzero(mask.to_numpy() & dates.isin(test_set).to_numpy())
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            fold = Fold(
                index=emitted,
                train_idx=train_idx,
                test_idx=test_idx,
                train_start=pd.Timestamp(train_dates[0]),
                train_end=pd.Timestamp(train_dates[-1]),
                test_start=pd.Timestamp(test_dates[0]),
                test_end=pd.Timestamp(test_dates[-1]),
                n_purged_dates=gap,
                meta={
                    "n_train_dates": len(train_dates),
                    "n_test_dates": len(test_dates),
                },
            )
            emitted += 1
            logger.info(fold.describe())
            yield fold


def inner_split(
    dates: pd.Series,
    train_idx: np.ndarray,
    n_folds: int = C.N_INNER_FOLDS,
    horizon: int = C.LABEL_HORIZON_DAYS,
    embargo: int = C.EMBARGO_DAYS,
) -> list[Fold]:
    """Purged folds *within* a training block, for hyperparameter search.

    Keeps tuning strictly inside the training data of the outer fold, so
    the outer test block stays untouched.
    """
    sub_dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True)).iloc[train_idx]
    lo, hi = sub_dates.min(), sub_dates.max()
    splitter = PurgedWalkForward(
        n_folds=n_folds,
        horizon=horizon,
        embargo=embargo,
        expanding=True,
        min_train_dates=max(60, C.MIN_TRAIN_DATES // 3),
    )
    return list(splitter.split(dates, date_range=(lo, hi)))


def chronological_holdout(
    dates: pd.Series,
    idx: np.ndarray,
    fraction: float = C.EARLY_STOPPING_FRACTION,
    embargo: int = C.EMBARGO_DAYS,
    horizon: int = C.LABEL_HORIZON_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Split *idx* into (fit, validation) by date, with a purge gap.

    Used for boosting early stopping: the validation slice is the tail of
    the training block, separated from the fit slice by the same gap the
    outer folds use, so the stopping round is not chosen on leaked data.
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    sub = dates.iloc[idx]
    unique_dates = np.sort(sub.unique())
    n = len(unique_dates)
    gap = max(horizon, C.PURGE_DAYS) + embargo
    n_val = max(1, int(n * fraction))
    val_lo_pos = n - n_val
    fit_hi_pos = val_lo_pos - gap - 1
    if fit_hi_pos < 1:
        # Training block too short to carve a purged validation slice —
        # fall back to using it whole for both, and let the caller's
        # fixed round count govern.
        return idx, idx

    fit_dates = set(unique_dates[: fit_hi_pos + 1])
    val_dates = set(unique_dates[val_lo_pos:])
    fit_idx = idx[sub.isin(fit_dates).to_numpy()]
    val_idx = idx[sub.isin(val_dates).to_numpy()]
    return fit_idx, val_idx


# ── leakage assertions ──────────────────────────────────────────────

def assert_no_leakage(fold: Fold, dates: pd.Series, horizon: int) -> None:
    """Raise if a fold violates the purge/embargo contract.

    Cheap enough to run on every fold, and it catches the class of bug
    that quietly inflates every downstream metric.
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    train_dates = dates.iloc[fold.train_idx]
    test_dates = dates.iloc[fold.test_idx]

    overlap = set(train_dates.unique()) & set(test_dates.unique())
    if overlap:
        raise AssertionError(
            f"fold {fold.index}: {len(overlap)} dates appear in both train and test"
        )

    if train_dates.max() >= test_dates.min():
        raise AssertionError(
            f"fold {fold.index}: train ends {train_dates.max().date()} "
            f"at or after test starts {test_dates.min().date()}"
        )

    # The gap is measured in trading days, so count intervening dates in
    # the full calendar rather than differencing the timestamps.
    all_dates = np.sort(dates.unique())
    gap_dates = all_dates[
        (all_dates > train_dates.max().to_datetime64())
        & (all_dates < test_dates.min().to_datetime64())
    ]
    if len(gap_dates) < horizon:
        raise AssertionError(
            f"fold {fold.index}: gap of {len(gap_dates)} trading days is "
            f"smaller than the {horizon}-day label horizon"
        )
    logger.debug(
        "fold %d leakage checks passed — %d trading days in gap",
        fold.index, len(gap_dates),
    )
