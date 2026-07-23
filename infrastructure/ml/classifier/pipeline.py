"""
Pipeline Orchestration
======================

Wires the pieces into one reproducible run:

1. **Build** the feature panel and attach labels (cached to parquet).
2. **Split** the timeline into a development period and an untouched
   evaluation period.
3. **Tune** LightGBM and XGBoost with Optuna on purged inner folds
   inside the development period only.
4. **Walk forward** across the evaluation period with the frozen
   parameters, scoring the models and all three baselines on identical
   test blocks.
5. **Explain** the final model with SHAP over held-out rows.
6. **Report** fold-level and aggregate metrics, plus lift over baseline.

Every fold passes :func:`cv.assert_no_leakage` before it is used.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from infrastructure.ml.classifier import (
    baseline,
    config as C,
    cv,
    evaluate,
    explain,
    features as F,
    labeling,
    models,
    tuning,
)

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Everything one pipeline run produced."""

    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    shap: Optional[explain.ShapResult] = None
    native_importance: Optional[pd.DataFrame] = None
    tuning: dict = field(default_factory=dict)
    label_stats: Optional[pd.DataFrame] = None
    calibration: Optional[pd.DataFrame] = None
    predictions: Optional[pd.DataFrame] = None
    meta: dict = field(default_factory=dict)


# ── data preparation ────────────────────────────────────────────────

def cache_path(universe: Optional[list[str]], include_intraday: bool):
    """Cache file keyed by universe and feature-family selection.

    Without the key, a quick ``--limit 20`` smoke run would overwrite the
    full-universe panel and every later run would silently train on 20
    symbols.
    """
    n = "full" if universe is None else f"n{len(universe)}"
    tf = "intraday" if include_intraday else "daily"
    return C.RESULTS_ROOT / f"panel_{n}_{tf}.parquet"


def build_dataset(
    universe: Optional[list[str]] = None,
    use_cache: bool = True,
    include_intraday: bool = True,
) -> pd.DataFrame:
    """Build (or load) the labelled panel.

    The panel is cached because the two DuckDB scans plus the per-symbol
    rolling features take a few minutes; re-tuning should not pay that
    cost every time.
    """
    path = cache_path(universe, include_intraday)
    if use_cache and path.exists():
        panel = pd.read_parquet(path)
        logger.info(
            "Loaded cached panel — %d rows x %d cols from %s",
            len(panel), panel.shape[1], path,
        )
        return panel

    t0 = time.time()
    panel = F.build_feature_panel(universe, include_intraday=include_intraday)
    panel = labeling.add_labels(panel)
    panel = panel.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)

    C.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    logger.info(
        "Built and cached panel in %.1fs — %d rows x %d cols",
        time.time() - t0, len(panel), panel.shape[1],
    )
    return panel


def prepare(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop unlabelled rows and resolve the model feature list."""
    labelled = panel[panel["label_valid"]].reset_index(drop=True)
    feats = F.feature_columns(labelled)

    # Guard against a feature that is constant or entirely missing over
    # the sample — it costs training time and confuses SHAP shares.
    usable = []
    for f in feats:
        col = labelled[f]
        if col.notna().sum() < 0.01 * len(col):
            logger.warning("Dropping %s — %.2f%% non-null", f, 100 * col.notna().mean())
            continue
        if col.nunique(dropna=True) <= 1:
            logger.warning("Dropping %s — constant", f)
            continue
        usable.append(f)

    logger.info(
        "Prepared dataset — %d labelled rows, %d features, base rate %.4f",
        len(labelled), len(usable), labelled["y"].mean(),
    )
    return labelled, usable


# ── walk-forward evaluation ─────────────────────────────────────────

def walk_forward(
    panel: pd.DataFrame,
    feats: list[str],
    model_params: dict[str, dict],
    eval_range: tuple[pd.Timestamp, pd.Timestamp],
    n_folds: int = C.N_OUTER_FOLDS,
    collect_predictions: bool = True,
) -> tuple[pd.DataFrame, dict, list, Optional[pd.DataFrame]]:
    """Score models and baselines over purged walk-forward folds.

    Returns ``(fold_table, aggregates, fitted_models, predictions)``.
    All strategies are scored on *identical* test blocks, so differences
    between them cannot come from different data.
    """
    splitter = cv.PurgedWalkForward(n_folds=n_folds)
    folds = list(splitter.split(panel["date"], date_range=eval_range))
    if not folds:
        raise RuntimeError("No usable folds — check date range and MIN_TRAIN_DATES.")

    strategies = list(model_params) + ["prior", "vol_cluster", "logit"]
    per_fold: dict[str, list[evaluate.Metrics]] = {s: [] for s in strategies}
    fitted: list[models.FittedModel] = []
    rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    for fold in folds:
        cv.assert_no_leakage(fold, panel["date"], C.LABEL_HORIZON_DAYS)

        train = panel.iloc[fold.train_idx]
        test = panel.iloc[fold.test_idx]
        y_train = train["y"].to_numpy(dtype=float)
        y_test = test["y"].to_numpy(dtype=float)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.warning("fold %d has a single class — skipped", fold.index)
            continue

        preds: dict[str, np.ndarray] = baseline.all_baselines(train, test, y_train)

        fit_idx, val_idx = cv.chronological_holdout(panel["date"], fold.train_idx)
        fit, val = panel.iloc[fit_idx], panel.iloc[val_idx]

        for kind, params in model_params.items():
            t0 = time.time()
            model = models.train(
                kind, fit, fit["y"].to_numpy(dtype=float), feats, params,
                X_val=val, y_val=val["y"].to_numpy(dtype=float),
            )
            preds[kind] = model.predict(test)
            fitted.append(model)
            logger.info(
                "fold %d %s — %d rounds, %.1fs",
                fold.index, kind, model.best_iteration, time.time() - t0,
            )

        for name, p in preds.items():
            m = evaluate.score(y_test, p)
            per_fold[name].append(m)
            rows.append({
                "fold": fold.index,
                "strategy": name,
                "test_start": fold.test_start.date(),
                "test_end": fold.test_end.date(),
                "n_train": len(fold.train_idx),
                **m.to_row(),
            })

        if collect_predictions:
            frame = test[["symbol", "date", "y", "fwd_abs_ret", "label_threshold"]].copy()
            frame["fold"] = fold.index
            for name, p in preds.items():
                frame[f"pred_{name}"] = p
            pred_frames.append(frame)

    fold_table = pd.DataFrame(rows)
    aggregates = {s: evaluate.aggregate(ms) for s, ms in per_fold.items() if ms}
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else None
    return fold_table, aggregates, fitted, predictions


def summarise(aggregates: dict) -> pd.DataFrame:
    """Aggregate table, ordered best-first by PR-AUC, with lift columns.

    ``lift_vs_logit`` is the honest headline: how much the boosted model
    adds over a linear model on the obvious volatility features.
    """
    df = pd.DataFrame(aggregates).T.reset_index().rename(columns={"index": "strategy"})
    if df.empty:
        return df
    df = df.sort_values("pr_auc", ascending=False).reset_index(drop=True)

    logit = df.loc[df["strategy"] == "logit", "pr_auc"]
    if len(logit) and logit.iloc[0] > 0:
        df["lift_vs_logit"] = df["pr_auc"] / logit.iloc[0]
    prior = df.loc[df["strategy"] == "prior", "pr_auc"]
    if len(prior) and prior.iloc[0] > 0:
        df["lift_vs_prior"] = df["pr_auc"] / prior.iloc[0]
    return df


# ── top-level run ───────────────────────────────────────────────────

def run(
    universe: Optional[list[str]] = None,
    use_cache: bool = True,
    include_intraday: bool = True,
    n_trials: int = C.OPTUNA_TRIALS,
    kinds: tuple[str, ...] = ("lgbm", "xgb"),
    skip_tuning: bool = False,
    run_shap: bool = True,
    save: bool = True,
) -> RunResult:
    """Execute the full pipeline and return every artefact it produced."""
    t_start = time.time()

    panel = build_dataset(universe, use_cache=use_cache,
                          include_intraday=include_intraday)
    label_stats = labeling.label_summary(panel)
    labelled, feats = prepare(panel)

    dev_range, eval_range = tuning.dev_eval_split(labelled["date"])

    # ── tuning (development period only) ────────────────────────────
    tuned: dict[str, dict] = {}
    tuning_info: dict = {}
    for kind in kinds:
        if skip_tuning:
            tuned[kind] = {}
            continue
        res = tuning.tune(labelled, feats, kind, dev_range, n_trials=n_trials)
        tuned[kind] = res.best_params
        tuning_info[kind] = {
            "best_value": res.best_value,
            "best_params": res.best_params,
            "n_trials": res.n_trials,
        }

    # ── walk-forward evaluation (untouched period) ──────────────────
    fold_table, aggregates, fitted, predictions = walk_forward(
        labelled, feats, tuned, eval_range
    )
    summary = summarise(aggregates)

    # ── explanation from the last fitted model of the best kind ─────
    shap_result = None
    native = None
    if run_shap and fitted:
        best_kind = next(
            (s for s in summary["strategy"] if s in tuned), None
        )
        chosen = [m for m in fitted if m.kind == best_kind]
        if chosen:
            model = chosen[-1]
            native = models.native_importance(model)
            held_out = labelled[
                (labelled["date"] >= eval_range[0]) & (labelled["date"] <= eval_range[1])
            ]
            shap_result = explain.explain(model, held_out)

    calibration = None
    if predictions is not None and not summary.empty:
        best = summary["strategy"].iloc[0]
        calibration = evaluate.calibration_table(
            predictions["y"], predictions[f"pred_{best}"]
        )

    result = RunResult(
        fold_metrics=fold_table,
        summary=summary,
        shap=shap_result,
        native_importance=native,
        tuning=tuning_info,
        label_stats=label_stats,
        calibration=calibration,
        predictions=predictions,
        meta={
            "n_rows": int(len(labelled)),
            "n_features": len(feats),
            "features": feats,
            "base_rate": float(labelled["y"].mean()),
            "dev_range": [str(dev_range[0].date()), str(dev_range[1].date())],
            "eval_range": [str(eval_range[0].date()), str(eval_range[1].date())],
            "label": {
                "horizon_days": C.LABEL_HORIZON_DAYS,
                "vol_multiple": C.LABEL_VOL_MULTIPLE,
                "atr_window": C.LABEL_ATR_WINDOW,
            },
            "cv": {
                "n_outer_folds": C.N_OUTER_FOLDS,
                "purge_days": C.PURGE_DAYS,
                "embargo_days": C.EMBARGO_DAYS,
            },
            "runtime_sec": round(time.time() - t_start, 1),
        },
    )

    if save:
        save_results(result)
    logger.info("Pipeline complete in %.1fs", time.time() - t_start)
    return result


def save_results(result: RunResult, root: Optional[object] = None) -> None:
    """Persist tables and metadata under ``data/processed/ml_classifier/``."""
    out = C.RESULTS_ROOT if root is None else root
    out.mkdir(parents=True, exist_ok=True)

    result.fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
    result.summary.to_csv(out / "summary.csv", index=False)
    if result.label_stats is not None:
        result.label_stats.to_csv(out / "label_stats.csv", index=False)
    if result.native_importance is not None:
        result.native_importance.to_csv(out / "native_importance.csv", index=False)
    if result.shap is not None:
        result.shap.importance.to_csv(out / "shap_importance.csv", index=False)
        explain.direction_table(result.shap).to_csv(
            out / "shap_direction.csv", index=False
        )
    if result.calibration is not None:
        result.calibration.to_csv(out / "calibration.csv", index=False)
    if result.predictions is not None:
        result.predictions.to_parquet(out / "predictions.parquet", index=False)

    (out / "run_meta.json").write_text(
        json.dumps({"meta": result.meta, "tuning": result.tuning}, indent=2),
        encoding="utf-8",
    )
    logger.info("Results written to %s", out)
