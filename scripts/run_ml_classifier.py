"""
Run the significant-move classification pipeline.

Examples
--------
Full run — rebuild the panel, tune both models, evaluate, explain::

    python scripts/run_ml_classifier.py --rebuild

Quick iteration on a small universe, no tuning::

    python scripts/run_ml_classifier.py --rebuild --limit 30 \
        --skip-tuning --no-intraday --no-shap

Re-evaluate from the cached panel with a bigger search budget::

    python scripts/run_ml_classifier.py --trials 100

Results land in ``data/processed/ml_classifier/``.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from infrastructure.ml.classifier import config as C
from infrastructure.ml.classifier import features as F
from infrastructure.ml.classifier import pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate the significant-move classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rebuild", action="store_true",
                   help="Rebuild the feature panel instead of using the cache.")
    p.add_argument("--limit", type=int, default=None,
                   help="Use only the first N symbols (fast smoke runs).")
    p.add_argument("--trials", type=int, default=C.OPTUNA_TRIALS,
                   help="Optuna trials per model.")
    p.add_argument("--models", default="lgbm,xgb",
                   help="Comma-separated model kinds to run.")
    p.add_argument("--skip-tuning", action="store_true",
                   help="Use library defaults instead of running Optuna.")
    p.add_argument("--no-intraday", action="store_true",
                   help="Skip the 15-minute feature family (much faster build).")
    p.add_argument("--no-shap", action="store_true",
                   help="Skip the SHAP attribution pass.")
    p.add_argument("--quiet", action="store_true", help="Warnings and errors only.")
    return p.parse_args(argv)


def _print_report(result: pipeline.RunResult) -> None:
    """Human-readable summary of the run."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    meta = result.meta
    print("\n" + "=" * 78)
    print("SIGNIFICANT-MOVE CLASSIFIER")
    print("=" * 78)
    print(f"rows           : {meta['n_rows']:,}")
    print(f"features       : {meta['n_features']}")
    print(f"base rate      : {meta['base_rate']:.4f}")
    print(f"label          : |ret(t+{meta['label']['horizon_days']})| > "
          f"{meta['label']['vol_multiple']}x ATR%({meta['label']['atr_window']})")
    print(f"dev period     : {meta['dev_range'][0]} .. {meta['dev_range'][1]}  (tuning only)")
    print(f"eval period    : {meta['eval_range'][0]} .. {meta['eval_range'][1]}  (reported below)")
    print(f"purge / embargo: {meta['cv']['purge_days']}d / {meta['cv']['embargo_days']}d")
    print(f"runtime        : {meta['runtime_sec']}s")

    if result.label_stats is not None and not result.label_stats.empty:
        print("\n--- label base rate by year " + "-" * 46)
        print(result.label_stats.to_string(index=False))

    print("\n--- walk-forward results (evaluation period) " + "-" * 29)
    cols = [c for c in [
        "strategy", "n_folds", "n_rows", "base_rate", "pr_auc", "pr_auc_std",
        "pr_auc_lift", "roc_auc", "brier", "prec@5%", "lift@5%",
        "lift_vs_prior", "lift_vs_logit",
    ] if c in result.summary.columns]
    print(result.summary[cols].to_string(index=False))

    if not result.fold_metrics.empty:
        print("\n--- per-fold PR-AUC " + "-" * 54)
        pivot = result.fold_metrics.pivot_table(
            index=["fold", "test_start", "test_end"],
            columns="strategy", values="pr_auc",
        )
        print(pivot.to_string())

    if result.tuning:
        print("\n--- tuned hyperparameters " + "-" * 48)
        for kind, info in result.tuning.items():
            print(f"{kind}: inner PR-AUC {info['best_value']:.5f} "
                  f"over {info['n_trials']} trials")
            for k, v in info["best_params"].items():
                print(f"    {k:20s} {v}")

    if result.shap is not None:
        print("\n--- SHAP importance, top 20 (held-out rows) " + "-" * 31)
        top = result.shap.importance.head(20)[
            ["feature", "mean_abs_shap", "share", "cumulative_share"]
        ]
        print(top.to_string(index=False))

    if result.calibration is not None and not result.calibration.empty:
        print("\n--- calibration of the best model " + "-" * 40)
        print(result.calibration.to_string(index=False))

    print(f"\nArtefacts written to {C.RESULTS_ROOT}")
    print("=" * 78 + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    universe = None
    if args.limit:
        universe = F.load_universe()[: args.limit]
        logging.info("Universe limited to %d symbols", len(universe))

    result = pipeline.run(
        universe=universe,
        use_cache=not args.rebuild,
        include_intraday=not args.no_intraday,
        n_trials=args.trials,
        kinds=tuple(k.strip() for k in args.models.split(",") if k.strip()),
        skip_tuning=args.skip_tuning,
        run_shap=not args.no_shap,
    )
    _print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
