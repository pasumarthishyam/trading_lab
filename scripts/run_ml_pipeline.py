"""
ML Pipeline Runner
==================

End-to-end orchestration of the two unsupervised systems in
``infrastructure/ml``: Isolation Forest anomaly detection over every
⟨stock, day⟩ session in the F&O universe, and K-Means regime
segmentation of the market calendar.

Every artefact is written to ``data/processed/ml/`` — feature matrices,
labels, serialised models (for scoring new sessions without refitting),
a self-contained HTML report, and a run manifest.

Usage
-----
    python scripts/run_ml_pipeline.py                  # full pipeline
    python scripts/run_ml_pipeline.py --anomaly-only   # anomaly detection only
    python scripts/run_ml_pipeline.py --cluster-only   # clustering only
    python scripts/run_ml_pipeline.py --reuse-features # skip feature rebuild
    python scripts/run_ml_pipeline.py --open           # open the report when done
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import joblib
import pandas as pd

from infrastructure.ml.anomaly import AnomalyResult, detect_anomalies
from infrastructure.ml.clustering import ClusterResult, cluster_market_days
from infrastructure.ml.dataset import (
    REPO_ROOT,
    build_market_day_features,
    build_stock_day_features,
)
from infrastructure.ml.report import build_ml_report

logger = logging.getLogger(__name__)

OUT_DIR: Path = REPO_ROOT / "data" / "processed" / "ml"

STOCK_FEATURES_PATH = OUT_DIR / "stock_day_features.parquet"
MARKET_FEATURES_PATH = OUT_DIR / "market_day_features.parquet"
ANOMALY_LABELS_PATH = OUT_DIR / "anomaly_labels.parquet"
CLUSTER_LABELS_PATH = OUT_DIR / "cluster_labels.parquet"
ANOMALY_MODEL_PATH = OUT_DIR / "isolation_forest.joblib"
CLUSTER_MODEL_PATH = OUT_DIR / "kmeans.joblib"
REPORT_PATH = OUT_DIR / "ml_report.html"
RUN_META_PATH = OUT_DIR / "run_meta.json"


# ── helpers ─────────────────────────────────────────────────────────

def _load_or_build(path: Path, builder, reuse: bool, what: str) -> pd.DataFrame:
    """Return a cached feature matrix when asked, else rebuild it."""
    if reuse and path.exists():
        df = pd.read_parquet(path)
        print(f"  [cached]  {what}: {len(df):,} rows  ({path.name})")
        return df
    t0 = time.time()
    df = builder()
    df.to_parquet(path, index=False)
    print(f"  [built]   {what}: {len(df):,} rows in {time.time() - t0:.1f}s")
    return df


def _save_anomaly_model(result: AnomalyResult) -> None:
    """Persist the fitted pipeline so new sessions can be scored later.

    Scaler and model travel together — scoring with a mismatched scaler
    silently produces garbage, so they are never saved separately.
    """
    joblib.dump(
        {
            "model": result.model,
            "scaler": result.scaler,
            "feature_names": result.feature_names,
            "score_threshold": result.stats["score_threshold"],
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "n_training_rows": result.stats["n_sessions"],
        },
        ANOMALY_MODEL_PATH,
    )


def _save_cluster_model(result: ClusterResult) -> None:
    joblib.dump(
        {
            "model": result.model,
            "scaler": result.scaler,
            "feature_names": result.feature_names,
            "cluster_names": result.stats["cluster_names"],
            "best_k": result.best_k,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "n_training_rows": result.stats["n_days"],
        },
        CLUSTER_MODEL_PATH,
    )


def _print_anomaly_summary(result: AnomalyResult) -> None:
    s = result.stats
    print()
    print("  Isolation Forest")
    print(f"    sessions scored     : {s['n_sessions']:,} "
          f"({s['n_symbols']} symbols, {s['date_min']} -> {s['date_max']})")
    print(f"    flagged             : {s['n_anomalies']:,} "
          f"({s['anomaly_rate'] * 100:.2f}%)")
    print(f"    known-event recall  : {s['known_events_detected']}/"
          f"{s['known_events_total']}  ({result.known_recall * 100:.0f}%)")
    print(f"    top features        : {', '.join(s['top3_features'])}")
    print()
    print("    ground truth:")
    for row in result.known_events.itertuples():
        mark = "PASS" if row.detected else "FAIL"
        pct = getattr(row, "score_percentile", float("nan"))
        print(f"      [{mark}] {row.symbol:<10s} "
              f"{pd.Timestamp(row.date):%Y-%m-%d}  "
              f"score={row.anomaly_score:+.4f}  top {pct:.3f}% most anomalous")


def _print_cluster_summary(result: ClusterResult) -> None:
    s = result.stats
    print()
    print("  K-Means")
    print(f"    market days         : {s['n_days']:,} "
          f"({s['date_min']} -> {s['date_max']})")
    print(f"    silhouette by K     : "
          + ", ".join(f"K={k}:{v:.3f}" for k, v in sorted(result.silhouette_scores.items())))
    print(f"    selected K          : {result.best_k} "
          f"(silhouette {s['best_silhouette']:.4f})")
    print(f"    stability (ARI)     : {result.stability_ari:.4f}")
    print()
    print("    regimes:")
    for cid in sorted(s["cluster_names"]):
        name = s["cluster_names"][cid]
        n = s["cluster_sizes"][cid]
        print(f"      {name:<32s} {n:>5,} days  ({n / s['n_days'] * 100:4.1f}%)")


def _scorecard(
    anomaly_result: AnomalyResult | None,
    cluster_result: ClusterResult | None,
) -> list[tuple[str, str, str, bool]]:
    """The pass/fail criteria, fixed before the models were fitted."""
    checks: list[tuple[str, str, str, bool]] = []

    if anomaly_result is not None:
        s = anomaly_result.stats
        recall = anomaly_result.known_recall
        rate = s["anomaly_rate"]
        top3 = s.get("top3_features", [])
        gap_top3 = any(
            f in ("abs_gap_pct", "gap_pct", "open_close_ratio") for f in top3
        )
        checks += [
            ("Known-event recall", "100%", f"{recall * 100:.0f}%", recall >= 1.0),
            ("Gap signature in top-3 features", "yes",
             "yes" if gap_top3 else "no", gap_top3),
            ("Anomaly budget", "0.1%-2%", f"{rate * 100:.2f}%",
             0.001 <= rate <= 0.02),
        ]

    if cluster_result is not None:
        s = cluster_result.stats
        n_unique = len(set(s["cluster_names"].values()))
        checks += [
            ("Silhouette score", "> 0.20", f"{s['best_silhouette']:.4f}",
             s["best_silhouette"] > 0.20),
            ("Distinct nameable regimes", f"{cluster_result.best_k}",
             f"{n_unique}", n_unique == cluster_result.best_k),
            ("Cluster stability (ARI)", "> 0.80",
             f"{cluster_result.stability_ari:.4f}",
             cluster_result.stability_ari > 0.80),
        ]

    return checks


def _print_scorecard(checks: list[tuple[str, str, str, bool]]) -> bool:
    print()
    print("=" * 72)
    print("  VALIDATION SCORECARD")
    print("=" * 72)
    print(f"  {'Check':<34s} {'Criteria':<10s} {'Actual':<10s} Result")
    print("  " + "-" * 68)
    for name, crit, actual, ok in checks:
        print(f"  {name:<34s} {crit:<10s} {actual:<10s} {'PASS' if ok else 'FAIL'}")
    passed = sum(1 for *_, ok in checks if ok)
    print("  " + "-" * 68)
    print(f"  {passed}/{len(checks)} checks passed")
    return passed == len(checks)


# ── main ────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the unsupervised ML pipeline (anomaly detection + clustering)",
    )
    ap.add_argument("--anomaly-only", action="store_true",
                    help="run only Isolation Forest anomaly detection")
    ap.add_argument("--cluster-only", action="store_true",
                    help="run only K-Means day clustering")
    ap.add_argument("--reuse-features", action="store_true",
                    help="reuse cached feature matrices instead of rebuilding")
    ap.add_argument("--contamination", default=None,
                    help="anomaly budget: a float (e.g. 0.005) or 'auto'")
    ap.add_argument("--open", action="store_true",
                    help="open the HTML report when finished")
    ap.add_argument("--verbose", action="store_true", help="enable INFO logging")
    args = ap.parse_args()

    if args.anomaly_only and args.cluster_only:
        ap.error("--anomaly-only and --cluster-only are mutually exclusive.")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s — %(message)s",
    )

    run_anomaly = not args.cluster_only
    run_cluster = not args.anomaly_only

    contamination: float | str = 0.005
    if args.contamination is not None:
        contamination = (
            args.contamination if args.contamination == "auto"
            else float(args.contamination)
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print()
    print("=" * 72)
    print("  UNSUPERVISED ML PIPELINE")
    print("=" * 72)

    anomaly_result: AnomalyResult | None = None
    cluster_result: ClusterResult | None = None
    stock_features: pd.DataFrame | None = None

    # ── anomaly detection ───────────────────────────────────────────
    if run_anomaly:
        print()
        print("  [1] Building stock-day feature matrix")
        stock_features = _load_or_build(
            STOCK_FEATURES_PATH, build_stock_day_features,
            args.reuse_features, "stock-day features",
        )

        print()
        print("  [2] Fitting Isolation Forest")
        t0 = time.time()
        anomaly_result = detect_anomalies(
            features=stock_features, contamination=contamination,
        )
        print(f"  [done]    fitted + scored in {time.time() - t0:.1f}s")

        anomaly_result.labels.to_parquet(ANOMALY_LABELS_PATH, index=False)
        _save_anomaly_model(anomaly_result)
        _print_anomaly_summary(anomaly_result)

    # ── clustering ──────────────────────────────────────────────────
    if run_cluster:
        print()
        print("  [3] Building market-day feature matrix")
        market_features = _load_or_build(
            MARKET_FEATURES_PATH, build_market_day_features,
            args.reuse_features, "market-day features",
        )

        print()
        print("  [4] Fitting K-Means")
        t0 = time.time()
        cluster_result = cluster_market_days(features=market_features)
        print(f"  [done]    K selected + fitted in {time.time() - t0:.1f}s")

        cluster_result.labels.to_parquet(CLUSTER_LABELS_PATH, index=False)
        _save_cluster_model(cluster_result)
        _print_cluster_summary(cluster_result)

    # ── report ──────────────────────────────────────────────────────
    report_path: Path | None = None
    if anomaly_result is not None and cluster_result is not None:
        print()
        print("  [5] Building HTML report")
        report_path = build_ml_report(
            anomaly_result, cluster_result, stock_features, REPORT_PATH,
        )
        print(f"  [done]    {report_path.name} "
              f"({report_path.stat().st_size / 1e6:.1f} MB, self-contained)")
    else:
        print()
        print("  [5] Report skipped - it covers both models; "
              "run without --anomaly-only/--cluster-only to build it.")

    # ── manifest ────────────────────────────────────────────────────
    checks = _scorecard(anomaly_result, cluster_result)
    all_passed = _print_scorecard(checks)

    meta = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "ran_anomaly": run_anomaly,
        "ran_cluster": run_cluster,
        "contamination": contamination,
        "checks": [
            {"name": n, "criteria": c, "actual": a, "passed": ok}
            for n, c, a, ok in checks
        ],
        "all_checks_passed": all_passed,
        "anomaly_stats": anomaly_result.stats if anomaly_result else None,
        "cluster_stats": cluster_result.stats if cluster_result else None,
        "anomaly_feature_importance": (
            {k: float(v) for k, v in anomaly_result.feature_importances.items()}
            if anomaly_result else None
        ),
        "silhouette_scores": (
            {str(k): float(v) for k, v in cluster_result.silhouette_scores.items()}
            if cluster_result else None
        ),
    }
    RUN_META_PATH.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print()
    print(f"  Artefacts written to {OUT_DIR}")
    print(f"  Total elapsed: {time.time() - t_start:.1f}s")
    print()

    if args.open and report_path is not None:
        webbrowser.open(report_path.as_uri())

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
