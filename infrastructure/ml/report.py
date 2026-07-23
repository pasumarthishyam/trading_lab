"""
ML Module — Static HTML Report
==============================

Renders the Isolation Forest and K-Means results into a single
self-contained, interactive HTML file — no kernel, no Jupyter, no
server.  Double-click it; it opens in any browser and works offline
(plotly.js is inlined once).

Colour follows the project's validated data-viz palette: one hue per
series for nominal categories, a single-hue ramp for magnitude, and a
blue↔red diverging scale with a neutral grey midpoint for the
standardised centroids (where zero genuinely means "average day").
Cluster hues follow the cluster, not its rank, so a regime keeps its
colour across every figure.

Usage
-----
    from infrastructure.ml.report import build_ml_report
    build_ml_report(anomaly_result, cluster_result, features)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from infrastructure.ml.anomaly import AnomalyResult, anomalies_per_symbol, top_anomalies
from infrastructure.ml.clustering import ClusterResult
from infrastructure.ml.dataset import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_REPORT_PATH: Path = REPO_ROOT / "data" / "processed" / "ml" / "ml_report.html"

_TEMPLATE = "plotly_white"

# ── palette (validated: see dataviz validate_palette.js) ────────────
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

# Categorical slots 1-3 — the only three needed (one per regime).
# This triple clears the all-pairs CVD and normal-vision floors.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
# Single hue for magnitude / one-series bars.
SEQ_BLUE = "#2a78d6"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

# Diverging: one hue per arm, light→dark outward, neutral grey midpoint.
DIVERGING = [
    [0.00, "#184f95"],
    [0.25, "#6da7ec"],
    [0.50, "#f0efec"],
    [0.75, "#ef8f8e"],
    [1.00, "#a82f2e"],
]

# Human-readable feature labels for chart axes.
_FEATURE_LABELS: dict[str, str] = {
    "return_pct": "Daily return",
    "abs_return_pct": "|Daily return|",
    "gap_pct": "Overnight gap",
    "abs_gap_pct": "|Overnight gap|",
    "range_pct": "Intraday range",
    "body_pct": "Candle body",
    "upper_wick_pct": "Upper wick",
    "lower_wick_pct": "Lower wick",
    "volume_ratio_20d": "Volume ratio (20d)",
    "volume_zscore_20d": "Volume z-score (20d)",
    "range_zscore_20d": "Range z-score (20d)",
    "return_zscore_20d": "Return z-score (20d)",
    "open_close_ratio": "Open / prev close",
    "session_range_pct": "Session range",
    "dvr_ratio": "DVR ratio",
    "vix_level": "VIX level",
    "vix_change_pct": "VIX change",
    "upper_wick_ratio": "Upper wick ratio",
    "lower_wick_ratio": "Lower wick ratio",
}


def _label(feature: str) -> str:
    return _FEATURE_LABELS.get(feature, feature)


# ── small helpers ───────────────────────────────────────────────────

def _fig_div(fig: go.Figure, height: int = 430) -> str:
    """Figure → HTML div (plotly.js injected once globally, not per figure)."""
    fig.update_layout(
        template=_TEMPLATE,
        margin=dict(l=70, r=40, t=50, b=70),
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            size=12,
            color=INK_SECONDARY,
        ),
        title=dict(font=dict(size=14, color=INK_PRIMARY)),
        legend=dict(orientation="h", y=-0.18, font=dict(color=INK_SECONDARY)),
        hoverlabel=dict(font_size=12),
    )
    # Recessive, solid hairline grid — never dashed.
    fig.update_xaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE,
                     linecolor=GRIDLINE, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE,
                     linecolor=GRIDLINE, tickfont=dict(color=INK_MUTED))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False, "displayModeBar": False})


def _table_html(df: pd.DataFrame, pct_cols=(), round_cols=(), int_cols=()) -> str:
    d = df.copy()
    for c in pct_cols:
        if c in d:
            d[c] = (d[c] * 100).round(2).astype(str) + "%"
    for c in round_cols:
        if c in d:
            d[c] = d[c].round(4)
    for c in int_cols:
        if c in d:
            d[c] = d[c].round(0).astype("Int64")
    return d.to_html(index=False, classes="rtab", border=0, na_rep="—")


def _stat_tile(value: str, label: str, ok: Optional[bool] = None) -> str:
    """A headline number.  The number *is* the chart — no one-bar bar charts."""
    cls = "" if ok is None else (" good" if ok else " bad")
    return (
        f"<div class='tile{cls}'><div class='tile-v'>{value}</div>"
        f"<div class='tile-l'>{label}</div></div>"
    )


def _cluster_colors(cluster_ids: list[int]) -> dict[int, str]:
    """Fixed hue per cluster — colour follows the entity, never its rank."""
    return {cid: SERIES[i % len(SERIES)] for i, cid in enumerate(sorted(cluster_ids))}


# ── anomaly figures ─────────────────────────────────────────────────

def _fig_feature_importance(importances: pd.Series) -> go.Figure:
    """Horizontal bar, single hue — these are nominal categories, so the
    bar length carries magnitude and colour carries nothing."""
    s = importances.sort_values()
    pct = s * 100
    fig = go.Figure()
    fig.add_bar(
        x=pct.to_numpy(),
        y=[_label(f) for f in s.index],
        orientation="h",
        marker_color=SEQ_BLUE,
        text=[f"{v:.1f}%" for v in pct],
        textposition="outside",
        textfont=dict(color=INK_SECONDARY, size=11),
        hovertemplate="%{y}<br>%{x:.2f}% of total<extra></extra>",
    )
    fig.update_layout(
        title="What drives the anomaly verdicts — permutation importance on flagged sessions",
        xaxis_title="Share of total score sensitivity (%)",
    )
    fig.update_xaxes(range=[0, float(pct.max()) * 1.18])
    return fig


def _fig_score_distribution(result: AnomalyResult) -> go.Figure:
    scores = result.labels["anomaly_score"].to_numpy()
    threshold = float(result.stats["score_threshold"])
    fig = go.Figure()
    fig.add_histogram(
        x=scores, nbinsx=120, marker_color=SEQ_BLUE,
        name="sessions",
        hovertemplate="score %{x:.4f}<br>%{y} sessions<extra></extra>",
    )
    # A threshold rule is a real threshold — dashing is meaningful here.
    fig.add_vline(
        x=threshold, line=dict(color=STATUS_CRITICAL, width=2, dash="dash"),
        annotation_text=f"  decision threshold ({threshold:.3f})",
        annotation_position="top right",
        annotation_font=dict(color=STATUS_CRITICAL, size=11),
    )
    fig.update_layout(
        title="Anomaly score distribution — everything left of the threshold is flagged",
        xaxis_title="Isolation Forest decision score (lower = more anomalous)",
        yaxis_title="Sessions",
        showlegend=False,
    )
    fig.update_yaxes(type="log", title="Sessions (log scale)")
    return fig


def _fig_anomalies_per_symbol(counts: pd.DataFrame) -> go.Figure:
    d = counts.sort_values("n_anomalies")
    fig = go.Figure()
    fig.add_bar(
        x=d["n_anomalies"].to_numpy(), y=d["symbol"].to_numpy(),
        orientation="h", marker_color=SEQ_BLUE,
        text=d["n_anomalies"].to_numpy(), textposition="outside",
        textfont=dict(color=INK_SECONDARY, size=11),
        hovertemplate="%{y}<br>%{x} flagged sessions<extra></extra>",
    )
    fig.update_layout(
        title="Most-flagged symbols",
        xaxis_title="Flagged sessions",
    )
    fig.update_xaxes(range=[0, float(d["n_anomalies"].max()) * 1.15])
    return fig


def _fig_anomaly_timeline(result: AnomalyResult) -> go.Figure:
    """Monthly counts — 1,500+ individual points would be an unreadable
    smear; the monthly aggregate is what actually shows the regimes."""
    flagged = result.labels[result.labels["is_anomaly"]].copy()
    flagged["month"] = pd.to_datetime(flagged["date"]).dt.to_period("M").dt.to_timestamp()
    monthly = flagged.groupby("month").size().rename("n").reset_index()

    fig = go.Figure()
    fig.add_bar(
        x=monthly["month"], y=monthly["n"], marker_color=SEQ_BLUE,
        hovertemplate="%{x|%b %Y}<br>%{y} flagged sessions<extra></extra>",
        name="flagged",
    )
    fig.update_layout(
        title="Flagged sessions per month — market-wide stress clusters in time",
        xaxis_title="Month", yaxis_title="Flagged sessions",
        showlegend=False,
    )
    return fig


def _fig_known_events(result: AnomalyResult) -> go.Figure:
    """Where the four ground-truth events sit in the score distribution."""
    ev = result.known_events.copy()
    if ev.empty or "score_percentile" not in ev.columns:
        return go.Figure()
    ev = ev.sort_values("score_percentile")
    labels = [
        f"{r.symbol} {pd.Timestamp(r.date):%Y-%m-%d}" for r in ev.itertuples()
    ]
    fig = go.Figure()
    fig.add_bar(
        x=ev["score_percentile"].to_numpy(), y=labels, orientation="h",
        marker_color=[STATUS_GOOD if d else STATUS_CRITICAL for d in ev["detected"]],
        text=[f"top {v:.3f}%" for v in ev["score_percentile"]],
        textposition="outside", textfont=dict(color=INK_SECONDARY, size=11),
        hovertemplate="%{y}<br>more anomalous than %{customdata:.3f}% of sessions"
                      "<extra></extra>",
        customdata=(100 - ev["score_percentile"]).to_numpy(),
    )
    fig.update_layout(
        title="Ground-truth corporate actions — position in the anomaly ranking "
              "(lower percentile = more anomalous)",
        xaxis_title="Score percentile across all sessions (%)",
        showlegend=False,
    )
    fig.update_xaxes(range=[0, max(float(ev["score_percentile"].max()) * 1.6, 0.5)])
    return fig


# ── clustering figures ──────────────────────────────────────────────

def _fig_silhouette(result: ClusterResult) -> go.Figure:
    ks = sorted(result.silhouette_scores)
    vals = [result.silhouette_scores[k] for k in ks]
    fig = go.Figure()
    fig.add_scatter(
        x=ks, y=vals, mode="lines+markers",
        line=dict(color=SEQ_BLUE, width=2),
        marker=dict(size=9, color=SEQ_BLUE),
        name="silhouette",
        hovertemplate="K=%{x}<br>silhouette %{y:.4f}<extra></extra>",
    )
    # Emphasis: highlight the selected K rather than labelling every point.
    fig.add_scatter(
        x=[result.best_k], y=[result.silhouette_scores[result.best_k]],
        mode="markers+text", marker=dict(size=15, color=SEQ_BLUE,
                                         line=dict(width=2, color=SURFACE)),
        text=[f" selected K={result.best_k}"], textposition="middle right",
        textfont=dict(color=INK_PRIMARY, size=12),
        showlegend=False, hoverinfo="skip",
    )
    fig.update_layout(
        title="K selection — mean silhouette score by cluster count",
        xaxis_title="K (number of clusters)", yaxis_title="Mean silhouette",
        showlegend=False,
    )
    fig.update_xaxes(dtick=1)
    return fig


def _fig_cluster_heatmap(result: ClusterResult) -> go.Figure:
    """Standardised centroids — zero means 'an average day', so a diverging
    scale with a neutral midpoint is the honest encoding."""
    z = result.centroid_z
    names = [result.stats["cluster_names"][int(i)] for i in z.index]
    limit = float(np.abs(z.to_numpy()).max())
    fig = go.Figure()
    fig.add_heatmap(
        z=z.to_numpy(),
        x=[_label(c) for c in z.columns],
        y=names,
        colorscale=DIVERGING, zmid=0, zmin=-limit, zmax=limit,
        colorbar=dict(title=dict(text="z", side="right"), thickness=12),
        hovertemplate="%{y}<br>%{x}: %{z:+.2f} SD from average<extra></extra>",
    )
    fig.update_layout(
        title="Regime fingerprints — standardised deviation from the average day",
        xaxis_title="", yaxis_title="",
    )
    fig.update_xaxes(tickangle=-35, showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def _fig_cluster_sizes(result: ClusterResult) -> go.Figure:
    sizes = result.profiles[["cluster_name", "n_days"]].reset_index()
    colors = _cluster_colors([int(i) for i in result.profiles.index])
    fig = go.Figure()
    for row in sizes.itertuples():
        fig.add_bar(
            x=[row.cluster_name], y=[row.n_days],
            marker_color=colors[int(row.cluster_id)],
            name=row.cluster_name,
            text=[f"{row.n_days}"], textposition="outside",
            textfont=dict(color=INK_SECONDARY, size=11),
            hovertemplate="%{x}<br>%{y} days<extra></extra>",
        )
    fig.update_layout(
        title="Days per regime", yaxis_title="Trading days", showlegend=True,
    )
    return fig


def _fig_cluster_timeline(result: ClusterResult) -> go.Figure:
    """Monthly regime mix — how the market's character shifts over time."""
    lab = result.labels.copy()
    lab["month"] = pd.to_datetime(lab["date"]).dt.to_period("M").dt.to_timestamp()
    counts = lab.groupby(["month", "cluster_id"]).size().unstack(fill_value=0)
    share = counts.div(counts.sum(axis=1), axis=0)
    colors = _cluster_colors([int(c) for c in counts.columns])

    fig = go.Figure()
    for cid in counts.columns:
        name = result.stats["cluster_names"][int(cid)]
        fig.add_bar(
            x=share.index, y=share[cid] * 100, name=name,
            marker=dict(
                color=colors[int(cid)],
                # 1px surface line = the 2px visual gap between segments.
                line=dict(width=1, color=SURFACE),
            ),
            hovertemplate="%{x|%b %Y}<br>" + name + ": %{y:.0f}% of days<extra></extra>",
        )
    fig.update_layout(
        barmode="stack",
        title="Regime mix by month — share of trading days in each regime",
        xaxis_title="Month", yaxis_title="Share of days (%)",
    )
    fig.update_yaxes(range=[0, 100])
    return fig


# ── assembly ────────────────────────────────────────────────────────

def build_ml_report(
    anomaly_result: AnomalyResult,
    cluster_result: ClusterResult,
    stock_features: Optional[pd.DataFrame] = None,
    report_path: Optional[Path] = None,
) -> Path:
    """Build the self-contained HTML report for both ML modules.

    Parameters
    ----------
    anomaly_result : AnomalyResult
    cluster_result : ClusterResult
    stock_features : DataFrame, optional
        Stock-day feature matrix, used to annotate the top-anomaly table.
    report_path : Path, optional
        Destination.  Defaults to ``data/processed/ml/ml_report.html``.
    """
    path = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    a_stats = anomaly_result.stats
    c_stats = cluster_result.stats

    sections: list[tuple[str, str, str]] = []

    # ── validation scorecard ────────────────────────────────────────
    recall = anomaly_result.known_recall
    rate = a_stats["anomaly_rate"]
    top3 = a_stats.get("top3_features", [])
    gap_in_top3 = any(f in ("abs_gap_pct", "gap_pct", "open_close_ratio") for f in top3)
    sil = c_stats["best_silhouette"]
    ari = c_stats["stability_ari"]

    checks = [
        ("Known-event recall",
         f"{a_stats['known_events_detected']}/{a_stats['known_events_total']} "
         f"corporate actions flagged", "100%", f"{recall * 100:.0f}%", recall >= 1.0),
        ("Feature-importance sanity",
         "A gap / split signature ranks top-3", "yes",
         "yes" if gap_in_top3 else "no", gap_in_top3),
        ("Anomaly budget",
         "Flagged share of all sessions", "0.1% – 2%",
         f"{rate * 100:.2f}%", 0.001 <= rate <= 0.02),
        ("Silhouette score",
         f"Best K = {cluster_result.best_k}", "> 0.20", f"{sil:.4f}", sil > 0.20),
        ("Cluster interpretability",
         "Distinct, nameable regime profiles",
         f"{cluster_result.best_k} distinct",
         f"{len(set(c_stats['cluster_names'].values()))} distinct",
         len(set(c_stats["cluster_names"].values())) == cluster_result.best_k),
        ("Cluster stability",
         "Mean pairwise ARI across 5 seeds", "> 0.80", f"{ari:.4f}", ari > 0.80),
    ]
    rows = "".join(
        f"<tr><td>{name}</td><td>{desc}</td><td>{crit}</td>"
        f"<td><b>{actual}</b></td>"
        f"<td class='{'pass' if ok else 'fail'}'>{'PASS' if ok else 'FAIL'}</td></tr>"
        for name, desc, crit, actual, ok in checks
    )
    n_passed = sum(1 for *_, ok in checks if ok)
    scorecard = (
        f"<table class='rtab scorecard'><thead><tr><th>Check</th><th>What it measures</th>"
        f"<th>Pass criteria</th><th>Actual</th><th>Result</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
    tiles = (
        "<div class='tiles'>"
        + _stat_tile(f"{recall * 100:.0f}%", "Known-event recall", recall >= 1.0)
        + _stat_tile(f"{a_stats['n_anomalies']:,}", "Sessions flagged")
        + _stat_tile(f"{rate * 100:.2f}%", "Anomaly rate", 0.001 <= rate <= 0.02)
        + _stat_tile(f"{sil:.3f}", "Silhouette", sil > 0.20)
        + _stat_tile(f"{ari:.3f}", "Stability (ARI)", ari > 0.80)
        + "</div>"
    )
    sections.append((
        "scorecard", "Validation scorecard",
        "<p class='desc'>Every criterion fixed <b>before</b> the models were fitted. "
        "The known-event check is the hard one: the detector is never told that "
        "corporate actions exist — it has to isolate them from session geometry alone."
        f"</p>{tiles}{scorecard}"))

    # ── anomaly sections ────────────────────────────────────────────
    ev = anomaly_result.known_events.copy()
    if not ev.empty:
        ev_display = pd.DataFrame({
            "symbol": ev["symbol"],
            "date": pd.to_datetime(ev["date"]).dt.strftime("%Y-%m-%d"),
            "open/prev_close": ev["ratio"].round(4) if "ratio" in ev else np.nan,
            "anomaly_score": ev["anomaly_score"].round(4),
            "score_percentile": ev["score_percentile"].round(4),
            "detected": np.where(ev["detected"], "PASS", "FAIL"),
        })
        ev_tbl = _table_html(ev_display)
    else:
        ev_tbl = "<p class='desc'>No registered corporate actions found.</p>"

    sections.append((
        "known", "1 · Ground-truth validation",
        "<p class='desc'>The four registered corporate actions from "
        "<code>_corporate_actions.json</code>. A demerger or split leaves an "
        "extreme open-to-previous-close discontinuity, so a competent detector "
        "must isolate them. <b>Score percentile</b> is the share of all sessions "
        "scoring lower — 0.02% means only ~1 session in 5,000 looked stranger."
        f"</p>{ev_tbl}{_fig_div(_fig_known_events(anomaly_result), 330)}"))

    sections.append((
        "importance", "2 · Feature importance",
        "<p class='desc'>Permutation importance measured <b>on the flagged "
        "sessions</b> — each feature is shuffled and the mean absolute shift in "
        "the model's own decision score is recorded. Measured across all sessions "
        "instead, importance is nearly uniform: 99.5% of rows are unremarkable on "
        "every axis at once, so that view answers a less useful question."
        f"</p>{_fig_div(_fig_feature_importance(anomaly_result.feature_importances), 470)}"))

    sections.append((
        "scores", "3 · Score distribution",
        "<p class='desc'>Decision scores for all "
        f"{a_stats['n_sessions']:,} sessions. The long left tail is the "
        "signal — a thin population of genuinely strange days, not a "
        "second mode. Counts are log-scaled so the tail stays visible."
        f"</p>{_fig_div(_fig_score_distribution(anomaly_result))}"))

    if stock_features is not None:
        top_tbl = _table_html(
            top_anomalies(anomaly_result, stock_features, 20).assign(
                date=lambda d: pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d"),
            ).drop(columns=["is_anomaly"]),
            pct_cols=["gap_pct", "return_pct", "range_pct"],
            round_cols=["anomaly_score", "volume_ratio_20d", "open_close_ratio"],
        )
        sections.append((
            "top", "4 · Most anomalous sessions",
            "<p class='desc'>The 20 strangest ⟨stock, day⟩ pairs in the "
            "dataset. These are recognisable market events — index-shock days, "
            "results crashes and circuit-limit moves — not data errors."
            f"</p>{top_tbl}"))

    sections.append((
        "bysymbol", "5 · Anomalies by symbol",
        "<p class='desc'>Flag counts concentrate in genuinely news-driven "
        "names rather than spreading evenly, which is what a working detector "
        "should do."
        f"</p>{_fig_div(_fig_anomalies_per_symbol(anomalies_per_symbol(anomaly_result, 20)), 500)}"))

    sections.append((
        "timeline", "6 · Anomalies over time",
        "<p class='desc'>Flagged sessions clustered by month. Market-wide "
        "stress shows up as a spike, not a constant background rate."
        f"</p>{_fig_div(_fig_anomaly_timeline(anomaly_result))}"))

    # ── clustering sections ─────────────────────────────────────────
    sections.append((
        "silhouette", "7 · K selection",
        "<p class='desc'>Every K in the requested range is scored by mean "
        "silhouette and the best wins — K is <b>selected</b>, not assumed."
        f"</p>{_fig_div(_fig_silhouette(cluster_result))}"))

    sections.append((
        "profiles", "8 · Regime fingerprints",
        "<p class='desc'>Each regime's standardised centroid. Zero is an "
        "average day, so the diverging scale reads as "
        "below-average ↔ above-average. Regime names are derived from these "
        "coordinates, not hand-assigned."
        f"</p>{_fig_div(_fig_cluster_heatmap(cluster_result), 380)}"))

    profile_cols = ["cluster_name", "n_days", "pct_days"] + [
        f"{c}_mean" for c in cluster_result.feature_names
    ]
    prof_tbl = _table_html(
        cluster_result.profiles[profile_cols].reset_index(),
        pct_cols=["pct_days"],
        round_cols=[f"{c}_mean" for c in cluster_result.feature_names],
    )
    sections.append((
        "profiletable", "9 · Regime profiles (original units)",
        "<p class='desc'>The same centroids in original units — the "
        "table view of the fingerprint above."
        f"</p>{prof_tbl}"))

    sections.append((
        "sizes", "10 · Regime sizes",
        "<p class='desc'>How the trading calendar splits across regimes."
        f"</p>{_fig_div(_fig_cluster_sizes(cluster_result), 380)}"))

    sections.append((
        "regimetime", "11 · Regime mix over time",
        "<p class='desc'>Monthly share of days in each regime. Crisis "
        "windows appear as a regime shift rather than a handful of odd days."
        f"</p>{_fig_div(_fig_cluster_timeline(cluster_result))}"))

    # ── header ──────────────────────────────────────────────────────
    verdict = (
        f"<b>{n_passed}/{len(checks)}</b> validation checks passed &nbsp;·&nbsp; "
        f"Isolation Forest recovered <b>{a_stats['known_events_detected']}/"
        f"{a_stats['known_events_total']}</b> ground-truth corporate actions "
        f"from {a_stats['n_sessions']:,} sessions &nbsp;·&nbsp; "
        f"K-Means found <b>{cluster_result.best_k}</b> stable regimes "
        f"(ARI {ari:.2f})"
    )
    meta_html = (
        f"<div class='meta'><b>Sessions:</b> {a_stats['n_sessions']:,} "
        f"({a_stats['n_symbols']} F&amp;O symbols, {a_stats['date_min']} → "
        f"{a_stats['date_max']}) &nbsp;|&nbsp; "
        f"<b>Market days:</b> {c_stats['n_days']:,} "
        f"({c_stats['date_min']} → {c_stats['date_max']}) &nbsp;|&nbsp; "
        f"<b>Seed:</b> {a_stats['random_state']} &nbsp;|&nbsp; "
        f"<b>Built:</b> {datetime.now():%Y-%m-%d %H:%M}</div>"
        f"<div class='verdict'>{verdict}</div>"
    )

    toc = "".join(f"<li><a href='#{a}'>{t}</a></li>" for a, t, _ in sections)
    body = "".join(
        f"<section id='{a}'><h2>{t} <a class='top' href='#toc'>↑ top</a></h2>{html}</section>"
        for a, t, html in sections
    )

    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Unsupervised ML — Anomaly Detection &amp; Day Clustering</title>
<script>{get_plotlyjs()}</script>
<style>
 body{{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:0;
      background:#f9f9f7;color:{INK_PRIMARY}}}
 header{{background:#10243e;color:#fff;padding:22px 32px}}
 header h1{{margin:0 0 6px;font-size:21px}}
 header>div{{font-size:14px;color:#e8eef7}}
 .meta{{font-size:13px;color:#cfe0f5;margin-top:8px}}
 .verdict{{margin-top:10px;font-size:15px;background:#1b3a63;padding:10px 14px;
          border-radius:6px;color:#fff}}
 #toc{{position:sticky;top:0;background:{SURFACE};border-bottom:1px solid {GRIDLINE};
       padding:12px 32px;z-index:9}}
 #toc ul{{margin:0;padding:0;list-style:none;display:flex;flex-wrap:wrap;gap:8px 18px}}
 #toc a{{text-decoration:none;color:#10243e;font-size:13px;font-weight:600}}
 #toc a:hover{{text-decoration:underline}}
 section{{padding:22px 32px;border-bottom:1px solid #ececec;background:{SURFACE}}}
 h2{{font-size:18px;color:#10243e;margin:0 0 6px}}
 h2 .top{{font-size:11px;font-weight:400;color:{INK_MUTED};margin-left:10px}}
 .desc{{color:{INK_SECONDARY};font-size:14px;margin:0 0 12px;max-width:900px;
        line-height:1.5}}
 code{{background:#f0efec;padding:1px 5px;border-radius:3px;font-size:12px}}
 .tiles{{display:flex;flex-wrap:wrap;gap:12px;margin:0 0 16px}}
 .tile{{background:{SURFACE};border:1px solid {GRIDLINE};border-radius:8px;
        padding:12px 18px;min-width:130px}}
 .tile-v{{font-size:26px;font-weight:600;color:{INK_PRIMARY};line-height:1.15}}
 .tile-l{{font-size:12px;color:{INK_MUTED};margin-top:2px}}
 .tile.good .tile-v{{color:{STATUS_GOOD}}}
 .tile.bad .tile-v{{color:{STATUS_CRITICAL}}}
 .tabwrap{{overflow-x:auto}}
 table.rtab{{border-collapse:collapse;font-size:13px;background:{SURFACE};
             font-variant-numeric:tabular-nums}}
 table.rtab th,table.rtab td{{border:1px solid {GRIDLINE};padding:5px 10px;
                              text-align:right}}
 table.rtab th{{background:#10243e;color:#fff;font-weight:600}}
 table.rtab td:first-child,table.rtab th:first-child{{text-align:left}}
 table.scorecard td:nth-child(2){{text-align:left}}
 td.pass{{color:{STATUS_GOOD};font-weight:700}}
 td.fail{{color:{STATUS_CRITICAL};font-weight:700}}
</style></head><body>
<header><h1>Unsupervised ML — Anomaly Detection &amp; Day Clustering</h1>
 <div>Isolation Forest over every ⟨stock, day⟩ session in the F&amp;O universe,
 and K-Means regime segmentation of the market calendar.</div>
 {meta_html}</header>
<nav id='toc'><ul>{toc}</ul></nav>
{body}
<footer style='padding:18px 32px;color:{INK_MUTED};font-size:12px'>
 Generated from the ML pipeline · regenerate:
 <code>python scripts/run_ml_pipeline.py</code></footer>
</body></html>"""

    path.write_text(doc, encoding="utf-8")
    logger.info("Report written to %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path
