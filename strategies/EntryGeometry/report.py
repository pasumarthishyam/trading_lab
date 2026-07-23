"""
Pass 1 — Distribution & Audit Report
====================================

Self-contained interactive HTML (``results/pass1_report.html``) plus the
distribution table and audit-sample CSVs.  No kernel, no Colab: regenerate
with one command, double-click to open.

    python strategies/EntryGeometry/report.py
    python strategies/EntryGeometry/report.py --open
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from strategies.EntryGeometry.config import (
    CONFIG, TRIGGERS_PATH, RUN_META_PATH, REPORT_PATH, DISTRIB_PATH, AUDIT_SAMPLE_PATH,
)
from strategies.EntryGeometry import analytics as A

_TEMPLATE = "plotly_white"


def _div(fig):
    fig.update_layout(template=_TEMPLATE, margin=dict(l=60, r=40, t=50, b=50),
                      height=380, legend=dict(orientation="h", y=-0.22))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False})


def _table(df, pct_cols=(), round_cols=(), int_cols=()):
    d = df.copy()
    for c in pct_cols:
        if c in d: d[c] = (d[c] * 100).round(1).astype(str) + "%"
    for c in round_cols:
        if c in d: d[c] = d[c].round(3)
    for c in int_cols:
        if c in d: d[c] = d[c].round(0).astype("Int64")
    return d.to_html(index=False, classes="rtab", border=0, na_rep="—")


def _hist(trig, col, title, vlines=()):
    vals, hi = A.histogram_data(trig, col)
    fig = go.Figure()
    fig.add_histogram(x=vals, nbinsx=50, marker_color="#3b6ea5")
    for v, lbl in vlines:
        if v <= hi:
            fig.add_vline(x=v, line=dict(color="crimson", dash="dash"),
                          annotation_text=lbl, annotation_position="top")
    fig.update_layout(title=title, xaxis_title=col, yaxis_title="triggers")
    return fig


def build_report() -> Path:
    trig = pd.read_parquet(TRIGGERS_PATH)
    meta = json.loads(Path(RUN_META_PATH).read_text(encoding="utf-8"))
    n_days = meta["n_test_days"]

    DISTRIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    perc = A.percentile_table(trig)
    readoff = A.huge_threshold_readoff(trig)
    sanity = A.sanity_stats(trig, n_days)
    audit = A.audit_sample(trig)

    # persist substrate CSVs
    perc.to_csv(DISTRIB_PATH, index=False)
    audit.to_csv(AUDIT_SAMPLE_PATH, index=False)

    sections = []

    # ── huge-candle read-off (the point of the pass) ────────────────
    readoff_tbl = _table(readoff, pct_cols=["pct_over_1x", "pct_over_1_5x", "pct_over_2x"],
                         round_cols=["median", "p75", "p90", "p95"])
    sections.append(("huge", "1 · 'Huge candle' read-off (the threshold lives here)",
        "<p class='desc'>The breakout-candle size relative to ATR (<b>atr_ratio</b>) and to "
        "trailing average range (<b>range_ratio</b>). Read the cutoff off the shape — e.g. "
        "the top decile (&gt;p90) are the thrusts.</p>" + readoff_tbl
        + _div(_hist(trig, "atr_ratio", "atr_ratio = bo_range / ATR(14)",
                     vlines=[(1, "1×"), (1.5, "1.5×"), (2, "2×")]))
        + _div(_hist(trig, "range_ratio", "range_ratio = bo_range / avg_range(14)",
                     vlines=[(1, "1×"), (2, "2×"), (3, "3×")]))))

    # ── full percentile table ───────────────────────────────────────
    sections.append(("perc", "2 · Percentile table (all metrics)",
        "<p class='desc'>Percentiles for the huge-candle metrics and the three (now four) "
        "stop-distance candidates. This CSV is saved as <code>pass1_distributions.csv</code>.</p>"
        + _table(perc, round_cols=["mean", "std"] + [f"p{int(p*100)}" for p in A.PCTS],
                 int_cols=["n"])))

    # ── stop distances ──────────────────────────────────────────────
    stop_hists = "".join(_div(_hist(trig, c, c)) for c in A.STOP_METRICS)
    sections.append(("stops", "3 · Stop-distance candidates (raw — none chosen)",
        "<p class='desc'>Distance from entry to each candidate stop, as a % of entry. "
        "Both swing variants (wick &amp; body), the 3rd-candle-back, and the breakout-candle "
        "stop are recorded so the stop rule can be chosen later from these shapes.</p>"
        + stop_hists))

    # ── direction / momentum context ────────────────────────────────
    pv = sanity["pos_vs_open"]
    fig_dir = go.Figure()
    cats = ["long", "short"]
    fig_dir.add_bar(x=cats, y=[pv.get("long_above", 0), pv.get("short_above", 0)],
                    name="above 9:15 open", marker_color="seagreen")
    fig_dir.add_bar(x=cats, y=[pv.get("long_below", 0), pv.get("short_below", 0)],
                    name="below 9:15 open", marker_color="indianred")
    fig_dir.update_layout(barmode="stack", title="Trigger direction vs position to 9:15 open",
                          yaxis_title="triggers")
    sections.append(("dir", "4 · Direction & momentum context",
        "<p class='desc'>Long/short split and where the entry sits vs the 9:15 open. "
        "Counter-trend cases (long-below, short-above) are surfaced here for the audit — "
        "<b>not</b> filtered.</p>"
        f"<p class='desc'>Long: <b>{sanity['long']}</b> &nbsp; Short: <b>{sanity['short']}</b></p>"
        + _div(fig_dir)))

    # ── triggers per day ────────────────────────────────────────────
    per_day = trig.groupby("date").size()
    figd = go.Figure(); figd.add_histogram(x=per_day.values, nbinsx=int(per_day.max()) or 1,
                                           marker_color="#6a51a3")
    figd.update_layout(title=f"Triggers per day (mean {per_day.mean():.1f})",
                       xaxis_title="triggers in a day", yaxis_title="days")
    sections.append(("perday", "5 · Triggers per day",
        f"<p class='desc'>{sanity['days_with_trigger']}/{n_days} days had ≥1 trigger "
        f"({sanity['pct_days_with_trigger']:.0f}%). Mean {sanity['triggers_per_day_mean']:.1f}, "
        f"median {sanity['triggers_per_day_median']:.0f}, max {sanity['triggers_per_day_max']}.</p>"
        + _div(figd)))

    # ── audit sample table ──────────────────────────────────────────
    a = audit.copy()
    a["date"] = a["date"].astype(str).str[:10]
    for c in ["level_price", "entry_price", "bo_range", "atr_at_entry"]:
        a[c] = a[c].round(2)
    for c in ["atr_ratio", "range_ratio"]:
        a[c] = a[c].round(2)
    sections.append(("audit", "6 · Audit sample (open these on TradingView)",
        "<p class='desc'>The most extreme atr_ratio rows (candidate 'huge' candles) plus a "
        "random normal sample. Open each on TradingView using only candles up to "
        "<code>trigger_time</code> and confirm the level sat on a real swing and a 5-min candle "
        "truly closed beyond the body-extreme. Saved as <code>pass1_audit_sample.csv</code>.</p>"
        + a.to_html(index=False, classes="rtab", border=0)))

    # ── assemble ────────────────────────────────────────────────────
    toc = "".join(f"<li><a href='#{a_}'>{t}</a></li>" for a_, t, _ in sections)
    body = "".join(f"<section id='{a_}'><h2>{t} <a class='top' href='#toc'>↑</a></h2>{h}</section>"
                   for a_, t, h in sections)
    meta_html = (
        f"<div class='meta'><b>Window:</b> {meta['test_start']} → {meta['test_end']} "
        f"({n_days} days) &nbsp;|&nbsp; <b>Basket:</b> top-{CONFIG['basket_size']} R-rank @ "
        f"{CONFIG['basket_lock_time']} &nbsp;|&nbsp; <b>swing_K:</b> {CONFIG['swing_K']} &nbsp;|&nbsp; "
        f"<b>ATR:</b> Wilder({CONFIG['atr_period']}) &nbsp;|&nbsp; <b>Triggers:</b> "
        f"{sanity['n_triggers']:,} &nbsp;|&nbsp; <b>Built:</b> {datetime.now():%Y-%m-%d %H:%M}</div>"
        "<div class='verdict'>Measure-and-audit only — no threshold, stop, exit or P&L applied. "
        "The 'huge' cutoff, swing K and stop rule are read off Section 1 &amp; 3, then set in Pass 2.</div>"
    )

    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Pass 1 — Entry Geometry Audit</title>
<script>{get_plotlyjs()}</script>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#fafafa;color:#1a1a1a}}
 header{{background:#1d3b2a;color:#fff;padding:22px 32px}} header h1{{margin:0 0 6px;font-size:21px}}
 .meta{{font-size:13px;color:#cfe9d8;margin-top:8px}}
 .verdict{{margin-top:10px;font-size:14px;background:#27503a;padding:10px 14px;border-radius:6px}}
 #toc{{position:sticky;top:0;background:#fff;border-bottom:1px solid #e3e3e3;padding:12px 32px;z-index:9}}
 #toc ul{{margin:0;padding:0;list-style:none;display:flex;flex-wrap:wrap;gap:8px 18px}}
 #toc a{{text-decoration:none;color:#1d3b2a;font-size:13px;font-weight:600}}
 section{{padding:22px 32px;border-bottom:1px solid #ececec}}
 h2{{font-size:18px;color:#1d3b2a;margin:0 0 6px}} h2 .top{{font-size:12px;color:#999;margin-left:8px}}
 .desc{{color:#444;font-size:14px;margin:0 0 10px;max-width:980px}}
 table.rtab{{border-collapse:collapse;font-size:12.5px;background:#fff}}
 table.rtab th,table.rtab td{{border:1px solid #e3e3e3;padding:4px 9px;text-align:right}}
 table.rtab th{{background:#1d3b2a;color:#fff}} table.rtab td:first-child,table.rtab th:first-child{{text-align:left}}
</style></head><body>
<header><h1>Pass 1 — Entry Geometry Measurement &amp; Audit</h1>
 <div>Replay the entry rules, measure candle/swing/stop geometry, and produce an auditable log — no thresholds applied.</div>
 {meta_html}</header>
<nav id='toc'><ul>{toc}</ul></nav>
{body}
<footer style='padding:18px 32px;color:#888;font-size:12px'>
 From pass1_triggers.parquet · regenerate: <code>python strategies/EntryGeometry/report.py</code></footer>
</body></html>"""

    REPORT_PATH.write_text(doc, encoding="utf-8")
    return REPORT_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    path = build_report()
    print(f"[OK] Report: {path}  ({path.stat().st_size/1e6:.1f} MB, self-contained)")
    print(f"     Distributions: {DISTRIB_PATH.name}  |  Audit sample: {AUDIT_SAMPLE_PATH.name}")
    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
