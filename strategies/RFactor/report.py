"""
R-Factor Backtest — Static HTML Report
======================================

Renders the seven outputs into a single self-contained, interactive HTML
file (``results/rfactor_report.html``) from the saved substrate — no
kernel, no Jupyter, no Colab.  Double-click the file; it opens in any
browser and works offline (plotly.js is inlined once).

All sections are always rendered, with a clickable table of contents at
the top so you can jump to the few you care about.

Usage
-----
    python strategies/RFactor/report.py
    python strategies/RFactor/report.py --open      # also launch in browser
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
from plotly.subplots import make_subplots

from strategies.RFactor.config import (
    CONFIG, PICKS_PATH, UNIVERSE_DAILY_PATH, SUMMARY_PATH, RUN_META_PATH,
    REPORT_PATH,
)
from strategies.RFactor import analytics

_TEMPLATE = "plotly_white"


# ── small helpers ───────────────────────────────────────────────────

def _fig_div(fig: go.Figure) -> str:
    """Figure -> HTML div (plotly.js injected once globally, not per fig)."""
    fig.update_layout(template=_TEMPLATE, margin=dict(l=60, r=40, t=50, b=80),
                      height=430, legend=dict(orientation="h", y=-0.25))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False})


def _table_html(df: pd.DataFrame, pct_cols=(), round_cols=(), int_cols=()) -> str:
    d = df.copy()
    for c in pct_cols:
        if c in d:
            d[c] = (d[c] * 100).round(1).astype(str) + "%"
    for c in round_cols:
        if c in d:
            d[c] = d[c].round(2)
    for c in int_cols:
        if c in d:
            d[c] = d[c].round(0).astype("Int64")
    return d.to_html(index=False, classes="rtab", border=0, na_rep="—")


# ── figure builders (one per output) ────────────────────────────────

def _fig_hit_rate(hr, chk):
    fig = go.Figure()
    fig.add_scatter(x=chk, y=hr["topn_hit"]*100, mode="lines+markers",
                    name=f"Top-{CONFIG['top_n']}")
    fig.add_scatter(x=chk, y=hr["top5_hit"]*100, mode="lines+markers",
                    name=f"Top-{CONFIG['top_subset']}")
    fig.add_scatter(x=chk, y=hr["base_rate"]*100, mode="lines",
                    name="Base rate (universe)", line=dict(dash="dash", color="grey"))
    fig.update_layout(title="Hit-rate curve — % of picks that moved ≥2% (from 9:15)",
                      yaxis_title="% hit", xaxis_title="checkpoint (IST)")
    return fig


def _fig_lift(hr, chk):
    fig = go.Figure()
    fig.add_bar(x=chk, y=hr["lift_topn"]*100, name=f"Top-{CONFIG['top_n']} lift")
    fig.add_bar(x=chk, y=hr["lift_top5"]*100, name=f"Top-{CONFIG['top_subset']} lift")
    fig.update_layout(barmode="group", title="Lift over base rate (the verdict)",
                      yaxis_title="lift (percentage points)", xaxis_title="checkpoint (IST)")
    return fig


def _fig_capturable(cc, chk):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=chk, y=cc["median_fav_move"]*100, mode="lines+markers",
                    name="Median favourable move after CP", line=dict(color="green"),
                    secondary_y=False)
    fig.add_scatter(x=chk, y=cc["pct_capturable"]*100, mode="lines+markers",
                    name=f"% capturable (≥{CONFIG['capture_bar']*100:.2f}%)",
                    line=dict(color="purple", dash="dash"), secondary_y=True)
    fig.update_yaxes(title_text="median capturable move (%)", secondary_y=False)
    fig.update_yaxes(title_text="% of picks capturable", secondary_y=True)
    fig.update_layout(title="Capturable slice after checkpoint — chooses entry time",
                      xaxis_title="checkpoint (IST)")
    return fig


def _fig_churn(ch):
    lbl = [f"{a}→{b}" for a, b in zip(ch["checkpoint"], ch["next_checkpoint"])]
    fig = go.Figure()
    fig.add_bar(x=lbl, y=ch["persistence"]*100, name="Persistence to next CP",
                marker_color="steelblue")
    fig.add_scatter(x=lbl, y=ch["churn"]*100, mode="lines+markers", name="Churn",
                    line=dict(color="crimson"))
    fig.update_layout(title="Top-N persistence vs churn between checkpoints",
                      yaxis_title="%", xaxis_title="checkpoint transition")
    return fig


def _fig_per_day(ppd, chk):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"Hits per pick-set (mean={ppd['n_hits'].mean():.1f})",
        "Mean hits by checkpoint"))
    vc = ppd["n_hits"].value_counts().sort_index()
    fig.add_bar(x=vc.index, y=vc.values, marker_color="teal",
                name="freq", showlegend=False, row=1, col=1)
    mbc = ppd.groupby("checkpoint")["n_hits"].mean().reindex(chk)
    fig.add_scatter(x=chk, y=mbc.values, mode="lines+markers", line=dict(color="teal"),
                    name="mean", showlegend=False, row=1, col=2)
    fig.update_xaxes(title_text=f"# of top-{CONFIG['top_n']} that hit", row=1, col=1)
    fig.update_xaxes(title_text="checkpoint", row=1, col=2)
    fig.update_layout(title_text="How many of the picks hit ≥2% on a day")
    return fig


def _fig_direction(ds, chk):
    fig = go.Figure()
    fig.add_bar(x=chk, y=ds["pct_up"]*100, name="up", marker_color="seagreen")
    fig.add_bar(x=chk, y=ds["pct_down"]*100, name="down", marker_color="indianred")
    fig.add_bar(x=chk, y=ds["pct_both"]*100, name="both", marker_color="grey")
    fig.update_layout(barmode="stack", title="Direction split among hitting picks",
                      yaxis_title="% of hits", xaxis_title="checkpoint (IST)")
    return fig


def _fig_magnitude(picks, mg, chk):
    p = picks.copy()
    p["abs_move"] = p[["max_up_pct", "max_down_pct"]].max(axis=1) * 100
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Magnitude of pick moves", "Share of picks reaching each bar"))
    fig.add_histogram(x=np.clip(p["abs_move"], 0, 12), nbinsx=40,
                      marker_color="darkorange", name="picks", showlegend=False,
                      row=1, col=1)
    for col, nm in [("pct_ge_2", "≥2%"), ("pct_ge_3", "≥3%"), ("pct_ge_5", "≥5%")]:
        fig.add_scatter(x=chk, y=mg[col]*100, mode="lines+markers", name=nm, row=1, col=2)
    fig.update_xaxes(title_text="max |move| from 9:15 (%)", row=1, col=1)
    fig.update_xaxes(title_text="checkpoint", row=1, col=2)
    fig.update_layout(title_text="Magnitude — are moves big enough to pay an option buyer?")
    return fig


# ── assembly ────────────────────────────────────────────────────────

def build_report() -> Path:
    picks = pd.read_parquet(PICKS_PATH)
    daily = pd.read_parquet(UNIVERSE_DAILY_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    meta = json.loads(Path(RUN_META_PATH).read_text(encoding="utf-8"))
    chk = CONFIG["checkpoints"]

    hr = analytics.hit_rate_curve(picks, daily)
    cc = analytics.capturable_curve(picks)
    ch = analytics.churn_curve(picks, chk)
    ppd = analytics.picks_per_day_distribution(picks)
    ds = analytics.direction_split(picks)
    mg = analytics.magnitude_distribution(picks)

    # Sections: (anchor, toc-title, html-body)
    sections: list[tuple[str, str, str]] = []

    # Summary table
    summ_tbl = _table_html(
        summary,
        pct_cols=["topn_hit", "top5_hit", "base_rate", "lift_topn", "lift_top5",
                  "median_fav_move", "pct_capturable", "churn"],
        round_cols=["mean_picks_hit"],
    )
    sections.append(("summary", "Summary (verdict table)",
                     f"<p class='desc'>Per-checkpoint top-N hit, base rate, "
                     f"<b>lift</b>, capturable slice and churn.</p>{summ_tbl}"))

    sections.append(("o12", "1 + 2 · Hit-rate & Lift",
                     "<p class='desc'>Top-N / top-5 hit rates vs the eligible-universe "
                     "base rate. The gap (<b>lift</b>) is the verdict.</p>"
                     + _fig_div(_fig_hit_rate(hr, chk)) + _fig_div(_fig_lift(hr, chk))))

    sections.append(("o3", "3 · Capturable after checkpoint",
                     "<p class='desc'>Favourable move still available after the "
                     "checkpoint (from the checkpoint price). <b>This curve picks the "
                     "entry time</b> — it falls as moves exhaust.</p>"
                     + _fig_div(_fig_capturable(cc, chk))))

    sections.append(("o4", "4 · Churn / persistence",
                     "<p class='desc'>How much of the top-N survives to the next "
                     "checkpoint. High early churn means later checkpoints earn their keep.</p>"
                     + _fig_div(_fig_churn(ch))))

    sections.append(("o5", "5 · Per-day distribution",
                     "<p class='desc'>On a typical day, how many of the picks gave a "
                     "≥2% move.</p>" + _fig_div(_fig_per_day(ppd, chk))))

    sections.append(("o6", "6 · Direction split",
                     "<p class='desc'>Up / down / both among hitting picks — confirms "
                     "the signal isn't secretly one-directional.</p>"
                     + _fig_div(_fig_direction(ds, chk))))

    sections.append(("o7", "7 · Magnitude distribution",
                     "<p class='desc'>How far picks travelled — confirms moves clear "
                     "≥3% / ≥5%, not just 2%.</p>"
                     + _fig_div(_fig_magnitude(picks, mg, chk))))

    # ── header / meta ───────────────────────────────────────────────
    verdict = (
        f"Top-{CONFIG['top_n']} lift avg <b>{hr['lift_topn'].mean()*100:.0f} pp</b>, "
        f"top-{CONFIG['top_subset']} lift avg <b>{hr['lift_top5'].mean()*100:.0f} pp</b> "
        f"over a <b>{meta['pooled_base_rate']*100:.0f}%</b> base rate."
    )
    meta_html = (
        f"<div class='meta'><b>Window:</b> {meta['test_start']} → {meta['test_end']} "
        f"({meta['n_test_days']} trading days) &nbsp;|&nbsp; "
        f"<b>Universe:</b> {meta['universe_size']} F&amp;O symbols &nbsp;|&nbsp; "
        f"<b>Dropped:</b> {', '.join(meta['dropped_degenerate_days']) or 'none'} &nbsp;|&nbsp; "
        f"<b>Built:</b> {datetime.now():%Y-%m-%d %H:%M}</div>"
        f"<div class='verdict'>{verdict}</div>"
    )

    toc = "".join(f"<li><a href='#{a}'>{t}</a></li>" for a, t, _ in sections)
    body = "".join(
        f"<section id='{a}'><h2>{t} <a class='top' href='#toc'>↑ top</a></h2>{html}</section>"
        for a, t, html in sections
    )

    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>R-Factor → Move Validation — Report</title>
<script>{get_plotlyjs()}</script>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#fafafa;color:#1a1a1a}}
 header{{background:#10243e;color:#fff;padding:22px 32px}}
 header h1{{margin:0 0 6px;font-size:21px}}
 .meta{{font-size:13px;color:#cfe0f5;margin-top:8px}}
 .verdict{{margin-top:10px;font-size:15px;background:#1b3a63;padding:10px 14px;border-radius:6px}}
 #toc{{position:sticky;top:0;background:#fff;border-bottom:1px solid #e3e3e3;padding:12px 32px;z-index:9}}
 #toc ul{{margin:0;padding:0;list-style:none;display:flex;flex-wrap:wrap;gap:8px 18px}}
 #toc a{{text-decoration:none;color:#10243e;font-size:13px;font-weight:600}}
 #toc a:hover{{text-decoration:underline}}
 section{{padding:22px 32px;border-bottom:1px solid #ececec}}
 h2{{font-size:18px;color:#10243e;margin:0 0 6px}}
 h2 .top{{font-size:11px;font-weight:400;color:#888;margin-left:10px}}
 .desc{{color:#444;font-size:14px;margin:0 0 10px;max-width:900px}}
 table.rtab{{border-collapse:collapse;font-size:13px;background:#fff}}
 table.rtab th,table.rtab td{{border:1px solid #e3e3e3;padding:5px 10px;text-align:right}}
 table.rtab th{{background:#10243e;color:#fff}}
 table.rtab td:first-child,table.rtab th:first-child{{text-align:left}}
</style></head><body>
<header><h1>R-Factor Ranking → Move Validation</h1>
 <div>Does ranking F&amp;O stocks by R-factor at a checkpoint select ≥2% movers — and by how much over random?</div>
 {meta_html}</header>
<nav id='toc'><ul>{toc}</ul></nav>
{body}
<footer style='padding:18px 32px;color:#888;font-size:12px'>
 Generated from rfactor_picks.parquet · regenerate: <code>python strategies/RFactor/report.py</code></footer>
</body></html>"""

    REPORT_PATH.write_text(doc, encoding="utf-8")
    return REPORT_PATH


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the R-Factor HTML report")
    ap.add_argument("--open", action="store_true", help="open in the default browser")
    args = ap.parse_args()
    path = build_report()
    size_mb = path.stat().st_size / 1e6
    print(f"[OK] Report written: {path}  ({size_mb:.1f} MB, self-contained)")
    print("     Double-click it, or re-run with --open.")
    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
