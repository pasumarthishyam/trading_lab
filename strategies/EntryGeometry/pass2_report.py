"""
Pass 2 — Comparison Report (Filter A vs B)
==========================================

Self-contained interactive HTML presenting the two momentum filters side by
side.  Presents both cleanly — does NOT pick a winner.

    python strategies/EntryGeometry/pass2_report.py
    python strategies/EntryGeometry/pass2_report.py --open
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

from strategies.EntryGeometry.config import CONFIG_P2, p2_paths
from strategies.EntryGeometry import pass2_metrics

_TEMPLATE = "plotly_white"
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def _color(fs, f):
    return _PALETTE[fs.index(f) % len(_PALETTE)]


def _div(fig):
    fig.update_layout(template=_TEMPLATE, margin=dict(l=60, r=40, t=50, b=50),
                      height=400, legend=dict(orientation="h", y=-0.2))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False})


def _load_logs(paths):
    """Auto-detect the filter_* subfolders present and load each trade log."""
    logs = {}
    for sub in sorted(paths.dir.glob("filter_*")):
        p = sub / "trades.parquet"
        if p.exists():
            f = sub.name.split("_", 1)[1]
            logs[f] = pd.read_parquet(p).sort_values(
                ["date", "trigger_time"]).reset_index(drop=True)
    return logs


# ── metric-table formatting ─────────────────────────────────────────
_ROWS = [
    ("Trades", "n_trades", "int"), ("No-trade days", "no_trade_days", "int"),
    ("Expectancy (R/trade)", "expectancy_R", "R"), ("Total R", "total_R", "num"),
    ("Win rate", "win_rate", "pct"),
    ("Avg winner (R)", "avg_win_R", "num"), ("Avg loser (R)", "avg_loss_R", "num"),
    ("% target", "pct_target", "pct"), ("% stop", "pct_stop", "pct"),
    ("% time-closed", "pct_time", "pct"), ("Time-close mean R", "time_close_mean_R", "num"),
    ("Max drawdown (R)", "max_drawdown_R", "num"),
    ("Max DD duration (trades)", "max_dd_duration_trades", "int"),
    ("Longest losing streak", "longest_losing_streak", "int"),
    ("Avg stop dist", "avg_stop_dist_pct", "pct"),
    ("Long / short", "ls", "ls"), ("Stop-first ties", "n_tie_stopfirst", "int"),
]


def _fmt(m, key, kind):
    if kind == "ls":
        return f"{m.get('n_long',0)} / {m.get('n_short',0)}"
    v = m.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if kind == "pct":
        return f"{v*100:.1f}%"
    if kind == "R":
        return f"{v:+.3f} R"
    if kind == "int":
        return f"{int(v)}"
    return f"{v:.2f}"


def _metrics_table(metrics, fs):
    head = "<tr><th>Metric</th>" + "".join(f"<th>Filter {f}</th>" for f in fs) + "</tr>"
    body = ""
    for label, key, kind in _ROWS:
        cells = "".join(f"<td>{_fmt(metrics[f], key, kind)}</td>" for f in fs)
        hl = " class='hl'" if key == "expectancy_R" else ""
        body += f"<tr{hl}><td>{label}</td>{cells}</tr>"
    return f"<table class='rtab'>{head}{body}</table>"


def build_report(test_name: str = "pass2_trade_backtest") -> Path:
    paths = p2_paths(test_name)
    logs = _load_logs(paths)
    meta = json.loads(paths.run_meta.read_text(encoding="utf-8"))
    n_days = meta["n_basket_days"]
    fs = list(logs.keys())
    metrics = {f: pass2_metrics.compute(logs[f], n_days) for f in logs}
    multi = len(fs) > 1

    sections = []

    # 1. verdict table
    vhdr = "Filter " + " vs ".join(fs) if multi else f"Filter {fs[0]}"
    sections.append(("verdict", f"1 · Verdict — {vhdr} (expectancy is the read)",
        f"<p class='desc'>{'Both filters, side by side. ' if multi else ''}"
        "<b>Expectancy per trade (R)</b> is the verdict — win rate looks low by design "
        "(3R target / 1R stop)." + (" Presented for comparison; no winner is chosen here."
        if multi else "") + "</p>" + _metrics_table(metrics, fs)))

    # 2. equity curves
    fig = go.Figure()
    for f, df in logs.items():
        eq = df["realized_R"].cumsum()
        fig.add_scatter(x=np.arange(1, len(df) + 1), y=eq, mode="lines",
                        name=f"Filter {f}", line=dict(color=_color(fs, f), width=2))
    fig.add_hline(y=0, line=dict(color="grey", dash="dot"))
    fig.update_layout(title="Cumulative R (equity curve, gross)",
                      xaxis_title="trade #", yaxis_title="cumulative R")
    sections.append(("equity", "2 · Equity curves (cumulative R)",
        "<p class='desc'>Gross, spot, 1 trade/day. Chronological.</p>" + _div(fig)))

    # 3. drawdown
    figd = go.Figure()
    for f, df in logs.items():
        eq = df["realized_R"].cumsum().to_numpy()
        dd = np.maximum.accumulate(eq) - eq
        figd.add_scatter(x=np.arange(1, len(df) + 1), y=-dd, mode="lines",
                         name=f"Filter {f}", line=dict(color=_color(fs, f)))
    figd.update_layout(title="Drawdown (R, from running peak)",
                       xaxis_title="trade #", yaxis_title="drawdown (R)")
    sections.append(("dd", "3 · Drawdown", "<p class='desc'>Depth of pain in R units — "
        "size positions so the worst run here is survivable.</p>" + _div(figd)))

    # 4. outcome split
    figo = go.Figure()
    for reason in ["target", "stop", "time"]:
        figo.add_bar(x=[f"Filter {f}" for f in logs],
                     y=[metrics[f][f"pct_{reason}"] * 100 for f in logs], name=reason)
    figo.update_layout(barmode="stack", title="Outcome split (% of trades)",
                       yaxis_title="% of trades")
    sections.append(("outcome", "4 · Outcome split", "<p class='desc'>Target (+3R) / stop "
        "(−1R) / 3:00 time-close (partial).</p>" + _div(figo)))

    # 5. R distribution
    figr = go.Figure()
    for f, df in logs.items():
        figr.add_histogram(x=df["realized_R"], name=f"Filter {f}",
                           marker_color=_color(fs, f), opacity=0.6, xbins=dict(size=0.5))
    figr.update_layout(barmode="overlay", title="Realised R per trade",
                       xaxis_title="realised R", yaxis_title="trades")
    sections.append(("rdist", "5 · Realised-R distribution",
        "<p class='desc'>The −1R stop wall, the +3R target spike, and the time-close spread "
        "between.</p>" + _div(figr)))

    # 6. audit log preview
    prev = []
    for f, df in logs.items():
        d = df.head(8).copy()
        d["filter"] = f       # already present from select_trades; ensure set
        prev.append(d)
    cols = ["filter", "date", "symbol", "direction", "r_rank", "trigger_time",
            "entry_price", "stop_price", "stop_dist_pct", "target_price",
            "atr_ratio", "exit_time", "exit_reason", "realized_R"]
    pv = pd.concat(prev)[cols].copy()
    pv["date"] = pv["date"].astype(str).str[:10]
    for c in ["entry_price", "stop_price", "target_price"]:
        pv[c] = pv[c].round(2)
    pv["stop_dist_pct"] = (pv["stop_dist_pct"] * 100).round(2).astype(str) + "%"
    pv["atr_ratio"] = pv["atr_ratio"].round(2); pv["realized_R"] = pv["realized_R"].round(2)
    sections.append(("audit", "6 · Trade-log preview (open on TradingView)",
        "<p class='desc'>First rows of each filter's log. Full logs: "
        "<code>filter_A/trades.csv</code>, <code>filter_B/trades.csv</code> "
        "(+ <code>.parquet</code>). Each row is stamped to find the bar on a chart.</p>"
        + pv.to_html(index=False, classes="rtab", border=0)))

    # 7. honesty notes
    sections.append(("honesty", "7 · Honesty notes (read before concluding)",
        "<ul class='desc'>"
        "<li><b>Fixed 3R is a proxy</b>, not the real exit (15-min level). Results will shift "
        "when level-based exits replace it.</li>"
        "<li><b>Entry at candle close is slightly optimistic</b> — live you act at the next "
        "candle's open, so live will be marginally worse.</li>"
        "<li><b>Spot &amp; gross.</b> No IV/theta/spread, no costs. Validates the underlying-move "
        "edge only; the option-premium and cost layers come after, and only if an edge shows.</li>"
        "<li><b>One regime.</b> A single rising window — necessary, not sufficient. Walk-forward "
        "across regimes is a later step.</li>"
        "<li><b>Read expectancy, not win rate.</b> With 3R:1R the win rate is low by design.</li>"
        "<li><b>Same-candle ties → stop-first</b> (pessimistic). Occurred "
        + " / ".join(f"{metrics[f]['n_tie_stopfirst']}×" for f in fs)
        + " — negligible on 5-min.</li>"
        "</ul>"))

    toc = "".join(f"<li><a href='#{a}'>{t}</a></li>" for a, t, _ in sections)
    body = "".join(f"<section id='{a}'><h2>{t} <a class='top' href='#toc'>↑</a></h2>{h}</section>"
                   for a, t, h in sections)
    exp = " vs ".join(f"{f}: {metrics[f]['expectancy_R']:+.3f}R" for f in logs)
    meta_html = (
        f"<div class='meta'><b>Window:</b> {meta['test_start']} → {meta['test_end']} "
        f"({n_days} basket days) &nbsp;|&nbsp; <b>Fixed:</b> 2× huge, 1% stop cap, 3R target, "
        f"3:00 force-close, 1 trade/day, spot, gross &nbsp;|&nbsp; <b>Built:</b> "
        f"{datetime.now():%Y-%m-%d %H:%M}</div>"
        f"<div class='verdict'>Expectancy — {exp} &nbsp;·&nbsp; both presented for comparison; "
        "no winner chosen.</div>"
    )
    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Pass 2 — Trade Backtest (A vs B)</title>
<script>{get_plotlyjs()}</script>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#fafafa;color:#1a1a1a}}
 header{{background:#3a2c10;color:#fff;padding:22px 32px}} header h1{{margin:0 0 6px;font-size:21px}}
 .meta{{font-size:13px;color:#ecd9b0;margin-top:8px}}
 .verdict{{margin-top:10px;font-size:15px;background:#53401b;padding:10px 14px;border-radius:6px}}
 #toc{{position:sticky;top:0;background:#fff;border-bottom:1px solid #e3e3e3;padding:12px 32px;z-index:9}}
 #toc ul{{margin:0;padding:0;list-style:none;display:flex;flex-wrap:wrap;gap:8px 18px}}
 #toc a{{text-decoration:none;color:#3a2c10;font-size:13px;font-weight:600}}
 section{{padding:22px 32px;border-bottom:1px solid #ececec}}
 h2{{font-size:18px;color:#3a2c10;margin:0 0 6px}} h2 .top{{font-size:12px;color:#999;margin-left:8px}}
 .desc{{color:#444;font-size:14px;margin:0 0 10px;max-width:980px}}
 ul.desc{{padding-left:20px}} ul.desc li{{margin:4px 0}}
 table.rtab{{border-collapse:collapse;font-size:13px;background:#fff}}
 table.rtab th,table.rtab td{{border:1px solid #e3e3e3;padding:5px 10px;text-align:right}}
 table.rtab th{{background:#3a2c10;color:#fff}} table.rtab td:first-child,table.rtab th:first-child{{text-align:left}}
 table.rtab tr.hl td{{background:#fff5e0;font-weight:700}}
</style></head><body>
<header><h1>Pass 2 — Full Trade Backtest (spot, gross)</h1>
 <div>3R-before-1R with a 3:00 force-close, on the Pass 1 triggers. Two momentum filters, compared.</div>
 {meta_html}</header>
<nav id='toc'><ul>{toc}</ul></nav>
{body}
<footer style='padding:18px 32px;color:#888;font-size:12px'>
 From filter_*/trades.parquet · regenerate: <code>python strategies/EntryGeometry/run_pass2.py</code> then <code>pass2_report.py</code></footer>
</body></html>"""
    paths.dir.mkdir(parents=True, exist_ok=True)
    paths.report.write_text(doc, encoding="utf-8")
    return paths.report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-name", default="pass2_trade_backtest")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    path = build_report(args.test_name)
    print(f"[OK] Report: {path}  ({path.stat().st_size/1e6:.1f} MB, self-contained)")
    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
