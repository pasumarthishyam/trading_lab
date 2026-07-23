"""
Pass 2.2 — K-sweep Report + Trade Log (rolling basket + breakeven)
==================================================================

Two self-contained HTML files:
  * report.html    — swing-K comparison: metrics table, equity curves, outcome split
  * trade_log.html — every trade across all K, live-filterable / sortable

    python strategies/EntryGeometry/pass2_2_report.py --open
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

from strategies.EntryGeometry.config import p2_paths
from strategies.EntryGeometry import pass2_metrics

_TEMPLATE = "plotly_white"
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def _div(fig):
    fig.update_layout(template=_TEMPLATE, margin=dict(l=60, r=40, t=50, b=50),
                      height=400, legend=dict(orientation="h", y=-0.2))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False})


def _load_k_logs(paths):
    logs = {}
    for sub in sorted(paths.dir.glob("K*")):
        p = sub / "trades.parquet"
        if p.exists():
            k = int(sub.name[1:])
            logs[k] = pd.read_parquet(p).sort_values(
                ["date", "trigger_time"]).reset_index(drop=True)
    return dict(sorted(logs.items()))


_ROWS = [
    ("Trades", "n_trades", "int"), ("No-trade days", "no_trade_days", "int"),
    ("Expectancy (R/trade)", "expectancy_R", "R"), ("Total R", "total_R", "num"),
    ("Win rate", "win_rate", "pct"),
    ("% target", "pct_target", "pct"), ("% breakeven", "pct_breakeven", "pct"),
    ("% stop", "pct_stop", "pct"), ("% time-closed", "pct_time", "pct"),
    ("Avg winner (R)", "avg_win_R", "num"), ("Avg loser (R)", "avg_loss_R", "num"),
    ("Max drawdown (R)", "max_drawdown_R", "num"),
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


def _metrics_table(metrics, ks, best_k):
    head = "<tr><th>Metric</th>" + "".join(
        f"<th class='{'best' if k==best_k else ''}'>K={k}</th>" for k in ks) + "</tr>"
    body = ""
    for label, key, kind in _ROWS:
        cells = "".join(
            f"<td class='{'best' if k==best_k else ''}'>{_fmt(metrics[k], key, kind)}</td>"
            for k in ks)
        hl = " class='hl'" if key == "expectancy_R" else ""
        body += f"<tr{hl}><td>{label}</td>{cells}</tr>"
    return f"<table class='rtab'>{head}{body}</table>"


def build_report(test_name: str = "pass2.2_rolling_be") -> Path:
    paths = p2_paths(test_name)
    logs = _load_k_logs(paths)
    meta = json.loads(paths.run_meta.read_text(encoding="utf-8"))
    n_days = meta["n_days"]
    ks = list(logs.keys())
    metrics = {k: pass2_metrics.compute(logs[k], n_days) for k in ks}
    best_k = max(ks, key=lambda k: metrics[k]["expectancy_R"])
    col = {k: _PALETTE[i % len(_PALETTE)] for i, k in enumerate(ks)}

    sec = []
    sec.append(("verdict", "1 · swing-K comparison (expectancy is the read)",
        "<p class='desc'>Rolling top-4 bucket (re-ranked every 15 min), breakeven→0R "
        "once +2R touched, Filter A, entry 09:30–12:30. Only <b>swing K</b> varies. "
        "Best expectancy column highlighted.</p>" + _metrics_table(metrics, ks, best_k)))

    # equity
    fig = go.Figure()
    for k in ks:
        fig.add_scatter(x=np.arange(1, len(logs[k]) + 1), y=logs[k]["realized_R"].cumsum(),
                        mode="lines", name=f"K={k}", line=dict(color=col[k], width=2))
    fig.add_hline(y=0, line=dict(color="grey", dash="dot"))
    fig.update_layout(title="Cumulative R by swing K", xaxis_title="trade #", yaxis_title="cum R")
    sec.append(("equity", "2 · Equity curves", "<p class='desc'>Gross, spot, 1 trade/day.</p>"
                + _div(fig)))

    # outcome split
    figo = go.Figure()
    for reason in ["target", "breakeven", "stop", "time"]:
        figo.add_bar(x=[f"K={k}" for k in ks],
                     y=[metrics[k][f"pct_{reason}"] * 100 for k in ks], name=reason)
    figo.update_layout(barmode="stack", title="Outcome split (%)", yaxis_title="% of trades")
    sec.append(("outcome", "3 · Outcome split",
        "<p class='desc'>+3R target / 0R breakeven / −1R stop / 3:00 time-close.</p>" + _div(figo)))

    # expectancy vs K bar
    fige = go.Figure()
    fige.add_bar(x=[f"K={k}" for k in ks], y=[metrics[k]["expectancy_R"] for k in ks],
                 marker_color=[col[k] for k in ks])
    fige.update_layout(title="Expectancy (R/trade) by K", yaxis_title="R/trade")
    sec.append(("exp", "4 · Expectancy by K", "<p class='desc'>The headline number per K.</p>"
                + _div(fige)))

    toc = "".join(f"<li><a href='#{a}'>{t}</a></li>" for a, t, _ in sec)
    body = "".join(f"<section id='{a}'><h2>{t} <a class='top' href='#toc'>↑</a></h2>{h}</section>"
                   for a, t, h in sec)
    exp = " · ".join(f"K={k}: {metrics[k]['expectancy_R']:+.3f}R" for k in ks)
    meta_html = (
        f"<div class='meta'><b>Window:</b> {meta['test_start']} → {meta['test_end']} "
        f"({n_days} days) &nbsp;|&nbsp; rolling top-4 · breakeven@2R · Filter A · "
        f"09:30–12:30 · spot, gross &nbsp;|&nbsp; built {datetime.now():%Y-%m-%d %H:%M}</div>"
        f"<div class='verdict'>{exp} &nbsp;→&nbsp; best K = <b>{best_k}</b></div>")

    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Pass 2.2 — Rolling + Breakeven — K sweep</title>
<script>{get_plotlyjs()}</script>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#fafafa;color:#1a1a1a}}
 header{{background:#243a10;color:#fff;padding:22px 32px}} header h1{{margin:0 0 6px;font-size:21px}}
 .meta{{font-size:13px;color:#d9ecb0;margin-top:8px}}
 .verdict{{margin-top:10px;font-size:15px;background:#3a5320;padding:10px 14px;border-radius:6px}}
 #toc{{position:sticky;top:0;background:#fff;border-bottom:1px solid #e3e3e3;padding:12px 32px;z-index:9}}
 #toc ul{{margin:0;padding:0;list-style:none;display:flex;flex-wrap:wrap;gap:8px 18px}}
 #toc a{{text-decoration:none;color:#243a10;font-size:13px;font-weight:600}}
 section{{padding:22px 32px;border-bottom:1px solid #ececec}}
 h2{{font-size:18px;color:#243a10;margin:0 0 6px}} h2 .top{{font-size:12px;color:#999;margin-left:8px}}
 .desc{{color:#444;font-size:14px;margin:0 0 10px;max-width:980px}}
 table.rtab{{border-collapse:collapse;font-size:13px;background:#fff}}
 table.rtab th,table.rtab td{{border:1px solid #e3e3e3;padding:5px 10px;text-align:right}}
 table.rtab th{{background:#243a10;color:#fff}} table.rtab td:first-child,table.rtab th:first-child{{text-align:left}}
 table.rtab tr.hl td{{background:#f2fbe6;font-weight:700}} .best{{background:#dff0c8 !important}}
</style></head><body>
<header><h1>Pass 2.2 — Rolling Basket + Breakeven (swing-K sweep)</h1>
 <div>Does re-ranking the bucket + a breakeven stop help, and which swing K is best?</div>
 {meta_html}</header>
<nav id='toc'><ul>{toc}</ul></nav>
{body}
<footer style='padding:18px 32px;color:#888;font-size:12px'>
 regenerate: <code>python strategies/EntryGeometry/run_pass2_2.py</code></footer>
</body></html>"""
    paths.dir.mkdir(parents=True, exist_ok=True)
    paths.report.write_text(doc, encoding="utf-8")
    return paths.report


# ── complete trade log (all K) ──────────────────────────────────────
_LOGCOLS = ["swing_K", "date", "symbol", "direction", "gov_checkpoint", "rank_at_cp",
            "trigger_time", "entry_price", "stop_source", "stop_price", "stop_dist_pct",
            "R_unit", "target_price", "be_price", "atr_ratio", "exit_time",
            "exit_reason", "realized_R"]


def build_tradelog(test_name: str = "pass2.2_rolling_be") -> Path:
    paths = p2_paths(test_name)
    logs = _load_k_logs(paths)
    if not logs:
        return paths.tradelog
    df = pd.concat(logs.values(), ignore_index=True)
    df["date"] = df["date"].astype(str).str[:10]
    for c in ["level_price", "entry_price", "stop_price", "target_price", "be_price"]:
        if c in df: df[c] = df[c].round(2)
    df["stop_dist_pct"] = (df["stop_dist_pct"] * 100).round(2)
    for c in ["R_unit", "atr_ratio", "realized_R"]:
        if c in df: df[c] = df[c].round(3)
    disp = df[[c for c in _LOGCOLS if c in df.columns]]

    head = "".join(f"<th onclick='sortT({i})'>{c}</th>" for i, c in enumerate(disp.columns))
    ri = list(disp.columns).index("realized_R")
    rows = []
    for _, r in disp.iterrows():
        cls = "win" if r["realized_R"] > 0 else ("loss" if r["realized_R"] < 0 else "be")
        rows.append(f"<tr class='{cls}'>" + "".join(f"<td>{r[c]}</td>" for c in disp.columns) + "</tr>")
    body = "".join(rows)

    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Pass 2.2 Trade Log</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#fafafa}}
 header{{background:#243a10;color:#fff;padding:14px 24px}} header h1{{margin:0;font-size:17px}}
 .bar{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:8px 24px;z-index:5}}
 #q{{width:340px;padding:6px 10px;font-size:13px;border:1px solid #ccc;border-radius:5px}}
 .hint{{color:#666;font-size:12.5px;margin-left:10px}}
 .wrap{{padding:0 24px 40px}}
 table{{border-collapse:collapse;font-size:12px;width:100%;background:#fff}}
 th,td{{border:1px solid #e6e6e6;padding:4px 8px;text-align:right;white-space:nowrap}}
 th{{background:#243a10;color:#fff;position:sticky;top:47px;cursor:pointer}}
 td:nth-child(2),td:nth-child(3),td:nth-child(4),td:nth-child(5){{text-align:left}}
 tr.win td{{background:#eef8ee}} tr.loss td{{background:#fdeeee}} tr.be td{{background:#f4f4f4}}
</style></head><body>
<header><h1>Pass 2.2 — Complete Trade Log (all K) · rolling + breakeven</h1></header>
<div class='bar'><input id='q' placeholder='filter (K, symbol, date, exit_reason ...)' oninput='filt()'>
 <span class='hint' id='cnt'></span><span class='hint'>green=win · grey=breakeven · red=loss · click header to sort</span></div>
<div class='wrap'><table id='t'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
<script>
 const tb=document.querySelector('#t tbody'),rows=[...tb.rows],cnt=document.getElementById('cnt');
 function upd(){{cnt.textContent=[...tb.rows].filter(r=>r.style.display!=='none').length+' / '+rows.length+' rows';}}
 function filt(){{const q=document.getElementById('q').value.toLowerCase();
   rows.forEach(r=>{{r.style.display=r.textContent.toLowerCase().includes(q)?'':'none';}});upd();}}
 let dir=1,last=-1;
 function sortT(i){{dir=(i===last)?-dir:1;last=i;
   const v=r=>{{const t=r.cells[i].textContent.trim();const n=parseFloat(t.replace('%',''));return isNaN(n)?t:n;}};
   [...tb.rows].sort((a,b)=>{{const x=v(a),y=v(b);return(x>y?1:x<y?-1:0)*dir;}}).forEach(r=>tb.appendChild(r));}}
 upd();
</script></body></html>"""
    paths.dir.mkdir(parents=True, exist_ok=True)
    paths.tradelog.write_text(doc, encoding="utf-8")
    return paths.tradelog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-name", default="pass2.2_rolling_be")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    rp = build_report(args.test_name)
    tl = build_tradelog(args.test_name)
    print(f"[OK] Report: {rp}  ({rp.stat().st_size/1e6:.1f} MB)")
    print(f"[OK] Trade log: {tl}  ({tl.stat().st_size/1e3:.0f} KB)")
    if args.open:
        webbrowser.open(rp.as_uri())


if __name__ == "__main__":
    main()
