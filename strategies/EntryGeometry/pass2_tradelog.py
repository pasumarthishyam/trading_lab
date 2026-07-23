"""
Pass 2 — Complete Trade Log (HTML)
==================================

Renders the FULL per-trade log to a single self-contained HTML file
(``trade_log.html``) with a live text filter and sortable, sticky-header
table.  Separate from the charts report, per request.

    python strategies/EntryGeometry/pass2_tradelog.py --test-name pass2.1_filterA_0930
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

import pandas as pd

from strategies.EntryGeometry.config import p2_paths

# Columns shown, in order (the complete trade record).
COLS = ["filter", "date", "symbol", "direction", "r_rank", "pos_vs_open",
        "trigger_time", "level_price", "entry_price", "stop_source",
        "stop_price", "stop_dist_pct", "R_unit", "target_price", "atr_ratio",
        "exit_time", "exit_reason", "realized_R", "tie_stopfirst"]


def _load_all(paths) -> pd.DataFrame:
    frames = []
    for sub in sorted(paths.dir.glob("filter_*")):
        p = sub / "trades.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["filter"] = sub.name.split("_", 1)[1]
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["filter", "date", "trigger_time"]).reset_index(drop=True)


def build_tradelog(test_name: str = "pass2_trade_backtest") -> Path:
    paths = p2_paths(test_name)
    df = _load_all(paths)
    meta = json.loads(paths.run_meta.read_text(encoding="utf-8")) if paths.run_meta.exists() else {}

    disp = df.copy()
    disp["date"] = disp["date"].astype(str).str[:10]
    for c in ["level_price", "entry_price", "stop_price", "target_price"]:
        if c in disp: disp[c] = disp[c].round(2)
    disp["stop_dist_pct"] = (disp["stop_dist_pct"] * 100).round(2)
    for c in ["R_unit", "atr_ratio", "realized_R"]:
        if c in disp: disp[c] = disp[c].round(3)
    disp = disp[[c for c in COLS if c in disp.columns]]

    # build rows; tag win/loss for row colouring
    head = "".join(f"<th onclick='sortT({i})'>{c}</th>" for i, c in enumerate(disp.columns))
    body = []
    ri = list(disp.columns).index("realized_R")
    for _, r in disp.iterrows():
        cls = "win" if r["realized_R"] > 0 else ("loss" if r["realized_R"] < 0 else "")
        tds = "".join(f"<td>{r[c]}</td>" for c in disp.columns)
        body.append(f"<tr class='{cls}'>{tds}</tr>")
    body_html = "".join(body)

    exp = ""
    if meta.get("metrics"):
        exp = " · ".join(f"Filter {f}: expectancy {m['expectancy_R']:+.3f}R "
                         f"({m['n_trades']} trades)" for f, m in meta["metrics"].items())

    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Pass 2 Trade Log — {test_name}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#fafafa;color:#1a1a1a}}
 header{{background:#3a2c10;color:#fff;padding:16px 24px}} header h1{{margin:0;font-size:18px}}
 .meta{{font-size:12.5px;color:#ecd9b0;margin-top:6px}}
 .bar{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:8px 24px;z-index:5}}
 #q{{width:340px;padding:6px 10px;font-size:13px;border:1px solid #ccc;border-radius:5px}}
 .count{{color:#666;font-size:12.5px;margin-left:10px}}
 .wrap{{padding:0 24px 40px}}
 table{{border-collapse:collapse;font-size:12px;width:100%;background:#fff}}
 th,td{{border:1px solid #e6e6e6;padding:4px 8px;text-align:right;white-space:nowrap}}
 th{{background:#3a2c10;color:#fff;position:sticky;top:49px;cursor:pointer;user-select:none}}
 th:hover{{background:#53401b}}
 td:first-child,td:nth-child(2),td:nth-child(3),td:nth-child(4){{text-align:left}}
 tr.win td{{background:#eef8ee}} tr.loss td{{background:#fdeeee}}
 tr:hover td{{background:#fff7e0}}
</style></head><body>
<header><h1>Pass 2 — Complete Trade Log · {test_name}</h1>
 <div class='meta'>{meta.get('test_start','?')} → {meta.get('test_end','?')} · {len(df)} trades · {exp} · built {datetime.now():%Y-%m-%d %H:%M}</div></header>
<div class='bar'><input id='q' placeholder='filter rows (symbol, date, reason, direction ...)'
 oninput='filt()'><span class='count' id='cnt'></span>
 <span class='count'>green=win · red=loss · click a header to sort</span></div>
<div class='wrap'><table id='t'><thead><tr>{head}</tr></thead><tbody>{body_html}</tbody></table></div>
<script>
 const tb=document.querySelector('#t tbody'), rows=[...tb.rows];
 const cnt=document.getElementById('cnt');
 function upd(){{cnt.textContent=[...tb.rows].filter(r=>r.style.display!=='none').length+' / '+rows.length+' rows';}}
 function filt(){{const q=document.getElementById('q').value.toLowerCase();
   rows.forEach(r=>{{r.style.display=r.textContent.toLowerCase().includes(q)?'':'none';}});upd();}}
 let dir=1,last=-1;
 function sortT(i){{dir=(i===last)?-dir:1;last=i;
   const v=r=>{{const t=r.cells[i].textContent.trim();const n=parseFloat(t.replace('%',''));return isNaN(n)?t:n;}};
   const s=[...tb.rows].sort((a,b)=>{{const x=v(a),y=v(b);return (x>y?1:x<y?-1:0)*dir;}});
   s.forEach(r=>tb.appendChild(r));}}
 upd();
</script></body></html>"""

    paths.dir.mkdir(parents=True, exist_ok=True)
    paths.tradelog.write_text(doc, encoding="utf-8")
    return paths.tradelog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-name", default="pass2_trade_backtest")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    path = build_tradelog(args.test_name)
    print(f"[OK] Trade log: {path}  ({path.stat().st_size/1e3:.0f} KB, self-contained)")
    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
