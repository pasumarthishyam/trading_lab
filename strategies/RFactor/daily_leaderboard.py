"""
Daily R-Factor Rank Leaderboard
===============================

For the test window (past ~1 year), records the top-N R-factor-ranked stocks
in the bucket at **every checkpoint, every day**, and renders a browsable HTML
where each symbol links straight to its TradingView chart (set the date
manually once the chart opens).

Checkpoints include 09:30 and 09:45 (09:30 is added to the RFactor defaults).
Output lives in its own folder: results/daily_rank_leaderboard/.

Usage
-----
    python strategies/RFactor/daily_leaderboard.py
    python strategies/RFactor/daily_leaderboard.py --top-n 10 --open
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import duckdb
import pandas as pd

from strategies.RFactor.config import CONFIG, RESULTS_DIR
from strategies.RFactor import engine

# Checkpoints for the leaderboard — RFactor defaults + 09:30 (so both 09:30
# and 09:45 are present), sorted.  15:00 and 15:30 are included for engine
# computation but only shown for EXTRA_CP_DATE in the HTML grid.
BASE_CHECKPOINTS = sorted(set(CONFIG["checkpoints"]) | {"09:30"})
EXTRA_CHECKPOINTS = ["15:00", "15:30"]
EXTRA_CP_DATE = "2026-07-03"  # only this date shows 15:00/15:30 rows
CHECKPOINTS = sorted(set(BASE_CHECKPOINTS) | set(EXTRA_CHECKPOINTS))

OUT_DIR = RESULTS_DIR / "daily_rank_leaderboard"
BASKET_SIZE = 4      # the traded bucket; highlighted in the view


# ── TradingView symbol / URL ────────────────────────────────────────

def tv_symbol(symbol: str) -> str:
    """RFactor symbol -> TradingView NSE ticker (special chars -> underscore)."""
    return "NSE:" + symbol.replace("&", "_").replace("-", "_")


def tv_url(symbol: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol={tv_symbol(symbol)}"


# ── build the leaderboard table ─────────────────────────────────────

def build_leaderboard(test_days_n: int, top_n: int) -> tuple[pd.DataFrame, dict]:
    con = duckdb.connect()
    rf = dict(CONFIG)
    rf["checkpoints"] = CHECKPOINTS
    lookback = rf["rvol_lookback"]

    cal = engine.trading_calendar(con)
    universe = engine.load_universe()
    n_test = min(test_days_n, len(cal) - lookback - 1)
    test_days = set(pd.Timestamp(d) for d in cal[-n_test:])
    cutoff = cal[-(n_test + lookback + 30)]

    agg = engine.aggregate_symbol_days(con, rf, cutoff, universe=universe)
    corp = engine.load_corp_action_dates()
    long_df, daily = engine.compute_features(agg, rf, corp, test_days)

    # drop degenerate days (muhurat / partial) — same rule as RFactor
    elig = daily.groupby("date")["eligible"].sum()
    dropped = sorted(d for d, c in elig.items() if c < 0.5 * len(universe))

    lb = long_df[(long_df["eligible"]) & (long_df["r_rank"] <= top_n)
                 & (~long_df["date"].isin(dropped))].copy()
    lb = lb[["date", "checkpoint", "r_rank", "symbol", "r_factor", "cv", "baseline"]]
    lb["r_rank"] = lb["r_rank"].astype(int)
    lb["in_basket"] = lb["r_rank"] <= BASKET_SIZE
    lb = lb.sort_values(["date", "checkpoint", "r_rank"]).reset_index(drop=True)

    days = sorted(lb["date"].unique())
    meta = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "test_start": str(pd.Timestamp(days[0]).date()),
        "test_end": str(pd.Timestamp(days[-1]).date()),
        "n_days": len(days),
        "checkpoints": CHECKPOINTS,
        "top_n": top_n,
        "basket_size": BASKET_SIZE,
        "universe_size": len(universe),
        "dropped_days": [str(pd.Timestamp(d).date()) for d in dropped],
    }
    return lb, meta


# ── HTML (grouped by day, one grid per day, symbols link to TV) ──────

def _day_grid(day_df: pd.DataFrame, top_n: int, day_checkpoints: list[str] | None = None) -> str:
    """One day's grid: rows = checkpoints, cols = rank 1..N, cells = TV links.

    Parameters
    ----------
    day_checkpoints : list[str] | None
        Checkpoints to render for this day.  Defaults to BASE_CHECKPOINTS.
        Pass the extended list (including 15:00/15:30) for the special date.
    """
    cps = day_checkpoints if day_checkpoints is not None else BASE_CHECKPOINTS
    piv = {(r.checkpoint, r.r_rank): r for r in day_df.itertuples()}
    head = "<th>checkpoint</th>" + "".join(
        f"<th class='{'bk' if k <= BASKET_SIZE else ''}'>#{k}</th>" for k in range(1, top_n + 1))
    rows = ""
    for cp in cps:
        cells = ""
        for k in range(1, top_n + 1):
            r = piv.get((cp, k))
            if r is None:
                cells += "<td>—</td>"
            else:
                cls = "bk" if k <= BASKET_SIZE else ""
                cells += (f"<td class='{cls}'><a href='{tv_url(r.symbol)}' target='_blank' "
                          f"title='R={r.r_factor:.2f}'>{r.symbol}</a></td>")
        rows += f"<tr><td class='cp'>{cp}</td>{cells}</tr>"
    return f"<table class='grid'><tr>{head}</tr>{rows}</table>"


def build_html(lb: pd.DataFrame, meta: dict) -> Path:
    top_n = meta["top_n"]
    days = sorted(lb["date"].unique(), reverse=True)   # newest first
    blocks = []
    for day in days:
        ds = pd.Timestamp(day).strftime("%Y-%m-%d (%a)")
        day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
        # Show 15:00/15:30 checkpoint rows only for the special date.
        if day_str == EXTRA_CP_DATE:
            day_cps = CHECKPOINTS  # includes 15:00/15:30
        else:
            day_cps = BASE_CHECKPOINTS
        grid = _day_grid(lb[lb["date"] == day], top_n, day_checkpoints=day_cps)
        blocks.append(f"<details class='day'><summary>{ds}</summary>{grid}</details>")
    body = "".join(blocks)

    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Daily R-Factor Leaderboard</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#fafafa;color:#1a1a1a}}
 header{{background:#10243e;color:#fff;padding:16px 24px}} header h1{{margin:0;font-size:18px}}
 .meta{{font-size:12.5px;color:#cfe0f5;margin-top:6px}}
 .bar{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:8px 24px;z-index:5}}
 #q{{width:300px;padding:6px 10px;font-size:13px;border:1px solid #ccc;border-radius:5px}}
 .hint{{color:#666;font-size:12.5px;margin-left:10px}}
 .wrap{{padding:10px 24px 50px}}
 details.day{{background:#fff;border:1px solid #e3e3e3;border-radius:6px;margin:8px 0;padding:4px 10px}}
 summary{{font-weight:600;font-size:14px;cursor:pointer;padding:6px 2px;color:#10243e}}
 table.grid{{border-collapse:collapse;font-size:12px;margin:6px 0 12px}}
 table.grid th,table.grid td{{border:1px solid #ececec;padding:3px 7px;text-align:center;white-space:nowrap}}
 table.grid th{{background:#10243e;color:#fff}} th.bk{{background:#1b5e20}}
 td.cp{{background:#f2f5f9;font-weight:600;text-align:right}}
 td.bk{{background:#eef8ee}} td.bk a{{font-weight:700}}
 a{{color:#0b5fa5;text-decoration:none}} a:hover{{text-decoration:underline}}
</style></head><body>
<header><h1>Daily R-Factor Rank Leaderboard — bucket at every checkpoint</h1>
 <div class='meta'>{meta['test_start']} → {meta['test_end']} · {meta['n_days']} trading days ·
 top-{top_n} of {meta['universe_size']} F&amp;O · checkpoints {', '.join(meta['checkpoints'])} ·
 green = traded bucket (top-{meta['basket_size']}) · built {meta['built'][:16].replace('T',' ')}</div></header>
<div class='bar'><input id='q' placeholder='jump to a date e.g. 2026-06-19'
 oninput='filt()'><span class='hint'>click a symbol → opens its TradingView chart (set the date there) ·
 hover a cell for its R value</span>
 <button onclick='toggle(true)'>expand all</button>
 <button onclick='toggle(false)'>collapse all</button></div>
<div class='wrap'>{body}</div>
<script>
 const days=[...document.querySelectorAll('details.day')];
 function filt(){{const q=document.getElementById('q').value.toLowerCase();
   days.forEach(d=>{{d.style.display=d.querySelector('summary').textContent.toLowerCase().includes(q)?'':'none';}});}}
 function toggle(o){{days.forEach(d=>{{if(d.style.display!=='none')d.open=o;}});}}
</script></body></html>"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "leaderboard.html"
    path.write_text(doc, encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description="Daily R-factor rank leaderboard")
    ap.add_argument("--test-days", type=int, default=CONFIG["test_period_days"])
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

    lb, meta = build_leaderboard(args.test_days, args.top_n)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lb.to_parquet(OUT_DIR / "leaderboard.parquet", index=False)
    # readable CSV with the TV link column baked in
    lb_csv = lb.copy()
    lb_csv["tradingview"] = lb_csv["symbol"].map(tv_url)
    lb_csv.to_csv(OUT_DIR / "leaderboard.csv", index=False)
    html = build_html(lb, meta)
    import json
    (OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("=" * 66)
    print("  DAILY R-FACTOR RANK LEADERBOARD")
    print(f"  {meta['test_start']} -> {meta['test_end']}  ({meta['n_days']} days)")
    print(f"  Checkpoints: {', '.join(CHECKPOINTS)}")
    print(f"  Top-{args.top_n} per checkpoint  |  bucket = top-{BASKET_SIZE} (highlighted)")
    print(f"  Rows: {len(lb):,}")
    print(f"  Saved -> {OUT_DIR}")
    print(f"    leaderboard.html  (open this — symbols link to TradingView)")
    print(f"    leaderboard.parquet / .csv (.csv has a tradingview URL column)")
    print("=" * 66)
    if args.open:
        webbrowser.open(html.as_uri())


if __name__ == "__main__":
    main()
