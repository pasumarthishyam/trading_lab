"""
RS Lift Test — HTML Report Generator
======================================
Generates the comparison report with:
- RS sanity summary
- Method comparison table (all methods, per checkpoint)
- Per-checkpoint lift curves (inline SVG / table)
- Per-method daily trade logs (1 best stock per day per checkpoint, TV linked)
- Audit sample: S1 selects X but B0 does not (and vice versa)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
from datetime import datetime
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "results"

TV_BASE = "https://www.tradingview.com/chart/?symbol=NSE:"


def tv_url(symbol: str) -> str:
    return TV_BASE + symbol.replace("&", "_").replace("-", "_")


def tv_link(symbol: str) -> str:
    return f"<a href='{tv_url(symbol)}' target='_blank'>{symbol}</a>"


METHOD_DESCRIPTIONS = {
    "B0_R_alone": (
        "B0 — R-alone (Baseline)",
        "Selects top-10 stocks by R-factor rank alone. R = cumulative volume at checkpoint "
        "divided by the 20-day baseline. This is the validated baseline — every other method must beat this."
    ),
    "B1_R_filterA": (
        "B1 — R + Filter A (momentum filter baseline)",
        "Adds the momentum filter (Filter A) on top of R-rank: a long trigger is only "
        "accepted if the 9:15 candle low is still the session low at checkpoint time "
        "(i.e., the 9:15 extreme has been held — price is trending with conviction). "
        "This is the current strategy's selection. RS must beat this to add value."
    ),
    "S1_sector_filter": (
        "S1 — R + Sector RS (filter mode)",
        "Among the top-R candidates, keeps only stocks with positive sector RS "
        "(stock is outperforming its official sector index from 9:15). "
        "For multi-sector stocks, uses the maximum RS across all sector memberships. "
        "COVERAGE NOTE: ~52 stocks (25%) have no sector mapping and are excluded from this filter — "
        "see section 5 of sanity check for the unmapped stock list."
    ),
    "S1_sector_rank": (
        "S1 — R + Sector RS (rank mode)",
        "Re-ranks the top-R candidates by sector RS (descending) and takes the top-10. "
        "Stocks without sector mapping are excluded. "
        "COVERAGE NOTE: same ~52 unmapped stocks as filter mode."
    ),
    "S1_sector_abs_rank": (
        "S1 — R + |Sector RS| (abs rank mode)",
        "Re-ranks by the ABSOLUTE value of sector RS — motivated by the U-shaped sanity finding "
        "(both Q1 weakest and Q4 strongest RS show high hit rates). Stocks moving far from their "
        "sector (either direction) rank highest. COVERAGE NOTE: same ~52 unmapped stocks excluded."
    ),
    "S2_market_filter": (
        "S2 — R + Market RS (filter mode)",
        "Among the top-R candidates, keeps only stocks with positive market RS "
        "(stock is outperforming its broad benchmark: Nifty50 for large-caps, "
        "MidcapSelect for midcaps, Sensex for Sensex members). Full 100% coverage."
    ),
    "S2_market_rank": (
        "S2 — R + Market RS (rank mode)",
        "Re-ranks the top-R candidates by market RS (descending). Full 100% coverage."
    ),
    "S2_market_abs_rank": (
        "S2 — R + |Market RS| (abs rank mode)",
        "Re-ranks by the ABSOLUTE value of market RS — motivated by the U-shaped sanity finding "
        "(both strong underperformers and outperformers show high hit rates). Full 100% coverage."
    ),
    "S3_both_filter": (
        "S3 — R + Both RS positive (filter mode)",
        "Requires both sector RS AND market RS to be positive. Most restrictive filter. "
        "Fallback to at-least-one-positive if fewer than 3 stocks survive the both-positive gate."
    ),
    "S3_both_rank": (
        "S3 — R + Both RS (combined rank mode)",
        "Re-ranks by RS_sector + RS_market combined score. Covers all stocks with at least "
        "one RS value; stocks with no sector contribute only their market RS."
    ),
    "S4_filterA_sectorRS": (
        "S4 — R + Filter A + Sector RS (rank mode)",
        "Stacks sector RS on top of the already-validated momentum filter. "
        "Tests whether RS adds lift BEYOND what Filter A already captures. "
        "If this barely beats B1, Filter A dominates and RS is redundant here."
    ),
    "S4_filterA_marketRS": (
        "S4 — R + Filter A + Market RS (rank mode)",
        "Same as above but using market RS (Nifty50) as the RS layer on top of Filter A."
    ),
    "S5_sectorRS_nofilter": (
        "S5 — R + Sector RS (rank mode, NO Filter A)",
        "Sector RS rank without any momentum filter. Stocks selected here differ from S4 "
        "because Filter A is absent — any top-R stock is eligible regardless of whether "
        "the 9:15 extreme was held. Compared with S4 to isolate the filter's contribution."
    ),
    "S5_marketRS_nofilter": (
        "S5 — R + Market RS (rank mode, NO Filter A)",
        "Market RS rank without any momentum filter. Same set of candidates as S2_market_rank "
        "but labelled as S5 for the explicit comparison with S4."
    ),
}

METHOD_ORDER = [
    "B0_R_alone", "B1_R_filterA",
    "S1_sector_filter", "S1_sector_rank", "S1_sector_abs_rank",
    "S2_market_filter", "S2_market_rank", "S2_market_abs_rank",
    "S3_both_filter", "S3_both_rank",
    "S4_filterA_sectorRS", "S4_filterA_marketRS",
    "S5_sectorRS_nofilter", "S5_marketRS_nofilter",
]


def _pct(v, decimals=2) -> str:
    if pd.isna(v):
        return "—"
    return f"{v*100:+.{decimals}f}%"


def _pct_plain(v, decimals=2) -> str:
    if pd.isna(v):
        return "—"
    return f"{v*100:.{decimals}f}%"


def _color_lift(v) -> str:
    """Green for positive lift, red for negative, grey for near-zero."""
    if pd.isna(v):
        return ""
    if v > 0.03:
        return "background:#c8f7c5"
    elif v > 0.01:
        return "background:#e8f8e5"
    elif v < -0.03:
        return "background:#ffc8c8"
    elif v < -0.01:
        return "background:#ffe8e8"
    return "background:#f8f8f8"


def build_summary_table_html(summary: pd.DataFrame, b0_lift_by_cp: dict, b1_lift_by_cp: dict) -> str:
    """Per-method, per-checkpoint aggregate table."""
    # Compute pooled (average across checkpoints) for each method
    pooled = (
        summary.groupby("method")
        .agg(
            n_total_picks=("n_total_picks", "sum"),
            hit_rate=("hit_rate", "mean"),
            base_rate=("base_rate", "first"),
            lift=("lift", "mean"),
            hit_rate_top5=("hit_rate_top5", "mean"),
            lift_top5=("lift_top5", "mean"),
            lift_delta_vs_B0=("lift_delta_vs_B0", "mean"),
            lift_delta_vs_B1=("lift_delta_vs_B1", "mean"),
        )
        .reset_index()
    )
    
    rows_html = ""
    for method in METHOD_ORDER:
        row = pooled[pooled["method"] == method]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        name, desc = METHOD_DESCRIPTIONS.get(method, (method, ""))
        lift_style = _color_lift(r["lift"])
        db0_style  = _color_lift(r["lift_delta_vs_B0"])
        db1_style  = _color_lift(r["lift_delta_vs_B1"])
        is_baseline = method in ("B0_R_alone", "B1_R_filterA")
        row_class = "baseline" if is_baseline else ""
        
        rows_html += f"""
        <tr class='{row_class}'>
            <td class='method-name' title='{desc}'>{name}</td>
            <td>{r['n_total_picks']:,}</td>
            <td>{_pct_plain(r['hit_rate'])}</td>
            <td>{_pct_plain(r['base_rate'])}</td>
            <td style='{lift_style}'>{_pct(r['lift'])}</td>
            <td>{_pct_plain(r['hit_rate_top5'])}</td>
            <td style='{lift_style}'>{_pct(r['lift_top5'])}</td>
            <td style='{db0_style}'>{_pct(r['lift_delta_vs_B0'])}</td>
            <td style='{db1_style}'>{_pct(r['lift_delta_vs_B1'])}</td>
        </tr>"""
    
    return f"""
    <table class='summary-table'>
        <caption>Pooled across all checkpoints and test days (248 days, 9 checkpoints)</caption>
        <thead>
            <tr>
                <th>Method</th>
                <th>N picks</th>
                <th>Hit rate<br><small>(top-10)</small></th>
                <th>Base rate</th>
                <th>Lift<br><small>(top-10)</small></th>
                <th>Hit rate<br><small>(top-5)</small></th>
                <th>Lift<br><small>(top-5)</small></th>
                <th>Delta<br><small>vs B0</small></th>
                <th>Delta<br><small>vs B1</small></th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    <p class='note'><b>Lift = hit rate − base rate. Positive = better than chance. 
    Delta vs B0 = improvement over R-alone. Delta vs B1 = improvement over R + Filter A.</b><br>
    Hover method name for description. Green &gt; +1pp vs B0, red &lt; 0 vs B0.</p>"""


def build_checkpoint_curves_html(summary: pd.DataFrame) -> str:
    """Per-checkpoint table showing lift per method, sorted by checkpoint."""
    cps = sorted(summary["checkpoint"].unique())
    
    # Header
    def _mth(m):
        name, desc = METHOD_DESCRIPTIONS.get(m, (m, ""))
        short = name.split(" — ")[0]
        return f"<th title='{desc}'>{short}</th>"
    
    method_headers = "".join(
        _mth(m) for m in METHOD_ORDER if m in summary["method"].values
    )
    
    rows_html = ""
    for cp in cps:
        cp_data = summary[summary["checkpoint"] == cp].set_index("method")
        cells = ""
        for m in METHOD_ORDER:
            if m not in summary["method"].values:
                continue
            if m not in cp_data.index:
                cells += "<td>—</td>"
                continue
            r = cp_data.loc[m]
            lift = r["lift"]
            style = _color_lift(lift)
            cells += f"<td style='{style}'>{_pct(lift)}</td>"
        rows_html += f"<tr><td class='cp'>{cp}</td>{cells}</tr>"
    
    return f"""
    <table class='checkpoint-table'>
        <caption>Lift (hit rate − base rate) at each checkpoint, per method</caption>
        <thead>
            <tr><th>Checkpoint</th>{method_headers}</tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>"""


def build_daily_log_html(daily_df: pd.DataFrame, method: str) -> str:
    """Daily log for one method: 1 best stock per day per checkpoint, TV-linked."""
    sub = daily_df[daily_df["method"] == method].copy()
    if len(sub) == 0:
        return "<p>No data for this method.</p>"
    
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values(["date", "checkpoint"])
    
    # Group by date
    rows_html = ""
    prev_date = None
    for _, row in sub.iterrows():
        dt = row["date"]
        if dt != prev_date:
            if prev_date is not None:
                rows_html += "</tbody></table>"
            ds = pd.Timestamp(dt).strftime("%Y-%m-%d (%a)")
            rows_html += f"""
            <details class='day'>
                <summary>{ds}</summary>
                <table class='log-table'>
                <thead><tr>
                    <th>Checkpoint</th><th>Symbol (TV link)</th>
                    <th>R-rank</th><th>R-factor</th>
                    <th>RS_sector%</th><th>RS_market%</th>
                    <th>Direction</th><th>Hit 2%?</th>
                    <th>Max Up%</th><th>Max Down%</th>
                </tr></thead>
                <tbody>"""
            prev_date = dt
        
        hit_cls = "hit" if row["hit_2pct"] else "miss"
        rs_s = f"{row['RS_sector']*100:+.2f}%" if pd.notna(row.get("RS_sector")) else "—"
        rs_m = f"{row['RS_market']*100:+.2f}%" if pd.notna(row.get("RS_market")) else "—"
        rows_html += f"""
                <tr class='{hit_cls}'>
                    <td class='cp'>{row['checkpoint']}</td>
                    <td>{tv_link(row['symbol'])}</td>
                    <td>{int(row['r_rank']) if pd.notna(row['r_rank']) else '—'}</td>
                    <td>{row['r_factor']:.2f}</td>
                    <td>{rs_s}</td>
                    <td>{rs_m}</td>
                    <td>{row['direction']}</td>
                    <td class='{hit_cls}'>{"YES" if row['hit_2pct'] else "no"}</td>
                    <td>{row['max_up_pct']*100:.2f}%</td>
                    <td>{row['max_down_pct']*100:.2f}%</td>
                </tr>"""
    
    if prev_date is not None:
        rows_html += "</tbody></table></details>"
    
    return rows_html


def build_audit_sample_html(long_with_rs: pd.DataFrame, top_n: int = 10) -> str:
    """
    Audit: show stock-days selected by best RS method but NOT by R-alone (and vice versa).
    Uses the 10:00 checkpoint as the reference.
    """
    ref_cp = "10:00"
    sub = long_with_rs[long_with_rs["checkpoint"] == ref_cp].copy()
    
    b0_sel = sub[sub["r_rank"] <= top_n].copy()
    
    # S1_sector_rank: sorted by RS_sector descending within top-R
    s1_sel = sub[sub["RS_sector"].notna()].sort_values(
        ["date", "RS_sector"], ascending=[True, False]
    ).groupby("date").head(top_n)
    
    b0_ids = set(zip(b0_sel["date"], b0_sel["symbol"]))
    s1_ids = set(zip(s1_sel["date"], s1_sel["symbol"]))
    
    only_s1 = s1_ids - b0_ids   # RS selected but R-alone did not
    only_b0  = b0_ids - s1_ids   # R-alone selected but RS did not
    
    def format_sample(id_set, source_df: pd.DataFrame, label: str) -> str:
        if not id_set:
            return f"<p>No exclusive {label} picks.</p>"
        sample = list(id_set)[:20]
        rows = ""
        for dt, sym in sorted(sample)[:20]:
            row = source_df[(source_df["date"] == dt) & (source_df["symbol"] == sym)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            hit = "YES" if r["hit_2pct"] else "no"
            hit_cls = "hit" if r["hit_2pct"] else "miss"
            rs_s = f"{r['RS_sector']*100:+.2f}%" if pd.notna(r.get("RS_sector")) else "—"
            rs_m = f"{r['RS_market']*100:+.2f}%" if pd.notna(r.get("RS_market")) else "—"
            rows += f"""<tr class='{hit_cls}'>
                <td>{pd.Timestamp(dt).strftime('%Y-%m-%d')}</td>
                <td>{tv_link(sym)}</td>
                <td>{int(r['r_rank']) if pd.notna(r['r_rank']) else '—'}</td>
                <td>{rs_s}</td><td>{rs_m}</td>
                <td class='{hit_cls}'>{hit}</td>
                <td>{r['max_up_pct']*100:.2f}%</td>
                <td>{r['max_down_pct']*100:.2f}%</td>
            </tr>"""
        return f"""<table class='audit-table'>
            <thead><tr><th>Date</th><th>Symbol</th><th>R-rank</th>
            <th>RS_sector%</th><th>RS_market%</th><th>Hit 2%?</th>
            <th>Max Up%</th><th>Max Down%</th></tr></thead>
            <tbody>{rows}</tbody></table>"""
    
    s1_only_html = format_sample(only_s1, s1_sel, "S1")
    b0_only_html = format_sample(only_b0, b0_sel, "B0")
    
    return f"""
    <p>Reference checkpoint: <b>10:00</b>. 
    N(only S1) = {len(only_s1)}, N(only B0) = {len(only_b0)} stock-days out of 248 test days.</p>
    <h4>Stock-days selected by S1 (Sector RS rank) but NOT by B0 (R-alone)</h4>
    {s1_only_html}
    <h4>Stock-days selected by B0 (R-alone) but NOT by S1 (Sector RS rank)</h4>
    {b0_only_html}
    """


def build_html_report(
    summary: pd.DataFrame,
    curves: pd.DataFrame,
    daily_picks: pd.DataFrame,
    extras: dict,
    sanity_notes: str,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    base_rate = extras["base_rate"]
    b0_by_cp = summary[summary["method"] == "B0_R_alone"].set_index("checkpoint")["lift"].to_dict()
    b1_by_cp = summary[summary["method"] == "B1_R_filterA"].set_index("checkpoint")["lift"].to_dict()
    
    summary_table = build_summary_table_html(summary, b0_by_cp, b1_by_cp)
    cp_curves_table = build_checkpoint_curves_html(summary)
    
    # Per-method daily logs
    daily_logs_html = ""
    for method in METHOD_ORDER:
        if method not in daily_picks["method"].values:
            continue
        name, desc = METHOD_DESCRIPTIONS.get(method, (method, ""))
        log_html = build_daily_log_html(daily_picks, method)
        daily_logs_html += f"""
        <details class='method-log'>
            <summary class='method-summary'>{name}</summary>
            <p class='method-desc'>{desc}</p>
            {log_html}
        </details>"""
    
    # Audit sample
    if "long_with_rs" in extras:
        audit_html = build_audit_sample_html(extras["long_with_rs"])
    else:
        audit_html = "<p>Audit sample not available.</p>"
    
    built_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RS Lift Test — Does Relative Strength Add Value?</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
          margin: 0; background: #f5f7fa; color: #1a1a2e; font-size: 13px; }}
  header {{ background: #10243e; color: #fff; padding: 20px 28px; }}
  header h1 {{ margin: 0 0 6px 0; font-size: 20px; }}
  .meta {{ font-size: 12px; color: #9ec5f5; }}
  .content {{ padding: 16px 28px 60px; max-width: 1400px; }}
  h2 {{ color: #10243e; border-bottom: 2px solid #10243e; padding-bottom: 4px; margin-top: 32px; }}
  h3 {{ color: #1b5e20; margin-top: 20px; }}
  h4 {{ color: #444; margin-top: 16px; }}
  
  /* Summary table */
  .summary-table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; margin: 12px 0; }}
  .summary-table th, .summary-table td {{ border: 1px solid #dde; padding: 5px 9px; text-align: center; }}
  .summary-table th {{ background: #10243e; color: #fff; white-space: nowrap; }}
  .summary-table .method-name {{ text-align: left; font-weight: 600; cursor: help; min-width: 260px; }}
  .summary-table caption {{ font-style: italic; font-size: 11px; color: #666; margin-bottom: 4px; }}
  .summary-table tr.baseline {{ background: #fffde7; }}
  .summary-table tr:hover {{ filter: brightness(0.96); }}
  
  /* Checkpoint curves table */
  .checkpoint-table {{ border-collapse: collapse; font-size: 12px; margin: 12px 0; }}
  .checkpoint-table th, .checkpoint-table td {{ border: 1px solid #dde; padding: 4px 8px; text-align: center; }}
  .checkpoint-table th {{ background: #10243e; color: #fff; font-size: 11px; writing-mode: inherit; }}
  .cp {{ background: #f2f5f9; font-weight: 600; text-align: right; }}
  
  /* Daily log */
  details.day {{ background: #fff; border: 1px solid #e3e3e3; border-radius: 5px; margin: 5px 0; padding: 3px 10px; }}
  details.day > summary {{ font-weight: 600; color: #10243e; cursor: pointer; padding: 5px 0; }}
  .log-table, .audit-table {{ border-collapse: collapse; font-size: 11.5px; margin: 6px 0 10px; width: 100%; }}
  .log-table th, .audit-table th {{ background: #10243e; color: #fff; padding: 3px 7px; text-align: center; }}
  .log-table td, .audit-table td {{ border: 1px solid #ececec; padding: 3px 7px; text-align: center; }}
  tr.hit td {{ background: #eef8ee; }}
  tr.miss td {{ background: #fff8f8; }}
  td.hit {{ color: #1b5e20; font-weight: 700; }}
  td.miss {{ color: #c62828; }}
  
  /* Method log */
  details.method-log {{ background: #fff; border: 1px solid #c5cae9; border-radius: 6px; margin: 10px 0; padding: 8px 14px; }}
  .method-summary {{ font-weight: 700; color: #283593; cursor: pointer; font-size: 14px; }}
  .method-desc {{ color: #555; font-style: italic; margin: 4px 0 8px; font-size: 12px; }}
  
  /* Note */
  .note {{ font-size: 11.5px; color: #555; background: #fff9c4; padding: 8px 12px; border-radius: 4px; border-left: 3px solid #f9a825; margin: 8px 0; }}
  .callout {{ background: #e8f5e9; padding: 10px 14px; border-radius: 5px; border-left: 4px solid #2e7d32; margin: 10px 0; }}
  .warn {{ background: #fff3e0; border-left: 4px solid #e65100; }}
  
  a {{ color: #0b5fa5; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  
  pre {{ background: #f1f3f5; padding: 10px; border-radius: 4px; font-size: 11px; overflow-x: auto; }}
  
  /* Search bar */
  .bar {{ position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd;
          padding: 6px 28px; z-index: 10; display: flex; align-items: center; gap: 12px; }}
  #q {{ width: 280px; padding: 5px 9px; border: 1px solid #ccc; border-radius: 4px; font-size: 12px; }}
  button {{ padding: 4px 10px; font-size: 11px; cursor: pointer; border: 1px solid #ccc; border-radius: 3px; }}
</style>
</head>
<body>
<header>
  <h1>Sector Relative Strength — Lift Test: Does RS Add Value?</h1>
  <div class='meta'>
    Test period: 248 trading days (2025-06-18 to 2026-06-19) &middot;
    Universe: 211 F&amp;O stocks &middot;
    Checkpoints: 09:25 to 11:30 (9 points) &middot;
    Built: {built_at}
  </div>
</header>

<div class='bar'>
  <input id='q' placeholder='Jump to date e.g. 2026-05-12' oninput='filt()'>
  <button onclick='toggle(true)'>Expand all</button>
  <button onclick='toggle(false)'>Collapse all</button>
  <span style='color:#666;font-size:11.5px'>Click symbol → opens TradingView chart (set date manually)</span>
</div>

<div class='content'>

<h2>0. What This Test Measures</h2>
<div class='callout'>
  <b>Question:</b> Does adding Relative Strength (RS) — stock vs its sector index and/or vs Nifty50 — 
  improve the quality of stock selection beyond R-factor rank alone?<br><br>
  <b>Outcome definition:</b> Same as the validated R-factor test — ≥2% move from the 9:15 open, 
  either direction, at any point in the session.<br><br>
  <b>Verdict metric: Lift = selected-stocks hit rate − base rate.</b> 
  Lift is the only number that matters. Raw hit rate without the base rate is uninterpretable.<br><br>
  <b>RS earns its place only if it beats B0 (R-alone) by a consistent, meaningful margin across checkpoints.</b>
  A one-checkpoint blip is noise. RS not beating B0 is a valid, useful result — it means added complexity 
  would be overfitting risk, and RS gets dropped.
</div>

<h2>1. RS Values — Sanity Check Summary</h2>
<pre>{sanity_notes}</pre>

<div class='callout warn'>
  <b>KEY FINDING from sanity check (Section 4 — Relevance signal):</b><br>
  RS quartile breakdown at 10:00 shows a <b>U-shaped pattern</b>: Q1 (most negative RS) and Q4 
  (most positive RS) both have higher hit rates (~50%) than Q2/Q3 (~22%). This means:
  <ul>
    <li>RS signals <b>magnitude of movement</b> (stock is moving far in either direction from its benchmark), 
    not directional alignment.</li>
    <li>The "positive RS = bullish" assumption is <b>incorrect for this outcome metric</b>.</li>
    <li><b>Filter mode (keep only positive RS) is expected to HURT performance</b> — it discards 
    the equally-informative negative-RS group (Q1).</li>
    <li>RS absolute value (|RS|) or extreme-RS rank may be a better predictor. 
    The rank mode results will show whether reordering by RS magnitude helps.</li>
  </ul>
  This is a pre-result insight from the sanity check — the lift table below will confirm it.
</div>

<div class='note'>
  <b>Unmapped stocks (~52):</b> About 52 F&O stocks have no sector mapping (broad-only; 
  examples: 360ONE, ABCAPITAL, ANGELONE, ASTRAL, BAJAJHLDNG, BDL, CAMS, CDSL, COCHINSHIP, CONCOR...). 
  These stocks have no RS_sector value and will be excluded from S1 (sector RS) filter/rank modes. 
  S1 results should be read with this coverage gap in mind. 
  S2 (market RS) and B0/B1 have full 100% coverage.
</div>

<h2>2. Selection Method Comparison — Pooled Results</h2>
{summary_table}

<h2>3. Per-Checkpoint Lift Curves</h2>
<p>Each cell = lift (hit rate − base rate) at that checkpoint for that method. 
Colour: green = beats B0 by &gt;1pp, red = below B0, white = near-zero.</p>
{cp_curves_table}

<h2>4. How to Read the Results</h2>
<div class='callout'>
  <ul>
    <li><b>Lift over base rate is the verdict</b>, not raw hit rate. Base rate = {base_rate*100:.2f}%.</li>
    <li><b>RS earns inclusion only if it beats B0 (and ideally B1) by a meaningful, consistent margin 
    across checkpoints</b> — a one-checkpoint blip is noise.</li>
    <li><b>Sector vs market importance</b> is read from S1 vs S2: 
    if S1 lift &gt;&gt; S2 lift → sector RS is the value-add. 
    If similar → both matter. If neither beats B0 → RS doesn't help, drop it.</li>
    <li><b>A negative or negligible result is a valid, useful outcome</b> — it means RS (or one 
    of its layers) doesn't improve selection and should not be added.</li>
    <li><b>S4 vs B1</b>: does RS add lift BEYOND what Filter A already provides? 
    If S4 barely beats B1, Filter A already captures most of the directional signal.</li>
    <li><b>S5 vs S4</b>: stocks selected differ because Filter A is absent in S5; 
    the difference shows the filter's contribution.</li>
  </ul>
</div>

<h2>5. Per-Method Daily Trade Logs</h2>
<p>Each entry shows the <b>top-ranked stock for each day/checkpoint</b> under that selection method. 
Click any symbol to open its TradingView chart (you'll need to set the date manually in TV). 
Green row = stock hit the ≥2% move that day.</p>
{daily_logs_html}

<h2>6. Audit Sample — What RS Changes vs R-alone (10:00 checkpoint)</h2>
<p>A sample of stock-days where S1 (sector RS rank) selected a different stock than B0 (R-alone), 
at the 10:00 checkpoint. Open the chart to see what RS is actually adding or removing.</p>
{audit_html}

</div>

<script>
  const allDetails = [...document.querySelectorAll('details')];
  const dayDetails = [...document.querySelectorAll('details.day')];
  function filt() {{
    const q = document.getElementById('q').value.toLowerCase();
    dayDetails.forEach(d => {{
      d.style.display = d.querySelector('summary').textContent.toLowerCase().includes(q) ? '' : 'none';
    }});
  }}
  function toggle(o) {{
    allDetails.forEach(d => {{ if (d.style.display !== 'none') d.open = o; }});
  }}
</script>
</body>
</html>"""
    
    out_path = OUT_DIR / "rs_lift_report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    print("This module is imported by run_lift_test.py — do not run directly.")
