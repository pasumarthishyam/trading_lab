"""
RS build summary (Deliverable 4)
================================

Membership counts per index, multi-sector counts, Nifty-50-only count,
nifty50_member counts, RS row counts, and any unavailable indices.
Writes data/rs/summary.json and prints a readable summary.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd

from infrastructure.rs.index_registry import INDEX_REGISTRY, human_name, SECTOR_KEYS

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "rs"


def summarize() -> dict:
    smap = json.loads((OUT / "sector_map.json").read_text(encoding="utf-8"))
    fetch = json.loads((OUT / "fetch_log.json").read_text(encoding="utf-8"))
    mem = pd.DataFrame(smap["memberships"])

    per_index = Counter(mem["benchmark_index"])
    by_type = Counter(mem["membership_type"])
    sector_mem = mem[mem["membership_type"] == "sector"]
    sectors_per_symbol = sector_mem.groupby("symbol").size()
    multi = int((sectors_per_symbol >= 2).sum())
    no_sector = sorted(smap.get("stocks_with_no_sector", []))
    n50 = Counter(smap["nifty50_member"].values())

    # RS row counts per (timeframe, benchmark) — read straight from the parquet
    con = duckdb.connect()
    rs_glob = (OUT / "rs_values").as_posix() + "/**/*.parquet"
    rs_counts = con.execute(
        f"SELECT timeframe, benchmark_index, COUNT(*) n FROM read_parquet('{rs_glob}', "
        f"hive_partitioning=true) GROUP BY timeframe, benchmark_index"
    ).fetchdf()
    total_rs = int(rs_counts["n"].sum())
    rs_by_tf = {tf: int(g["n"].sum()) for tf, g in rs_counts.groupby("timeframe")}

    # index coverage read directly from the stored parquet (robust to fetch_log)
    from infrastructure.rs.index_registry import data_folder as _folder
    idx_root = ROOT / "data" / "raw" / "indices"
    coverage = {}
    for k in INDEX_REGISTRY:
        cov = {}
        for tf in ("5min", "1min"):
            p = idx_root / _folder(k) / f"{tf}.parquet"
            if p.exists():
                df = con.execute(
                    f"SELECT MIN(CAST(date AS TIMESTAMP)::DATE) a, "
                    f"MAX(CAST(date AS TIMESTAMP)::DATE) b, COUNT(*) n "
                    f"FROM read_parquet('{p.as_posix()}')").fetchone()
                cov[tf] = f"{df[0]} -> {df[1]} ({df[2]:,} rows)"
        coverage[k] = cov

    have_data = [k for k in INDEX_REGISTRY if coverage.get(k)]
    no_data = [k for k in INDEX_REGISTRY if not coverage.get(k)]
    summary = {
        "assumption": smap["assumption"],
        "indices_with_data": len(have_data),
        "indices_no_data": no_data,
        "index_coverage": coverage,
        "membership_rows_per_index": {k: per_index[k] for k in sorted(per_index)},
        "membership_rows_by_type": dict(by_type),
        "stocks_in_2plus_sector_indices": multi,
        "stocks_with_no_sector": {"count": len(no_sector), "symbols": no_sector},
        "nifty50_member_counts": {"member": n50.get(True, 0), "non_member": n50.get(False, 0)},
        "rs_rows_total": total_rs,
        "rs_rows_by_timeframe": rs_by_tf,
        "sector_indices_with_no_fo_members": [
            k for k in SECTOR_KEYS if per_index.get(k, 0) == 0],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _print(s: dict) -> None:
    print("=" * 66)
    print("  SECTOR RELATIVE STRENGTH — BUILD SUMMARY")
    print("=" * 66)
    print(f"  Indices with data: {s['indices_with_data']}  |  no data (skipped): "
          f"{s['indices_no_data'] or 'none'}")
    tfline = ", ".join(f"{tf}={n:,}" for tf, n in sorted(s["rs_rows_by_timeframe"].items()))
    print(f"  RS rows total:   {s['rs_rows_total']:,}  (point-in-time)  [{tfline}]")
    print(f"\n  Membership rows per index (sector + Nifty50 market benchmark):")
    for k, n in s["membership_rows_per_index"].items():
        print(f"      {human_name(k):<26} {k:<16} {n:>4}")
    print(f"\n  Membership rows by type:           {s['membership_rows_by_type']}")
    print(f"  Stocks in >=2 sector indices:      {s['stocks_in_2plus_sector_indices']}")
    print(f"  Stocks with NO sector (broad only): {s['stocks_with_no_sector']['count']}"
          f"  (only the Nifty 50 broad benchmark)")
    print(f"  Nifty 50 members / non-members:    "
          f"{s['nifty50_member_counts']['member']} / {s['nifty50_member_counts']['non_member']}")
    if s["sector_indices_with_no_fo_members"]:
        print(f"  Sector indices with no F&O members: "
              f"{s['sector_indices_with_no_fo_members']}")
    print(f"\n  ASSUMPTION: {s['assumption']}")
    print("=" * 66)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    _print(summarize())
