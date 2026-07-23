"""
Build the multi-membership sector map (sector_map.json + .csv)
=============================================================

Intersects the (draft) constituent lists with the F&O universe and emits one
row per (symbol, benchmark_index) membership, plus a per-symbol nifty50_member
flag.  Every stock gets a Nifty 50 benchmark row (market benchmark); genuine
Nifty 50 constituents also carry nifty50_member = true.

Human-readable, source-cited per membership — this is the file the user
verifies against NSE.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from infrastructure.rs.index_registry import INDEX_REGISTRY, human_name, MARKET_BENCHMARK
from infrastructure.rs.membership_source import (
    NIFTY50, SECTOR_CONSTITUENTS, BROAD_CONSTITUENTS, SOURCE_NIFTY50, source_for,
    NO_INDEX_DATA_KEYS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST = REPO_ROOT / "data" / "raw" / "stocks" / "_fo_universe_manifest.json"
OUT_DIR = REPO_ROOT / "data" / "rs"
MAP_JSON = OUT_DIR / "sector_map.json"
MAP_CSV = OUT_DIR / "sector_map.csv"

ASSUMPTION = ("Membership changes slowly; the CURRENT mapping is applied across "
              "ALL history (2020->latest) — acceptable low-risk approximation. "
              "Nifty 50 is the BROAD market benchmark applied to EVERY stock; "
              "Midcap Select & Sensex are sector-type scopes (membership-based), "
              "alongside the sector indices. Most lists are official NSE CSVs; "
              "Healthcare/Oil&Gas/Consumer-Durables/Media are still DRAFT. Nifty "
              "Cement membership is recorded but its index is not a Kite instrument, "
              "so RS vs Cement is not computed.")


def load_universe() -> list[str]:
    return sorted(json.loads(MANIFEST.read_text(encoding="utf-8"))["symbols"])


def build() -> dict:
    universe = load_universe()
    uset = set(universe)

    # sanity: warn if a source list names a symbol outside the F&O universe
    # (harmless, just not carried downstream) — collected for the summary.
    unknown = {}
    for key, members in SECTOR_CONSTITUENTS.items():
        extra = [s for s in members if s not in uset]
        if extra:
            unknown[key] = extra

    def _row(sym, key, mtype, source):
        return {
            "symbol": sym, "benchmark_index": key, "benchmark_name": human_name(key),
            "membership_type": mtype, "source": source,
            "index_data_available": key not in NO_INDEX_DATA_KEYS,
            "override": False, "override_reason": "",
        }

    # Sector-type scopes = the sector indices PLUS Midcap Select & Sensex
    # (treated as sector scopes, per user + the platform's "sector scope" view).
    sector_all = {**SECTOR_CONSTITUENTS, **BROAD_CONSTITUENTS}

    rows = []
    nifty50_member = {}
    no_sector = []
    for sym in universe:
        has_sector = False
        for key in sector_all:
            if sym in sector_all[key]:
                rows.append(_row(sym, key, "sector", source_for(key)))
                has_sector = True
        # Nifty 50 = the BROAD market index, applied to EVERY stock.
        nifty50_member[sym] = sym in NIFTY50
        rows.append(_row(sym, MARKET_BENCHMARK, "broad", SOURCE_NIFTY50))
        if not has_sector:
            no_sector.append(sym)

    df = pd.DataFrame(rows).sort_values(["symbol", "membership_type", "benchmark_index"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": str(date.today()),
        "assumption": ASSUMPTION,
        "universe_size": len(universe),
        "nifty50_member": nifty50_member,
        "stocks_with_no_sector": no_sector,
        "memberships": df.to_dict(orient="records"),
        "source_notes": {"nifty50": SOURCE_NIFTY50,
                         "official_and_draft": "see per-row 'source'; index_data_available flags Cement"},
        "unknown_symbols_in_source_lists": unknown,
    }
    MAP_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    df.to_csv(MAP_CSV, index=False)
    return payload


if __name__ == "__main__":
    p = build()
    print(f"[OK] sector_map.json -> {MAP_JSON}  ({len(p['memberships'])} membership rows)")
    print(f"     sector_map.csv  -> {MAP_CSV}")
    if p["unknown_symbols_in_source_lists"]:
        print("     note: source lists named non-F&O symbols:",
              p["unknown_symbols_in_source_lists"])
