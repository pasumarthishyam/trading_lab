"""
Sector-RS Index Registry
========================

Canonical benchmark identifiers used across the RS layer, each mapped to its
Kite index tradingsymbol + instrument token (for fetching) and its data folder
under ``data/raw/indices/`` (for loading).

The three already-present indices (Nifty 50, Nifty Bank, Nifty Financial
Services) keep their existing folder names (NIFTY / BANKNIFTY / FINNIFTY); the
newly-fetched sectoral indices get their own folders.

``benchmark_index`` values in ``sector_map.json`` and ``rs_values`` use the
canonical keys below.
"""

from __future__ import annotations

# canonical_key -> (human_name, kite_tradingsymbol, instrument_token, data_folder)
INDEX_REGISTRY: dict[str, tuple[str, str, int, str]] = {
    "NIFTY50":         ("Nifty 50",                 "NIFTY 50",          256265, "NIFTY"),
    "NIFTYBANK":       ("Nifty Bank",               "NIFTY BANK",        260105, "BANKNIFTY"),
    "NIFTYFINSERVICE": ("Nifty Financial Services", "NIFTY FIN SERVICE", 257801, "FINNIFTY"),
    "NIFTYPVTBANK":    ("Nifty Private Bank",       "NIFTY PVT BANK",    271113, "NIFTYPVTBANK"),
    "NIFTYPSUBANK":    ("Nifty PSU Bank",           "NIFTY PSU BANK",    262921, "NIFTYPSUBANK"),
    "NIFTYIT":         ("Nifty IT",                 "NIFTY IT",          259849, "NIFTYIT"),
    "NIFTYPHARMA":     ("Nifty Pharma",             "NIFTY PHARMA",      262409, "NIFTYPHARMA"),
    "NIFTYHEALTHCARE": ("Nifty Healthcare",         "NIFTY HEALTHCARE",  288521, "NIFTYHEALTHCARE"),
    "NIFTYAUTO":       ("Nifty Auto",               "NIFTY AUTO",        263433, "NIFTYAUTO"),
    "NIFTYFMCG":       ("Nifty FMCG",               "NIFTY FMCG",        261897, "NIFTYFMCG"),
    "NIFTYMETAL":      ("Nifty Metal",              "NIFTY METAL",       263689, "NIFTYMETAL"),
    "NIFTYENERGY":     ("Nifty Energy",             "NIFTY ENERGY",      261641, "NIFTYENERGY"),
    "NIFTYOILGAS":     ("Nifty Oil & Gas",          "NIFTY OIL AND GAS", 289033, "NIFTYOILGAS"),
    "NIFTYREALTY":     ("Nifty Realty",             "NIFTY REALTY",      261129, "NIFTYREALTY"),
    "NIFTYMEDIA":      ("Nifty Media",              "NIFTY MEDIA",       263945, "NIFTYMEDIA"),
    "NIFTYCONSRDURBL": ("Nifty Consumer Durables",  "NIFTY CONSR DURBL", 288777, "NIFTYCONSRDURBL"),
    # Nifty Cement: genuine sector but NOT a Kite index instrument (token 0) —
    # membership is recorded; RS is not computable until index data exists.
    "NIFTYCEMENT":     ("Nifty Cement",             "NIFTY CEMENT",      0,      "NIFTYCEMENT"),
    # Broad / size benchmarks (added on user request — "sometimes those perform well").
    "NIFTYMIDSELECT":  ("Nifty Midcap Select",      "NIFTY MID SELECT",  288009, "NIFTYMIDSELECT"),
    "SENSEX":          ("BSE Sensex",               "SENSEX",            265,    "SENSEX"),
}

# Broad/size benchmarks (not sectors): Nifty 50 (market), Midcap Select, Sensex.
BROAD_KEYS = ["NIFTYMIDSELECT", "SENSEX"]

# Nifty 50 is the universal market benchmark.
MARKET_BENCHMARK = "NIFTY50"

# Sector indices (everything except the market benchmark and broad/size indices).
SECTOR_KEYS = [k for k in INDEX_REGISTRY
               if k != MARKET_BENCHMARK and k not in ("NIFTYMIDSELECT", "SENSEX")]


def human_name(key: str) -> str:
    return INDEX_REGISTRY[key][0]


def data_folder(key: str) -> str:
    return INDEX_REGISTRY[key][3]


def token(key: str) -> int:
    return INDEX_REGISTRY[key][2]


def kite_symbol(key: str) -> str:
    return INDEX_REGISTRY[key][1]
