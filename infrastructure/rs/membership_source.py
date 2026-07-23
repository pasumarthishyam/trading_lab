"""
Index Membership — Source Lists
===============================

Constituent lists used to build ``sector_map.json``.

* **OFFICIAL** (``OFFICIAL_KEYS``): F&O-intersected constituents taken directly
  from NSE/niftyindices official constituent CSVs the user supplied (2026-07).
  These are authoritative.
* **DRAFT**: the four indices for which no official CSV was supplied
  (Healthcare, Oil & Gas, Consumer Durables, Media) — compiled, still pending
  verification against the official niftyindices.com CSVs.

Only F&O-universe symbols are listed (the builder also intersects with the
manifest as a safety net).  Membership changes slowly; the current mapping is
applied across all history — stated in the build summary.

NIFTYCEMENT is a genuine (new) sector but is **not a Kite index instrument**, so
its price data can't be fetched and RS against it isn't computable yet — the
membership is recorded and flagged.
"""

from __future__ import annotations

SOURCE_OFFICIAL = "NSE/niftyindices official constituent CSV (user-provided, 2026-07)"
SOURCE_NIFTY50 = "NSE official NIFTY 50 constituent CSV (user-provided, 2026-07)"
SOURCE_DRAFT = ("DRAFT: compiled — VERIFY against niftyindices.com official "
                "constituent CSV (no official list supplied yet)")
SOURCE_CEMENT = ("NSE official Nifty Cement constituent CSV (user-provided, 2026-07) "
                 "— index NOT available in Kite; RS pending index data")

# Indices whose lists below come from official NSE CSVs.
OFFICIAL_KEYS = {
    "NIFTYBANK", "NIFTYPVTBANK", "NIFTYPSUBANK", "NIFTYFINSERVICE", "NIFTYIT",
    "NIFTYPHARMA", "NIFTYAUTO", "NIFTYFMCG", "NIFTYMETAL", "NIFTYENERGY",
    "NIFTYREALTY", "NIFTYCEMENT",
}
# Benchmarks with no fetchable Kite index instrument (RS not computable yet).
NO_INDEX_DATA_KEYS = {"NIFTYCEMENT"}

# ── Nifty 50 (market benchmark + membership) — OFFICIAL ──────────────
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA",
    "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH",
    "HDFCBANK", "HDFCLIFE", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDIGO",
    "INFY", "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI",
    "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
    "SBILIFE", "SHRIRAMFIN", "SBIN", "SUNPHARMA", "TCS", "TATACONSUM", "TMPV",
    "TATASTEEL", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

# ── Sector indices — canonical_key -> F&O constituents ──────────────
SECTOR_CONSTITUENTS: dict[str, list[str]] = {
    # ---- OFFICIAL (from user-supplied NSE CSVs) ----
    "NIFTYBANK": [
        "AUBANK", "AXISBANK", "BANKBARODA", "CANBK", "FEDERALBNK", "HDFCBANK",
        "ICICIBANK", "IDFCFIRSTB", "INDUSINDBK", "KOTAKBANK", "PNB", "SBIN",
        "UNIONBANK", "YESBANK",
    ],
    "NIFTYPVTBANK": [
        "AXISBANK", "BANDHANBNK", "FEDERALBNK", "HDFCBANK", "ICICIBANK",
        "IDFCFIRSTB", "INDUSINDBK", "KOTAKBANK", "RBLBANK", "YESBANK",
    ],
    "NIFTYPSUBANK": [
        "BANKBARODA", "BANKINDIA", "CANBK", "INDIANB", "PNB", "SBIN", "UNIONBANK",
    ],
    "NIFTYFINSERVICE": [
        "AXISBANK", "BSE", "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "HDFCBANK",
        "HDFCLIFE", "ICICIBANK", "ICICIGI", "JIOFIN", "KOTAKBANK", "LICHSGFIN",
        "MFSL", "MUTHOOTFIN", "PFC", "RECLTD", "SBICARD", "SBILIFE", "SHRIRAMFIN",
        "SBIN",
    ],
    "NIFTYIT": [
        "COFORGE", "HCLTECH", "INFY", "LTM", "MPHASIS", "OFSS", "PERSISTENT",
        "TCS", "TECHM", "WIPRO",
    ],
    "NIFTYPHARMA": [
        "ALKEM", "AUROPHARMA", "BIOCON", "CIPLA", "DIVISLAB", "DRREDDY",
        "GLENMARK", "LAURUSLABS", "LUPIN", "MANKIND", "SUNPHARMA", "TORNTPHARM",
        "ZYDUSLIFE",
    ],
    "NIFTYAUTO": [
        "ASHOKLEY", "BAJAJ-AUTO", "BHARATFORG", "BOSCHLTD", "EICHERMOT",
        "EXIDEIND", "HEROMOTOCO", "M&M", "MARUTI", "MOTHERSON", "SONACOMS",
        "TVSMOTOR", "TMPV", "TIINDIA", "UNOMINDA",
    ],
    "NIFTYFMCG": [
        "BRITANNIA", "COLPAL", "DABUR", "GODREJCP", "HINDUNILVR", "ITC",
        "MARICO", "NESTLEIND", "PATANJALI", "RADICO", "TATACONSUM", "UNITDSPR",
        "VBL",
    ],
    "NIFTYMETAL": [
        "APLAPOLLO", "ADANIENT", "HINDALCO", "HINDZINC", "JSWSTEEL",
        "JINDALSTEL", "NMDC", "NATIONALUM", "SAIL", "TATASTEEL", "VEDL",
    ],
    "NIFTYENERGY": [  # official Nifty Energy is now a broad energy index
        "ABB", "ADANIENSOL", "ADANIGREEN", "ADANIPOWER", "BHEL", "BPCL",
        "CGPOWER", "COALINDIA", "GAIL", "GVT&D", "HINDPETRO", "POWERINDIA",
        "IOC", "INOXWIND", "JSWENERGY", "NHPC", "NTPC", "ONGC", "OIL",
        "PETRONET", "POWERGRID", "RELIANCE", "SIEMENS", "SUZLON", "TATAPOWER",
    ],
    "NIFTYREALTY": [
        "DLF", "GODREJPROP", "LODHA", "OBEROIRLTY", "PHOENIXLTD", "PRESTIGE",
    ],
    "NIFTYCEMENT": [  # official, but no Kite index -> RS pending
        "ULTRACEMCO", "GRASIM", "AMBUJACEM", "SHREECEM", "DALBHARAT",
    ],

    # ---- DRAFT (no official CSV supplied yet) ----
    "NIFTYHEALTHCARE": [
        "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "APOLLOHOSP", "MAXHEALTH",
        "FORTIS", "TORNTPHARM", "ZYDUSLIFE", "LUPIN", "AUROPHARMA", "ALKEM",
        "MANKIND", "LAURUSLABS", "GLENMARK", "BIOCON",
    ],
    "NIFTYOILGAS": [
        "RELIANCE", "ONGC", "IOC", "BPCL", "GAIL", "HINDPETRO", "PETRONET", "OIL",
    ],
    "NIFTYCONSRDURBL": [
        "TITAN", "HAVELLS", "VOLTAS", "CROMPTON", "DIXON", "AMBER",
        "BLUESTARCO", "KALYANKJIL", "PGEL",
    ],
    "NIFTYMEDIA": [],  # no F&O-universe constituents
}

# ── Broad / size benchmarks (OFFICIAL, user-supplied) ───────────────
# Added on user request as additional benchmarks. Nifty 50 is now applied to
# its 50 members ONLY (not universally); a stock in Midcap Select gets Midcap
# Select as its broad benchmark instead of Nifty 50.
BROAD_CONSTITUENTS: dict[str, list[str]] = {
    "NIFTYMIDSELECT": [  # Nifty Midcap Select (F&O members)
        "AUBANK", "ASHOKLEY", "AUROPHARMA", "BSE", "BHARATFORG", "BHEL", "DIXON",
        "FORTIS", "HEROMOTOCO", "HINDPETRO", "INDIANB", "INDUSTOWER",
        "INDUSINDBK", "NAUKRI", "LICI", "LUPIN", "MARICO", "PAYTM", "POLICYBZR",
        "PERSISTENT", "POLYCAB", "SRF", "SUZLON", "SWIGGY", "YESBANK",
    ],
    "SENSEX": [  # BSE Sensex 30 (F&O members; all are also Nifty 50 members)
        "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJFINANCE", "BAJAJFINSV",
        "BEL", "BHARTIARTL", "ETERNAL", "HCLTECH", "HDFCBANK", "HINDUNILVR",
        "ICICIBANK", "INFY", "INDIGO", "ITC", "KOTAKBANK", "LT", "M&M",
        "MARUTI", "NTPC", "POWERGRID", "RELIANCE", "SBIN", "SUNPHARMA", "TCS",
        "TATASTEEL", "TECHM", "TITAN", "TRENT", "ULTRACEMCO",
    ],
}
OFFICIAL_KEYS |= set(BROAD_CONSTITUENTS)


def source_for(key: str) -> str:
    if key in NO_INDEX_DATA_KEYS:
        return SOURCE_CEMENT
    if key in OFFICIAL_KEYS:
        return SOURCE_OFFICIAL
    return SOURCE_DRAFT
