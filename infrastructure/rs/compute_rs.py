"""
Compute the RS tidy table (rs_values)
=====================================

For each F&O stock, at each 5-min candle, for EACH of its benchmarks (all
sector memberships + Nifty 50):

    RS = (stock%move from its 9:15 open) - (benchmark%move from its 9:15 open)

Point-in-time: uses only each series' own 9:15 open and its value at that
candle.  Stock and benchmark are aligned by exact timestamp; a candle is
emitted only where BOTH series have that minute — a missing benchmark minute
yields no row (i.e. null / not fabricated, never forward-filled).

Output: tidy long table partitioned by benchmark, at
``data/rs/rs_values/benchmark_index=<KEY>/*.parquet`` with columns
``symbol, timestamp, benchmark_index, membership_type, rs_value``.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import duckdb
import pandas as pd

from infrastructure.rs.index_registry import INDEX_REGISTRY, data_folder

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STOCKS_ROOT = REPO_ROOT / "data" / "raw" / "stocks"
INDICES_ROOT = REPO_ROOT / "data" / "raw" / "indices"
OUT_DIR = REPO_ROOT / "data" / "rs"
RS_DIR = OUT_DIR / "rs_values"
MAP_CSV = OUT_DIR / "sector_map.csv"

SESS_OPEN, SESS_END = "09:15", "15:30"


def _move_cte(name: str, source_sql: str, label_col: str) -> str:
    """Build a %-move-from-9:15-open CTE for a set of parquet files.

    ``label_col`` is a SQL expression giving the series id (symbol / benchmark).
    """
    return f"""
    {name}_min AS (
        SELECT {label_col} AS series, date AS ts,
               CAST(date AS TIMESTAMP)::DATE AS d,
               CAST(date AS TIMESTAMP)::TIME AS t,
               open, close
        FROM read_parquet({source_sql}, filename=true)
        WHERE CAST(date AS TIMESTAMP)::TIME BETWEEN TIME '{SESS_OPEN}' AND TIME '{SESS_END}'
    ),
    {name}_open AS (
        SELECT series, d, arg_min(open, t) AS o0915
        FROM {name}_min GROUP BY series, d
    ),
    {name}_move AS (
        SELECT m.series, m.ts, (m.close - o.o0915) / o.o0915 AS move
        FROM {name}_min m JOIN {name}_open o USING (series, d)
        WHERE o.o0915 IS NOT NULL AND o.o0915 <> 0
    )"""


def compute(timeframe: str = "5min", universe: list[str] | None = None) -> dict:
    con = duckdb.connect()

    membership = pd.read_csv(MAP_CSV)[["symbol", "benchmark_index", "membership_type"]]
    if universe is not None:
        membership = membership[membership["symbol"].isin(set(universe))]
    symbols = sorted(membership["symbol"].unique())

    # explicit POSIX file lists (avoids Windows glob/regex issues)
    stock_paths = [(STOCKS_ROOT / s / f"{timeframe}.parquet").as_posix()
                   for s in symbols if (STOCKS_ROOT / s / f"{timeframe}.parquet").exists()]
    stock_list = "[" + ", ".join(f"'{p}'" for p in stock_paths) + "]"

    # benchmark move: UNION ALL of each available index with its canonical label
    bench_used, bench_missing = [], []
    bench_selects = []
    for key in INDEX_REGISTRY:
        if key not in set(membership["benchmark_index"]):
            continue  # benchmark nobody maps to (e.g. NIFTYMEDIA with no F&O members)
        p = INDICES_ROOT / data_folder(key) / f"{timeframe}.parquet"
        if not p.exists():
            bench_missing.append(key)
            continue
        bench_used.append(key)
        bench_selects.append(f"""
        SELECT '{key}' AS series, date AS ts, CAST(date AS TIMESTAMP)::DATE AS d,
               CAST(date AS TIMESTAMP)::TIME AS t, open, close
        FROM read_parquet('{p.as_posix()}')
        WHERE CAST(date AS TIMESTAMP)::TIME BETWEEN TIME '{SESS_OPEN}' AND TIME '{SESS_END}'""")
    bench_union = " UNION ALL ".join(bench_selects)

    con.register("membership", membership)

    sql = f"""
    WITH {_move_cte('stk', stock_list, f"regexp_extract(filename, 'stocks/([^/]+)/{timeframe}', 1)")},
    bench_min AS ( {bench_union} ),
    bench_open AS (
        SELECT series, d, arg_min(open, t) AS o0915 FROM bench_min GROUP BY series, d
    ),
    bench_move AS (
        SELECT b.series, b.ts, (b.close - o.o0915) / o.o0915 AS move
        FROM bench_min b JOIN bench_open o USING (series, d)
        WHERE o.o0915 IS NOT NULL AND o.o0915 <> 0
    )
    SELECT sm.series AS symbol, sm.ts AS timestamp,
           mem.benchmark_index, mem.membership_type,
           (sm.move - bm.move) AS rs_value, '{timeframe}' AS timeframe
    FROM stk_move sm
    JOIN membership mem ON sm.series = mem.symbol
    JOIN bench_move bm ON bm.series = mem.benchmark_index AND bm.ts = sm.ts
    """

    # Note: stock filenames come back with OS separators; normalise before regex.
    # DuckDB on Windows returns backslashes for globs but forward slashes for an
    # explicit file list, so the 'stocks/(...)/5min' regex works here.

    # clear only this timeframe's partition subtree, leave others intact
    tf_dir = RS_DIR / f"timeframe={timeframe}"
    if tf_dir.exists():
        shutil.rmtree(tf_dir)
    RS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Writing rs_values (timeframe=%s, partitioned by benchmark) ...", timeframe)
    con.execute(
        f"COPY ({sql}) TO '{RS_DIR.as_posix()}' "
        f"(FORMAT PARQUET, PARTITION_BY (timeframe, benchmark_index), OVERWRITE_OR_IGNORE)"
    )

    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{tf_dir.as_posix()}/**/*.parquet')"
    ).fetchone()[0]
    return {"timeframe": timeframe, "n_rows": int(n_rows), "benchmarks_used": bench_used,
            "benchmarks_missing_data": bench_missing, "n_symbols": len(symbols)}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="5min", choices=["1min", "5min"])
    args = ap.parse_args()
    r = compute(timeframe=args.timeframe)
    print(f"[OK] rs_values[{r['timeframe']}]: {r['n_rows']:,} rows, "
          f"{len(r['benchmarks_used'])} benchmarks -> {RS_DIR}/timeframe={r['timeframe']}")
    if r["benchmarks_missing_data"]:
        print(f"     benchmarks with no {args.timeframe} data (skipped):", r["benchmarks_missing_data"])
