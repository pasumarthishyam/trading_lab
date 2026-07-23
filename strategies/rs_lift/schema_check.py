"""Quick schema check for RS values and substrate parquets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
con = duckdb.connect()

# RS schema
rs_glob = (REPO / "data/rs/rs_values/**/*.parquet").as_posix()
schema = con.execute(
    f"DESCRIBE SELECT * FROM read_parquet('{rs_glob}', hive_partitioning=true) LIMIT 1"
).fetchdf()
print("RS schema:")
print(schema.to_string())
print()

# Sample RS rows
sample = con.execute(
    f"SELECT * FROM read_parquet('{rs_glob}', hive_partitioning=true) LIMIT 5"
).fetchdf()
print("RS sample rows:")
print(sample)
print("RS dtypes:", sample.dtypes.to_dict())
print()

# Picks schema
picks = pd.read_parquet(REPO / "strategies/RFactor/results/move_validation/substrate/picks.parquet")
print("Picks cols:", list(picks.columns))
print("Picks date dtype:", picks["date"].dtype, "| sample:", picks["date"].head(3).tolist())
print("Picks shape:", picks.shape)
print()

# Daily schema
daily = pd.read_parquet(REPO / "strategies/RFactor/results/move_validation/substrate/universe_daily.parquet")
print("Daily cols:", list(daily.columns))
print("Daily date dtype:", daily["date"].dtype, "| sample:", daily["date"].head(3).tolist())
print("Daily shape:", daily.shape)
print()

# Sector map
smap = pd.read_csv(REPO / "data/rs/sector_map.csv")
print("Sector map cols:", list(smap.columns))
print("Sector map shape:", smap.shape)
print("membership_type counts:")
print(smap["membership_type"].value_counts())
print("Sample multi-sector stock:")
print(smap[smap["membership_type"] == "sector"].groupby("symbol").size().nlargest(5))
