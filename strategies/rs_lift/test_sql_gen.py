"""Quick SQL generation test"""
import sys
sys.path.insert(0, 'c:/Users/Hp/Documents/trading_lab')

CHECKPOINTS = ["09:25", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30"]

fa_cols = ", ".join(
    f"COALESCE(a.cummin_{cp.replace(':', '')} >= o.low0915 - 1e-9, false) AS fa_long_{cp.replace(':', '')},  "
    f"COALESCE(a.cummax_{cp.replace(':', '')} <= o.high0915 + 1e-9, false) AS fa_short_{cp.replace(':', '')}"
    for cp in CHECKPOINTS
)
print("fa_cols sample (first 200 chars):", fa_cols[:200])
print()

# Test SQL parse with duckdb in-memory
import duckdb
con = duckdb.connect()
try:
    test_sql = f"""
    SELECT 1 AS x,
        {fa_cols}
    FROM (SELECT 1 AS col_a) agg,
         (SELECT 1 AS col_o, 0.5 AS low0915, 1.5 AS high0915) o
    """
    # Rename col references to be valid
    print("SQL snippet (first 400):", test_sql[:400])
except Exception as e:
    print("Error:", e)
