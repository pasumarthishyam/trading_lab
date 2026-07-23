"""
Initial Data Fetch Script
==========================

Orchestrates the first-time data download for the VCF strategy.
Fetches all four datasets in priority order, validates immediately
after each fetch, and stops on first validation failure.

All from_dates are computed dynamically from today minus the API's
max_days limit (with safety margin):
    - Daily:    today - 1950 days  (API limit 2000, 50-day margin)
    - 15-minute: today - 195 days  (API limit 200, 5-day margin)
    - 1-minute:  today - 58 days   (API limit 60, 2-day margin)

Usage:
    python scripts/fetch_initial_data.py

Prerequisites:
    - ``ZERODHA_ACCESS_TOKEN`` is set  (run ``generate_token.py`` first)
"""

import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Add project root to sys.path so infrastructure imports work
# regardless of whether pip install -e . has been run.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ── Dynamic date calculation ────────────────────────────────────────
# All from_dates respect Zerodha API max_days limits with safety margins.
# See infrastructure/data/ingestion.py TIMEFRAME_MAP for limits.

TODAY: str = str(date.today())

# Daily:  API limit = 2000 days.  Safety margin = 50 days.
DAILY_FROM: str = str(date.today() - timedelta(days=1950))

# 15-min: API limit = 200 days.   Safety margin = 5 days.
FIFTEEN_MIN_FROM: str = str(date.today() - timedelta(days=195))

# 1-min:  API limit = 60 days.    Safety margin = 2 days.
ONE_MIN_FROM: str = str(date.today() - timedelta(days=58))

# Rate-limit delay between API calls (seconds).
_INTER_FETCH_DELAY: float = 0.5

# ── Fetch plan ──────────────────────────────────────────────────────

FETCH_PLAN: list[dict[str, str]] = [
    {
        "label": "NIFTY daily",
        "symbol": "NIFTY",
        "asset_type": "index",
        "timeframe": "daily",
        "from_date": DAILY_FROM,
        "to_date": TODAY,
    },
    {
        "label": "INDIAVIX daily",
        "symbol": "INDIAVIX",
        "asset_type": "volatility",
        "timeframe": "daily",
        "from_date": DAILY_FROM,
        "to_date": TODAY,
    },
    {
        "label": "NIFTY 15min",
        "symbol": "NIFTY",
        "asset_type": "index",
        "timeframe": "15min",
        "from_date": FIFTEEN_MIN_FROM,
        "to_date": TODAY,
    },
    {
        "label": "NIFTY 1min",
        "symbol": "NIFTY",
        "asset_type": "index",
        "timeframe": "1min",
        "from_date": ONE_MIN_FROM,
        "to_date": TODAY,
    },
]


def main() -> None:
    """Execute the full fetch-and-validate pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from infrastructure.data.ingestion import fetch_and_store
    from infrastructure.data.validation import validate

    print("=" * 60)
    print("  Initial Data Fetch -- VCF Strategy")
    print(f"  Today: {TODAY}")
    print("=" * 60)

    total = len(FETCH_PLAN)

    for i, job in enumerate(FETCH_PLAN, start=1):
        label = job["label"]
        print(f"\n[{i}/{total}] Fetching {label}...")
        print(f"         {job['from_date']} -> {job['to_date']}")

        # ── Fetch ───────────────────────────────────────────────────
        df = fetch_and_store(
            symbol=job["symbol"],
            asset_type=job["asset_type"],
            timeframe=job["timeframe"],
            from_date=job["from_date"],
            to_date=job["to_date"],
        )
        print(f"    [OK]  Fetched {len(df):,} rows")

        # ── Validate ────────────────────────────────────────────────
        # VIX is a volatility measure — spike check is meaningless for it.
        # Real events (COVID, tariff shocks) cause 50-70%+ daily swings.
        # Disable for volatility by setting threshold to 100%.
        spike_threshold = 1.00 if job["asset_type"] == "volatility" else 0.10
        report = validate(
            symbol=job["symbol"],
            asset_type=job["asset_type"],
            timeframe=job["timeframe"],
            spike_threshold=spike_threshold,
        )

        if report["clean"]:
            print(f"    [OK]  Validation PASSED -- {report['rows']:,} rows, clean")
        else:
            print(f"    [FAIL]  Validation FAILED:")
            for issue in report["issues"]:
                print(f"       - {issue}")
            print(f"\n[STOP]  Fix {label} issues before proceeding.")
            sys.exit(1)

        # Rate-limit delay before next fetch.
        if i < total:
            time.sleep(_INTER_FETCH_DELAY)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  [SUCCESS]  All datasets fetched and validated!")
    print("=" * 60)
    print(f"\n  Files saved under: data/raw/")
    print(f"    - indices/NIFTY/daily.parquet")
    print(f"    - indices/NIFTY/15min.parquet")
    print(f"    - indices/NIFTY/1min.parquet")
    print(f"    - volatility/INDIAVIX/daily.parquet")
    print(f"\n  Next step: build features with strategies/VCF/feature_builder.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
