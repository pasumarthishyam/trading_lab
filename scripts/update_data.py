"""
Periodic Data Update Script
============================

One-command update for all VCF instruments.  Calls
``batch_update()`` which handles incremental fetches,
rate limiting, and exponential backoff automatically.

Usage:
    python scripts/update_data.py

Schedule:
    - 1-minute:  run every 45–50 days  (60-day API window)
    - 15-minute: run every 3–4 months  (200-day API window)
    - Daily:     run every 6 months    (2000-day API window)

Running this script covers all three at once — the ``update()``
function inside ``batch_update`` will skip instruments that are
already current.
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path so infrastructure imports work
# regardless of whether pip install -e . has been run.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main() -> None:
    """Run batch update for all VCF instruments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from infrastructure.data.ingestion import batch_update

    print("=" * 60)
    print("  Periodic Data Update — All VCF Instruments")
    print("=" * 60)

    batch_update([
        ("NIFTY", "index", "1min"),
        ("NIFTY", "index", "15min"),
        ("NIFTY", "index", "daily"),
        ("INDIAVIX", "volatility", "daily"),
    ])

    print("\n✅  Batch update complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
