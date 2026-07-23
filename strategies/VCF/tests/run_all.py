"""
VCF Test Runner
===============

Discovers and runs all VCF tests in dependency order:
  Foundation → Move Characteristics → VIX Regime →
  Options Simulation → Framework Performance

Usage::

    # Run all tests
    python -m strategies.VCF.tests.run_all

    # Run a specific category
    python -m strategies.VCF.tests.run_all --category move_characteristics

    # Dry run — just list discovered tests
    python -m strategies.VCF.tests.run_all --dry-run
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from infrastructure.db import get_connection

# Category execution order (dependency chain).
_CATEGORY_ORDER = [
    "foundation",
    "move_characteristics",
    "vix_regime",
    "options_simulation",
    "framework_performance",
]

# Maps category → list of module paths (auto-discovered).
_TESTS_DIR = Path(__file__).resolve().parent


def discover_tests(category_filter: str | None = None) -> list[dict]:
    """Find all test modules in the tests directory.

    Returns a list of dicts with keys: category, module_path, filename.
    """
    found = []
    categories = [category_filter] if category_filter else _CATEGORY_ORDER

    for category in categories:
        cat_dir = _TESTS_DIR / category
        if not cat_dir.is_dir():
            continue

        for py_file in sorted(cat_dir.glob("t[0-9][0-9]_*.py")):
            module_name = (
                f"strategies.VCF.tests.{category}.{py_file.stem}"
            )
            found.append({
                "category": category,
                "module_path": module_name,
                "filename": py_file.name,
            })

    return found


def run_all(category_filter: str | None = None, dry_run: bool = False):
    """Discover and run all tests."""
    tests = discover_tests(category_filter)

    print("=" * 60)
    print("  VCF TEST SUITE")
    print("=" * 60)
    print(f"\n  Discovered {len(tests)} test(s):\n")

    for t in tests:
        print(f"    [{t['category']:25s}]  {t['filename']}")

    if dry_run:
        print("\n  (Dry run — no tests executed)")
        return

    if not tests:
        print("\n  No test files found. Create test files matching")
        print("  pattern t[0-9][0-9]_*.py in category subdirectories.")
        return

    print(f"\n{'─' * 60}")

    results = []
    for t in tests:
        try:
            module = importlib.import_module(t["module_path"])

            # Find the VCFTest subclass in the module.
            test_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and hasattr(attr, "execute")
                    and hasattr(attr, "TEST_NUMBER")
                    and attr.TEST_NUMBER > 0
                ):
                    test_class = attr
                    break

            if test_class is None:
                print(f"\n  ⚠️  No VCFTest subclass found in {t['filename']}")
                results.append({**t, "status": "skipped"})
                continue

            instance = test_class()
            instance.execute()
            results.append({**t, "status": "success"})

        except Exception as e:
            logging.exception("Test %s failed", t["filename"])
            results.append({**t, "status": "failed", "error": str(e)})

    # Print summary grid.
    print(f"\n{'=' * 60}")
    print("  TEST SUITE SUMMARY")
    print(f"{'=' * 60}\n")

    icons = {"success": "✅", "failed": "❌", "skipped": "⚠️"}
    for r in results:
        icon = icons.get(r["status"], "?")
        print(f"    {icon}  {r['filename']:40s}  {r['status']}")

    passed = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"\n  Total: {len(results)}  |  ✅ {passed}  |  ❌ {failed}  |  ⚠️ {skipped}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run VCF test suite")
    parser.add_argument(
        "--category", type=str, default=None,
        choices=_CATEGORY_ORDER,
        help="Run only tests in this category",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List discovered tests without running them",
    )
    args = parser.parse_args()

    run_all(category_filter=args.category, dry_run=args.dry_run)
