"""
Trading Lab — Database Connection
==================================

Thin wrapper around SQLite connection for the trading_lab research
platform.  All Python tests and scripts import ``get_connection()``
from here.

Usage::

    from infrastructure.db import get_connection

    conn = get_connection()
    conn.execute("SELECT * FROM strategies")
"""

import sqlite3
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "trading_lab.db"


def get_db_path() -> Path:
    """Return the absolute path to the SQLite database file."""
    return _DB_PATH


def get_connection() -> sqlite3.Connection:
    """Return a connection to the trading_lab SQLite database.

    The connection uses:
    - WAL journal mode for concurrent read/write safety
    - Foreign key enforcement
    - Row factory for dict-style access

    Returns
    -------
    sqlite3.Connection
    """
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn
