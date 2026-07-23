"""
VCF Test Base Class
===================

Every VCF test inherits from ``VCFTest``.  This base class handles:

- Output directory creation (``results/{category}/{test_id}/``)
- Chart saving (PNG + Plotly JSON) with DB registration
- Table saving (CSV + JSON) with DB registration
- Insight logging with severity levels
- Terminal output capture — every print statement is recorded
- Run metadata (timing, config snapshot, data window)

Usage::

    from strategies.VCF.tests.base_test import VCFTest

    class T07IntradayRange(VCFTest):
        TEST_NUMBER = 7
        TEST_NAME   = "Intraday Range Distribution"
        CATEGORY    = "move_characteristics"

        def run(self):
            df = self.load_master()
            # ... your analysis ...
            self.save_chart(fig, "range_box", title="Range by VIX Regime")
            self.save_table(stats_df, "regime_stats", title="Regime Statistics")
            self.log_insight("golden_mean", "211pt",
                             "Golden zone averages 211pt daily range",
                             severity="important")

    if __name__ == "__main__":
        T07IntradayRange().execute()
"""

import io
import json
import logging
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from infrastructure.db import get_connection
from strategies.VCF.config import CONFIG

logger = logging.getLogger(__name__)

# Timezone for IST
_IST = timezone(timedelta(hours=5, minutes=30))

# VCF strategy root
_VCF_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_ROOT = _VCF_ROOT / "results"


class _TeeWriter:
    """Write to both a StringIO buffer and the real stdout."""

    def __init__(self, real_stdout: io.TextIOBase):
        self._real = real_stdout
        self._buffer = io.StringIO()

    def write(self, text: str):
        self._real.write(text)
        self._buffer.write(text)

    def flush(self):
        self._real.flush()

    def getvalue(self) -> str:
        return self._buffer.getvalue()


class VCFTest:
    """Base class for all VCF tests.

    Subclasses must define:
    - ``TEST_NUMBER``: int — the test number (1–28)
    - ``TEST_NAME``: str — human-readable name
    - ``CATEGORY``: str — category slug (e.g. "move_characteristics")

    Subclasses must implement:
    - ``run(self)`` — the test logic
    """

    TEST_NUMBER: int = 0
    TEST_NAME: str = ""
    CATEGORY: str = ""

    # Set to the instrument symbol (default NIFTY).
    INSTRUMENT: str = "NIFTY"

    def __init__(self):
        self._test_id_str = f"t{self.TEST_NUMBER:02d}"
        self._output_dir = _RESULTS_ROOT / self.CATEGORY / self._test_id_str
        self._charts_dir = self._output_dir / "charts"
        self._tables_dir = self._output_dir / "tables"

        # Create output directories.
        self._charts_dir.mkdir(parents=True, exist_ok=True)
        self._tables_dir.mkdir(parents=True, exist_ok=True)

        # Internal state for the current run.
        self._conn = None
        self._run_id = None
        self._start_time = None
        self._master_df = None
        self._tee = None
        self._insights_buffer = []
        self._charts_buffer = []
        self._tables_buffer = []

    # ── Public API — Data ───────────────────────────────────────────

    def load_master(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Load the VCF master DataFrame.

        Parameters
        ----------
        start, end : str, optional
            Date bounds (inclusive), format ``"YYYY-MM-DD"``.

        Returns
        -------
        pd.DataFrame
        """
        from strategies.VCF.feature_builder import load_master

        df = load_master(start_date=start, end_date=end)
        self._master_df = df
        return df

    # ── Public API — Save Results ───────────────────────────────────

    def save_chart(
        self,
        fig,
        name: str,
        title: str = "",
        chart_type: str = "",
        description: str = "",
    ) -> Path:
        """Save a chart to the results directory and register in SQLite.

        Handles both matplotlib and plotly figures:
        - matplotlib → PNG only
        - plotly → PNG + JSON (for interactive dashboard rendering)

        Parameters
        ----------
        fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
            The chart figure.
        name : str
            Filename (without extension), e.g. ``"range_by_regime_box"``.
        title : str
            Display title for the dashboard.
        chart_type : str
            Chart type: "bar", "box", "line", "heatmap", "scatter", "histogram".
        description : str
            Optional description of what the chart shows.

        Returns
        -------
        Path
            Path to the saved PNG file.
        """
        png_path = self._charts_dir / f"{name}.png"
        plotly_json_str = None

        # Detect figure type and save accordingly.
        try:
            import plotly.graph_objects as go
            if isinstance(fig, go.Figure):
                # Plotly figure — save PNG + JSON.
                fig.write_image(str(png_path), width=1200, height=700, scale=2)
                plotly_json_str = fig.to_json()
            else:
                raise TypeError("Not a plotly figure")
        except (ImportError, TypeError):
            # Matplotlib figure.
            fig.savefig(
                str(png_path), dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none",
            )

        # Relative path for DB storage (relative to VCF results root).
        rel_path = png_path.relative_to(_RESULTS_ROOT).as_posix()

        self._charts_buffer.append({
            "filename": rel_path,
            "title": title or name.replace("_", " ").title(),
            "chart_type": chart_type,
            "description": description,
            "plotly_json": plotly_json_str,
        })

        print(f"    📊 Chart saved: {name}.png")
        return png_path

    def save_table(
        self,
        df: pd.DataFrame,
        name: str,
        title: str = "",
        description: str = "",
    ) -> Path:
        """Save a DataFrame as CSV and register in SQLite.

        The full table data is also stored as JSON in the database
        for direct rendering in the dashboard (no CSV parsing needed).

        Parameters
        ----------
        df : pd.DataFrame
            The data to save.
        name : str
            Filename (without extension), e.g. ``"range_by_regime"``.
        title : str
            Display title for the dashboard.
        description : str
            Optional description of what the table contains.

        Returns
        -------
        Path
            Path to the saved CSV file.
        """
        csv_path = self._tables_dir / f"{name}.csv"
        df.to_csv(str(csv_path))

        # Convert to JSON for dashboard rendering.
        # Reset index so index columns appear as regular columns.
        df_for_json = df.reset_index() if df.index.name else df
        data_json = df_for_json.to_json(orient="records", date_format="iso")

        rel_path = csv_path.relative_to(_RESULTS_ROOT).as_posix()

        self._tables_buffer.append({
            "filename": rel_path,
            "title": title or name.replace("_", " ").title(),
            "description": description,
            "row_count": len(df),
            "column_count": len(df.columns),
            "data_json": data_json,
        })

        print(f"    📋 Table saved: {name}.csv ({len(df)} rows × {len(df.columns)} cols)")
        return csv_path

    def log_insight(
        self,
        key: str,
        value: str,
        text: str,
        severity: str = "info",
    ) -> None:
        """Record an insight / finding from this test.

        Parameters
        ----------
        key : str
            Machine-readable key, e.g. ``"golden_zone_swing_rate"``.
        value : str
            The value, e.g. ``"91.7%"``.
        text : str
            Human-readable explanation. This is what shows on the
            dashboard insight card.
        severity : str
            One of ``"info"``, ``"important"``, ``"critical"``.
        """
        if severity not in ("info", "important", "critical"):
            raise ValueError(f"severity must be info/important/critical, got {severity!r}")

        self._insights_buffer.append({
            "insight_key": key,
            "insight_value": value,
            "insight_text": text,
            "severity": severity,
        })

        icons = {"info": "ℹ️", "important": "⚠️", "critical": "🔴"}
        print(f"    {icons[severity]} Insight [{severity}]: {text}")

    # ── Execution Entry Point ───────────────────────────────────────

    def execute(self, notes: str = "") -> None:
        """Run the test with full infrastructure wrapping.

        This is the method you call to run the test.  It:
        1. Opens DB connection and creates a test_run row
        2. Captures all terminal output
        3. Calls ``self.run()`` (your analysis logic)
        4. Writes all buffered results to SQLite
        5. Records duration and status

        Parameters
        ----------
        notes : str, optional
            Any manual notes to attach to this run.
        """
        self._conn = get_connection()
        self._start_time = time.time()

        # Start capturing terminal output.
        real_stdout = sys.stdout
        self._tee = _TeeWriter(real_stdout)
        sys.stdout = self._tee

        # Look up test_id and instrument_id from DB.
        test_row = self._conn.execute(
            "SELECT t.id FROM tests t "
            "JOIN test_categories tc ON t.category_id = tc.id "
            "WHERE tc.slug = ? AND t.test_number = ?",
            (self.CATEGORY, self.TEST_NUMBER),
        ).fetchone()

        if not test_row:
            sys.stdout = real_stdout
            raise RuntimeError(
                f"Test {self.CATEGORY}/{self._test_id_str} not found in DB. "
                f"Run scripts/init_db.py first."
            )

        inst_row = self._conn.execute(
            "SELECT id FROM instruments WHERE symbol = ?",
            (self.INSTRUMENT,),
        ).fetchone()

        if not inst_row:
            sys.stdout = real_stdout
            raise RuntimeError(
                f"Instrument {self.INSTRUMENT!r} not found in DB. "
                f"Run scripts/init_db.py first."
            )

        db_test_id = test_row["id"]
        db_instrument_id = inst_row["id"]
        run_at = datetime.now(_IST).isoformat(timespec="seconds")

        # Snapshot relevant config.
        config_snap = json.dumps({
            k: v for section in ("VCF", "MARKET")
            for k, v in CONFIG[section].items()
            if not callable(v)
        }, default=str)

        # Create test_run row.
        cursor = self._conn.execute(
            "INSERT INTO test_runs "
            "(test_id, instrument_id, run_at, status, config_snapshot) "
            "VALUES (?, ?, ?, 'running', ?)",
            (db_test_id, db_instrument_id, run_at, config_snap),
        )
        self._run_id = cursor.lastrowid
        self._conn.commit()

        # Run the test.
        print("=" * 60)
        print(f"  {self._test_id_str.upper()} — {self.TEST_NAME}")
        print(f"  Category: {self.CATEGORY}")
        print(f"  Instrument: {self.INSTRUMENT}")
        print(f"  Run at: {run_at}")
        print("=" * 60)

        status = "success"
        try:
            self.run()
        except Exception as e:
            status = "failed"
            logger.exception("Test %s failed", self._test_id_str)
            print(f"\n  ❌ TEST FAILED: {e}")
            raise
        finally:
            duration = time.time() - self._start_time

            # Determine data window from master DataFrame if loaded.
            data_start = data_end = None
            row_count = None
            if self._master_df is not None and len(self._master_df) > 0:
                data_start = str(self._master_df.index.min().date())
                data_end = str(self._master_df.index.max().date())
                row_count = len(self._master_df)

            # Capture terminal output.
            terminal_output = self._tee.getvalue()
            sys.stdout = real_stdout

            # Write all buffered data to SQLite.
            self._flush_to_db(
                status=status,
                duration=duration,
                data_start=data_start,
                data_end=data_end,
                row_count=row_count,
                terminal_output=terminal_output,
                notes=notes,
            )

            self._conn.close()

            # Print summary.
            print(f"\n{'=' * 60}")
            print(f"  ✅ {self._test_id_str.upper()} — COMPLETE")
            print(f"  Duration:  {duration:.1f}s")
            print(f"  Charts:    {len(self._charts_buffer)}")
            print(f"  Tables:    {len(self._tables_buffer)}")
            print(f"  Insights:  {len(self._insights_buffer)}")
            if data_start:
                print(f"  Data:      {data_start} to {data_end} ({row_count:,} rows)")
            print(f"  Status:    {status}")
            print(f"{'=' * 60}")

    def run(self):
        """Override this method with your test logic."""
        raise NotImplementedError("Subclass must implement run()")

    # ── Private ─────────────────────────────────────────────────────

    def _flush_to_db(
        self,
        status: str,
        duration: float,
        data_start: str | None,
        data_end: str | None,
        row_count: int | None,
        terminal_output: str,
        notes: str,
    ) -> None:
        """Write all buffered results to SQLite in one transaction."""

        # Update the test_run row.
        self._conn.execute(
            "UPDATE test_runs SET "
            "status = ?, duration_seconds = ?, "
            "data_window_start = ?, data_window_end = ?, "
            "row_count = ?, terminal_output = ?, notes = ? "
            "WHERE id = ?",
            (status, round(duration, 2), data_start, data_end,
             row_count, terminal_output, notes, self._run_id),
        )

        # Insert insights.
        for ins in self._insights_buffer:
            self._conn.execute(
                "INSERT INTO insights "
                "(test_run_id, insight_key, insight_value, "
                " insight_text, severity) "
                "VALUES (?, ?, ?, ?, ?)",
                (self._run_id, ins["insight_key"], ins["insight_value"],
                 ins["insight_text"], ins["severity"]),
            )

        # Insert charts.
        for ch in self._charts_buffer:
            self._conn.execute(
                "INSERT INTO charts "
                "(test_run_id, filename, title, chart_type, "
                " description, plotly_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (self._run_id, ch["filename"], ch["title"],
                 ch["chart_type"], ch["description"], ch["plotly_json"]),
            )

        # Insert tables.
        for tb in self._tables_buffer:
            self._conn.execute(
                "INSERT INTO data_tables "
                "(test_run_id, filename, title, description, "
                " row_count, column_count, data_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self._run_id, tb["filename"], tb["title"],
                 tb["description"], tb["row_count"],
                 tb["column_count"], tb["data_json"]),
            )

        self._conn.commit()
        logger.info(
            "Flushed %d insights, %d charts, %d tables to DB (run_id=%d)",
            len(self._insights_buffer),
            len(self._charts_buffer),
            len(self._tables_buffer),
            self._run_id,
        )
