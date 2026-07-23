"""
T07 — Intraday Range Distribution
==================================

Profiles Nifty's daily high-low range across:
- VIX regimes (tight, functional, golden, elevated, spreads_only, no_trade)
- Day of week
- Year
- Capture zone feasibility (what % of days have range ≥ 100pt, 150pt, 200pt)

Uses daily data from the VCF master DataFrame for full history coverage.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategies.VCF.tests.base_test import VCFTest
from strategies.VCF.config import CONFIG


class T07IntradayRange(VCFTest):
    TEST_NUMBER = 7
    TEST_NAME = "Intraday Range Distribution"
    CATEGORY = "move_characteristics"

    def run(self):
        df = self.load_master()

        # ── Use pre-computed session range ──────────────────────────
        df = df.copy()
        df["range"] = df["session_range"]
        df["day_name"] = df.index.day_name()
        df["year"] = df.index.year

        # Filter to rows with valid VIX regime and range.
        valid = df[df["vix_regime"].notna() & df["range"].notna()].copy()
        print(f"\n  Working with {len(valid):,} valid trading days")

        # ── 1. Overall Statistics ───────────────────────────────────
        print("\n  [1/5] Computing overall statistics...")
        overall = valid["range"].describe()
        overall["median"] = valid["range"].median()
        overall_df = pd.DataFrame(overall, columns=["value"]).round(2)
        self.save_table(overall_df, "overall_stats",
                        title="Overall Range Statistics",
                        description="Descriptive statistics for Nifty daily range")

        self.log_insight(
            "mean_daily_range",
            f"{valid['range'].mean():.0f}pt",
            f"Average daily Nifty range is {valid['range'].mean():.0f} points "
            f"across {len(valid):,} trading days",
            severity="info",
        )

        # ── 2. Range by VIX Regime ──────────────────────────────────
        print("  [2/5] Analysing range by VIX regime...")

        regime_order = list(CONFIG["VCF"]["vix_bands"].keys())
        regime_stats = (
            valid.groupby("vix_regime")["range"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reindex(regime_order)
            .dropna(subset=["count"])
            .round(1)
        )
        regime_stats.columns = ["Days", "Mean", "Median", "Std", "Min", "Max"]
        self.save_table(regime_stats, "range_by_regime",
                        title="Daily Range by VIX Regime",
                        description="Range statistics broken down by VIX regime classification")

        # Box plot
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_data = [valid[valid["vix_regime"] == r]["range"].dropna()
                     for r in regime_order if r in valid["vix_regime"].values]
        plot_labels = [r for r in regime_order if r in valid["vix_regime"].values]
        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0", "#795548"]
        for patch, color in zip(bp["boxes"], colors[:len(plot_labels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title("Daily Range Distribution by VIX Regime", fontsize=14, pad=15)
        ax.set_ylabel("Range (points)")
        ax.set_xlabel("VIX Regime")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self.save_chart(fig, "range_by_regime_box",
                        title="Range Distribution by VIX Regime",
                        chart_type="box",
                        description="Box plot showing daily range spread across VIX regimes")
        plt.close(fig)

        # Find golden zone stats for insight.
        if "golden" in regime_stats.index:
            golden = regime_stats.loc["golden"]
            self.log_insight(
                "golden_zone_mean_range",
                f"{golden['Mean']:.0f}pt",
                f"Golden zone (VIX 13-18) averages {golden['Mean']:.0f}pt daily range "
                f"across {golden['Days']:.0f} days — highest tradeable regime",
                severity="important",
            )

        # ── 3. Range by Day of Week ─────────────────────────────────
        print("  [3/5] Analysing range by day of week...")

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        day_stats = (
            valid.groupby("day_name")["range"]
            .agg(["count", "mean", "median", "std"])
            .reindex(day_order)
            .dropna(subset=["count"])
            .round(1)
        )
        day_stats.columns = ["Days", "Mean", "Median", "Std"]
        self.save_table(day_stats, "range_by_weekday",
                        title="Daily Range by Weekday",
                        description="Range statistics by day of week")

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(day_stats))
        bars = ax.bar(x, day_stats["Mean"], color="#2196F3", alpha=0.8, width=0.6)
        ax.bar(x, day_stats["Median"], color="#FF9800", alpha=0.5, width=0.3, label="Median")
        ax.set_xticks(x)
        ax.set_xticklabels(day_stats.index, rotation=0)
        ax.set_title("Average Daily Range by Weekday", fontsize=14, pad=15)
        ax.set_ylabel("Range (points)")
        ax.legend(["Mean", "Median"])
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self.save_chart(fig, "range_by_weekday_bar",
                        title="Range by Weekday",
                        chart_type="bar",
                        description="Mean and median daily range by day of week")
        plt.close(fig)

        # ── 4. Range by Year ────────────────────────────────────────
        print("  [4/5] Analysing range by year...")

        year_stats = (
            valid.groupby("year")["range"]
            .agg(["count", "mean", "median", "std"])
            .round(1)
        )
        year_stats.columns = ["Days", "Mean", "Median", "Std"]
        self.save_table(year_stats, "range_by_year",
                        title="Daily Range by Year",
                        description="Range statistics by calendar year")

        # ── 5. Capture Zone Feasibility ─────────────────────────────
        print("  [5/5] Computing capture zone feasibility...")

        thresholds = [100, 150, 200, 250]
        cap_data = []
        for regime in regime_order:
            subset = valid[valid["vix_regime"] == regime]
            if len(subset) == 0:
                continue
            row = {"regime": regime, "days": len(subset)}
            for thr in thresholds:
                hit = (subset["range"] >= thr).sum()
                row[f"ge_{thr}pt_count"] = hit
                row[f"ge_{thr}pt_pct"] = round(hit / len(subset) * 100, 1)
            cap_data.append(row)

        # Add overall row.
        overall_row = {"regime": "ALL", "days": len(valid)}
        for thr in thresholds:
            hit = (valid["range"] >= thr).sum()
            overall_row[f"ge_{thr}pt_count"] = hit
            overall_row[f"ge_{thr}pt_pct"] = round(hit / len(valid) * 100, 1)
        cap_data.append(overall_row)

        cap_df = pd.DataFrame(cap_data).set_index("regime")
        self.save_table(cap_df, "capture_feasibility",
                        title="Capture Zone Feasibility",
                        description="Percentage of days where range exceeds various thresholds, by regime")

        # Stacked bar chart for capture feasibility.
        fig, ax = plt.subplots(figsize=(12, 6))
        regimes_plot = [r for r in regime_order if r in cap_df.index]
        x = np.arange(len(regimes_plot))
        width = 0.2
        colors_cap = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]

        for i, thr in enumerate(thresholds):
            vals = [cap_df.loc[r, f"ge_{thr}pt_pct"] for r in regimes_plot]
            ax.bar(x + i * width, vals, width, label=f"≥{thr}pt",
                   color=colors_cap[i], alpha=0.8)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(regimes_plot, rotation=0)
        ax.set_title("Capture Zone Feasibility by VIX Regime", fontsize=14, pad=15)
        ax.set_ylabel("% of Days")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self.save_chart(fig, "capture_feasibility_bar",
                        title="Capture Zone Feasibility by Regime",
                        chart_type="bar",
                        description="Percentage of days achieving each range threshold by regime")
        plt.close(fig)

        # Distribution histogram.
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(valid["range"], bins=60, color="#2196F3", alpha=0.7, edgecolor="white")
        for thr, color in zip([100, 150, 200], ["green", "orange", "red"]):
            ax.axvline(thr, color=color, linestyle="--", linewidth=2,
                       label=f"{thr}pt threshold")
        ax.set_title("Daily Range Distribution — All Days", fontsize=14, pad=15)
        ax.set_xlabel("Range (points)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self.save_chart(fig, "range_distribution_hist",
                        title="Daily Range Distribution",
                        chart_type="histogram",
                        description="Histogram of daily range with capture zone thresholds")
        plt.close(fig)

        # Key capture feasibility insights.
        overall_100 = cap_df.loc["ALL", "ge_100pt_pct"]
        overall_150 = cap_df.loc["ALL", "ge_150pt_pct"]
        self.log_insight(
            "overall_100pt_rate",
            f"{overall_100}%",
            f"{overall_100}% of all trading days have a range ≥100 points — "
            f"the lower bound of the capture zone is achievable on most days",
            severity="important",
        )
        self.log_insight(
            "overall_150pt_rate",
            f"{overall_150}%",
            f"{overall_150}% of all trading days have a range ≥150 points — "
            f"the upper bound of the capture zone",
            severity="info",
        )

        # DVR ratio by regime (range as % of expected move).
        if "dvr" in valid.columns:
            print("\n  [Bonus] Computing DVR ratio by regime...")
            valid_dvr = valid[valid["dvr"] > 0].copy()
            valid_dvr["dvr_ratio"] = valid_dvr["range"] / valid_dvr["dvr"]

            dvr_stats = (
                valid_dvr.groupby("vix_regime")["dvr_ratio"]
                .agg(["count", "mean", "median", "std"])
                .reindex(regime_order)
                .dropna(subset=["count"])
                .round(3)
            )
            dvr_stats.columns = ["Days", "Mean_Ratio", "Median_Ratio", "Std_Ratio"]
            self.save_table(dvr_stats, "dvr_ratio_by_regime",
                            title="DVR Ratio by VIX Regime",
                            description="Actual range as proportion of expected move (DVR)")

            overall_dvr = valid_dvr["dvr_ratio"].median()
            self.log_insight(
                "median_dvr_ratio",
                f"{overall_dvr:.2f}",
                f"Market typically uses {overall_dvr:.0%} of its expected daily move — "
                f"a ratio of {overall_dvr:.2f} (median across all regimes)",
                severity="info",
            )

        print("\n  ✅ T07 analysis complete.")


if __name__ == "__main__":
    T07IntradayRange().execute()
