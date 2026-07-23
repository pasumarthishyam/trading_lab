"""
T08 — Directional Move Frequency
=================================

Measures how often Nifty makes clean directional swings that exceed
the capture zone thresholds (100pt, 150pt).

Analyses across:
- Multiple swing reversal thresholds (20pt, 30pt, 40pt)
- VIX regimes
- Directionality ratio (how much of daily range is one clean swing)
- False start counts per regime

Uses 1-minute swing data from the VCF master DataFrame.
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


class T08DirectionalMoveFreq(VCFTest):
    TEST_NUMBER = 8
    TEST_NAME = "Directional Move Frequency"
    CATEGORY = "move_characteristics"

    def run(self):
        df = self.load_master()
        df = df.copy()
        df["range"] = df["session_range"]

        thresholds = CONFIG["VCF"]["swing_reversal_thresholds"]
        default_thr = CONFIG["VCF"]["swing_reversal_default"]
        regime_order = list(CONFIG["VCF"]["vix_bands"].keys())

        # ── 1. Swing stats by threshold ─────────────────────────────
        print("\n  [1/4] Analysing swing magnitude by reversal threshold...")

        threshold_rows = []
        for thr in thresholds:
            mag_col = f"swing_{thr}_magnitude"
            count_col = f"swing_{thr}_count"

            subset = df[df[mag_col].notna()].copy()
            if len(subset) == 0:
                print(f"    ⚠️ No data for threshold {thr}pt — skipping")
                continue

            n = len(subset)
            ge_100 = (subset[mag_col] >= 100).sum()
            ge_150 = (subset[mag_col] >= 150).sum()
            median_mag = subset[mag_col].median()
            mean_mag = subset[mag_col].mean()
            mean_count = subset[count_col].mean() if count_col in subset.columns else None

            threshold_rows.append({
                "reversal_threshold": f"{thr}pt",
                "days": n,
                "ge_100pt_pct": round(ge_100 / n * 100, 1),
                "ge_150pt_pct": round(ge_150 / n * 100, 1),
                "median_swing": round(median_mag, 1),
                "mean_swing": round(mean_mag, 1),
                "mean_false_starts": round(mean_count, 1) if mean_count else None,
            })

        if threshold_rows:
            thr_df = pd.DataFrame(threshold_rows).set_index("reversal_threshold")
            self.save_table(thr_df, "swing_by_threshold",
                            title="Swing Statistics by Reversal Threshold",
                            description="How swing detection sensitivity affects measured move frequency")

            # Threshold sensitivity bar chart.
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(thr_df))
            width = 0.3
            ax.bar(x - width / 2, thr_df["ge_100pt_pct"], width,
                   label="≥100pt swing", color="#4CAF50", alpha=0.8)
            ax.bar(x + width / 2, thr_df["ge_150pt_pct"], width,
                   label="≥150pt swing", color="#FF9800", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(thr_df.index)
            ax.set_title("Swing Frequency by Reversal Threshold", fontsize=14, pad=15)
            ax.set_ylabel("% of Days")
            ax.set_xlabel("Reversal Threshold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            self.save_chart(fig, "threshold_sensitivity",
                            title="Threshold Sensitivity Analysis",
                            chart_type="bar",
                            description="Impact of reversal threshold on measured swing frequency")
            plt.close(fig)

        # ── 2. Swing stats by VIX regime (default threshold) ────────
        print(f"  [2/4] Analysing swings by VIX regime ({default_thr}pt threshold)...")

        mag_col = f"swing_{default_thr}_magnitude"
        count_col = f"swing_{default_thr}_count"
        swing_valid = df[df[mag_col].notna()].copy()

        if len(swing_valid) > 0:
            regime_rows = []
            for regime in regime_order:
                subset = swing_valid[swing_valid["vix_regime"] == regime]
                if len(subset) == 0:
                    continue
                n = len(subset)
                ge_100 = (subset[mag_col] >= 100).sum()
                ge_150 = (subset[mag_col] >= 150).sum()
                mean_swing = subset[mag_col].mean()
                mean_starts = subset[count_col].mean() if count_col in subset.columns else None

                regime_rows.append({
                    "regime": regime,
                    "days": n,
                    "ge_100pt_pct": round(ge_100 / n * 100, 1),
                    "ge_150pt_pct": round(ge_150 / n * 100, 1),
                    "mean_swing": round(mean_swing, 1),
                    "false_starts": round(mean_starts, 1) if mean_starts else None,
                })

            if regime_rows:
                regime_df = pd.DataFrame(regime_rows).set_index("regime")
                self.save_table(regime_df, "swing_by_regime",
                                title=f"Swing Stats by VIX Regime ({default_thr}pt threshold)",
                                description="Directional move frequency and quality by VIX regime")

                # Bar chart.
                fig, ax = plt.subplots(figsize=(12, 6))
                x = np.arange(len(regime_df))
                width = 0.3
                ax.bar(x - width / 2, regime_df["ge_100pt_pct"], width,
                       label="≥100pt swing", color="#4CAF50", alpha=0.8)
                ax.bar(x + width / 2, regime_df["ge_150pt_pct"], width,
                       label="≥150pt swing", color="#FF9800", alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(regime_df.index, rotation=0)
                ax.set_title(f"Swing Frequency by VIX Regime ({default_thr}pt threshold)",
                             fontsize=14, pad=15)
                ax.set_ylabel("% of Days")
                ax.set_xlabel("VIX Regime")
                ax.legend()
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                self.save_chart(fig, "swing_by_regime_bar",
                                title="Swing Frequency by Regime",
                                chart_type="bar",
                                description="Directional swing hit rate by VIX regime")
                plt.close(fig)

                # Insights from regime data.
                if "golden" in regime_df.index:
                    golden = regime_df.loc["golden"]
                    self.log_insight(
                        "golden_zone_100pt_rate",
                        f"{golden['ge_100pt_pct']}%",
                        f"Golden zone (VIX 13-18) produces 100pt+ clean swings on "
                        f"{golden['ge_100pt_pct']}% of days — highest tradeable hit rate",
                        severity="important",
                    )
                    if golden["false_starts"] is not None:
                        self.log_insight(
                            "golden_zone_false_starts",
                            f"{golden['false_starts']:.0f}",
                            f"Golden zone averages {golden['false_starts']:.0f} false starts "
                            f"before the real directional move — patience required",
                            severity="important",
                        )

        # ── 3. Directionality Analysis ──────────────────────────────
        print("  [3/4] Computing directionality ratio...")

        if len(swing_valid) > 0 and "range" in swing_valid.columns:
            swing_valid_dir = swing_valid[
                (swing_valid["range"] > 0) & (swing_valid[mag_col] > 0)
            ].copy()
            swing_valid_dir["directionality"] = (
                swing_valid_dir[mag_col] / swing_valid_dir["range"]
            )

            dir_rows = []
            for regime in regime_order:
                subset = swing_valid_dir[swing_valid_dir["vix_regime"] == regime]
                if len(subset) == 0:
                    continue
                dir_rows.append({
                    "regime": regime,
                    "days": len(subset),
                    "avg_range": round(subset["range"].mean(), 1),
                    "avg_swing": round(subset[mag_col].mean(), 1),
                    "directionality": round(subset["directionality"].mean() * 100, 1),
                })

            if dir_rows:
                dir_df = pd.DataFrame(dir_rows).set_index("regime")
                self.save_table(dir_df, "directionality_analysis",
                                title="Directionality Ratio by VIX Regime",
                                description="What fraction of daily range is captured in one clean directional swing")

                # Scatter plot: range vs swing by regime.
                fig, ax = plt.subplots(figsize=(10, 8))
                colors_map = {
                    "tight": "#4CAF50", "functional": "#2196F3",
                    "golden": "#FF9800", "elevated": "#F44336",
                    "spreads_only": "#9C27B0", "no_trade": "#795548",
                }
                for regime in regime_order:
                    subset = swing_valid_dir[swing_valid_dir["vix_regime"] == regime]
                    if len(subset) == 0:
                        continue
                    ax.scatter(subset["range"], subset[mag_col],
                               label=regime, color=colors_map.get(regime, "gray"),
                               alpha=0.6, s=40)

                # 100% directionality line.
                max_range = swing_valid_dir["range"].max()
                ax.plot([0, max_range], [0, max_range], "k--", alpha=0.3,
                        label="100% directional")
                ax.set_title("Daily Range vs Largest Clean Swing", fontsize=14, pad=15)
                ax.set_xlabel("Daily Range (points)")
                ax.set_ylabel(f"Largest Swing ({default_thr}pt threshold)")
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3)
                plt.tight_layout()
                self.save_chart(fig, "range_vs_swing_scatter",
                                title="Range vs Directional Swing",
                                chart_type="scatter",
                                description="Relationship between total daily range and clean directional move")
                plt.close(fig)

                # Directionality insight.
                if "golden" in dir_df.index:
                    golden_dir = dir_df.loc["golden", "directionality"]
                    self.log_insight(
                        "golden_directionality",
                        f"{golden_dir}%",
                        f"Golden zone has {golden_dir}% directionality — only "
                        f"~{golden_dir:.0f}% of the daily range is one clean move. "
                        f"The rest is chop and counter-movement",
                        severity="critical",
                    )

        # ── 4. Swing magnitude histogram ────────────────────────────
        print("  [4/4] Creating swing magnitude histogram...")

        if len(swing_valid) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(swing_valid[mag_col], bins=40, color="#2196F3",
                    alpha=0.7, edgecolor="white")
            for thr, color, ls in [(100, "green", "--"), (150, "orange", "--")]:
                ax.axvline(thr, color=color, linestyle=ls, linewidth=2,
                           label=f"{thr}pt capture zone")
            ax.set_title(f"Swing Magnitude Distribution ({default_thr}pt threshold)",
                         fontsize=14, pad=15)
            ax.set_xlabel("Swing Magnitude (points)")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            self.save_chart(fig, "swing_magnitude_hist",
                            title="Swing Magnitude Distribution",
                            chart_type="histogram",
                            description="Distribution of largest clean swing magnitudes")
            plt.close(fig)

            # Overall insight.
            median_swing = swing_valid[mag_col].median()
            n_swing_days = len(swing_valid)
            overall_100_pct = (swing_valid[mag_col] >= 100).mean() * 100
            self.log_insight(
                "overall_swing_100pt",
                f"{overall_100_pct:.1f}%",
                f"{overall_100_pct:.1f}% of days with swing data ({n_swing_days} days) "
                f"produce a ≥100pt clean directional swing. Median swing: {median_swing:.0f}pt",
                severity="important",
            )

        print("\n  ✅ T08 analysis complete.")


if __name__ == "__main__":
    T08DirectionalMoveFreq().execute()
