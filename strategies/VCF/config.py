"""
VCF Strategy Configuration
==========================

All parameters for the Volatility Capture Framework strategy.
Accessed as ``CONFIG["DATA"]``, ``CONFIG["MARKET"]``, etc.

Convention for band tuples: ``(lower, upper)`` represents the
half-open interval **[lower, upper)** — inclusive lower bound,
exclusive upper bound.
"""

CONFIG: dict = {
    # ── Data identifiers ────────────────────────────────────────────
    "DATA": {
        "nifty_symbol": "NIFTY",
        "vix_symbol": "INDIAVIX",
        "nifty_asset_type": "index",
        "vix_asset_type": "volatility",
        "primary_timeframes": ["1min", "15min", "daily"],
    },

    # ── Market session parameters ───────────────────────────────────
    "MARKET": {
        "session_start": "09:20",
        "session_end": "15:20",
        "trade_start": "09:45",
        "trade_end": "14:30",
        "lot_size": 25,
        "expiry_day": "Tuesday",
        "no_trade_days": ["Tuesday"],
    },

    # ── VCF framework parameters ────────────────────────────────────
    "VCF": {
        "dvr_divisor": 16,
        "capture_zone_min": 100,
        "capture_zone_max": 150,
        "option_target_points_min": 50,
        "option_target_points_max": 75,
        "option_target_return_min": 0.20,
        "option_target_return_max": 0.50,
        "atm_delta": 0.50,

        # VIX regime bands.
        # Convention: each tuple is [lower, upper) — inclusive lower,
        # exclusive upper.  For example, "golden" covers VIX values
        # where 13 <= VIX < 18.
        "vix_bands": {
            "tight":        (0, 11),
            "functional":   (11, 13),
            "golden":       (13, 18),
            "elevated":     (18, 22),
            "spreads_only": (22, 25),
            "no_trade":     (25, 9999),
        },

        "vix_direction_downgrade_threshold": 0.05,
        "swing_reversal_thresholds": [20, 30, 40],
        "swing_reversal_default": 30,
        "time_stop_minutes": 45,
        "time_stop_max_minutes": 50,
        "time_stop_exception_note": (
            "Setup-specific overrides must be defined in that setup's "
            "own config. Never decided in the moment."
        ),
        "sl_max_premium_pct": 0.35,
        "sl_preferred_premium_pct": 0.175,
        "opening_range_minutes": 15,
        "transaction_costs": {
            "optimistic": 5,
            "realistic": 8,
            "conservative": 12,
        },
    },

    # ── Backtest parameters ─────────────────────────────────────────
    "BACKTEST": {
        "risk_per_trade_pct": 0.02,
        "risk_per_trade_max_pct": 0.05,
        "max_capital_per_trade_pct": 0.25,
        "monte_carlo_simulations": 10000,
        "monte_carlo_min_sample": 150,
        "walkforward_train_pct": 0.70,
        "bs_error_margin": 0.175,
        "min_sample_for_significance": 30,
        "significance_level": 0.05,
    },
}
