"""
Step 6: Cross-Platform Comparison

Compares price patterns, volume distributions, and volatility between
Polymarket and Kalshi for matched markets. Identifies divergence windows
and correlates them with whale activity.

Usage:
    poetry run python scripts/06_cross_platform_comparison.py

Outputs:
    data/processed/price_comparison.csv
    data/processed/divergence_analysis.json
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

PROCESSED_DIR = Path("data/processed")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── 1. Load trade data ───────────────────────────────────────────────
section("1. Loading cross-platform trade data")

poly_df = pd.read_csv(PROCESSED_DIR / "poly_trades_all_matched.csv")
poly_df["timestamp"] = pd.to_datetime(poly_df["timestamp"], utc=True)

kalshi_df = pd.read_csv(PROCESSED_DIR / "kalshi_trades_all_matched.csv")
kalshi_df["create_date"] = pd.to_datetime(kalshi_df["create_date"], utc=True)

print(f"Polymarket: {len(poly_df):,} trades")
print(f"Kalshi: {len(kalshi_df):,} trades")


# ── 2. Compute hourly price series ───────────────────────────────────
section("2. Computing hourly price series")

# Polymarket: volume-weighted average price per hour
poly_hourly = poly_df.groupby(poly_df["timestamp"].dt.floor("1h")).apply(
    lambda x: np.average(x["price"], weights=x["size"]) if len(x) > 0 else np.nan
)
poly_hourly.name = "poly_price"

# Kalshi: count-weighted average price per hour
kalshi_hourly = kalshi_df.groupby(kalshi_df["create_date"].dt.floor("1h")).apply(
    lambda x: np.average(x["price"], weights=x["count"]) if len(x) > 0 else np.nan
)
kalshi_hourly.name = "kalshi_price"

# Align timestamps
aligned = pd.DataFrame({
    "poly_price": poly_hourly,
    "kalshi_price": kalshi_hourly,
}).dropna()

print(f"Aligned hourly periods: {len(aligned)}")


# ── 3. Compute metrics ───────────────────────────────────────────────
section("3. Computing comparison metrics")

if len(aligned) > 1:
    correlation, pvalue = pearsonr(aligned["poly_price"], aligned["kalshi_price"])
    print(f"Price correlation: {correlation:.3f} (p-value: {pvalue:.4f})")

    # Mean absolute divergence
    divergence = abs(aligned["poly_price"] - aligned["kalshi_price"]).mean()
    print(f"Mean absolute divergence: {divergence:.4f}")

    # Max divergence
    max_divergence = abs(aligned["poly_price"] - aligned["kalshi_price"]).max()
    print(f"Max divergence: {max_divergence:.4f}")

    # Volatility comparison
    poly_volatility = aligned["poly_price"].std()
    kalshi_volatility = aligned["kalshi_price"].std()
    print(f"\nVolatility:")
    print(f"  Polymarket: {poly_volatility:.4f}")
    print(f"  Kalshi: {kalshi_volatility:.4f}")

    # Identify divergence windows (>5% gap)
    aligned["divergence"] = abs(aligned["poly_price"] - aligned["kalshi_price"])
    divergence_threshold = 0.05
    divergence_periods = aligned[aligned["divergence"] > divergence_threshold]
    print(f"\nDivergence periods (>5%): {len(divergence_periods)}")


# ── 4. Trade volume comparison ───────────────────────────────────────
section("4. Trade volume comparison")

poly_hourly_volume = poly_df.groupby(poly_df["timestamp"].dt.floor("1h"))["size"].sum()
kalshi_hourly_volume = kalshi_df.groupby(kalshi_df["create_date"].dt.floor("1h"))["count"].sum()

print(f"Polymarket avg hourly volume: {poly_hourly_volume.mean():,.0f} shares")
print(f"Kalshi avg hourly volume: {kalshi_hourly_volume.mean():,.0f} contracts")

# Volume spike analysis
poly_q75 = poly_hourly_volume.quantile(0.75)
poly_spikes = (poly_hourly_volume > poly_q75).sum()
print(f"Polymarket volume spikes (>Q75): {poly_spikes}")


# ── 5. Save results ──────────────────────────────────────────────────
section("5. Saving results")

aligned.to_csv(PROCESSED_DIR / "price_comparison.csv")
print("Saved: price_comparison.csv")

divergence_data = {
    "correlation": float(correlation) if len(aligned) > 1 else None,
    "mean_divergence": float(divergence) if len(aligned) > 1 else None,
    "max_divergence": float(max_divergence) if len(aligned) > 1 else None,
    "poly_volatility": float(poly_volatility) if len(aligned) > 1 else None,
    "kalshi_volatility": float(kalshi_volatility) if len(aligned) > 1 else None,
    "divergence_periods": int(len(divergence_periods)) if len(aligned) > 1 else 0,
}

with open(PROCESSED_DIR / "divergence_analysis.json", "w") as f:
    json.dump(divergence_data, f, indent=2)
print("Saved: divergence_analysis.json")

print("\nPhase 6 complete. Ready for Phase 7 (Visualization).")
