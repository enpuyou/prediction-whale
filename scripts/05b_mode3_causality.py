"""
Mode 3 Temporal Causality Analysis: Does suspicious wallet trading PRECEDE retail trading?

Tests the manipulation hypothesis: if large suspicious wallets systematically
move before retail traders, that's evidence of price-leading behavior (Mode 3).

Method: Granger-style lead-lag analysis.
  - Split trades into 15-minute time bins
  - For each market, compute suspicious-wallet volume in window T
  - Correlate with retail-wallet volume in window T+1 (1-step lead)
  - Also test T+2 (30 min lead), T+3 (45 min), T+4 (60 min)
  - Positive correlation at T+k with near-zero at T-k = evidence of leading

Usage:
    poetry run python scripts/05b_mode3_causality.py

Outputs:
    data/processed/mode3_causality_results.json
    data/processed/mode3_causality_summary.txt
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROCESSED_DIR = Path("data/processed")
BIN_MINUTES = 15  # time bin width
MAX_LAG = 4       # max lag steps to test (4 × 15min = 60min)


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── 1. Load data ──────────────────────────────────────────────────────
section("1. Loading data")

df = pd.read_csv(PROCESSED_DIR / "poly_trades_all_matched.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["time_bin"] = df["timestamp"].dt.floor(f"{BIN_MINUTES}min")

features_df = pd.read_csv(PROCESSED_DIR / "wallet_features.csv")
suspicious_set = set(features_df.loc[features_df["cluster"] == -1, "wallet"])
retail_set = set(features_df.loc[features_df["cluster"] != -1, "wallet"])

df["actor"] = df["proxyWallet"].apply(
    lambda w: "suspicious" if w in suspicious_set else "retail"
)

print(f"Total trades: {len(df):,}")
print(f"Suspicious wallets: {len(suspicious_set):,}")
print(f"Retail wallets: {len(retail_set):,}")
print(f"Suspicious trades: {(df['actor']=='suspicious').sum():,}")
print(f"Retail trades: {(df['actor']=='retail').sum():,}")

markets = df["conditionId"].unique()
print(f"Markets: {len(markets):,}")
print(f"Time bin width: {BIN_MINUTES} min, max lag tested: {MAX_LAG} steps = {MAX_LAG*BIN_MINUTES} min")


# ── 2. Per-market lead-lag analysis ──────────────────────────────────
section("2. Per-market lead-lag analysis")

lag_correlations = {lag: [] for lag in range(-MAX_LAG, MAX_LAG + 1)}
market_results = []
markets_with_both = 0

for i, market in enumerate(markets):
    mdf = df[df["conditionId"] == market]
    susp = mdf[mdf["actor"] == "suspicious"]
    ret = mdf[mdf["actor"] == "retail"]

    if susp.empty or ret.empty:
        continue

    # Volume per time bin
    susp_ts = susp.groupby("time_bin")["size"].sum()
    ret_ts = ret.groupby("time_bin")["size"].sum()

    # Union of all bins with activity
    all_bins = susp_ts.index.union(ret_ts.index)
    if len(all_bins) < 10:  # need minimum bins for correlation
        continue

    susp_aligned = susp_ts.reindex(all_bins, fill_value=0).values.astype(float)
    ret_aligned = ret_ts.reindex(all_bins, fill_value=0).values.astype(float)

    markets_with_both += 1

    # Test each lag: does suspicious(T) predict retail(T+lag)?
    # Positive lag = suspicious leads retail
    # Negative lag = retail leads suspicious
    market_lags = {}
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        if lag > 0:
            x = susp_aligned[:-lag]
            y = ret_aligned[lag:]
        elif lag < 0:
            x = susp_aligned[-lag:]
            y = ret_aligned[:lag]
        else:
            x = susp_aligned
            y = ret_aligned

        if len(x) < 5 or x.std() == 0 or y.std() == 0:
            corr = 0.0
        else:
            corr, _ = stats.pearsonr(x, y)
            if np.isnan(corr):
                corr = 0.0
        market_lags[lag] = corr
        lag_correlations[lag].append(corr)

    # Find peak positive lag (where suspicious best predicts retail)
    positive_lags = {k: v for k, v in market_lags.items() if k > 0}
    peak_lag = max(positive_lags, key=positive_lags.get) if positive_lags else 0
    peak_corr = positive_lags.get(peak_lag, 0)

    market_results.append({
        "market": str(market),
        "lag_correlations": {str(k): float(v) for k, v in market_lags.items()},
        "peak_positive_lag": int(peak_lag),
        "peak_positive_corr": float(peak_corr),
        "n_bins": int(len(all_bins)),
        "susp_trades": int(len(susp)),
        "retail_trades": int(len(ret)),
    })

    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(markets)} markets...")

print(f"\nMarkets with both suspicious and retail trades: {markets_with_both:,}")
print(f"Markets included in analysis: {len(market_results):,}")


# ── 3. Aggregate lag-correlation profile ─────────────────────────────
section("3. Aggregate lag-correlation profile")

mean_corr = {}
for lag in range(-MAX_LAG, MAX_LAG + 1):
    vals = lag_correlations[lag]
    if vals:
        mean_corr[lag] = float(np.mean(vals))
    else:
        mean_corr[lag] = 0.0

print("\nMean cross-correlation (suspicious volume predicts retail volume):")
print(f"  Lag -4 (-60min, retail leads): {mean_corr.get(-4, 0):.4f}")
print(f"  Lag -3 (-45min):               {mean_corr.get(-3, 0):.4f}")
print(f"  Lag -2 (-30min):               {mean_corr.get(-2, 0):.4f}")
print(f"  Lag -1 (-15min):               {mean_corr.get(-1, 0):.4f}")
print(f"  Lag  0 (simultaneous):         {mean_corr.get( 0, 0):.4f}")
print(f"  Lag +1 (+15min, susp leads):   {mean_corr.get( 1, 0):.4f}")
print(f"  Lag +2 (+30min):               {mean_corr.get( 2, 0):.4f}")
print(f"  Lag +3 (+45min):               {mean_corr.get( 3, 0):.4f}")
print(f"  Lag +4 (+60min):               {mean_corr.get( 4, 0):.4f}")

# Peak-lag distribution across markets
peak_lag_counts = {}
for r in market_results:
    pl = r["peak_positive_lag"]
    peak_lag_counts[pl] = peak_lag_counts.get(pl, 0) + 1

print("\nDistribution of peak lead lag across markets (which lag has highest positive corr):")
for lag in range(1, MAX_LAG + 1):
    count = peak_lag_counts.get(lag, 0)
    pct = count / len(market_results) * 100 if market_results else 0
    print(f"  Lag +{lag} ({lag*BIN_MINUTES:2d}min): {count:3d} markets ({pct:.1f}%)")

# Strong-lead markets: peak corr > 0.3 and peak lag > 0
strong_lead_markets = [r for r in market_results if r["peak_positive_corr"] > 0.3]
print(f"\nMarkets with strong suspicious lead (peak corr > 0.3): {len(strong_lead_markets):,}")
if strong_lead_markets:
    strong_lead_markets_sorted = sorted(strong_lead_markets, key=lambda x: -x["peak_positive_corr"])
    print("  Top 5 markets by lead strength:")
    for r in strong_lead_markets_sorted[:5]:
        lag_min = r["peak_positive_lag"] * BIN_MINUTES
        print(f"    {r['market'][:20]}... lag={lag_min}min, corr={r['peak_positive_corr']:.3f}, "
              f"susp_trades={r['susp_trades']}, retail_trades={r['retail_trades']}")


# ── 4. Statistical test: is the positive lead signal significant? ─────
section("4. Statistical significance test")

# Compare mean correlation at lag+1 vs lag-1 (directional asymmetry)
lead_vals = lag_correlations[1]   # suspicious leads retail
follow_vals = lag_correlations[-1]  # retail leads suspicious

if len(lead_vals) > 1 and len(follow_vals) > 1:
    t_stat, p_val = stats.ttest_rel(lead_vals, follow_vals)
    print(f"\nPaired t-test: mean_corr(lag+1) vs mean_corr(lag-1)")
    print(f"  Lag+1 mean: {np.mean(lead_vals):.4f} ± {np.std(lead_vals):.4f}")
    print(f"  Lag-1 mean: {np.mean(follow_vals):.4f} ± {np.std(follow_vals):.4f}")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
    if p_val < 0.05 and np.mean(lead_vals) > np.mean(follow_vals):
        lead_significance = "SIGNIFICANT (p<0.05): suspicious wallets systematically lead retail"
    elif p_val < 0.05:
        lead_significance = "SIGNIFICANT (p<0.05): retail systematically leads suspicious"
    else:
        lead_significance = "NOT SIGNIFICANT (p>=0.05): no directional lead detected"
    print(f"  Interpretation: {lead_significance}")
else:
    t_stat, p_val = 0.0, 1.0
    lead_significance = "Insufficient data for significance test"

# Fraction of markets where suspicious leads at any positive lag
n_lead = sum(1 for r in market_results if r["peak_positive_corr"] > 0.1)
n_total = len(market_results)
pct_lead = n_lead / n_total * 100 if n_total > 0 else 0
print(f"\nFraction of markets with any suspicious lead signal (corr>0.1): "
      f"{n_lead}/{n_total} = {pct_lead:.1f}%")


# ── 5. Interpretation ─────────────────────────────────────────────────
section("5. Interpretation")

# Assess Mode 3 risk based on results
lead_mean = mean_corr.get(1, 0)
lag_mean = mean_corr.get(-1, 0)
directional_asymmetry = lead_mean - lag_mean

if directional_asymmetry > 0.05 and p_val < 0.05:
    mode3_risk = "HIGH"
    mode3_interpretation = (
        "Suspicious wallets systematically trade BEFORE retail wallets. "
        "This is consistent with active price manipulation: large traders "
        "move first, then retail follows as prices shift. "
        f"The directional asymmetry is {directional_asymmetry:.3f} "
        f"(positive = suspicious leads), significant at p={p_val:.3f}."
    )
elif directional_asymmetry > 0.02:
    mode3_risk = "MODERATE"
    mode3_interpretation = (
        "A weak tendency for suspicious wallets to trade slightly before retail, "
        f"but the effect is small (asymmetry={directional_asymmetry:.3f}) "
        f"and not statistically conclusive (p={p_val:.3f}). "
        "Could reflect information advantages rather than active manipulation."
    )
else:
    mode3_risk = "LOW"
    mode3_interpretation = (
        "No clear temporal leadership pattern: suspicious wallets do NOT "
        "systematically trade before retail wallets. "
        f"Directional asymmetry is only {directional_asymmetry:.3f} "
        f"(p={p_val:.3f}). "
        "This weakens the active manipulation hypothesis (Mode 3)."
    )

print(f"\n  Mode 3 (Manipulation) Risk: {mode3_risk}")
print(f"\n  {mode3_interpretation}")

print("\n  Reminder of what each mode means:")
print("  Mode 1 (Herding): traders copy each other → measured by DBSCAN outlier rate")
print("  Mode 2 (Minority price-setting): one group's volume dominates → measured by Diversity score")
print(f"  Mode 3 (Active manipulation): large trader moves first, retail follows → this analysis")


# ── 6. Save results ───────────────────────────────────────────────────
section("6. Saving results")

results = {
    "method": "Lead-lag cross-correlation (Granger-style)",
    "bin_minutes": BIN_MINUTES,
    "max_lag_steps": MAX_LAG,
    "markets_analyzed": len(market_results),
    "markets_with_data": markets_with_both,
    "aggregate_lag_correlations": {str(k): v for k, v in mean_corr.items()},
    "directional_asymmetry": float(directional_asymmetry),
    "lead_mean_corr": float(mean_corr.get(1, 0)),
    "follow_mean_corr": float(mean_corr.get(-1, 0)),
    "ttest_t": float(t_stat),
    "ttest_p": float(p_val),
    "statistical_result": lead_significance,
    "pct_markets_with_lead_signal": float(pct_lead),
    "mode3_risk": mode3_risk,
    "mode3_interpretation": mode3_interpretation,
    "top_markets": sorted(market_results, key=lambda x: -x["peak_positive_corr"])[:20],
}

with open(PROCESSED_DIR / "mode3_causality_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved: mode3_causality_results.json")

summary_lines = [
    "Mode 3 Temporal Causality Analysis Summary",
    "=" * 60,
    "",
    f"Method: Lead-lag cross-correlation ({BIN_MINUTES}-min bins, max lag {MAX_LAG} steps = {MAX_LAG*BIN_MINUTES} min)",
    f"Markets analyzed: {len(market_results):,}",
    "",
    "Aggregate lag-correlation profile (suspicious → retail):",
]
for lag in range(-MAX_LAG, MAX_LAG + 1):
    prefix = "susp leads →" if lag > 0 else ("retail leads →" if lag < 0 else "simultaneous ")
    summary_lines.append(f"  Lag {lag:+d} ({lag*BIN_MINUTES:+4d}min) [{prefix}]: {mean_corr.get(lag, 0):.4f}")

summary_lines += [
    "",
    f"Directional asymmetry (lead - follow): {directional_asymmetry:.4f}",
    f"t-test: t={t_stat:.3f}, p={p_val:.4f}",
    f"Statistical result: {lead_significance}",
    "",
    f"Mode 3 Risk: {mode3_risk}",
    f"Interpretation: {mode3_interpretation}",
]

with open(PROCESSED_DIR / "mode3_causality_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))
print("Saved: mode3_causality_summary.txt")

print("\nMode 3 analysis complete.")
