"""
Step 4: Feature Engineering & DBSCAN Clustering

Builds wallet feature vectors from Polymarket trade data and runs
unsupervised clustering to identify behavioral archetypes (retail,
market-makers, whales, suspicious).

Usage:
    poetry run python scripts/04_feature_engineering_dbscan.py

Outputs:
    data/processed/wallet_features.csv
    data/processed/dbscan_clusters.json
    data/processed/cluster_summary.txt
"""

import json
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── 1. Load and prepare trade data ───────────────────────────────────
section("1. Loading Polymarket trade data")

df = pd.read_csv(PROCESSED_DIR / "poly_trades_all_matched.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Total trades: {len(df):,}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Unique wallets: {df['proxyWallet'].nunique():,}")


# ── 2. Engineer wallet features ──────────────────────────────────────
section("2. Engineering wallet features")

wallet_features = []

for wallet, wdf in df.groupby("proxyWallet"):
    # Basic activity
    total_volume = wdf["size"].sum()
    num_trades = len(wdf)
    avg_trade_size = wdf["size"].mean()

    # Temporal patterns
    time_range_seconds = (wdf["timestamp"].max() - wdf["timestamp"].min()).total_seconds()
    time_range_hours = max(time_range_seconds / 3600, 1)
    trade_freq_per_hour = num_trades / time_range_hours

    # Hour-of-day distribution
    wdf_copy = wdf.copy()
    wdf_copy["hour"] = wdf_copy["timestamp"].dt.hour
    hourly_dist = wdf_copy["hour"].value_counts(normalize=True).reindex(range(24), fill_value=0)
    timing_entropy = entropy(hourly_dist.values)

    # Buy/sell ratio
    buy_ratio = (wdf["side"] == "BUY").mean()

    # Markets active in
    num_conditions = wdf["conditionId"].nunique()

    # Max trade size (whale indicator)
    max_trade_size = wdf["size"].max()

    # Price variance (volatile traders signal)
    price_std = wdf["price"].std() if len(wdf) > 1 else 0

    wallet_features.append({
        "wallet": wallet,
        "total_volume": total_volume,
        "num_trades": num_trades,
        "avg_trade_size": avg_trade_size,
        "max_trade_size": max_trade_size,
        "trade_freq_per_hour": trade_freq_per_hour,
        "buy_ratio": buy_ratio,
        "timing_entropy": timing_entropy,
        "num_conditions": num_conditions,
        "price_std": price_std,
        "pct_volume": total_volume / df["size"].sum(),
    })

features_df = pd.DataFrame(wallet_features)
print(f"\nFeatures computed for {len(features_df)} wallets")
print(f"\nFeature summary:")
print(features_df[["total_volume", "num_trades", "avg_trade_size", "buy_ratio"]].describe())


# ── 3. Run DBSCAN with parameter search ──────────────────────────────
section("3. Running DBSCAN clustering")

feature_cols = [c for c in features_df.columns if c != "wallet"]
X = StandardScaler().fit_transform(features_df[feature_cols])

best_score = -1
best_params = None
best_labels = None

print("\nGrid searching eps and min_samples...")
for eps in np.arange(0.3, 1.6, 0.2):
    for min_samples in [3, 5, 10]:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        if n_clusters > 1 and n_noise > 0:
            # Only score clusters with actual outliers
            mask = labels != -1
            if mask.sum() > 1:
                score = silhouette_score(X[mask], labels[mask])
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)
                    best_labels = labels
                    print(f"  eps={eps:.1f}, min_samples={min_samples}: "
                          f"clusters={n_clusters}, noise={n_noise}, score={score:.3f} ✓")

print(f"\nBest parameters: eps={best_params[0]:.1f}, min_samples={best_params[1]}")

features_df["cluster"] = best_labels


# ── 4. Analyze clusters ──────────────────────────────────────────────
section("4. Cluster analysis")

cluster_summary = []
for cluster_id in sorted(set(best_labels)):
    mask = features_df["cluster"] == cluster_id
    cluster_data = features_df[mask]

    if cluster_id == -1:
        label = "SUSPICIOUS (outliers)"
    else:
        # Infer label from cluster centroids
        mean_vol = cluster_data["total_volume"].mean()
        mean_freq = cluster_data["trade_freq_per_hour"].mean()
        mean_trades = cluster_data["num_trades"].mean()
        if mean_vol > df["size"].sum() / 1000 and mean_freq < 5:
            label = "WHALE/INFORMED"
        elif mean_freq > 20 and cluster_data["buy_ratio"].mean().abs() - 0.5 < 0.1:
            label = "MARKET_MAKER"
        else:
            label = "RETAIL"

    print(f"\nCluster {cluster_id}: {label}")
    print(f"  Count: {len(cluster_data)}")
    print(f"  Avg volume: ${cluster_data['total_volume'].mean():,.0f}")
    print(f"  Avg trades: {cluster_data['num_trades'].mean():.0f}")
    print(f"  Avg buy_ratio: {cluster_data['buy_ratio'].mean():.2f}")
    print(f"  Avg timing_entropy: {cluster_data['timing_entropy'].mean():.2f}")

    cluster_summary.append({
        "cluster_id": cluster_id,
        "label": label,
        "count": len(cluster_data),
        "avg_volume": cluster_data["total_volume"].mean(),
        "avg_trades": cluster_data["num_trades"].mean(),
        "avg_buy_ratio": cluster_data["buy_ratio"].mean(),
    })


# ── 5. Save results ──────────────────────────────────────────────────
section("5. Saving results")

# Save full feature matrix
features_df.to_csv(PROCESSED_DIR / "wallet_features.csv", index=False)
print("Saved: wallet_features.csv")

# Save cluster assignments
cluster_export = features_df[["wallet", "cluster"]].to_dict("records")
with open(PROCESSED_DIR / "dbscan_clusters.json", "w") as f:
    json.dump(cluster_export, f, indent=2)
print("Saved: dbscan_clusters.json")

# Save summary
with open(PROCESSED_DIR / "cluster_summary.txt", "w") as f:
    f.write("DBSCAN Cluster Summary\n")
    f.write("=" * 60 + "\n\n")
    for summary in cluster_summary:
        f.write(f"Cluster {summary['cluster_id']}: {summary['label']}\n")
        f.write(f"  Wallets: {summary['count']}\n")
        f.write(f"  Avg volume: ${summary['avg_volume']:,.0f}\n")
        f.write(f"  Avg trades: {summary['avg_trades']:.0f}\n")
        f.write(f"  Avg buy ratio: {summary['avg_buy_ratio']:.2f}\n\n")

print("Saved: cluster_summary.txt")

print("\nPhase 4 complete. Ready for Phase 5 (Network Graph & Wisdom Score).")
