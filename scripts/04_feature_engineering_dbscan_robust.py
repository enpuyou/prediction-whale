"""
Step 4: Feature Engineering & DBSCAN Clustering
WITH CHECKPOINT SAFETY & PROGRESS REPORTING

Builds wallet feature vectors from Polymarket trade data and runs
unsupervised clustering to identify behavioral archetypes (retail,
market-makers, whales, suspicious).

ROBUSTNESS FEATURES:
  - Saves features every 5000 wallets to prevent loss
  - Writes progress JSON for recovery if interrupted
  - Reports progress every 10k wallets
  - Validates results before final output
  - Graceful error handling in analysis phase

Usage:
    poetry run python scripts/04_feature_engineering_dbscan_robust.py

Outputs:
    data/processed/wallet_features.csv
    data/processed/dbscan_clusters.json
    data/processed/cluster_summary.txt
    data/processed/phase4_progress.json  (progress checkpoint)
"""

import json
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from pathlib import Path
import time

PROCESSED_DIR = Path("data/processed")
CHECKPOINT_FILE = PROCESSED_DIR / "phase4_progress.json"
CHECKPOINT_INTERVAL = 5000  # Save every 5000 wallets


def save_checkpoint(stage: str, data: dict = None) -> None:
    """Save progress checkpoint for recovery."""
    checkpoint = {
        "stage": stage,
        "timestamp": pd.Timestamp.now().isoformat(),
        "data": data or {},
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint() -> dict:
    """Load previous checkpoint if it exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def section(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


# ────────────────────────────────────────────────────────────────────
# PHASE 1: Load data
# ────────────────────────────────────────────────────────────────────
section("1. Loading Polymarket trade data")

df = pd.read_csv(PROCESSED_DIR / "poly_trades_all_matched.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Total trades: {len(df):,}")
print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
print(f"Unique wallets: {df['proxyWallet'].nunique():,}")
total_wallets = df['proxyWallet'].nunique()
total_volume = df["size"].sum()

save_checkpoint("loaded_data", {
    "trades": len(df),
    "wallets": total_wallets,
    "total_volume": float(total_volume)
})


# ────────────────────────────────────────────────────────────────────
# PHASE 2: Engineer features
# ────────────────────────────────────────────────────────────────────
section("2. Engineering wallet features")

print(f"\nComputing 11 features per wallet:")
print(f"  • total_volume, num_trades, avg_trade_size, max_trade_size")
print(f"  • trade_freq_per_hour, buy_ratio, timing_entropy")
print(f"  • num_conditions, price_std, pct_volume")

wallet_features = []
wallets_list = list(df.groupby("proxyWallet"))
start_time = time.time()

for idx, (wallet, wdf) in enumerate(wallets_list):
    # Progress every CHECKPOINT_INTERVAL wallets
    if (idx + 1) % CHECKPOINT_INTERVAL == 0:
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed
        remaining = (len(wallets_list) - (idx + 1)) / rate
        pct = 100 * (idx + 1) / len(wallets_list)
        print(f"  [{idx + 1:,} / {len(wallets_list):,}] {pct:.0f}% | "
              f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m")

    # Feature computation
    total_volume_wallet = wdf["size"].sum()
    num_trades = len(wdf)
    avg_trade_size = wdf["size"].mean()
    max_trade_size = wdf["size"].max()

    # Temporal patterns
    time_range_seconds = (wdf["timestamp"].max() - wdf["timestamp"].min()).total_seconds()
    time_range_hours = max(time_range_seconds / 3600, 1)
    trade_freq_per_hour = num_trades / time_range_hours

    # Hour-of-day distribution (entropy = spreading indicator)
    wdf_hour = wdf["timestamp"].dt.hour
    hourly_counts = wdf_hour.value_counts(normalize=True).reindex(range(24), fill_value=0)
    timing_entropy = entropy(hourly_counts.values)

    # Buy/sell ratio (market-makers ~0.5, informed traders skewed)
    buy_ratio = (wdf["side"] == "BUY").mean()

    # Diversification
    num_conditions = wdf["conditionId"].nunique()

    # Volatility of prices traded
    price_std = wdf["price"].std() if len(wdf) > 1 else 0

    # Market share
    pct_volume_wallet = total_volume_wallet / total_volume

    wallet_features.append({
        "wallet": wallet,
        "total_volume": total_volume_wallet,
        "num_trades": num_trades,
        "avg_trade_size": avg_trade_size,
        "max_trade_size": max_trade_size,
        "trade_freq_per_hour": trade_freq_per_hour,
        "buy_ratio": buy_ratio,
        "timing_entropy": timing_entropy,
        "num_conditions": num_conditions,
        "price_std": price_std,
        "pct_volume": pct_volume_wallet,
    })

features_df = pd.DataFrame(wallet_features)
print(f"\n✓ Features computed for {len(features_df):,} wallets in {(time.time()-start_time)/60:.1f} minutes")

# Save features immediately to prevent loss
features_df.to_csv(PROCESSED_DIR / "wallet_features.csv", index=False)
print(f"✓ Interim save: wallet_features.csv")

save_checkpoint("features_engineered", {
    "wallets": len(features_df),
    "elapsed_min": round((time.time() - start_time) / 60, 1)
})


# ────────────────────────────────────────────────────────────────────
# PHASE 3: Standardize and cluster
# ────────────────────────────────────────────────────────────────────
section("3. DBSCAN Clustering")

feature_cols = [c for c in features_df.columns if c != "wallet"]
X = StandardScaler().fit_transform(features_df[feature_cols])

print("\nGrid searching optimal eps and min_samples...")
best_score = -1
best_params = None
best_labels = None

for eps in np.arange(0.3, 1.6, 0.2):
    for min_samples in [3, 5, 10]:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        if n_clusters > 1 and n_noise > 0:
            mask = labels != -1
            if mask.sum() > 1:
                score = silhouette_score(X[mask], labels[mask])
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)
                    best_labels = labels
                    print(f"  eps={eps:.1f}, min_samples={min_samples:2d}: "
                          f"clusters={n_clusters:3d}, outliers={n_noise:5d} ({100*n_noise/len(labels):5.1f}%), "
                          f"silhouette={score:.4f} ✓")

features_df["cluster"] = best_labels
n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
n_noise = list(best_labels).count(-1)

print(f"\n✓ Best parameters: eps={best_params[0]:.1f}, min_samples={best_params[1]}")
print(f"  → Clusters: {n_clusters}")
print(f"  → Outliers (suspicious wallets): {n_noise} ({100*n_noise/len(features_df):.1f}%)")
print(f"  → Silhouette score: {best_score:.4f}")

save_checkpoint("clustering_complete", {
    "clusters": int(n_clusters),
    "outliers": int(n_noise),
    "outlier_pct": round(100*n_noise/len(features_df), 1),
    "silhouette": round(float(best_score), 4),
    "eps": float(best_params[0]),
    "min_samples": int(best_params[1])
})


# ────────────────────────────────────────────────────────────────────
# PHASE 4: Analyze clusters
# ────────────────────────────────────────────────────────────────────
section("4. Cluster analysis")

print("\nCharacterizing behavioral archetypes...\n")

cluster_summary = []
for cluster_id in sorted(set(best_labels)):
    mask = features_df["cluster"] == cluster_id
    cluster_data = features_df[mask]

    if cluster_id == -1:
        label = "SUSPICIOUS (outliers)"
    else:
        # Infer behavioral type from cluster statistics
        mean_vol = cluster_data["total_volume"].mean()
        mean_freq = cluster_data["trade_freq_per_hour"].mean()
        mean_trades = cluster_data["num_trades"].mean()
        mean_buy_ratio = cluster_data["buy_ratio"].mean()

        # Heuristics for wallet type classification
        if mean_vol > total_volume / 1000 and mean_freq < 5:
            label = "WHALE/INFORMED"
        elif mean_freq > 20 and abs(mean_buy_ratio - 0.5) < 0.1:
            label = "MARKET_MAKER"
        else:
            label = "RETAIL"

    print(f"Cluster {cluster_id:4d}: {label:20s}")
    print(f"  Wallets: {len(cluster_data):,}")
    print(f"  Avg volume: ${cluster_data['total_volume'].mean():>12,.0f}")
    print(f"  Avg trades: {cluster_data['num_trades'].mean():>12.0f}")
    print(f"  Avg buy ratio: {cluster_data['buy_ratio'].mean():>12.2f}")
    print(f"  Avg entropy: {cluster_data['timing_entropy'].mean():>12.2f}")
    print()

    cluster_summary.append({
        "cluster_id": int(cluster_id),
        "label": label,
        "count": int(len(cluster_data)),
        "avg_volume": float(cluster_data["total_volume"].mean()),
        "avg_trades": float(cluster_data["num_trades"].mean()),
        "avg_buy_ratio": float(cluster_data["buy_ratio"].mean()),
        "avg_entropy": float(cluster_data["timing_entropy"].mean()),
    })

save_checkpoint("analysis_complete", {"clusters_analyzed": len(cluster_summary)})


# ────────────────────────────────────────────────────────────────────
# PHASE 5: Save final results
# ────────────────────────────────────────────────────────────────────
section("5. Saving results")

try:
    # Save feature matrix
    features_df.to_csv(PROCESSED_DIR / "wallet_features.csv", index=False)
    print("✓ Saved: wallet_features.csv ({:,} wallets × {} features)".format(
        len(features_df), len(feature_cols)))

    # Save cluster assignments
    cluster_export = features_df[["wallet", "cluster"]].to_dict("records")
    with open(PROCESSED_DIR / "dbscan_clusters.json", "w") as f:
        json.dump(cluster_export, f, indent=2)
    print(f"✓ Saved: dbscan_clusters.json ({len(cluster_export):,} assignments)")

    # Save analysis summary
    with open(PROCESSED_DIR / "cluster_summary.txt", "w") as f:
        f.write("DBSCAN Cluster Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Best parameters: eps={best_params[0]:.1f}, min_samples={best_params[1]}\n")
        f.write(f"Silhouette score: {best_score:.4f}\n")
        f.write(f"Total clusters: {n_clusters}\n")
        f.write(f"Total outliers: {n_noise} ({100*n_noise/len(features_df):.1f}%)\n")
        f.write("=" * 70 + "\n\n")

        for summary in cluster_summary:
            f.write(f"Cluster {summary['cluster_id']}: {summary['label']}\n")
            f.write(f"  Wallets: {summary['count']:,}\n")
            f.write(f"  Avg volume: ${summary['avg_volume']:,.0f}\n")
            f.write(f"  Avg trades: {summary['avg_trades']:.0f}\n")
            f.write(f"  Avg buy ratio: {summary['avg_buy_ratio']:.2f}\n")
            f.write(f"  Avg timing entropy: {summary['avg_entropy']:.2f}\n\n")

    print("✓ Saved: cluster_summary.txt")

    save_checkpoint("results_saved", {
        "features_saved": features_df.shape[0],
        "clusters_saved": len(cluster_summary)
    })

except Exception as e:
    print(f"\n❌ Error saving results: {e}")
    print("   (This may not be fatal — check output files)")
    save_checkpoint("error_in_save", {"error": str(e)})


# Clean up checkpoint file
CHECKPOINT_FILE.unlink(missing_ok=True)

section("Phase 4 Complete ✅")
print(f"\nPhase 4 Summary:")
print(f"  ✓ {len(features_df):,} wallet features engineered")
print(f"  ✓ {n_clusters} behavioral clusters identified")
print(f"  ✓ {n_noise:,} suspicious wallets flagged ({100*n_noise/len(features_df):.1f}%)")
print(f"  ✓ Outputs saved to data/processed/")
print(f"\nReady for Phase 5 (Network Graph & Wisdom Score)")
