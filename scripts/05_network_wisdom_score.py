"""
Step 5: Network Graph & Wisdom Score Computation

Builds wallet interaction network from Polymarket trades and computes
composite Wisdom Score measuring crowd integrity:
  - Volume concentration (Gini coefficient)
  - Network centralization (Freeman's centrality)
  - Modularity (community structure)
  - DBSCAN clustering overlap with graph communities

Usage:
    poetry run python scripts/05_network_wisdom_score.py

Outputs:
    data/processed/wisdom_score_summary.json
    data/processed/network_stats.txt
"""

import json
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from itertools import combinations

PROCESSED_DIR = Path("data/processed")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def gini_coefficient(values):
    """Compute Gini coefficient (0 = perfect equality, 1 = perfect inequality)."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n


# ── 1. Load data ─────────────────────────────────────────────────────
section("1. Loading data")

df = pd.read_csv(PROCESSED_DIR / "poly_trades_all_matched.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

clusters = json.load(open(PROCESSED_DIR / "dbscan_clusters.json"))
cluster_map = {c["wallet"]: c["cluster"] for c in clusters}

print(f"Trades: {len(df):,}")
print(f"Unique wallets: {df['proxyWallet'].nunique():,}")
print(f"Cluster assignments loaded: {len(cluster_map):,}")


# ── 2. Build wallet interaction network ──────────────────────────────
section("2. Building wallet interaction network")

G = nx.Graph()

# Add nodes with cluster attributes
for wallet in df["proxyWallet"].unique():
    cluster_id = cluster_map.get(wallet, -1)
    G.add_node(wallet, cluster=cluster_id)

print(f"Nodes: {G.number_of_nodes()}")

# Add edges: wallets trading same side in same 1-hour window
# (indicates potential coordination or information leakage)
df["time_bin"] = df["timestamp"].dt.floor("1h")
edge_count = 0
for (time_bin, side), group in df.groupby(["time_bin", "side"]):
    wallets_in_bin = group["proxyWallet"].unique()
    if len(wallets_in_bin) < 2:
        continue
    for w1, w2 in combinations(wallets_in_bin, 2):
        if G.has_edge(w1, w2):
            G[w1][w2]["weight"] += 1
        else:
            G.add_edge(w1, w2, weight=1)
            edge_count += 1

print(f"Edges: {G.number_of_edges():,}")
print(f"Network density: {nx.density(G):.4f}")


# ── 3. Compute network metrics ───────────────────────────────────────
section("3. Computing network metrics")

# Volume concentration
wallet_volumes = df.groupby("proxyWallet")["size"].sum()
top5_share = wallet_volumes.nlargest(5).sum() / wallet_volumes.sum()
gini = gini_coefficient(wallet_volumes.values)
print(f"\nVolume concentration:")
print(f"  Top 5 wallets share: {top5_share:.1%}")
print(f"  Gini coefficient: {gini:.3f}")

# Network centralization
if G.number_of_nodes() > 0:
    dc = nx.degree_centrality(G)
    max_dc = max(dc.values()) if dc else 0
    centralization = (
        sum(max_dc - v for v in dc.values()) / ((len(G) - 1) * (len(G) - 2))
        if len(G) > 1
        else 0
    )
    print(f"\nNetwork centralization:")
    print(f"  Max degree centrality: {max_dc:.4f}")
    print(f"  Freeman centralization: {centralization:.4f}")
else:
    centralization = 0

# Community detection (Louvain)
if G.number_of_edges() > 0:
    try:
        communities = list(nx.community.louvain_communities(G, weight="weight", seed=42))
        modularity = nx.community.modularity(G, communities, weight="weight")
        print(f"\nCommunity structure:")
        print(f"  Communities: {len(communities)}")
        print(f"  Modularity: {modularity:.4f}")
        print(f"  Community sizes: {sorted([len(c) for c in communities], reverse=True)[:5]}")
    except:
        communities = []
        modularity = 0
        print("  Could not compute communities (insufficient network connectivity)")
else:
    communities = []
    modularity = 0


# ── 4. Cross-validate: DBSCAN clusters vs graph communities ──────────
section("4. Cross-validating clustering methods")

if communities:
    # For each graph community, check overlap with DBSCAN clusters
    for comm_idx, community in enumerate(communities[:5]):  # Check top 5
        community_clusters = [cluster_map.get(w, -1) for w in community]
        most_common_cluster = max(set(community_clusters), key=community_clusters.count)
        cluster_overlap = community_clusters.count(most_common_cluster) / len(community)
        print(f"  Community {comm_idx} ({len(community)} wallets): "
              f"{cluster_overlap:.1%} in DBSCAN cluster {most_common_cluster}")


# ── 5. Compute composite Wisdom Score ────────────────────────────────
section("5. Computing Wisdom Score")

# Wisdom Score formula (0 = whale manipulation, 100 = true crowd)
# Lower concentration, lower centralization, higher modularity = higher score
wisdom_score = 100 * (1 - (
    0.35 * top5_share +          # Volume concentration (0-1)
    0.35 * gini +                # Gini coefficient (0-1)
    0.20 * centralization +      # Network centralization (0-1)
    0.10 * (1 - modularity)      # Low modularity = coordination (flip it)
))

wisdom_score = max(0, min(100, wisdom_score))  # Clamp to [0, 100]

print(f"\nWisdom Score Components:")
print(f"  Volume concentration (top 5): {top5_share:.1%} (weight: 0.35)")
print(f"  Gini coefficient: {gini:.3f} (weight: 0.35)")
print(f"  Network centralization: {centralization:.4f} (weight: 0.20)")
print(f"  Modularity: {modularity:.3f} (weight: 0.10)")
print(f"\n  ↓")
print(f"  WISDOM SCORE: {wisdom_score:.1f} / 100")


# ── 6. Interpretation ────────────────────────────────────────────────
section("6. Interpretation")

if wisdom_score > 75:
    rating = "HIGH - Strong crowd wisdom, minimal whale manipulation risk"
elif wisdom_score > 50:
    rating = "MODERATE - Some concentration but diversified participation"
elif wisdom_score > 25:
    rating = "LOW - Significant concentration, whale influence likely"
else:
    rating = "CRITICAL - Extreme concentration, market likely whale-dominated"

print(f"\nRating: {rating}")


# ── 7. Save results ──────────────────────────────────────────────────
section("7. Saving results")

wisdom_data = {
    "wisdom_score": float(wisdom_score),
    "rating": rating,
    "metrics": {
        "top5_volume_share": float(top5_share),
        "gini_coefficient": float(gini),
        "network_centralization": float(centralization),
        "modularity": float(modularity),
    },
    "network": {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": float(nx.density(G)),
        "communities": len(communities),
    },
}

with open(PROCESSED_DIR / "wisdom_score_summary.json", "w") as f:
    json.dump(wisdom_data, f, indent=2)
print("Saved: wisdom_score_summary.json")

with open(PROCESSED_DIR / "network_stats.txt", "w") as f:
    f.write("Network Analysis Summary\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Wisdom Score: {wisdom_score:.1f} / 100\n")
    f.write(f"Rating: {rating}\n\n")
    f.write("Network Metrics:\n")
    f.write(f"  Nodes: {G.number_of_nodes():,}\n")
    f.write(f"  Edges: {G.number_of_edges():,}\n")
    f.write(f"  Density: {nx.density(G):.4f}\n")
    f.write(f"  Communities: {len(communities)}\n")
    f.write(f"  Modularity: {modularity:.4f}\n\n")
    f.write("Volume Concentration:\n")
    f.write(f"  Top 5 wallets: {top5_share:.1%}\n")
    f.write(f"  Gini coefficient: {gini:.3f}\n\n")
    f.write("Network Centralization:\n")
    f.write(f"  Freeman centralization: {centralization:.4f}\n")

print("Saved: network_stats.txt")

print("\nPhase 5 complete. Ready for Phase 6 (Cross-Platform Comparison).")
