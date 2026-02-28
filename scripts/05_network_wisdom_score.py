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
print("\nVolume concentration:")
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
    print("\nNetwork centralization:")
    print(f"  Max degree centrality: {max_dc:.4f}")
    print(f"  Freeman centralization: {centralization:.4f}")
else:
    centralization = 0

# Community detection (Louvain)
if G.number_of_edges() > 0:
    try:
        communities = list(nx.community.louvain_communities(G, weight="weight", seed=42))
        modularity = nx.community.modularity(G, communities, weight="weight")
        print("\nCommunity structure:")
        print(f"  Communities: {len(communities)}")
        print(f"  Modularity: {modularity:.4f}")
        print(f"  Community sizes: {sorted([len(c) for c in communities], reverse=True)[:5]}")
    except Exception:
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


# ── 5. Compute Wisdom of Crowds Mechanism Score ───────────────────────
section("5. Computing Wisdom of Crowds Mechanism Score")

# Surowiecki's four crowd wisdom conditions, each scored 0–100:
#
# Diversity Score: fraction of volume NOT controlled by behaviorally-suspicious
#   wallets (DBSCAN cluster == -1, i.e. outliers/anomalous traders).
#   Unlike top-N filtering (which conflates size with anomaly), this
#   asks: "Does the behaviorally-anomalous group dominate price-setting?"
#   High score = volume spread across many behaviorally-normal traders.
features_df = pd.read_csv(PROCESSED_DIR / "wallet_features.csv")
suspicious_wallets = set(
    features_df.loc[features_df["cluster"] == -1, "wallet"]
)
suspicious_volume = wallet_volumes[wallet_volumes.index.isin(suspicious_wallets)].sum()
total_volume = wallet_volumes.sum()
suspicious_volume_share = suspicious_volume / total_volume if total_volume > 0 else 0
top10_share = wallet_volumes.nlargest(10).sum() / total_volume  # kept for reference
diversity_score = max(0, min(100, (1 - suspicious_volume_share) * 100))

# Independence Score: inverse of DBSCAN coordination signal
#   High outlier rate (cluster == -1) = low independence
outlier_wallets = sum(1 for v in cluster_map.values() if v == -1)
total_wallets = len(cluster_map)
outlier_rate = outlier_wallets / total_wallets if total_wallets > 0 else 0
independence_score = max(0, min(100, (1 - outlier_rate) * 100))

# Decentralization Score: inverse of network centralization
#   Low Freeman centralization = no dominant hub = high score
decentralization_score = max(0, min(100, (1 - centralization) * 100))

# Aggregation Score: how well does price track public information?
#   Approximated by market modularity — high community structure
#   suggests price emerges from diverse independent groups (placeholder
#   until Google Trends correlation is available)
aggregation_score = max(0, min(100, modularity * 100))

# Composite: equally-weighted average of all four
wisdom_score = (diversity_score + independence_score + decentralization_score + aggregation_score) / 4

print("\nWisdom of Crowds Mechanism Score — Sub-scores:")
print(f"  Diversity Score     : {diversity_score:.1f} / 100  (volume outside behaviorally-suspicious group; suspicious share = {suspicious_volume_share:.1%})")
print(f"  Independence Score  : {independence_score:.1f} / 100  (low DBSCAN coordination signal)")
print(f"  Decentralization    : {decentralization_score:.1f} / 100  (no dominant hub in network)")
print(f"  Aggregation Score   : {aggregation_score:.1f} / 100  (community modularity proxy)")
print("\n  ↓")
print(f"  WISDOM OF CROWDS MECHANISM SCORE: {wisdom_score:.1f} / 100")


# ── 6. Interpretation ────────────────────────────────────────────────
section("6. Interpretation")

if wisdom_score > 70:
    signal_label = "Crowd Wisdom Signal"
    signal_description = (
        "Price reflects the aggregated beliefs of a diverse, independent group. "
        "Safe to cite as crowd consensus."
    )
elif wisdom_score >= 40:
    signal_label = "Expert Opinion Signal"
    signal_description = (
        "Price is influenced by a smaller set of sophisticated or high-volume traders. "
        "Reflects informed opinion, not broad public consensus."
    )
else:
    signal_label = "Concentrated Capital Signal"
    signal_description = (
        "Price is driven by a small number of large wallets. "
        "Does not represent crowd consensus — likely reflects concentrated positions."
    )

rating = signal_label

print(f"\n  Signal Type : {signal_label}")
print(f"  Description : {signal_description}")


# ── 7. Save results ──────────────────────────────────────────────────
section("7. Saving results")

wisdom_data = {
    "wisdom_score": float(wisdom_score),
    "signal_label": signal_label,
    "rating": rating,
    "sub_scores": {
        "diversity_score": float(diversity_score),
        "independence_score": float(independence_score),
        "decentralization_score": float(decentralization_score),
        "aggregation_score": float(aggregation_score),
    },
    "metrics": {
        "top5_volume_share": float(top5_share),
        "top10_volume_share": float(top10_share),
        "suspicious_volume_share": float(suspicious_volume_share),
        "gini_coefficient": float(gini),
        "network_centralization": float(centralization),
        "modularity": float(modularity),
        "outlier_rate": float(outlier_rate),
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
    f.write(f"Wisdom of Crowds Mechanism Score: {wisdom_score:.1f} / 100\n")
    f.write(f"Signal Label: {signal_label}\n\n")
    f.write("Sub-Scores (Surowiecki Conditions):\n")
    f.write(f"  Diversity Score    : {diversity_score:.1f} / 100\n")
    f.write(f"  Independence Score : {independence_score:.1f} / 100\n")
    f.write(f"  Decentralization   : {decentralization_score:.1f} / 100\n")
    f.write(f"  Aggregation Score  : {aggregation_score:.1f} / 100\n\n")
    f.write("Network Metrics:\n")
    f.write(f"  Nodes: {G.number_of_nodes():,}\n")
    f.write(f"  Edges: {G.number_of_edges():,}\n")
    f.write(f"  Density: {nx.density(G):.4f}\n")
    f.write(f"  Communities: {len(communities)}\n")
    f.write(f"  Modularity: {modularity:.4f}\n\n")
    f.write("Volume Concentration:\n")
    f.write(f"  Top 5 wallets: {top5_share:.1%}\n")
    f.write(f"  Top 10 wallets: {top10_share:.1%}\n")
    f.write(f"  Suspicious-group (DBSCAN cluster -1): {suspicious_volume_share:.1%}\n")
    f.write(f"  Gini coefficient: {gini:.3f}\n\n")
    f.write("Network Centralization:\n")
    f.write(f"  Freeman centralization: {centralization:.4f}\n")

print("Saved: network_stats.txt")


# ── 8. Marketing Recommendation Card ─────────────────────────────────
section("8. Campaign Timing Recommendation")


def generate_marketing_recommendation(
    market_name: str,
    wisdom_score: float,
    signal_label: str,
    dominant_wallet_type: str,
    days_until_resolution: int,
    current_probability: float | None = None,
) -> None:
    """Print a plain-English marketing recommendation card.

    Parameters
    ----------
    market_name : str
        Human-readable name of the prediction market.
    wisdom_score : float
        Composite Wisdom of Crowds Mechanism Score (0–100).
    signal_label : str
        One of 'Crowd Wisdom Signal', 'Expert Opinion Signal',
        'Concentrated Capital Signal'.
    dominant_wallet_type : str
        e.g. 'Retail', 'Whale', 'Market Maker', 'Suspicious'.
    days_until_resolution : int
        Days remaining until the market resolves.
    current_probability : float | None
        Current YES probability (0–1). Used for citation copy.
    """
    # ── Signal type explanation ───────────────────────────────────────
    if signal_label == "Crowd Wisdom Signal":
        signal_explanation = (
            "This market aggregates the beliefs of a large, diverse, and "
            "independent group of traders — the conditions Surowiecki identifies "
            "as necessary for a crowd to be smarter than any individual."
        )
    elif signal_label == "Expert Opinion Signal":
        signal_explanation = (
            "This market is driven by a smaller set of sophisticated or "
            "high-volume traders. The number reflects informed opinion "
            "rather than broad public consensus."
        )
    else:
        signal_explanation = (
            "This market is dominated by a small number of large capital "
            "positions. The probability reflects concentrated bets, not "
            "the aggregated judgment of a diverse crowd."
        )

    # ── Citation guidance ─────────────────────────────────────────────
    prob_str = f"{current_probability*100:.0f}%" if current_probability is not None else "X%"
    if wisdom_score >= 70:
        citation = (
            f'Suggested copy: "Prediction markets give {market_name} '
            f'a {prob_str} probability."'
        )
    elif wisdom_score >= 40:
        citation = (
            f'Suggested copy: "Sophisticated traders currently price '
            f'{market_name} at {prob_str}."'
        )
    else:
        citation = (
            "Do not cite this number directly — it reflects concentrated "
            "capital positions, not crowd consensus. Reference the market "
            "as an indicator of large-trader sentiment only."
        )

    # ── Campaign timing action ────────────────────────────────────────
    if wisdom_score >= 70 and days_until_resolution > 7:
        action = "PROCEED"
        rationale = (
            f"High crowd wisdom signal ({wisdom_score:.0f}/100) with "
            f"{days_until_resolution}d until resolution — safe window "
            "to build campaign around this probability."
        )
    elif wisdom_score >= 40 or days_until_resolution > 14:
        action = "MONITOR"
        rationale = (
            f"Score of {wisdom_score:.0f}/100 ({signal_label}) and "
            f"{days_until_resolution}d to resolution. Check again in "
            "3–5 days; cite with appropriate qualification."
        )
    else:
        action = "HOLD"
        rationale = (
            f"Score of {wisdom_score:.0f}/100 ({signal_label}) driven "
            f"by {dominant_wallet_type} wallets. Citing this number "
            "risks misleading the audience — wait for broader participation "
            "or use a different signal."
        )

    # ── Print card ────────────────────────────────────────────────────
    width = 70
    print("\n" + "╔" + "═" * (width - 2) + "╗")
    print("║" + "  CAMPAIGN TIMING RECOMMENDATION CARD".center(width - 2) + "║")
    print("║" + f"  {market_name}".center(width - 2) + "║")
    print("╠" + "═" * (width - 2) + "╣")

    print("║" + "  1. SIGNAL TYPE".ljust(width - 2) + "║")
    print("║" + f"     {signal_label}".ljust(width - 2) + "║")
    for line in [signal_explanation[i:i+62] for i in range(0, len(signal_explanation), 62)]:
        print("║" + f"     {line}".ljust(width - 2) + "║")
    print("║" + "".ljust(width - 2) + "║")

    print("║" + "  2. CITATION GUIDANCE".ljust(width - 2) + "║")
    for line in [citation[i:i+62] for i in range(0, len(citation), 62)]:
        print("║" + f"     {line}".ljust(width - 2) + "║")
    print("║" + "".ljust(width - 2) + "║")

    print("║" + "  3. CAMPAIGN TIMING ACTION".ljust(width - 2) + "║")
    print("║" + f"     ▶  {action}".ljust(width - 2) + "║")
    for line in [rationale[i:i+62] for i in range(0, len(rationale), 62)]:
        print("║" + f"     {line}".ljust(width - 2) + "║")

    print("║" + "".ljust(width - 2) + "║")
    print("║" + f"  Wisdom Score: {wisdom_score:.1f}/100   Dominant Type: {dominant_wallet_type}   Days left: {days_until_resolution}".ljust(width - 2) + "║")
    print("╚" + "═" * (width - 2) + "╝")


# Determine dominant wallet type from cluster distribution
cluster_counts_raw = {}
for v in cluster_map.values():
    cluster_counts_raw[v] = cluster_counts_raw.get(v, 0) + 1

outlier_count = cluster_counts_raw.get(-1, 0)
non_outlier_total = sum(v for k, v in cluster_counts_raw.items() if k != -1)
outlier_pct = outlier_count / total_wallets if total_wallets > 0 else 0

if outlier_pct > 0.15:
    dominant_type = "Suspicious/Coordinated"
elif top5_share > 0.30:
    dominant_type = "Whale"
else:
    dominant_type = "Retail"

generate_marketing_recommendation(
    market_name="Polymarket Multi-Market Analysis",
    wisdom_score=wisdom_score,
    signal_label=signal_label,
    dominant_wallet_type=dominant_type,
    days_until_resolution=30,   # placeholder — update per market
    current_probability=None,
)

print("\nPhase 5 complete. Ready for Phase 6 (Cross-Platform Comparison).")
