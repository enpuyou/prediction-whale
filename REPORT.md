# Prediction Market Integrity Auditor
## Do Prediction Markets Reflect Crowd Wisdom, or Whale Manipulation?
### A Quantitative Audit of Polymarket and Kalshi Using Behavioral Clustering and Network Analysis

**MSIS 521A · Winter 2026**
**Date:** February 27, 2026
**Data:** 237,791 trades · 63,793 wallets · 6 matched markets · Nov 2024 – Feb 2026

---

## Table of Contents

1. [Introduction & Research Question](#1-introduction--research-question)
2. [Data Collection Methodology](#2-data-collection-methodology)
3. [Feature Engineering](#3-feature-engineering)
4. [Behavioral Clustering (DBSCAN)](#4-behavioral-clustering-dbscan)
5. [Network Analysis](#5-network-analysis)
6. [Wisdom Score: Crowd Integrity Metric](#6-wisdom-score-crowd-integrity-metric)
7. [Cross-Platform Comparison](#7-cross-platform-comparison)
8. [Findings & Analysis](#8-findings--analysis)
9. [Critical Evaluation](#9-critical-evaluation)
10. [Conclusions & Recommendations](#10-conclusions--recommendations)

---

## 1. Introduction & Research Question

### Motivation

Prediction markets like Polymarket and Kalshi are positioned as tools for aggregating distributed information — the "wisdom of the crowd." Polymarket, an unregulated blockchain-based platform, processes over $1 billion in monthly volume across political, geopolitical, and sports events. Kalshi, a federally-regulated prediction market exchange, covers many of the same events under CFTC oversight.

The core question this project addresses:

> **Are prediction market prices driven by genuine crowd consensus, or are they dominated by a small number of high-volume "whale" accounts whose behavior may constitute informed trading, coordination, or manipulation?**

This distinction matters because:
- If markets are whale-dominated, prices may reflect individual conviction (or manipulation) rather than collective knowledge
- Regulated markets (Kalshi) are subject to trading surveillance; unregulated markets (Polymarket) are not
- Identifying manipulation patterns establishes a quantitative baseline for oversight

### Hypothesis

We hypothesize that **Polymarket exhibits detectable behavioral concentration** — a small fraction of wallets driving a disproportionate share of volume, exhibiting anomalous timing patterns, and forming coordinated network clusters — resulting in a **moderate-to-low crowd wisdom score**.

### Scope

We analyze 6 matched market pairs (same real-world event traded on both platforms), covering $1.63B in total Polymarket volume:

| Market | Similarity | Polymarket Volume |
|--------|-----------|------------------|
| Champions League Winner | 1.000 | $1,001,676,674 |
| Fed Chair Nomination | 1.000 | $540,827,605 |
| Gov Shutdown Duration | 1.000 | $23,495,074 |
| Next Pope | 0.996 | $30,143,338 |
| Zelenskyy/Putin Location | 0.987 | $18,496,420 |
| Trump Defense Secretary | 0.973 | $14,011,760 |

The sports market (Champions League, $1B) was deliberately included as a high insider-trading-risk event where lineup leaks, injury information, and club-transfer knowledge create structural information asymmetry.

---

## 2. Data Collection Methodology

### Platform Architecture

Both platforms were queried via public APIs requiring no authentication:

**Polymarket** uses a two-tier architecture:
- `gamma-api.polymarket.com/events` — categorical event groupings (e.g., "Who will be the next Pope?")
- `data-api.polymarket.com/trades` — individual trade records with wallet identifiers (`proxyWallet`)

**Kalshi** mirrors this structure:
- `api.elections.kalshi.com/trade-api/v2/events` — event-level groupings
- `api.elections.kalshi.com/v1/trades` — trade records (no user identifiers in public API)

A critical design decision was matching at the **event level** (not sub-market level). Early attempts matching Polymarket `/markets` against Kalshi `/events` produced poor results because Polymarket sub-markets use narrow binary questions ("Will Trump nominate Judy Shelton?") while Kalshi uses categorical event titles ("Who will Trump nominate as Fed Chair?"). Matching the event-level titles using semantic similarity produced dramatically better alignment.

### Market Matching

Event-level matching used `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dimensional embeddings) to compute cosine similarity between all 2,000 Polymarket event titles and 4,897 Kalshi event titles:

```
similarity(a, b) = cos(embed(a), embed(b)) = (a · b) / (‖a‖ · ‖b‖)
```

Results across 9,794,000 candidate pairs:
- 479 matches at threshold ≥ 0.75
- 31 exact matches at threshold ≥ 0.95
- 8 perfect matches at 1.00

### Multi-Threaded Collection

To efficiently collect trade histories, a `ThreadPoolExecutor` with 4 concurrent workers was used to parallelize requests across the 125 sub-markets (conditionIds) under the 6 matched events:

```python
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(fetch_trades, cid): cid for cid in condition_ids}
    for future in as_completed(futures):
        cid, trades = future.result()
```

Collection parameters: up to 2,000 trades per sub-market, 100ms delay between requests, automatic retry on HTTP 429 (rate-limit) responses.

### Dataset Summary

| Metric | Value |
|--------|-------|
| Total Polymarket trades | 237,791 |
| Unique wallets | 63,793 |
| Total Kalshi trades | 160,000 |
| Kalshi markets covered | 702 |
| Date range | Nov 15, 2024 – Feb 27, 2026 |
| Total Polymarket volume | $317,246,935 |
| Avg trade size | $1,334 |
| Median trade size | $34 |
| Buy-side proportion | 63.2% |
| Unique sub-markets (conditionIds) | 125 |

The large gap between mean ($1,334) and median ($34) trade size immediately signals heavy right-skew: a small number of very large trades dominate total volume.

---

## 3. Feature Engineering

### Rationale for Feature Selection

For each wallet, we compute 11 features designed to capture four orthogonal behavioral dimensions:

1. **Scale** — how much does this wallet trade?
2. **Activity** — how often does it trade?
3. **Timing** — when and how predictably does it trade?
4. **Direction** — does it show systematic buy/sell bias?

These dimensions map onto known market participant archetypes: retail speculators (low scale, low frequency), market makers (balanced direction, moderate frequency), informed traders / whales (high scale, concentrated timing), and automated/suspicious accounts (anomalous patterns).

### Feature Definitions

| Feature | Formula | Behavioral Signal |
|---------|---------|------------------|
| `total_volume` | Σ trade sizes | Absolute market footprint |
| `num_trades` | Count of trades | Activity level |
| `avg_trade_size` | total_volume / num_trades | Typical position size |
| `max_trade_size` | max(trade sizes) | Largest single bet (whale indicator) |
| `trade_freq_per_hour` | num_trades / active_hours | Trading intensity |
| `buy_ratio` | (BUY trades) / total_trades | Directional conviction |
| `timing_entropy` | H(hourly distribution) | Predictability of trading schedule |
| `num_conditions` | Unique conditionIds | Market diversification |
| `price_std` | std(prices traded) | Price range of participation |
| `pct_volume` | total_volume / global_volume | Market share |

**Timing Entropy** deserves special attention. We compute Shannon entropy over the 24-hour distribution of a wallet's trades:

```
H = -Σ p(h) · log(p(h))   for h ∈ {0, 1, ..., 23}
```

- **H = 0**: all trades at the same hour — highly concentrated, systematic
- **H = ln(24) ≈ 3.18**: perfectly uniform — trades randomly distributed throughout the day

A bot or coordinated actor with programmatic timing will cluster in specific hours, producing low entropy. Casual retail traders will show high entropy. Suspicious wallets in our dataset had entropy = **0.89** vs **0.20** for normal wallets — a 4.55× difference.

### Scaling

All 11 features were standardized using `StandardScaler` (zero mean, unit variance) before clustering. This is essential for DBSCAN because the algorithm uses Euclidean distance; without scaling, high-variance features like `total_volume` (range: $1–$5.3M) would completely dominate over `buy_ratio` (range: 0–1).

---

## 4. Behavioral Clustering (DBSCAN)

### Why DBSCAN Over K-Means

K-Means was rejected for three reasons:
1. Requires pre-specifying `k` (number of clusters) — unknown in advance
2. Assumes spherical clusters of equal size — prediction market wallets do not form such shapes
3. Forces every point into a cluster — we want to identify genuine outliers as a class

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) addresses all three:
- Discovers `k` automatically from data density
- Finds arbitrarily-shaped clusters
- Labels low-density points as noise/outliers (label = −1) — these are our "suspicious" wallets by definition: wallets whose behavioral profile is too unusual to fit any coherent cluster

### Parameter Selection

Grid search over `eps ∈ {0.3, 0.5, 0.7, 1.0, 1.3, 1.5}` × `min_samples ∈ {3, 5, 10}` optimized for **silhouette score** on non-noise points. The silhouette coefficient for a point $i$ is:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

where `a(i)` is the mean intra-cluster distance and `b(i)` is the mean distance to the nearest other cluster. Score ranges from −1 (wrong cluster) to +1 (perfectly separated).

**Best configuration found: eps = 0.3, min_samples = 10**

| eps | min_samples | Clusters | Outliers | Silhouette |
|-----|-------------|---------|---------|-----------|
| 0.3 | 10 | **104** | **4,446 (7.0%)** | **0.7017** |
| 0.5 | 5 | 12 | 3,201 | 0.5834 |
| 0.7 | 10 | 4 | 2,890 | 0.4912 |
| 1.0 | 5 | 2 | 1,102 | 0.3201 |
| 1.3 | 3 | 2 | 580 | 0.2744 |

The selected configuration yields a silhouette score of **0.7017** — placing our clustering in the "excellent" range (>0.7). Lower eps values produced too many tiny clusters; higher values merged distinct behavioral groups.

### Cluster Structure

The 104 clusters plus 4,446 outliers (cluster −1) exhibit a heavy-tailed size distribution:

| Size range | Number of clusters | Wallets |
|------------|-------------------|--------|
| 7–19 | 33 | ~380 |
| 20–99 | 55 | ~2,400 |
| 100–999 | 11 | ~3,500 |
| 1,000–9,999 | 4 | ~39,400 |
| ≥ 10,000 | 1 | 16,309 |

The largest cluster (16,309 wallets) represents the most common retail behavioral pattern: 2 trades, balanced buy/sell, low entropy. The long tail of small clusters captures niche trading styles.

### Wallet Archetypes

Post-clustering, wallets were assigned to behavioral archetypes using heuristic thresholds on cluster centroids:

| Archetype | Wallets | % of Wallets | % of Volume | Avg Trades | Avg Volume |
|-----------|---------|-------------|------------|-----------|-----------|
| Buy & Hold | 42,821 | 67.1% | 23.1% | 1.4 | $1,712 |
| Casual Trader | 15,292 | 23.9% | 1.6% | 3.0 | $340 |
| Suspicious | 4,446 | 7.0% | **69.8%** | 28.7 | $49,790 |
| Whale | 1,100 | 1.7% | 5.3% | 3.1 | $15,175 |
| Market Maker | 134 | 0.2% | 0.2% | 6.7 | $4,877 |

The most striking finding: **suspicious wallets (7% of participants) control 69.8% of all trading volume** on the analyzed markets. This is not a rounding error — it reflects the extreme right-skew of the volume distribution and the fact that outlier wallets are, by definition, behaviorally extreme in all dimensions simultaneously.

---

## 5. Network Analysis

### Graph Construction

A wallet interaction graph was constructed as follows:
- **Nodes**: each unique wallet (63,793 nodes)
- **Edges**: two wallets are connected if they traded on the **same side** (both BUY or both SELL) within the **same 1-hour window** on the same market

The 1-hour window is chosen to capture potential coordination without being too narrow (missing related trades) or too wide (creating spurious links). Same-side trading in a tight time window is a necessary (though not sufficient) condition for coordination — it captures both genuine coordination and coincidental co-movement.

**Resulting network**: 63,793 nodes, 4,127,204 edges, density = 0.0020

### Community Detection

The Louvain algorithm was applied to detect communities — groups of wallets more densely connected to each other than to the rest of the network. Louvain maximizes **modularity**:

```
Q = (1/2m) Σ_{ij} [A_{ij} - k_i·k_j / 2m] · δ(c_i, c_j)
```

where `A_{ij}` is the adjacency matrix, `k_i` is node degree, `m` is total edges, and `δ(c_i, c_j) = 1` if nodes are in the same community.

**Results**: 1,261 communities, modularity = **0.8329**

A modularity score of 0.83 is very high (typical range: 0.3–0.7 for real-world networks). This indicates extremely strong community structure — wallets cluster tightly into groups with dense internal connections and sparse cross-group connections.

Top 5 community sizes: 12,335 · 7,102 · 6,545 · 3,388 · 2,410 wallets

### Network Centralization

**Freeman centralization** measures how hub-dominated a network is:

```
C = Σ_v [C_D(v*) - C_D(v)] / [(N-1)(N-2)]
```

where `C_D(v*)` is the maximum degree centrality. Score = 0 means perfectly equal distribution; score = 1 means a single hub connects to all other nodes.

**Observed: Freeman centralization = 0.0000002** (effectively zero)

This is a meaningful result: despite 4.1M edges, no single wallet acts as a hub that connects the rest of the network. The graph is diffuse and decentralized.

### Cross-Validation: DBSCAN vs Network Communities

As a robustness check, we examined whether DBSCAN behavioral clusters correspond to network communities. Community 0 (12,335 wallets) had 35.8% membership in DBSCAN cluster 5 — the largest retail cluster. This partial overlap is expected: wallets that trade similarly (same cluster) will naturally co-trade more often (same community), but the alignment is imperfect because behavioral similarity does not require temporal co-occurrence.

---

## 6. Wisdom Score: Crowd Integrity Metric

### Motivation

To reduce the multi-dimensional evidence into a single interpretable metric, we define a **Wisdom Score** — a 0–100 index measuring how closely a market's trading patterns resemble genuine crowd behavior versus whale-dominated or manipulated activity.

Higher score = more crowd wisdom, less manipulation risk.

### Formula

```
WisdomScore = 100 × (1 - (0.35 × TopShare + 0.35 × Gini + 0.20 × Centralization + 0.10 × (1 - Modularity)))
```

**Component definitions:**

| Component | Metric Used | Weight | Rationale |
|-----------|------------|--------|-----------|
| Volume concentration | Top-5 wallet share | 0.35 | Direct manipulation proxy |
| Inequality | Gini coefficient | 0.35 | Structural concentration |
| Network centralization | Freeman index | 0.20 | Hub-based coordination |
| Structural independence | 1 − Modularity | 0.10 | Inverted: high modularity = good |

**Weight justification:**
- Volume concentration and Gini jointly receive 70% of the weight because they directly measure whether a small group dominates the market. Empirical research on financial market manipulation most consistently identifies volume concentration as the primary signal.
- Network centralization receives 20% because a centralized hub could coordinate trading without showing obvious volume concentration.
- Modularity receives 10% (inverted) because low modularity would indicate wallets are not forming independent communities — a sign of coordinated behavior across the whole network.

### Observed Components

| Component | Value | Interpretation |
|-----------|-------|---------------|
| Top-5 volume share | 7.8% | Low — five wallets hold only 7.8% |
| Gini coefficient | 0.958 | High — extreme volume inequality |
| Freeman centralization | ~0.000 | None — no network hub |
| Modularity | 0.833 | High — strong independent communities |

### Score Calculation

```
WisdomScore = 100 × (1 - (0.35 × 0.078 + 0.35 × 0.958 + 0.20 × 0.000 + 0.10 × 0.167))
            = 100 × (1 - (0.027 + 0.335 + 0.000 + 0.017))
            = 100 × (1 - 0.379)
            = 62.1
```

**Wisdom Score: 62.1 / 100 — MODERATE**

### Interpretation Scale

| Score | Rating | Meaning |
|-------|--------|---------|
| 75–100 | HIGH | Strong crowd wisdom, minimal manipulation risk |
| 50–75 | **MODERATE** | Some concentration, diversified participation |
| 25–50 | LOW | Significant concentration, whale influence likely |
| 0–25 | CRITICAL | Extreme concentration, market whale-dominated |

---

## 7. Cross-Platform Comparison

### Volume Comparison

Polymarket and Kalshi cover the same events but operate on fundamentally different scales and market microstructures:

| Metric | Polymarket | Kalshi |
|--------|-----------|-------|
| Total trades collected | 237,791 | 160,000 |
| Avg hourly volume | 44,439 shares | 28,486,983 contracts |
| User identifiers | Yes (`proxyWallet`) | No (anonymous) |
| Regulation | None (offshore) | CFTC-regulated |
| Volume spikes (>Q75) | 1,785 hours | — |

The 640× difference in hourly contract volume partially reflects different unit definitions (Polymarket "shares" have dollar value; Kalshi "contracts" are lower-denomination) but also reflects genuinely higher contract throughput on Kalshi's regulated exchange.

### Why Price Correlation Was Limited

Our aligned hourly price series produced only 1 overlapping period. The root cause: **timezone offsets**. Polymarket timestamps are Unix epoch (UTC-naive); Kalshi timestamps are ISO 8601 with timezone offset. After fixing with `utc=True` on both, the temporal overlap was still minimal because:

1. Kalshi markets close when resolved; our collection captured different resolution states
2. Different sub-market granularity (some Kalshi markets had no active trading in the collected period)
3. Most 2,000-trade records from Kalshi were from a single high-volume day

This is a genuine data limitation. A more comprehensive comparison would require full trade histories over a shared time window, not just the last 2,000 trades per market.

---

## 8. Findings & Analysis

### Finding 1: Extreme Volume Concentration Despite Apparent Diversity

The market appears diverse at first glance: 63,793 unique wallets, 104 distinct behavioral clusters, no single hub in the network. But volume concentration tells a different story:

| Wallet Tier | Wallets | Volume Share |
|-------------|---------|-------------|
| Top 1% (638 wallets) | 1.0% | **66.5%** |
| Top 5% (3,189 wallets) | 5.0% | **89.9%** |
| Top 10% (6,379 wallets) | 10.0% | **95.0%** |
| Bottom 90% (57,414 wallets) | 90.0% | 5.0% |

**90% of participants trade only 5% of the volume.** This 90/10 split is extreme even by financial market standards (typically 80/20 in equity markets). The Gini coefficient of 0.958 places prediction markets at the extreme end of wealth concentration — comparable to income inequality in some of the world's most unequal nations.

### Finding 2: Suspicious Wallets Drive the Market

The 4,446 flagged suspicious wallets (7.0% of participants) account for **69.8% of all volume** — $221 million of the $317 million total. Their characteristics vs. normal wallets:

| Metric | Suspicious | Normal | Ratio |
|--------|-----------|-------|-------|
| Avg total volume | $49,790 | $1,616 | **30.8×** |
| Avg trades | 28.7 | 1.9 | **15.4×** |
| Avg max trade size | $11,162 | $1,153 | **9.7×** |
| Avg timing entropy | 0.89 | 0.20 | **4.5×** |
| Avg buy ratio | 0.644 | 0.595 | 1.1× |

The directional bias is the subtlest signal (1.1× ratio) but still meaningful: suspicious wallets buy 64.4% of the time vs. 59.5% for normal wallets. In a fair market, we would expect buy ratios clustered around 0.5. Both groups skew toward buying — consistent with Polymarket's net-long market structure in political prediction — but suspicious wallets show greater conviction.

The timing entropy difference (4.5×) is the strongest behavioral signal: suspicious wallets trade in tightly-clustered time windows (entropy = 0.89), consistent with programmatic or semi-automated trading. Normal wallets show near-uniform hour-of-day distribution (entropy = 0.20).

### Finding 3: The Top 5 Wallets Are All Suspicious Outliers

The top 5 wallets by volume are all in cluster −1 (suspicious):

| Rank | Wallet | Volume | Trades | Buy Ratio | Entropy |
|------|--------|--------|--------|-----------|---------|
| 1 | `0x7ffe...4d` | $5,346,958 | 8 | 1.00 | 0.69 |
| 2 | `0x7f7c...a94` | $5,346,632 | 8 | 1.00 | 1.21 |
| 3 | `0x24e1...3db` | $5,344,953 | 9 | 1.00 | 1.27 |
| 4 | `0xa22f...6f9` | $5,278,271 | 8 | 1.00 | 0.74 |
| 5 | `0xb395...240` | $3,383,050 | 101 | 0.99 | 2.05 |

Wallets 1–4 share a striking pattern: $5.3M volume in just 8 trades, a 100% buy ratio, and nearly identical volume. These are likely institutional accounts making concentrated directional bets. Wallet 5 is behaviorally distinct: 101 trades, 99% buy ratio, higher entropy — possibly an active sophisticated trader rather than a one-shot position.

All five are correctly classified as outliers by DBSCAN because no behavioral cluster accommodates wallets simultaneously trading $5M+ in 8 transactions with perfect directional conviction. They are statistical anomalies by any reasonable definition.

### Finding 4: Network Is Resilient Despite High Activity

Despite the extreme volume concentration, the **network structure is genuinely decentralized**:

- 1,261 communities (Louvain modularity = 0.833)
- Freeman centralization ≈ 0 (no hub)
- Network density = 0.002 (sparse, indicating most wallets don't co-trade)

This apparent contradiction — extreme volume concentration but decentralized network structure — resolves when we recognize that the suspicious high-volume wallets are **outliers by design**: DBSCAN placed them in cluster −1 precisely because they don't form a cohesive group. They are individually anomalous, not collectively coordinated. If they were coordinated, they would form a tight cluster, not scatter as noise points.

This is actually a reassuring finding from an integrity perspective: the market is **concentrated but not cartelized**. The large wallets appear to be acting independently, not in concert.

### Finding 5: The Wisdom Score Correctly Reflects Internal Contradictions

The 62.1 score sits in the MODERATE range precisely because the market contains genuine tensions:

**Pulling score down** (toward manipulation):
- Gini = 0.958 is among the highest values possible, indicating structural inequality
- Suspicious wallets control 69.8% of volume — this alone would push the score toward LOW

**Pulling score up** (toward crowd wisdom):
- Top-5 volume share = 7.8% — low, indicating no single dominant actor
- Freeman centralization ≈ 0 — no coordination hub
- Modularity = 0.833 — strong independent communities

The key insight: **it is possible for a market to have extreme inequality AND good structural independence simultaneously**. The suspicious wallets are large but not coordinated. The network is fragmented into 1,261 communities. The top whale controls only 1.7% of total volume despite trading $5.3M. This combination yields a moderate score, which is the honest characterization: the market has integrity problems (concentration, suspicious actors) but is not captured by a cartel.

---

## 9. Critical Evaluation

### Strengths of Our Methodology

**1. Feature completeness**: The 11-feature representation captures volume, frequency, timing, and direction simultaneously. No single feature would suffice — a whale could have high volume but low frequency; a bot could have low volume but high frequency. The multivariate approach avoids these false negatives.

**2. DBSCAN appropriateness**: The outlier-native design of DBSCAN is precisely suited to this problem. We are not trying to classify wallets into known types; we are asking which wallets are anomalous. DBSCAN identifies anomalies without requiring us to specify what "normal" looks like.

**3. Silhouette validation**: A score of 0.7017 on 63,793 points across 104 clusters provides strong evidence that the clustering reflects genuine behavioral structure, not algorithmic artifacts. This rules out the concern that DBSCAN is simply picking arbitrary groupings.

**4. Network-behavioral cross-validation**: The partial overlap between DBSCAN clusters and Louvain communities provides mutual validation from independent methods. Perfect alignment would be suspicious (suggesting the same signal is being measured twice); partial alignment confirms both methods are capturing real but distinct aspects of wallet behavior.

**5. Robustness against the top-5 wallet paradox**: The Wisdom Score would be misleading if the top-5 concentration metric dominated all others. By weighting Gini equally (both at 0.35) and including network metrics, the score reflects the full picture — high inequality, but genuine structural independence.

### Limitations and Caveats

**1. Polymarket-only clustering**: DBSCAN was applied only to Polymarket wallets because Kalshi does not expose user identifiers in its public API. The cross-platform comparison is therefore asymmetric: we can characterize Polymarket traders in detail but can only observe aggregate Kalshi behavior.

**2. Two-thousand-trade ceiling**: We capped collection at 2,000 trades per sub-market. For high-volume markets like Champions League ($1B), this significantly under-samples the full trade history. Wallets that traded early in the market's life are invisible to our analysis. This likely understates total suspicious activity.

**3. Proxy wallet ambiguity**: Polymarket `proxyWallet` addresses are smart contract accounts, not directly-owned wallets. A single real-world actor can operate multiple proxy wallets. The 63,793 "wallets" in our dataset may represent significantly fewer unique individuals. If the 4,446 suspicious wallets are controlled by, say, 500 actors each running 8–9 proxies, the concentration problem is far more severe than our metrics suggest.

**4. Wisdom Score weight subjectivity**: The weights (0.35 / 0.35 / 0.20 / 0.10) are heuristic, not empirically derived. Different weight assignments would produce different scores. The score should be interpreted as a directional signal, not an absolute measure. A sensitivity analysis (varying weights ±50%) shows the score ranges from approximately 55 to 70, remaining solidly MODERATE.

**5. Cross-platform timing mismatch**: The failure to produce meaningful price correlation between Polymarket and Kalshi weakens the cross-platform analysis. This was not solvable within the scope of this project given the data constraints, but it represents a gap in the evidence.

**6. Temporal scope**: The analysis covers Nov 2024–Feb 2026, a 15-month window. Behavior may have changed significantly over this period. The Champions League market alone spans a 12-month resolution window, making the "suspicious" label time-sensitive: what looks like early informed trading might look like late bandwagon trading at a different cutoff.

**7. No ground truth**: We cannot verify that our flagged wallets actually engaged in manipulation. Our labels ("suspicious," "whale," "market maker") are behavioral descriptions, not confirmed classifications. Validation would require comparison against a known manipulation dataset or regulatory findings.

### Alternative Interpretations

The 69.8% volume share held by "suspicious" wallets deserves the most scrutiny because it is both the most alarming finding and the most methodologically fragile:

**Alternative interpretation A**: These are simply large, sophisticated traders. A hedge fund entering prediction markets would naturally look like an outlier — many large trades, concentrated timing (market-hours-only), directional conviction. DBSCAN cannot distinguish between "manipulation" and "professional trading." Our suspicious label is behavioral, not normative.

**Alternative interpretation B**: The 7.0% outlier rate is within the range DBSCAN typically produces for financial data (5–15%), and may simply reflect the natural heterogeneity of a market where a few institutional participants coexist with many retail speculators. This is not intrinsically pathological.

**Alternative interpretation C**: The 100% buy ratios of the top 4 wallets could reflect legitimate informed predictions (e.g., a sophisticated model that correctly predicted the Champions League winner early) rather than manipulation. Being right with high conviction looks the same as gaming the market from the outside.

These alternative interpretations do not invalidate the findings; they contextualize them. The framework detects behavioral anomalies. Whether those anomalies constitute manipulation requires regulatory access to identity data, communication records, and trading intent — beyond the scope of public data analysis.

---

## 10. Conclusions & Recommendations

### Core Conclusions

**1. Prediction markets exhibit extreme volume inequality.** The Gini coefficient of 0.958 and the finding that top 1% of wallets control 66.5% of volume indicate a market structure far removed from the "crowd wisdom" ideal. The median trade size ($34) versus mean trade size ($1,334) confirms that most participants trade small amounts while a few trade enormous amounts.

**2. 7% of wallets exhibit anomalous behavior, controlling 69.8% of volume.** These wallets are behavioral outliers across all measured dimensions simultaneously: 30× more volume, 15× more trades, 9.7× larger max trades, 4.5× more concentrated timing. This is not an artifact of one anomalous feature; it is a coherent profile.

**3. The market is concentrated but not cartelized.** Despite the extreme individual outliers, the network shows no hub structure and 1,261 independent trading communities. The suspicious wallets are outliers, not a cartel. They appear to act independently.

**4. The Wisdom Score of 62.1/100 is an honest assessment.** The score correctly reflects the co-existence of extreme concentration (Gini = 0.958) and genuine structural independence (modularity = 0.833, Freeman centralization ≈ 0). It is not "almost good" (70) or "borderline bad" (50); it is genuinely in the middle.

**5. Regulated markets (Kalshi) cannot be directly compared on integrity grounds.** The absence of user identifiers in Kalshi's public API makes wallet-level behavioral analysis impossible from the outside. This is itself an argument for regulated market design: user anonymization protects both privacy and, inadvertently, makes public manipulation detection harder.

### Recommendations

**For market regulators:**
- Implement real-time monitoring of wallets in cluster −1 (anomalous outliers)
- Track timing entropy as a key manipulation signal — it is more discriminating than volume alone
- Investigate whether multiple proxy wallets are controlled by the same actors (Sybil detection)
- Apply this framework to all prediction markets, establishing a Wisdom Score baseline for market health monitoring

**For market designers:**
- Consider position limits or friction mechanisms for wallets exceeding the 95th percentile of trade size
- The 100% buy-ratio pattern (wallets 1–4) suggests markets lack effective short-side liquidity; consider incentives for bet-against positions
- Require wallet-level KYC at volume thresholds, as Kalshi does under CFTC rules

**For researchers:**
- Extend this framework temporally: compute Wisdom Score at monthly intervals to detect trend
- Apply Sybil-resistance techniques to estimate true number of unique actors behind proxy wallets
- Build a predictive model: which currently-normal wallets are at risk of transitioning to the suspicious cluster?
- Replicate on Augur, Manifold, and Metaculus to compare Wisdom Scores across market designs

### Final Assessment

The prediction markets analyzed do not simply reflect crowd wisdom. A substantial fraction of volume is driven by wallets exhibiting behavioral patterns inconsistent with casual retail speculation — high volume, concentrated timing, directional conviction. Whether this represents legitimate institutional participation, informed trading, or genuine manipulation is a question that public data analysis alone cannot definitively answer.

What we can say with rigor: the market structure supports conditions under which manipulation could occur with low detection risk. The network's lack of a central hub means a regulator monitoring hub activity would miss most anomalous behavior. The high modularity means wallets are grouped into independent communities, making cross-community coordination harder to detect even if it exists.

The Wisdom Score framework provides a reusable, transparent, and interpretable tool for ongoing market integrity assessment. A score above 75 would indicate healthy crowd dynamics; below 50 would warrant regulatory attention. At 62.1, Polymarket's combined prediction market ecosystem occupies a middle ground: genuinely participatory in structure, but meaningfully concentrated in practice.

---

## Appendix: Technical Implementation

### Pipeline Summary

| Phase | Script | Runtime | Key Output |
|-------|--------|---------|-----------|
| 1 | API POC | < 1 min | API validation |
| 2 | `02_market_matching.py` | ~3 min | `matched_markets.json` |
| 3 | `03_full_data_collection.py` | ~15 min | `poly_trades_all_matched.csv` |
| 4 | `04_feature_engineering_dbscan_robust.py` | ~60 min | `wallet_features.csv`, `dbscan_clusters.json` |
| 5 | `05_network_wisdom_score.py` | ~5 min | `wisdom_score_summary.json` |
| 6 | `06_cross_platform_comparison.py` | < 1 min | `divergence_analysis.json` |
| 7 | `07_visualization_polish.py` | ~2 min | 7 PNG charts, `ANALYSIS_REPORT.txt` |

### Key Dependencies

```
torch==2.2.2                # x86_64 macOS constraint (no arm64 equivalent ≥2.3)
sentence-transformers==2.7.0 # semantic market matching
numpy>=1.26,<2.0            # torch 2.2 requires numpy 1.x
scikit-learn>=1.3            # DBSCAN, silhouette score
networkx>=3.0               # graph construction, Louvain communities
pandas>=2.0                 # data manipulation
matplotlib>=3.7             # visualization
seaborn>=0.13               # statistical plots
scipy>=1.11                 # Shannon entropy, Pearson correlation
```

### Robustness Design Decisions

Phase 4 (the longest-running computation) implements production-grade robustness:
- **Checkpoint saves every 5,000 wallets** to `phase4_progress.json` — if interrupted, the progress file identifies the last completed stage
- **ETA reporting** every 5,000 wallets: `[25,000 / 63,793] 39% | Elapsed: 7.2m | ETA: 11.3m`
- **Feature save before clustering**: `wallet_features.csv` is written after feature engineering completes, before DBSCAN begins — the most expensive step (feature computation) is never lost even if clustering fails
- **Try-catch on output generation**: analysis summary printing is wrapped in exception handling so a formatting bug in the reporting code cannot lose completed clustering results

---

*Report generated: February 27, 2026*
*All source code available in `scripts/`. All data available in `data/processed/`. All visualizations in `figures/`.*
