# Prediction Market Integrity Auditor

**Do Prediction Markets Reflect Crowd Wisdom, or Whale Manipulation?**

A comprehensive quantitative analysis of trading behavior, wallet clustering, network structure, and market integrity across Polymarket and Kalshi prediction markets.

## 📊 Project Overview

This project analyzes 237,791 trades from 63,793 unique wallets across 6 matched prediction market pairs to quantify whale manipulation risk using behavioral clustering (DBSCAN), network analysis (Louvain communities), and a composite Wisdom Score measuring market integrity.

**Key Finding**: Suspicious wallets (7% of participants) control **69.8% of trading volume**, despite low network centralization and strong community structure.

## 📁 Repository Structure

```
prediction-whale/
├── REPORT.md                          ← Main comprehensive report (read this first!)
├── README.md                          ← This file
│
├── scripts/
│   ├── 02_market_matching.py          Phase 2: Event-level semantic matching
│   ├── 03_full_data_collection.py     Phase 3: Multi-threaded API collection
│   ├── 04_feature_engineering_dbscan_robust.py    Phase 4: Clustering with checkpoints
│   ├── 05_network_wisdom_score.py     Phase 5: Network & Wisdom Score computation
│   ├── 06_cross_platform_comparison.py            Phase 6: Cross-platform analysis
│   └── 07_visualization_polish.py     Phase 7: Publication-ready visualizations
│
├── data/
│   ├── raw/                           Raw API responses (JSON)
│   │   ├── poly_all_events.json       2,000 Polymarket events
│   │   ├── kalshi_all_events.json     4,897 Kalshi events
│   │   └── [market-specific trade files]
│   │
│   └── processed/
│       ├── matched_markets.json       6 final market pairs with metadata
│       ├── poly_trades_all_matched.csv            237,791 trades (42 MB)
│       ├── kalshi_trades_all_matched.csv          160,000 trades (12 MB)
│       ├── wallet_features.csv        63,793 wallets × 11 features (8 MB)
│       ├── dbscan_clusters.json       Cluster assignments (5.2 MB)
│       ├── cluster_summary.txt        Human-readable cluster analysis
│       ├── wisdom_score_summary.json   Wisdom Score & network metrics
│       ├── network_stats.txt          Network analysis summary
│       ├── price_comparison.csv       Hourly price alignment
│       └── divergence_analysis.json   Cross-platform divergence
│
├── figures/
│   ├── 01_cluster_distribution.png              Cluster sizes & outlier %
│   ├── 02_wallet_archetypes.png                 Buy&Hold, Casual, Suspicious, Whale, MM
│   ├── 03_feature_correlation.png               Feature importance & correlations
│   ├── 04_wisdom_score_breakdown.png            Integrity gauge & components
│   ├── 05_outlier_detection.png                 Suspicious vs normal characteristics
│   ├── 06_volume_concentration.png              Lorenz curve & top 20 wallets
│   ├── 07_temporal_patterns.png                 Trading patterns by hour/day
│   └── ANALYSIS_REPORT.txt           Executive summary
│
└── pyproject.toml                     Poetry dependency configuration
```

## 🎯 Key Metrics At A Glance

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Wisdom Score** | 62.1 / 100 | MODERATE market integrity |
| **Suspicious Wallets** | 4,446 (7.0%) | Behavioral outliers via DBSCAN |
| **Volume Concentration** | Top 1% = 66.5% | Extreme inequality (Gini = 0.958) |
| **Network Communities** | 1,261 | Highly decentralized structure |
| **Clustering Quality** | Silhouette = 0.7017 | EXCELLENT (>0.7) |
| **Total Participants** | 63,793 | Broad but unequal participation |
| **Total Volume** | $317.2M | Across 6 major prediction markets |

## 📈 The Core Finding

```
Suspicious Wallets (7.0% of participants) control 69.8% of all trading volume

Volume Distribution:
  Top 1%   (638 wallets)   → 66.5% of volume
  Top 5%   (3,189 wallets) → 89.9% of volume
  Top 10%  (6,379 wallets) → 95.0% of volume
  Bottom 90% (57,414 wallets) → 5.0% of volume
```

Suspicious wallet characteristics vs. normal wallets:
- **30.8× higher average volume** ($49,790 vs $1,616)
- **15.4× more trades** (28.7 vs 1.9)
- **9.7× larger max trade** ($11,162 vs $1,153)
- **4.5× more concentrated timing** (entropy 0.89 vs 0.20)

## 🔬 Methodology Overview

### Phase 1: Market Matching
- Semantic similarity matching of event titles using sentence-transformers
- 2,000 Polymarket events vs 4,897 Kalshi events
- Result: 6 matched pairs (8 perfect 1.0 similarity scores)

### Phase 2: Data Collection
- Multi-threaded collection (ThreadPoolExecutor, 4 workers)
- 237,791 Polymarket trades, 160,000 Kalshi trades
- Covers 15 months (Nov 2024 – Feb 2026)

### Phase 3: Feature Engineering
11 wallet features capturing volume, activity, timing, and direction:
- Volume: `total_volume`, `avg_trade_size`, `max_trade_size`, `pct_volume`
- Activity: `num_trades`, `trade_freq_per_hour`
- Timing: `timing_entropy` (0 = clustered, 3.18 = uniform)
- Direction: `buy_ratio` (0 = all sells, 1 = all buys)
- Diversification: `num_conditions`, `price_std`

### Phase 4: Behavioral Clustering
- **Algorithm**: DBSCAN (density-based, naturally finds outliers)
- **Parameters**: eps=0.3, min_samples=10 (optimized via silhouette score)
- **Results**: 104 clusters + 4,446 outliers (noise cluster = −1)
- **Quality**: Silhouette score = 0.7017 (excellent, >0.4 is good)

### Phase 5: Network Analysis
- **Graph**: Wallets connected if co-trading same side within 1-hour window
- **Structure**: 63,793 nodes, 4,127,204 edges, density = 0.002
- **Communities**: 1,261 communities detected via Louvain algorithm
- **Modularity**: 0.833 (very high, indicates strong independent groups)
- **Centralization**: 0.0000 (no hub structure)

### Phase 6: Wisdom Score
Composite metric combining:
- **Volume concentration** (top-5 share: 7.8%, weight 0.35)
- **Wealth inequality** (Gini coefficient: 0.958, weight 0.35)
- **Network centralization** (Freeman index: ~0, weight 0.20)
- **Community independence** (modularity: 0.833, weight 0.10)

Formula: `WisdomScore = 100 × (1 - (0.35×TopShare + 0.35×Gini + 0.20×Centralization + 0.10×(1−Modularity)))`

Result: **62.1 / 100 (MODERATE)**

## 📖 How To Read This Project

**If you have 5 minutes:**
→ Read the Key Metrics section above and the Finding summary

**If you have 30 minutes:**
→ Read REPORT.md sections 1, 8 (Findings), and 10 (Conclusions)

**If you have 2 hours:**
→ Read REPORT.md in full (all 571 lines)

**If you want to deep-dive:**
→ Load `data/processed/wallet_features.csv` into your analysis tool and replicate the findings

## 🛠️ Running the Analysis

### Prerequisites
```bash
# Using Poetry
poetry install

# Environment
Python 3.11.7
torch==2.2.2 (x86_64 macOS only)
sentence-transformers==2.7.0
scikit-learn>=1.3
networkx>=3.0
pandas>=2.0
matplotlib>=3.7
seaborn>=0.13
```

### Execute Pipeline
```bash
# Phase 2: Market Matching
poetry run python scripts/02_market_matching.py

# Phase 3: Data Collection
poetry run python scripts/03_full_data_collection.py

# Phase 4: Clustering (60 minutes)
poetry run python scripts/04_feature_engineering_dbscan_robust.py

# Phase 5: Network & Wisdom Score
poetry run python scripts/05_network_wisdom_score.py

# Phase 6: Cross-Platform Comparison
poetry run python scripts/06_cross_platform_comparison.py

# Phase 7: Visualization
poetry run python scripts/07_visualization_polish.py
```

## 🎨 Visualizations

All charts in `figures/` are publication-ready (300 DPI PNG):

1. **Cluster Distribution** — Wallet clustering results, outlier rates, cluster sizes
2. **Wallet Archetypes** — Buy&Hold, Casual Trader, Suspicious, Whale, Market Maker profiles
3. **Feature Correlation** — Feature importance and multivariate relationships
4. **Wisdom Score Breakdown** — Integrity gauge with component contributions
5. **Outlier Detection** — Suspicious vs normal wallet characteristics
6. **Volume Concentration** — Lorenz curve and top 20 wallets
7. **Temporal Patterns** — Trading volume and frequency by hour/day

## ⚠️ Limitations & Caveats

1. **Proxy wallet ambiguity**: Polymarket `proxyWallet` addresses are smart contracts, not individuals. A single actor can operate multiple proxies.

2. **Incomplete trade history**: Collection capped at 2,000 trades per sub-market. High-volume markets (Champions League, $1B) are significantly under-sampled.

3. **Kalshi anonymity**: Kalshi public API provides no wallet identifiers, making comparable behavioral analysis impossible.

4. **No ground truth**: "Suspicious" is a statistical label, not a confirmed classification. We cannot verify actual manipulation without regulatory data.

5. **Wisdom Score weights**: The component weights (0.35, 0.35, 0.20, 0.10) are heuristic. Sensitivity analysis shows the score ranges 55–70 under reasonable weight variations.

6. **Cross-platform timing mismatch**: Only 1 overlapping hourly period between Polymarket and Kalshi after timezone normalization.

## 🔮 Future Extensions

1. **Temporal Evolution**: Compute Wisdom Score monthly to detect trend
2. **Sybil Detection**: Estimate how many unique actors control the 4,446 suspicious wallets
3. **Predictive Modeling**: Which currently-normal wallets are at risk of becoming suspicious?
4. **Cross-Market Comparison**: Apply framework to Augur, Manifold, Metaculus
5. **Real-Time Monitoring**: API for live wallet behavior alerts

## 📚 References & Data Sources

**APIs Used:**
- Polymarket Gamma API: `gamma-api.polymarket.com`
- Polymarket Data API: `data-api.polymarket.com`
- Kalshi Public API: `api.elections.kalshi.com`

**All data queried**: November 15, 2024 – February 27, 2026

**No authentication required** — all endpoints are public

## 📧 Citation

If you use this framework or findings in your work, please cite:

```
Prediction Market Integrity Auditor (2026)
MSIS 521A Winter 2026 Final Project
Analysis of 237,791 trades across Polymarket and Kalshi prediction markets
Available: https://github.com/[your-repo]
```

## 📄 License

This analysis and all source code is provided as-is for educational and research purposes.

---

**Project Status**: ✅ **COMPLETE** (All 7 Phases Finished)

Last updated: February 27, 2026
