# Prediction Market Integrity Auditor - Jupyter Notebooks for Google Colab

**Run a complete prediction market analysis in Google Colab - zero local setup required!**

These 6 Jupyter notebooks analyze Polymarket and Kalshi to detect whale manipulation using semantic event matching, multi-threaded data collection, DBSCAN clustering, network analysis, and Wisdom Score computation.

All notebooks are ready to run in Google Colab without any local dependencies. Simply download the `.ipynb` files and upload them to Colab.

## 📚 Notebook Overview

| Phase | File | Runtime | Task | Output |
| --- | --- | --- | --- | --- |
| **2** | `02_market_matching.ipynb` | ~5 min | Match Polymarket & Kalshi events using semantic similarity (sentence-transformers) | matched_markets.json (6 pairs) |
| **3** | `03_full_data_collection.ipynb` | ~15 min | Multi-threaded API collection from both platforms (2000 trades/market) | 237k+ Poly + 160k+ Kalshi trades |
| **4** | `04_feature_engineering_dbscan.ipynb` | ~60 min | Engineer 11 wallet features + DBSCAN clustering with parameter grid search | 104 clusters + 4,446 outliers (7%) |
| **5** | `05_network_wisdom_score.ipynb` | ~5 min | Build wallet interaction network (Louvain communities) + compute Surowiecki Wisdom Score | Wisdom Score: 76.6/100 (Crowd Wisdom Signal w/ Diversity warning) |
| **6** | `06_cross_platform_comparison.ipynb` | ~2 min | Hourly price alignment + correlation + divergence analysis | price_comparison.csv |
| **7** | `07_visualization_polish.ipynb` | ~3 min | Create 8 publication-ready charts (300 DPI) + analysis report, including Mode 3 temporal causality | figures/ + ANALYSIS_REPORT.txt |

## 🚀 Quick Start in Google Colab

1. **Go to** [Google Colab](https://colab.research.google.com/)
2. **Upload notebook** → File menu → Open notebook → Upload tab
3. **Run cells in order** → Shift + Enter
4. **Download results** → Folder icon on left → Right-click file → Download

## ✅ Expected Results After All Phases

| File | Size | Contains |
|------|------|----------|
| matched_markets.json | < 1 KB | 6 market pairs |
| poly_trades_all_matched.csv | 42 MB | 237,791 trades from Polymarket |
| kalshi_trades_all_matched.csv | 12 MB | 160,000 trades from Kalshi |
| wallet_features.csv | 8 MB | 63,793 wallets × 11 features |
| dbscan_clusters.json | 5.2 MB | Cluster assignments |
| wisdom_score_summary.json | < 1 KB | Wisdom Score = 76.6/100 (Surowiecki sub-scores) |
| mode3_causality_results.json | < 1 KB | Mode 3 temporal causality test (LOW risk, p=0.531) |
| 8 × PNG charts | 3.0 MB | Visualizations (300 DPI) incl. Figure 8 (Mode 3) |
| ANALYSIS_REPORT.txt | < 100 KB | Comprehensive findings with three failure modes |

**Total**: ~78 MB of analysis outputs

## 📋 Recommended Execution Order

```
1. 02_market_matching.ipynb         (5 min)
   ↓
2. 03_full_data_collection.ipynb    (15 min)
   ↓
3. 04_feature_engineering_dbscan.ipynb (60 min) ⚠️ longest
   ↓
4. 05_network_wisdom_score.ipynb    (5 min)
   ↓
5. 06_cross_platform_comparison.ipynb (2 min)
   ↓
6. 07_visualization_polish.ipynb     (3 min)
```

**Total Runtime**: ~90 minutes

## 🔧 Colab Tips

### Mounting Google Drive (Optional)
Save results to Drive for easy access:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Then save files to `/content/drive/My Drive/prediction-whale/`

### Managing Files
- **Download from Colab**: Click folder icon on left → Right-click file → Download
- **Upload to Colab**: Drag & drop or use folder icon → Upload

### Runtime Limits
- Free tier: 12 hours continuous, then resets
- Phase 4 takes ~60 min; may need to run overnight
- Each notebook is independent (restart between phases if needed)

### Memory Issues
- Colab provides ~12 GB RAM (sufficient for this project)
- If issues occur: restart kernel or use Colab Pro ($10/month)

## 📊 Key Findings Summary

```
Wisdom Score: 76.6 / 100 (CROWD WISDOM SIGNAL with Diversity Warning)

SUROWIECKI CONDITIONS:
  - Diversity    : 30.2/100  ← FAILING (suspicious group controls 69.8% volume)
  - Independence : 93.0/100  ← STRONG (no lockstep herding detected)
  - Decentralization: 100.0/100 ← STRONG (no network hub, 1,261 communities)
  - Aggregation  : 83.3/100  ← STRONG (modularity 0.833)

THREE FAILURE MODES:
  - Mode 1 (Herding)   : LOW  — suspicious wallets don't move in lockstep
  - Mode 2 (Minority price-setting): HIGH — suspicious group dominates volume
  - Mode 3 (Active manipulation): LOW — no temporal lead pattern (p=0.531)

Suspicious Wallets (7% of participants) control 69.8% of volume:
  - Top 1%  (638 wallets)  → 66.5% of volume
  - Top 5%  (3,189 wallets) → 89.9% of volume
  - Top 10% (6,379 wallets) → 95.0% of volume

104 behavioral clusters identified:
  - Buy & Hold: 42,821 wallets (23.1% of volume)
  - Casual Trader: 15,292 wallets (1.6% of volume)
  - Suspicious: 4,446 wallets (69.8% of volume) ⚠️
  - Whale: 1,100 wallets (5.3% of volume)
  - Market Maker: 134 wallets (0.2% of volume)

Network Analysis:
  - 1,261 communities (highly decentralized)
  - Freeman centralization = 0.0000 (no hub)
  - Modularity = 0.833 (very strong communities)
```

## ❓ Troubleshooting

**"ModuleNotFoundError: No module named X"**
→ Run the pip install cell first (each notebook has one)

**"Failed to fetch data from API"**
→ APIs might be rate-limited; wait 30 seconds and retry

**"Memory limit exceeded"**
→ Colab ran out of RAM; restart kernel and continue with next phase

**"Runtime disconnected"**
→ Colab disconnected after inactivity; reconnect and resume

## 📖 For More Details

- **[COLAB_GUIDE.md](../COLAB_GUIDE.md)** - Complete Colab setup, expected runtimes, troubleshooting
- **[REPORT.md](../REPORT.md)** - Detailed methodology, findings, and critical evaluation
- **[README.md](../README.md)** - Project overview, architecture, and key concepts

## 🎯 What You'll Learn

- **Semantic matching**: How to align events across prediction platforms
- **Feature engineering**: 11 wallet features for behavioral classification
- **DBSCAN clustering**: Unsupervised detection of suspicious traders (7% outlier rate)
- **Network analysis**: Wallet interaction graphs and Louvain communities (1,261)
- **Wisdom Score**: Surowiecki framework — four conditions (Diversity, Independence, Decentralization, Aggregation)
- **Three failure modes**: Herding vs. minority price-setting vs. active manipulation
- **Temporal causality**: Lead-lag analysis to test whether whales move before retail traders
- **Market integrity**: Concrete metrics for evaluating crowd wisdom and manipulation risk

---

**All notebooks are ready to run immediately in Colab!** ✅

Start with Phase 2 and run through Phase 7 (~90 minutes total). Each notebook is independent and can be rerun anytime.
