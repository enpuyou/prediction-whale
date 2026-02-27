# 📓 Google Colab Guide - Prediction Market Integrity Auditor

Run the analysis in Google Colab without installing anything locally!

## 🚀 Quick Start

1. **Download a notebook** from the `notebooks/` folder
2. **Go to [Google Colab](https://colab.research.google.com/)**
3. **Upload the notebook** (File → Open notebook → Upload tab)
4. **Run cells in order** (Shift + Enter)

## 📚 Available Notebooks

### Phase 2: Market Matching
- **File**: `02_market_matching.ipynb`
- **Runtime**: ~5 minutes
- **What it does**: Matches Polymarket and Kalshi events using semantic similarity
- **Output**: `matched_markets.json` (6 market pairs)
- **Dependencies**: requests, pandas, sentence-transformers

### Phase 3: Full Data Collection
- **File**: `03_full_data_collection.ipynb`
- **Runtime**: ~15 minutes
- **What it does**: Collects 237k+ Polymarket trades and 160k+ Kalshi trades
- **Output**: `poly_trades_all_matched.csv`, `kalshi_trades_all_matched.csv`
- **Note**: Downloads ~54 MB of data
- **Dependencies**: requests, pandas

### Phase 4: Feature Engineering & DBSCAN
- **File**: `04_feature_engineering_dbscan.ipynb`
- **Runtime**: ~60 minutes (longest phase)
- **What it does**: Engineers 11 wallet features, runs DBSCAN clustering
- **Output**: `wallet_features.csv` (8 MB), `dbscan_clusters.json` (5.2 MB)
- **Features**: 104 clusters, 4,446 outliers, silhouette=0.7017
- **Dependencies**: pandas, numpy, scikit-learn, scipy

### Phase 5: Network & Wisdom Score
- **File**: `05_network_wisdom_score.ipynb`
- **Runtime**: ~5 minutes
- **What it does**: Builds wallet interaction network, computes Wisdom Score
- **Output**: `wisdom_score_summary.json`, `network_stats.txt`
- **Result**: Wisdom Score = 62.1/100 (MODERATE)
- **Dependencies**: pandas, networkx, numpy

### Phase 6: Cross-Platform Comparison
- **File**: `06_cross_platform_comparison.ipynb`
- **Runtime**: ~2 minutes
- **What it does**: Compares prices and volumes between Polymarket and Kalshi
- **Output**: `price_comparison.csv`, `divergence_analysis.json`
- **Dependencies**: pandas, numpy, scipy

### Phase 7: Visualization & Report
- **File**: `07_visualization_polish.ipynb`
- **Runtime**: ~3 minutes
- **What it does**: Creates 7 publication-ready charts
- **Output**: PNG charts and `ANALYSIS_REPORT.txt`
- **Charts**: Cluster distribution, archetypes, correlation, Wisdom Score, outliers, concentration, temporal
- **Dependencies**: pandas, matplotlib, seaborn, numpy

## 📋 Recommended Execution Order

**Option A: Run All Phases (90 minutes total)**
```
1. 02_market_matching.ipynb         (5 min)
2. 03_full_data_collection.ipynb    (15 min)
3. 04_feature_engineering_dbscan.ipynb (60 min) ⚠️ longest
4. 05_network_wisdom_score.ipynb    (5 min)
5. 06_cross_platform_comparison.ipynb (2 min)
6. 07_visualization_polish.ipynb     (3 min)
```

**Option B: Skip to Analysis (10 minutes)**
If you already have the data, start with Phase 4 using pre-downloaded files.

**Option C: Visualization Only (3 minutes)**
If you have all feature files, run Phase 7 directly.

## 🔧 Colab Tips

### Mounting Google Drive (Optional)
If you want to save results to Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then save files to `/content/drive/My Drive/prediction-whale/`

### Managing Files
- Download files from Colab: Click the folder icon on left, right-click file → Download
- Upload files to Colab: Folder icon → Upload (or just drag-drop)

### Runtime Limits
- Colab free tier: 12 hours continuous, then resets
- Each notebook is independent — if one crashes, restart and continue with next
- Phase 4 takes ~60 min; may need to run overnight

### Memory Issues
If you get memory errors:
- Colab provides ~12 GB RAM; enough for this project
- If issues persist: Run Phase 4 in smaller batches or use Colab Pro ($10/month)

## 📊 Expected Results

After running all phases, you'll have:

| File | Size | Contains |
|------|------|----------|
| matched_markets.json | < 1 KB | 6 market pairs |
| poly_trades_all_matched.csv | 42 MB | 237,791 trades |
| kalshi_trades_all_matched.csv | 12 MB | 160,000 trades |
| wallet_features.csv | 8 MB | 63,793 wallets × 11 features |
| dbscan_clusters.json | 5.2 MB | Cluster assignments |
| wisdom_score_summary.json | < 1 KB | Wisdom Score = 62.1/100 |
| 7 × PNG charts | 2.7 MB | Visualizations |

**Total**: ~78 MB of analysis outputs

## 🎯 Key Findings (From Running These Notebooks)

```
Wisdom Score: 62.1 / 100 (MODERATE)

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
→ Run the pip install cell first (in each notebook)

**"Failed to fetch data from API"**
→ APIs might be rate-limited; wait 30 seconds and retry

**"Memory limit exceeded"**
→ Colab ran out of RAM; restart kernel and continue with next phase

**"Runtime disconnected"**
→ Colab disconnected after 30 min inactivity; reconnect and resume

## 📞 Questions?

Read the comprehensive report: `REPORT.md` in the same folder as these notebooks.

**Key sections:**
- Section 3: Feature Engineering (explains the 11 wallet features)
- Section 4: DBSCAN Methodology (explains clustering)
- Section 5: Network Analysis (explains graph structure)
- Section 6: Wisdom Score (explains the integrity metric)
- Section 8: Key Findings (explains the results)

---

**All notebooks are ready to run immediately in Colab!** ✅
