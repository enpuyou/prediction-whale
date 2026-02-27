# Prediction Market Integrity Auditor - Jupyter Notebooks for Google Colab

All notebooks are ready to run in Google Colab without any local setup. Simply download the `.ipynb` files and upload them to Colab.

## 📚 Notebook Overview

| Phase | File | Runtime | Task | Notes |
|-------|------|---------|------|-------|
| **2** | `02_market_matching.ipynb` | ~5 min | Match Polymarket & Kalshi events using semantic similarity | Requires: matched_markets.json from Phase 1 |
| **3** | `03_full_data_collection.ipynb` | ~15 min | Collect 237k+ Polymarket & 160k+ Kalshi trades | Multi-threaded API collection (~54 MB download) |
| **4** | `04_feature_engineering_dbscan.ipynb` | ~60 min | Engineer 11 wallet features + DBSCAN clustering | **Longest phase** - identify 104 clusters + 4,446 outliers |
| **5** | `05_network_wisdom_score.ipynb` | ~5 min | Build wallet interaction network + compute Wisdom Score | Outputs: 62.1/100 (MODERATE integrity) |
| **6** | `06_cross_platform_comparison.ipynb` | ~2 min | Compare prices/volumes across platforms | Identify price divergence windows |
| **7** | `07_visualization_polish.ipynb` | ~3 min | Generate 7 publication-ready charts + summary report | Creates all PNG visualizations |

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
| wisdom_score_summary.json | < 1 KB | Wisdom Score = 62.1/100 |
| 7 × PNG charts | 2.7 MB | Visualizations (300 DPI) |
| ANALYSIS_REPORT.txt | < 100 KB | Comprehensive findings |

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
→ Run the pip install cell first (each notebook has one)

**"Failed to fetch data from API"**
→ APIs might be rate-limited; wait 30 seconds and retry

**"Memory limit exceeded"**
→ Colab ran out of RAM; restart kernel and continue with next phase

**"Runtime disconnected"**
→ Colab disconnected after inactivity; reconnect and resume

## 📖 For More Details

- See `COLAB_GUIDE.md` for complete setup instructions
- Read `REPORT.md` for methodology and detailed analysis
- Check `README.md` in project root for architecture overview

---

**All notebooks are ready to run immediately in Colab!** ✅

Choose a phase above and start analyzing prediction markets for whale manipulation.
