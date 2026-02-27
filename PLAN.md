# Prediction Market Integrity Auditor — Project Plan

## Cross-Platform Wisdom of Crowds Analysis: Polymarket vs. Kalshi

**MSIS 521 A (Gold) | Winter 2026 | Team Project**

---

## 1. One-Liner

A Python tool that takes any prediction market, quantifies whether its price reflects genuine crowd wisdom or whale manipulation, and exposes how both regulated (Kalshi) and unregulated (Polymarket) platforms fail the "wisdom of crowds" test in structurally different ways.

---

## 2. Problem & Client

**Client:** A hypothetical financial media organization (or consumer protection agency) that cites prediction market odds in reporting and needs to verify whether those odds are trustworthy before publishing them.

**Problem:** Journalists and analysts increasingly cite Polymarket/Kalshi odds as objective probabilities ("markets give X a 73% chance"). But nobody has a tool to check whether that 73% was set by 50,000 independent thinkers or by 3 wallets with $10M each. Our tool answers: **"Should you trust this number?"**

---

## 3. Course Material Alignment

| Course Topic | Session | How Our Project Maps |
|---|---|---|
| Web Analytics / Google Trends | 01 | Volume trend analysis across platforms over time |
| Text Representation / NLP | 02 | Sentence embeddings for cross-platform market matching |
| Transformers / BERT | 03 | Hugging Face sentence-transformer for semantic similarity; media framing analysis (backup component) |
| Image / Pattern Recognition | 04 | Feature engineering on raw trade data, anomaly/pattern detection |
| SEM / Social Media / UGC | 05 | Platform marketing narrative vs. data reality; "prediction laundering" in media |
| Social Networks / Wisdom of Crowds | 06 | **Core framework:** NetworkX graph analysis, Surowiecki's four conditions, crowd wisdom scoring |

---

## 4. ML/AI Components

### 4.1 Primary ML Component — Unsupervised Wallet Behavior Classification (DBSCAN)

**What it does:** Classifies every Polymarket wallet in a given market into behavioral archetypes — not through manual rules, but through unsupervised clustering on engineered features. This is the difference between saying "these 5 wallets are big" (descriptive) and "a model identified these wallets as behaviorally anomalous" (analytical).

**Why DBSCAN over K-means:**

- Does not require specifying number of clusters in advance (we don't know how many behavioral types exist)
- Naturally identifies outliers as "noise" points — these are your suspicious wallets
- Finds clusters of arbitrary shape (whale behavior doesn't form neat spheres in feature space)

**Features per wallet (engineered from raw trade data):**

| Feature | What It Captures |
|---|---|
| `total_volume` | Scale of activity |
| `num_trades` | Frequency of participation |
| `avg_trade_size` | Typical bet size |
| `trade_frequency_per_hour` | How actively they trade |
| `buy_sell_ratio` | Directional bias vs. market-making |
| `timing_entropy` | Whether trades are spread across hours (high entropy = organic) or clustered (low entropy = bot/coordinated) |
| `num_markets_active` | Specialist vs. platform-wide player |
| `portfolio_value` | Total capital on platform |
| `round_trip_frequency` | Buy then sell same asset within 5-min window — wash trading signal |
| `pct_volume_in_market` | Concentration of market-moving power |
| `avg_time_before_resolution` | Trades close to resolution = possible informed trading |
| `win_rate` | Accuracy on resolved markets |

**Expected clusters:**

- **Market Makers:** High volume, high frequency, balanced buy/sell, many markets, tight spread capture
- **Whales / Informed Traders:** Large trades, few markets, trades cluster near resolution, high win rate
- **Retail:** Small trades, few markets, irregular timing, average win rate
- **Suspicious / Coordinated:** Flagged as DBSCAN outliers OR separate dense cluster. High round-trip frequency, repeated counterparties, volume that doesn't move price

**Validation:** Cross-reference DBSCAN cluster labels against the NetworkX graph communities. If the model's "suspicious" cluster overlaps with tightly-connected graph communities (wallets that trade heavily with each other), that's converging evidence from two independent methods.

### 4.2 Supporting ML Component — Sentence Transformer for Market Matching

**What it does:** Automatically matches equivalent markets across Polymarket and Kalshi using semantic similarity rather than manual lookup.

**Model:** `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face. Pre-trained, no fine-tuning needed. Encodes text into 384-dimensional vectors; cosine similarity measures semantic equivalence.

**Why include this:** Manual matching works for 3–5 markets. But using a transformer demonstrates Session 03 knowledge, makes the tool genuinely reusable, and is a clean 30 lines of code that adds real functionality.

**Implementation:**
```
1. Pull all market titles from both platforms
2. Encode with sentence-transformer
3. Compute pairwise cosine similarity
4. Return matches above 0.75 threshold
5. Manual confirmation step (human in the loop)
```

### 4.3 Backup/Bonus Component — Media Citation Framing Analysis (NLP)

**Only pursue this if ahead of schedule.** This connects the on-platform manipulation to how it gets laundered into public discourse.

**What it does:** Scrape news articles/headlines that mention Polymarket or Kalshi odds. Run them through a Hugging Face zero-shot classifier (e.g., `facebook/bart-large-mnli`) to categorize how the odds are framed:

- **Unqualified fact:** "Markets give Trump a 65% chance"
- **Hedged speculation:** "Betting markets suggest roughly 65% odds"
- **Contextualized:** "Polymarket, where a small number of large traders dominate volume, currently prices Trump at 65%"

Then cross-reference: when an article cites a specific number, was that number recently moved by a whale trade? If yes, that's a documented instance of "prediction laundering" — a whale's capital-driven bet getting stripped of context and presented as collective intelligence.

**Why it's backup:** This requires web scraping news articles, which adds scope and potential fragility. The core project is strong without it. But if you have time, it transforms the presentation from "here's what's happening on the platforms" to "here's how it reaches the public."

---

## 5. Data Sources

All data comes from **public APIs requiring zero authentication.**

| Source | Base URL | Auth | Key Data |
|---|---|---|---|
| Polymarket Gamma API | `gamma-api.polymarket.com` | None | Market metadata, volume, liquidity, events |
| Polymarket Data API | `data-api.polymarket.com` | None | Trades **(with wallet addresses)**, positions, top holders, leaderboard |
| Polymarket CLOB (read) | `clob.polymarket.com` | None | Order books, prices, spreads |
| Kalshi Trade API | `api.elections.kalshi.com/trade-api/v2` | None (read) | Market metadata, trades **(NO user IDs)**, candlesticks |
| Hugging Face | Local model download | None | `sentence-transformers/all-MiniLM-L6-v2` |

**The critical asymmetry:** Polymarket exposes wallet addresses per trade → enables network analysis. Kalshi exposes nothing about who trades → impossible to audit. This asymmetry IS the thesis.

---

## 6. Step-by-Step Build Plan

### Step 1: Environment Setup & API Proof of Concept — Day 1

**Goal:** Validate every data source works before committing to the project. This is the single most important day — if something fails here, you pivot early.

**Actions:**

1. Create Python environment:
   ```
   pip install requests pandas numpy networkx scikit-learn sentence-transformers matplotlib seaborn
   ```

2. Test Polymarket Gamma API (market metadata):
   ```python
   import requests
   resp = requests.get("https://gamma-api.polymarket.com/markets", params={
       "active": "true", "limit": 20, "order": "volume", "ascending": "false"
   })
   markets = resp.json()
   for m in markets:
       print(m["question"], m.get("volume"), m["conditionId"])
   ```

3. Test Polymarket Data API (trades with wallet IDs — **this is the critical test**):
   ```python
   condition_id = markets[0]["conditionId"]
   resp = requests.get("https://data-api.polymarket.com/trades", params={
       "market": condition_id, "limit": 10
   })
   trades = resp.json()
   # CONFIRM: does each trade have a "proxyWallet" field?
   print(trades[0].keys())
   print(trades[0]["proxyWallet"])
   ```

4. Test Polymarket top holders:
   ```python
   resp = requests.get("https://data-api.polymarket.com/top-holders", params={
       "conditionIds": condition_id, "limit": 20
   })
   ```

5. Test Polymarket leaderboard:
   ```python
   resp = requests.get("https://data-api.polymarket.com/v1/leaderboard", params={
       "timePeriod": "ALL", "orderBy": "VOL", "limit": 50
   })
   ```

6. Test Kalshi API (trades without user IDs):
   ```python
   resp = requests.get("https://api.elections.kalshi.com/trade-api/v2/trades", params={
       "limit": 10
   })
   kalshi_trades = resp.json()["trades"]
   print(kalshi_trades[0].keys())
   # CONFIRM: no user/wallet identifier present
   ```

7. Save all raw responses as JSON for offline dev.

**Deliverable:** Jupyter notebook with all 6 API calls succeeding. Screenshot of Polymarket trade showing `proxyWallet` field. If any endpoint returns unexpected data, document the issue.

**Go/No-Go Decision:** If `proxyWallet` is not in Polymarket trade data, the network analysis component doesn't work. Fallback: use on-chain data from Polygon via Dune Analytics (free tier). But based on current docs, `proxyWallet` is confirmed present.

---

### Step 2: Market Selection & Transformer-Based Matching — Day 2

**Goal:** Select 3–5 markets that exist on both platforms using the sentence transformer.

**Actions:**

1. Pull all active + recently closed markets from Polymarket:
   ```python
   all_poly_markets = []
   offset = 0
   while True:
       resp = requests.get("https://gamma-api.polymarket.com/markets", params={
           "limit": 100, "offset": offset
       })
       batch = resp.json()
       if not batch:
           break
       all_poly_markets.extend(batch)
       offset += len(batch)
   ```

2. Pull all Kalshi markets:
   ```python
   all_kalshi_markets = []
   cursor = None
   while True:
       params = {"limit": 1000}
       if cursor:
           params["cursor"] = cursor
       resp = requests.get(
           "https://api.elections.kalshi.com/trade-api/v2/markets", params=params
       )
       data = resp.json()
       all_kalshi_markets.extend(data["markets"])
       cursor = data.get("cursor")
       if not cursor:
           break
   ```

3. Load sentence transformer and compute embeddings:
   ```python
   from sentence_transformers import SentenceTransformer
   from sklearn.metrics.pairwise import cosine_similarity

   model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

   poly_titles = [m["question"] for m in all_poly_markets]
   kalshi_titles = [m["title"] for m in all_kalshi_markets]

   poly_embeddings = model.encode(poly_titles)
   kalshi_embeddings = model.encode(kalshi_titles)

   sim_matrix = cosine_similarity(poly_embeddings, kalshi_embeddings)
   ```

4. Extract top matches:
   ```python
   import numpy as np

   for i in range(len(poly_titles)):
       best_j = np.argmax(sim_matrix[i])
       score = sim_matrix[i][best_j]
       if score > 0.75:
           print(f"MATCH ({score:.2f}): Poly: {poly_titles[i]}")
           print(f"              Kalshi: {kalshi_titles[best_j]}")
           print()
   ```

5. Manually review matches. Select 3–5 high-volume markets. Prefer: Fed rate decisions, major political outcomes, economic indicators.

6. Create a mapping file:
   ```python
   matched_markets = [
       {"poly_conditionId": "0x...", "kalshi_ticker": "KXFED...", "question": "Will the Fed cut rates in March?"},
       # ...
   ]
   ```

**Deliverable:** `matched_markets.json` with confirmed pairs. Notebook section showing transformer similarity scores.

---

### Step 3: Full Data Collection — Days 2–3

**Goal:** Pull complete trade histories and wallet profiles for all selected markets.

**Actions:**

1. **Polymarket trades** — for each matched market, paginate through all trades:
   ```python
   import time

   def get_all_poly_trades(condition_id):
       trades = []
       offset = 0
       while True:
           resp = requests.get("https://data-api.polymarket.com/trades", params={
               "market": condition_id, "limit": 100, "offset": offset
           })
           if resp.status_code == 429:
               time.sleep(2)
               continue
           batch = resp.json()
           if not batch:
               break
           trades.extend(batch)
           offset += len(batch)
           time.sleep(0.5)
       return trades
   ```

2. **Polymarket top holders** per market (max 20 per token).

3. **Polymarket leaderboard** — top 50 by volume (ALL time).

4. **Polymarket wallet enrichment** — for each unique wallet above $100 volume threshold:
   ```python
   def get_wallet_profile(wallet):
       positions = requests.get("https://data-api.polymarket.com/positions",
                                params={"user": wallet}).json()
       value = requests.get("https://data-api.polymarket.com/value",
                            params={"user": wallet}).json()
       return {
           "num_markets": len(positions),
           "total_value": value[0]["value"] if value else 0,
       }
   ```

5. **Kalshi trades** — for each matched market:
   ```python
   def get_all_kalshi_trades(ticker):
       trades = []
       cursor = None
       while True:
           params = {"ticker": ticker, "limit": 1000}
           if cursor:
               params["cursor"] = cursor
           resp = requests.get(
               "https://api.elections.kalshi.com/trade-api/v2/trades", params=params
           )
           data = resp.json()
           trades.extend(data["trades"])
           cursor = data.get("cursor")
           if not cursor:
               break
           time.sleep(0.5)
       return trades
   ```

6. Save everything as CSV/JSON files.

**Deliverable:** Complete data files. Summary table:

| Market | Poly Trades | Poly Unique Wallets | Kalshi Trades | Date Range |
|---|---|---|---|---|
| Fed March Rate | ? | ? | ? | ? |
| ... | ... | ... | ... | ... |

---

### Step 4: Feature Engineering & DBSCAN Clustering — Days 3–4

**Goal:** Build wallet feature vectors and run unsupervised classification.

**Actions:**

1. Create per-wallet feature DataFrame from Polymarket trade data:

   ```python
   import pandas as pd
   import numpy as np
   from scipy.stats import entropy

   df = pd.DataFrame(all_trades)
   df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
   df["hour"] = df["timestamp"].dt.hour

   wallet_features = []
   for wallet, wdf in df.groupby("proxyWallet"):
       hourly_dist = wdf["hour"].value_counts(normalize=True).reindex(range(24), fill_value=0)

       # Round-trip detection: same wallet buys then sells same asset within 5 min
       wdf_sorted = wdf.sort_values("timestamp")
       round_trips = 0
       # (simplified — compare consecutive opposite-side trades on same asset)

       wallet_features.append({
           "wallet": wallet,
           "total_volume": wdf["size"].sum(),
           "num_trades": len(wdf),
           "avg_trade_size": wdf["size"].mean(),
           "trade_freq_per_hour": len(wdf) / max((wdf["timestamp"].max() - wdf["timestamp"].min()).total_seconds() / 3600, 1),
           "buy_sell_ratio": (wdf["side"] == "BUY").mean(),
           "timing_entropy": entropy(hourly_dist),
           "num_markets": wallet_profiles.get(wallet, {}).get("num_markets", 1),
           "portfolio_value": wallet_profiles.get(wallet, {}).get("total_value", 0),
           "pct_volume": wdf["size"].sum() / df["size"].sum(),
           "round_trip_freq": round_trips / max(len(wdf), 1),
       })

   features_df = pd.DataFrame(wallet_features)
   ```

2. Standardize and cluster:

   ```python
   from sklearn.preprocessing import StandardScaler
   from sklearn.cluster import DBSCAN
   from sklearn.metrics import silhouette_score

   feature_cols = [c for c in features_df.columns if c != "wallet"]
   X = StandardScaler().fit_transform(features_df[feature_cols])

   # Grid search eps and min_samples
   best_score = -1
   for eps in [0.3, 0.5, 0.7, 1.0, 1.5]:
       for min_samples in [3, 5, 10]:
           db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
           labels = db.labels_
           if len(set(labels)) > 1 and -1 in labels:
               score = silhouette_score(X[labels != -1], labels[labels != -1])
               if score > best_score:
                   best_score = score
                   best_params = (eps, min_samples)
                   best_labels = labels

   features_df["cluster"] = best_labels
   ```

3. Label clusters by inspecting centroids:
   ```python
   # Examine mean feature values per cluster
   features_df.groupby("cluster")[feature_cols].mean()
   # Map cluster IDs to human labels based on feature patterns
   ```

4. Visualize with PCA or t-SNE:
   ```python
   from sklearn.decomposition import PCA

   pca = PCA(n_components=2)
   coords = pca.fit_transform(X)
   features_df["pca_x"] = coords[:, 0]
   features_df["pca_y"] = coords[:, 1]

   # Scatter plot colored by cluster
   ```

**Deliverable:** Feature matrix. Cluster assignments. PCA/t-SNE scatter plot. Table of cluster centroids with interpretations.

---

### Step 5: Network Graph & Wisdom Score — Days 4–5

**Goal:** Construct the wallet interaction graph and compute the composite integrity metric.

**Actions:**

1. Build the NetworkX graph:

   ```python
   import networkx as nx
   from itertools import combinations

   G = nx.Graph()

   # Add nodes with attributes
   for _, row in features_df.iterrows():
       G.add_node(row["wallet"],
                  volume=row["total_volume"],
                  cluster=row["cluster"],
                  portfolio_value=row["portfolio_value"])

   # Add edges: wallets trading same side in same 1-hour window
   df["time_bin"] = df["timestamp"].dt.floor("1h")
   for (time_bin, side), group in df.groupby(["time_bin", "side"]):
       wallets_in_bin = group["proxyWallet"].unique()
       if len(wallets_in_bin) < 2:
           continue
       for w1, w2 in combinations(wallets_in_bin, 2):
           if G.has_edge(w1, w2):
               G[w1][w2]["weight"] += 1
           else:
               G.add_edge(w1, w2, weight=1)
   ```

2. Compute metrics:

   ```python
   # Volume concentration
   wallet_volumes = df.groupby("proxyWallet")["size"].sum().sort_values(ascending=False)
   top5_share = wallet_volumes.head(5).sum() / wallet_volumes.sum()

   # Gini coefficient
   def gini(values):
       sorted_vals = np.sort(values)
       n = len(sorted_vals)
       index = np.arange(1, n + 1)
       return (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n

   gini_coeff = gini(wallet_volumes.values)

   # Network centralization (Freeman)
   dc = nx.degree_centrality(G)
   max_dc = max(dc.values())
   centralization = sum(max_dc - v for v in dc.values()) / ((len(G)-1) * (len(G)-2))

   # Community detection
   communities = nx.community.louvain_communities(G, weight="weight")
   modularity = nx.community.modularity(G, communities)

   # Composite Wisdom Score (0 = whale control, 100 = true crowd)
   wisdom_score = 100 * (1 - (
       0.30 * top5_share +
       0.30 * gini_coeff +
       0.20 * centralization +
       0.20 * (1 - modularity)
   ))
   ```

3. Cross-validate clusters vs. graph communities:
   ```python
   # For each Louvain community, check what % of wallets are in DBSCAN's "suspicious" cluster
   # High overlap = strong converging evidence
   ```

**Deliverable:** NetworkX graph. Wisdom Score per market. Metrics summary table. Graph community → DBSCAN cluster overlap analysis.

---

### Step 6: Cross-Platform Comparison — Day 5

**Goal:** Compare matched markets between Polymarket and Kalshi.

**Actions:**

1. Align timestamps and resample both platforms' trades to hourly price series:
   ```python
   # Polymarket: volume-weighted average price per hour
   poly_hourly = poly_df.groupby(poly_df["timestamp"].dt.floor("1h")).apply(
       lambda x: np.average(x["price"], weights=x["size"])
   )

   # Kalshi: volume-weighted average price per hour
   kalshi_hourly = kalshi_df.groupby(kalshi_df["created_time"].dt.floor("1h")).apply(
       lambda x: np.average(x["yes_price"], weights=x["count"])
   )
   ```

2. Plot price divergence time series (overlay both platforms).

3. Trade-size distribution histograms side by side.

4. Volume-by-hour heatmaps (when does each platform trade most?).

5. Compute price correlation and identify divergence windows (>5% gap).

6. For divergence windows: check if Polymarket whale cluster was disproportionately active during those periods.

**Deliverable:** Comparison charts. Divergence table with whale activity flags.

---

### Step 7: Visualization Polish — Days 5–6

**Goal:** Create presentation-ready outputs.

**Core visualizations (6 charts minimum):**

1. **Network graph** — force-directed layout. Nodes sized by volume. Colors: red = DBSCAN suspicious/outlier, orange = whale/informed, blue = market maker, gray = retail. This is the demo showpiece.

2. **Wisdom Score cards** — one per market. Show composite score + four sub-components as a bar/radar chart.

3. **Volume concentration curve** — cumulative % of volume by top-N wallets. Classic Pareto visualization.

4. **DBSCAN cluster scatter** — 2D PCA/t-SNE projection colored by cluster label.

5. **Cross-platform price divergence** — time series overlay with shaded divergence windows.

6. **Trade-size histograms** — Polymarket vs. Kalshi side by side for same market.

**Optional (if time permits):**

7. Animated network graph showing how trading relationships evolve over time.
8. Heatmap of whale activity by hour-of-day and day-of-week.

---

### Step 8: Presentation — Days 6–7

**15-minute structure:**

| Time | Section | Content |
|---|---|---|
| 0:00–2:00 | **Hook** | "Polymarket says X has a 73% chance. Bloomberg printed it. CNN cited it. But who is the crowd?" |
| 2:00–4:00 | **The Theory** | Wisdom of crowds (Session 06): diversity, independence, decentralization, aggregation. Both platforms claim to satisfy all four. |
| 4:00–7:00 | **The Tool** | Architecture walkthrough: API data → transformer matching → feature engineering → DBSCAN clustering → NetworkX graph → Wisdom Score. Show the pipeline, explain design choices. |
| 7:00–11:00 | **Demo & Findings** | Live or recorded. Show: (1) network graph with whale nodes lit up, (2) cluster scatter plot, (3) wisdom scores across markets, (4) cross-platform divergence charts. |
| 11:00–13:00 | **The Punchline** | The transparency paradox. Polymarket shows you the manipulation but nobody builds tools to interpret it. Kalshi hides it by regulation. Neither platform is incentivized to build this tool because it would undermine their core product — the appearance of crowd wisdom. |
| 13:00–15:00 | **So What** | Consumer protection implications. Media literacy. Regulatory gaps. What a trustworthy prediction market would look like. |

---

## 7. Risk Registry

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| API rate limiting blocks data collection | Medium | High | Add sleep(0.5) between requests. Exponential backoff on 429s. Cache aggressively. Fallback: Dune Analytics free tier for Polymarket on-chain data. |
| `proxyWallet` field missing from trade data | Low | Critical | Confirmed present in Polymarket docs and third-party implementations. Day 1 proof-of-concept validates this. |
| Too few unique wallets for meaningful graph | Medium | Medium | Pre-screen markets for wallet diversity in Step 1. Select high-volume markets with 100+ unique wallets. |
| DBSCAN finds only 1 cluster or all noise | Medium | Medium | Grid search eps/min_samples. If DBSCAN fails, fall back to K-means with k=4 (less elegant but functional). |
| No overlapping markets between platforms | Low | High | Fed rate, election, and economic markets exist on both platforms. Transformer matching in Step 2 will surface them. |
| Sentence-transformer model too large to run locally | Low | Low | all-MiniLM-L6-v2 is 80MB — runs on any laptop. No GPU needed. |
| Teammate scope disagreements | Medium | Medium | Project is modular — data collection, ML/clustering, network analysis, visualization, and presentation can be owned by different people. |

---

## 8. Team Division of Labor (Suggested)

| Track | Owner | Steps |
|---|---|---|
| **Data Engineering** | Person A | Steps 1, 3 — API calls, pagination, data cleaning, storage |
| **ML / Analytics** | Person B | Steps 2, 4 — Transformer matching, feature engineering, DBSCAN clustering |
| **Network Analysis & Scoring** | Person C | Steps 5, 6 — NetworkX graph, wisdom score, cross-platform comparison |
| **Visualization & Presentation** | All | Steps 7, 8 — Charts, deck, narrative (everyone contributes) |

---

## 9. Tech Stack Summary

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| Environment | Jupyter Notebook (per course requirement) |
| Data collection | `requests` |
| Data manipulation | `pandas`, `numpy` |
| ML — clustering | `scikit-learn` (DBSCAN, StandardScaler, PCA, silhouette_score) |
| ML — text matching | `sentence-transformers` (Hugging Face) |
| Network analysis | `networkx` |
| Visualization | `matplotlib`, `seaborn` |
| Optional NLP backup | `transformers` (Hugging Face zero-shot classifier) |

---

## 10. Deliverables Checklist

- [ ] Python code (Jupyter Notebook) — submitted before Session 07
- [ ] Presentation deck (PowerPoint/PDF) — submitted before Session 07
- [ ] Live or recorded demo — presented during Session 07
- [ ] AI Disclosure statement (per syllabus GenAI policy)

---

## 11. AI Disclosure (Draft)

*Per the course GenAI policy, we used Claude Opus 4 to assist with: brainstorming project direction, researching API documentation for Polymarket and Kalshi, structuring the project plan, and debugging Python code during development. All analytical decisions (market selection, feature engineering choices, clustering parameter tuning, wisdom score formula design, and interpretation of results) were made by the team. We validated all AI-assisted code by running it against live API data and verifying outputs manually.*
