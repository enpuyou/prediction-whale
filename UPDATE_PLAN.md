Here are the discrete tasks, written so you can paste them directly into Claude Code one at a time:

---

## Task 1: Reframe the Wisdom Score Methodology

**Paste this:**

> In our existing Wisdom Score formula, we currently compute a single score that implies whale-dominated markets are "less accurate." Refactor the scoring to instead measure the four Surowiecki crowd wisdom conditions explicitly and independently. The score should not claim to measure accuracy — it should measure whether the market is functioning as a crowd wisdom mechanism. The four sub-scores should be:
>
> - **Diversity Score** (0–100): What fraction of total volume comes from wallets outside the top 10? High diversity = broad participation.
> - **Independence Score** (0–100): Inverse of the DBSCAN coordination signal. High round-trip frequency and co-movement between wallets lowers this score.
> - **Decentralization Score** (0–100): Inverse of network centralization from NetworkX. Measures whether the graph has dominant hub nodes.
> - **Aggregation Score** (0–100): How well does the price track external signals? Use Pearson correlation between weekly price movement and Google Trends volume for the market keyword. High correlation = price is aggregating real public information.
>
> The composite Wisdom of Crowds Mechanism Score is the equally-weighted average of all four. Add a label to the output: if the score is above 70 print "Crowd Wisdom Signal", between 40–70 print "Expert Opinion Signal", below 40 print "Concentrated Capital Signal." These labels are more accurate and more interesting than a simple high/low score.

---

## Task 2: Add Google Trends Integration

**Paste this:**

> Add a new module to the notebook that pulls Google Trends data for each matched market using the `pytrends` library. For each market, define a search keyword (e.g., "Federal Reserve rate cut" for a Fed market). Pull weekly search interest for the same date range as the Polymarket trade data. Then compute two things:
>
> 1. Pearson correlation between weekly price change on Polymarket and weekly Google Trends volume change. Store this as `trends_correlation` and use it as the Aggregation Score in Task 1.
>
> 2. A dual-axis time series chart per market: Polymarket price on the left axis, Google Trends volume on the right axis, with DBSCAN-flagged whale-activity time windows shaded in red. Title the chart "[Market Name]: Prediction Market Price vs. Public Search Interest."
>
> If trends_correlation is below 0.3 and there are active whale windows in the same period, add a flag: "Price movement decoupled from public interest during whale activity."

---

## Task 3: Separate Informed Whale from Manipulative Whale in DBSCAN Output

**Paste this:**

> Our current DBSCAN clustering labels large wallets generically as "whales." Refactor the cluster labeling logic to distinguish two types of large trader based on feature patterns:
>
> - **Informed Trader:** High `win_rate` (above 0.6), high `avg_time_before_resolution` is LOW (trades close to resolution date), low `round_trip_frequency`, active in few markets (`num_markets` below 5). These wallets may be moving the market because they have better information.
>
> - **Structural Manipulator:** High `round_trip_frequency`, low `win_rate`, trades spread across many markets, high `pct_volume_in_market`. These wallets are moving the market through capital pressure not information.
>
> Update the network graph visualization to use different colors: orange for Informed Trader, red for Structural Manipulator, blue for Market Maker, gray for Retail. Add a summary line to the output: "X wallets identified as potentially informed traders. Y wallets flagged as structural manipulators."

---

## Task 4: Add Campaign Timing Recommendation Output

**Paste this:**

> After all scores are computed, add a final function called `generate_marketing_recommendation()` that takes the market name, Wisdom of Crowds Mechanism Score, the dominant wallet type from Task 3, trends_correlation, and days until market resolution as inputs. It should return a plain-English marketing recommendation in three parts:
>
> 1. **Signal Type:** One of "Crowd Wisdom Signal", "Expert Opinion Signal", or "Concentrated Capital Signal" from Task 1 — with a one-sentence explanation of what that means for a brand strategist.
>
> 2. **Citation Guidance:** How a marketing team should characterize this number if they publish it. If Score >= 70, suggest: "Prediction markets give X a Y% probability." If Score 40–70, suggest: "Sophisticated traders currently price X at Y%." If Score < 40, suggest: "Do not cite this number directly — it reflects concentrated capital positions, not crowd consensus."
>
> 3. **Campaign Timing Action:** One of PROCEED / MONITOR / HOLD with a one-sentence rationale that references the specific metrics driving the decision.
>
> Format the output as a printed card that could be shown during the demo, not just a dictionary.

---

## Task 5: Historical Validation — Run Tool on 2024 US Election Market

**Paste this:**

> Add a section to the notebook titled "Historical Validation." Pull historical trade data from Polymarket for the 2024 US Presidential Election market (search Gamma API for the relevant conditionId). Run the full pipeline — DBSCAN clustering, NetworkX graph, Wisdom of Crowds Mechanism Score, and campaign timing recommendation — on this historical data.
>
> The goal is to show two things side by side:
> 1. The Wisdom Score was low (whale-dominated, independence condition failed)
> 2. The final price was accurate — Trump did win
>
> Add a text cell below the output that explicitly states: "This market was dominated by a small number of large traders, meaning it was not functioning as a crowd wisdom mechanism. However, the final probability estimate proved accurate — suggesting the large traders may have been informed rather than manipulative. Our tool correctly identifies the mechanism (not crowd wisdom) without incorrectly claiming the price was wrong."
>
> This is the intellectual honesty slide in the presentation — it actually strengthens the argument rather than weakening it.

---

## Task 6: Update All Chart Titles and Labels to Marketing Language

**Paste this:**

> Go through all existing visualizations and update titles, axis labels, and legends to use marketing-facing language rather than data science language. Specifically:
>
> - Network graph title: "Who Is Actually Moving This Market?" not "Wallet Interaction Network"
> - Cluster scatter plot title: "Trader Behavior Archetypes" with legend labels: "Informed Traders", "Structural Manipulators", "Market Makers", "Retail Participants"
> - Volume concentration curve title: "How Concentrated Is the Trading Power?" with a callout annotation marking the top 5 wallet threshold
> - Cross-platform comparison title: "Do Polymarket and Kalshi Tell the Same Story?"
> - Google Trends overlay title: "[Market]: Is Price Tracking Real Public Interest?"
>
> The goal is that every chart should be readable by a CMO with no data science background in under 10 seconds.

---

## Sequence to Follow

Do them in order — Tasks 1 and 3 are the conceptually important ones that change how the tool thinks about whales. Task 2 adds the Google Trends data. Task 4 produces the marketing output. Task 5 is the historical case study that validates everything. Task 6 is polish.

Tasks 1, 3, and 4 together are the core of what makes this intellectually defensible and marketingfacing at the same time. If you run short on time, those three plus Task 6 are the minimum set.
