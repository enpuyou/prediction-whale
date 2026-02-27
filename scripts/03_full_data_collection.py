"""
Step 3: Full Data Collection for All Matched Markets (Multi-threaded)

Collects recent trade histories (last 2000 trades per market) in parallel
for all 6 selected cross-platform market pairs.

Usage:
    poetry run python scripts/03_full_data_collection.py

Outputs:
    data/raw/kalshi_trades_KXFEDCHAIRNOM-29.json
    data/raw/kalshi_trades_KXGOVTSHUTLENGTH-26FEB07.json
    ... (one per matched market)
    data/processed/poly_trades_all_matched.csv
    data/processed/kalshi_trades_all_matched.csv
"""

import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MAX_TRADES_PER_MARKET = 2000
POLYMARKET_BATCH_SIZE = 100
KALSHI_BATCH_SIZE = 1000
REQUEST_DELAY = 0.1
MAX_WORKERS = 4


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def load_matched_markets():
    with open("data/processed/matched_markets.json") as f:
        return json.load(f)


def fetch_kalshi_market_trades(ticker: str, max_trades: int = MAX_TRADES_PER_MARKET):
    """Fetch trades for a single Kalshi market."""
    trades = []
    cursor = None
    trade_count = 0

    while trade_count < max_trades:
        params = {
            "ticker": ticker,
            "limit": min(KALSHI_BATCH_SIZE, max_trades - trade_count),
        }
        if cursor:
            params["cursor"] = cursor
        try:
            resp = requests.get(
                "https://api.elections.kalshi.com/v1/trades",
                params=params,
                timeout=10,
            )
            if resp.status_code == 429:
                time.sleep(2)
                continue
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("trades", [])
            if not batch:
                break
            trades.extend(batch)
            trade_count += len(batch)
            cursor = data.get("cursor")
            if not cursor or trade_count >= max_trades:
                break
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"  ERROR on {ticker}: {e}")
            break

    return ticker, trades


def fetch_polymarket_condition_trades(cond_id: str, max_trades: int = MAX_TRADES_PER_MARKET):
    """Fetch trades for a single Polymarket condition."""
    trades = []
    offset = 0
    trade_count = 0

    while trade_count < max_trades:
        try:
            resp = requests.get(
                "https://data-api.polymarket.com/trades",
                params={
                    "market": cond_id,
                    "limit": POLYMARKET_BATCH_SIZE,
                    "offset": offset,
                },
                timeout=10,
            )
            if resp.status_code == 429:
                time.sleep(2)
                continue
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            trades.extend(batch)
            trade_count += len(batch)
            offset += len(batch)
            if trade_count >= max_trades:
                break
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"  ERROR on {cond_id}: {e}")
            break

    return cond_id, trades


# ── 1. Collect Kalshi trades (parallelized) ──────────────────────────
section("1. Collecting Kalshi trades (multi-threaded)")

matched = load_matched_markets()
kalshi_events = json.load(open("data/raw/kalshi_all_events.json"))
all_kalshi_trades = []

for market_idx, m in enumerate(matched, 1):
    kalshi_ticker = m.get("kalshi_event_ticker")
    question = m["question"]
    print(f"\n[{market_idx}/{len(matched)}] {question[:60]}")
    print(f"  Event: {kalshi_ticker}")

    # Find event to get its sub-market tickers
    event = next(
        (e for e in kalshi_events if e.get("event_ticker") == kalshi_ticker), None
    )
    if not event:
        print(f"  WARNING: Event {kalshi_ticker} not found in raw data")
        continue

    # Fetch all markets under this event
    markets = []
    cursor = None
    while True:
        params = {"event_ticker": kalshi_ticker, "limit": 200, "status": "open"}
        if cursor:
            params["cursor"] = cursor
        try:
            resp = requests.get(
                "https://api.elections.kalshi.com/trade-api/v2/markets",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("markets", [])
            if not batch:
                break
            markets.extend(batch)
            cursor = data.get("cursor")
            if not cursor:
                break
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"  ERROR fetching markets: {e}")
            break

    print(f"  {len(markets)} markets. Fetching trades in parallel...")

    # Fetch trades for all markets in parallel
    market_trades = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_kalshi_market_trades, m["ticker"]): m["ticker"]
            for m in markets
        }
        for future in as_completed(futures):
            ticker, trades = future.result()
            market_trades.extend(trades)
            print(f"    {ticker}: {len(trades)} trades")

    print(f"  Total trades: {len(market_trades)}")
    all_kalshi_trades.extend(market_trades)

    # Save per-event file
    fname = RAW_DIR / f"kalshi_trades_{kalshi_ticker}.json"
    fname.write_text(json.dumps(market_trades, indent=2))
    print(f"  saved → {fname.name}")

print(f"\nTotal Kalshi trades: {len(all_kalshi_trades)}")


# ── 2. Collect Polymarket trades (parallelized) ──────────────────────
section("2. Collecting Polymarket trades (multi-threaded)")

all_poly_trades = []
poly_events = json.load(open("data/raw/poly_all_events.json"))

for market_idx, m in enumerate(matched, 1):
    poly_event_id = str(m.get("poly_event_id"))
    question = m["question"]
    print(f"\n[{market_idx}/{len(matched)}] {question[:60]}")

    # Find event in raw data to get its sub-market conditionIds
    event = next((e for e in poly_events if str(e["id"]) == poly_event_id), None)
    if not event:
        print(f"  WARNING: Event {poly_event_id} not found in raw data")
        continue

    condition_ids = event["market_condition_ids"]
    print(f"  {len(condition_ids)} sub-markets. Fetching trades in parallel...")

    # Fetch trades for all conditions in parallel
    market_trades = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_polymarket_condition_trades, cid): cid
            for cid in condition_ids
        }
        for future in as_completed(futures):
            cid, trades = future.result()
            market_trades.extend(trades)
            print(f"    {cid[:16]}...: {len(trades)} trades")

    print(f"  Total trades: {len(market_trades)}")
    all_poly_trades.extend(market_trades)

    # Save per-event file
    fname = RAW_DIR / f"poly_trades_{poly_event_id}.json"
    fname.write_text(json.dumps(market_trades, indent=2))
    print(f"  saved → {fname.name}")

print(f"\nTotal Polymarket trades: {len(all_poly_trades)}")


# ── 3. Convert to CSVs for analysis ──────────────────────────────────
section("3. Converting to CSV format")

# Polymarket trades CSV
if all_poly_trades:
    poly_df = pd.DataFrame(all_poly_trades)
    # Keep key fields
    keep_cols = [
        "proxyWallet",
        "side",
        "price",
        "size",
        "timestamp",
        "conditionId",
        "eventSlug",
        "outcome",
    ]
    poly_df = poly_df[[c for c in keep_cols if c in poly_df.columns]]
    poly_df["timestamp"] = pd.to_datetime(poly_df["timestamp"], unit="s")
    poly_df = poly_df.sort_values("timestamp")
    poly_df.to_csv(PROCESSED_DIR / "poly_trades_all_matched.csv", index=False)
    print(f"Polymarket: {len(poly_df)} trades → poly_trades_all_matched.csv")

# Kalshi trades CSV
if all_kalshi_trades:
    kalshi_df = pd.DataFrame(all_kalshi_trades)
    # Keep key fields
    keep_cols = ["ticker", "price", "price_dollars", "count", "create_date", "taker_side"]
    kalshi_df = kalshi_df[[c for c in keep_cols if c in kalshi_df.columns]]
    kalshi_df["create_date"] = pd.to_datetime(kalshi_df["create_date"])
    kalshi_df = kalshi_df.sort_values("create_date")
    kalshi_df.to_csv(PROCESSED_DIR / "kalshi_trades_all_matched.csv", index=False)
    print(f"Kalshi: {len(kalshi_df)} trades → kalshi_trades_all_matched.csv")


# ── 4. Summary statistics ──────────────────────────────────────────
section("4. Summary Statistics")

print(f"\nPolymarket:")
print(f"  Total trades: {len(all_poly_trades):,}")
if all_poly_trades:
    poly_df = pd.DataFrame(all_poly_trades)
    print(f"  Unique wallets: {poly_df['proxyWallet'].nunique():,}")
    print(f"  Price range: {poly_df['price'].min():.4f} - {poly_df['price'].max():.4f}")
    print(f"  Total volume (shares): {poly_df['size'].sum():,.0f}")
    print(
        f"  Date range: {poly_df['timestamp'].min()} to {poly_df['timestamp'].max()}"
    )

print(f"\nKalshi:")
print(f"  Total trades: {len(all_kalshi_trades):,}")
if all_kalshi_trades:
    kalshi_df = pd.DataFrame(all_kalshi_trades)
    print(f"  Unique markets: {kalshi_df['ticker'].nunique()}")
    print(f"  Price range: {kalshi_df['price'].min()} - {kalshi_df['price'].max()}")
    print(f"  Total volume (contracts): {kalshi_df['count'].sum():,.0f}")
    print(
        f"  Date range: {kalshi_df['create_date'].min()} to {kalshi_df['create_date'].max()}"
    )

print("\nPhase 3 complete. Ready for Phase 4 (Feature Engineering & DBSCAN).")
