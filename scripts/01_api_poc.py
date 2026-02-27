"""
Step 1: API Proof of Concept
Validates all data sources. Run this before any other work.

GO/NO-GO gate: Polymarket trades must contain 'proxyWallet'.

Usage:
    poetry run python scripts/01_api_poc.py
"""

import json
import sys
import time
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def save(name: str, data) -> None:
    path = RAW_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    count = len(data) if isinstance(data, list) else 1
    print(f"  saved {count} record(s) → {path}")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── Test 1: Polymarket Gamma API ────────────────────────────────
section("Test 1: Polymarket Gamma API — Market Metadata")

resp = requests.get(
    "https://gamma-api.polymarket.com/markets",
    params={"active": "true", "limit": 20, "order": "volume", "ascending": "false"},
)
resp.raise_for_status()
markets = resp.json()

print(f"status:   {resp.status_code}")
print(f"markets:  {len(markets)}")
print(f"fields:   {list(markets[0].keys())[:8]}")
print("top 5:")
for m in markets[:5]:
    print(f"  [{m.get('volume', 'N/A')}] {m['question'][:70]}")
    print(f"    conditionId: {m.get('conditionId', 'MISSING')}")

save("poly_markets_top20", markets)


# ── Test 2: Polymarket Data API — Trades (GO/NO-GO gate) ────────
section("Test 2: Polymarket Data API — Trades (proxyWallet gate)")

condition_id = markets[0]["conditionId"]
print(f"market:   {markets[0]['question'][:60]}")
print(f"conditionId: {condition_id}")

resp = requests.get(
    "https://data-api.polymarket.com/trades",
    params={"market": condition_id, "limit": 10},
)
resp.raise_for_status()
trades = resp.json()

print(f"status:  {resp.status_code}")
print(f"trades:  {len(trades)}")
print(f"fields:  {list(trades[0].keys())}")

if "proxyWallet" in trades[0]:
    print(f"\n✅ GO: proxyWallet confirmed — {trades[0]['proxyWallet']}")
else:
    print("\n❌ NO-GO: proxyWallet MISSING — network analysis cannot proceed")
    print("   Fallback: Dune Analytics on-chain data")
    sys.exit(1)

save("poly_trades_sample", trades)


# ── Test 3: Polymarket Data API — Holders ───────────────────────
# Endpoint: /holders?market=<conditionId>
# Returns list of {token, holders: [{proxyWallet, asset, ...}]}
section("Test 3: Polymarket Data API — Holders per Market")

resp = requests.get(
    "https://data-api.polymarket.com/holders",
    params={"market": condition_id, "limit": 20},
)
resp.raise_for_status()
holders_resp = resp.json()

print(f"status:  {resp.status_code}")
print(f"tokens returned: {len(holders_resp)}")
for token_entry in holders_resp[:2]:
    token = token_entry.get("token", "?")
    holders = token_entry.get("holders", [])
    print(f"  token: {token[:30]}...  holders: {len(holders)}")
    if holders:
        print(f"  fields: {list(holders[0].keys())}")
        print(f"  top holder wallet: {holders[0].get('proxyWallet', 'MISSING')}")

save("poly_holders_sample", holders_resp)


# ── Test 4: Polymarket Data API — Leaderboard ──────────────────
section("Test 4: Polymarket Data API — Leaderboard")

resp = requests.get(
    "https://data-api.polymarket.com/v1/leaderboard",
    params={"timePeriod": "ALL", "orderBy": "VOL", "limit": 50},
)
resp.raise_for_status()
leaderboard = resp.json()

print(f"status:  {resp.status_code}")
records = leaderboard if isinstance(leaderboard, list) else leaderboard.get("data", [])
print(f"records: {len(records)}")
if records:
    print(f"fields:  {list(records[0].keys())}")
    print("top 3:")
    for r in records[:3]:
        print(f"  {r}")

save("poly_leaderboard_top50", leaderboard)


# ── Test 5: Kalshi API — Trades (confirm no user ID) ────────────
section("Test 5: Kalshi API — Trades (confirm no user identifier)")

resp = requests.get(
    "https://api.elections.kalshi.com/v1/trades",
    params={"limit": 10},
)
resp.raise_for_status()
kalshi_data = resp.json()
kalshi_trades = kalshi_data.get("trades", [])

print(f"status:  {resp.status_code}")
print(f"trades:  {len(kalshi_trades)}")

if kalshi_trades:
    print(f"fields:  {list(kalshi_trades[0].keys())}")
    # taker_side means buy/sell direction, not a user identity field — exclude it
    user_fields = [
        k for k in kalshi_trades[0].keys()
        if any(x in k.lower() for x in ["user", "wallet", "account", "trader"])
        or (k.lower() in ["maker", "taker"])
    ]
    if user_fields:
        print(f"⚠️  user-like fields found: {user_fields} — inspect manually")
    else:
        print("✅ confirmed: no user/wallet identifier in Kalshi trades (thesis asymmetry holds)")
    print(f"\nsample trade:\n{json.dumps(kalshi_trades[0], indent=2)}")

save("kalshi_trades_sample", kalshi_data)


# ── Test 6: Kalshi API — Market Metadata ────────────────────────
section("Test 6: Kalshi API — Market Metadata")

resp = requests.get(
    "https://api.elections.kalshi.com/trade-api/v2/markets",
    params={"limit": 10, "status": "open"},
)
resp.raise_for_status()
kalshi_market_data = resp.json()
kalshi_markets = kalshi_market_data.get("markets", [])

print(f"status:  {resp.status_code}")
print(f"markets: {len(kalshi_markets)}")
if kalshi_markets:
    print(f"fields:  {list(kalshi_markets[0].keys())}")
    print("sample markets:")
    for m in kalshi_markets[:5]:
        title = m.get("title") or m.get("question") or "N/A"
        print(f"  [{m.get('ticker')}] {title[:70]}")
        print(f"    volume={m.get('volume')} last_price={m.get('last_price')}")

save("kalshi_markets_sample", kalshi_market_data)


# ── Summary ─────────────────────────────────────────────────────
section("Summary")

saved = sorted(RAW_DIR.glob("*.json"))
print(f"saved {len(saved)} raw data files:")
for f in saved:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<40} {size_kb:.1f} KB")

print("\nAll 6 endpoints passed. Ready for Phase 2.")
