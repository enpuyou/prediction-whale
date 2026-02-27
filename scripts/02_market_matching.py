"""
Step 2: Market Matching via Sentence Transformer
Fetches EVENTS from both Polymarket and Kalshi, encodes titles with
all-MiniLM-L6-v2, and finds cross-platform pairs by cosine similarity.

Key insight: Both platforms have an event layer that groups binary
sub-markets into categorical questions. Matching at the event level
(not individual market level) produces semantically correct pairs.

Usage:
    poetry run python scripts/02_market_matching.py

Outputs:
    data/raw/poly_all_events.json
    data/raw/kalshi_all_events.json
    data/processed/candidate_matches.json
"""

import json
import time
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SIMILARITY_THRESHOLD = 0.75


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ── 1. Fetch Polymarket EVENTS (not individual markets) ──────────
# The /events endpoint returns categorical event groupings.
# Each event has a title like "Who will Trump nominate as Fed Chair?"
# and contains multiple binary sub-markets.
# We filter to volume > $10k to keep meaningful events only.
section("1. Fetching Polymarket events (volume > $10k)")

poly_events = []
offset = 0
MAX_POLY = 2000
while len(poly_events) < MAX_POLY:
    resp = requests.get(
        "https://gamma-api.polymarket.com/events",
        params={
            "limit": 100,
            "offset": offset,
            "active": "true",
            "order": "volume",
            "ascending": "false",
        },
    )
    resp.raise_for_status()
    batch = resp.json()
    if not batch:
        break
    # Filter by volume
    batch = [e for e in batch if to_float(e.get("volume") or 0) > 10_000]
    if not batch:
        break  # sorted by volume desc — stop when below threshold
    poly_events.extend(batch)
    offset += 100
    print(f"  fetched {len(poly_events)} qualifying events...", end="\r")
    time.sleep(0.3)

print(f"\nTotal Polymarket events (vol > $10k): {len(poly_events)}")

# Save lightweight version (strip sub-markets to save disk)
poly_events_light = []
for e in poly_events:
    poly_events_light.append({
        "id": e.get("id"),
        "title": e.get("title"),
        "slug": e.get("slug"),
        "volume": e.get("volume"),
        "num_markets": len(e.get("markets", [])),
        "market_condition_ids": [
            m.get("conditionId") for m in e.get("markets", [])
        ],
        "active": e.get("active"),
        "closed": e.get("closed"),
    })
(RAW_DIR / "poly_all_events.json").write_text(
    json.dumps(poly_events_light, indent=2)
)
print("  saved → data/raw/poly_all_events.json")


# ── 2. Fetch all Kalshi events ───────────────────────────────────
section("2. Fetching all Kalshi events (paginated)")

kalshi_events = []
cursor = None
while True:
    params = {"limit": 200, "status": "open"}
    if cursor:
        params["cursor"] = cursor
    resp = requests.get(
        "https://api.elections.kalshi.com/trade-api/v2/events",
        params=params,
    )
    resp.raise_for_status()
    data = resp.json()
    batch = data.get("events", [])
    if not batch:
        break
    kalshi_events.extend(batch)
    cursor = data.get("cursor")
    print(f"  fetched {len(kalshi_events)} events...", end="\r")
    if not cursor:
        break
    time.sleep(0.3)

print(f"\nTotal Kalshi events: {len(kalshi_events)}")
(RAW_DIR / "kalshi_all_events.json").write_text(
    json.dumps(kalshi_events, indent=2)
)
print("  saved → data/raw/kalshi_all_events.json")


# ── 3. Encode with sentence transformer ─────────────────────────
section("3. Encoding event titles with all-MiniLM-L6-v2")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

poly_titles = [e.get("title", "") for e in poly_events]
kalshi_titles = [e.get("title", "") for e in kalshi_events]

print(f"Encoding {len(poly_titles)} Polymarket event titles...")
poly_embeddings = model.encode(poly_titles, show_progress_bar=True)

print(f"Encoding {len(kalshi_titles)} Kalshi event titles...")
kalshi_embeddings = model.encode(kalshi_titles, show_progress_bar=True)

sim_matrix = cosine_similarity(poly_embeddings, kalshi_embeddings)
print(f"Similarity matrix shape: {sim_matrix.shape}")


# ── 4. Extract top matches ───────────────────────────────────────
section(f"4. Extracting matches (threshold={SIMILARITY_THRESHOLD})")

matches = []
for i in range(len(poly_titles)):
    best_j = int(np.argmax(sim_matrix[i]))
    score = float(sim_matrix[i][best_j])
    if score >= SIMILARITY_THRESHOLD:
        pe = poly_events[i]
        ke = kalshi_events[best_j]
        matches.append({
            "similarity": round(score, 4),
            "poly_title": poly_titles[i],
            "kalshi_title": kalshi_titles[best_j],
            "poly_event_id": pe.get("id"),
            "poly_slug": pe.get("slug"),
            "poly_volume": pe.get("volume"),
            "poly_num_markets": len(pe.get("markets", [])),
            "poly_condition_ids": [
                m.get("conditionId") for m in pe.get("markets", [])
            ],
            "kalshi_event_ticker": ke.get("event_ticker"),
            "kalshi_category": ke.get("category"),
        })

# Sort by similarity descending
matches.sort(key=lambda x: x["similarity"], reverse=True)

print(f"Found {len(matches)} matches above {SIMILARITY_THRESHOLD} threshold")
(PROCESSED_DIR / "candidate_matches.json").write_text(
    json.dumps(matches, indent=2)
)
print(f"All {len(matches)} candidates saved → data/processed/candidate_matches.json")


# ── 5. Show results by quality tiers ─────────────────────────────
section("5. Match quality breakdown")

tier_exact = [m for m in matches if m["similarity"] >= 0.95]
tier_high = [m for m in matches if 0.85 <= m["similarity"] < 0.95]
tier_mid = [m for m in matches if 0.75 <= m["similarity"] < 0.85]

print(f"  Exact matches  (>=0.95): {len(tier_exact)}")
print(f"  High matches (0.85-0.95): {len(tier_high)}")
print(f"  Mid matches  (0.75-0.85): {len(tier_mid)}")

if tier_exact:
    print("\n── Exact matches (>=0.95) ──")
    for m in tier_exact[:20]:
        print(f"  [{m['similarity']:.4f}] POLY: {m['poly_title'][:60]}")
        print(f"           KALSHI: {m['kalshi_title'][:60]}")
        print(f"           poly_vol=${to_float(m['poly_volume']):,.0f}  sub-markets={m['poly_num_markets']}")
        print()

if tier_high:
    print("\n── High matches (0.85-0.95) ──")
    for m in tier_high[:20]:
        print(f"  [{m['similarity']:.4f}] POLY: {m['poly_title'][:60]}")
        print(f"           KALSHI: {m['kalshi_title'][:60]}")
        print(f"           poly_vol=${to_float(m['poly_volume']):,.0f}  sub-markets={m['poly_num_markets']}")
        print()

# High-volume candidates
section("6. High-volume cross-platform candidates")

high_vol = [
    m for m in matches
    if to_float(m["poly_volume"]) > 100_000
]
high_vol.sort(key=lambda x: to_float(x["poly_volume"]), reverse=True)

print(f"Matches with Polymarket volume > $100k: {len(high_vol)}")
print("\nTop 20 by volume:")
for m in high_vol[:20]:
    print(f"  [{m['similarity']:.2f} | ${to_float(m['poly_volume']):>14,.0f}] {m['poly_title'][:55]}")
    print(f"    → {m['kalshi_title'][:70]}")

print("\nReview the matches above, then manually confirm 3-5 pairs.")
print("Edit data/processed/matched_markets.json with your final selections.")
