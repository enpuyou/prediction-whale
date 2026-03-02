"""Pull price and archetype breakdown per market for the demo app."""
import pandas as pd
import numpy as np

poly = pd.read_csv("data/processed/poly_trades_all_matched.csv")
wf   = pd.read_csv("data/processed/wallet_features.csv")

# Archetype mapping matching notebook logic
archetype_map = {
    5: "Buy & Hold", 4: "Casual Trader", -1: "Suspicious",
    12: "Whale", 13: "Market Maker"
}
wf["archetype"] = wf["cluster"].map(archetype_map).fillna("Other Retail")

merged = poly.merge(
    wf[["wallet", "cluster", "archetype"]],
    left_on="proxyWallet", right_on="wallet", how="left"
)
merged["size"] = merged["size"].astype(float)
merged["archetype"] = merged["archetype"].fillna("Other Retail")

# Archetype breakdown of volume per market
print("=== Archetype volume share per market ===")
for slug, grp in merged.groupby("eventSlug"):
    total = grp["size"].sum()
    a = grp.groupby("archetype")["size"].sum().sort_values(ascending=False)
    print(f"\n{slug}")
    for arch, vol in a.items():
        print(f"  {arch}: {vol/total*100:.1f}%  (vol={vol:.0f})")

# Latest price per outcome per market (last 50 trades per conditionId)
print("\n\n=== Latest prices per market (last trades) ===")
poly["timestamp"] = pd.to_datetime(poly["timestamp"], errors="coerce", utc=True)
poly["size"] = poly["size"].astype(float)

for slug, grp in poly.groupby("eventSlug"):
    print(f"\n{slug}")
    for cid, cgrp in grp.groupby("conditionId"):
        last = cgrp.sort_values("timestamp").tail(5)
        avg_price = last["price"].mean()
        outcome = last["outcome"].iloc[-1] if "outcome" in last.columns else "?"
        print(f"  {outcome[:40]:<42} last_price={avg_price:.3f}")

print("\n\n=== Price comparison CSV ===")
pc = pd.read_csv("data/processed/price_comparison.csv")
print(pc.to_string())
