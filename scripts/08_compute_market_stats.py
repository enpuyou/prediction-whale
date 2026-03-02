"""Compute real per-market stats from actual trade + wallet data."""
import pandas as pd
import numpy as np
import json

poly = pd.read_csv("data/processed/poly_trades_all_matched.csv")
wf   = pd.read_csv("data/processed/wallet_features.csv")

wf["is_suspicious"] = (wf["cluster"] == -1).astype(int)

merged = poly.merge(
    wf[["wallet", "cluster", "is_suspicious"]],
    left_on="proxyWallet", right_on="wallet", how="left"
)
merged["is_suspicious"] = merged["is_suspicious"].fillna(0).astype(int)
merged["size"] = merged["size"].astype(float)


def gini(x):
    v = np.array(x, dtype=float)
    v = v[v > 0]
    if len(v) == 0:
        return 0.0
    v = np.sort(v)
    n = len(v)
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


results = {}
for slug, grp in merged.groupby("eventSlug"):
    total_trades   = len(grp)
    unique_traders = grp["proxyWallet"].nunique()
    total_vol      = grp["size"].sum()

    wvol = grp.groupby("proxyWallet")["size"].sum().sort_values(ascending=False)
    gini_val    = gini(wvol.values)
    top5_share  = wvol.head(5).sum()  / total_vol * 100
    top10_share = wvol.head(10).sum() / total_vol * 100

    susp_vol     = grp.loc[grp["is_suspicious"] == 1, "size"].sum()
    susp_vol_pct = susp_vol / total_vol * 100 if total_vol > 0 else 0
    susp_wallets = int(grp.loc[grp["is_suspicious"] == 1, "proxyWallet"].nunique())

    results[slug] = {
        "total_trades":    total_trades,
        "unique_traders":  unique_traders,
        "total_volume":    round(total_vol, 0),
        "gini":            round(gini_val, 4),
        "top5_share_pct":  round(top5_share, 2),
        "top10_share_pct": round(top10_share, 2),
        "susp_vol_pct":    round(susp_vol_pct, 2),
        "susp_wallets":    susp_wallets,
    }

for slug, r in results.items():
    print(f"\n--- {slug} ---")
    for k, v in r.items():
        print(f"  {k}: {v}")

print("\n\n=== JSON ===")
print(json.dumps(results, indent=2))
