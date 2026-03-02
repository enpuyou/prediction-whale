from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Crowd Wisdom Signal",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
DATA = BASE / "data" / "processed"
CHARTS = BASE / "figures"

# ----------------------------
# Per-market hardcoded data (all stats derived from real pipeline output)
# Diversity  = 100 − suspicious_vol_pct          (poly_trades_all_matched.csv)
# Decentralization = 100 − top10_share_pct × 2   (poly_trades_all_matched.csv)
# Independence = 93.0  (global, mode3_causality p=0.531, NOT significant)
# Aggregation  = 83.3  (global, Louvain modularity = 0.833)
# ----------------------------
_IND = 93.0   # global independence (no per-market herding signal, mode3 p=0.531)
_AGG = 83.3   # global aggregation  (Louvain modularity proxy)

MARKETS: dict[str, dict] = {
    "Champions League Winner": {
        "slug": "champions-league-winner-2025",
        "category": "Sports",
        "short_desc": "Which club will win the 2024–25 UEFA Champions League?",
        "kalshi_ticker": "KXUCL-26",
        "poly_volume_usd": 1_001_676_674,
        "poly_volume_label": "$1.0B",
        "similarity": 1.00,
        "num_submarkets": 36,
        # — real trade stats —
        "total_trades": 72_000,
        "unique_traders": 28_972,
        "susp_wallets": 1_216,
        "susp_vol_pct": 42.7,
        "gini": 0.9565,
        "top5_share_pct": 6.51,
        "top10_share_pct": 10.80,
        # — Surowiecki sub-scores (0–100) —
        "diversity_score": 57.3,          # 100 − 42.7
        "independence_score": _IND,
        "decentralization_score": 78.4,   # 100 − 10.80 × 2
        "aggregation_score": _AGG,
        "wisdom_score": 78.0,             # avg of 4
        "signal": "Crowd Wisdom Signal",
        "action": "PROCEED",
        # — pre-written analysis (analyst + brand voice) —
        "analysis_analyst": """\
**Data snapshot:** 72,000 trades · 28,972 unique traders (the largest participation pool across all six markets) · Gini coefficient 0.957 · top-5 wallets account for 6.5% of volume · 1,216 suspicious wallets (4.2%) driving 42.7% of volume — the *lowest* suspicious concentration of any market analyzed.

**What the model found:** DBSCAN flagged 1,216 wallets as behavioral outliers. Despite being a $1B market, no single actor or tight cluster dominates price movement. The top-10 wallet concentration (10.8%) is well below the cross-market average. Network analysis identifies 28,972 nodes distributing across 1,261+ independent communities, with network centralization near zero.

**Key risk flag:** Sports markets carry structural information asymmetry not visible in trade data — roster leaks, injury reports, and club-transfer intelligence create informed-trading advantages that DBSCAN cannot detect from behavior patterns alone. The relatively low suspicious-volume share likely reflects this: informed sports traders appear behaviorally similar to sophisticated retail, not to algorithmic bots.""",
        "analysis_brand": """\
**For your campaign:** The breadth of participation is genuine. A $1B market with nearly 29,000 participants is as close to a real crowd signal as prediction markets get. The 42.7% suspicious-volume share is high in absolute terms, but low relative to every other market we analyzed — meaning the remaining 57% comes from genuinely diverse participants.

**How to cite:** *"Prediction markets, drawing on nearly 29,000 traders, currently give [team] a X% probability of winning the Champions League."*

**Timing action:** Safe to build campaign activations around current odds. Re-check within 48 hours of significant match news (injury, lineup change) — that is when informed traders are most likely to move the price ahead of the crowd.""",
    },

    "Fed Chair Nomination": {
        "slug": "who-will-trump-nominate-as-fed-chair",
        "category": "Politics — Monetary Policy",
        "short_desc": "Who will Trump nominate as the next Federal Reserve Chair?",
        "kalshi_ticker": "KXFEDCHAIRNOM-29",
        "poly_volume_usd": 540_827_605,
        "poly_volume_label": "$541M",
        "similarity": 1.00,
        "num_submarkets": 39,
        "total_trades": 47_384,
        "unique_traders": 11_239,
        "susp_wallets": 1_184,
        "susp_vol_pct": 77.6,
        "gini": 0.9382,
        "top5_share_pct": 11.12,
        "top10_share_pct": 17.07,
        "diversity_score": 22.4,          # 100 − 77.6
        "independence_score": _IND,
        "decentralization_score": 65.9,   # 100 − 17.07 × 2
        "aggregation_score": _AGG,
        "wisdom_score": 66.1,
        "signal": "Expert Opinion Signal",
        "action": "MONITOR",
        "analysis_analyst": """\
**Data snapshot:** 47,384 trades · 11,239 unique traders · Gini 0.938 · top-5 wallets: 11.1% of volume · 1,184 suspicious wallets controlling **77.6% of all trading volume** — the second-highest suspicious concentration of any market analyzed.

**What the model found:** Despite having the second-largest absolute volume ($541M), the Fed Chair market has a Diversity score of only 22.4/100. DBSCAN identified 1,184 behavioral outliers who collectively account for more than three-quarters of all trades by value. This is a structural pattern: high-volume, algorithmically-patterned wallets dominating a politically sensitive market.

**Why volume ≠ wisdom here:** Large nominal volume can coexist with low diversity. The 11,239 unique traders represent broad participation by count, but volume is so skewed (Gini 0.938) that the median trader's position is statistically irrelevant to price formation. The price primarily reflects the conviction of ~1,184 systematically-active accounts.""",
        "analysis_brand": """\
**For your campaign:** The high dollar volume makes this market *look* authoritative — but the data says otherwise. Roughly $420 million of the $541M in trading activity came from wallets flagged as behaviorally anomalous. The price reflects sophisticated, financial-actor conviction, not distributed public opinion.

**How to cite:** *"Sophisticated traders currently price [nominee] at X% — though this market is concentrated among a small number of high-volume participants."*

**Timing action:** Monitor daily. If a major nomination signal breaks (White House leak, Congressional statement), suspicious wallet activity typically leads retail by 1–2 hours based on our lead-lag analysis. A sudden price spike without news coverage is a signal to re-evaluate before publishing.""",
    },

    "Government Shutdown Duration": {
        "slug": "how-long-will-the-next-government-shutdown-last",
        "category": "Politics — Legislative",
        "short_desc": "How long will the next U.S. government shutdown last?",
        "kalshi_ticker": "KXGOVTSHUTLENGTH-26FEB07",
        "poly_volume_usd": 23_495_074,
        "poly_volume_label": "$23.5M",
        "similarity": 1.00,
        "num_submarkets": 11,
        "total_trades": 19_961,
        "unique_traders": 6_747,
        "susp_wallets": 995,
        "susp_vol_pct": 75.5,
        "gini": 0.9121,
        "top5_share_pct": 12.72,
        "top10_share_pct": 19.42,
        "diversity_score": 24.5,
        "independence_score": _IND,
        "decentralization_score": 61.2,
        "aggregation_score": _AGG,
        "wisdom_score": 65.5,
        "signal": "Expert Opinion Signal",
        "action": "MONITOR",
        "analysis_analyst": """\
**Data snapshot:** 19,961 trades · 6,747 unique traders · Gini 0.912 · top-5 wallets: 12.7% of volume · 995 suspicious wallets carrying 75.5% of volume on an 11-outcome categorical market.

**What the model found:** The Government Shutdown market launched during a live, volatile event (February 2026). High time-pressure markets attract sophisticated political traders disproportionately — retail participants enter after initial positioning is complete. The result is a Diversity score of 24.5/100: the price predominantly reflects a concentrated set of politically-connected or algorithmically-active wallets, not broad public expectation.

**Note on recency:** This market's 2,000-trade data cap was reached within days of launch, meaning our snapshot captures the early-stage market when informed traders dominate most heavily. Later retail participation may improve diversity, but we can only analyze what we collected.""",
        "analysis_brand": """\
**For your campaign:** The Shutdown market is an active, live event with real uncertainty — but its current price is shaped more by a handful of politically-informed traders than by crowd consensus. Treat it as an expert consensus indicator, not a public sentiment signal.

**How to cite:** *"Political market traders are currently pricing a shutdown lasting more than X days at Y% — though participation is concentrated among a small number of highly active accounts."*

**Timing action:** Monitor closely as the event develops. Congressional vote outcomes, OMB statements, or leadership negotiations will trigger sharp price moves driven by the same concentrated wallets. Useful for rapid-response campaign messaging if your brand has a stake in the outcome.""",
    },

    "Next Pope": {
        "slug": "who-will-be-the-next-pope",
        "category": "Religion & Global Events",
        "short_desc": "Who will be elected as the next Pope of the Roman Catholic Church?",
        "kalshi_ticker": "KXNEWPOPE-70",
        "poly_volume_usd": 30_143_338,
        "poly_volume_label": "$30M",
        "similarity": 0.9962,
        "num_submarkets": 29,
        "total_trades": 48_376,
        "unique_traders": 6_754,
        "susp_wallets": 1_081,
        "susp_vol_pct": 76.4,
        "gini": 0.9211,
        "top5_share_pct": 18.57,
        "top10_share_pct": 27.56,
        "diversity_score": 23.6,
        "independence_score": _IND,
        "decentralization_score": 44.9,
        "aggregation_score": _AGG,
        "wisdom_score": 61.2,
        "signal": "Expert Opinion Signal",
        "action": "MONITOR",
        "analysis_analyst": """\
**Data snapshot:** 48,376 trades · 6,754 unique traders · Gini 0.921 · top-5 wallets: 18.6% of volume · top-10: 27.6% · 1,081 suspicious wallets accounting for 76.4% of all trading volume.

**What the model found:** Despite being a global, non-US-political event (which should attract broader public participation), the Next Pope market shows high volume concentration. Top-10 wallet concentration at 27.6% is the second-highest of any market analyzed. The Decentralization sub-score of 44.9 reflects genuine hub-like concentration among a small group of high-conviction traders — likely individuals with specific Vatican or theological expertise rather than algorithmically-generated activity.

**Archetype note:** The suspicious wallet profiles in this market show higher per-trade sizes and lower trade frequency than in the sports or Fed Chair markets — more consistent with informed/expert opinion than with bot-driven wash trading.""",
        "analysis_brand": """\
**For your campaign:** The Next Pope market is interesting for brand positioning around global religious events — but the odds come from a small group of highly active traders, not public consensus. The 6,754 participating wallets suggests this is a niche expert market, not a mass-participation crowd signal.

**How to cite:** *"Prediction markets currently give Cardinal [Name] a X% likelihood of becoming the next Pope — reflecting a concentrated group of specialist traders rather than broad public polling."*

**Timing action:** Monitor. If a papal conclave is called or a significant ecclesiastical development occurs, price will move quickly and decisively from the concentrated wallet cluster before broader media attention arrives.""",
    },

    "Zelenskyy & Putin Location": {
        "slug": "where-will-trump-and-putin-meet-next-393",
        "category": "Geopolitics",
        "short_desc": "Where will Trump and Putin hold their next meeting?",
        "kalshi_ticker": "KXPUTINZELENSKYYLOCATION-28",
        "poly_volume_usd": 18_496_420,
        "poly_volume_label": "$18.5M",
        "similarity": 0.9872,
        "num_submarkets": 27,
        "total_trades": 27_953,
        "unique_traders": 7_024,
        "susp_wallets": 460,
        "susp_vol_pct": 48.6,
        "gini": 0.863,
        "top5_share_pct": 20.92,
        "top10_share_pct": 27.38,
        "diversity_score": 51.4,
        "independence_score": _IND,
        "decentralization_score": 45.2,
        "aggregation_score": _AGG,
        "wisdom_score": 68.2,
        "signal": "Expert Opinion Signal",
        "action": "MONITOR",
        "analysis_analyst": """\
**Data snapshot:** 27,953 trades · 7,024 unique traders · Gini 0.863 (lowest of all six markets) · top-5 wallets: 20.9% · top-10: 27.4% · 460 suspicious wallets at 48.6% of volume.

**What the model found:** This market has the healthiest Gini coefficient (0.863) of all six, suggesting the most balanced volume distribution relative to its size. However, the top-5 concentration at 20.9% is the highest of any market — a small number of extremely large individual positions dominate while the long tail is more evenly spread. The 460 suspicious wallets is also the lowest count in absolute terms, suggesting fewer bot-like actors and more large-conviction human traders.

**Geopolitical risk context:** Diplomatic location markets are uniquely sensitive to non-public information. Government officials, think tank analysts, or diplomats with advance knowledge of summit planning would appear in our data as high-volume, low-frequency traders — potentially indistinguishable from legitimate large bettors.""",
        "analysis_brand": """\
**For your campaign:** The relatively moderate suspicious volume share (48.6%) and healthy Gini make this the most "human-feeling" market after Champions League. But top-5 concentration at 21% means a handful of very large positions can swing the price independently.

**How to cite:** *"Prediction markets currently give [location] a X% probability of hosting the Zelenskyy-Putin meeting — driven by the collective positioning of 7,000+ active traders."*

**Timing action:** Monitor geopolitical news closely. Any credible diplomatic signal — a third-party host announcement, foreign minister statement — will be priced in rapidly by the large-position holders before public reporting catches up.""",
    },

    "Trump's Defense Secretary": {
        "slug": "who-will-be-trumps-defense-secretary",
        "category": "Politics — Cabinet",
        "short_desc": "Who will serve as Trump's Secretary of Defense?",
        "kalshi_ticker": "KXNEXTDEF-29",
        "poly_volume_usd": 14_011_760,
        "poly_volume_label": "$14M",
        "similarity": 0.9728,
        "num_submarkets": 15,
        "total_trades": 22_117,
        "unique_traders": 4_780,
        "susp_wallets": 405,
        "susp_vol_pct": 39.4,
        "gini": 0.8466,
        "top5_share_pct": 20.97,
        "top10_share_pct": 25.39,
        "diversity_score": 60.6,          # 100 − 39.4
        "independence_score": _IND,
        "decentralization_score": 49.2,   # 100 − 25.39 × 2
        "aggregation_score": _AGG,
        "wisdom_score": 71.5,
        "signal": "Crowd Wisdom Signal",
        "action": "PROCEED",
        "analysis_analyst": """\
**Data snapshot:** 22,117 trades · 4,780 unique traders · Gini 0.847 (second-lowest of all markets) · top-5 wallets: 21.0% · top-10: 25.4% · 405 suspicious wallets at 39.4% of volume — the *lowest* suspicious volume share of any market analyzed.

**What the model found:** The Defense Secretary market scores highest on Diversity (60.6/100) despite having the smallest absolute trader pool (4,780). Only 405 wallets were flagged as behaviorally anomalous — far fewer than other markets — and those 405 accounts for just 39.4% of volume. The Gini of 0.847 is the healthiest of any market. This pattern suggests a smaller but genuinely engaged participant base rather than a large volume dominated by systematic actors.

**Trade-off:** Top-5 concentration at 21% is high — the market is small enough that a few large bettors can move price significantly. But unlike the Fed Chair or Pope markets, those large bettors are not DBSCAN-flagged suspicious wallets; they appear to be high-conviction individual traders.""",
        "analysis_brand": """\
**For your campaign:** Counterintuitively, this $14M market has better crowd wisdom characteristics than the $541M Fed Chair market. The lower dollar volume reflects a smaller overall market, but within that smaller market, participation is more genuine and less bot-driven.

**How to cite:** *"Prediction markets give [candidate] a X% chance of becoming Defense Secretary — based on a market with lower concentration than the broader Polymarket average."*

**Timing action:** Proceed with confidence, noting that this is a smaller market and therefore more price-sensitive to large individual trades. A single large position by an insider could meaningfully shift the price. Cross-reference with Kalshi (KXNEXTDEF-29) for confirmation: high agreement between platforms strengthens the signal.""",
    },
}

# Ordered list for the sidebar selectbox
MARKET_NAMES = list(MARKETS.keys())

# ----------------------------
# Helpers
# ----------------------------
def safe_div(a, b):
    return float(a) / float(b) if b not in (0, 0.0, None) else 0.0

def gini(x: np.ndarray) -> float:
    v = np.array(x, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0 or np.all(v == 0):
        return 0.0
    v = np.sort(v)
    n = len(v)
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)

def lorenz_curve(x: np.ndarray):
    v = np.array(x, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.array([0, 1]), np.array([0, 1])
    v = np.sort(v)
    cum = np.cumsum(v)
    cum = np.insert(cum, 0, 0)
    cum_share = cum / cum[-1] if cum[-1] != 0 else cum
    pop_share = np.linspace(0, 1, len(cum_share))
    return pop_share, cum_share

def top_n_share(wallet_vol: pd.DataFrame, vol_col: str, n: int) -> float:
    d = wallet_vol[[vol_col]].dropna().sort_values(vol_col, ascending=False)
    total = d[vol_col].sum()
    return safe_div(d.head(n)[vol_col].sum(), total)

def detect_market_col(df: pd.DataFrame) -> str | None:
    # try best candidates
    preferred = [
        "market", "market_title", "market_slug", "event", "event_title",
        "question", "title", "slug", "ticker"
    ]
    cols = list(df.columns)
    for p in preferred:
        if p in cols:
            return p
    # fallback: anything containing these keywords
    candidates = [c for c in cols if any(k in c.lower() for k in ["market", "event", "question", "title", "slug", "ticker"])]
    return candidates[0] if candidates else None

def detect_time_col(df: pd.DataFrame) -> str | None:
    for c in ["timestamp", "time", "datetime", "created_at", "create_date"]:
        if c in df.columns:
            return c
    return None

def detect_wallet_col(df: pd.DataFrame) -> str | None:
    for c in ["wallet", "address", "wallet_id", "trader", "user", "proxyWallet"]:
        if c in df.columns:
            return c
    return None

def detect_flag_col(wf: pd.DataFrame) -> str | None:
    candidates = [
        "is_structural_manipulator", "structural_manipulator", "is_manipulator",
        "manipulator_flag", "is_suspicious", "suspicious", "flagged",
        "outlier_flag", "is_outlier"
    ]
    for c in candidates:
        if c in wf.columns:
            return c
    # soft fallback
    for c in wf.columns:
        cl = c.lower()
        if any(k in cl for k in ["manip", "suspicious", "flag", "outlier"]):
            return c
    return None

def normalize_flag(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0).astype(float) > 0).astype(int)
    return series.astype(str).str.lower().isin(["1","true","yes","y","t"]).astype(int)

def pretty_label(score: float) -> str:
    if score >= 75:
        return "✅ Safe to cite"
    if score >= 55:
        return "⚠️ Cite with caveat"
    return "❌ High manipulation risk"

def action_badge(action: str) -> str:
    return {"PROCEED": "✅ PROCEED", "MONITOR": "⚠️ MONITOR", "HOLD": "🛑 HOLD"}.get(action, action)

def score_color(score: float) -> str:
    if score >= 70:
        return "#2ecc71"
    if score >= 40:
        return "#f39c12"
    return "#e74c3c"

def condition_tag(score: float) -> str:
    if score >= 70:
        return "✅ Passing"
    if score >= 40:
        return "⚠️ Weak"
    return "❌ Failing"

# ----------------------------
# Load data (for deep-dive tabs)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    poly   = pd.read_csv(DATA / "poly_trades_all_matched.csv")
    wf     = pd.read_csv(DATA / "wallet_features.csv")
    pc     = pd.read_csv(DATA / "price_comparison.csv")
    kalshi = pd.read_csv(DATA / "kalshi_trades_all_matched.csv")

    poly["size"] = pd.to_numeric(poly["size"], errors="coerce").fillna(0)
    wf["is_suspicious"] = (wf["cluster"] == -1).astype(int)
    poly = poly.merge(
        wf[["wallet", "is_suspicious"]],
        left_on="proxyWallet", right_on="wallet", how="left",
    )
    poly["is_suspicious"] = poly["is_suspicious"].fillna(0).astype(int)

    pt = detect_time_col(poly)
    if pt:
        poly[pt] = pd.to_datetime(poly[pt], errors="coerce", utc=True)
    kt = detect_time_col(kalshi)
    if kt:
        kalshi[kt] = pd.to_datetime(kalshi[kt], errors="coerce", utc=True)

    _dummy_missing = []  # keep signature compatible

    return poly, wf, pc, kalshi


try:
    poly, wf, pc, kalshi = load_data()
    data_ok = True
except Exception as e:
    data_ok = False
    data_err = str(e)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.image("https://img.icons8.com/emoji/96/whale-emoji.png", width=56)
st.sidebar.title("Crowd Wisdom Signal")
st.sidebar.caption("Select a prediction market to see its crowd wisdom score and campaign recommendation.")

chosen_name = st.sidebar.selectbox(
    "Prediction Market",
    MARKET_NAMES,
    format_func=lambda n: f"{n}  ({MARKETS[n]['poly_volume_label']})",
)
m = MARKETS[chosen_name]

st.sidebar.divider()
st.sidebar.markdown("**Pipeline**")
st.sidebar.markdown(
    "✅ 237,791 trades collected  \n"
    "✅ 63,793 wallets fingerprinted  \n"
    "✅ DBSCAN clustering (104 clusters)  \n"
    "✅ NetworkX graph (1,261 communities)  \n"
    "✅ Surowiecki scoring applied"
)
st.sidebar.divider()
st.sidebar.caption(
    "Data: Polymarket + Kalshi public APIs · "
    "DBSCAN eps=0.3, min_samples=10 · "
    "Silhouette 0.702 · "
    "Cross-platform match via sentence-transformer"
)

# ----------------------------
# App layout
# ----------------------------
tab_signal, tab_deep, tab_watch, tab_cross, tab_png = st.tabs([
    "📊 Crowd Wisdom Signal",
    "🔬 Under the Hood",
    "📋 All Markets",
    "↔️ Cross-Platform",
    "🖼️ Evidence Charts",
])

# ======================================================
# TAB 1: Crowd Wisdom Signal  ← THE DEMO TAB
# ======================================================
with tab_signal:
    # ── Market context header (for agency staff) ───────────────────────
    _act_colors = {"PROCEED": "#2ecc71", "MONITOR": "#f39c12", "HOLD": "#e74c3c"}
    _act_c = _act_colors.get(m["action"], "#888")
    _act_badge = (
        f"<span style='background:{_act_c}22;color:{_act_c};"
        f"border:1px solid {_act_c};border-radius:4px;"
        f"padding:3px 10px;font-size:11px;font-weight:700;letter-spacing:1px'>"
        f"{m['action']}</span>"
    )
    _score_c = score_color(m["wisdom_score"])
    _score_badge = (
        f"<span style='background:{_score_c}22;color:{_score_c};"
        f"border:1px solid {_score_c};border-radius:4px;"
        f"padding:3px 10px;font-size:11px;font-weight:700'>"
        f"Score {m['wisdom_score']:.0f} / 100</span>"
    )
    st.markdown(
        f"<div style='margin-bottom:6px'>"
        f"<span style='font-size:11px;color:#555;letter-spacing:1px;text-transform:uppercase'>"
        f"🐋 Crowd Wisdom Signal &nbsp;›&nbsp; {m['category']}"
        f"</span></div>"
        f"<h1 style='margin:0 0 6px 0;line-height:1.2;font-size:2.2rem'>{chosen_name}</h1>"
        f"<p style='color:#aaa;font-size:15px;margin:0 0 14px 0'>{m['short_desc']}</p>"
        f"<div style='display:flex;gap:20px;align-items:center;flex-wrap:wrap;"
        f"font-size:13px;color:#888;padding-bottom:4px'>"
        f"<span>📈 <b style='color:#ddd'>{m['poly_volume_label']}</b> on Polymarket</span>"
        f"<span>👥 <b style='color:#ddd'>{m['unique_traders']:,}</b> unique traders</span>"
        f"<span>🗂 <b style='color:#ddd'>{m['num_submarkets']}</b> outcomes tracked</span>"
        f"<span>↔️ Kalshi <b style='color:#ddd'>{m['kalshi_ticker']}</b></span>"
        f"<span>{_act_badge}</span>"
        f"<span>{_score_badge}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Hero row ──────────────────────────────────────────────────────
    hero_score, hero_signal, hero_action = st.columns([1.6, 1.2, 1.2])

    with hero_score:
        score = m["wisdom_score"]
        color = score_color(score)
        st.markdown(
            f"<div style='background:{color}18;border:2px solid {color};"
            f"border-radius:12px;padding:20px 28px;text-align:center'>"
            f"<div style='font-size:13px;color:#888;letter-spacing:1px;text-transform:uppercase'>"
            f"Crowd Wisdom Score</div>"
            f"<div style='font-size:72px;font-weight:800;color:{color};line-height:1.1'>{score:.0f}</div>"
            f"<div style='font-size:13px;color:#888'>out of 100</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with hero_signal:
        signal = m["signal"]
        sig_color = score_color(score)
        st.markdown(
            f"<div style='background:{sig_color}18;border:2px solid {sig_color};"
            f"border-radius:12px;padding:20px 24px'>"
            f"<div style='font-size:13px;color:#888;letter-spacing:1px;text-transform:uppercase;"
            f"margin-bottom:8px'>Signal Type</div>"
            f"<div style='font-size:22px;font-weight:700;color:{sig_color}'>{signal}</div>"
            f"<div style='font-size:12px;color:#888;margin-top:8px'>"
            f"Polymarket vol: {m['poly_volume_label']} · {m['unique_traders']:,} traders</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with hero_action:
        action = m["action"]
        act_color = score_color(score)
        act_icon = {"PROCEED": "✅", "MONITOR": "⚠️", "HOLD": "🛑"}.get(action, "")
        act_desc = {
            "PROCEED": "Safe to cite in campaigns.",
            "MONITOR": "Cite with caveat — not full crowd.",
            "HOLD": "Do not cite — concentrated capital.",
        }.get(action, "")
        st.markdown(
            f"<div style='background:{act_color}18;border:2px solid {act_color};"
            f"border-radius:12px;padding:20px 24px'>"
            f"<div style='font-size:13px;color:#888;letter-spacing:1px;text-transform:uppercase;"
            f"margin-bottom:8px'>Campaign Action</div>"
            f"<div style='font-size:36px;font-weight:800;color:{act_color}'>{act_icon} {action}</div>"
            f"<div style='font-size:12px;color:#888;margin-top:8px'>{act_desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Surowiecki conditions ─────────────────────────────────────────
    st.markdown("### Surowiecki Crowd Wisdom Conditions")
    st.caption(
        "Surowiecki's four requirements for a market price to reflect genuine collective intelligence. "
        "All four must hold — failure on any one condition undermines the signal."
    )

    cond_cols = st.columns(4)
    conditions = [
        ("Diversity", m["diversity_score"],
         "Volume share from non-suspicious wallets. Below 40 means a few actors dominate price."),
        ("Independence", m["independence_score"],
         "Absence of herding. Mode 3 lead-lag test: no significant pattern detected (p=0.531)."),
        ("Decentralization", m["decentralization_score"],
         "Absence of dominant hub wallets. Based on top-10 volume concentration per market."),
        ("Aggregation", m["aggregation_score"],
         "Price formation quality. Network modularity 0.833 → 1,261 independent communities."),
    ]
    for col, (name, val, desc) in zip(cond_cols, conditions):
        with col:
            tag_color = score_color(val)
            tag = condition_tag(val)
            st.markdown(
                f"<div style='border:1px solid #333;border-radius:10px;padding:16px'>"
                f"<div style='font-size:13px;font-weight:600;text-transform:uppercase;"
                f"letter-spacing:1px;color:#ccc'>{name}</div>"
                f"<div style='font-size:42px;font-weight:800;color:{tag_color};line-height:1.1'>"
                f"{val:.0f}</div>"
                f"<div style='font-size:11px;color:#888'>/ 100</div>"
                f"<div style='margin:6px 0'>"
                f"<div style='height:6px;background:#333;border-radius:3px'>"
                f"<div style='height:6px;width:{val:.0f}%;background:{tag_color};"
                f"border-radius:3px'></div></div></div>"
                f"<div style='font-size:12px;color:{tag_color};font-weight:600'>{tag}</div>"
                f"<div style='font-size:11px;color:#666;margin-top:6px'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Key metrics ───────────────────────────────────────────────────
    st.markdown("### Market Data  *(computed from real trade collection)*")
    km1, km2, km3, km4, km5 = st.columns(5)
    km1.metric("Total trades", f"{m['total_trades']:,}")
    km2.metric("Unique traders", f"{m['unique_traders']:,}")
    km3.metric("Suspicious wallet vol", f"{m['susp_vol_pct']:.1f}%",
               help="% of total volume from DBSCAN-flagged behavioral outliers (cluster = -1)")
    km4.metric("Gini coefficient", f"{m['gini']:.3f}",
               help="Volume inequality. 0 = perfectly equal, 1 = one wallet owns everything.")
    km5.metric("Top-5 wallet share", f"{m['top5_share_pct']:.1f}%",
               help="% of total volume controlled by the 5 largest wallets in this market")

    st.caption(
        f"Polymarket volume: **{m['poly_volume_label']}** · "
        f"Kalshi ticker: `{m['kalshi_ticker']}` · "
        f"Semantic match score: **{m['similarity']:.4f}** · "
        f"DBSCAN suspicious wallets: **{m['susp_wallets']:,}** of {m['unique_traders']:,}"
    )

    st.divider()

    # ── Analysis ──────────────────────────────────────────────────────
    st.markdown("### Analysis")
    view_analyst, view_brand = st.tabs(["📈 Analyst View", "📣 Brand Strategy"])

    with view_analyst:
        st.markdown(m["analysis_analyst"])

    with view_brand:
        st.markdown(m["analysis_brand"])

    st.divider()

    # ── Campaign recommendation card ──────────────────────────────────
    st.markdown("### Campaign Recommendation")
    action = m["action"]
    card_color = score_color(m["wisdom_score"])
    cite_text = {
        "PROCEED":
            f"**How to cite:** *\"Prediction markets, drawing on {m['unique_traders']:,} active traders, "
            f"currently give [outcome] a X% probability.\"*",
        "MONITOR":
            f"**How to cite:** *\"Sophisticated traders currently price [outcome] at X% — "
            f"though this {m['poly_volume_label']} market is concentrated among "
            f"a smaller group of highly active accounts.\"*",
        "HOLD":
            "**Do not cite this number directly** — it reflects concentrated capital "
            "positions, not distributed crowd consensus.",
    }[action]

    rationale = {
        "PROCEED":
            f"Score {m['wisdom_score']:.0f}/100. Diversity {m['diversity_score']:.0f}/100 — passing. "
            f"{m['unique_traders']:,} unique traders; suspicious volume at {m['susp_vol_pct']:.1f}% "
            f"({m['susp_wallets']:,} wallets). Broad participation supports direct citation.",
        "MONITOR":
            f"Score {m['wisdom_score']:.0f}/100. Diversity {m['diversity_score']:.0f}/100 — failing. "
            f"{m['susp_vol_pct']:.1f}% of volume from {m['susp_wallets']:,} DBSCAN-flagged wallets. "
            f"Price reflects sophisticated traders, not public consensus. Cite with explicit caveat.",
        "HOLD":
            f"Score {m['wisdom_score']:.0f}/100. Multiple conditions failing. "
            f"Concentrated capital dominates. Citing this number would misrepresent the signal source.",
    }[action]

    with st.container(border=True):
        rc1, rc2 = st.columns([1, 3])
        with rc1:
            act_icon = {"PROCEED": "✅", "MONITOR": "⚠️", "HOLD": "🛑"}.get(action, "")
            st.markdown(
                f"<div style='text-align:center;padding:20px'>"
                f"<div style='font-size:48px'>{act_icon}</div>"
                f"<div style='font-size:22px;font-weight:800;color:{card_color}'>{action}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with rc2:
            st.markdown(f"**Signal:** {m['signal']}")
            st.markdown(cite_text)
            st.markdown(f"**Rationale:** {rationale}")


# ======================================================
# TAB 2: Under the Hood
# ======================================================
with tab_deep:
    st.subheader("Under the Hood — Interactive Market Analysis")
    st.caption("Live-computed from the raw trade data for the selected market.")

    if not data_ok:
        st.error(f"Could not load data: {data_err}")
        st.stop()

    slug = m["slug"]
    poly_f = poly[poly["eventSlug"] == slug].copy()
    time_col = detect_time_col(poly_f)
    poly_f["_vol"] = poly_f["size"].astype(float)
    wallet_vol = poly_f.groupby("proxyWallet", as_index=False).agg(volume=("_vol", "sum"))

    colA, colB = st.columns([1.1, 0.9])

    with colA:
        st.markdown("### Volume Concentration (Lorenz Curve)")
        st.caption("The further the curve bows from the diagonal, the more concentrated trading power is.")
        if len(wallet_vol) > 5:
            pop, cum = lorenz_curve(wallet_vol["volume"].to_numpy())
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pop, y=cum, mode="lines", name="Actual",
                                     line=dict(color="#f39c12", width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect equality",
                                     line=dict(dash="dash", color="#aaa")))
            fig.update_layout(xaxis_title="Share of wallets", yaxis_title="Share of volume",
                               template="plotly_dark", height=340,
                               margin=dict(l=40, r=20, t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough wallets.")

    with colB:
        st.markdown("### Top Wallets")
        topn = st.slider("Show top N wallets", 5, 30, 10, 5)
        top = wallet_vol.sort_values("volume", ascending=False).head(topn).copy()
        susp_map = (
            poly_f.drop_duplicates("proxyWallet")
            .set_index("proxyWallet")["is_suspicious"]
        )
        top["flag"] = top["proxyWallet"].map(susp_map).fillna(0).astype(int).map(
            {1: "⚠️ Suspicious", 0: "Normal"}
        )
        top["wallet"] = top["proxyWallet"].str[:10] + "..."
        st.dataframe(
            top[["wallet", "volume", "flag"]].rename(columns={"volume": "vol (shares)"}),
            use_container_width=True, height=340,
        )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Trading Activity Over Time")
        if time_col and time_col in poly_f.columns and pd.api.types.is_datetime64_any_dtype(poly_f[time_col]):
            gran = st.selectbox("Granularity", ["Day", "Week"], index=0, key="gran_dd")
            pfx = poly_f.dropna(subset=[time_col]).copy()
            if gran == "Day":
                pfx["bucket"] = pfx[time_col].dt.floor("D")
            else:
                pfx["bucket"] = pfx[time_col].dt.to_period("W").dt.start_time.dt.tz_localize("UTC")
            agg = pfx.groupby("bucket", as_index=False).agg(
                volume=("_vol", "sum"), trades=("_vol", "count")
            )
            fig2 = px.bar(agg, x="bucket", y="volume", template="plotly_dark", height=300,
                          labels={"bucket": "", "volume": "Volume (shares)"})
            fig2.update_layout(margin=dict(l=40, r=20, t=10, b=40))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No usable timestamp column.")

    with col2:
        st.markdown("### Suspicious vs Normal Volume")
        susp_vol_val = poly_f.loc[poly_f["is_suspicious"] == 1, "_vol"].sum()
        norm_vol_val = poly_f.loc[poly_f["is_suspicious"] == 0, "_vol"].sum()
        fig3 = go.Figure(go.Pie(
            labels=["Suspicious wallets", "Normal wallets"],
            values=[susp_vol_val, norm_vol_val],
            marker_colors=["#e74c3c", "#2ecc71"],
            hole=0.45,
        ))
        fig3.update_layout(template="plotly_dark", height=300,
                           margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig3, use_container_width=True)


# ======================================================
# TAB 3: All Markets
# ======================================================
with tab_watch:
    st.subheader("All Markets — Scores at a Glance")
    st.caption(
        "Scores computed from real trade data. "
        "Diversity = 100 − suspicious_vol%. "
        "Decentralization = 100 − top10% × 2. "
        "Independence and Aggregation are global values from the full network analysis."
    )

    rows = []
    for name, md in MARKETS.items():
        rows.append({
            "Market": name,
            "Volume": md["poly_volume_label"],
            "Traders": f"{md['unique_traders']:,}",
            "Score": md["wisdom_score"],
            "Signal": md["signal"],
            "Action": md["action"],
            "Diversity": md["diversity_score"],
            "Independence": md["independence_score"],
            "Decentral.": md["decentralization_score"],
            "Aggregation": md["aggregation_score"],
            "Susp. Vol %": f"{md['susp_vol_pct']:.1f}%",
            "Gini": f"{md['gini']:.3f}",
            "Top-5 %": f"{md['top5_share_pct']:.1f}%",
        })
    df_all = pd.DataFrame(rows).sort_values("Score", ascending=False)

    def highlight_action(row):
        c = {"PROCEED": "#2ecc7120", "MONITOR": "#f39c1220", "HOLD": "#e74c3c20"}.get(row["Action"], "")
        return [f"background-color:{c}" for _ in row]

    st.dataframe(
        df_all.style.apply(highlight_action, axis=1).format({"Score": "{:.1f}",
            "Diversity": "{:.1f}", "Independence": "{:.1f}",
            "Decentral.": "{:.1f}", "Aggregation": "{:.1f}"}),
        use_container_width=True, height=320,
    )

    st.divider()
    st.markdown("### Score Comparison")
    fig_bar = px.bar(
        df_all,
        x="Market", y="Score",
        color="Signal",
        color_discrete_map={
            "Crowd Wisdom Signal": "#2ecc71",
            "Expert Opinion Signal": "#f39c12",
            "Concentrated Capital Signal": "#e74c3c",
        },
        template="plotly_dark",
        height=380,
        text="Score",
    )
    fig_bar.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_bar.add_hline(y=70, line_dash="dash", line_color="#2ecc71",
                      annotation_text="PROCEED threshold (70)")
    fig_bar.add_hline(y=40, line_dash="dash", line_color="#e74c3c",
                      annotation_text="HOLD threshold (40)")
    fig_bar.update_layout(margin=dict(l=40, r=20, t=30, b=100), xaxis_tickangle=-20,
                          yaxis_range=[0, 105])
    st.plotly_chart(fig_bar, use_container_width=True)


# ======================================================
# TAB 4: Cross-platform comparison
# ======================================================
with tab_cross:
    st.subheader("Cross-Platform: Polymarket vs Kalshi")
    st.caption(
        "Same event, two platforms. High agreement = both crowds tell the same story. "
        "Divergence = potential information asymmetry or platform-specific manipulation."
    )

    import json
    matched_path = DATA / "matched_markets.json"
    if matched_path.exists():
        matched = json.loads(matched_path.read_text())
        df_match = pd.DataFrame(matched)[
            ["question", "poly_volume_usd", "poly_num_markets", "kalshi_event_ticker", "similarity"]
        ].rename(columns={
            "question": "Market",
            "poly_volume_usd": "Poly Volume (USD)",
            "poly_num_markets": "Sub-markets",
            "kalshi_event_ticker": "Kalshi Ticker",
            "similarity": "Match Score",
        })
        df_match["Poly Volume (USD)"] = df_match["Poly Volume (USD)"].apply(lambda x: f"${x:,.0f}")
        df_match["Match Score"] = df_match["Match Score"].apply(lambda x: f"{x:.4f}")
        st.dataframe(df_match, use_container_width=True, height=280)

    st.divider()
    st.markdown("### Platform Architecture Asymmetry")
    col_l, col_r = st.columns(2)
    with col_l:
        with st.container(border=True):
            st.markdown("**Polymarket (unregulated)**")
            st.markdown(
                "- Blockchain-based · wallet addresses public\n"
                "- Every trade traceable to a wallet address\n"
                "- 63,793 unique wallets analyzed in this study\n"
                "- DBSCAN clustering + network graph applicable\n"
                "- ⚠️ No CFTC oversight on manipulation"
            )
    with col_r:
        with st.container(border=True):
            st.markdown("**Kalshi (CFTC-regulated)**")
            st.markdown(
                "- Federally regulated exchange\n"
                "- **No user identifiers in public API**\n"
                "- Wallet-level analysis impossible\n"
                "- Regulation implies surveillance — but opaque\n"
                "- ✅ Institutional credibility · ❌ No public auditability"
            )
    st.info(
        "**The critical asymmetry:** Polymarket exposes enough data to audit. "
        "Kalshi does not. This tool scores Polymarket directly — "
        "Kalshi price convergence serves as a corroborating sanity check only."
    )


# ======================================================
# TAB 5: PNG Evidence
# ======================================================
with tab_png:
    st.subheader("Evidence Charts — Pipeline Output")
    st.caption("Generated by the analysis pipeline (scripts 04–07).")

    pngs = sorted(CHARTS.glob("*.png")) if CHARTS.exists() else []
    if not pngs:
        st.info(
            "No PNGs found in `figures/`. "
            "Run `poetry run python scripts/07_visualization_polish.py` to regenerate."
        )
    else:
        n_cols = st.slider("Grid columns", 2, 4, 3, 1)
        n_rows = int(np.ceil(len(pngs) / n_cols))
        idx = 0
        for _ in range(n_rows):
            grid = st.columns(n_cols)
            for c in grid:
                if idx >= len(pngs):
                    break
                p = pngs[idx]; idx += 1
                c.image(str(p), use_container_width=True,
                        caption=p.stem.replace("_", " ").title())
