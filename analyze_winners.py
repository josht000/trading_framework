"""
Winning-trade analysis: what do the profitable trades have in common, and how clustered / cyclical /
market-driven are they?

  python analyze_winners.py            # build reports/trades_enriched.parquet (cached) and write reports/winning_trades_report.md
  python analyze_winners.py --rebuild  # rebuild the enriched table first

Universe: every executed-candidate trade of the base strategy (pm_target25_v2 exits, 13-month atr_pivot, no entry filters),
dev and the locked test period together, labelled by `period`. This is descriptive analysis; predictive models are
trained on dev only and scored on test.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.backtest.market_regime import attach_regime
from src.backtest.pit_metrics import dedupe_trades
from src.utils import timing

TAG = "lb13"
VARIANT = "pm_target25_v2"
OUT = "reports"
ENRICHED = os.path.join(OUT, "trades_enriched.parquet")
NY = "America/New_York"

CATEGORIES = {
    "earnings": ("earnings", "eps", "revenue", "quarter", "q1 ", "q2 ", "q3 ", "q4 ", "results", "guidance", "outlook"),
    "contract_partnership": ("contract", "award", "order", "agreement", "partnership", "collaborat", "selected", "supply", "deal"),
    "clinical_fda": ("fda", "clinical", "trial", "phase 1", "phase 2", "phase 3", "approval", "clearance", "patent"),
    "offering_dilution": ("offering", "private placement", "registered direct", "at-the-market", "atm ", "dilut", "warrant",
                          "reverse split", "shelf"),
    "m_and_a": ("acquire", "acquisition", "merger", "buyout", "tender offer", "to be acquired", "takeover", "strategic alternatives"),
    "analyst": ("upgrade", "downgrade", "price target", "initiates", "analyst"),
    "theme_hype": ("artificial intelligence", " ai ", "quantum", "bitcoin", "crypto", "drone", "defense", "rare earth",
                   "greenland", "uranium", "nuclear", "gold", "oil ", "lithium", "space", "robot"),
}
GENERIC = ("stocks moving", "stocks are moving", "movers", "stocks hitting", "stocks that hit", "shares are trading",
           "trading higher", "trading lower", "session", "mid-day", "pre-market", "premarket", "after-market",
           "moving in", "earnings scheduled", "analyzing", "short interest", "analyst ratings", "options activity",
           "unusual options", "nasdaq", "dow ", "s&p 500", "futures", "52-week")
SECTOR_MAP = {  # yfinance sector -> SimFin sector
    "Technology": "Technology", "Healthcare": "Healthcare", "Financial Services": "Financial Services",
    "Consumer Cyclical": "Consumer Cyclical", "Consumer Defensive": "Consumer Defensive", "Energy": "Energy",
    "Industrials": "Industrials", "Basic Materials": "Basic Materials", "Real Estate": "Real Estate",
    "Utilities": "Utilities", "Communication Services": "Communication Services",
}


def load_trades():
    with timing.stage("load trade inputs"):
        parts = []
        for holdout, period in ((False, "dev"), (True, "test")):
            p = os.path.join("results", "holdout" if holdout else "", f"pit_trades_atr_pivot_{TAG}_{VARIANT}.csv")
            t = pd.read_csv(p, keep_default_na=False, na_values=[""])
            t["period"] = period
            parts.append(t)
        t = pd.concat(parts, ignore_index=True)
        t["entry_time"] = t["entry_time"].astype(str)
        t = dedupe_trades(t).copy()
        t["day"] = pd.to_datetime(t["entry_time"].str[:10])
    with timing.stage("regime attach"):
        return attach_regime(t.reset_index(drop=True))


def add_market_overnight(t):
    """Next-session close-to-open gap and entry-day close-to-close return for SPY / IWM / QQQ (analysis only, uses the future)."""
    for sym in ("SPY", "IWM", "QQQ"):
        d = pd.read_parquet(os.path.join("data", "pit", "market", f"{sym}_daily.parquet")).sort_values("date").set_index("date")
        gap = d["open"].shift(-1) / d["close"] - 1
        t[f"{sym.lower()}_gap_next"] = gap.reindex(t["day"]).to_numpy()
        t[f"{sym.lower()}_cc_next"] = (d["close"].shift(-1) / d["close"] - 1).reindex(t["day"]).to_numpy()
    return t


def add_simfin(t):
    """Sector / industry, point-in-time shares outstanding and market cap, and earnings publish dates from SimFin."""
    import simfin as sf
    from src.data.simfin_store import SimFinStore

    store = SimFinStore()
    comp = store.load_companies()
    ind = sf.load_industries()
    info = comp[["IndustryId"]].join(ind, on="IndustryId")
    info = info[~info.index.duplicated(keep="last")]
    t["sector"] = t["symbol"].map(info["Sector"])
    t["industry"] = t["symbol"].map(info["Industry"])
    sp = store.load_shareprices()
    sp.index = sp.index.set_levels(pd.to_datetime(sp.index.levels[1]).normalize(), level=1)
    inc = store.load_income()
    shares, mcap, earn_next, earn_prev = [], [], [], []
    by_sym = {s: g for s, g in t.groupby("symbol")}
    pub = {}
    for s in by_sym:
        try:
            x = inc.xs(s, level="Ticker")
            pub[s] = pd.to_datetime(x["Publish Date"]).dropna().sort_values().to_numpy()
        except KeyError:
            pub[s] = np.array([], dtype="datetime64[ns]")
    out = pd.DataFrame(index=t.index, columns=["shares_pit", "mcap_pit", "earn_next_0_3d", "earn_prev_1_3d", "has_earn_data"], dtype=float)
    for s, g in by_sym.items():
        try:
            d = sp.xs(s, level="Ticker")
        except KeyError:
            d = None
        for i, r in g.iterrows():
            if d is not None:
                prior = d.loc[:r["day"]]
                if len(prior):
                    out.at[i, "shares_pit"] = prior["Shares Outstanding"].iloc[-1]
                    out.at[i, "mcap_pit"] = prior["Shares Outstanding"].iloc[-1] * prior["Close"].iloc[-1]
            p = pub[s]
            out.at[i, "has_earn_data"] = float(len(p) > 0)
            if len(p):
                day = np.datetime64(r["day"])
                out.at[i, "earn_next_0_3d"] = float(((p >= day) & (p <= day + np.timedelta64(3, "D"))).any())
                out.at[i, "earn_prev_1_3d"] = float(((p >= day - np.timedelta64(3, "D")) & (p < day)).any())
    return t.join(out)


def add_yfinance(t):
    p = os.path.join("data", "pit", "enrichment", "yf_info.parquet")
    if not os.path.exists(p):
        return t
    y = pd.read_parquet(p).set_index("symbol")
    t["yf_sector"] = t["symbol"].map(y["sector"]).map(SECTOR_MAP)
    t["yf_industry"] = t["symbol"].map(y["industry"])
    t["float_now"] = t["symbol"].map(y["floatShares"])
    t["short_pct_float_now"] = t["symbol"].map(y["shortPercentOfFloat"])
    t["insider_pct_now"] = t["symbol"].map(y["heldPercentInsiders"])
    t["sector"] = t["sector"].fillna(t["yf_sector"])
    t["industry"] = t["industry"].fillna(t["yf_industry"])
    t["sector"] = t["sector"].fillna("Unknown")
    return t


def classify(headlines):
    """Return {category: bool} for a list of lower-cased headlines that are not generic mover lists."""
    out = {k: False for k in CATEGORIES}
    for h in headlines:
        for k, words in CATEGORIES.items():
            if any(w in h for w in words):
                out[k] = True
    return out


def add_news(t):
    """Pre-entry (24h) and post-entry (entry -> next 10:00 ET) headline features. Post-entry is analysis-only."""
    news = pd.read_parquet(os.path.join("data", "pit", "news", "news.parquet"))
    low = news["headline"].fillna("").str.lower() + " " + news["summary"].fillna("").str.lower().str[:200]
    gen = news["headline"].fillna("").str.lower().apply(lambda x: any(g in x for g in GENERIC))
    news = news.assign(low=low, generic=gen)
    by = {s: g.sort_values("created_at") for s, g in news.groupby("symbol")}
    rows = []
    for sym, ts, day in zip(t["symbol"], pd.to_datetime(t["entry_time"], utc=True), t["day"]):
        g = by.get(sym)
        r = {"pre_specific_24h": 0, "post_n": 0, "post_specific": 0, "post_mover_list": 0, "post_any_news": 0, "pre_mover_list": 0}
        r.update({f"post_{k}": False for k in CATEGORIES})
        r.update({f"pre_{k}": False for k in CATEGORIES})
        if g is not None:
            end = pd.Timestamp(day, tz=NY).tz_convert("UTC") + pd.Timedelta(days=1, hours=14)  # next day ~10:00 ET
            post = g[(g["created_at"] > ts) & (g["created_at"] <= end)]
            pre = g[(g["created_at"] > ts - pd.Timedelta(hours=24)) & (g["created_at"] <= ts)]
            r["post_n"], r["post_any_news"] = len(post), int(len(post) > 0)
            r["post_mover_list"] = int(post["generic"].sum())
            sp = post[~post["generic"]]
            r["post_specific"] = len(sp)
            r.update({f"post_{k}": v for k, v in classify(sp["low"]).items()})
            pr = pre[~pre["generic"]]
            r["pre_specific_24h"] = len(pr)
            r["pre_mover_list"] = int(pre["generic"].sum())
            r.update({f"pre_{k}": v for k, v in classify(pr["low"]).items()})
        rows.append(r)
    return t.join(pd.DataFrame(rows, index=t.index))


def prepare(rebuild=False):
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(ENRICHED) and not rebuild:
        with timing.stage("load enriched table"):
            return pd.read_parquet(ENRICHED)
    t = load_trades()
    for name, fn in (
        ("market loads", add_market_overnight),
        ("simfin loads", add_simfin),
        ("yfinance loads", add_yfinance),
        ("news loads", add_news),
    ):
        with timing.stage(name):
            t = fn(t)
    t["winner"] = t["ret"] > 0
    t["big"] = t["ret"] >= 0.10
    t["monster"] = t["ret"] >= 0.25
    t["week"] = t["day"].dt.to_period("W-FRI").dt.start_time
    t["month"] = t["day"].dt.to_period("M").dt.start_time
    with timing.stage("write enriched table"):
        t.to_parquet(ENRICHED, index=False)
    return t


METHODS = """
## Methods and sources

- **Overnight-return literature.** Lou, Polk and Skouras ("A Tug of War: Overnight Versus Intraday Expected Returns") find that momentum returns accrue overnight; Gamm ("A Surprise That Keeps You Awake") documents attention-driven overnight buying after large earnings surprises; Gregoire (Journal of Accounting Research, 2022) shows after-hours prices move through quotes more than trades, so trade-price returns understate and delay the move. Our entries are momentum names held overnight, and our fills use trade bars, so realized fills are optimistic compared to quote-based reality.
- **Selection-bias control.** Bailey and Lopez de Prado's probabilistic / deflated Sharpe ratio (section 11), Benjamini-Hochberg FDR on every family of subgroup and rule tests (sections 4, 5, 8), and a locked chronological test period (sections 7, 8, 12).
- **Purged walk-forward machine learning.** Expanding monthly folds with an embargo, gradient boosting with permutation importance, and a remove-and-retrain check because attribution methods such as SHAP describe the fitted model, including its tails, and can rank features highly that do not generalize (see recent walk-forward TreeSHAP studies).
- **Clustering and regime tests.** Permutation tests on week-level dispersion of big winners (a simple stand-in for self-exciting / Hawkes clustering), change-point test on mean return, sector sympathy permutation test, NMF topic modelling of company descriptions for themes beyond industry codes, k-means archetypes.
- **Data.** Trades from `data/pit` (delisted names included). Sector/industry and point-in-time shares outstanding, market cap and earnings publish dates from SimFin; sector fallback, current float and short interest (snapshots, not point-in-time) and company descriptions from yfinance; headlines from Alpaca (Benzinga) which cover few small-cap press releases, so absence of news in this report means "not found in this feed".
- **Caveats.** About 40% of trades are winners but 10 trades produce more than the entire net profit; every statistic about big winners rests on 157 trades (75 monsters). Float and short interest from yfinance are today's values. Post-entry news is hindsight and cannot be used as a filter.
"""


def build_report(T, out_dir=OUT, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import winners_analysis as wa
    import winners_models as wm
    from explore import STRATEGIES, apply_spec
    from src.backtest.pit_metrics import simulate_account

    wa.OUT, wa.FIG, wa.TAB = out_dir, os.path.join(out_dir, "figures"), os.path.join(out_dir, "tables")
    os.makedirs(wa.FIG, exist_ok=True)
    os.makedirs(wa.TAB, exist_ok=True)
    mkt = {s: pd.read_parquet(os.path.join("data", "pit", "market", f"{s}_daily.parquet")).set_index("date")["close"] for s in ("SPY", "IWM", "QQQ")}
    cand = pd.read_parquet(os.path.join("data", "pit", "candidates.parquet"))
    daily = pd.read_parquet(os.path.join("data", "pit", "daily_pit.parquet"), columns=["date", "symbol", "close"])
    md = []
    nl = chr(10)
    md.append("# " + (title or "Winning-trade analysis: base strategy (13-month atr_pivot, pm_target25_v2 exits)") + nl)
    md.append(f"Generated from {len(T)} de-duplicated trades, {T['day'].min().date()} to {T['day'].max().date()} "
              f"({int((T['period'] == 'dev').sum())} dev, {int((T['period'] == 'test').sum())} test), {T['symbol'].nunique()} symbols, "
              f"{int(T['winner'].sum())} winners ({100 * T['winner'].mean():.1f}%), {int(T['big'].sum())} big winners (>= +10%), {int(T['monster'].sum())} monsters (>= +25%). "
              "Returns are per trade after 0.1%/side commission and slippage (0.15% regular / 0.4% extended hours); entry = 15:55 ET, exit rules as in README." + nl)
    out = {}

    def section(name, fn, *args):
        try:
            out[name] = fn(*args)
        except Exception as e:
            md.append(f"## {name}: skipped ({type(e).__name__}: {e}); too few trades in this subset for this analysis." + nl)

    section("conc", wa.sec_concentration, T, md)
    section("time", wa.sec_time, T, md, mkt)
    section("cat", wa.sec_catalysts, T, md)
    section("sector", wa.sec_sector, T, md)
    section("topics", wa.sec_topics, T, md)
    section("float", wa.sec_float, T, md)
    section("market", wa.sec_market, T, md, cand)
    section("hedge", wa.sec_hedge, T, md)
    section("models", wm.sec_models, T, md)
    section("rules", wm.sec_rules, T, md)
    section("arch", wm.sec_archetypes, T, md)
    section("event", wm.sec_event_study, T, md, daily)
    strategies = {}
    dev = T[T["period"] == "dev"]
    days = pd.bdate_range(dev["day"].min(), dev["day"].max())
    if out_dir == OUT:
        for name in ("base", "S1", "S2", "H3", "H2"):
            _, _, dr = simulate_account(apply_spec(dev, STRATEGIES[name]), all_days=days)
            strategies[name] = dr
    else:
        _, _, dr = simulate_account(dev, all_days=days)
        strategies["this strategy"] = dr
    section("rigor", wm.sec_rigor, T, md, strategies)
    section("robust", wa.sec_robust, T, md)
    md.append(METHODS)
    path = os.path.join(out_dir, "winning_trades_report.md")
    header = ""
    hp = os.path.join(out_dir, "summary_header.md")
    if os.path.exists(hp):
        header = open(hp, encoding="utf-8").read() + nl + nl
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + nl.join(md))
    print("wrote", path)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--prep-only", action="store_true")
    a = ap.parse_args()
    T = prepare(a.rebuild)
    print(T.shape, T[["winner", "big", "monster"]].mean().round(3).to_dict())
    if not a.prep_only:
        build_report(T)
