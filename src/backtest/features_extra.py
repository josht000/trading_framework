"""
Extra point-in-time features for the trade-ranking work. Everything is computed from information published at or before the
15:55 ET entry (windows end at the entry time, inclusive), except columns prefixed `post_` which look ahead and are for analysis only.

  EDGAR (data/pit/edgar/filings.parquet, `build_pit_dataset.py edgar`): 8-K counts and item flags since the prior session
      close, offering / dilution forms in the prior 10 days, distress items (3.01 delisting notice, 1.03 bankruptcy, 4.02 non-reliance,
      late-filing forms) in the prior 30 days.
  Benzinga (data/pit/news/news.parquet): company-specific / event headlines since the prior session close.
  Theme heat: how many other candidate breakouts share the company's description topic (NMF on descriptions) within the last 7 days.
  OTC: ticker present in Massive's OTC / pink-sheet reference (`build_pit_dataset.py otc`) -> excluded from trading.
"""

import os

import numpy as np
import pandas as pd

NY = "America/New_York"
PIT = os.path.join("data", "pit")

GENERIC = ("stocks moving", "stocks are moving", "movers", "stocks hitting", "stocks that hit", "shares are trading",
           "trading higher", "trading lower", "session", "mid-day", "pre-market", "premarket", "after-market",
           "moving in", "earnings scheduled", "analyzing", "short interest", "analyst ratings", "options activity",
           "unusual options", "nasdaq", "dow ", "s&p 500", "futures", "52-week")
EVENT = ("announce", "reports", "secures", "signs", "receives", "completes", "approval", "approves", "clearance", "trial",
         "results", "contract", "award", "agreement", "partnership", "acquir", "launch", "order", "selected", "grant",
         "raises", "upgrade", "initiates", "patent", "milestone", "expands", "wins")
MOVERS = ("stocks moving", "stocks are moving", "movers", "stocks hitting", "stocks that hit", "shares are trading",
          "trading higher", "trading lower", "moving in", "mid-day", "session", "52-week")
NEG = ("offering", "private placement", "registered direct", "at-the-market", "dilut", "warrant", "reverse split",
       "going concern", "bankrupt", "delist", "deficiency", "downgrade", "lawsuit", "investigation", "default")


def prior_close_utc(day):
    prev = pd.Timestamp(day) - pd.offsets.BDay(1)
    return pd.Timestamp(prev.strftime("%Y-%m-%d") + " 16:00", tz=NY).tz_convert("UTC")


def _entry_utc(t):
    return pd.to_datetime(t["entry_time"].astype(str), utc=True)


# ---------------------------------------------------------------- EDGAR
def add_edgar(t):
    path = os.path.join(PIT, "edgar", "filings.parquet")
    cols = ["e_has_data", "e_8k_since_close", "e_8k_24h", "e_item_101", "e_item_201", "e_item_202", "e_item_502", "e_item_701_801",
            "e_item_302_10d", "e_offering_10d", "e_distress_30d", "e_delist_notice_90d", "e_late_30d", "e_other_distress_30d", "post_e_8k"]
    if not os.path.exists(path):
        return t.assign(**{c: np.nan for c in cols})
    f = pd.read_parquet(path)
    by = {s: g.sort_values("acceptance_utc") for s, g in f.groupby("symbol")}
    rows = []
    for sym, ts, day in zip(t["symbol"], _entry_utc(t), t["day"]):
        r = dict.fromkeys(cols, 0)
        g = by.get(sym)
        r["e_has_data"] = int(g is not None)
        if g is not None:
            pc = prior_close_utc(day)
            acc = g["acceptance_utc"]
            k8 = g[g["form"].isin(["8-K", "8-K/A"])]
            sc = k8[(k8["acceptance_utc"] > pc) & (k8["acceptance_utc"] <= ts)]
            r["e_8k_since_close"] = len(sc)
            r["e_8k_24h"] = int(((k8["acceptance_utc"] > ts - pd.Timedelta(hours=24)) & (k8["acceptance_utc"] <= ts)).sum())
            items = ",".join(sc["items"].fillna("").astype(str)).split(",")
            items = {i.strip() for i in items if i.strip()}
            r["e_item_101"] = int("1.01" in items)
            r["e_item_201"] = int("2.01" in items)
            r["e_item_202"] = int("2.02" in items)
            r["e_item_502"] = int("5.02" in items)
            r["e_item_701_801"] = int(bool(items & {"7.01", "8.01"}))
            w10 = g[(g["acceptance_utc"] > ts - pd.Timedelta(days=10)) & (g["acceptance_utc"] <= ts)]
            i10 = ",".join(w10[w10["form"].isin(["8-K", "8-K/A"])]["items"].fillna("").astype(str)).split(",")
            r["e_item_302_10d"] = int("3.02" in {i.strip() for i in i10})
            r["e_offering_10d"] = int(w10["form"].str.startswith(("424B", "S-1", "S-3", "F-1", "F-3")).any())
            w30 = g[(g["acceptance_utc"] > ts - pd.Timedelta(days=30)) & (g["acceptance_utc"] <= ts)]
            i30 = {i.strip() for i in ",".join(w30[w30["form"].isin(["8-K", "8-K/A"])]["items"].fillna("").astype(str)).split(",") if i.strip()}
            r["e_distress_30d"] = int(bool(i30 & {"3.01", "1.03", "4.02"}) or w30["form"].isin(["NT 10-K", "NT 10-Q"]).any())
            w90 = g[(g["acceptance_utc"] > ts - pd.Timedelta(days=90)) & (g["acceptance_utc"] <= ts)]
            i90 = {i.strip() for i in ",".join(w90[w90["form"].isin(["8-K", "8-K/A"])]["items"].fillna("").astype(str)).split(",") if i.strip()}
            r["e_delist_notice_90d"] = int("3.01" in i90)
            r["e_late_30d"] = int(w30["form"].isin(["NT 10-K", "NT 10-Q"]).any())
            r["e_other_distress_30d"] = int(bool(i30 & {"1.03", "4.02"}))
            end = pd.Timestamp(day, tz=NY).tz_convert("UTC") + pd.Timedelta(days=1, hours=14)
            r["post_e_8k"] = int(((k8["acceptance_utc"] > ts) & (k8["acceptance_utc"] <= end)).sum())
        rows.append(r)
    return t.join(pd.DataFrame(rows, index=t.index))


# ---------------------------------------------------------------- Benzinga since prior close
def add_news_since_close(t):
    path = os.path.join(PIT, "news", "news.parquet")
    cols = ["n_specific_sc", "n_event_sc", "n_neg_sc", "n_mover_5d"]
    if not os.path.exists(path):
        return t.assign(**{c: 0 for c in cols})
    n = pd.read_parquet(path)
    h = n["headline"].fillna("").str.lower()
    n = n.assign(low=h + " " + n["summary"].fillna("").str.lower().str[:200],
                 generic=h.apply(lambda x: any(g in x for g in GENERIC)),
                 mover=h.apply(lambda x: any(g in x for g in MOVERS)))
    movers_news = {s: g["created_at"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy() for s, g in n[n["mover"]].groupby("symbol")}
    n = n[~n["generic"]]
    n["event"] = n["low"].apply(lambda x: any(e in x for e in EVENT))
    n["neg"] = n["low"].apply(lambda x: any(e in x for e in NEG))
    by = {s: g for s, g in n.groupby("symbol")}
    out = []
    for sym, ts, day in zip(t["symbol"], _entry_utc(t), t["day"]):
        g = by.get(sym)
        mv = movers_news.get(sym)
        n_mv = 0 if mv is None else int(((mv > np.datetime64(ts.tz_convert("UTC").tz_localize(None) - pd.Timedelta(days=5)))
                                         & (mv <= np.datetime64(ts.tz_convert("UTC").tz_localize(None)))).sum())
        if g is None:
            out.append((0, 0, 0, n_mv))
            continue
        w = g[(g["created_at"] > prior_close_utc(day)) & (g["created_at"] <= ts)]
        out.append((len(w), int(w["event"].sum()), int(w["neg"].sum()), n_mv))
    return t.join(pd.DataFrame(out, columns=cols, index=t.index))


# ---------------------------------------------------------------- themes
def topic_map(k=14):
    """symbol -> NMF topic id from yfinance business descriptions (cached)."""
    cache = os.path.join(PIT, "enrichment", "topics.parquet")
    src = os.path.join(PIT, "enrichment", "yf_info.parquet")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(src):
        return pd.read_parquet(cache)
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    y = pd.read_parquet(src).set_index("symbol")["longBusinessSummary"].dropna()
    vec = TfidfVectorizer(stop_words="english", min_df=8, max_df=0.35, ngram_range=(1, 2), max_features=6000)
    X = vec.fit_transform(y.values)
    nmf = NMF(n_components=k, random_state=0, init="nndsvda", max_iter=500)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    label = {i: ", ".join(terms[np.argsort(-nmf.components_[i])[:4]]) for i in range(k)}
    df = pd.DataFrame({"symbol": y.index, "topic": W.argmax(axis=1), "strength": W.max(axis=1)})
    df["topic_label"] = df["topic"].map(label)
    df.loc[df["strength"] <= 0.015, "topic"] = -1
    df.to_parquet(cache, index=False)
    return df


def add_theme_heat(t, cand, days=7):
    """Number of other candidate breakouts in the same topic (and the same sector) within the last `days` calendar days incl. today."""
    tm = topic_map().set_index("symbol")["topic"]
    c = cand.copy()
    c["topic"] = c["symbol"].map(tm)
    c = c[c["topic"].notna() & (c["topic"] >= 0)]
    by_topic = {k: g[["date", "symbol"]] for k, g in c.groupby("topic")}
    heat = []
    for sym, day in zip(t["symbol"], t["day"]):
        tp = tm.get(sym, np.nan)
        if pd.isna(tp) or tp < 0 or tp not in by_topic:
            heat.append(0)
            continue
        g = by_topic[tp]
        w = g[(g["date"] <= day) & (g["date"] > day - pd.Timedelta(days=days)) & (g["symbol"] != sym)]
        heat.append(w["symbol"].nunique())
    t = t.assign(theme_heat=heat, topic=t["symbol"].map(tm))
    sec = t.groupby(["day", "sector"])["symbol"].transform("nunique") - 1
    return t.assign(sector_heat=np.where(t["sector"] == "Unknown", 0, sec))


# ---------------------------------------------------------------- top daily gainers (movers list proxy from our own bars)
def top_gainers(top_n=20, min_price=1.0, min_dollar_volume=500_000):
    path = os.path.join(PIT, "enrichment", f"top_gainers_{top_n}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    d = pd.read_parquet(os.path.join(PIT, "daily_pit.parquet"), columns=["date", "symbol", "close", "volume"]).sort_values(["symbol", "date"])
    d["prev"] = d.groupby("symbol")["close"].shift(1)
    d["ret"] = d["close"] / d["prev"] - 1
    d = d[(d["prev"] >= min_price) & (d["close"] * d["volume"] >= min_dollar_volume)].dropna(subset=["ret"])
    d["rank"] = d.groupby("date")["ret"].rank(ascending=False, method="first")
    out = d[d["rank"] <= top_n][["date", "symbol", "rank", "ret"]]
    out.to_parquet(path, index=False)
    return out


def add_movers(t, top_n=20, sessions=5):
    """n_top_gainer_days: previous `sessions` sessions on which the stock was among the day's top `top_n` gainers (price >= $1,
    dollar volume >= $0.5M, whole listed universe including delisted names). Known before the entry day starts."""
    tg = top_gainers(top_n)
    by_day = {d: set(g["symbol"]) for d, g in tg.groupby("date")}
    days = sorted(by_day)
    idx = pd.Series(range(len(days)), index=days)
    out = []
    for sym, day in zip(t["symbol"], t["day"]):
        pos = days.__len__() if day not in idx.index else int(idx[day])
        if day not in idx.index:
            pos = int(np.searchsorted(np.array(days, dtype="datetime64[ns]"), np.datetime64(day)))
        prev = days[max(0, pos - sessions):pos]
        out.append(sum(sym in by_day[d] for d in prev))
    return t.assign(n_top_gainer_days=out)


# ---------------------------------------------------------------- OTC
def add_otc(t):
    path = os.path.join(PIT, "otc_reference.parquet")
    if not os.path.exists(path):
        return t.assign(is_otc=False)
    otc = set(pd.read_parquet(path)["ticker"])
    return t.assign(is_otc=t["symbol"].isin(otc))
