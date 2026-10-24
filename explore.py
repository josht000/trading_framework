"""
Entry-filter exploration on the DEV set (3-month holdout never loaded).

  python explore.py [--variant pm_target25_v2] [--min-trades 300]

For each filter (market regime from SPY/IWM/QQQ, 200 SMA / EMA position, news and dilution flags)
and a few stacks, reports hit rate, expectancy and "stay alive" account stats on a $5,000 account
(<= 5 overnight trades, $1,000 each, $1,250 for boosted trades). Split-half columns (h1 / h2)
show whether a result holds in both halves of the dev period.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.backtest.market_regime import attach_regime
from src.backtest.pit_metrics import dedupe_trades, simulate_account


def _mcap_column(t):
    return t["mcap_final"] if "mcap_final" in t.columns else t["mcap_pit"]

METHOD = "atr_pivot" + (("_" + os.getenv("RESULTS_TAG")) if os.getenv("RESULTS_TAG") else "")

FILTERS = {
    "all": lambda t: t["ret"].notna(),
    # market regime (known at 15:55)
    "spy_day>=0": lambda t: t["spy_day"] >= 0,
    "spy_day>=0.1%": lambda t: t["spy_day"] >= 0.001,
    "spy_day>=0.25%": lambda t: t["spy_day"] >= 0.0025,
    "spy>sma50": lambda t: t["spy_sma50_up"] == 1,
    "spy_ret20>0": lambda t: t["spy_ret20"] > 0,
    "iwm_day>=0": lambda t: t["iwm_day"] >= 0,
    "shares<=100M": lambda t: t["shares_pit"].fillna(t["float_now"]) <= 100e6,
    "shares<30M": lambda t: t["shares_pit"].fillna(t["float_now"]) < 30e6,
    "mcap<200M": lambda t: _mcap_column(t) < 200e6,
    "headroom>=10%": lambda t: t["headroom"] >= 0.10,
    "headroom>=20%": lambda t: t["headroom"] >= 0.20,
    "headroom>=30%": lambda t: t["headroom"] >= 0.30,
    "headroom<20%": lambda t: t["headroom"] < 0.20,
    "iwm_day>=0.25%": lambda t: t["iwm_day"] >= 0.0025,
    "iwm>sma50": lambda t: t["iwm_sma50_up"] == 1,
    "iwm_ret20>0": lambda t: t["iwm_ret20"] > 0,
    "iwm_ret5>0": lambda t: t["iwm_ret5"] > 0,
    "qqq_day>=0.1%": lambda t: t["qqq_day"] >= 0.001,
    # the stock's own moving averages
    "above_sma200": lambda t: t["above_sma200"] == 1,
    "below_sma200": lambda t: t["above_sma200"] == 0,
    "no_sma200_overhead<=15%": lambda t: ~t["dist_sma200"].between(-0.15, 0),
    "ema9_ext<=25%": lambda t: t["dist_ema9"] <= 0.25,
    "ema20_ext<=40%": lambda t: t["dist_ema20"] <= 0.40,
    # news (Alpaca/Benzinga headlines up to 15:55)
    "news_24h>=1": lambda t: t["news_24h"] >= 1,
    "news_5d>=1": lambda t: t["news_5d"] >= 1,
    "no_news_24h": lambda t: t["news_24h"] == 0,
    "news_pos_24h>=1": lambda t: t["news_pos_24h"] >= 1,
    "news_score_24h>=1": lambda t: t["news_score_24h"] >= 1,
    "news_neg_24h==0": lambda t: t["news_neg_24h"] == 0,
    "specific_news_24h>=1": lambda t: t["specific_24h"] >= 1,
    "event_news_24h>=1": lambda t: t["event_24h"] >= 1,
    "no_event_news_24h": lambda t: t["event_24h"] == 0,
    "no_dilution_10d": lambda t: t["dilution_10d"] == 0,
}
STACKS = {
    "news_24h & no_dilution": ["news_24h>=1", "no_dilution_10d"],
    "spy_day>=0.1% & news_24h": ["spy_day>=0.1%", "news_24h>=1"],
    "iwm_day>=0 & news_24h": ["iwm_day>=0", "news_24h>=1"],
    "iwm_day>=0 & no_dilution": ["iwm_day>=0", "no_dilution_10d"],
    "spy_day>=0.1% & iwm_day>=0": ["spy_day>=0.1%", "iwm_day>=0"],
    "iwm_day>=0 & news_pos & no_dilution": ["iwm_day>=0", "news_pos_24h>=1", "no_dilution_10d"],
    "above_sma200 & news_24h": ["above_sma200", "news_24h>=1"],
    "above_sma200 & iwm_day>=0": ["above_sma200", "iwm_day>=0"],
    "spy_day>=0.1% & news_pos & no_dilution": ["spy_day>=0.1%", "news_pos_24h>=1", "no_dilution_10d"],
    "above_sma200 & iwm_day>=0.25%": ["above_sma200", "iwm_day>=0.25%"],
    "above_sma200 & spy_day>=0": ["above_sma200", "spy_day>=0"],
    "above_sma200 & iwm_day>=0 & no_dilution": ["above_sma200", "iwm_day>=0", "no_dilution_10d"],
    "above_sma200 & iwm_day>=0 & event_news": ["above_sma200", "iwm_day>=0", "event_news_24h>=1"],
    "above_sma200 & iwm_day>=0 & specific_news": ["above_sma200", "iwm_day>=0", "specific_news_24h>=1"],
    "above_sma200 & iwm_day>=0 & no_event_news": ["above_sma200", "iwm_day>=0", "no_event_news_24h"],
    "above_sma200 & iwm_day>=0 & spy_day>=0": ["above_sma200", "iwm_day>=0", "spy_day>=0"],
    "headroom>=20% & above_sma200": ["headroom>=20%", "above_sma200"],
    "headroom>=20% & above_sma200 & iwm_day>=0 & spy_day>=0": ["headroom>=20%", "above_sma200", "iwm_day>=0", "spy_day>=0"],
    "headroom>=20% & iwm_day>=0 & spy_day>=0": ["headroom>=20%", "iwm_day>=0", "spy_day>=0"],
    "headroom>=30% & above_sma200 & iwm_day>=0 & spy_day>=0": ["headroom>=30%", "above_sma200", "iwm_day>=0", "spy_day>=0"],
}


GENERIC = ("stocks moving", "stocks are moving", "movers", "stocks hitting", "stocks that hit", "shares are trading",
           "trading higher", "trading lower", "session", "mid-day", "pre-market", "premarket", "after-market",
           "moving in", "earnings scheduled", "analyzing", "short interest", "analyst ratings", "options activity",
           "unusual options", "nasdaq", "dow ", "s&p 500", "futures", "52-week")
EVENT = ("announce", "reports", "secures", "signs", "receives", "completes", "approval", "approves", "clearance", "trial",
         "results", "contract", "award", "agreement", "partnership", "acquir", "launch", "order", "selected", "grant",
         "raises", "upgrade", "initiates", "patent", "milestone", "expands", "wins")


def add_news_quality(t):
    """Company-specific headlines (mover lists / calendars / index recaps removed) in the 24h before entry."""
    news = pd.read_parquet(os.path.join("data", "pit", "news", "news.parquet"))
    h = news["headline"].fillna("").str.lower()
    generic = h.apply(lambda x: any(g in x for g in GENERIC))
    news = news.assign(event=(~generic) & h.apply(lambda x: any(e in x for e in EVENT)), specific=~generic)
    by = {s: g for s, g in news.groupby("symbol")}
    spec, evt = [], []
    for sym, ts in zip(t["symbol"], pd.to_datetime(t["entry_time"], utc=True)):
        g = by.get(sym)
        if g is None:
            spec.append(0)
            evt.append(0)
            continue
        w = g[(g["created_at"] > ts - pd.Timedelta(hours=24)) & (g["created_at"] <= ts)]
        spec.append(int(w["specific"].sum()))
        evt.append(int(w["event"].sum()))
    return t.assign(specific_24h=spec, event_24h=evt)


def load_dev(variant):
    t = pd.read_csv(os.path.join("results", f"pit_trades_{METHOD}_{variant}.csv"), parse_dates=["entry_time"])
    return add_news_quality(attach_regime(dedupe_trades(t)))


def boost_rule(t):
    return ((t["news_pos_24h"] >= 1) & (t["dilution_10d"] == 0) & (t["iwm_day"] >= 0)).to_numpy()


def evaluate(t, days, boost=None):
    if len(t) == 0:
        return None
    st, eq, _ = simulate_account(t, boost=boost, all_days=days)
    return st


def row(name, t, days, mid, boost=None):
    st = evaluate(t, days, boost)
    if st is None:
        return None
    d = pd.to_datetime(t["entry_time"].astype(str).str[:10])
    h1, h2 = t[d <= mid], t[d > mid]
    out = {"filter": name, **{k: st[k] for k in ("trades", "win%", "mean%", "PF", "total%", "maxDD%", "min$", "worst_day%", "loss_streak_days", "sharpe")}}
    out["win%_h1"] = 100 * (h1["ret"] > 0).mean() if len(h1) else np.nan
    out["win%_h2"] = 100 * (h2["ret"] > 0).mean() if len(h2) else np.nan
    out["mean%_h1"] = 100 * h1["ret"].mean() if len(h1) else np.nan
    out["mean%_h2"] = 100 * h2["ret"].mean() if len(h2) else np.nan
    return out


STRATEGIES = {  # named candidates: exit variant + entry filters
    "base": dict(variant="pm_target25_v2", keys=["all"]),
    "S1": dict(variant="pm_target25_v2", keys=["above_sma200", "iwm_day>=0", "spy_day>=0"]),
    "S2": dict(variant="pm_target25_v2", keys=["above_sma200", "iwm_day>=0.25%"]),
    "H1": dict(variant="pm_target25_v2", keys=["headroom>=20%"]),
    "H3": dict(variant="pm_target25_v2", keys=["headroom>=20%", "above_sma200"]),
    "H2": dict(variant="pm_target25_v2", keys=["headroom>=20%", "above_sma200", "iwm_day>=0", "spy_day>=0"]),
}


def apply_spec(t, spec):
    m = np.ones(len(t), dtype=bool)
    for k in spec["keys"]:
        m &= FILTERS[k](t).fillna(False).to_numpy()
    return t[m]


def export_and_report(name, holdout=False):
    """Write the executed trades of a named strategy; holdout=True scores the locked final test set."""
    from src.backtest.pit_metrics import select_trades
    spec = STRATEGIES[name]
    p = os.path.join("results", "holdout" if holdout else "", f"pit_trades_{METHOD}_{spec['variant']}.csv")
    t = pd.read_csv(p, parse_dates=["entry_time"])
    t = add_news_quality(attach_regime(dedupe_trades(t)))
    day = pd.to_datetime(t["entry_time"].astype(str).str[:10])
    days = pd.bdate_range(day.min(), day.max())
    out = {}
    for label, sp in (("base", STRATEGIES["base"]), (name, spec)):
        s = apply_spec(t, sp)
        st, eq, dr = simulate_account(s, all_days=days)
        out[label] = st
        if label == name:
            sel = select_trades(s)
            sel["frac"] = 0.20
            dest = os.path.join("results", "holdout" if holdout else "", f"pit_trades_{METHOD}_{name}.csv")
            sel.drop(columns=["day"]).to_csv(dest, index=False)
    print(pd.DataFrame(out).T.to_string(float_format=lambda x: f"{x:.2f}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="pm_target25_v2")
    ap.add_argument("--min-trades", type=int, default=300)
    ap.add_argument("--export", default=None, help="write executed dev trades of a STRATEGIES entry")
    ap.add_argument("--test", default=None, help="score a STRATEGIES entry once on the locked final test set")
    a = ap.parse_args()
    if a.export:
        return export_and_report(a.export)
    if a.test:
        return export_and_report(a.test, holdout=True)
    t = load_dev(a.variant)
    days = pd.bdate_range(pd.to_datetime(t["entry_time"].astype(str).str[:10]).min(),
                          pd.to_datetime(t["entry_time"].astype(str).str[:10]).max())
    mid = days[len(days) // 2]
    rows = []
    for name, f in FILTERS.items():
        r = row(name, t[f(t).fillna(False)], days, mid)
        if r:
            rows.append(r)
    for name, keys in STACKS.items():
        m = np.ones(len(t), dtype=bool)
        for k in keys:
            m &= FILTERS[k](t).fillna(False).to_numpy()
        r = row(name, t[m], days, mid)
        if r:
            rows.append(r)
    res = pd.DataFrame(rows).set_index("filter")
    res.to_csv(os.path.join("results", f"explore_{a.variant}.csv"))
    with pd.option_context("display.width", 300, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print(f"base variant {a.variant}, dev trades after dedupe: {len(t)}; min trades shown {a.min_trades}")
        print(res[res.trades >= a.min_trades].sort_values("win%", ascending=False))
        print("\nfewer than min trades:", ", ".join(res[res.trades < a.min_trades].index))
        # boost test on the unfiltered and a filtered set
        print("\nboost rule (25% instead of 20% when pos news & no dilution & IWM up):")
        for name, mask in (("all", np.ones(len(t), bool)), ("news_24h>=1", FILTERS["news_24h>=1"](t).fillna(False).to_numpy())):
            s = t[mask]
            for lab, b in (("no boost", None), ("boost", boost_rule)):
                st = evaluate(s, days, b)
                print(f"  {name:12s} {lab:9s} total {st['total%']:7.1f}%  maxDD {st['maxDD%']:6.1f}%  min$ {st['min$']:8.0f}  win {st['win%']:.1f}%")


if __name__ == "__main__":
    main()
