"""
Fetch company descriptors for the traded symbols (current snapshot, not point-in-time):
yfinance sector / industry / float / short interest, cached in data/pit/enrichment/yf_info.parquet.

  python fetch_enrichment.py
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

KEYS = ["sector", "industry", "floatShares", "sharesOutstanding", "marketCap", "shortPercentOfFloat",
        "sharesShort", "heldPercentInsiders", "heldPercentInstitutions", "country", "quoteType", "longBusinessSummary"]


def one(sym):
    import yfinance as yf
    try:
        info = yf.Ticker(sym).info or {}
    except Exception:
        info = {}
    return {"symbol": sym, **{k: info.get(k) for k in KEYS}}


def main():
    t = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in (
        "results/pit_trades_atr_pivot_lb13_pm_target25_v2.csv", "results/holdout/pit_trades_atr_pivot_lb13_pm_target25_v2.csv")])
    cand = pd.read_parquet("data/pit/candidates.parquet", columns=["symbol"])
    syms = sorted(set(t["symbol"].unique()) | set(cand["symbol"].unique()))
    out = "data/pit/enrichment/yf_info.parquet"
    old = pd.read_parquet(out) if os.path.exists(out) else pd.DataFrame(columns=["symbol"])
    syms = [s for s in syms if s not in set(old["symbol"])]
    rows = []
    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(one, syms), 1):
            rows.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(syms)}", flush=True)
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"yfinance info: {df['sector'].notna().sum()}/{len(df)} symbols with a sector, "
          f"{df['floatShares'].notna().sum()} with float")


if __name__ == "__main__":
    main()
