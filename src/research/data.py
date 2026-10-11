"""Cached daily-bar loader for research (split-adjusted Alpaca SIP bars)."""

import os
from typing import Dict, List

import pandas as pd

CACHE_DIR = os.path.join("data", "cache", "daily")
WARMUP_DAYS = 420


def _cache_path(symbol: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol.upper()}_1Day_split.parquet")


def load_daily(symbols: List[str], start: str, end: str, refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Daily bars per symbol covering start - warmup .. end, cached as parquet.
    Warmup bars are included so ATR/pivots are warm at `start`. When the cache does not
    cover the request, the fetch is widened to the union of the cached and requested
    ranges so repeated calls with different dates do not shrink the cache.
    """
    from src.data.alpaca_client import AlpacaDataClient

    os.makedirs(CACHE_DIR, exist_ok=True)
    want_start = pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)
    want_end = pd.Timestamp(end)

    out, plan = {}, {}
    for s in symbols:
        path = _cache_path(s)
        fetch_start, fetch_end = want_start, want_end
        if os.path.exists(path) and not refresh:
            df = pd.read_parquet(path)
            idx = pd.DatetimeIndex(df.index)
            if idx.tz is not None:
                idx = idx.tz_convert("America/New_York").tz_localize(None)
            lo, hi = idx.min().normalize(), idx.max().normalize()
            if lo <= want_start + pd.Timedelta(days=7) and hi >= want_end - pd.Timedelta(days=7):
                out[s] = df
                continue
            fetch_start, fetch_end = min(lo, want_start), max(hi, want_end)
        plan.setdefault((fetch_start, fetch_end), []).append(s)

    if plan:
        client = AlpacaDataClient()
        for (a, b), group in plan.items():
            fetched = client.fetch_daily_bars(group, a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"))
            for s, df in fetched.items():
                df.to_parquet(_cache_path(s))
                out[s] = df
            for s in set(group) - set(fetched):
                print(f"  {s}: no daily data returned")

    return {s: out[s] for s in symbols if s in out}
