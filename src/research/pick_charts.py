"""
Pick chart dates for labeling WITHOUT looking at any detector.

A resistance detector can only be judged on charts that actually have overhead
resistance, so dates are chosen by simple rules that do not use the detectors:

  - price within [min_price, max_price] and 20-day average volume >= min_avg_volume
    (the strategy's tradable universe)
  - close is min_ratio..max_ratio of the prior 252-bar high: clearly below the 1-year
    high (has overhead supply) but not a collapsed stock
  - 20-day return >= min_ret20: rising toward the overhead levels
  - at least min_bars_above bars in the last 252 traded above price by 2% or more
  - enough history, and forward bars after the date so you can see what happened

Dates are then sampled at random (fixed seed) with a minimum spacing, so the set is
reproducible and not cherry-picked.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

NY_TZ = "America/New_York"


@dataclass(frozen=True)
class PickConfig:
    min_price: float = 1.0
    max_price: float = 20.0
    min_avg_volume: float = 500_000
    min_ratio: float = 0.55
    max_ratio: float = 0.90
    min_ret20: float = 0.05
    min_bars_above: int = 20
    window: int = 252
    min_history: int = 260
    forward_buffer: int = 15
    min_spacing: int = 40
    per_symbol: int = 1
    max_total: int = 0
    seed: int = 7


def _normalize(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(NY_TZ).tz_localize(None)
    df.index = idx.normalize()
    df = df.sort_index()
    return df[~df.index.duplicated(keep="last")]


def eligible_dates(daily: pd.DataFrame, cfg: PickConfig) -> pd.DataFrame:
    df = _normalize(daily)
    close, high = df["close"].to_numpy(float), df["high"].to_numpy(float)
    n, w = len(df), cfg.window
    hi_prior = pd.Series(high).rolling(w).max().to_numpy()
    ratio = close / hi_prior
    ret20 = pd.Series(close).pct_change(20).to_numpy()
    vol20 = df["volume"].rolling(20).mean().to_numpy()

    bars_above = np.full(n, np.nan)
    for k in range(w - 1, n):
        bars_above[k] = np.sum(high[k - w + 1:k + 1] > close[k] * 1.02)

    ok = (
        (np.arange(n) >= cfg.min_history) & (np.arange(n) <= n - 1 - cfg.forward_buffer)
        & (close >= cfg.min_price) & (close <= cfg.max_price) & (vol20 >= cfg.min_avg_volume)
        & (ratio >= cfg.min_ratio) & (ratio <= cfg.max_ratio)
        & (ret20 >= cfg.min_ret20) & (bars_above >= cfg.min_bars_above)
    )
    out = pd.DataFrame({"asof": df.index, "close": close, "ratio_to_252d_high": ratio, "ret20": ret20,
                        "avg_vol20": vol20, "bars_above": bars_above, "pos": np.arange(n)})
    return out[ok].reset_index(drop=True)


def pick_dates(daily_by_symbol: Dict[str, pd.DataFrame], start, end, cfg: PickConfig,
               exclude: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Pick dates per symbol. `exclude` (columns symbol, asof) are charts already used: new picks
    stay at least min_spacing bars away from them."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    used = {}
    if exclude is not None and not exclude.empty:
        for sym, g in exclude.assign(asof=pd.to_datetime(exclude["asof"])).groupby(exclude["symbol"].str.upper()):
            used[sym] = list(g["asof"].unique())

    rows = []
    for i, (symbol, daily) in enumerate(sorted(daily_by_symbol.items())):
        el = eligible_dates(daily, cfg)
        el = el[(el["asof"] >= start) & (el["asof"] <= end)]
        if el.empty:
            continue
        index = _normalize(daily).index
        taken = [int(index.searchsorted(pd.Timestamp(a))) for a in used.get(symbol.upper(), [])]
        rng = np.random.default_rng(cfg.seed + i)
        new = 0
        for j in rng.permutation(len(el)):
            row = el.iloc[j]
            if all(abs(int(row["pos"]) - c) >= cfg.min_spacing for c in taken):
                taken.append(int(row["pos"]))
                new += 1
                rows.append({"symbol": symbol, **row.drop("pos").to_dict(), "eligible_days": len(el)})
            if new >= cfg.per_symbol:
                break
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if cfg.max_total and len(out) > cfg.max_total:
        keep = np.sort(np.random.default_rng(cfg.seed).choice(len(out), cfg.max_total, replace=False))
        out = out.iloc[keep]
    return out.sort_values(["symbol", "asof"]).reset_index(drop=True)
