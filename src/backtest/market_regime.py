"""
Market-regime features for each signal date from SPY / IWM / QQQ (data/pit/market/, built by
`build_pit_dataset.py market`). Everything is known at the 15:55 ET entry:

  <m>_day      price at 15:55 vs prior close
  <m>_ret5/20  prior-close return over the previous 5 / 20 sessions
  <m>_sma50/200_up  15:55 price above the 50/200-day SMA of prior closes (1/0)
  <m>_dist_sma200   15:55 price / SMA200 - 1
"""

import os

import numpy as np
import pandas as pd

MARKET_DIR = os.path.join("data", "pit", "market")
NY = "America/New_York"


def market_table(symbol):
    d = pd.read_parquet(os.path.join(MARKET_DIR, f"{symbol}_daily.parquet")).sort_values("date").set_index("date")
    m = pd.read_parquet(os.path.join(MARKET_DIR, f"{symbol}_minute.parquet"))
    ts = pd.to_datetime(m["timestamp"], utc=True).dt.tz_convert(NY)
    m = m.assign(ts=ts)
    m = m[(m["ts"].dt.time >= pd.Timestamp("09:30").time()) & (m["ts"].dt.time <= pd.Timestamp("15:55").time())]
    p1555 = m.groupby(m["ts"].dt.tz_localize(None).dt.normalize())["close"].last()  # last bar at or before 15:55
    c = d["close"]
    prev = c.shift(1)
    out = pd.DataFrame(index=d.index)
    price = p1555.reindex(d.index)
    out["day"] = price / prev - 1
    out["ret5"] = prev / c.shift(6) - 1
    out["ret20"] = prev / c.shift(21) - 1
    for n in (50, 200):
        sma = c.shift(1).rolling(n).mean()
        out[f"sma{n}_up"] = (price > sma).astype(float).where(sma.notna())
    out["dist_sma200"] = price / c.shift(1).rolling(200).mean() - 1
    return out


def regime_table(symbols=("SPY", "IWM", "QQQ")):
    parts = [market_table(s).add_prefix(f"{s.lower()}_") for s in symbols]
    return pd.concat(parts, axis=1)


def attach_regime(trades, symbols=("SPY", "IWM", "QQQ")):
    """Add regime columns to a trades frame (matches on entry date)."""
    reg = regime_table(symbols)
    day = pd.to_datetime(trades["entry_time"].astype(str).str[:10])
    return trades.join(reg.reindex(day.values).set_index(trades.index))


def summary(symbol="SPY", months=3):
    """How the market has been trending: returns and SMA position over the last `months` months."""
    d = pd.read_parquet(os.path.join(MARKET_DIR, f"{symbol}_daily.parquet")).sort_values("date").set_index("date")
    c = d["close"]
    last = c.index[-1]
    start = last - pd.DateOffset(months=months)
    w = c[c.index >= start]
    return {
        "from": str(w.index[0].date()), "to": str(last.date()),
        "return%": 100 * (w.iloc[-1] / w.iloc[0] - 1),
        "max_drawdown%": 100 * (w / w.cummax() - 1).min(),
        "pct_days_above_sma50": 100 * (c > c.rolling(50).mean()).loc[w.index].mean(),
        "pct_days_above_sma200": 100 * (c > c.rolling(200).mean()).loc[w.index].mean(),
        "last_vs_sma200%": 100 * (c.iloc[-1] / c.rolling(200).mean().iloc[-1] - 1),
        "up_days%": 100 * (c.pct_change().loc[w.index] > 0).mean(),
    }
