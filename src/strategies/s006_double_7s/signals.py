"""S006 Double 7s: signals on a wide ETF total-return panel (date x symbol)."""

import numpy as np
import pandas as pd


def compute_signals(panel, cfg):
    """tradable/pool masks + entry (close > SMA200 AND close at N-day low) / exit (close at N-day high)."""
    close = panel["close"]
    lb = int(cfg["lookback"])
    close_df = pd.DataFrame(close)
    roll_low = close_df.rolling(lb, min_periods=lb).min().to_numpy()
    roll_high = close_df.rolling(lb, min_periods=lb).max().to_numpy()
    sma = close_df.rolling(int(cfg["sma_window"]), min_periods=int(cfg["sma_window"])).mean().to_numpy()
    history = np.isfinite(close).cumsum(0)
    tradable = np.isfinite(close) & (history >= int(cfg["min_history"]))
    signal = tradable & (close > sma) & (close <= roll_low)
    exit_now = close >= roll_high
    atr = None
    return {"signal": signal, "exit_now": exit_now, "atr": atr, "tradable": tradable, "pool": tradable,
            "roll_low": roll_low, "roll_high": roll_high}


def build_trades(panel, sig, cfg, fill):
    from src.strategies._engines import swing_trades

    return swing_trades(panel, sig["signal"], sig["exit_now"], cfg, fill,
                        stop_atr=sig["atr"], stop_mult=float(cfg.get("disaster_stop_atr", 0.0)))
