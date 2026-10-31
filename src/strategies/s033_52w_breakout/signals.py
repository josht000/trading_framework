"""S033 52-week-high breakout: signals on the merged ETF + liquid-stock TR panel."""

import numpy as np
import pandas as pd


def compute_signals(panel, cfg, rank, etf_cols):
    """entry close >= rolling N-day high (inclusive); exit close <= rolling M-day low."""
    close = panel["close"]
    hi_n, lo_n = int(cfg["entry_window"]), int(cfg["exit_window"])
    close_df = pd.DataFrame(close)
    roll_high = close_df.rolling(hi_n, min_periods=hi_n).max().to_numpy()
    roll_low = close_df.rolling(lo_n, min_periods=lo_n).min().to_numpy()
    history = np.isfinite(close).cumsum(0)
    is_etf = np.zeros(close.shape[1], bool)
    is_etf[etf_cols] = True
    stock_ok = ~is_etf[None, :] & (rank <= int(cfg["universe_top_n"]))
    etf_ok = is_etf[None, :] & np.ones_like(close, bool)
    tradable = np.isfinite(close) & (history >= int(cfg["min_history"])) & (stock_ok | etf_ok)
    signal = tradable & (close >= roll_high)
    exit_now = close <= roll_low
    return {"signal": signal, "exit_now": exit_now, "tradable": tradable, "pool": tradable,
            "roll_high": roll_high, "roll_low": roll_low}


def build_trades(panel, sig, cfg, fill):
    from src.strategies._engines import swing_trades

    return swing_trades(panel, sig["signal"], sig["exit_now"], cfg, fill,
                        delisting_return=float(cfg.get("delisting_return", -0.30)))
