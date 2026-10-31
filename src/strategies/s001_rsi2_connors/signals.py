"""S001 RSI(2) Classic (Connors): signals on a wide ETF total-return panel (date x symbol)."""

import numpy as np
import pandas as pd


def wilder_rsi(close_df, n):
    """Wilder RSI(n) column-wise on a price DataFrame."""
    d = close_df.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = up / dn
    return (100.0 - 100.0 / (1.0 + rs))


def wilder_atr(high_df, low_df, close_df, n):
    """Wilder ATR(n): ewm(alpha=1/n) of the true range."""
    pc = close_df.shift(1)
    tr = pd.concat([high_df - low_df, (high_df - pc).abs(), (low_df - pc).abs()]).groupby(level=0).max()
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def compute_signals(panel, cfg, high=None, low=None):
    """Return tradable/pool masks plus the entry, exit and pyramid-add signal matrices."""
    close = panel["close"]
    close_df = pd.DataFrame(close)
    n = int(cfg["rsi_len"])
    rsi = wilder_rsi(close_df, n).to_numpy()
    sma_gate = close_df.rolling(int(cfg["sma_window"]), min_periods=int(cfg["sma_window"])).mean().to_numpy()
    if cfg.get("gate", "sma200") == "ret21":
        w = int(cfg.get("gate_ret_window", 21))
        gate = np.full(close.shape, False)
        gate[w:] = close[w:] / close[:-w] - 1.0 > 0.0
        gate &= np.isfinite(close)
    else:
        gate = close > sma_gate
    sma_exit = close_df.rolling(int(cfg["exit_sma"]), min_periods=int(cfg["exit_sma"])).mean().to_numpy()
    history = np.isfinite(close).cumsum(0)
    tradable = np.isfinite(close) & (history >= int(cfg["min_history"]))
    if cfg.get("exit_mode", "sma5") == "rsi":
        exit_now = rsi > float(cfg["exit_rsi"])
    else:
        exit_now = close > sma_exit
    signal = tradable & gate & (rsi < float(cfg["rsi_threshold"]))
    add_signal = (rsi < float(cfg.get("add_rsi", 5))) if cfg.get("sizing", "single") == "pyramid3" else None
    atr = None
    if float(cfg.get("disaster_stop_atr", 0.0)) > 0 and high is not None and low is not None:
        atr = wilder_atr(pd.DataFrame(high), pd.DataFrame(low), close_df, int(cfg["atr_len"])).to_numpy()
    return {"rsi": rsi, "signal": signal, "exit_now": exit_now, "add_signal": add_signal, "atr": atr,
            "tradable": tradable, "pool": tradable}


def build_trades(panel, sig, cfg, fill):
    from src.strategies._engines import swing_trades

    return swing_trades(panel, sig["signal"], sig["exit_now"], cfg, fill,
                        add_signal=sig["add_signal"], max_units=int(cfg.get("max_units", 3)),
                        stop_atr=sig["atr"], stop_mult=float(cfg.get("disaster_stop_atr", 0.0)),
                        extras={"rsi2": sig["rsi"]})
