"""S021 unconditional and SMA-gated close-to-open returns."""

import numpy as np
import pandas as pd


def compute_signals(panel, aux, cfg):
    close = panel["close"][:, 0]
    overnight = aux["overnight_return"][:, 0]
    n_rows, n_symbols = panel["close"].shape
    trade_weights = np.zeros((n_rows, n_symbols), dtype=float)
    tradable = np.zeros((n_rows, n_symbols), dtype=bool)
    valid = np.isfinite(overnight)
    if cfg.get("participation_filter", "none") == "sma50":
        sma = pd.Series(close).rolling(50, min_periods=50).mean().to_numpy()
        allowed = np.zeros(n_rows, dtype=bool)
        allowed[1:] = close[:-1] > sma[:-1]
        valid &= allowed
    trade_weights[valid, 0] = 1.0
    tradable[1:, 0] = True
    return {
        "tradable": tradable,
        "pool": tradable.copy(),
        "trade_weights": trade_weights,
        "overnight_entry": aux["overnight_entry"],
        "overnight_exit": aux["overnight_exit"],
        "benchmark_returns": np.nan_to_num(overnight),
        "trend_exposure": trade_weights[:, 0],
        "benchmarks": {
            f"{cfg['symbol']} overnight": np.nan_to_num(overnight),
            f"{cfg['symbol']} total return": np.nan_to_num(aux["total_return"]),
        },
    }
