import numpy as np
import pandas as pd

from src.strategies.s017_night_trading.signals import compute_signals


def test_s017_formation_is_point_in_time_and_applies_bh_top_quintile():
    dates = pd.bdate_range("2019-10-01", periods=330)
    n = len(dates)
    symbols = [f"S{i}" for i in range(5)] + ["SPY"]
    rng = np.random.default_rng(7)
    overnight = rng.normal(0.0, 0.005, (n, len(symbols)))
    overnight[:, 0] += 0.01
    overnight[0] = np.nan
    close = np.column_stack([20 * np.cumprod(1 + rng.normal(0, 0.002, n)) for _ in symbols])
    open_ = close.copy()
    open_[1:] = close[:-1] * (1 + overnight[1:])
    rank = np.ones_like(close)
    rank[:, -1] = np.inf
    panel = {"dates": dates, "symbols": symbols, "open": open_, "close": close,
             "volume": np.full_like(close, 100_000.0)}
    aux = {
        "rank": rank,
        "overnight_return": overnight,
        "overnight_entry": np.vstack([np.full((1, len(symbols)), np.nan), close[:-1]]),
        "overnight_exit": open_,
        "spy_total_return": np.zeros(n),
    }
    cfg = {
        "universe_top_n": 5, "formation_cadence": "annual", "formation_window": 252,
        "min_t_stat": 2.0, "fdr_q": 0.05, "top_fraction": 0.20,
        "extra_filter": "none", "holding_mode": "overnight_only", "delisting_return": -0.30,
    }
    base = compute_signals(panel, aux, cfg)
    formation_row = np.flatnonzero((dates.year == 2020) & (dates.month == 12))[-1]
    changed = dict(panel, open=panel["open"].copy(), close=panel["close"].copy())
    changed["close"][formation_row + 1:] *= 2
    changed_aux = dict(aux, overnight_return=overnight.copy())
    changed_aux["overnight_return"][formation_row + 1:] += 0.5
    future = compute_signals(changed, changed_aux, cfg)
    assert len(base["formation_summary"]) == 1
    assert base["formation_summary"][0]["universe"] == 5
    assert base["formation_summary"][0]["selected"] <= 1
    assert np.array_equal(base["membership"], future["membership"])
    assert np.nanmax(base["trade_weights"]) <= 1.0
