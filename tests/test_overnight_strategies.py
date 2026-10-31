import numpy as np
import pandas as pd
import pytest

from src.strategies._overnight import build_overnight_trades
from src.strategies.s021_index_overnight.signals import compute_signals as s021_signals
from src.strategies.s105_overnight_carry.signals import compute_signals as s105_signals


def small_panel(n=80):
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = (100 * np.cumprod(1 + np.sin(np.arange(n) / 5) * 0.002))[:, None]
    open_ = np.vstack([close[0], close[:-1] * (1 + np.cos(np.arange(n - 1) / 4)[:, None] * 0.003)])
    return {"dates": dates, "symbols": ["SPY"], "open": open_, "close": close, "volume": np.ones_like(close)}


def test_s021_unconditional_and_sma_gate_use_completed_data_only():
    panel = small_panel()
    overnight = np.r_[np.nan, np.linspace(-0.002, 0.003, len(panel["dates"]) - 1)]
    entry = np.vstack([np.full((1, 1), np.nan), panel["close"][:-1]])
    exit_ = entry * (1 + overnight[:, None])
    aux = {"overnight_return": overnight[:, None], "overnight_entry": entry,
           "overnight_exit": exit_, "total_return": np.zeros(len(overnight))}
    base = s021_signals(panel, aux, {"symbol": "SPY", "participation_filter": "none"})
    gated = s021_signals(panel, aux, {"symbol": "SPY", "participation_filter": "sma50"})
    assert base["trade_weights"][1:, 0].tolist() == [1.0] * (len(overnight) - 1)
    assert not gated["trade_weights"][:50, 0].any()
    changed = dict(panel, close=panel["close"].copy())
    changed["close"][60:] *= 2
    future = s021_signals(changed, aux, {"symbol": "SPY", "participation_filter": "sma50"})
    assert np.array_equal(gated["trade_weights"][:60], future["trade_weights"][:60])


def test_s105_targets_overnight_vol_and_respects_leverage_cap():
    panel = small_panel()
    rng = np.random.default_rng(2)
    overnight = np.r_[np.nan, rng.normal(0.0002, 0.01, len(panel["dates"]) - 1)]
    entry = np.vstack([np.full((1, 1), np.nan), panel["close"][:-1]])
    exit_ = entry * (1 + overnight[:, None])
    aux = {"overnight_return": overnight[:, None], "overnight_entry": entry,
           "overnight_exit": exit_, "total_return": np.zeros(len(overnight))}
    cfg = {"vol_window": 20, "target_vol": 0.10, "max_leverage": 1.0, "resize_band": 0.20}
    sig = s105_signals(panel, aux, cfg)
    weights = sig["trade_weights"][:, 0]
    assert np.nanmax(weights) <= 1.0
    assert np.nanmin(weights) >= 0.0
    assert np.flatnonzero(weights > 0)[0] >= 21
    changed = overnight.copy()
    changed[60:] *= 10
    changed_exit = entry * (1 + changed[:, None])
    future_aux = dict(aux, overnight_return=changed[:, None], overnight_exit=changed_exit)
    future = s105_signals(panel, future_aux, cfg)
    assert np.array_equal(weights[:60], future["trade_weights"][:60, 0])


def test_overnight_trades_charge_both_sides_on_the_overnight_interval():
    panel = small_panel(3)
    sig = {
        "trade_weights": np.array([[0.0], [1.0], [0.0]]),
        "overnight_entry": np.array([[np.nan], [100.0], [101.0]]),
        "overnight_exit": np.array([[np.nan], [101.0], [102.0]]),
    }
    trades = build_overnight_trades(panel, sig, {"cost_bps_per_side": 5.0}, "close_to_open")
    assert len(trades) == 1
    assert trades.iloc[0]["entry_idx"] == trades.iloc[0]["exit_idx"] == 1
    assert trades.iloc[0]["ret"] == pytest.approx(0.01 - 2 * 0.0005)
