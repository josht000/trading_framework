import numpy as np
import pandas as pd
import pytest

from src.strategies.s052_earnings_premium.signals import build_trades, compute_signals


def synthetic_inputs():
    dates = pd.bdate_range("2016-01-04", "2021-12-31")
    symbols = ["AAA", "SPY"]
    close = np.column_stack([
        np.full(len(dates), 20.0) * np.cumprod(1 + np.sin(np.arange(len(dates)) / 19) * 0.001),
        np.full(len(dates), 100.0) * np.cumprod(1 + np.cos(np.arange(len(dates)) / 23) * 0.001),
    ])
    open_ = close * np.column_stack([
        1 + np.sin(np.arange(len(dates)) / 7) * 0.002,
        1 + np.cos(np.arange(len(dates)) / 9) * 0.002,
    ])
    volume = np.full(close.shape, 1000.0)
    panel = {"dates": dates, "symbols": symbols, "open": open_, "close": close, "volume": volume}
    price_panel = {
        "dates": dates,
        "symbols": ["AAA"],
        "open": open_[:, :1],
        "close": close[:, :1],
        "volume": volume[:, :1],
    }
    events = pd.to_datetime(["2019-04-15", "2019-07-15", "2019-10-15", "2020-01-15"])
    aux = {"event_dates": {"AAA": events.to_numpy()}, "share_reports": {}}
    cfg = {
        "min_price": 5.0, "min_history": 1, "cadence": "monthly",
        "volume_lookback_months": 48, "volume_lag_months": 3,
        "volume_quintile": 0.20, "sort_mode": "volume", "weighting": "equal",
        "leg": "long_spy_hedge", "cost_bps_per_side": 5.0, "delisting_return": -0.30,
    }
    return panel, price_panel, aux, cfg


def test_monthly_signal_uses_only_prior_publish_dates_and_lagged_volume():
    panel, price_panel, aux, cfg = synthetic_inputs()
    sig = compute_signals(panel, price_panel, aux, cfg)
    dates = pd.DatetimeIndex(panel["dates"])
    march = int(np.flatnonzero(dates == pd.Timestamp("2020-03-31"))[0])
    assert march in sig["selected"]
    assert sig["selected"][march]["long_idx"].tolist() == [0]
    assert sig["selected"][march]["spy_weight"] == -1.0

    changed = dict(price_panel, volume=price_panel["volume"].copy())
    changed["volume"][dates > pd.Timestamp("2020-01-31")] *= 100
    future_events = dict(aux, event_dates={"AAA": np.r_[aux["event_dates"]["AAA"], np.datetime64("2021-04-15")]})
    future = compute_signals(panel, changed, future_events, cfg)
    assert future["selected"][march]["long_idx"].tolist() == [0]


def test_monthly_trades_have_balanced_spy_hedge_and_registered_costs():
    panel, price_panel, aux, cfg = synthetic_inputs()
    sig = compute_signals(panel, price_panel, aux, cfg)
    trades = build_trades(panel, sig, cfg, "next_open")
    assert len(trades) > 0
    first = trades[trades["signal_idx"] == trades["signal_idx"].min()]
    assert set(first["symbol"]) == {"AAA", "SPY"}
    assert first.loc[first["symbol"] == "AAA", "trade_weight"].sum() == pytest.approx(1.0)
    assert first.loc[first["symbol"] == "SPY", "trade_weight"].sum() == pytest.approx(-1.0)
    assert first["entry_idx"].iloc[0] == first["signal_idx"].iloc[0] + 1
    assert (first["ret"] <= first["ret_gross"]).all()
