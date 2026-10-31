import numpy as np
import pandas as pd
import pytest
import yaml

import run as run_module
from src.data.alpaca_history import merge_bar_file, normalize_bars
from src.data.daily_panel import load_panel
from src.data.liquid_universe import load_liquid_ranks, write_liquid_ranks
from src.framework.daily_study import DailyStudy
from src.framework.preflight import check_study_inputs
from src.framework.registry import get_strategy
from src.strategies._engines import rebalance_trades, swing_trades
from src.strategies.s001_rsi2_connors.signals import compute_signals as s001_signals
from src.strategies.s003_rsi4_2575.signals import compute_signals as s003_signals
from src.strategies.s006_double_7s.signals import compute_signals as s006_signals
from src.strategies.s033_52w_breakout.signals import compute_signals as s033_signals
from src.strategies.s041_mom_12_1.signals import compute_signals as s041_signals
from src.strategies.s042_sector_momentum.signals import compute_signals as s042_signals


def panel_from_close(close, symbols=None, dates=None):
    close = np.asarray(close, dtype=float)
    if close.ndim == 1:
        close = close[:, None]
    symbols = symbols or [f"S{i}" for i in range(close.shape[1])]
    dates = pd.DatetimeIndex(dates if dates is not None else pd.bdate_range("2020-01-02", periods=close.shape[0]))
    return {
        "dates": dates,
        "symbols": symbols,
        "open": close * 1.001,
        "close": close,
        "volume": np.full(close.shape, 100_000.0),
    }


def test_daily_studies_register_through_the_framework():
    names = [
        "s001_rsi2_connors", "s003_rsi4_2575", "s006_double_7s",
        "s033_52w_breakout", "s041_mom_12_1", "s042_sector_momentum",
        "s075_residual_reversal",
    ]
    assert all(isinstance(get_strategy(name), DailyStudy) for name in names)


def test_s001_and_s003_use_configured_rsi_period_without_lookahead():
    close = np.column_stack([
        100 * np.exp(np.cumsum(np.sin(np.arange(80) * 0.8) * 0.01)),
        50 * np.exp(np.cumsum(np.cos(np.arange(80) * 0.6) * 0.01)),
    ])
    panel = panel_from_close(close)
    cfg = {
        "rsi_len": 2, "sma_window": 4, "exit_sma": 3, "exit_mode": "sma5",
        "exit_rsi": 70, "min_history": 1, "gate": "ret21", "gate_ret_window": 1,
        "rsi_threshold": 10, "sizing": "single", "disaster_stop_atr": 0.0, "atr_len": 14,
    }
    base = s001_signals(panel, cfg)
    assert base["rsi"].shape == close.shape
    assert np.isnan(base["rsi"][:2]).all()

    cfg4 = dict(cfg, rsi_len=4, rsi_threshold=25)
    four = s003_signals(panel, cfg4)
    assert np.isnan(four["rsi"][:4]).all()

    changed = close.copy()
    changed[50:] *= 1.5
    future = s001_signals(panel_from_close(changed), cfg)
    assert np.array_equal(base["signal"][:50], future["signal"][:50])


def test_s006_double_seven_uses_inclusive_rolling_extrema():
    close = np.array([10, 9, 8, 9, 10], dtype=float)
    panel = panel_from_close(close)
    sig = s006_signals(panel, {"lookback": 3, "sma_window": 2, "min_history": 1})
    assert sig["roll_low"][2, 0] == 8
    assert sig["roll_high"][2, 0] == 10
    assert sig["exit_now"][4, 0]
    assert sig["tradable"][0, 0]


def test_s033_breakout_accepts_equal_highs_and_respects_pool_masks():
    close = np.array([[1, 1], [2, 2], [3, 3], [3, 2], [2, 3]], dtype=float)
    panel = panel_from_close(close, ["ETF", "STK"])
    rank = np.ones(close.shape)
    cfg = {"entry_window": 3, "exit_window": 2, "min_history": 2, "universe_top_n": 1}
    sig = s033_signals(panel, cfg, rank, np.array([0]))
    assert sig["signal"][2].all()
    assert sig["signal"][3, 0]
    assert not sig["signal"][3, 1]
    assert sig["exit_now"][4, 0]
    assert not sig["tradable"][0].any()


def test_s041_12_1_score_skips_latest_month_and_rebalances_at_month_end():
    dates = pd.bdate_range("2020-01-01", periods=270)
    t = np.arange(len(dates), dtype=float)
    close = np.column_stack([10 * (1 + 0.001 * t), 20 * (1 + 0.002 * t)])
    panel = panel_from_close(close, ["A", "B"], dates)
    rank = np.ones(close.shape)
    cfg = {
        "mom_window": 252, "skip_window": 21, "universe_top_n": 2,
        "industry_neutral": False, "breadth_mode": "top", "breadth": 1,
        "min_breadth": 1, "decile": 0.1,
    }
    sig = s041_signals(panel, cfg, rank)
    assert sig["score"][252, 0] == pytest.approx(close[231, 0] / close[0, 0] - 1)
    expected = np.searchsorted(dates.values, pd.Series(dates).groupby(dates.to_period("M")).max().to_numpy())
    assert np.array_equal(sig["rebalance_rows"], expected)


def test_s042_sector_momentum_limits_tradable_names_and_uses_monthly_marks():
    dates = pd.bdate_range("2020-01-01", periods=270)
    t = np.arange(len(dates), dtype=float)
    close = np.column_stack([10 * (1 + 0.001 * t), 20 * (1 + 0.002 * t)])
    panel = panel_from_close(close, ["XLB", "SPY"], dates)
    sig = s042_signals(
        panel,
        {"signal": "mom12", "min_history": 1, "rebalance": "monthly"},
        np.array([0]),
    )
    assert sig["tradable"][:, 0].all()
    assert not sig["tradable"][:, 1].any()
    assert sig["score"][252, 0] == pytest.approx(close[252, 0] / close[0, 0] - 1)
    assert len(sig["rebalance_rows"]) > 8


def test_shared_trade_engines_preserve_fill_timing_and_rebalance_selection():
    dates = pd.bdate_range("2021-01-04", periods=6)
    close = np.array([[10, 10], [9, 10], [10, 10], [11, 10], [12, 11], [13, 12]], dtype=float)
    panel = panel_from_close(close, ["A", "B"], dates)
    signal = np.zeros(close.shape, dtype=bool)
    exits = np.zeros(close.shape, dtype=bool)
    signal[1, 0] = True
    exits[3, 0] = True
    cfg = {"cost_bps_per_side": 0.0, "book_slots": 1}
    close_trade = swing_trades(panel, signal, exits, cfg, "close").iloc[0]
    open_trade = swing_trades(panel, signal, exits, cfg, "next_open").iloc[0]
    assert (close_trade.entry_idx, close_trade.exit_idx) == (1, 3)
    assert (open_trade.entry_idx, open_trade.exit_idx) == (2, 4)

    score = np.array([[1, 2]] * len(dates), dtype=float)
    tradable = np.ones(close.shape, dtype=bool)
    trades = rebalance_trades(panel, score, tradable, np.array([0, 3]), 1, cfg, "close")
    assert trades.iloc[0].symbol == "B"
    assert trades.iloc[0].exit_reason == "rebalance"


def test_liquidity_cache_round_trip_and_preflight_manifest(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=3)
    ranks = np.array([[1, 2], [2, 1], [1, 2]], dtype=np.float32)
    rank_dir = tmp_path / "ranks"
    write_liquid_ranks(rank_dir, dates, ["A", "B"], ranks)
    loaded_dates, symbols, loaded_ranks = load_liquid_ranks(rank_dir)
    assert loaded_dates.equals(dates)
    assert symbols == ["A", "B"]
    assert np.array_equal(loaded_ranks, ranks)

    bars = tmp_path / "bars"
    bars.mkdir()
    (bars / "A_1Day_tr.parquet").touch()
    manifest = check_study_inputs(
        "smoke",
        {"data_dir": str(bars), "data_suffix": "_1Day_tr.parquet", "symbols": "A", "ranks_dir": str(rank_dir)},
        root=tmp_path,
        results_dir=tmp_path / "results",
    )
    assert manifest["inputs"]["data_dir"]["symbols"] == 1
    assert manifest["inputs"]["ranks_dir"]["symbols"] == 2
    assert (tmp_path / "results" / "smoke" / "input_manifest.json").is_file()
    with pytest.raises(FileNotFoundError, match="missing"):
        check_study_inputs(
            "smoke",
            {"data_dir": str(bars), "data_suffix": "_1Day_tr.parquet", "symbols": "MISSING"},
            root=tmp_path,
        )


def test_alpaca_daily_bar_normalization_and_merge(tmp_path):
    path = tmp_path / "AAA_1Day_tr.parquet"
    first = normalize_bars([{
        "t": "2024-01-02T00:00:00Z", "o": 10, "h": 12, "l": 9, "c": 11,
        "v": 1000, "n": 20, "vw": 10.5,
    }], "AAA")
    merge_bar_file(path, first)
    update = normalize_bars([{
        "t": "2024-01-02T00:00:00Z", "o": 10, "h": 13, "l": 9, "c": 12,
        "v": 1200, "n": 25, "vw": 11.0,
    }], "AAA")
    merge_bar_file(path, update)
    saved = pd.read_parquet(path)
    assert len(saved) == 1
    assert saved.iloc[0]["close"] == 12


def test_daily_panel_accepts_naive_local_date_indexes(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_CACHE", "0")
    dates = pd.date_range("2020-03-09", periods=3, freq="B")
    pd.DataFrame({"open": [10.0, 11.0, 12.0], "close": [10.5, 11.5, 12.5],
                  "volume": [100.0, 110.0, 120.0]}, index=dates).to_parquet(
        tmp_path / "AAA_1Day_yahoo.parquet"
    )
    panel = load_panel(tmp_path, symbols=["AAA"], suffix="_1Day_yahoo.parquet")
    assert list(panel["dates"].date) == list(dates.date)
    assert panel["close"][:, 0].tolist() == [10.5, 11.5, 12.5]


def test_run_check_data_writes_manifest_without_simulating(tmp_path, monkeypatch):
    bars = tmp_path / "bars"
    bars.mkdir()
    (bars / "A_1Day_tr.parquet").touch()
    config = tmp_path / "study.yaml"
    config.write_text(
        yaml.safe_dump({
            "data_dir": "bars",
            "data_suffix": "_1Day_tr.parquet",
            "symbols": "A",
        }),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    run_module.main([
        "--strategy", "s001_rsi2_connors", "--config", str(config), "--check-data",
    ])
    out = tmp_path / "results" / "s001_rsi2_connors"
    assert (out / "input_manifest.json").is_file()
    assert not (out / "partition_dev.csv").exists()
