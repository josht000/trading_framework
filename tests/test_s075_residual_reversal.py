from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import run as run_module
from src.framework import daily_partition as DP
from src.framework.base import RunContext, Strategy
from src.framework.pipeline import run as run_pipeline
from src.strategies.s075_residual_reversal import signals as S

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "s075_residual_reversal.yaml").read_text())


def small_cfg(**kw):
    cfg = dict(CFG, universe_top_n=30, peer_top_n=40, min_price=1.0, min_history=50, sma_window=20, n_draws=20, n_boot=50)
    cfg.update(kw)
    return cfg


def synthetic_panel(n_sym=40, n_rows=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_rows)
    ret = rng.normal(0.0003, 0.015, (n_rows, n_sym))
    close = 20 * np.cumprod(1 + ret, axis=0)
    open_ = close * (1 + rng.normal(0, 0.003, close.shape))
    volume = rng.uniform(1e5, 1e6, close.shape)
    symbols = [f"S{i:02d}" for i in range(n_sym)]
    sector_map = pd.DataFrame({"Industry": [f"I{i % 4}" for i in range(n_sym)], "Sector": ["X"] * n_sym}, index=symbols)
    return {"dates": dates, "symbols": symbols, "open": open_, "close": close, "volume": volume}, sector_map


def test_peer_returns_leave_one_out_and_min_peers():
    ret = np.array([[0.1, 0.2, 0.3, 0.4]])
    groups = np.array([0, 0, 0, 1])
    out = S.peer_returns(ret, np.ones_like(ret, bool), groups, min_peers=2)
    assert out[0, :3] == pytest.approx([0.25, 0.2, 0.15])
    assert np.isnan(out[0, 3])


def test_rolling_beta_recovers_slope_and_uses_shift():
    x = np.random.default_rng(1).normal(0, 0.01, (120, 2))
    beta = S.rolling_beta(2.5 * x, x, 60, 40)
    assert np.nanmax(np.abs(beta[59:] - 2.5)) < 1e-6
    assert np.isnan(beta[:39]).all()
    shifted = S.shift_down(beta)
    assert np.isnan(shifted[0]).all() and shifted[60, 0] == pytest.approx(beta[59, 0])


def test_signals_have_no_lookahead():
    panel, sector_map = synthetic_panel()
    cfg = small_cfg()
    base = S.compute_signals(panel, sector_map, cfg)
    changed = dict(panel, close=panel["close"].copy(), volume=panel["volume"].copy())
    changed["close"][250:] *= 1.5
    changed["volume"][250:] *= 3
    other = S.compute_signals(changed, sector_map, cfg)
    for key in ("resid5", "beta", "signal", "tradable", "pool"):
        a, b = base[key][:250], other[key][:250]
        assert np.array_equal(a, b, equal_nan=True) if a.dtype != bool else np.array_equal(a, b)
    assert base["signal"].any()


def hand_sig(resid, signal_rows, n_rows):
    signal = np.zeros((n_rows, 1), bool)
    signal[signal_rows, 0] = True
    return {"resid5": np.asarray(resid, float).reshape(-1, 1), "beta": np.full((n_rows, 1), 1.0), "signal": signal,
            "industry": np.array(["I"], dtype=object), "sector": np.array(["S"], dtype=object)}


def hand_panel(close, open_=None):
    close = np.asarray(close, float).reshape(-1, 1)
    open_ = close * 0.99 if open_ is None else np.asarray(open_, float).reshape(-1, 1)
    return {"dates": pd.bdate_range("2021-01-04", periods=len(close)), "symbols": ["AAA"], "close": close, "open": open_}


def test_trade_exit_on_residual_recovery_close_and_next_open():
    cfg = small_cfg(cost_bps_per_side=0.0)
    close = [10, 10, 9, 9.5, 9.8, 10.2, 10.4, 10.5, 10.6, 10.7]
    resid = [np.nan, np.nan, -0.1, -0.05, -0.02, 0.01, 0.0, 0.0, 0.0, 0.0]
    panel, sig = hand_panel(close), hand_sig(resid, [2], len(close))
    t = S.build_trades(panel, sig, cfg, "close").iloc[0]
    assert (t.entry_idx, t.exit_idx, t.exit_reason) == (2, 5, "resid")
    assert t.ret == pytest.approx(10.2 / 9 - 1)
    o = S.build_trades(panel, sig, cfg, "next_open").iloc[0]
    assert (o.entry_idx, o.exit_idx) == (3, 6)
    assert o.entry_px == pytest.approx(9.5 * 0.99) and o.exit_px == pytest.approx(10.4 * 0.99)


def test_trade_cap_delisting_and_costs():
    cfg = small_cfg(cost_bps_per_side=10.0, max_hold=3)
    close = [10, 9, 9, 9, 9, 9, 9]
    resid = [-0.1] + [-0.1] * 6
    t = S.build_trades(hand_panel(close), hand_sig(resid, [1], 7), cfg, "close").iloc[0]
    assert (t.exit_idx, t.exit_reason) == (4, "cap")
    assert t.ret == pytest.approx(9 * (1 - 0.001) / (9 * (1 + 0.001)) - 1)
    close = [10, 9, 9.5, np.nan, np.nan, np.nan, np.nan]
    d = S.build_trades(hand_panel(close, close), hand_sig(resid, [1], 7), small_cfg(cost_bps_per_side=0.0), "close").iloc[0]
    assert (d.exit_idx, d.exit_reason) == (2, "delisted")


def test_one_position_per_symbol_until_exit():
    cfg = small_cfg(cost_bps_per_side=0.0)
    close = [10, 9, 9, 9.5, 10, 10, 10, 10]
    resid = [-0.1, -0.1, -0.1, -0.1, 0.1, -0.1, 0.1, 0.1]
    trades = S.build_trades(hand_panel(close), hand_sig(resid, [1, 2, 3, 5], 8), cfg, "close")
    assert list(trades["signal_idx"]) == [1, 5]
    assert list(trades["exit_idx"]) == [4, 6]


def test_account_compounds_to_trade_return():
    close = np.array([[10.0], [9.0], [9.5], [10.2], [10.0]])
    exp = DP.expand_positions([1], [3])
    r, count = DP.account_series(exp, np.array([0]), np.array([9.0]), np.array([10.2]), close, 0.001, 5)
    assert np.prod(1 + r) - 1 == pytest.approx(10.2 * 0.999 / (9.0 * 1.001) - 1)
    assert list(count) == [0, 1, 1, 1, 0]
    r2, _ = DP.account_series(exp, np.array([0]), np.array([9.0]), np.array([10.2]), close, 0.0, 5, slots=4)
    assert r2[2] == pytest.approx((9.5 / 9.0 - 1) / 4)


def test_partition_identity_and_determinism():
    panel, sector_map = synthetic_panel(n_rows=420)
    cfg = small_cfg(cost_bps_per_side=5.0, book_slots=5)
    sig = S.compute_signals(panel, sector_map, cfg)
    trades = S.build_trades(panel, sig, cfg, "close")
    assert len(trades) > 30
    n = len(panel["dates"])
    ret = sig["ret"]
    b = np.where(sig["tradable"], np.nan_to_num(ret), 0.0).sum(1) / np.maximum((sig["tradable"] & ~np.isnan(ret)).sum(1), 1)
    import pandas as pd_
    close_ff = pd_.DataFrame(panel["close"]).ffill().to_numpy()
    book = {"close_ff": close_ff, "entry_mat": panel["close"], "exit_mat": panel["close"], "cost": 0.0005, "n_rows": n, "b_day": b, "slots": 5}
    rows = np.arange(n)
    masks = {"dev": (rows >= 260) & (rows < 350), "test": rows >= 350}
    one, _ = DP.evaluate_fill(trades, book, masks, sig["tradable"], sig["pool"], 1.0, cfg)
    two, _ = DP.evaluate_fill(trades, book, masks, sig["tradable"], sig["pool"], 1.0, cfg)
    for period, res in one.items():
        assert res["skill"] == pytest.approx(res["total_log"] - res["trend"] - res["bias"])
        assert res == two[period]
        assert {"random_pick_universe", "random_pick_pool", "skill_pct", "bias", "trend"} <= set(res)
    assert one["test"]["bias"] == 0.0


def write_synthetic_dir(path, n_sym=40, n_rows=700):
    panel, sector_map = synthetic_panel(n_sym, n_rows)
    index = pd.DatetimeIndex(panel["dates"]).tz_localize("UTC") + pd.Timedelta(hours=5)
    path.mkdir()
    for j, symbol in enumerate(panel["symbols"]):
        frame = pd.DataFrame({"symbol": symbol, "open": panel["open"][:, j], "high": panel["close"][:, j], "low": panel["close"][:, j],
                              "close": panel["close"][:, j], "volume": panel["volume"][:, j]}, index=index)
        frame.index.name = "timestamp"
        frame.to_parquet(path / f"{symbol}_1Day_split.parquet")
    return sector_map


def write_fake_simfin(root):
    path = root / "data" / "simfin"
    path.mkdir(parents=True)
    (path / "us-companies.csv").touch()
    (path / "industries.csv").touch()


def test_run_py_data_dir_switch_end_to_end(tmp_path, monkeypatch):
    data_dir = tmp_path / "bars"
    sector_map = write_synthetic_dir(data_dir)
    write_fake_simfin(tmp_path)
    cfg = small_cfg(data_dir="does/not/exist", partition_split_date="2022-03-01", min_history=100)
    config = tmp_path / "s075.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    import src.strategies.s075_residual_reversal.strategy as strat
    monkeypatch.setattr(strat, "load_sector_map", lambda: sector_map)
    monkeypatch.setenv("DATA_CACHE", "0")
    monkeypatch.chdir(tmp_path)
    ctx = run_module.main(["--strategy", "s075_residual_reversal", "--config", str(config), "--data-dir", str(data_dir)])
    assert ctx.cfg["data_dir"] == str(data_dir)
    part = pd.read_csv(tmp_path / "results" / "s075_residual_reversal" / "partition_dev.csv")
    assert {"trend", "random_pick_pool", "random_pick_universe", "skill", "bias", "skill_pct"} <= set(part.columns)
    assert set(part["fill"]) == {"close", "next_open"}
    assert (tmp_path / "results" / "s075_residual_reversal" / "partition_test.csv").exists()
    assert (tmp_path / "results" / "s075_residual_reversal" / "trades_close.csv").exists()
    assert not (tmp_path / "reports").exists()


def test_missing_data_dir_is_an_error(tmp_path):
    from src.strategies.s075_residual_reversal.strategy import S075ResidualReversal
    ctx = RunContext(cfg=dict(CFG, data_dir=None), paths={}, args=None, flags={})
    with pytest.raises(ValueError, match="data-dir"):
        S075ResidualReversal().build_table(ctx, False)


def test_pipeline_uses_strategy_evaluate_hook():
    class Hooked(Strategy):
        name = "hooked"

        def build_trades(self, ctx):
            return pd.DataFrame()

        def build_table(self, ctx, rebuild):
            return pd.DataFrame()

        def setups(self, ctx, table):
            return {}

        def rank_experiments(self, ctx, table):
            pass

        def charts(self, ctx, table, setups):
            pass

        def evaluate(self, ctx, table, setups):
            return "custom"

    ctx = run_pipeline(RunContext(cfg={}, paths={}, args=None, flags={}), Hooked(), ["partition"])
    assert ctx.data["evaluation"] == "custom"


def test_new_components_default_to_baseline_and_work():
    panel, sector_map = synthetic_panel(n_rows=400)
    base = S.compute_signals(panel, sector_map, small_cfg())
    explicit = S.compute_signals(panel, sector_map, small_cfg(signal_mode="raw", skip_days=0, regime_mode="none", min_abs_resid=0.0))
    assert np.array_equal(base["signal"], explicit["signal"])
    skipped = S.compute_signals(panel, sector_map, small_cfg(skip_days=2))
    assert not skipped["signal"][:2].any()
    assert skipped["signal"].sum() <= base["signal"].sum() + 1
    nosma = S.compute_signals(panel, sector_map, small_cfg(sma_window=0))
    assert nosma["signal"].sum() >= base["signal"].sum()
    z = S.compute_signals(panel, sector_map, small_cfg(signal_mode="zscore"))
    assert z["z"] is not None and z["signal"].any()
    big = S.compute_signals(panel, sector_map, small_cfg(min_abs_resid=0.05))
    assert (big["resid5"][big["signal"]] <= -0.05).all()


def test_regime_block_and_exit_rules():
    b = np.r_[np.zeros(30), np.full(5, -0.03), np.zeros(10)]
    cfg = {"regime_mode": "ew_drawdown", "regime_window": 5, "regime_threshold": 0.10}
    blocked = S.regime_block(b, cfg)
    assert blocked[:30].sum() == 0 and blocked[34]
    assert S.regime_block(b, {"regime_mode": "none"}) is None
    close = [10, 10, 9, 9.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    resid = [np.nan, np.nan, -0.1] + [-0.1] * 7
    sig = hand_sig(resid, [2], len(close))
    stop = S.build_trades(hand_panel(close), sig, small_cfg(cost_bps_per_side=0.0, stop_loss=0.10), "close").iloc[0]
    assert (stop.exit_reason, stop.exit_idx) == ("stop", 4)
    fixed = S.build_trades(hand_panel(close), sig, small_cfg(cost_bps_per_side=0.0, exit_mode="fixed", fixed_hold=3), "close").iloc[0]
    assert (fixed.exit_reason, fixed.exit_idx) == ("cap", 5)
    tp = S.build_trades(hand_panel([10, 10, 9, 9.5, 10.0, 10, 10, 10, 10, 10]), sig, small_cfg(cost_bps_per_side=0.0, take_profit=0.08), "close").iloc[0]
    assert (tp.exit_reason, tp.exit_idx) == ("target", 4)


def test_variant_overlay_and_env_export(tmp_path):
    from src.framework.config import env_overrides, load_config
    path = tmp_path / "c.yaml"
    path.write_text("a: 1\nvariants:\n  v1: {a: 2, exit_mode: fixed}\n", encoding="utf-8")
    cfg = load_config("x", path, env={}, variant="v1")
    assert (cfg["a"], cfg["exit_mode"], cfg["results_tag"], cfg["variant"]) == (2, "fixed", "v1", "v1")
    base = load_config("x", path, env={})
    assert base["a"] == 1 and base["variant"] == "baseline" and not base["results_tag"]
    with pytest.raises(ValueError, match="unknown variant"):
        load_config("x", path, env={}, variant="nope")
    assert "VARIANTS" not in env_overrides(cfg)


def test_run_py_variant_batch_writes_prefixed_files_and_one_png(tmp_path, monkeypatch):
    data_dir = tmp_path / "bars"
    sector_map = write_synthetic_dir(data_dir)
    write_fake_simfin(tmp_path)
    cfg = small_cfg(partition_split_date="2022-03-01", min_history=100, variants={"nosma": {"sma_window": 0}})
    config = tmp_path / "s075.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    import src.strategies.s075_residual_reversal.strategy as strat
    monkeypatch.setattr(strat, "load_sector_map", lambda: sector_map)
    monkeypatch.setenv("DATA_CACHE", "0")
    monkeypatch.chdir(tmp_path)
    run_module.main(["--strategy", "s075_residual_reversal", "--config", str(config), "--data-dir", str(data_dir), "--variant", "baseline,nosma"])
    out = tmp_path / "results" / "s075_residual_reversal"
    assert (out / "partition_dev.csv").exists() and (out / "nosma__partition_dev.csv").exists()
    assert sorted(p.name for p in out.glob("*.png")) == ["nosma__summary.png", "summary.png"]
    comparison = pd.read_csv(out / "variants_comparison.csv")
    assert set(comparison["variant"]) == {"baseline", "nosma"} and "selection_haircut" in comparison


def test_peer_level_sector_vs_industry():
    panel, sector_map = synthetic_panel()
    ind = S.compute_signals(panel, sector_map, small_cfg(peer_level="industry"))
    sec = S.compute_signals(panel, sector_map, small_cfg(peer_level="sector"))
    assert ind["resid5"].shape == sec["resid5"].shape
    assert not np.array_equal(np.nan_to_num(ind["resid5"]), np.nan_to_num(sec["resid5"]))
