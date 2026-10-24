import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.backtest import partition as P

DAYS = pd.bdate_range("2025-01-02", periods=300)


def make_trades(rng, n_per_day=8, alpha=0.0, beta=0.0, bench_sd=0.01, noise=0.05):
    """Synthetic overnight trades: ret = alpha + beta * bench + noise; benchmark window returns injected via monkeypatch."""
    rows, bench = [], {}
    for d in DAYS:
        b = rng.normal(0, bench_sd)
        bench[d] = b
        for i in range(n_per_day):
            rows.append({"symbol": f"S{i}", "entry_time": f"{d.date()} 15:55:00-05:00", "exit_time": f"{(d + pd.offsets.BDay(1)).date()} 09:30:00-05:00",
                         "ret": alpha + beta * b + rng.normal(0, noise), "rel_vol": rng.random(), "atr_pct": 0.05, "prev_close": 10.0 + i + 0.01 * len(rows), "price": 10.5 + i, "day": d, "b": b})
    return pd.DataFrame(rows), bench


@pytest.fixture
def fake_bench(monkeypatch):
    state = {}

    def wr(trades, sym):
        return trades["b"].astype(float) if "b" in trades else pd.Series(0.0, index=trades.index)
    monkeypatch.setattr(P, "window_returns", wr)
    return state


def test_identity_and_pure_beta(fake_bench):
    t, _ = make_trades(np.random.default_rng(1), beta=1.0, noise=0.0)
    r = P.partition(t, DAYS, bias={"value": 0.0, "src": "t"}, n_draws=200, n_boot=200)
    assert r["total_log"] == pytest.approx(r["trend_iwm"] + r["skill"] + r["bias"], abs=1e-12)
    r2 = P.partition(t, DAYS, bias={"value": 0.03, "src": "t"}, n_draws=200, n_boot=200)
    assert r2["total_log"] == pytest.approx(r2["trend_iwm"] + r2["skill"] + r2["bias"], abs=1e-12)
    assert r["beta_iwm"] == pytest.approx(1.0, abs=0.15)
    assert abs(r["skill"]) < 0.25 * abs(r["trend_iwm"]) + 0.05


def test_injected_alpha(fake_bench):
    t, _ = make_trades(np.random.default_rng(2), alpha=0.004, beta=0.0)
    r = P.partition(t, DAYS, bias={"value": 0.0, "src": "t"}, n_draws=200, n_boot=500)
    # 5 positions x 20% x 0.4% = 0.4% per day
    assert r["skill"] == pytest.approx(0.004 * len(DAYS), rel=0.35)
    assert r["skill_p"] < 0.05


def test_random_pick_no_skill(fake_bench):
    ps, skills = [], []
    for s in range(30):
        t, _ = make_trades(np.random.default_rng(100 + s), n_per_day=8)
        r = P.partition(t, DAYS[:60], bias={"value": 0.0, "src": "t"}, n_draws=300, n_boot=100, seed=s)
        ps.append(r["random_p"])
        skills.append(r["selection_skill"])
    ps = np.array(ps)
    assert 0.3 < ps.mean() < 0.7 and ps.min() < 0.3 < 0.7 < ps.max()
    assert abs(np.mean(skills)) < 0.03


def test_expected_max_noise_matches_monte_carlo():
    rng = np.random.default_rng(3)
    for n in (10, 100, 1000):
        mc = rng.normal(size=(2000, n)).max(1).mean()
        assert P.expected_max_noise(1.0, n) == pytest.approx(mc, rel=0.06)
    assert P.expected_max_noise(0.2, 100) == pytest.approx(0.2 * P.expected_max_noise(1.0, 100))


def _noise_frame(rng, n_days=300):
    days = pd.bdate_range("2025-01-02", periods=n_days)
    return pd.DataFrame({"day": days, "x": rng.normal(0, 0.01, n_days), "y": rng.normal(0, 0.01, (n_days, 200)).tolist()})


def test_walk_forward_bias_detects_overfit():
    rng = np.random.default_rng(4)
    n = 300
    days = pd.bdate_range("2025-01-02", periods=n)
    rules = rng.choice([-1.0, 1.0], size=(200, n))
    ret = rng.normal(0, 0.01, n)
    df = pd.DataFrame({"day": days, "i": np.arange(n), "ret": ret})

    def ev(t, p, dd):
        return float((rules[p, t["i"].to_numpy()] * t["ret"].to_numpy()).sum())

    def tune(t):
        return int(np.argmax([(rules[k, t["i"].to_numpy()] * t["ret"].to_numpy()).sum() for k in range(200)]))

    res = P.walk_forward_bias(df, tune, ev, folds=5, embargo_days=2, n_days_period=n)
    assert res["bias"] > 0 and res["folds"] >= 3
    gap = (res["is_per_day"] - res["oos_per_day"]) * n
    assert res["bias"] == pytest.approx(gap)
    assert res["is_per_day"] > 0.0004 and abs(res["oos_per_day"]) < res["is_per_day"]
    fixed = [P.walk_forward_bias(df, lambda t, k=k: k, ev, folds=5, embargo_days=2, n_days_period=n)["bias"] for k in range(200)]
    assert abs(np.mean(fixed)) < 0.25 * res["bias"]


def test_skill_pct_na_when_total_negative(fake_bench):
    t, _ = make_trades(np.random.default_rng(5), alpha=-0.004)
    r = P.partition(t, DAYS, bias={"value": 0.0, "src": "t"}, n_draws=100, n_boot=100)
    assert r["total_log"] < 0 and np.isnan(r["skill_pct"]) and r["skill_pct_note"]


def test_registry_roundtrip(tmp_path):
    p = str(tmp_path / "reg.jsonl")
    P.log_trials("exit", 3, per_trial_total_log=[0.1, 0.2, 0.3], notes="x", days=100, path=p)
    P.log_trials("ranking", None, notes="unknown", estimated=True, invalid=False, path=p)
    reg = P.load_registry(p)
    assert list(reg["area"]) == ["exit", "ranking"] and reg.loc[0, "n_trials"] == 3
    assert reg.loc[0, "per_trial_total_log"] == [0.1, 0.2, 0.3] and bool(reg.loc[1, "estimated"])
    hb = P.haircut_B(["exit", "ranking"], 100, sd_fallback=0.2, registry=reg)
    assert hb["unknown"] == [] or "ranking" in hb["unknown"]


def test_random_universe_and_pool_are_reported_separately(fake_bench):
    t, _ = make_trades(np.random.default_rng(9), n_per_day=5, alpha=0.01, noise=0.0)
    universe = pd.concat([t, t.assign(symbol=t["symbol"] + "_extra",
                                      ret=-0.01, prev_close=t["prev_close"] + 100,
                                      price=t["price"] + 100)], ignore_index=True)
    r = P.partition(t, DAYS, pool=t, universe=universe, bias={"value": 0.0, "src": "t"},
                    n_draws=200, n_boot=100)
    assert r["days_no_choice_pool"] == len(DAYS)
    assert r["days_no_choice_universe"] == 0
    assert r["random_pick_universe"] < r["total_log"]
    assert np.isnan(P.partition(t.assign(ret=-0.01), DAYS, bias={"value": 0.0, "src": "t"},
                                n_draws=50, n_boot=50)["skill_pct"])


def test_pooled_haircut_reports_effective_trial_count():
    reg = pd.DataFrame([
        {"area": "exit", "n_trials": 20, "per_trial_total_log": list(np.linspace(-.1, .1, 20)),
         "days": 100, "invalid": False},
        {"area": "ranking", "n_trials": 10, "per_trial_total_log": list(np.linspace(-.05, .05, 10)),
         "days": 100, "invalid": False},
    ])
    h = P.haircut_B(["exit", "ranking"], 100, registry=reg)
    assert h["N_eff"] == h["N_raw"] == 30
    assert h["sd"] > 0 and h["value"] >= 0
