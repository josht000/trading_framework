import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import rank_test
from src.backtest import pit_metrics
from src.backtest.pit_runner import (
    _atr_pct,
    _feature_views,
    _features,
    _ma_features,
    _news_features,
    _path,
)
from src.framework import data_cache
from src.strategies.humbled_bullish_overnight import HumbledBullishOvernight


def _synthetic():
    rows = []
    for day in pd.bdate_range("2025-01-02", periods=7):
        for i in range(8):
            rows.append({
                "symbol": f"S{i}",
                "entry_time": f"{day.date()} 15:55",
                "prev_close": 10.0 + i,
                "price": 10.5 + i,
                "ret": np.sin(i + day.day) / 10,
                "rank": float(i % 3) if i != 5 else np.nan,
            })
    return pd.DataFrame(rows)


def test_numba_selector_matches_reference():
    frame = _synthetic()
    expected = pit_metrics.select_trades_py(frame, 5, "rank")
    actual = pit_metrics.select_trades(frame, 5, "rank")
    assert_frame_equal(expected, actual)


def test_numba_disabled_uses_reference(monkeypatch):
    frame = _synthetic()
    expected = pit_metrics.select_trades_py(frame, 3, "rank")
    monkeypatch.setenv("NUMBA_DISABLE", "1")
    actual = pit_metrics.select_trades(frame, 3, "rank")
    assert_frame_equal(expected, actual)


def test_numba_selector_matches_ranked_table_fixture():
    frame = pd.read_parquet("reports/ranked_table.parquet").head(600)
    expected = pit_metrics.select_trades_py(frame, 5, "rel_vol")
    actual = pit_metrics.select_trades(frame, 5, "rel_vol")
    assert_frame_equal(expected, actual)


def test_weight_trial_path_matches_reference():
    table = pd.read_parquet("reports/ranked_table.parquet")
    dev = table[table["period"] == "dev"].head(800)
    expected = rank_test._weight_search_py(dev, 12, np.random.default_rng(20261104))
    actual = rank_test._weight_search_numba(dev, 12, np.random.default_rng(20261104))
    pd.testing.assert_frame_equal(expected, actual, check_exact=True)


def test_feature_views_match_reference_on_real_symbols():
    candidates = pd.read_parquet("data/pit/candidates.parquet")
    daily = pd.read_parquet(
        "data/pit/daily_pit.parquet",
        columns=["date", "symbol", "open", "high", "low", "close", "volume", "delisted_utc"],
    )
    symbols = sorted(candidates["symbol"].unique())[:30]
    _, minute_by = data_cache.load_minute(symbols, by_symbol=True)
    daily_by = {s: g.set_index("date").sort_index() for s, g in daily.groupby("symbol")}
    cand_dates = {s: set(pd.DatetimeIndex(g["date"]).date) for s, g in candidates.groupby("symbol")}
    strategy = HumbledBullishOvernight(
        min_volume=500_000, min_price=2.0, max_price=20.0, max_market_cap=2_000_000_000,
        min_relative_volume=2.0, news_lookback_days=3, min_news_articles=1,
        require_market_cap=False, require_news_catalyst=False, resistance_method="atr_pivot",
        min_breakout_atr=0.0, lazy_resistance=True, lookback_bars=273,
    )
    checked = 0
    for symbol in symbols:
        if symbol not in daily_by or symbol not in minute_by:
            continue
        sig = strategy.apply_strategy(daily_by[symbol], minute_by[symbol], {}, entry_dates=cand_dates[symbol])
        entries, exits = sig.index[sig["entry"]], sig.index[sig["exit"]]
        tm, atr = sig.index.time, _atr_pct(daily_by[symbol])
        views = _feature_views(sig, daily_by[symbol])
        for ts in entries:
            day, later = ts.date(), exits[exits > ts]
            prior = atr[atr.index < pd.Timestamp(day)]
            prior_atr = float(prior.iloc[-1]) if len(prior) else float("nan")
            ep = float(sig.at[ts, "close"])
            old = {
                **_features(sig, ts, ep, daily_by[symbol], day, prior_atr),
                **_path(sig, tm, ts, ep, day, later),
                **_ma_features(daily_by[symbol], day, ep),
                **_news_features(None, ts),
            }
            new = {
                **_features(sig, ts, ep, daily_by[symbol], day, prior_atr, views),
                **_path(sig, tm, ts, ep, day, later, views),
                **_ma_features(daily_by[symbol], day, ep, views),
                **_news_features(None, ts),
            }
            for key, expected in old.items():
                actual = new[key]
                if isinstance(expected, float) and np.isnan(expected) and isinstance(actual, float) and np.isnan(actual):
                    continue
                assert expected == actual, (symbol, ts, key, expected, actual)
            checked += 1
    assert checked > 0
