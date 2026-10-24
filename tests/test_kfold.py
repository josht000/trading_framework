import itertools

import numpy as np
import pandas as pd
import pytest

from src.backtest import kfold
from src.backtest import partition


def _trades(days, returns):
    rows = []
    for d, ret in zip(days, returns):
        for j in range(5):
            rows.append({
                "symbol": f"S{j}_{d.day}", "entry_time": f"{d.date()} 15:55:00-05:00",
                "exit_time": f"{d.date()} 09:30:00-05:00", "ret": ret,
                "rel_vol": 5 - j, "prev_close": 10 + j, "price": 10.5 + j,
                "day": d,
            })
    return pd.DataFrame(rows)


def test_folds_partition_days_once_and_contiguously():
    days = pd.bdate_range("2025-01-02", periods=25)
    folds = kfold.chronological_folds(days)
    assert [len(x) for x in folds] == [5] * 5
    assert pd.DatetimeIndex(np.concatenate([x.to_numpy() for x in folds])).equals(days)
    assert all(folds[i][-1] < folds[i + 1][0] for i in range(4))


def test_positive_configuration_has_five_profitable_folds(monkeypatch):
    monkeypatch.setattr(partition, "window_returns", lambda t, sym: pd.Series(0.0, index=t.index))
    days = pd.bdate_range("2025-01-02", periods=25)
    selected = _trades(days, [.01] * len(days))
    pieces = [selected]
    for i in range(19):
        neg = _trades(days, [-.01] * len(days))
        neg["symbol"] = neg["symbol"] + f"_neg{i}"
        neg["prev_close"] += 100 + i
        neg["price"] += 100 + i
        pieces.append(neg)
    pool = pd.concat(pieces, ignore_index=True)
    rows, summary = kfold.kfold_stability(selected, days, pool=pool, n_draws=100)
    assert summary["folds_profit"] == "5/5"
    assert summary["kfold_p"] < 1e-8
    assert rows["profitable"].all()


def test_poisson_binomial_matches_enumeration():
    p = np.array([.2, .5, .8])
    expected = sum(
        np.prod([p[i] if mask[i] else 1 - p[i] for i in range(3)])
        for mask in itertools.product([False, True], repeat=3)
        if sum(mask) >= 2
    )
    assert kfold.poisson_binomial_tail(p, 2) == pytest.approx(expected)


def test_one_huge_fold_is_concentrated():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(partition, "window_returns", lambda t, sym: pd.Series(0.0, index=t.index))
    try:
        days = pd.bdate_range("2025-01-02", periods=25)
        returns = [1.0] * 5 + [-.1] * 20
        _, summary = kfold.kfold_stability(_trades(days, returns), days, n_draws=100)
        assert summary["folds_profit"] == "1/5"
        assert summary["best_fold_share"] > 1
    finally:
        monkeypatch.undo()
