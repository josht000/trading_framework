from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.framework import Setup, get_strategy
from src.framework.base import RunContext, Strategy
from src.framework.config import load_config
from src.framework.evaluate import evaluate
from src.framework.pipeline import run
from src.framework.report import table


def test_registry_lookup_and_unknown_error():
    assert get_strategy("atr_pivot").name == "atr_pivot"
    with pytest.raises(ValueError, match="atr_pivot"):
        get_strategy("does_not_exist")


def test_config_precedence(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("results_tag: yaml\nlookback_bars: 10\n", encoding="utf-8")
    cfg = load_config("atr_pivot", path, env={"RESULTS_TAG": "env"}, cli={"results_tag": "cli"})
    assert cfg["results_tag"] == "cli"
    assert cfg["lookback_bars"] == 10
    assert load_config("atr_pivot", path, env={})["results_tag"] == "yaml"


def test_pipeline_stage_order_and_subset():
    class Dummy(Strategy):
        name = "dummy"
        needs = ()

        def __init__(self):
            self.calls = []

        def build_trades(self, ctx):
            self.calls.append("trades")
            return pd.DataFrame()

        def build_table(self, ctx, rebuild):
            self.calls.append("table")
            return pd.DataFrame()

        def setups(self, ctx, table):
            self.calls.append("setups")
            return {}

        def rank_experiments(self, ctx, table):
            self.calls.append("rank")

        def charts(self, ctx, table, setups):
            self.calls.append("charts")

    dummy = Dummy()
    ctx = RunContext(cfg={}, paths={}, args=None, flags={})
    run(ctx, dummy, ["rank"])
    assert dummy.calls == ["table", "setups", "rank"]


def test_report_table_smoke(capsys):
    table(pd.DataFrame({"setup": ["base"], "score": [1.25]}), title="smoke")
    assert "smoke" in capsys.readouterr().out


def test_evaluate_tiny_partition(monkeypatch):
    from src.backtest import partition as P
    monkeypatch.setattr(P, "window_returns", lambda trades, sym: pd.Series(0.0, index=trades.index))
    days = pd.bdate_range("2025-01-02", periods=5)
    rows = []
    for period, offset in (("dev", 0), ("test", 10)):
        for i, day in enumerate(days):
            for symbol in ("AAA", "BBB"):
                rows.append({
                    "symbol": symbol,
                    "period": period,
                    "day": day,
                    "entry_time": f"{day.date()} 15:55",
                        "exit_time": f"{(day + pd.offsets.BDay(1)).date()} 09:30",
                        "ret": 0.01 + offset * 0.0,
                        "rel_vol": float(i + 1),
                        "prev_close": 10.0,
                        "price": 10.5,
                })
    frame = pd.DataFrame(rows)
    setups = {"base": Setup("base", lambda x: x, "rel_vol")}
    result = evaluate(RunContext({}, {}, None), frame, setups)
    assert set(result["results"]) == {("base", "dev"), ("base", "test")}
    assert {"total_log", "trend_iwm", "skill", "bias"}.issubset(result["results"][("base", "dev")]["partition"])
