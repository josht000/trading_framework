"""Generic account, k-fold, and partition evaluation."""

import importlib

import pandas as pd

from src.backtest import kfold as K
from src.backtest import partition as P
from src.backtest.pit_metrics import simulate_account


def _generic(ctx, table, setups):
    out = {}
    for name, setup in setups.items():
        for period in ("dev", "test"):
            frame = table[table["period"] == period].copy()
            selected = setup.select(frame)
            days = pd.bdate_range(frame["day"].min(), frame["day"].max())
            account = simulate_account(selected, max_pos=5, rank=setup.rank_col, all_days=days)
            krows, ksummary = K.kfold_stability(selected, days, rank=setup.rank_col, pool=selected, label=f"{name}|{period}")
            part = P.run_partition(selected, days, rank=setup.rank_col, pool=selected, areas=["signal", "exit"], oos=period == "test")
            out[(name, period)] = {"account": account[0], "kfold": ksummary, "partition": part, "kfold_rows": krows}
    return out


def evaluate(ctx, trades_or_table, setups):
    """Run one complete evaluation and return account/k-fold/partition results."""
    table = trades_or_table
    if isinstance(table, pd.DataFrame) and {"period", "day", "is_otc"}.issubset(table.columns):
        report = importlib.import_module("scripts.partition_report")
        names = getattr(report, "DEFAULT", list(setups))
        results = report.compute(table, names)
        report.write(results)
        return {"results": results, "setups": setups}
    return {"results": _generic(ctx, table, setups), "setups": setups}
