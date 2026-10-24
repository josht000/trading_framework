"""Thin adapter around the existing ATR-pivot study modules."""

from pathlib import Path

import pandas as pd

import main as legacy_main
import rank_final
import rank_test
from src.framework.base import Setup, Strategy
from src.framework.registry import register
from src.utils import timing


@register("atr_pivot")
class AtrPivotStrategy(Strategy):
    name = "atr_pivot"
    needs = ("daily", "minute")
    exposure_window = {"entry": "15:55", "exit": "next-session"}

    def build_trades(self, ctx):
        legacy_main.run_pit_backtest()
        tag = ctx.cfg.get("results_tag", "")
        prefix = "atr_pivot" + (f"_{tag}" if tag else "")
        variants = ctx.cfg.get("exit_variants", "v2")
        names = list(__import__("src.backtest.exit_variants", fromlist=["VARIANTS"]).VARIANTS)
        names = names if variants == "all" else [x.strip() for x in variants.split(",") if x.strip()]
        frames = []
        for name in names:
            for holdout, period in ((False, "dev"), (True, "test")):
                path = Path("results") / ("holdout" if holdout else "") / f"pit_trades_{prefix}_{name}.csv"
                if path.exists():
                    frame = pd.read_csv(path)
                    frame["variant"] = name
                    frame["period"] = period
                    frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def build_table(self, ctx, rebuild):
        return rank_test.build_table(rebuild)

    def setups(self, ctx, table):
        defs, prep, ridge = rank_final.strategies(table)
        ctx.data["rank_defs"] = defs
        ctx.data["rank_prep"] = prep
        ctx.data["ridge"] = ridge
        return {
            name: Setup(
                label=name,
                select=lambda frame, name=name: defs[name][0](prep(frame, ridge)),
                rank_col=rank,
                model=ridge,
            )
            for name, (_, rank) in defs.items()
        }

    def rank_experiments(self, ctx, table):
        rank_test.run_experiments(table, int(ctx.cfg.get("iters", 700)), final=False)
        if ctx.flags.get("test"):
            rank_test.run_experiments(table, final=True)

    def charts(self, ctx, table, setups):
        defs, prep, ridge = ctx.data["rank_defs"], ctx.data["rank_prep"], ctx.data["ridge"]
        rank_final.charts(table)
        rank_final.winners_report(table, defs, prep, ridge)
