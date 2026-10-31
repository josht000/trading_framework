"""S033 52-Week-High Breakout (Donchian-style): framework adapter (ETF + stock TR bars, pooled, long-only)."""

import numpy as np
import pandas as pd

from src.data.daily_panel import load_panel
from src.framework import report
from src.framework.registry import register
from src.strategies import _bookstudy as BS
from src.strategies._bookstudy import DailyStudy
from src.strategies._engines import align_ranks, merge_panels
from tools.build_liquid_ranks import load_liquid_ranks

from .signals import build_trades, compute_signals

NAME = "s033_52w_breakout"
TR_SUFFIX = "_1Day_tr.parquet"


@register(NAME)
class S033Breakout(DailyStudy):
    name = NAME
    TITLE = "S033 52-Week-High Breakout"
    timing_null = True

    def load(self, ctx, cfg):
        rebuild = bool(ctx.flags.get("rebuild_cache"))
        suffix = cfg.get("data_suffix", TR_SUFFIX)
        etfs = [x.strip() for x in str(cfg["etf_symbols"]).split(",") if x.strip()]
        with __import__("src.utils.timing", fromlist=["timing"]).stage("panels"):
            etf_panel = load_panel(cfg["etf_data_dir"], symbols=etfs, rebuild=rebuild, suffix=suffix)
            stock_panel = load_panel(cfg["data_dir"], rebuild=rebuild, suffix=suffix)
            panel = merge_panels(etf_panel, stock_panel) if cfg.get("include_stocks", True) else etf_panel
        rank_dates, rank_symbols, rank = load_liquid_ranks(cfg.get("ranks_dir", "data/cache/liquid_ranks"))
        rank = align_ranks(panel["dates"], panel["symbols"], rank_dates, rank_symbols, rank)
        etf_cols = np.array([panel["symbols"].index(s) for s in etf_panel["symbols"]])
        return panel, {"rank": rank, "etf_cols": etf_cols}

    def signals(self, panel, aux, cfg):
        return compute_signals(panel, cfg, aux["rank"], aux["etf_cols"])

    def trades_for(self, panel, sig, cfg, fill):
        return build_trades(panel, sig, cfg, fill)

    def rank_col(self):
        return "ret_gross"

    def extra_summary(self, ctx, table, out):
        etf_set = set(str(ctx.cfg["etf_symbols"]).split(","))
        table = table.assign(kind=np.where(table["symbol"].isin(etf_set), "etf", "stock"))
        rows = []
        for (fill, period, kind), t in table.groupby(["fill", "period", "kind"]):
            rows.append({"fill": fill, "period": period, "kind": kind, **BS.period_stats(t)})
        by_kind = pd.DataFrame(rows)
        by_kind.to_csv(out / f"{BS.prefix(ctx.cfg)}per_kind.csv", index=False)
        report.table(by_kind, floatfmt=".3f")

    def baselines(self, cfg, state, strat_r):
        ew_idx = np.cumprod(1.0 + state["b_day"])
        sma = pd.Series(ew_idx).rolling(200, min_periods=200).mean().to_numpy()
        gate = np.concatenate([[False], (ew_idx > sma)[:-1]])
        return {"ew_buy_hold": state["b_day"],
                "sma200_timing": np.where(gate, state["b_day"], 0.0),
                "vol_matched_ew": BS.vol_matched(state["b_day"], strat_r, state["masks"]["dev"])}

    def intro(self, cfg, state):
        return [
            f"- Universe: {cfg['etf_symbols']} + liquid_top_{cfg['universe_top_n']} stocks (pooled; ETF TR + stock TR bars); "
            f"book slots {cfg['book_slots']}.",
            f"- Rule: close >= highest close of last {cfg['entry_window']} sessions (inclusive); exit close <= lowest close of last {cfg['exit_window']} sessions. No stop.",
            "- `next_open` is the primary fill; `close` is the same-close UPPER BOUND.",
            "- Delisting convention: a held name whose series ends mid-sample exits at last close x (1 - 0.30) (catalog performance-delisting assumption).",
            "- Baselines: EW universe buy-and-hold, lagged SMA200 timing on the EW index, dev-vol-matched EW, exposure-matched random timing.",
            "- Limitations: index-only power is ~14 trades since 2000 (per the source) - the pooled ETF+stock book supplies the sample.",
        ]

    def caption(self, cfg, state):
        return (f"variant={cfg['variant']} | {cfg['entry_window']}d-high / {cfg['exit_window']}d-low breakout, ETF+top{cfg['universe_top_n']} stocks | "
                f"{cfg['cost_bps_per_side']}bps/side slots={cfg['book_slots']} | {state['dates'][0].date()}..{state['dates'][-1].date()}")
