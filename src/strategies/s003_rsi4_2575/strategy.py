"""S003 RSI(4) 25/75 (Connors): framework adapter (ETF TR daily bars, research scale, long-only)."""

import numpy as np
import pandas as pd

from src.data.daily_panel import load_panel
from src.framework import report
from src.framework.registry import register
from src.strategies import _bookstudy as BS
from src.strategies._bookstudy import DailyStudy

from .signals import build_trades, compute_signals

NAME = "s003_rsi4_2575"
TR_SUFFIX = "_1Day_tr.parquet"


@register(NAME)
class S003Rsi4(DailyStudy):
    name = NAME
    TITLE = "S003 RSI(4) 25/75 (Connors)"
    timing_null = True

    def load(self, ctx, cfg):
        symbols = [x.strip() for x in str(cfg["symbols"]).split(",") if x.strip()]
        panel = load_panel(cfg["data_dir"], symbols=symbols, rebuild=bool(ctx.flags.get("rebuild_cache")),
                           suffix=cfg.get("data_suffix", TR_SUFFIX))
        return panel, {}

    def signals(self, panel, aux, cfg):
        return compute_signals(panel, cfg)

    def trades_for(self, panel, sig, cfg, fill):
        return build_trades(panel, sig, cfg, fill)

    def rank_col(self):
        return "rsi4"

    def extra_summary(self, ctx, table, out):
        rows = []
        for (fill, period, symbol), t in table.groupby(["fill", "period", "symbol"]):
            rows.append({"fill": fill, "period": period, "symbol": symbol, **BS.period_stats(t)})
        by_sym = pd.DataFrame(rows)
        by_sym.to_csv(out / f"{BS.prefix(ctx.cfg)}per_etf.csv", index=False)
        report.table(by_sym.pivot_table(index="symbol", columns=["fill", "period"], values="trades", fill_value=0).reset_index())

    def baselines(self, cfg, state, strat_r):
        ew_idx = np.cumprod(1.0 + state["b_day"])
        sma = pd.Series(ew_idx).rolling(int(cfg["sma_window"]), min_periods=int(cfg["sma_window"])).mean().to_numpy()
        gate = np.concatenate([[False], (ew_idx > sma)[:-1]])
        return {"ew_buy_hold": state["b_day"],
                "sma200_timing": np.where(gate, state["b_day"], 0.0),
                "vol_matched_ew": BS.vol_matched(state["b_day"], strat_r, state["masks"]["dev"])}

    def intro(self, cfg, state):
        return [
            f"- Universe: {cfg['symbols']} (ETF total-return bars); each symbol at most one position, book slots {cfg['book_slots']}.",
            f"- Rule: close > SMA{cfg['sma_window']} AND RSI({cfg['rsi_len']}) < {cfg['rsi_threshold']}; exit RSI({cfg['rsi_len']}) > {cfg['exit_rsi']} (exit level 55 was hindsight-selected in the source - the 75 variant is Connors' original).",
            "- `next_open` is the primary fill; `close` is the same-close UPPER BOUND. `moc_proxy` needs 15:50 ETF minute bars, which do not exist in the repo.",
            "- Baselines: EW buy-and-hold of the universe, SMA200 timing on that EW index (lagged), dev-vol-matched EW, 500-draw exposure-matched random timing.",
        ]

    def caption(self, cfg, state):
        return (f"variant={cfg['variant']} | RSI({cfg['rsi_len']})<{cfg['rsi_threshold']}>{cfg['exit_rsi']} gate=SMA{cfg['sma_window']} | "
                f"{cfg['cost_bps_per_side']}bps/side slots={cfg['book_slots']} | {cfg['data_dir']} {state['dates'][0].date()}..{state['dates'][-1].date()}")
