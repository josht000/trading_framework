"""S006 Double 7s: framework adapter (ETF TR daily bars, pooled, research scale, long-only)."""

import numpy as np
import pandas as pd

from src.data.daily_panel import load_panel
from src.framework import report
from src.framework.registry import register
from src.strategies import _bookstudy as BS
from src.strategies._bookstudy import DailyStudy
from src.strategies._engines import load_field

from .signals import build_trades, compute_signals

NAME = "s006_double_7s"
TR_SUFFIX = "_1Day_tr.parquet"


@register(NAME)
class S006Double7s(DailyStudy):
    name = NAME
    TITLE = "S006 Double 7s"
    timing_null = True

    def load(self, ctx, cfg):
        suffix = cfg.get("data_suffix", TR_SUFFIX)
        symbols = [x.strip() for x in str(cfg["symbols"]).split(",") if x.strip()]
        panel = load_panel(cfg["data_dir"], symbols=symbols, rebuild=bool(ctx.flags.get("rebuild_cache")), suffix=suffix)
        aux = {}
        if float(cfg.get("disaster_stop_atr", 0.0)) > 0:
            aux["high"] = load_field(cfg["data_dir"], panel["symbols"], panel["dates"], "high", suffix)
            aux["low"] = load_field(cfg["data_dir"], panel["symbols"], panel["dates"], "low", suffix)
        return panel, aux

    def signals(self, panel, aux, cfg):
        sig = compute_signals(panel, cfg)
        if float(cfg.get("disaster_stop_atr", 0.0)) > 0 and "high" in aux:
            from src.strategies.s001_rsi2_connors.signals import wilder_atr
            sig["atr"] = wilder_atr(pd.DataFrame(aux["high"]), pd.DataFrame(aux["low"]),
                                    pd.DataFrame(panel["close"]), int(cfg["atr_len"])).to_numpy()
        return sig

    def trades_for(self, panel, sig, cfg, fill):
        return build_trades(panel, sig, cfg, fill)

    def rank_col(self):
        return "ret_gross"

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
            f"- Universe: {cfg['symbols']} pooled (ETF total-return bars); each symbol at most one position, book slots {cfg['book_slots']}. SPY alone gives ~50 trades/decade so the pooled ETF book is the baseline.",
            f"- Rule: close > SMA{cfg['sma_window']} AND close <= lowest close of last {cfg['lookback']} sessions (inclusive, ties count); exit close >= highest close of last {cfg['lookback']} sessions. No stop in the baseline.",
            "- `next_open` is the primary fill; `close` is the same-close UPPER BOUND (moc_proxy needs 15:50 ETF minute bars, which do not exist in the repo).",
            "- Baselines: EW buy-and-hold, SMA200 timing on the EW index (lagged), dev-vol-matched EW, 500-draw exposure-matched random timing.",
        ]

    def caption(self, cfg, state):
        return (f"variant={cfg['variant']} | Double7s lookback={cfg['lookback']} gate=SMA{cfg['sma_window']} stop={cfg.get('disaster_stop_atr', 0)}xATR | "
                f"{cfg['cost_bps_per_side']}bps/side slots={cfg['book_slots']} | {cfg['data_dir']} {state['dates'][0].date()}..{state['dates'][-1].date()}")
