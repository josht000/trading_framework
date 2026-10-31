"""S001 RSI(2) Classic (Connors) - cluster control: framework adapter (ETF TR daily bars, research scale, long-only)."""

import numpy as np
import pandas as pd

from src.data.daily_panel import load_panel
from src.framework import DailyStudy, daily_study as BS, report
from src.framework.registry import register
from src.strategies._engines import load_field

from .signals import build_trades, compute_signals

NAME = "s001_rsi2_connors"
TR_SUFFIX = "_1Day_tr.parquet"


@register(NAME)
class S001Rsi2Connors(DailyStudy):
    name = NAME
    TITLE = "S001 RSI(2) Classic (Connors)"
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
        return compute_signals(panel, cfg, high=aux.get("high"), low=aux.get("low"))

    def trades_for(self, panel, sig, cfg, fill):
        return build_trades(panel, sig, cfg, fill)

    def rank_col(self):
        return "rsi2"

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
            f"- Universe: {cfg['symbols']} (ETF total-return bars `{cfg.get('data_suffix', TR_SUFFIX)}`); each symbol at most one position, book slots {cfg['book_slots']}.",
            f"- Rule: close > gate ({cfg.get('gate', 'sma200')}) AND RSI({cfg['rsi_len']}) < {cfg['rsi_threshold']}; exit {cfg.get('exit_mode', 'sma5')}; sizing {cfg.get('sizing', 'single')}; disaster stop x{cfg.get('disaster_stop_atr', 0)} ATR({cfg['atr_len']}).",
            "- `next_open` is the primary fill (signal on session t, fill at open t+1); `close` is the same-close UPPER BOUND. `moc_proxy` needs 15:50 ETF minute bars, which do not exist in the repo.",
            "- Baselines: EW buy-and-hold of the tradable ETF universe, SMA200 timing on that EW index (signal lagged one session), dev-vol-matched EW, and 500-draw exposure-matched random timing.",
            "- Limitations: moc_proxy fill unavailable; XLC enters the universe ~2019-07 (250-bar history rule).",
        ]

    def caption(self, cfg, state):
        return (f"variant={cfg['variant']} | RSI({cfg['rsi_len']})<{cfg['rsi_threshold']} gate={cfg.get('gate', 'sma200')} exit={cfg.get('exit_mode', 'sma5')} "
                f"sizing={cfg.get('sizing', 'single')} stop={cfg.get('disaster_stop_atr', 0)}xATR | {cfg['cost_bps_per_side']}bps/side slots={cfg['book_slots']} | "
                f"{cfg['data_dir']} {state['dates'][0].date()}..{state['dates'][-1].date()}")
