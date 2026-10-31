"""S041 12-1 cross-sectional momentum: framework adapter (stock TR bars, monthly top decile, research scale)."""

import numpy as np
import pandas as pd

from src.data.daily_panel import load_panel
from src.framework import DailyStudy, daily_study as BS
from src.framework.registry import register
from src.strategies._engines import align_ranks
from src.data.liquid_universe import load_liquid_ranks

from .signals import build_trades, compute_signals

NAME = "s041_mom_12_1"
TR_SUFFIX = "_1Day_tr.parquet"


@register(NAME)
class S041Momentum(DailyStudy):
    name = NAME
    TITLE = "S041 12-1 Cross-Sectional Momentum"

    def load(self, ctx, cfg):
        panel = load_panel(cfg["data_dir"], rebuild=bool(ctx.flags.get("rebuild_cache")),
                           suffix=cfg.get("data_suffix", TR_SUFFIX))
        rank_dates, rank_symbols, rank = load_liquid_ranks(cfg.get("ranks_dir", "data/cache/liquid_ranks"))
        aux = {"rank": align_ranks(panel["dates"], panel["symbols"], rank_dates, rank_symbols, rank)}
        if cfg.get("industry_neutral"):
            from src.data.daily_panel import load_sector_map
            ind = load_sector_map()["Industry"].reindex(panel["symbols"]).fillna("")
            panel["industry"] = ind.to_numpy()
        return panel, aux

    def signals(self, panel, aux, cfg):
        return compute_signals(panel, cfg, aux["rank"])

    def trades_for(self, panel, sig, cfg, fill):
        return build_trades(panel, sig, cfg, fill)

    def rank_col(self):
        return "ret_gross"

    def baselines(self, cfg, state, strat_r):
        return {"ew_universe": state["b_day"],
                "vol_matched_ew": BS.vol_matched(state["b_day"], strat_r, state["masks"]["dev"])}

    def intro(self, cfg, state):
        return [
            f"- Universe: liquid_top_{cfg['universe_top_n']} PIT (60d median dollar-volume rank among common stocks, price>=5, >=250 bars), "
            f"{len(state['panel']['symbols'])} names in the TR cache; monthly top {'decile' if cfg.get('breadth_mode','decile')=='decile' else cfg['breadth']} by {cfg['mom_window']}-{cfg['skip_window']} total return, equal weight, book slots {cfg['book_slots']}.",
            "- `close` fill is the month-end close named by the rule (still an upper bound); `next_open` is the robustness fill. Costs 15 bps/side (stock default).",
            "- Delisting convention: a held name whose series ends mid-sample exits at last close x (1 + delisting_return) - the catalog's -30% performance-delisting assumption.",
            "- Baselines: equal-weight liquid universe (trend), dev-vol-matched EW (Borsa-style benchmark), random k-of-N picks (built in).",
            "- Limitations: SimFin industry map is a current snapshot; stock TR comes from Alpaca adjustment=all (dividend-adjusted, delisted names kept).",
        ]

    def caption(self, cfg, state):
        return (f"variant={cfg['variant']} | 12-1 mom top-{cfg.get('breadth_mode','decile')}({cfg.get('decile') if cfg.get('breadth_mode','decile')=='decile' else cfg['breadth']}) "
                f"ind_neutral={cfg.get('industry_neutral', False)} | {cfg['cost_bps_per_side']}bps/side slots={cfg['book_slots']} | "
                f"{state['dates'][0].date()}..{state['dates'][-1].date()}")
