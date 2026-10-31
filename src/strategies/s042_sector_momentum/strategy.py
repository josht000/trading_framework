"""S042 Sector Momentum Rotation: framework adapter (sector ETF TR bars, monthly top 3, research scale)."""

import numpy as np
import pandas as pd

from src.data.daily_panel import load_panel
from src.framework.registry import register
from src.strategies import _bookstudy as BS
from src.strategies._bookstudy import DailyStudy

from .signals import build_trades, compute_signals

NAME = "s042_sector_momentum"
TR_SUFFIX = "_1Day_tr.parquet"
BENCH = ("SPY", "SHY")


@register(NAME)
class S042SectorMomentum(DailyStudy):
    name = NAME
    TITLE = "S042 Sector Momentum Rotation"

    def load(self, ctx, cfg):
        sectors = [x.strip() for x in str(cfg["symbols"]).split(",") if x.strip()]
        wanted = sorted(set(sectors) | set(BENCH))
        panel = load_panel(cfg["data_dir"], symbols=wanted, rebuild=bool(ctx.flags.get("rebuild_cache")),
                           suffix=cfg.get("data_suffix", TR_SUFFIX))
        sector_idx = np.array([panel["symbols"].index(s) for s in sectors if s in panel["symbols"]])
        bench = {}
        for s in BENCH:
            if s in panel["symbols"]:
                j = panel["symbols"].index(s)
                r = np.full(len(panel["dates"]), np.nan)
                c = panel["close"][:, j]
                r[1:] = c[1:] / c[:-1] - 1.0
                bench[s] = np.nan_to_num(r)
        return panel, {"sector_idx": sector_idx, "bench": bench}

    def signals(self, panel, aux, cfg):
        return compute_signals(panel, cfg, aux["sector_idx"])

    def trades_for(self, panel, sig, cfg, fill):
        return build_trades(panel, sig, cfg, fill)

    def rank_col(self):
        return "ret_gross"

    def baselines(self, cfg, state, strat_r):
        spy = state["aux"]["bench"].get("SPY", np.zeros(len(state["dates"])))
        shy = state["aux"]["bench"].get("SHY", np.zeros(len(state["dates"])))
        return {"ew_sectors": state["b_day"], "spy": spy, "sixty_forty": 0.6 * spy + 0.4 * shy,
                "vol_matched_ew": BS.vol_matched(state["b_day"], strat_r, state["masks"]["dev"])}

    def intro(self, cfg, state):
        return [
            f"- Universe: {cfg['symbols']} SPDR sector ETFs (TR bars) + SPY/SHY benchmark columns; PIT by data availability + {cfg['min_history']}-bar history rule (XLC enters ~2019, XLRE ~2016-11).",
            f"- Rule: month-end rank by {cfg.get('signal', 'mom12')} ({'12-month total return' if cfg.get('signal','mom12')=='mom12' else cfg.get('signal')}), hold top {cfg['breadth']} equal-weight, rebalance {cfg.get('rebalance','monthly')}; book slots {cfg['book_slots']}.",
            "- `close` fill is the month-end close named by the rule (upper bound); `next_open` is the robustness fill. Costs 5 bps/side (liquid ETF).",
            "- Baselines: equal-weight sector basket (trend), SPY, 60/40 (SPY+SHY TR), dev-vol-matched EW, random k-of-N picks.",
            "- Limitations: no published PIT sector-membership table - PIT is approximated by listing dates + the history rule; the Sep-2018 XLK/XLY/XLC reconstitution is handled by the same rule.",
        ]

    def caption(self, cfg, state):
        return (f"variant={cfg['variant']} | sector momentum signal={cfg.get('signal','mom12')} top{cfg['breadth']} {cfg.get('rebalance','monthly')} | "
                f"{cfg['cost_bps_per_side']}bps/side slots={cfg['book_slots']} | {state['dates'][0].date()}..{state['dates'][-1].date()}")
