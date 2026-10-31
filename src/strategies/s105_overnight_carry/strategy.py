"""S105 overnight SPY carry with volatility targeting."""

import numpy as np
import pandas as pd

from src.data.daily_panel import load_panel
from src.framework.daily_study import prefix
from src.framework import report
from src.framework.registry import register
from src.utils import timing
from src.strategies._overnight import OvernightDailyStudy, build_overnight_trades, load_index_panel

from .signals import compute_signals

NAME = "s105_overnight_carry"
TR_SUFFIX = "_1Day_tr.parquet"


def _align_return_series(source, target_dates):
    close = source["close"][:, 0]
    ret = np.zeros(len(close), dtype=float)
    valid = np.isfinite(close[1:]) & np.isfinite(close[:-1]) & (close[:-1] > 0)
    rows = np.flatnonzero(valid) + 1
    ret[rows] = close[rows] / close[rows - 1] - 1.0
    out = np.zeros(len(target_dates), dtype=float)
    positions = pd.DatetimeIndex(target_dates).get_indexer(pd.DatetimeIndex(source["dates"]))
    good = positions >= 0
    out[positions[good]] = ret[good]
    return out


@register(NAME)
class S105OvernightCarry(OvernightDailyStudy):
    name = NAME
    TITLE = "S105 Volatility-Targeted Overnight Carry"
    timing_null = True

    def load(self, ctx, cfg):
        with timing.stage("raw SPY overnight series"):
            panel, aux = load_index_panel(
                "SPY", cfg["raw_data_dir"], cfg["corporate_actions_file"], cfg["yahoo_data_dir"],
                start_date=cfg["start_date"], rebuild=bool(ctx.flags.get("rebuild_cache")),
            )
        with timing.stage("SPY total-return comparator"):
            tr = load_panel(cfg["tr_data_dir"], symbols=["SPY"], suffix=cfg.get("tr_data_suffix", TR_SUFFIX),
                            rebuild=bool(ctx.flags.get("rebuild_cache")))
            aux["total_return"] = _align_return_series(tr, panel["dates"])
        return panel, aux

    def signals(self, panel, aux, cfg):
        with timing.stage("overnight volatility and band sizing"):
            return compute_signals(panel, aux, cfg)

    def trades_for(self, panel, sig, cfg, fill):
        return build_overnight_trades(panel, sig, cfg, fill)

    def rank_col(self):
        return "ret"

    def baselines(self, cfg, state, strat_r):
        return {
            "s021_unscaled_overnight": state["aux"]["overnight_return"][:, 0],
            "spy_total_return": state["aux"]["total_return"],
        }

    def intro(self, cfg, state):
        return [
            f"- 20-session realized volatility of dividend-correct close-to-open returns; target={cfg['target_vol']:.0%}, "
            f"cap={cfg['max_leverage']:.1f}x, resize band={cfg['resize_band']:.0%}.",
            f"- Baseline sizing uses overnight volatility only; cost={cfg['cost_bps_per_side']} bps/side, "
            f"with S021's 0.5/1/2/5 bps ladder.",
            "- Daily closes proxy the MOC auction for 2016-2023; close-based fills are an upper bound where minute bars are unavailable.",
            "- Null: S021 unscaled SPY overnight returns on identical data. This is a research hypothesis: the cited volatility-scaling evidence is for momentum, not overnight carry.",
            "- Research-scale exposure is separate from the $5,000 paper-scale 1x account; the 2x variant assumes margin.",
        ]

    def caption(self, cfg, state):
        return (
            f"variant={cfg['variant']} | overnight vol={cfg['vol_window']}d target={cfg['target_vol']:.0%} "
            f"cap={cfg['max_leverage']:.1f}x band={cfg['resize_band']:.0%} | "
            f"{cfg['cost_bps_per_side']}bps/side | {state['dates'][0].date()}..{state['dates'][-1].date()}"
        )

    def extra_summary(self, ctx, table, out):
        state = self._state(ctx)
        weights = state["sig"]["trade_weights"][:, 0]
        mask = weights > 0
        resize = np.r_[False, np.abs(np.diff(weights)) > 1e-12]
        rows = []
        for period, period_mask in state["masks"].items():
            active = mask & period_mask
            resize_active = resize & period_mask
            rows.append({
                "period": period,
                "n_overnights": int(active.sum()),
                "mean_exposure": float(weights[active].mean()) if active.any() else np.nan,
                "max_exposure": float(weights[active].max()) if active.any() else np.nan,
                "resize_events": int(resize_active.sum()),
                "resize_turnover_annual": float(np.abs(np.diff(weights, prepend=0))[period_mask].sum() * 252 / max(period_mask.sum(), 1)),
                "overnight_round_trip_turnover_annual": float(2 * weights[period_mask].sum() * 252 / max(period_mask.sum(), 1)),
            })
        pd.DataFrame(rows).to_csv(out / f"{prefix(ctx.cfg)}exposure_and_turnover.csv", index=False)
        report.section(f"{NAME} exposure and resizing")
        report.table(pd.DataFrame(rows), floatfmt=".3f")
