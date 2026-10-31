"""S021 SPY close-to-open premium using raw bars and ex-date cash dividends."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.daily_panel import load_panel
from src.framework.daily_study import prefix
from src.framework import report
from src.framework.registry import register
from src.utils import timing
from src.strategies._overnight import OvernightDailyStudy, build_overnight_trades, load_index_panel

from .signals import compute_signals

NAME = "s021_index_overnight"
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
class S021IndexOvernight(OvernightDailyStudy):
    name = NAME
    TITLE = "S021 Index Overnight Premium"
    timing_null = True

    def load(self, ctx, cfg):
        symbol = cfg["symbol"]
        with timing.stage("raw price and corporate-action series"):
            panel, aux = load_index_panel(
                symbol, cfg["raw_data_dir"], cfg["corporate_actions_file"], cfg["yahoo_data_dir"],
                start_date=cfg["start_date"], rebuild=bool(ctx.flags.get("rebuild_cache")),
            )
        with timing.stage("total-return benchmark"):
            tr = load_panel(cfg["tr_data_dir"], symbols=[symbol], suffix=cfg.get("tr_data_suffix", TR_SUFFIX),
                            rebuild=bool(ctx.flags.get("rebuild_cache")))
            aux["total_return"] = _align_return_series(tr, panel["dates"])
        aux["symbol"] = symbol
        return panel, aux

    def signals(self, panel, aux, cfg):
        return compute_signals(panel, aux, cfg)

    def trades_for(self, panel, sig, cfg, fill):
        return build_overnight_trades(panel, sig, cfg, fill)

    def rank_col(self):
        return "ret"

    def baselines(self, cfg, state, strat_r):
        return {
            "unconditional_overnight": state["aux"]["overnight_return"][:, 0],
            f"{cfg['symbol']}_buy_hold_TR": state["aux"]["total_return"],
        }

    def intro(self, cfg, state):
        return [
            f"- Instrument: {cfg['symbol']}; entry at the prior close and exit at the next open, dividend/split corrected.",
            f"- Participation gate: {cfg['participation_filter']}; cost={cfg['cost_bps_per_side']} bps/side; "
            f"cost sensitivity is the {cfg['cost_ladder_multipliers']}x ladder.",
            "- Daily closes proxy the MOC auction for 2016-2023; treat close-based fills as an upper bound because minute bars do not cover the full sample.",
            "- The SMA(50) gate uses only the previous completed close. The primary rule is unconditional.",
            "- Baselines: same-instrument unconditional overnight drift, same-instrument total-return buy-and-hold, and random timing.",
            "- Note: the catalog's SMA(50) gate is an adaptation; Cooper, Cliff & Gulen support the market-level overnight/intraday decomposition.",
        ]

    def caption(self, cfg, state):
        return (f"variant={cfg['variant']} | {cfg['symbol']} close-to-open filter={cfg['participation_filter']} | "
                f"{cfg['cost_bps_per_side']}bps/side | {state['dates'][0].date()}..{state['dates'][-1].date()}")

    def extra_summary(self, ctx, table, out):
        state = self._state(ctx)
        dates = pd.DatetimeIndex(state["dates"])
        gross = np.asarray(state["aux"]["overnight_return"][:, 0], dtype=float)
        weights = state["sig"]["trade_weights"][:, 0]
        weekday_rows = []
        for day in range(5):
            sel = (dates.dayofweek == day) & (weights > 0) & np.isfinite(gross)
            values = gross[sel]
            weekday_rows.append({
                "weekday": ("Mon", "Tue", "Wed", "Thu", "Fri")[day],
                "n": len(values),
                "mean_gross_pct": 100 * float(values.mean()) if len(values) else np.nan,
                "win_pct": 100 * float((values > 0).mean()) if len(values) else np.nan,
            })
        pre = prefix(ctx.cfg)
        pd.DataFrame(weekday_rows).to_csv(out / f"{pre}weekday_returns.csv", index=False)

        periods = [
            ("2016-19", "2016-01-01", "2020-01-01"),
            ("2020", "2020-01-01", "2021-01-01"),
            ("2021", "2021-01-01", "2022-01-01"),
            ("2022", "2022-01-01", "2023-01-01"),
            ("2023-26", "2023-01-01", "2027-01-01"),
        ]
        rows = []
        for label, start, end in periods:
            sel = (dates >= start) & (dates < end) & (weights > 0) & np.isfinite(gross)
            values = gross[sel]
            rows.append({
                "period": label, "n": len(values),
                "gross_total_log": float(np.log1p(values).sum()) if len(values) else 0.0,
                "gross_mean_bps": float(values.mean() * 1e4) if len(values) else np.nan,
                "break_even_bps_per_side": float(values.mean() * 1e4 / 2) if len(values) else np.nan,
            })
        pd.DataFrame(rows).to_csv(out / f"{pre}subperiods_and_break_even.csv", index=False)
        report.section(f"{NAME} weekday and break-even costs")
        report.table(pd.DataFrame(weekday_rows), floatfmt=".3f")
