"""S052 earnings-announcement premium: monthly, point-in-time stock baskets."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.daily_panel import load_panel
from src.framework import DailyStudy
from src.framework.registry import register
from src.utils import timing

from .signals import build_trades, compute_signals

NAME = "s052_earnings_premium"
TR_SUFFIX = "_1Day_tr.parquet"
SPLIT_SUFFIX = "_1Day_split.parquet"


def _simfin_inputs(path):
    frame = pd.read_csv(
        path,
        sep=";",
        usecols=["Ticker", "Publish Date", "Shares (Basic)"],
        parse_dates=["Publish Date"],
        dtype={"Ticker": "string"},
    )
    frame = frame.dropna(subset=["Ticker", "Publish Date"]).copy()
    frame["Ticker"] = frame["Ticker"].str.upper()
    frame = frame.drop_duplicates(["Ticker", "Publish Date"], keep="last")
    events, shares = {}, {}
    for symbol, rows in frame.sort_values("Publish Date").groupby("Ticker", sort=False):
        events[str(symbol)] = pd.DatetimeIndex(rows["Publish Date"].drop_duplicates()).sort_values().to_numpy()
        valid_shares = rows.dropna(subset=["Shares (Basic)"])
        shares[str(symbol)] = (
            valid_shares["Publish Date"].to_numpy(dtype="datetime64[ns]"),
            valid_shares["Shares (Basic)"].to_numpy(float),
        )
    return events, shares


def _join_panels(left, right, extra_symbol):
    dates = pd.DatetimeIndex(np.union1d(left["dates"].values, right["dates"].values))
    symbols = list(left["symbols"]) + [extra_symbol]
    arrays = {}
    left_rows = np.searchsorted(dates.values, left["dates"].values)
    right_rows = np.searchsorted(dates.values, right["dates"].values)
    for field in ("open", "close", "volume"):
        values = np.full((len(dates), len(symbols)), np.nan)
        values[left_rows, :len(left["symbols"])] = left[field]
        values[right_rows, -1] = right[field][:, 0]
        arrays[field] = values
    return {"dates": dates, "symbols": symbols, **arrays}


def _daily_return(close):
    result = np.zeros(len(close), dtype=float)
    valid = np.isfinite(close[1:]) & np.isfinite(close[:-1]) & (close[:-1] > 0)
    rows = np.flatnonzero(valid) + 1
    result[rows] = close[rows] / close[rows - 1] - 1.0
    return result


@register(NAME)
class S052EarningsPremium(DailyStudy):
    name = NAME
    TITLE = "S052 Earnings Announcement Premium"

    def load(self, ctx, cfg):
        with timing.stage("SimFin publish dates"):
            events, shares = _simfin_inputs(cfg["earnings_file"])
        root = Path(cfg["data_dir"])
        available = {p.name[:-len(cfg["data_suffix"])] for p in root.glob(f"*{cfg['data_suffix']}")}
        symbols = sorted(set(events) & available)
        if not symbols:
            raise FileNotFoundError("no SimFin tickers match the total-return stock cache")
        with timing.stage("stock total-return panel"):
            stocks = load_panel(
                root, symbols=symbols, suffix=cfg.get("data_suffix", TR_SUFFIX),
                rebuild=bool(ctx.flags.get("rebuild_cache")),
            )
        with timing.stage("split-adjusted universe panel"):
            prices = load_panel(
                cfg["price_data_dir"], symbols=symbols, suffix=cfg.get("price_data_suffix", SPLIT_SUFFIX),
                rebuild=bool(ctx.flags.get("rebuild_cache")),
            )
        with timing.stage("SPY benchmark panel"):
            spy = load_panel(
                cfg["spy_data_dir"], symbols=["SPY"], suffix=cfg.get("spy_data_suffix", TR_SUFFIX),
                rebuild=bool(ctx.flags.get("rebuild_cache")),
            )
        panel = _join_panels(stocks, spy, "SPY")
        spy_return = _daily_return(panel["close"][:, -1])
        return panel, {
            "events": events,
            "share_reports": shares,
            "price_panel": prices,
            "spy_return": spy_return,
            "n_simfin": len(events),
            "n_stock_bars": len(stocks["symbols"]),
        }

    def signals(self, panel, aux, cfg):
        with timing.stage("earnings schedule and volume sort"):
            return compute_signals(panel, aux["price_panel"], {
                "event_dates": aux["events"],
                "share_reports": aux["share_reports"],
            }, cfg)

    def trades_for(self, panel, sig, cfg, fill):
        return build_trades(panel, sig, cfg, fill)

    def rank_col(self):
        return "ret"

    def baselines(self, cfg, state, strat_r):
        return {
            "spy_total_return": state["aux"]["spy_return"],
            "equal_weight_stock_universe": state["sig"]["universe_returns"],
        }

    def intro(self, cfg, state):
        return [
            f"- Universe: SimFin Publish Date stocks with price >= ${cfg['min_price']:.0f}, "
            f">= {cfg['min_history']} prior bars, and exactly four announcements in the prior 12 months; "
            f"{state['aux']['n_stock_bars']} tickers have total-return bars.",
            f"- Expected announcement month is the same calendar month as the prior-year Publish Date. "
            f"Volume concentration = announcement-month share volume / total share volume over "
            f"{cfg['volume_lookback_months']} months, lagged {cfg['volume_lag_months']} months; "
            f"the baseline uses the top {cfg['volume_quintile']:.0%}.",
            f"- Monthly entry/exit; weighting={cfg['weighting']}, leg={cfg['leg']}; "
            f"gross exposure is 2x for the SPY hedge; {cfg['cost_bps_per_side']} bps/side, net of costs.",
            "- Baselines: random picks from the same event-eligible universe, equal-weight stock universe, and SPY total return.",
            "- Source note: R1's paper check reports value-weighted portfolios; the catalog's first-listed paper-scale baseline is equal weight. "
            "The `value_weight` variant uses latest published SimFin basic shares as a point-in-time market-cap proxy.",
        ]

    def caption(self, cfg, state):
        return (
            f"variant={cfg['variant']} | cadence={cfg['cadence']} sort={cfg['sort_mode']} "
            f"weight={cfg['weighting']} leg={cfg['leg']} | "
            f"{cfg['cost_bps_per_side']}bps/side | {state['dates'][0].date()}..{state['dates'][-1].date()}"
        )
