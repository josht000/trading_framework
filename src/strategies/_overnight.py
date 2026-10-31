"""Shared raw-price and corporate-action helpers for overnight studies."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.daily_panel import load_panel
from src.framework import DailyStudy

RAW_SUFFIX = "_1Day_raw.parquet"


def action_matrices(dates, symbols, actions_path):
    dates = pd.DatetimeIndex(dates)
    symbol_idx = {symbol: i for i, symbol in enumerate(symbols)}
    date_idx = {date: i for i, date in enumerate(dates)}
    dividends = np.zeros((len(dates), len(symbols)), dtype=float)
    split_factor = np.ones((len(dates), len(symbols)), dtype=float)
    path = Path(actions_path)
    if not path.is_file():
        raise FileNotFoundError(f"corporate-action file missing: {path}")
    actions = pd.read_parquet(path)
    for row in actions.itertuples(index=False):
        symbol = getattr(row, "symbol", None)
        if symbol not in symbol_idx:
            continue
        date = pd.Timestamp(getattr(row, "ex_date")).normalize()
        i, j = date_idx.get(date), symbol_idx[symbol]
        if i is None:
            continue
        action_type = getattr(row, "action_type", "cash_dividend")
        if action_type == "cash_dividend":
            rate = getattr(row, "rate", 0.0)
            if pd.notna(rate):
                dividends[i, j] += float(rate)
        elif action_type in ("forward_split", "reverse_split", "unit_split"):
            old_value, new_value = getattr(row, "old_rate", 0.0), getattr(row, "new_rate", 0.0)
            old_rate = float(old_value) if pd.notna(old_value) else 0.0
            new_rate = float(new_value) if pd.notna(new_value) else 0.0
            if old_rate > 0 and new_rate > 0:
                split_factor[i, j] *= new_rate / old_rate
    return dividends, split_factor


def add_overnight_fields(panel, dividends, split_factor):
    effective_open = (panel["open"] + dividends) * split_factor
    overnight_entry = np.full(panel["close"].shape, np.nan)
    overnight_entry[1:] = panel["close"][:-1]
    overnight_return = np.full(panel["close"].shape, np.nan)
    overnight_return[1:] = effective_open[1:] / panel["close"][:-1] - 1.0
    return {
        "dividend": dividends,
        "split_factor": split_factor,
        "effective_open": effective_open,
        "overnight_entry": overnight_entry,
        "overnight_exit": effective_open,
        "overnight_return": overnight_return,
    }


def load_raw_panel(data_dir, symbols, actions_path, *, rebuild=False):
    panel = load_panel(data_dir, symbols=symbols, suffix=RAW_SUFFIX, rebuild=rebuild)
    dividends, splits = action_matrices(panel["dates"], panel["symbols"], actions_path)
    return panel, add_overnight_fields(panel, dividends, splits)


def _local_dates(index):
    dates = pd.DatetimeIndex(index)
    if dates.tz is not None:
        dates = dates.tz_convert("America/New_York").tz_localize(None)
    return dates.normalize()


def load_index_panel(symbol, raw_dir, actions_path, yahoo_dir, *, start_date="2016-01-04", rebuild=False):
    """Use Yahoo raw OHLC/actions for SPY through 2025, Alpaca raw bars afterwards."""
    if symbol != "SPY":
        return load_raw_panel(raw_dir, [symbol], actions_path, rebuild=rebuild)

    yahoo_path = Path(yahoo_dir) / "SPY_1Day_yahoo.parquet"
    if not yahoo_path.is_file():
        return load_raw_panel(raw_dir, [symbol], actions_path, rebuild=rebuild)
    yahoo = pd.read_parquet(yahoo_path)
    yahoo.index = _local_dates(yahoo.index)
    yahoo = yahoo.loc[yahoo.index >= pd.Timestamp(start_date)].copy()
    yahoo["split_factor"] = 1.0
    has_split = (yahoo["split_numerator"] > 0) & (yahoo["split_denominator"] > 0)
    yahoo.loc[has_split, "split_factor"] = (
        yahoo.loc[has_split, "split_numerator"] / yahoo.loc[has_split, "split_denominator"]
    )
    yahoo = yahoo[["open", "close", "volume", "dividend", "split_factor"]]

    alpaca, aux = load_raw_panel(raw_dir, [symbol], actions_path, rebuild=rebuild)
    alpaca_frame = pd.DataFrame({
        "open": alpaca["open"][:, 0],
        "close": alpaca["close"][:, 0],
        "volume": alpaca["volume"][:, 0],
        "dividend": aux["dividend"][:, 0],
        "split_factor": aux["split_factor"][:, 0],
    }, index=alpaca["dates"])
    alpaca_frame = alpaca_frame.loc[alpaca_frame.index > pd.Timestamp("2025-12-31")]
    combined = pd.concat([yahoo, alpaca_frame]).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    dividends, splits = action_matrices(combined.index, [symbol], actions_path)
    combined["dividend"] = dividends[:, 0]
    combined["split_factor"] = splits[:, 0]
    panel = {
        "dates": pd.DatetimeIndex(combined.index),
        "symbols": [symbol],
        "open": combined[["open"]].to_numpy(float),
        "close": combined[["close"]].to_numpy(float),
        "volume": combined[["volume"]].to_numpy(float),
    }
    dividend = combined[["dividend"]].to_numpy(float)
    split_factor = combined[["split_factor"]].to_numpy(float)
    return panel, add_overnight_fields(panel, dividend, split_factor)


def join_panels(left, left_aux, right, right_aux):
    dates = pd.DatetimeIndex(np.union1d(left["dates"].values, right["dates"].values))
    symbols = list(left["symbols"]) + list(right["symbols"])
    arrays = {field: np.full((len(dates), len(symbols)), np.nan) for field in ("open", "close", "volume")}
    auxiliary = {field: np.full((len(dates), len(symbols)), np.nan) for field in
                 ("effective_open", "overnight_entry", "overnight_exit", "overnight_return")}
    auxiliary["dividend"] = np.zeros((len(dates), len(symbols)), dtype=float)
    auxiliary["split_factor"] = np.ones((len(dates), len(symbols)), dtype=float)
    auxiliary["split_factor"].fill(1.0)
    left_rows = np.searchsorted(dates.values, left["dates"].values)
    right_rows = np.searchsorted(dates.values, right["dates"].values)
    n_left = len(left["symbols"])
    for field in arrays:
        arrays[field][left_rows, :n_left] = left[field]
        arrays[field][right_rows, n_left:] = right[field]
    for field in auxiliary:
        auxiliary[field][left_rows, :n_left] = left_aux[field]
        auxiliary[field][right_rows, n_left:] = right_aux[field]
    return {"dates": dates, "symbols": symbols, **arrays}, auxiliary


class OvernightDailyStudy(DailyStudy):
    """DailyStudy adapter that accounts for close-to-next-open return intervals."""

    def book(self, state, cfg, fill):
        book = super().book(state, cfg, fill)
        book["entry_mat"] = state["aux"]["overnight_entry"]
        book["exit_mat"] = state["aux"]["overnight_exit"]
        return book


def build_overnight_trades(panel, sig, cfg, fill):
    dates = pd.DatetimeIndex(panel["dates"])
    symbols = list(panel["symbols"])
    weights = np.asarray(sig["trade_weights"], dtype=float)
    if weights.shape != panel["close"].shape:
        raise ValueError("overnight trade_weights must match the daily symbol panel")
    entries = np.asarray(sig["overnight_entry"], dtype=float)
    exits = np.asarray(sig["overnight_exit"], dtype=float)
    if entries.shape != weights.shape or exits.shape != weights.shape:
        raise ValueError("overnight entry/exit matrices must match trade_weights")
    cost = float(cfg["cost_bps_per_side"]) / 1e4
    fixed = np.asarray(sig.get("random_fixed", np.zeros(weights.shape, dtype=bool)), dtype=bool)
    rows = []
    for day in range(1, len(dates)):
        for column in np.flatnonzero(weights[day]):
            entry_px, exit_px = entries[day, column], exits[day, column]
            if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px <= 0:
                continue
            weight = float(weights[day, column])
            gross = weight * (exit_px / entry_px - 1.0)
            rows.append({
                "fill": fill,
                "symbol": symbols[column],
                "signal_idx": day - 1,
                "entry_idx": day,
                "exit_idx": day,
                "signal_date": dates[day - 1],
                "entry_date": dates[day],
                "exit_date": dates[day],
                "entry_px": float(entry_px),
                "exit_px": float(exit_px),
                "trade_weight": weight,
                "entry_cost_weight": abs(weight),
                "exit_cost_weight": abs(weight),
                "ret": gross - 2 * cost * abs(weight),
                "ret_gross": gross,
                "hold": 1,
                "exit_reason": "overnight",
                "sym_idx": column,
                "random_fixed": bool(fixed[day, column]),
            })
    return pd.DataFrame(rows, columns=[
        "fill", "symbol", "signal_idx", "entry_idx", "exit_idx", "signal_date", "entry_date", "exit_date",
        "entry_px", "exit_px", "trade_weight", "entry_cost_weight", "exit_cost_weight", "ret", "ret_gross",
        "hold", "exit_reason", "sym_idx", "random_fixed",
    ])
