"""S052 monthly earnings-announcement baskets with point-in-time event features."""

import math

import numpy as np
import pandas as pd


def month_end_rows(dates):
    periods = pd.DatetimeIndex(dates).to_period("M")
    return np.flatnonzero(np.r_[periods[1:] != periods[:-1], True])


def week_end_rows(dates):
    periods = pd.DatetimeIndex(dates).to_period("W-FRI")
    return np.flatnonzero(np.r_[periods[1:] != periods[:-1], True])


def _month_volume(price_panel, stock_symbols):
    columns = [price_panel["symbols"].index(s) for s in stock_symbols]
    frame = pd.DataFrame(price_panel["volume"][:, columns], index=price_panel["dates"], columns=stock_symbols)
    monthly = frame.groupby(pd.DatetimeIndex(frame.index).to_period("M")).sum(min_count=1)
    return monthly.index, np.nan_to_num(monthly.to_numpy(float).T)


def _volume_cumsums(month_periods, monthly_volume, event_months, symbols):
    period_to_row = {period: i for i, period in enumerate(month_periods)}
    event_volume = np.zeros_like(monthly_volume)
    for j, symbol in enumerate(symbols):
        rows = [period_to_row[p] for p in event_months.get(symbol, ()) if p in period_to_row]
        if rows:
            event_volume[j, rows] = monthly_volume[j, rows]
    return period_to_row, np.cumsum(monthly_volume, axis=1), np.cumsum(event_volume, axis=1)


def _volume_concentration(period_to_row, volume_cum, event_cum, target_period, lookback, lag):
    end_period = target_period - int(lag)
    end = period_to_row.get(end_period)
    start = period_to_row.get(end_period - int(lookback) + 1)
    if end is None or start is None:
        return np.full(volume_cum.shape[0], np.nan)
    total = volume_cum[:, end] - (volume_cum[:, start - 1] if start else 0.0)
    event_volume = event_cum[:, end] - (event_cum[:, start - 1] if start else 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, event_volume / total, np.nan)


def _top_quintile(eligible, score, fraction):
    candidates = np.flatnonzero(eligible & np.isfinite(score))
    selected = np.zeros(len(eligible), dtype=bool)
    if not len(candidates):
        return selected
    take = max(1, int(math.ceil(len(candidates) * float(fraction))))
    order = np.argsort(score[candidates], kind="stable")
    selected[candidates[order[-take:]]] = True
    return selected


def _weekly_expected(dates, target_week):
    prior_start = pd.Timestamp(target_week) - pd.DateOffset(years=1)
    if not len(dates):
        return False
    return bool(np.any(np.abs((dates - prior_start).days) <= 7))


def compute_signals(panel, price_panel, aux, cfg):
    dates = pd.DatetimeIndex(panel["dates"])
    symbols = list(panel["symbols"])
    stock_symbols = [s for s in symbols if s != "SPY"]
    stock_cols = np.array([symbols.index(s) for s in stock_symbols], dtype=int)
    price_cols = [price_panel["symbols"].index(s) for s in stock_symbols]
    price_rows = np.searchsorted(price_panel["dates"].values, dates.values)
    price_valid = price_rows < len(price_panel["dates"])
    price_valid &= price_panel["dates"].values[np.minimum(price_rows, len(price_panel["dates"]) - 1)] == dates.values
    raw_close = np.full((len(dates), len(stock_symbols)), np.nan)
    raw_close[price_valid] = price_panel["close"][price_rows[price_valid]][:, price_cols]
    history = np.isfinite(raw_close).cumsum(0)
    tradable_stocks = np.isfinite(raw_close) & (raw_close >= float(cfg["min_price"])) & (history >= int(cfg["min_history"]))

    cadence = cfg.get("cadence", "monthly")
    all_rows = month_end_rows(dates) if cadence == "monthly" else week_end_rows(dates)
    month_periods, monthly_volume = _month_volume(price_panel, stock_symbols)
    event_dates = aux["event_dates"]
    event_index = {s: pd.DatetimeIndex(event_dates.get(s, ())) for s in stock_symbols}
    event_months = {s: set(event_index[s].to_period("M")) for s in stock_symbols}
    period_to_row, volume_cum, event_cum = _volume_cumsums(month_periods, monthly_volume, event_months, stock_symbols)
    shares = aux["share_reports"]
    selected = {}
    eligible_pool = np.zeros((len(dates), len(symbols)), dtype=bool)
    score = np.full((len(dates), len(symbols)), np.nan)
    trend_exposure = np.zeros(len(dates), dtype=float)
    universe_return = np.zeros(len(dates), dtype=float)
    market_return = np.zeros(len(dates), dtype=float)
    daily_ret = np.full(panel["close"].shape, np.nan)
    daily_ret[1:] = panel["close"][1:] / panel["close"][:-1] - 1.0
    live = tradable_stocks & np.isfinite(daily_ret[:, stock_cols])
    universe_return = np.where(live, np.nan_to_num(daily_ret[:, stock_cols]), 0.0).sum(1) / np.maximum(live.sum(1), 1)
    spy_col = symbols.index("SPY")
    market_return = np.nan_to_num(daily_ret[:, spy_col])

    active_rows = []
    period_end = {}
    signal_ready = np.zeros(len(dates), dtype=bool)
    for row in all_rows:
        signal_date = dates[row]
        if cadence == "monthly":
            target_period = signal_date.to_period("M") + 1
            target_week = None
            end_row = int(np.searchsorted(dates.values, target_period.end_time.to_datetime64(), side="right") - 1)
        else:
            next_row = min(row + 1, len(dates) - 1)
            target_week = (dates[next_row] - pd.Timedelta(days=dates[next_row].weekday())).normalize()
            week_of_month = (target_week.day - 1) // 7 + 1
            if target_week.month not in (1, 4, 7, 10) or week_of_month not in (3, 4):
                continue
            target_period = target_week.to_period("M")
            end_row = min(row + 5, len(dates) - 1)
        if end_row <= row or end_row >= len(dates):
            continue
        volume_ratio = _volume_concentration(
            period_to_row, volume_cum, event_cum, target_period,
            cfg["volume_lookback_months"], cfg["volume_lag_months"],
        )
        if not np.isfinite(volume_ratio).any():
            continue
        exact_four = np.zeros(len(stock_symbols), dtype=bool)
        expected = np.zeros(len(stock_symbols), dtype=bool)
        event_cutoff = signal_date - pd.DateOffset(months=12)
        for j, symbol in enumerate(stock_symbols):
            reports = event_index[symbol]
            recent_count = np.searchsorted(reports, signal_date, side="right") - np.searchsorted(reports, event_cutoff, side="right")
            exact_four[j] = recent_count == 4
            if cadence == "monthly":
                expected[j] = target_period - 12 in event_months.get(symbol, set())
            else:
                expected[j] = _weekly_expected(reports, target_week)
        eligible = tradable_stocks[row] & exact_four & np.isfinite(volume_ratio)
        eligible_pool[row, stock_cols] = eligible
        score[row, stock_cols] = volume_ratio
        top = _top_quintile(eligible, volume_ratio, cfg["volume_quintile"]) if cfg.get("sort_mode", "volume") == "volume" else eligible
        long_sel = eligible & expected & top
        short_sel = eligible & ~expected & top if cfg.get("leg", "long_spy_hedge") == "long_short" else np.zeros_like(eligible)

        def weights_for(mask):
            idx = np.flatnonzero(mask)
            if not len(idx):
                return idx, np.array([], dtype=float)
            if cfg.get("weighting", "equal") == "value":
                caps = np.zeros(len(idx), dtype=float)
                for k, j in enumerate(idx):
                    symbol = stock_symbols[j]
                    report_dates, share_counts = shares.get(symbol, (np.array([], dtype="datetime64[ns]"), np.array([], dtype=float)))
                    asof = np.searchsorted(report_dates, signal_date.to_datetime64(), side="right") - 1
                    if asof >= 0:
                        caps[k] = raw_close[row, j] * share_counts[asof]
                good = np.isfinite(caps) & (caps > 0)
                idx, caps = idx[good], caps[good]
                if not len(caps):
                    return np.array([], dtype=int), np.array([], dtype=float)
                return idx, caps / caps.sum()
            return idx, np.full(len(idx), 1.0 / len(idx))

        long_idx, long_weights = weights_for(long_sel)
        short_idx, short_weights = weights_for(short_sel)
        if len(long_idx) and cadence == "monthly":
            spy_weight = -1.0 if cfg.get("leg", "long_spy_hedge") == "long_spy_hedge" else 0.0
        else:
            spy_weight = 0.0
        selected[int(row)] = {
            "end_row": int(end_row),
            "long_idx": long_idx,
            "long_weights": long_weights,
            "short_idx": short_idx,
            "short_weights": short_weights,
            "spy_weight": spy_weight,
        }
        active_rows.append(int(row))
        if len(long_idx):
            signal_ready[row:end_row + 1] = True
            trend_exposure[row:end_row + 1] = 2.0

    tradable = np.zeros((len(dates), len(symbols)), dtype=bool)
    tradable[:, stock_cols] = tradable_stocks & signal_ready[:, None]
    pool = eligible_pool
    if cfg.get("leg", "long_spy_hedge") == "long_spy_hedge":
        benchmark = 0.5 * (universe_return - market_return)
    else:
        benchmark = np.zeros(len(dates), dtype=float)
    return {
        "tradable": tradable,
        "pool": pool,
        "score": score,
        "selected": selected,
        "active_rows": np.asarray(active_rows, dtype=int),
        "stock_symbols": stock_symbols,
        "trend_exposure": trend_exposure,
        "benchmark_returns": benchmark,
        "benchmarks": {"SPY total return": market_return, "equal-weight stocks": universe_return},
        "universe_returns": universe_return,
    }


def build_trades(panel, sig, cfg, fill):
    dates = pd.DatetimeIndex(panel["dates"])
    symbols = list(panel["symbols"])
    sym_idx = {s: i for i, s in enumerate(symbols)}
    entry_mat = panel["open"] if fill == "next_open" else panel["close"]
    exit_mat = panel["open"] if fill == "next_open" else panel["close"]
    cost = float(cfg["cost_bps_per_side"]) / 1e4
    last_valid = np.where(np.isfinite(panel["close"]), np.arange(len(dates))[:, None], -1).max(0)
    active_rows = sig["active_rows"]
    weights_by_row = []
    for signal_row in active_rows:
        selection = sig["selected"][int(signal_row)]
        weights = {}
        for ids, values, direction in (
            (selection["long_idx"], selection["long_weights"], 1.0),
            (selection["short_idx"], selection["short_weights"], -1.0),
        ):
            for j, value in zip(ids, values):
                weights[sig["stock_symbols"][int(j)]] = direction * float(value)
        if selection["spy_weight"]:
            weights["SPY"] = float(selection["spy_weight"])
        weights_by_row.append(weights)

    rows = []
    for i, signal_row in enumerate(active_rows):
        selection = sig["selected"][int(signal_row)]
        end_row = selection["end_row"]
        entry_row = int(signal_row + 1) if fill == "next_open" else int(signal_row)
        desired_exit = int(end_row + 1) if fill == "next_open" else int(end_row)
        if entry_row >= len(dates):
            continue
        next_weights = weights_by_row[i + 1] if i + 1 < len(weights_by_row) else {}
        previous_weights = weights_by_row[i - 1] if i else {}
        window_end = desired_exit >= len(dates)
        for symbol, signed_weight in weights_by_row[i].items():
            k = sym_idx[symbol]
            entry_px = entry_mat[entry_row, k]
            position_exit = min(desired_exit, len(dates) - 1)
            exit_reason = "window_end" if window_end else "rebalance"
            exit_px = exit_mat[position_exit, k]
            if window_end:
                exit_px = panel["close"][position_exit, k]
            if desired_exit <= last_valid[k] and not window_end:
                pass
            elif last_valid[k] < len(dates) - 1 and last_valid[k] >= entry_row:
                position_exit = int(last_valid[k])
                exit_px = panel["close"][position_exit, k] * (1 + float(cfg["delisting_return"]))
                exit_reason = "delisted"
            elif not np.isfinite(exit_px):
                exit_px = panel["close"][position_exit, k]
            if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px <= 0:
                continue

            entry_fee_weight = abs(signed_weight - previous_weights.get(symbol, 0.0))
            if exit_reason == "delisted":
                exit_fee_weight = abs(signed_weight)
            elif exit_reason == "window_end":
                exit_fee_weight = 0.0
            elif next_weights.get(symbol, 0.0) == 0.0:
                exit_fee_weight = abs(signed_weight)
            else:
                exit_fee_weight = 0.0
            gross = signed_weight * (exit_px / entry_px - 1.0)
            ret = gross - cost * (entry_fee_weight + exit_fee_weight)
            rows.append({
                "fill": fill, "symbol": symbol, "signal_idx": int(signal_row), "entry_idx": entry_row,
                "exit_idx": position_exit, "signal_date": dates[signal_row], "entry_date": dates[entry_row],
                "exit_date": dates[position_exit], "entry_px": float(entry_px), "exit_px": float(exit_px),
                "trade_weight": signed_weight, "entry_cost_weight": entry_fee_weight,
                "exit_cost_weight": exit_fee_weight, "ret": ret, "ret_gross": gross,
                "hold": int(position_exit - entry_row + 1), "exit_reason": exit_reason,
                "sym_idx": k, "random_fixed": symbol == "SPY",
            })
    cols = ["fill", "symbol", "signal_idx", "entry_idx", "exit_idx", "signal_date", "entry_date", "exit_date",
            "entry_px", "exit_px", "trade_weight", "entry_cost_weight", "exit_cost_weight",
            "ret", "ret_gross", "hold", "exit_reason", "sym_idx", "random_fixed"]
    return pd.DataFrame(rows, columns=cols)
