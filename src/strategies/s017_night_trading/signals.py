"""S017 annual/periodic overnight-bias selection with BH false-discovery control."""

import math

import numpy as np
import pandas as pd


def _formation_rows(dates, frequency):
    periods = pd.DatetimeIndex(dates).to_period("M")
    rows = np.flatnonzero(np.r_[periods[1:] != periods[:-1], True])
    selected = []
    for row in rows:
        month = periods[row].month
        if frequency == "annual" and month != 12:
            continue
        if frequency == "quarterly" and month not in (3, 6, 9, 12):
            continue
        if periods[row] == periods[-1] and dates[row].day < periods[row].days_in_month - 3:
            continue
        selected.append(int(row))
    return np.asarray(selected, dtype=int)


def _bh_discoveries(p_values, universe, q):
    valid = np.flatnonzero(universe & np.isfinite(p_values))
    discoveries = np.zeros(len(universe), dtype=bool)
    if not len(valid):
        return discoveries, np.nan
    order = valid[np.argsort(p_values[valid], kind="stable")]
    thresholds = float(q) * np.arange(1, len(order) + 1) / len(order)
    passing = np.flatnonzero(p_values[order] <= thresholds)
    if not len(passing):
        return discoveries, np.nan
    cutoff = p_values[order[passing[-1]]]
    discoveries[valid] = p_values[valid] <= cutoff
    return discoveries, float(cutoff)


def _top_quintile(means, universe, fraction):
    valid = np.flatnonzero(universe & np.isfinite(means))
    selected = np.zeros(len(universe), dtype=bool)
    if len(valid):
        take = max(1, int(np.ceil(len(valid) * float(fraction))))
        order = np.argsort(means[valid], kind="stable")
        selected[valid[order[-take:]]] = True
    return selected


def compute_signals(panel, aux, cfg):
    dates = pd.DatetimeIndex(panel["dates"])
    symbols = list(panel["symbols"])
    stock_cols = np.array([i for i, symbol in enumerate(symbols) if symbol != "SPY"], dtype=int)
    overnight = np.asarray(aux["overnight_return"], dtype=float)
    intraday = np.full(panel["close"].shape, np.nan)
    valid_intraday = np.isfinite(panel["open"]) & (panel["open"] > 0) & np.isfinite(panel["close"])
    intraday[valid_intraday] = panel["close"][valid_intraday] / panel["open"][valid_intraday] - 1.0
    rank = np.asarray(aux["rank"], dtype=float)
    universe_top_n = int(cfg["universe_top_n"])
    cadence = cfg.get("formation_cadence", "annual")
    formation_rows = _formation_rows(dates, cadence)
    selected_at_formation = []
    summary = []
    cohort_members = []
    min_t = float(cfg.get("min_t_stat", 2.0))
    q = float(cfg.get("fdr_q", 0.05))
    window = int(cfg.get("formation_window", 252))
    top_fraction = float(cfg.get("top_fraction", 0.20))
    for row in formation_rows:
        start = row - window + 1
        if start < 1:
            continue
        sample = overnight[start:row + 1, stock_cols]
        universe = rank[row, stock_cols] <= universe_top_n
        finite = np.isfinite(sample)
        count = finite.sum(0)
        means = np.divide(
            np.nansum(sample, axis=0), count,
            out=np.full(len(stock_cols), np.nan), where=count > 0,
        )
        centered = np.where(finite, sample - means[None, :], 0.0)
        variance = np.divide(
            np.square(centered).sum(0), count - 1,
            out=np.full(len(stock_cols), np.nan), where=count > 1,
        )
        std = np.sqrt(np.maximum(variance, 0.0))
        t_stat = np.divide(
            means, std / np.sqrt(np.maximum(count, 1)),
            out=np.full(len(stock_cols), np.nan), where=(std > 0) & (count >= window),
        )
        p_value = np.array([0.5 * math.erfc(float(t) / np.sqrt(2.0)) if np.isfinite(t) else np.nan for t in t_stat])
        eligible = universe & (count >= window) & np.isfinite(t_stat)
        fdr, fdr_cutoff = _bh_discoveries(p_value, eligible, q)
        top = _top_quintile(means, eligible, top_fraction)
        selection = eligible & (t_stat > min_t) & fdr & top
        if cfg.get("extra_filter", "none") == "negative_intraday":
            intraday_sample = intraday[start:row + 1, stock_cols]
            intraday_mean = np.nanmean(intraday_sample, axis=0)
            selection &= np.isfinite(intraday_mean) & (intraday_mean < 0)
        selected = np.zeros(len(symbols), dtype=bool)
        selected[stock_cols] = selection
        selected_at_formation.append((int(row), selected))
        cohort_members.append(tuple(symbols[j] for j in np.flatnonzero(selected)))
        summary.append({
            "formation_date": dates[row],
            "formation_year": int(dates[row].year),
            "universe": int(universe.sum()),
            "valid_252d": int(eligible.sum()),
            "t_gt_threshold": int((eligible & (t_stat > min_t)).sum()),
            "bh_discoveries": int(fdr.sum()),
            "bh_p_cutoff": fdr_cutoff,
            "top_quintile": int(top.sum()),
            "selected": int(selection.sum()),
            "mean_selected_overnight_pct": 100 * float(np.nanmean(means[selection])) if selection.any() else np.nan,
            "selected_symbols": tuple(symbols[j] for j in np.flatnonzero(selected)),
        })

    membership = np.zeros(panel["close"].shape, dtype=bool)
    for i, (row, selection) in enumerate(selected_at_formation):
        stop = selected_at_formation[i + 1][0] if i + 1 < len(selected_at_formation) else len(dates)
        membership[row:stop] = selection[None, :]

    tradable = (rank <= universe_top_n) & np.isfinite(panel["close"])
    pool = tradable.copy()
    overnight_returns = overnight.copy()
    overnight_entry = np.asarray(aux["overnight_entry"], dtype=float).copy()
    overnight_exit = np.asarray(aux["overnight_exit"], dtype=float).copy()
    last_valid = np.where(np.isfinite(panel["close"]), np.arange(len(dates))[:, None], -1).max(0)
    delisted = np.zeros(panel["close"].shape, dtype=bool)
    for day in range(1, len(dates) - 1):
        cols = (last_valid == day - 1) & np.isfinite(overnight_entry[day])
        delisted[day, cols] = True
        overnight_returns[day, cols] = float(cfg.get("delisting_return", -0.30))
        overnight_exit[day, cols] = overnight_entry[day, cols] * (1 + float(cfg.get("delisting_return", -0.30)))
    trade_weights = np.zeros(panel["close"].shape, dtype=float)
    trend_exposure = np.zeros(len(dates), dtype=float)
    universe_overnight = np.full(len(dates), np.nan)
    universe_intraday = np.full(len(dates), np.nan)
    holding_mode = cfg.get("holding_mode", "overnight_only")
    spread_return = overnight_returns - intraday
    for day in range(1, len(dates)):
        candidate = tradable[day - 1, stock_cols] & np.isfinite(overnight_returns[day, stock_cols])
        if not candidate.any():
            continue
        universe_overnight[day] = float(np.nanmean(overnight[day, stock_cols][candidate]))
        if np.isfinite(intraday[day, stock_cols][candidate]).any():
            universe_intraday[day] = float(np.nanmean(intraday[day, stock_cols][candidate]))
        selected = membership[day - 1, stock_cols]
        if holding_mode == "full_day":
            active_return = intraday[day, stock_cols]
        elif holding_mode == "overnight_intraday":
            active_return = spread_return[day, stock_cols]
        else:
            active_return = overnight_returns[day, stock_cols]
        live = selected & np.isfinite(active_return)
        if not live.any():
            continue
        trade_weights[day, stock_cols[live]] = 1.0 / int(live.sum())
        trend_exposure[day] = 2.0 if holding_mode == "overnight_intraday" else 1.0

    if holding_mode == "full_day":
        benchmark = np.nan_to_num(universe_intraday)
        external_returns = {
            "SPY total return": np.nan_to_num(aux["spy_total_return"]),
            "SPY intraday": np.nan_to_num(intraday[:, symbols.index("SPY")]),
        }
    elif holding_mode == "overnight_intraday":
        benchmark = 0.5 * np.nan_to_num(universe_overnight - universe_intraday)
        external_returns = {
            "SPY overnight": np.nan_to_num(overnight[:, symbols.index("SPY")]),
            "SPY intraday": np.nan_to_num(intraday[:, symbols.index("SPY")]),
        }
    else:
        benchmark = np.nan_to_num(universe_overnight)
        external_returns = {
            "SPY overnight": np.nan_to_num(overnight[:, symbols.index("SPY")]),
            "SPY total return": np.nan_to_num(aux["spy_total_return"]),
        }
    return {
        "tradable": tradable,
        "pool": pool,
        "trade_weights": trade_weights,
        "membership": membership,
        "formation_rows": formation_rows,
        "formation_summary": summary,
        "cohort_members": cohort_members,
        "overnight_returns": overnight_returns,
        "overnight_entry": overnight_entry,
        "overnight_exit": overnight_exit,
        "delisted": delisted,
        "intraday_returns": intraday,
        "spread_returns": spread_return,
        "universe_overnight": np.nan_to_num(universe_overnight),
        "universe_intraday": np.nan_to_num(universe_intraday),
        "benchmark_returns": benchmark,
        "trend_exposure": trend_exposure,
        "benchmarks": external_returns,
    }


def build_trades(panel, sig, cfg, fill):
    dates = pd.DatetimeIndex(panel["dates"])
    symbols = list(panel["symbols"])
    mode = cfg.get("holding_mode", "overnight_only")
    weights = sig["trade_weights"]
    overnight = sig["overnight_returns"]
    intraday = sig["intraday_returns"]
    spread = sig["spread_returns"]
    cost = float(cfg["cost_bps_per_side"]) / 1e4
    rows = []
    for day in range(1, len(dates)):
        members = np.flatnonzero(weights[day] > 0)
        if not len(members):
            continue
        weight = 1.0 / len(members)
        for col in members:
            if mode == "full_day":
                entry_px, exit_px = panel["open"][day, col], panel["close"][day, col]
                entry_cost, exit_cost = weight, weight
            elif mode == "overnight_intraday":
                entry_px, exit_px = 1.0, 1.0 + spread[day, col]
                entry_cost, exit_cost = 2 * weight, 2 * weight
            else:
                entry_px = sig["overnight_entry"][day, col]
                exit_px = sig["overnight_exit"][day, col]
                entry_cost, exit_cost = weight, weight
            if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px <= 0:
                continue
            gross = weight * (exit_px / entry_px - 1.0)
            rows.append({
                "fill": fill, "symbol": symbols[col], "signal_idx": day - 1, "entry_idx": day, "exit_idx": day,
                "signal_date": dates[day - 1], "entry_date": dates[day], "exit_date": dates[day],
                "entry_px": float(entry_px), "exit_px": float(exit_px), "trade_weight": weight,
                "entry_cost_weight": entry_cost, "exit_cost_weight": exit_cost,
                "ret": gross - cost * (entry_cost + exit_cost), "ret_gross": gross,
                "hold": 1,
                "exit_reason": "delisted" if sig["delisted"][day, col] else ("overnight" if mode == "overnight_only" else mode),
                "sym_idx": int(col), "random_fixed": False,
            })
    return pd.DataFrame(rows, columns=[
        "fill", "symbol", "signal_idx", "entry_idx", "exit_idx", "signal_date", "entry_date", "exit_date",
        "entry_px", "exit_px", "trade_weight", "entry_cost_weight", "exit_cost_weight", "ret", "ret_gross",
        "hold", "exit_reason", "sym_idx", "random_fixed",
    ])
