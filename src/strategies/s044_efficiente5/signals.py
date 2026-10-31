"""S044 monthly discrete mean-variance allocation and portfolio construction."""

import numpy as np
import pandas as pd

from src.utils import timing

from .optimizer import solve_integer_portfolio

ASSETS = ("SPY", "IWM", "EFA", "TLT", "LQD", "HYG", "EEM", "EMB",
          "IYR", "GSG", "GLD", "TIP", "BIL")


def rebalance_rows(dates, frequency):
    period = "M" if frequency == "monthly" else "W-FRI"
    dates = pd.DatetimeIndex(dates)
    periods = dates.to_period(period)
    return pd.Series(np.arange(len(dates))).groupby(periods).max().to_numpy(int)


def _returns(close):
    result = np.full(close.shape, np.nan)
    result[1:] = close[1:] / close[:-1] - 1.0
    return result


def compute_signals(panel, cfg):
    symbols = list(panel["symbols"])
    asset_idx = np.array([symbols.index(symbol) for symbol in ASSETS], dtype=int)
    close = panel["close"][:, asset_idx]
    open_ = panel["open"][:, asset_idx]
    returns = _returns(close)
    n_rows = len(panel["dates"])
    window = int(cfg["lookback"])
    rebalance = rebalance_rows(panel["dates"], cfg.get("rebalance", "monthly"))
    weights_by_row, diagnostics = {}, []
    trend_exposure = np.zeros(n_rows, dtype=float)
    for row in rebalance:
        if row < window:
            continue
        sample = returns[row - window + 1:row + 1]
        if len(sample) != window or not np.isfinite(sample).all():
            continue
        with timing.stage("discrete integer allocation"):
            mean = sample.mean(0)
            covariance = np.cov(sample, rowvar=False, ddof=1)
            weights, target, mean_portfolio, realized, min_var_fallback, solver_calls = solve_integer_portfolio(
                mean, covariance,
                target_vol=float(cfg["target_vol"]),
                max_target_vol=float(cfg.get("max_target_vol", 0.50)),
                grid_path=cfg.get("weight_grid_path", "data/cache/s044_efficiente5_weight_grid_v1.npy"),
            )
        weights_by_row[int(row)] = weights
        next_row_pos = int(np.searchsorted(rebalance, row, side="right"))
        end = int(rebalance[next_row_pos]) if next_row_pos < len(rebalance) else n_rows - 1
        trend_exposure[int(row) + 1:end + 2] = 2.0
        diagnostics.append({
            "signal_date": panel["dates"][row],
            "target_vol": target,
            "realized_vol": realized,
            "mean_daily_return": mean_portfolio,
            "minimum_variance_fallback": bool(min_var_fallback),
            "solver_calls": int(solver_calls),
            "weights": weights.copy(),
        })

    tradable = np.isfinite(panel["close"])
    if len(tradable):
        tradable[: min(window, len(tradable))] = False
    pool = tradable.copy()
    bil_idx, spy_idx, tlt_idx = (symbols.index(s) for s in ("BIL", "SPY", "TLT"))
    daily = returns[:, asset_idx]
    count = np.isfinite(daily).sum(1)
    ew = np.divide(np.nansum(daily, axis=1), count, out=np.zeros(n_rows), where=count > 0)
    bil = returns[:, bil_idx]
    spy = returns[:, spy_idx]
    tlt = returns[:, tlt_idx]
    b_day = np.nan_to_num(0.5 * (ew - bil))
    benchmarks = {
        "SPY total return": np.nan_to_num(spy),
        "60/40 SPY-TLT": np.nan_to_num(0.6 * spy + 0.4 * tlt),
    }
    return {
        "tradable": tradable,
        "pool": pool,
        "asset_idx": asset_idx,
        "weights_by_row": weights_by_row,
        "rebalance_rows": rebalance,
        "optimization_diagnostics": diagnostics,
        "benchmark_returns": b_day,
        "trend_exposure": trend_exposure,
        "benchmarks": benchmarks,
        "daily_asset_returns": daily,
        "equal_weight_excess": np.nan_to_num(ew - bil),
        "spy_return": np.nan_to_num(spy),
        "sixty_forty": np.nan_to_num(0.6 * spy + 0.4 * tlt),
        "bil_return": np.nan_to_num(bil),
    }


def build_trades(panel, sig, cfg, fill):
    dates = pd.DatetimeIndex(panel["dates"])
    symbols = list(panel["symbols"])
    asset_idx = sig["asset_idx"]
    rebalance = np.asarray(sorted(sig["weights_by_row"]), dtype=int)
    entry_mat = panel["open"] if fill == "next_open" else panel["close"]
    exit_mat = panel["open"] if fill == "next_open" else panel["close"]
    cost = float(cfg["cost_bps_per_side"]) / 1e4
    rows = []
    weights_by_row = sig["weights_by_row"]
    for i, signal_row in enumerate(rebalance):
        weights = np.asarray(weights_by_row[int(signal_row)], dtype=float)
        previous = np.asarray(weights_by_row[int(rebalance[i - 1])], dtype=float) if i else np.zeros(len(ASSETS))
        following = np.asarray(weights_by_row[int(rebalance[i + 1])], dtype=float) if i + 1 < len(rebalance) else np.zeros(len(ASSETS))
        next_row = int(rebalance[i + 1]) if i + 1 < len(rebalance) else len(dates) - 1
        entry_row = int(signal_row + 1) if fill == "next_open" else int(signal_row)
        requested_exit = int(next_row + 1) if fill == "next_open" else int(next_row)
        window_end = requested_exit >= len(dates)
        exit_row = min(requested_exit, len(dates) - 1)
        if entry_row >= len(dates):
            continue
        for j, asset in enumerate(ASSETS):
            weight = weights[j]
            if weight <= 0:
                continue
            col = asset_idx[j]
            entry_px = entry_mat[entry_row, col]
            exit_px = panel["close"][exit_row, col] if window_end else exit_mat[exit_row, col]
            if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px <= 0:
                continue
            entry_fee = abs(weight - previous[j])
            exit_fee = 0.0 if window_end or following[j] > 0 else abs(weight)
            gross = weight * (exit_px / entry_px - 1.0)
            rows.append({
                "fill": fill, "symbol": asset, "signal_idx": int(signal_row),
                "entry_idx": entry_row, "exit_idx": exit_row,
                "signal_date": dates[signal_row], "entry_date": dates[entry_row], "exit_date": dates[exit_row],
                "entry_px": float(entry_px), "exit_px": float(exit_px), "trade_weight": float(weight),
                "entry_cost_weight": entry_fee, "exit_cost_weight": exit_fee,
                "ret": gross - cost * (entry_fee + exit_fee), "ret_gross": gross,
                "hold": int(exit_row - entry_row + 1), "exit_reason": "window_end" if window_end else "rebalance",
                "sym_idx": int(col), "random_fixed": False,
            })

        bil_col = symbols.index("BIL")
        entry_px = entry_mat[entry_row, bil_col]
        exit_px = panel["close"][exit_row, bil_col] if window_end else exit_mat[exit_row, bil_col]
        if np.isfinite(entry_px) and np.isfinite(exit_px) and entry_px > 0:
            weight = -1.0
            gross = weight * (exit_px / entry_px - 1.0)
            rows.append({
                "fill": fill, "symbol": "BIL funding", "signal_idx": int(signal_row),
                "entry_idx": entry_row, "exit_idx": exit_row,
                "signal_date": dates[signal_row], "entry_date": dates[entry_row], "exit_date": dates[exit_row],
                "entry_px": float(entry_px), "exit_px": float(exit_px), "trade_weight": weight,
                "entry_cost_weight": 0.0, "exit_cost_weight": 0.0,
                "ret": gross, "ret_gross": gross,
                "hold": int(exit_row - entry_row + 1), "exit_reason": "cash_financing",
                "sym_idx": bil_col, "random_fixed": True,
            })

    columns = [
        "fill", "symbol", "signal_idx", "entry_idx", "exit_idx", "signal_date", "entry_date", "exit_date",
        "entry_px", "exit_px", "trade_weight", "entry_cost_weight", "exit_cost_weight",
        "ret", "ret_gross", "hold", "exit_reason", "sym_idx", "random_fixed",
    ]
    return pd.DataFrame(rows, columns=columns)
