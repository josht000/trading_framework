"""S105 volatility targeting from the overnight return actually held."""

import numpy as np


def compute_signals(panel, aux, cfg):
    overnight = aux["overnight_return"][:, 0]
    n_rows = len(overnight)
    window = int(cfg["vol_window"])
    target = float(cfg["target_vol"])
    cap = float(cfg["max_leverage"])
    band = float(cfg["resize_band"])
    exposure = np.zeros(n_rows, dtype=float)
    desired_path = np.full(n_rows, np.nan)
    current = 0.0
    for day in range(window + 1, n_rows):
        sample = overnight[day - window:day]
        sample = sample[np.isfinite(sample)]
        if len(sample) < window:
            continue
        annual_vol = float(sample.std(ddof=1) * np.sqrt(252))
        if not np.isfinite(annual_vol) or annual_vol <= 0:
            continue
        desired = min(cap, target / annual_vol)
        desired_path[day] = desired
        if current <= 0 or band <= 0 or abs(desired / current - 1.0) > band:
            current = desired
        exposure[day] = current

    tradable = np.zeros(panel["close"].shape, dtype=bool)
    tradable[window:, 0] = np.isfinite(overnight[window:])
    trade_weights = exposure[:, None]
    return {
        "tradable": tradable,
        "pool": tradable.copy(),
        "trade_weights": trade_weights,
        "overnight_entry": aux["overnight_entry"],
        "overnight_exit": aux["overnight_exit"],
        "benchmark_returns": np.nan_to_num(overnight),
        "trend_exposure": exposure,
        "benchmarks": {
            "SPY overnight": np.nan_to_num(overnight),
            "SPY total return": np.nan_to_num(aux["total_return"]),
        },
        "desired_exposure": desired_path,
        "overnight_returns": overnight,
    }
