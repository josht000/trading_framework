"""S041 12-1 cross-sectional momentum: monthly top-k of the liquid universe by t-252..t-21 TR return."""

import numpy as np
import pandas as pd

from src.strategies._engines import align_ranks, rebalance_trades


def month_end_rows(dates):
    """Last session of each calendar month (row indices)."""
    months = pd.Series(dates).groupby(dates.to_period("M")).max()
    return np.searchsorted(np.asarray(dates), months.to_numpy())


def compute_signals(panel, cfg, rank):
    """tradable (liquid top-N PIT), score = 12-1 month TR momentum, rebalance row indices."""
    close = panel["close"]
    mom = np.full(close.shape, np.nan)
    skip, win = int(cfg["skip_window"]), int(cfg["mom_window"])
    mom[win:] = close[win - skip: mom.shape[0] - skip if skip else mom.shape[0]] / close[: mom.shape[0] - win] - 1.0
    n = int(cfg["universe_top_n"])
    tradable = np.isfinite(close) & (rank <= n)
    if cfg.get("industry_neutral"):
        ind = np.asarray(panel.get("industry", [""] * len(panel["symbols"])), dtype=object)
        for code in set(ind[ind != ""]):
            cols = np.flatnonzero(ind == code)
            sub = mom[:, cols]
            mean = np.where(np.isfinite(sub) & (rank[:, cols] <= n), sub, np.nan)
            with np.errstate(invalid="ignore"):
                row_mean = np.nanmean(mean, axis=1)
            mom[:, cols] = np.where(np.isnan(row_mean[:, None]), np.nan, sub - row_mean[:, None])
    return {"tradable": tradable, "pool": tradable, "score": mom,
            "rebalance_rows": month_end_rows(panel["dates"])}


def build_trades(panel, sig, cfg, fill):
    k = int(cfg["breadth"]) if cfg.get("breadth_mode", "decile") == "top" else None
    if k is None:
        elig = (sig["tradable"] & np.isfinite(sig["score"])).sum(1)
        k = max(int(cfg.get("min_breadth", 1)), int(np.median(elig[elig > 0] * float(cfg["decile"])))) if (elig > 0).any() else 1
    return rebalance_trades(panel, sig["score"], sig["tradable"], sig["rebalance_rows"], k, cfg, fill,
                            delisting_return=float(cfg.get("delisting_return", -0.30)))
