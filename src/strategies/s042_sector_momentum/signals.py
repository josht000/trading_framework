"""S042 sector momentum rotation: monthly top-k of the SPDR sector universe by trailing return rank."""

import numpy as np
import pandas as pd

from src.strategies._engines import rebalance_trades


def rebalance_rows(dates, freq):
    period = "W" if freq == "weekly" else "M"
    groups = pd.Series(dates).groupby(dates.to_period(period)).max()
    return np.searchsorted(np.asarray(dates), groups.to_numpy())


def compute_signals(panel, cfg, sector_idx):
    """tradable = the sector columns with enough history; score per signal mode."""
    close = panel["close"]
    mode = cfg.get("signal", "mom12")
    score = np.full(close.shape, np.nan)
    if mode == "rankavg":
        ranks = np.full(close.shape, np.nan)
        for w in (21, 63, 126, 189, 252):
            r = np.full(close.shape, np.nan)
            r[w:] = close[w:] / close[:-w] - 1.0
            order = np.argsort(np.argsort(-np.where(np.isfinite(r), r, -np.inf), axis=1, kind="stable"), axis=1, kind="stable")
            ranks = np.nansum(np.stack([np.where(np.isnan(ranks), 0, ranks), np.where(np.isfinite(r), order, np.nan)]), axis=0)
        score = -ranks
    else:
        windows = {"mom12": (252,), "blend13": (21, 63)}[mode]
        acc = np.zeros(close.shape)
        for w in windows:
            r = np.full(close.shape, np.nan)
            r[w:] = close[w:] / close[:-w] - 1.0
            acc += np.where(np.isfinite(r), r, np.nan)
        score = acc / len(windows)
    history = np.isfinite(close).cumsum(0)
    is_sector = np.zeros(close.shape[1], bool)
    is_sector[sector_idx] = True
    tradable = is_sector[None, :] & np.isfinite(close) & (history >= int(cfg["min_history"]))
    return {"tradable": tradable, "pool": tradable, "score": score,
            "rebalance_rows": rebalance_rows(panel["dates"], cfg.get("rebalance", "monthly"))}


def build_trades(panel, sig, cfg, fill):
    return rebalance_trades(panel, sig["score"], sig["tradable"], sig["rebalance_rows"],
                            int(cfg["breadth"]), cfg, fill)
