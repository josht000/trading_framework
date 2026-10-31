"""PIT liquidity-rank cache shared by stock-universe data tools and studies."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.daily_panel import liquidity_ranks, load_panel
from src.data.market_universes import ETF_LIKE_SYMBOLS


def compute_liquid_ranks(data_dir, *, exclude=ETF_LIKE_SYMBOLS):
    """Return aligned dates, common-stock symbols, and PIT trailing-dollar-volume ranks."""
    panel = load_panel(data_dir)
    symbols = np.asarray(panel["symbols"], dtype=object)
    keep = np.array([symbol not in exclude for symbol in symbols], dtype=bool)
    stock_symbols = symbols[keep].tolist()
    ranks = liquidity_ranks(panel["close"][:, keep], panel["volume"][:, keep]).astype(np.float32)
    return panel["dates"], stock_symbols, ranks


def write_liquid_ranks(out_dir, dates, symbols, ranks):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ranks.npy", ranks)
    (out / "index.json").write_text(
        json.dumps({"dates": [str(date.date()) for date in dates], "symbols": list(symbols)}),
        encoding="utf-8",
    )


def load_liquid_ranks(root="data/cache/liquid_ranks"):
    """Return `(dates, symbols, rank_matrix)` from a built liquid-rank cache."""
    root = Path(root)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    return pd.DatetimeIndex(index["dates"]), index["symbols"], np.load(root / "ranks.npy")


def ever_top_symbols(symbols, ranks, top_n):
    """Sorted stock symbols that entered the PIT top-N universe on any session."""
    hit = np.any(ranks <= int(top_n), axis=0)
    return sorted(symbol for symbol, is_hit in zip(symbols, hit) if is_hit)
