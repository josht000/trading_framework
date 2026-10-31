"""Precompute the PIT liquid-universe rank matrix over the stock cache and cache it.

Rank per session = trailing 60-day median dollar volume rank among common stocks
(ETF symbols excluded), price >= $5 and >= 250 prior bars required - the catalog's
liquid_top_N universe rule. Stored as int16 npy + a json index; strategies load it
instead of recomputing ranks on 11k symbols each run.

  python tools/build_liquid_ranks.py --out data/cache/liquid_ranks
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.daily_panel import liquidity_ranks, load_panel  # noqa: E402
from tools.fetch_etf_daily import ETF_SYMBOLS  # noqa: E402

ETF_LIKE = set(ETF_SYMBOLS) | {"GLD", "SLV", "USO", "CPER", "GSG", "EMB", "IYR", "TIP", "UUP", "DBC", "VNQ",
                              "BIL", "SQQQ", "GDX", "GDXJ", "ACWX", "EWJ", "XHB"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cache/liquid_ranks")
    args = parser.parse_args()
    out = Path(args.out)
    panel = load_panel(ROOT / "data" / "cache" / "daily_10years")
    symbols = np.asarray(panel["symbols"], dtype=object)
    keep = np.array([s not in ETF_LIKE for s in symbols])
    close, volume = panel["close"][:, keep], panel["volume"][:, keep]
    symbols = symbols[keep]
    rank = liquidity_ranks(close, volume).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ranks.npy", rank)
    (out / "index.json").write_text(json.dumps({"dates": [str(d.date()) for d in panel["dates"]],
                                               "symbols": symbols.tolist()}), encoding="utf-8")
    ever500 = int((rank <= 500).any(0).sum())
    print(f"ranks {rank.shape}; {ever500} names ever in top-500 -> {out}")


def load_liquid_ranks(root="data/cache/liquid_ranks"):
    """(dates, symbols, rank float matrix) - importable helper for strategies."""
    root = Path(root)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    return pd.DatetimeIndex(index["dates"]), index["symbols"], np.load(root / "ranks.npy")


if __name__ == "__main__":
    main()
