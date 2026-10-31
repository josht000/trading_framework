"""Write the union of symbols that ever rank in the trailing-60d-dollar-volume top N.

Used to bound the total-return refetch to the names the liquid_top_N universe can
actually pick (S041/S033), plus a buffer.

  python tools/liquid_union.py --n 600 --out data/cache/daily_10years_tr/_universe.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.daily_panel import liquidity_ranks, load_panel  # noqa: E402
from tools.fetch_etf_daily import ETF_SYMBOLS  # noqa: E402

EXCLUDE = set(ETF_SYMBOLS) | {"GLD", "SLV", "USO", "CPER", "GSG", "EMB", "IYR", "TIP", "UUP", "DBC", "VNQ"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    panel = load_panel(ROOT / "data" / "cache" / "daily_10years")
    rank = liquidity_ranks(panel["close"], panel["volume"])
    hit = (rank <= args.n).any(axis=0)
    symbols = sorted(s for s, h in zip(panel["symbols"], hit) if h and s not in EXCLUDE)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(symbols) + "\n", encoding="utf-8")
    print(f"{int(hit.sum())} names ever in top {args.n}; {len(symbols)} after ETF exclusion -> {args.out}")


if __name__ == "__main__":
    main()
