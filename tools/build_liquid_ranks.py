"""Build the stock-only PIT liquidity-rank cache used by S033 and S041.

  python tools/build_liquid_ranks.py --out data/cache/liquid_ranks
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.liquid_universe import compute_liquid_ranks, write_liquid_ranks  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "data" / "cache" / "liquid_ranks"))
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "cache" / "daily_10years"))
    args = parser.parse_args()
    dates, symbols, ranks = compute_liquid_ranks(args.data_dir)
    write_liquid_ranks(args.out, dates, symbols, ranks)
    ever500 = int((ranks <= 500).any(axis=0).sum())
    print(f"ranks {ranks.shape}; {len(dates)} sessions, {len(symbols)} stocks, {ever500} ever in top-500 -> {args.out}")


if __name__ == "__main__":
    main()
