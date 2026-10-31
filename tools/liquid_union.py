"""Write stock symbols that ever enter the PIT trailing-dollar-volume top N.

  python tools/liquid_union.py --n 600 --out data/cache/daily_10years_tr/_universe.txt
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.liquid_universe import compute_liquid_ranks, ever_top_symbols  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "cache" / "daily_10years"))
    args = parser.parse_args()
    dates, symbols, ranks = compute_liquid_ranks(args.data_dir)
    names = ever_top_symbols(symbols, ranks, args.n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"{len(names)} common stocks ever in top {args.n} across {len(dates)} sessions -> {out}")


if __name__ == "__main__":
    main()
