"""Backfill split+dividend (total-return) daily bars into data/cache/daily_10years_tr.

The main cache is split-adjusted only (`adjustment=split`), so buy-and-hold,
momentum and rotation studies that must consume the catalog's `tr` series
understate returns by the dividend yield. This refetches symbols with
`adjustment=all` and writes `<SYM>_1Day_tr.parquet` files (same schema).

  python tools/fetch_tr_daily.py --symbols SPY QQQ TLT
  python tools/fetch_tr_daily.py --file data/cache/daily_10years_tr/_universe.txt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.alpaca_history import START, fetch_bars_batch

OUT_DIR = ROOT / "data" / "cache" / "daily_10years_tr"
_out_dir = OUT_DIR


def out_path(symbol):
    return _out_dir / f"{symbol.replace('/', '_')}_1Day_tr.parquet"


def fetch_symbol(session, symbol, next_request, start, end):
    counts = fetch_bars_batch(
        session, [symbol], next_request, start, end,
        adjustment="all", out_dir=_out_dir, suffix="_1Day_tr.parquet",
    )
    return counts[symbol]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", help="symbols to fetch")
    parser.add_argument("--file", help="text file with one symbol per line")
    parser.add_argument("--start", default=str(START.date()))
    parser.add_argument("--end", help="default: latest bar in the split cache + 1 day")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="output folder for <SYM>_1Day_tr.parquet files")
    args = parser.parse_args()

    symbols = list(args.symbols or [])
    if args.file:
        symbols += [x.strip() for x in Path(args.file).read_text().splitlines() if x.strip()]
    symbols = sorted(set(symbols))
    if not symbols:
        raise SystemExit("no symbols: pass --symbols or --file")

    os.chdir(ROOT)
    load_dotenv(ROOT / ".env")
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET are required")

    global _out_dir
    _out_dir = Path(args.out_dir)
    state_dir = _out_dir / "_state"
    start = pd.Timestamp(args.start)
    if args.end:
        end = pd.Timestamp(args.end)
    else:
        sample = pd.read_parquet(ROOT / "data" / "cache" / "daily_10years" / "SPY_1Day_split.parquet")
        end = sample.index.max() + pd.Timedelta(days=1)

    _out_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    next_request = [time.monotonic()]
    failed = []
    done = 0
    for i, symbol in enumerate(symbols, 1):
        marker = state_dir / f"{symbol}.done"
        if marker.exists() and out_path(symbol).exists():
            done += 1
            continue
        try:
            n = fetch_symbol(session, symbol, next_request, start, end)
            if n:
                marker.write_text(f"{n}\n", encoding="utf-8")
                done += 1
            else:
                failed.append(symbol)
        except Exception as exc:
            failed.append(f"{symbol}: {exc}")
        if i % 200 == 0:
            print(f"  {i}/{len(symbols)} fetched", flush=True)

    print(f"{done}/{len(symbols)} symbols written to {_out_dir}")
    if failed:
        print(f"{len(failed)} failed/empty: {failed[:40]}")
    print("Complete.", flush=True)


if __name__ == "__main__":
    main()
