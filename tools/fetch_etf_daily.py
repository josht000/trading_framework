"""Backfill ETF daily bars into data/cache/daily_10years.

The main fetch_daily_10years.py universe skips reference type "ETF", so the
index/sector ETFs the catalog strategies need are absent from the cache.
This reuses its merge/fetch helpers to pull the catalog's ETF list over the
same 2016-01-04 -> present window with the same per-symbol parquet layout.
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

from src.data.alpaca_history import (
    START,
    cache_path as _cache_path,
    fetch_bars_batch,
    read_daily_cache,
)
from src.data.market_universes import ETF_SYMBOLS

OUT_DIR = ROOT / "data" / "cache" / "daily_10years"
STATE_DIR = OUT_DIR / "_state"
SUFFIX = "_1Day_split.parquet"


def cache_path(symbol):
    return _cache_path(OUT_DIR, symbol, SUFFIX)


def read_cache(path):
    return read_daily_cache(path)

def cache_end():
    """End boundary (exclusive) matching the latest bar already in the cache.

    Alpaca's free SIP plan rejects end dates past the next session, and the
    equity cache was trimmed to a common boundary, so mirror that date.
    """
    sample = read_cache(cache_path("GLD"))
    return sample.index.max() + pd.Timedelta(days=1)


def report_range(symbols):
    for symbol in symbols:
        path = cache_path(symbol)
        if not path.exists():
            print(f"  {symbol:5s} MISSING")
            continue
        frame = read_cache(path)
        if frame.empty:
            print(f"  {symbol:5s} EMPTY")
            continue
        print(f"  {symbol:5s} {frame.index.min().date()} -> "
              f"{frame.index.max().date()}  {len(frame)} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true",
                        help="show symbols and current coverage without downloading")
    parser.add_argument("--symbols", nargs="*", default=ETF_SYMBOLS,
                        help="override the ETF symbol list")
    args = parser.parse_args()

    os.chdir(ROOT)
    symbols = sorted(set(args.symbols))
    end = cache_end()
    print(f"ETF daily cache: {len(symbols)} symbols; "
          f"{START.date()}..{(end - pd.Timedelta(days=1)).date()}; output={OUT_DIR}")
    if args.plan:
        report_range(symbols)
        return

    load_dotenv(ROOT / ".env")
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET are required")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    next_request = [time.monotonic()]
    failed = []
    batch_size = 5
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        marker = STATE_DIR / f"etf_{'_'.join(batch)}.done"
        if marker.exists():
            print(f"  skip {batch} (done marker)", flush=True)
            continue
        try:
            fetch_bars_batch(
                session, batch, next_request, START, end,
                adjustment="split", out_dir=OUT_DIR, suffix=SUFFIX,
            )
            marker.write_text("\n".join(batch) + "\n", encoding="utf-8")
            print(f"  fetched {batch}", flush=True)
        except Exception as exc:
            failed.append((batch, str(exc)))
            print(f"FAILED {batch}: {exc}", flush=True)

    report_range(symbols)
    if failed:
        print(f"{len(failed)} failed batches; rerun to retry.", flush=True)
        raise SystemExit(1)
    print("Complete.", flush=True)


if __name__ == "__main__":
    main()
