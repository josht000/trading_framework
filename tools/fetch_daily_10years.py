"""Build a full Alpaca daily-bar cache from its free-tier history floor.

Reuses the existing Alpaca PIT bars (2023 onward) and the smaller legacy
per-symbol cache, then backfills 2016-01-04 through 2023-08-31.
"""

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.alpaca_history import (
    BAR_COLUMNS,
    START,
    cache_path as _cache_path,
    fetch_bars_batch,
    merge_bar_file,
    read_daily_cache,
)
from src.data.pit_dataset import NY, universe_symbols

BACKFILL_END = pd.Timestamp("2023-09-01")  # exclusive
OUT_DIR = ROOT / "data" / "cache" / "daily_10years"
STATE_DIR = OUT_DIR / "_state"


def cache_path(symbol):
    return _cache_path(OUT_DIR, symbol, "_1Day_split.parquet")


def read_cache(path):
    return read_daily_cache(path)


def merge_write(symbol, bars):
    merge_bar_file(cache_path(symbol), bars[BAR_COLUMNS])


def seed_existing():
    """Copy all already-downloaded Alpaca daily data into the requested layout."""
    frames = []
    parts_dir = ROOT / "data" / "pit" / "alpaca_daily"
    for part in sorted(parts_dir.glob("part_*.parquet")):
        frame = pd.read_parquet(part)
        if not frame.empty:
            ts = pd.to_datetime(frame["date"]).dt.tz_localize(NY).dt.tz_convert("UTC")
            frame = frame.drop(columns=["date"]).assign(timestamp=ts)
            frames.append(frame)

    if frames:
        data = pd.concat(frames, ignore_index=True)
        for i, (symbol, group) in enumerate(data.groupby("symbol", sort=False), 1):
            merge_write(symbol, group[BAR_COLUMNS])
            if i % 500 == 0:
                print(f"  reused Alpaca PIT bars for {i}/{data['symbol'].nunique()} symbols", flush=True)

    legacy_dir = ROOT / "data" / "cache" / "daily"
    for i, path in enumerate(sorted(legacy_dir.glob("*_1Day_split.parquet")), 1):
        symbol = path.name.removesuffix("_1Day_split.parquet")
        if "/" in symbol or "\\" in symbol:
            continue
        old = pd.read_parquet(path).reset_index()
        if "timestamp" not in old:
            old = old.rename(columns={old.columns[0]: "timestamp"})
        merge_write(symbol, old[BAR_COLUMNS])
        if i % 100 == 0:
            print(f"  reused legacy cache files: {i}", flush=True)


def fetch_batch(session, symbols, next_request, start=START, end=BACKFILL_END):
    fetch_bars_batch(
        session, symbols, next_request, start, end,
        adjustment="split", out_dir=OUT_DIR, suffix="_1Day_split.parquet",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=5,
                        help="symbols per request (default 5 keeps the 2016-2023 history near one API page)")
    parser.add_argument("--plan", action="store_true", help="show universe and request count without downloading")
    parser.add_argument("--trim-after", type=date.fromisoformat,
                        help="remove bars after this ET calendar date, without making API requests")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    os.chdir(ROOT)
    if args.trim_after:
        cutoff = pd.Timestamp(args.trim_after)
        files = list(OUT_DIR.glob("*_1Day_split.parquet"))
        for i, path in enumerate(files, 1):
            frame = pd.read_parquet(path)
            keep = frame.index.tz_convert(NY).date <= args.trim_after
            if not keep.all():
                frame.loc[keep].to_parquet(path)
            if i % 1000 == 0 or i == len(files):
                print(f"  checked {i}/{len(files)} files against {cutoff.date()}", flush=True)
        return

    load_dotenv(ROOT / ".env")
    symbols = universe_symbols(START)
    batches = [symbols[i:i + args.batch_size] for i in range(0, len(symbols), args.batch_size)]
    print(f"Alpaca free-tier daily cache: {len(symbols)} symbols; "
          f"backfill {START.date()}..{(BACKFILL_END - pd.Timedelta(days=1)).date()}; "
          f"{len(batches)} request batches; output={OUT_DIR}", flush=True)
    if args.plan:
        return

    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET are required")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    seed_existing()

    session = requests.Session()
    session.headers.update({
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    })
    next_request = [time.monotonic()]
    started = time.monotonic()
    failed = []
    completed = 0
    for batch_index, batch in enumerate(batches):
        marker = STATE_DIR / f"batch_{batch_index:05d}.done"
        if marker.exists():
            completed += 1
            continue
        try:
            fetch_batch(session, batch, next_request)
            marker.write_text("\n".join(batch) + "\n", encoding="utf-8")
        except Exception as exc:
            failed.append((batch_index, batch, str(exc)))
            print(f"FAILED batch {batch_index + 1}/{len(batches)}: {exc}", flush=True)
            continue
        completed += 1
        if completed % 25 == 0 or completed == len(batches):
            elapsed = (time.monotonic() - started) / 60
            print(f"  completed {completed}/{len(batches)} batches; "
                  f"{elapsed:.1f} min; last symbols={batch[0]}..{batch[-1]}", flush=True)

    # The existing all-universe Alpaca pull ends at its latest saved session.
    recent_start = None
    for part in sorted((ROOT / "data" / "pit" / "alpaca_daily").glob("part_*.parquet")):
        dates = pd.read_parquet(part, columns=["date"])["date"]
        if len(dates):
            latest = pd.Timestamp(dates.max()).normalize()
            recent_start = latest + pd.Timedelta(days=1) if recent_start is None else max(
                recent_start, latest + pd.Timedelta(days=1))
    if recent_start is not None:
        recent_end = pd.Timestamp.now(tz=NY).normalize().tz_localize(None) + pd.Timedelta(days=1)
        if recent_start < recent_end:
            recent_symbols = universe_symbols(recent_start)
            recent_batches = [
                recent_symbols[i:i + args.batch_size]
                for i in range(0, len(recent_symbols), args.batch_size)
            ]
            print(f"Fetching latest sessions {recent_start.date()}.."
                  f"{(recent_end - pd.Timedelta(days=1)).date()} for "
                  f"{len(recent_symbols)} listed symbols", flush=True)
            for batch_index, batch in enumerate(recent_batches):
                marker = STATE_DIR / (
                    f"recent_{recent_start.date()}_{recent_end.date()}_{batch_index:05d}.done")
                if marker.exists():
                    continue
                try:
                    fetch_batch(session, batch, next_request, recent_start, recent_end)
                    marker.write_text("\n".join(batch) + "\n", encoding="utf-8")
                except Exception as exc:
                    failed.append((f"recent-{batch_index}", batch, str(exc)))
                    print(f"FAILED recent batch {batch_index + 1}/{len(recent_batches)}: {exc}", flush=True)
                if (batch_index + 1) % 25 == 0 or batch_index + 1 == len(recent_batches):
                    print(f"  latest sessions: {batch_index + 1}/{len(recent_batches)} batches", flush=True)

    if failed:
        print(f"Finished with {len(failed)} failed batches. Rerun to retry the unmarked batches.", flush=True)
        raise SystemExit(1)
    print(f"Complete: {completed}/{len(batches)} batches in "
          f"{(time.monotonic() - started) / 60:.1f} min; "
          f"{len(list(OUT_DIR.glob('*_1Day_split.parquet')))} per-symbol files.", flush=True)


if __name__ == "__main__":
    main()
