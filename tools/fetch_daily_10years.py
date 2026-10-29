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

from src.data.pit_dataset import NY, universe_symbols

START = pd.Timestamp("2016-01-04")
BACKFILL_END = pd.Timestamp("2023-09-01")  # exclusive
OUT_DIR = ROOT / "data" / "cache" / "daily_10years"
STATE_DIR = OUT_DIR / "_state"
API_URL = "https://data.alpaca.markets/v2/stocks/bars"
PAGE_LIMIT = 10_000
REQUEST_INTERVAL = 0.31  # Alpaca free plan: 200 requests/minute
BAR_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close",
               "volume", "trade_count", "vwap"]


def cache_path(symbol):
    return OUT_DIR / f"{symbol.replace('/', '_')}_1Day_split.parquet"


def read_cache(path):
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "open", "high", "low", "close",
                                     "volume", "trade_count", "vwap"],
                            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    return pd.read_parquet(path)


def merge_write(symbol, bars):
    path = cache_path(symbol)
    old = read_cache(path)
    new = bars.set_index("timestamp")
    new.index = pd.to_datetime(new.index, utc=True)
    new.index.name = "timestamp"
    merged = pd.concat([old, new]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.to_parquet(path)


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


def alpaca_page(session, symbols, params, next_request):
    """Fetch one API page, respecting the free-tier request ceiling."""
    for attempt in range(6):
        wait = next_request[0] - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        next_request[0] = time.monotonic() + REQUEST_INTERVAL
        response = session.get(API_URL, params=params, timeout=90)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 5:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else min(2 ** attempt, 30))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Alpaca request failed for {symbols}")


def fetch_batch(session, symbols, next_request, start=START, end=BACKFILL_END):
    params = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        "start": start.strftime("%Y-%m-%dT00:00:00Z"),
        "end": end.strftime("%Y-%m-%dT00:00:00Z"),
        "limit": PAGE_LIMIT,
        "adjustment": "split",
        "feed": "sip",
        "sort": "asc",
    }
    output = {symbol: [] for symbol in symbols}
    while True:
        payload = alpaca_page(session, symbols, params, next_request)
        for symbol, rows in (payload.get("bars") or {}).items():
            output.setdefault(symbol, []).extend(rows)
        token = payload.get("next_page_token")
        if not token:
            break
        params["page_token"] = token

    for symbol, rows in output.items():
        if not rows:
            continue
        bars = pd.DataFrame(rows).rename(columns={
            "t": "timestamp", "o": "open", "h": "high", "l": "low",
            "c": "close", "v": "volume", "n": "trade_count", "vw": "vwap",
        })
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars.insert(0, "symbol", symbol)
        merge_write(symbol, bars[BAR_COLUMNS])


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
