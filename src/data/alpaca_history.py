"""Reusable Alpaca daily-history request and parquet helpers."""

import time
from pathlib import Path

import pandas as pd

API_URL = "https://data.alpaca.markets/v2/stocks/bars"
PAGE_LIMIT = 10_000
REQUEST_INTERVAL = 0.31
START = pd.Timestamp("2016-01-04")
BAR_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close",
               "volume", "trade_count", "vwap"]


def alpaca_page(session, symbols, params, next_request):
    """Fetch one API page while respecting Alpaca's free-tier request ceiling."""
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


def normalize_bars(rows, symbol):
    """Convert Alpaca's compact bar rows into the repository's daily-bar schema."""
    bars = pd.DataFrame(rows).rename(columns={
        "t": "timestamp", "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "n": "trade_count", "vw": "vwap",
    })
    if bars.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars.insert(0, "symbol", symbol)
    return bars[BAR_COLUMNS]


def merge_bar_file(path, bars):
    """Merge normalized bars into a per-symbol parquet file, preferring new rows."""
    path = Path(path)
    if bars.empty:
        return 0
    new = bars.set_index("timestamp")
    new.index = pd.to_datetime(new.index, utc=True)
    new.index.name = "timestamp"
    if path.exists():
        old = pd.read_parquet(path)
        merged = pd.concat([old, new]).sort_index()
    else:
        merged = new.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path)
    return len(merged)


def cache_path(out_dir, symbol, suffix):
    return Path(out_dir) / f"{symbol.replace('/', '_')}{suffix}"


def read_daily_cache(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(
            columns=["symbol", "open", "high", "low", "close", "volume", "trade_count", "vwap"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )
    return pd.read_parquet(path)


def fetch_bars_batch(session, symbols, next_request, start, end, *, adjustment, out_dir, suffix):
    """Fetch one page-paginated daily-bar batch and merge each symbol to parquet."""
    symbols = list(symbols)
    params = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        "start": start.strftime("%Y-%m-%dT00:00:00Z"),
        "end": end.strftime("%Y-%m-%dT00:00:00Z"),
        "limit": PAGE_LIMIT,
        "adjustment": adjustment,
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

    written = {}
    for symbol, rows in output.items():
        bars = normalize_bars(rows, symbol)
        written[symbol] = merge_bar_file(cache_path(out_dir, symbol, suffix), bars) if len(bars) else 0
    return written
