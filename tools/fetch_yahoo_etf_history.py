"""Fetch Yahoo daily OHLCV, dividends, splits, and adjusted close for ETF studies.

Example:
  python tools/fetch_yahoo_etf_history.py --start 2008-01-01 --end 2026-01-01
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.liquid_universe import load_liquid_ranks

OUT_DIR = ROOT / "data" / "cache" / "etf_yahoo_2008_2025"
SYMBOLS = ("SPY", "IWM", "EFA", "TLT", "LQD", "HYG", "EEM", "EMB",
           "IYR", "GSG", "GLD", "TIP", "BIL")
BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"


def annual_formation_symbols(top_n):
    dates, symbols, ranks = load_liquid_ranks()
    selected = set()
    for year in sorted(set(dates.year)):
        rows = (dates.year == year).nonzero()[0]
        if len(rows) and dates[rows[-1]].month == 12:
            selected.update(symbols[i] for i in (ranks[rows[-1]] <= top_n).nonzero()[0])
    return sorted(selected)


def _timestamp(value):
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


def _fetch(session, symbol, start, end):
    url = f"{BASE_URL}/{symbol}"
    params = {
        "period1": _timestamp(start),
        "period2": _timestamp(end),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    for attempt in range(6):
        response = session.get(url, params=params, timeout=45)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 5:
                response.raise_for_status()
            time.sleep(min(2 ** attempt, 30))
            continue
        response.raise_for_status()
        payload = response.json()["chart"]
        if payload.get("error"):
            raise ValueError(f"{symbol}: {payload['error']}")
        result = payload.get("result")
        if not result:
            raise ValueError(f"{symbol}: Yahoo returned no chart data")
        result = result[0]
        quote = result["indicators"]["quote"][0]
        adjusted = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
        if adjusted is None:
            raise ValueError(f"{symbol}: adjusted close missing")
        timezone_name = result.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
        dates = pd.to_datetime(result.get("timestamp", []), unit="s", utc=True)
        dates = dates.tz_convert(timezone_name).tz_localize(None).normalize()
        frame = pd.DataFrame({
            "symbol": symbol,
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "adj_close": adjusted,
            "volume": quote["volume"],
        }, index=dates)
        frame.index.name = "date"

        events = result.get("events", {})
        dividends = {int(k): v["amount"] for k, v in events.get("dividends", {}).items()}
        splits = {int(k): v for k, v in events.get("splits", {}).items()}
        utc_dates = dates.tz_localize(timezone_name).tz_convert("UTC")
        stamps = [int(ts.timestamp()) for ts in utc_dates]
        frame["dividend"] = [dividends.get(ts, 0.0) for ts in stamps]
        frame["split_numerator"] = [splits.get(ts, {}).get("numerator", 0.0) for ts in stamps]
        frame["split_denominator"] = [splits.get(ts, {}).get("denominator", 0.0) for ts in stamps]
        frame.index = utc_dates
        frame.index.name = "timestamp"
        return frame.dropna(subset=["open", "close", "adj_close"])
    raise RuntimeError(f"Yahoo request failed for {symbol}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--symbols-file")
    parser.add_argument("--include-symbols", nargs="*", default=[])
    parser.add_argument("--ever-top-n", type=int, default=0)
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--end", default="2026-01-01", help="exclusive end date")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    symbols = set(args.symbols) | set(args.include_symbols)
    if args.symbols_file:
        symbols.update(x.strip() for x in Path(args.symbols_file).read_text(encoding="utf-8").splitlines() if x.strip())
    if args.ever_top_n:
        symbols.update(annual_formation_symbols(args.ever_top_n))
    if not symbols:
        symbols.update(SYMBOLS)
    symbols = sorted(symbols)
    out_dir = Path(args.out_dir)
    print(f"{len(symbols)} symbols; {args.start}..{args.end}; output={out_dir}")
    if args.plan:
        print("Plan only; no network request made.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; TradingFrameworkResearch/1.0)"})
    failed = []
    for i, symbol in enumerate(symbols, 1):
        path = out_dir / f"{symbol}_1Day_yahoo.parquet"
        if path.exists() and not args.force:
            continue
        try:
            frame = _fetch(session, symbol, args.start, args.end)
            frame.to_parquet(path)
            print(f"{i}/{len(symbols)} {symbol}: {len(frame)} bars, "
                  f"{frame.index.min().date()}..{frame.index.max().date()}", flush=True)
        except Exception as exc:
            failed.append((symbol, str(exc)))
            print(f"FAILED {symbol}: {exc}", flush=True)
        time.sleep(0.8)
    if failed:
        print(f"{len(failed)} symbols failed: {failed}")
        raise SystemExit(1)
    print("Complete.")


if __name__ == "__main__":
    main()
