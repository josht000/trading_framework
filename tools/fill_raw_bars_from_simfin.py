"""Fill Alpaca raw-bar gaps from the local SimFin daily price archive.

SimFin daily OHLC is split-adjusted and does not replace the separate
corporate-action cache. Use this only for a verified symbol/date gap.

Example:
  python tools/fill_raw_bars_from_simfin.py --symbols MPW --start 2023-09-01 --end 2025-10-11
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "data" / "simfin" / "us-shareprices-daily.csv"
OUT_DIR = ROOT / "data" / "cache" / "daily_10years_raw"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", required=True, help="inclusive date")
    parser.add_argument("--end", required=True, help="exclusive date")
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    symbols = set(args.symbols)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    source, out_dir = Path(args.source), Path(args.out_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.plan:
        print(f"Would scan {source} for {len(symbols)} symbols, {start.date()}..{end.date()}.")
        return

    collected = {symbol: [] for symbol in symbols}
    with source.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream, delimiter=";"):
            symbol = row.get("Ticker")
            if symbol not in symbols or not row.get("Date"):
                continue
            date = pd.Timestamp(row["Date"])
            if not start <= date < end:
                continue
            collected[symbol].append({
                "date": date,
                "open": row.get("Open"),
                "high": row.get("High"),
                "low": row.get("Low"),
                "close": row.get("Close"),
                "volume": row.get("Volume"),
            })

    from src.data.alpaca_history import merge_bar_file

    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol, records in collected.items():
        if not records:
            print(f"{symbol}: no SimFin rows in range")
            continue
        frame = pd.DataFrame(records)
        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["open", "close", "volume"])
        local = pd.DatetimeIndex(frame["date"]).tz_localize("America/New_York").tz_convert("UTC")
        bars = pd.DataFrame({
            "symbol": symbol,
            "open": frame["open"].to_numpy(float),
            "high": frame["high"].to_numpy(float),
            "low": frame["low"].to_numpy(float),
            "close": frame["close"].to_numpy(float),
            "volume": frame["volume"].to_numpy(float),
            "trade_count": np.nan,
            "vwap": np.nan,
        }, index=local)
        bars.index.name = "timestamp"
        path = out_dir / f"{symbol.replace('/', '_')}_1Day_raw.parquet"
        rows = merge_bar_file(path, bars)
        print(f"{symbol}: merged {len(frame)} rows; cache now has {rows} bars")


if __name__ == "__main__":
    main()
