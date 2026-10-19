"""
Build the point-in-time daily dataset from free sources (Massive, MarketParquet, yfinance).

  python build_pit_dataset.py reference            # Massive ticker reference, active + delisted
  python build_pit_dataset.py massive [--days N]   # grouped daily bars, resumable (~12.5s per date)
  python build_pit_dataset.py mp                   # MarketParquet free last-365-days files
  python build_pit_dataset.py alpaca-daily         # Alpaca daily bars for the whole universe (primary source)
  python build_pit_dataset.py candidates           # daily prefilter -> data/pit/candidates.parquet
  python build_pit_dataset.py alpaca-minute        # 1-min bars for candidate symbol-days
  python build_pit_dataset.py build                # merge into data/pit/daily_pit.parquet
  python build_pit_dataset.py validate             # coverage + cross-source checks, writes report.txt
"""

import argparse
import sys
import os
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from src.data import pit_dataset as pit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["reference", "massive", "mp", "alpaca-daily", "build", "candidates", "alpaca-minute", "validate"])
    ap.add_argument("--days", type=int, default=None, help="limit number of dates fetched (massive)")
    ap.add_argument("--start", default=None, help="start date YYYY-MM-DD (massive); default 2 years back")
    ap.add_argument("--end", default=None, help="end date YYYY-MM-DD (massive); default yesterday")
    a = ap.parse_args()

    if a.step == "reference":
        pit.fetch_reference()
    elif a.step == "massive":
        end = date.fromisoformat(a.end) if a.end else date.today() - timedelta(days=1)
        start = date.fromisoformat(a.start) if a.start else end - timedelta(days=728)
        pit.fetch_massive_daily(start, end, max_days=a.days)
    elif a.step == "mp":
        pit.fetch_marketparquet()
    elif a.step == "alpaca-daily":
        end = date.fromisoformat(a.end) if a.end else date.today() - timedelta(days=1)
        start = date.fromisoformat(a.start) if a.start else date(2024, 9, 1)
        pit.fetch_alpaca_daily(start, end)
    elif a.step == "candidates":
        import pandas as pd
        c = pit.daily_candidates(pd.read_parquet(pit.PIT_DIR / "daily_pit.parquet"))
        c = c[c["date"] >= "2024-10-18"]
        c.to_parquet(pit.PIT_DIR / "candidates.parquet", index=False)
        print(f"{len(c)} candidate symbol-days, {c['symbol'].nunique()} symbols, {c['date'].nunique()} dates")
    elif a.step == "alpaca-minute":
        import pandas as pd
        pit.fetch_alpaca_minute(pd.read_parquet(pit.PIT_DIR / "candidates.parquet"))
    elif a.step == "build":
        pit.build_daily()
    else:
        pit.validate()


if __name__ == "__main__":
    main()
