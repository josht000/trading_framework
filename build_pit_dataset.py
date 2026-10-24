"""
Build the point-in-time daily dataset from free sources (Massive, MarketParquet, yfinance).

  python build_pit_dataset.py reference            # Massive ticker reference, active + delisted
  python build_pit_dataset.py massive [--days N]   # grouped daily bars, resumable (~12.5s per date)
  python build_pit_dataset.py mp                   # MarketParquet free last-365-days files
  python build_pit_dataset.py alpaca-daily         # Alpaca daily bars for the whole universe (primary source)
  python build_pit_dataset.py market               # SPY/QQQ/IWM daily (from 2023-09) + 1-min bars -> data/pit/market/
  python build_pit_dataset.py news                 # Alpaca news for the candidate symbols -> data/pit/news/news.parquet
  python build_pit_dataset.py otc                  # Massive OTC / pink-sheet ticker list (excluded from trading)
  python build_pit_dataset.py edgar                # SEC EDGAR 8-K (item codes), offering and late-filing forms per candidate symbol
  python build_pit_dataset.py candidates           # daily prefilter -> data/pit/candidates.parquet
  python build_pit_dataset.py alpaca-minute        # 1-min bars for candidate symbol-days
  python build_pit_dataset.py build                # merge into data/pit/daily_pit.parquet
  python build_pit_dataset.py validate             # integrity checks (exit 1 on duplicates/bad bars/stale windows), then coverage + cross-source checks, writes report.txt
  python build_pit_dataset.py check                # integrity checks only (also run after candidates / alpaca-minute / build)
"""

import argparse
import sys
import os
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from src.data import pit_dataset as pit
from src.data.pit_integrity import require_valid_pit_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["reference", "massive", "mp", "alpaca-daily", "market", "news", "otc", "edgar", "build", "candidates", "alpaca-minute", "validate", "check"])
    ap.add_argument("--days", type=int, default=None, help="limit number of dates fetched (massive)")
    ap.add_argument("--start", default=None, help="start date YYYY-MM-DD (massive); default 2 years back")
    ap.add_argument("--end", default=None, help="end date YYYY-MM-DD (massive); default yesterday")
    ap.add_argument("--lookback-bars", type=int, default=None, help="atr_pivot lookback in daily bars (candidates); 273 = 13 months")
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
        start = date.fromisoformat(a.start) if a.start else pit.DAILY_START
        pit.fetch_alpaca_daily(start, end)
    elif a.step == "market":
        pit.fetch_market()
    elif a.step == "news":
        import pandas as pd
        pit.fetch_news(pd.read_parquet(pit.PIT_DIR / "candidates.parquet")["symbol"].unique())
    elif a.step == "otc":
        pit.fetch_otc_reference()
    elif a.step == "edgar":
        import pandas as pd
        pit.fetch_edgar(pd.read_parquet(pit.PIT_DIR / "candidates.parquet")["symbol"].unique())
    elif a.step == "candidates":
        import pandas as pd
        c = pit.daily_candidates(pd.read_parquet(pit.PIT_DIR / "daily_pit.parquet"), lookback_bars=a.lookback_bars)
        c = c[c["date"] >= "2024-10-18"]
        c.to_parquet(pit.PIT_DIR / "candidates.parquet", index=False)
        print(f"{len(c)} candidate symbol-days, {c['symbol'].nunique()} symbols, {c['date'].nunique()} dates")
    elif a.step == "alpaca-minute":
        import pandas as pd
        pit.fetch_alpaca_minute(pd.read_parquet(pit.PIT_DIR / "candidates.parquet"))
    elif a.step == "build":
        pit.build_daily()
    elif a.step == "validate":
        pit.validate()
    else:
        require_valid_pit_data()
    if a.step in ("candidates", "alpaca-minute", "build"):
        require_valid_pit_data(trades=False, windows=False)


if __name__ == "__main__":
    main()
