"""Rebuild results/exit/windows.npz (+ windows_index.parquet) from the cleaned minute files.

trade_id = row position in concat(dev table, holdout table); duplicate-security trades get no window.
The old cache is kept as windows_pre_dedupe.npz.  No returns are read from the holdout table.

  C:/data/envs/trading_framework/Scripts/python.exe scripts/build_exit_windows.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest.exit_sim import build_windows, load_windows
from src.data.pit_integrity import DEV_TRADES, TEST_TRADES, WINDOWS, dedupe_trades

OUT = WINDOWS.parent


def read_trades():
    kw = dict(keep_default_na=False, na_values=[""])  # ticker "NAN" must not become NaN
    dev, test = (pd.read_csv(p, **kw).assign(period=n) for p, n in ((DEV_TRADES, "dev"), (TEST_TRADES, "test")))
    return pd.concat([dev, test], ignore_index=True)


def main():
    t = read_trades()
    old = OUT / "windows_pre_dedupe.npz"
    if WINDOWS.exists() and not old.exists():
        shutil.copy2(WINDOWS, old)
    print(f"trades: {len(t)} (dev {int((t.period == 'dev').sum())}, test {int((t.period == 'test').sum())}); "
          f"after dedupe_trades {len(dedupe_trades(t))}")
    ws = build_windows(t, ROOT / "data" / "pit" / "alpaca_minute", WINDOWS, print)
    ws = load_windows(WINDOWS)
    idx = pd.read_parquet(OUT / "windows_index.parquet")
    print(f"windows: {len(ws)} ({idx.period.value_counts().to_dict()}), index rows {len(idx)}")


if __name__ == "__main__":
    main()
