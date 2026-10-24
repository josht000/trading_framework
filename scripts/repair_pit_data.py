"""Repair existing PIT files in place: drop duplicate bars / alias securities / unusable daily bars.

Every file that changes is first copied to data/pit/_backup_dedupe_<timestamp>/ (same relative path); the cleaned
file is written to a temp file, re-read and verified before it replaces the original.  Dry run unless --apply.

  C:/data/envs/trading_framework/Scripts/python.exe scripts/repair_pit_data.py [--apply]
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.pit_integrity import PIT_DIR, clean_daily, clean_minute, dedupe_on, drop_daily_aliases, traded_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    backup = PIT_DIR / f"_backup_dedupe_{time.strftime('%Y%m%d_%H%M%S')}"
    daily = pd.read_parquet(PIT_DIR / "daily_pit.parquet")
    jobs = [(PIT_DIR / "daily_pit.parquet", clean_daily, ["symbol", "date"], daily)]
    cands = pd.read_parquet(PIT_DIR / "candidates.parquet")
    new = dedupe_on(drop_daily_aliases(cands, daily, traded_pairs()), ["symbol", "date"]).reset_index(drop=True)
    gone = set(zip(cands["date"], cands["symbol"])) - set(zip(new["date"], new["symbol"]))
    jobs.append((PIT_DIR / "candidates.parquet", lambda c: new, ["symbol", "date"], None))
    jobs.append((PIT_DIR / "news" / "news.parquet", lambda n: dedupe_on(n, ["symbol", "created_at", "headline"]).reset_index(drop=True),
                 ["symbol", "created_at", "headline"], None))
    drop_gone = lambda f: (lambda m: clean_minute(m[[(pd.Timestamp(f.stem), s) not in gone for s in m["symbol"]]]))
    jobs += [(f, drop_gone(f), ["symbol", "timestamp"], None) for f in sorted((PIT_DIR / "alpaca_minute").glob("*.parquet"))]
    jobs += [(f, lambda m: dedupe_on(m, ["timestamp"]).reset_index(drop=True), ["timestamp"], None)
             for f in sorted((PIT_DIR / "market").glob("*_minute.parquet"))]
    changed = removed = 0
    for path, fn, key, loaded in jobs:
        x = loaded if loaded is not None else pd.read_parquet(path)
        if not len(x):
            continue
        y = fn(x)
        if len(y) == len(x):
            continue
        changed += 1
        removed += len(x) - len(y)
        rel = path.relative_to(PIT_DIR)
        print(f"{rel}: {len(x)} -> {len(y)} rows (-{len(x) - len(y)})")
        if not a.apply:
            continue
        (backup / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup / rel)
        tmp = path.with_suffix(".tmp")
        y.to_parquet(tmp, index=False)
        z = pd.read_parquet(tmp)
        assert len(z) == len(y) and not z.duplicated(key).any() and list(z.columns) == list(x.columns), f"verify failed {rel}"
        assert (backup / rel).stat().st_size == path.stat().st_size
        tmp.replace(path)
    print(f"{changed} files changed, {removed} rows removed" + (f"; backups in {backup}" if a.apply and changed else " (dry run)"))


if __name__ == "__main__":
    main()
