"""Build (or --check) results/exit/oracle.parquet + oracle_meta.json: per-trade hindsight exit diagnostics.

  C:/data/envs/trading_framework/Scripts/python.exe scripts/build_exit_oracle.py [--check] [--perms 200]

Derived from results/exit/windows.npz; separate from data/pit (no hindsight in the PIT set).  --check exits 1 if stale.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest import exit_oracle as O
from src.backtest.exit_sim import load_windows
from src.data.pit_integrity import ORACLE, ORACLE_META, WINDOWS, WINDOWS_INDEX, oracle_staleness, require_valid_pit_data, sha256_file


def check():
    if not ORACLE.exists():
        print("STALE: oracle.parquet missing")
        return 1
    bad = oracle_staleness()
    m = json.loads(ORACLE_META.read_text()) if ORACLE_META.exists() else {}
    for f, h in m.get("code_sha256", {}).items():
        if sha256_file(ROOT / "src" / "backtest" / f) != h:
            print(f"NOTE: {f} changed since the oracle was built (not counted as stale; rebuild if simulator logic changed)")
    print("\n".join(f"STALE: {b}" for b in bad) if bad else f"oracle OK ({m.get('rows')} rows, built {m.get('created_utc')})")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--perms", type=int, default=O.N_PERM)
    a = ap.parse_args()
    if a.check:
        sys.exit(check())
    require_valid_pit_data()
    t0 = time.time()
    index = pd.read_parquet(WINDOWS_INDEX)
    ws = load_windows(WINDOWS)
    assert set(index["trade_id"]) == {w.trade_id for w in ws}, "windows_index.parquet does not match windows.npz"
    out = O.build_oracle(ws, index, a.perms)
    out.to_parquet(ORACLE, index=False)
    meta = O.make_meta(out)
    meta["params"]["rw_permutations"] = a.perms
    ORACLE_META.write_text(json.dumps(meta, indent=2))
    print(f"wrote {ORACLE} ({len(out)} rows, {len(out.columns)} cols) in {time.time() - t0:.0f}s")
    print(out["hindsight_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
