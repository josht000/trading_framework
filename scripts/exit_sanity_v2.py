"""Sanity checks on real dev windows after the exit_sim fixes (dev only)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from src.backtest.exit_sim import load_windows, simulate, baseline_policy, fixed_time_policy, _clock, _ts

dev = pd.read_csv(ROOT / "results/pit_trades_atr_pivot_lb13_pm_target25_v2.csv", parse_dates=["entry_time"])
W = [w for w in load_windows(ROOT / "results/exit/windows.npz") if w.trade_id < len(dev)]
print("dev windows", len(W))
leg = simulate(W, baseline_policy(), "legacy")
d = (leg["ret"].to_numpy() - dev.iloc[leg["trade_id"]]["ret"].to_numpy())
print("legacy vs stored: n=%d max|diff|=%.3g  n(|diff|>1e-9)=%d" % (len(d), np.nanmax(np.abs(d)), int((np.abs(d) > 1e-9).sum())))
bad = np.flatnonzero(np.abs(d) > 1e-9)[:5]
print(pd.DataFrame({"i": leg["trade_id"].iloc[bad], "mine": leg["ret"].iloc[bad], "stored": dev.iloc[leg["trade_id"].iloc[bad]]["ret"].to_numpy(), "reason": leg["reason"].iloc[bad], "stored_reason": dev.iloc[leg["trade_id"].iloc[bad]]["exit_reason"].to_numpy()}))
print("\nfixed-time (realistic):")
for c in ["09:31", "09:35", "09:40", "09:45", "10:00", "10:15", "10:30"]:
    r = simulate(W, fixed_time_policy(c), "realistic")["ret"].dropna()
    print(f"  close {c}: n={len(r)} mean={r.mean():.4%} hit={(r > 0).mean():.2%}")
print("\ntrailing exit-minute distribution (trail 3%, activate 0, deadline 10:15, realistic decision minute):")
r = simulate(W, {"name": "smart", "keep_ah": False, "trail": .03, "activate": 0.0, "deadline": "10:15"}, "realistic")
mins = []
for w, di, rs in zip(W, r["decision_idx"], r["reason"]):
    mins.append(int(_clock(w)[1][di]) - 570)
r["m"] = mins
print(r["reason"].value_counts().to_string())
print(pd.Series(mins).describe().to_string())
print(pd.cut(pd.Series(mins), [-1000, -1, 0, 4, 9, 19, 29, 44, 45, 1000]).value_counts().sort_index().to_string())
