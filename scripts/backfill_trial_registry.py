"""Back-fill results/trial_registry.jsonl with approximate (estimated=True) entries for the search work done before the registry existed.

  C:/data/envs/trading_framework/Scripts/python.exe scripts/backfill_trial_registry.py [--force]

Counts come from the code and result files; per-trial totals are in total log return of the 5 x 20% account over the stated business
days (approximations are named in the notes). Entries whose count cannot be established carry n_trials = null.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.backtest import partition as P
from src.backtest.pit_metrics import select_trades, simulate_account

DEV_DAYS = len(pd.bdate_range("2024-10-18", "2026-06-15"))


def account_total(path):
    t = pd.read_csv(path)
    t["entry_time"] = t["entry_time"].astype(str)
    if "rel_vol" not in t:
        return np.nan
    return float(np.log1p(simulate_account(t, max_pos=5, rank="rel_vol")[2]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    reg = P.load_registry()
    if len(reg) and not a.force:
        raise SystemExit(f"registry already has {len(reg)} entries; use --force to append the back-fill again")
    base = pd.read_csv("results/pit_trades_atr_pivot_lb13_pm_target25_v2.csv")
    base["entry_time"] = base["entry_time"].astype(str)
    n_sel = len(select_trades(base, 5, "rel_vol"))
    tbl = Path("reports/exit/tables")
    tot = lambda df: (0.2 * n_sel * df["mean_ret"]).to_numpy()  # approx total log: 20% x selected trades x mean trade return
    v2 = pd.read_csv(tbl / "v2_trailing.csv")
    grid = pd.read_csv(tbl / "v2_trailing_grid.csv")
    fix = pd.read_csv(tbl / "v2_baselines_and_fixed_time.csv")
    kw = dict(estimated=True, source="backfill", stat_kind="account_total_log")
    P.log_trials("exit", len(v2) + len(grid) + len(fix), np.concatenate([tot(v2), tot(grid), tot(fix)]), days=DEV_DAYS,
                 notes=f"exit_rerun_v2: {len(v2)} trailing random + {len(grid)} grid + {len(fix)} baseline/fixed-time (valid re-run). per-trial total approximated as 0.2 * {n_sel} selected trades * mean trade return", **kw)
    old = sum(len(pd.read_csv(f)) for f in glob.glob(str(tbl / "*.csv")) if os.path.basename(f).split(".")[0] in
              ("trailing", "momentum", "scale_out", "pm_ah", "combos", "learned", "optimal_stopping", "fixed_time") or os.path.basename(f).startswith("robustness_"))
    P.log_trials("exit", old, None, invalid=True, notes=f"first exit_campaign run (pre-fix simulator, results invalid; report says 6,001 attempted): {old} rows in the family tables. Excluded from haircut B", **kw)
    variants = [f for f in sorted(glob.glob("results/pit_trades_atr_pivot_*.csv")) if not any(k in f for k in ("_S1", "_S2", "_H1", "_H2", "_H3", "rolling"))]
    P.log_trials("exit", len(variants), [account_total(f) for f in variants], days=DEV_DAYS,
                 notes="legacy exit-variant trade files results/pit_trades_atr_pivot_*.csv (v1, v2, w30_*, ah_full, atr_stop, confirm2, pm*/stop* variants, lb13); per-trial = account total log over its own file", **kw)
    ex = pd.read_csv("results/explore_pm_target25_v2.csv")
    import explore
    P.log_trials("entry_filters", len(explore.FILTERS) + len(explore.STACKS), np.log1p(ex["total%"] / 100), days=DEV_DAYS,
                 notes=f"explore.py: {len(explore.FILTERS)} single filters + {len(explore.STACKS)} stacks on pm_target25_v2 (S1/H3/H2 come from these); per-trial from results/explore_pm_target25_v2.csv ({len(ex)} rows)", **kw)
    it = pd.read_csv("results/iterations_dev.csv")
    P.log_trials("entry_filters", len(it), np.log1p(it["total%"] / 100), days=DEV_DAYS, notes="iterate.py iterations_dev.csv (exit variants + filters + sizing off v2 / pm25 bases; different sizing rules)", **kw)
    P.log_trials("entry_filters", 7, None, notes="rank_test.STRATS (7 named filter stacks: base, S1, H3, H2 and <=100M variants), not separately scored in a table of totals", **kw)
    ws = pd.read_csv("reports/ranking/weight_search.csv")
    P.log_trials("ranking", 700, np.log1p(ws["train_total%"] / 100), days=int(0.6 * DEV_DAYS),
                 notes=f"rank_test.tune_weights weights x threshold random search: 700 draws, {len(ws)} valid; per-trial = train segment (60% of dev days) total log", **kw)
    P.log_trials("ranking", 62, None, notes="rank_test dev_experiments/ridge: hygiene 4, default-rank/threshold 8, STRATS x 4 rankings 28, ablation 12, scales 3, boost 3, ridge thresholds 4 (counted by hand from code)", **kw)
    P.log_trials("signal", None, None, notes="signal detector / lookback choice (atr_pivot, lookback 13 months, sweep_resistance GRIDS): count not established; sweep_resistance tuned on label F1 not on returns", **kw)
    print(P.load_registry()[["area", "n_trials", "days", "estimated", "invalid", "notes"]].to_string(max_colwidth=70))


if __name__ == "__main__":
    main()
