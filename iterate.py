"""
Offline strategy iteration on per-trade tables written by main.py (results/pit_trades_atr_pivot_<variant>.csv).

  python iterate.py                 # 12 iterations off each base on the DEV set (3 months held out)
  python iterate.py --test NAME,... # score named iterations once on the held-out final test set

An iteration = an exit variant (already simulated by main.py) + entry filters on features known at
15:55 + portfolio rules (max positions/day, sizing, ranking). Goal: lower max drawdown without
giving up Sharpe. Iterations 1-9 are single changes off the base; 10-12 were picked by hand after
seeing 1-9 (an automatic "combine whatever improved drawdown" rule found nothing to combine).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.backtest.pit_metrics import daily_returns, dedupe_trades, summarize

METHOD = "atr_pivot"
BASES = {"v2": "v2", "pm25": "pm_target25_v2"}
FILTERS = {
    "sdr<=0.8": lambda t: t["sig_day_ret"] <= 0.8,
    "atr<=0.15": lambda t: t["atr_pct"] <= 0.15,
    "range<=0.6": lambda t: t["range_pct"] <= 0.6,
}
SINGLES = [  # (key, description, spec patch). "exit" patches name the exit variant per base.
    ("pmstop15", "pre-market stop -15%", {"exit": "pmstop15"}),
    ("pmstop25", "pre-market stop -25%", {"exit": "pmstop25"}),
    ("sdr<=0.8", "skip signal-day move > +80%", {"filters": ["sdr<=0.8"]}),
    ("atr<=0.15", "skip daily ATR > 15% of price", {"filters": ["atr<=0.15"]}),
    ("range<=0.6", "skip intraday range > 60%", {"filters": ["range<=0.6"]}),
    ("slots", "1/5 capital per trade (cash on thin days)", {"sizing": "slots"}),
    ("pos3", "max 3 positions/day", {"max_pos": 3}),
    ("rank_liq", "rank by 20d dollar volume", {"rank": "dvol20"}),
    ("stop15", "after-hours stop -15%", {"exit": "stop15"}),
]
ROUND2 = [  # chosen after seeing round 1: no single change improved drawdown at similar Sharpe
    ("invatr", {"sizing": "invatr"}),
    ("invatr+stop15", {"sizing": "invatr", "exit": "stop15"}),
    ("slots+stop15", {"sizing": "slots", "exit": "stop15"}),
]
EXIT_NAMES = {  # (base, patch key) -> exit variant file name
    ("v2", "pmstop15"): "v2_pmstop15", ("v2", "pmstop25"): "v2_pmstop25", ("v2", "stop15"): "v2_stop15",
    ("pm25", "pmstop15"): "pm25_pmstop15", ("pm25", "pmstop25"): "pm25_pmstop25", ("pm25", "stop15"): "pm25_stop15",
}


def load(variant, holdout=False):
    p = os.path.join("results", "holdout" if holdout else "", f"pit_trades_{METHOD}_{variant}.csv")
    t = pd.read_csv(p, parse_dates=["entry_time"])
    return t


def evaluate(spec, holdout=False, days=None, window=None):
    t = dedupe_trades(load(spec["variant"], holdout))
    for f in spec.get("filters", []):
        t = t[FILTERS[f](t)]
    if window is not None:
        d = pd.to_datetime(t["entry_time"].astype(str).str[:10])
        t = t[(d >= window[0]) & (d <= window[1])]
    if t.empty:
        return None
    return summarize(t, spec.get("max_pos", 5), spec.get("rank", "rel_vol"), spec.get("sizing", "full"), days)


def full_days(holdout):
    t = load("v1", holdout)
    d = pd.to_datetime(t["entry_time"].astype(str).str[:10])
    return pd.bdate_range(d.min(), d.max())


def run_dev():
    days = full_days(False)
    mid = days[len(days) // 2]
    rows, specs = [], {}
    for bname, bvar in BASES.items():
        base = {"variant": bvar}
        res = {"base": evaluate(base, days=days)}
        specs[f"{bname}_base"] = base
        singles = {}
        for i, (key, desc, patch) in enumerate(SINGLES, 1):
            spec = dict(base)
            if "exit" in patch:
                spec["variant"] = EXIT_NAMES[(bname, patch["exit"])]
            spec.update({k: v for k, v in patch.items() if k != "exit"})
            singles[key] = (spec, patch)
            specs[f"{bname}_i{i:02d}_{key}"] = spec
            res[f"i{i:02d} {key}"] = evaluate(spec, days=days)
        for j, (key, patch) in enumerate(ROUND2, 10):
            spec = dict(base)
            if "exit" in patch:
                spec["variant"] = EXIT_NAMES[(bname, patch["exit"])]
            spec.update({k: v for k, v in patch.items() if k != "exit"})
            specs[f"{bname}_i{j}_{key}"] = spec
            res[f"i{j} {key}"] = evaluate(spec, days=days)
        for name, r in res.items():
            if r is None:
                continue
            key = name.split()[0]
            spec = specs[f"{bname}_base"] if name == "base" else next(
                v for k, v in specs.items() if k.startswith(f"{bname}_{key}_") or k == f"{bname}_{key}")
            h1 = evaluate(spec, days=days, window=(days[0], mid))
            h2 = evaluate(spec, days=days, window=(mid + pd.Timedelta(days=1), days[-1]))
            rows.append({"strategy": bname, "iteration": name, **r,
                         "sharpe_h1": h1["sharpe"] if h1 else np.nan, "sharpe_h2": h2["sharpe"] if h2 else np.nan,
                         "dd_h1": h1["maxDD%"] if h1 else np.nan, "dd_h2": h2["maxDD%"] if h2 else np.nan})
    out = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/iterations_dev.csv", index=False)
    with open("results/iteration_specs.json", "w") as f:
        json.dump(specs, f, indent=1)
    with pd.option_context("display.width", 280, "display.max_columns", 30, "display.max_colwidth", 60,
                           "display.float_format", "{:.2f}".format):
        for bname in BASES:
            print(f"\n=== {bname} (dev set) ===")
            print(out[out.strategy == bname].drop(columns="strategy").set_index("iteration"))
    return out


def run_test(names):
    specs = json.load(open("results/iteration_specs.json"))
    days = full_days(True)
    rows = []
    for n in names:
        r = evaluate(specs[n], holdout=True, days=days)
        rows.append({"spec": n, **(r or {})})
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print(pd.DataFrame(rows).set_index("spec"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=None, help="comma list of spec names from results/iteration_specs.json")
    a = ap.parse_args()
    run_test(a.test.split(",")) if a.test else run_dev()
