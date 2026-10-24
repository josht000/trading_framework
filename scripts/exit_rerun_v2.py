"""Re-run of the fixed_time and trailing exit experiments after the exit_sim fixes.

DEV ONLY (entries before the locked test period); realistic fills; same 5 date-block CV objective
(`_score` from exit_campaign.py) for baselines and candidates.  Writes reports/exit/tables/v2_*.csv
and reports/exit/figures/v2_*.png.

  C:/data/envs/trading_framework/Scripts/python.exe scripts/exit_rerun_v2.py --trials 1500 --procs 16
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exit_campaign import _masks, _paired_bootstrap, _score, score_vs_baseline
from src.backtest.exit_sim import baseline_policy, fixed_time_policy, load_windows, simulate
from src.backtest import kfold
from src.backtest.market_regime import attach_regime
from src.backtest.pit_metrics import dedupe_trades
from src.data.pit_integrity import require_valid_pit_data

TABLES = ROOT / "reports" / "exit" / "tables"
FIGS = ROOT / "reports" / "exit" / "figures"
CLOSES = ["09:31", "09:35", "09:40", "09:45", "10:00", "10:15", "10:30"]
SEED = 20261022
W = T = None


def load_dev():
    """Dev trades (pre locked test) with regime columns; duplicated securities removed (dedupe_trades)."""
    import os
    os.chdir(ROOT)
    dev = pd.read_csv(ROOT / "results" / "pit_trades_atr_pivot_lb13_pm_target25_v2.csv")
    dev["entry_time"] = dev["entry_time"].astype(str)
    assert pd.to_datetime(dev["entry_time"], utc=True).max() < pd.Timestamp("2026-06-16", tz="UTC")
    dev = attach_regime(dev)
    keep = dev.index.isin(dedupe_trades(dev).index)
    ws = [w for w in load_windows(ROOT / "results" / "exit" / "windows.npz") if w.trade_id < len(dev) and keep[w.trade_id]]
    trades = dev.iloc[[w.trade_id for w in ws]].reset_index(drop=True)
    return ws, trades


def _init():
    global W, T
    W, T = load_dev()


def _run(policy):
    res = simulate(W, policy, "realistic")
    return _score(res, T)


def run_many(policies, procs):
    with Pool(procs, initializer=_init) as pool:
        return pool.map(_run, policies, chunksize=4)


def stats_rows(label, res, trades):
    masks = _masks(trades)
    rows = []
    for name in ("base", "S1", "H3"):
        r = res["ret"][masks[name]].dropna()
        rows.append({"policy": label, "subset": name, "n": len(r), "hit": (r > 0).mean(), "mean": r.mean(),
                     "median": r.median(),
                     "pf": r[r > 0].sum() / abs(r[r <= 0].sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1500)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--skip-validate", action="store_true")
    a = ap.parse_args()
    if not a.skip_validate:
        require_valid_pit_data()
    t0 = time.time()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    ws, trades = load_dev()
    _masks(trades)
    print(f"dev windows after dedupe: {len(ws)}; S1 uses IWM/SPY: {_masks.s1_full}; "
          f"S1 n={int(_masks(trades)['S1'].sum())} H3 n={int(_masks(trades)['H3'].sum())}", flush=True)

    # ---- baselines (same scoring function, identical trade set, realistic fills)
    base_pols = {"a_v2_pm25_0800": baseline_policy(), **{f"fixed_close_{c}": fixed_time_policy(c) for c in CLOSES}}
    results = {k: simulate(ws, p, "realistic") for k, p in base_pols.items()}
    scores = {k: _score(results[k], trades) for k in base_pols}
    A, B = scores["a_v2_pm25_0800"], scores["fixed_close_09:31"]
    rows = []
    for k, sc in scores.items():
        rows.append({"policy": k, "params": json.dumps(base_pols[k], sort_keys=True), **sc,
                     **{f"{x}_vs_a": v for x, v in score_vs_baseline(sc, A).items()},
                     **{f"{x}_vs_b": v for x, v in score_vs_baseline(sc, B).items()}})
    fx = pd.DataFrame(rows)
    fx.to_csv(TABLES / "v2_baselines_and_fixed_time.csv", index=False)
    fx[fx["policy"].str.startswith("fixed")].to_csv(TABLES / "v2_fixed_time.csv", index=False)
    print(fx[["policy", "score", "mean_ret", "hit_rate", "m_base", "m_S1", "m_H3", "dJ_vs_a", "dJ_vs_b"]].to_string(), flush=True)

    # ---- trailing random search
    rng = np.random.default_rng(SEED)
    pols = [{"name": "smart", "keep_ah": False, "deadline": str(rng.choice(["09:45", "10:00", "10:15", "10:30"])),
             "trail": float(rng.uniform(.01, .16)), "activate": float(rng.uniform(0, .30)),
             "peak_from": str(rng.choice(["entry", "open"]))} for _ in range(a.trials)]
    sc = run_many(pols, a.procs)
    tr = pd.DataFrame([{"trial": i, "params": json.dumps(p, sort_keys=True), **p, **s,
                        **{f"{x}_vs_a": v for x, v in score_vs_baseline(s, A).items()},
                        **{f"{x}_vs_b": v for x, v in score_vs_baseline(s, B).items()}}
                       for i, (p, s) in enumerate(zip(pols, sc))])
    tr.to_csv(TABLES / "v2_trailing.csv", index=False)
    print(f"trailing done {time.time() - t0:.0f}s", flush=True)

    # ---- trailing grid (deadline 10:15), both peak_from
    gp = [{"name": "smart", "keep_ah": False, "deadline": "10:15", "trail": t / 100, "activate": ac / 100,
           "peak_from": pf} for pf in ("open", "entry") for t in range(1, 17) for ac in (0, 2, 5, 10, 15, 20, 30)]
    gs = run_many(gp, a.procs)
    grid = pd.DataFrame([{**p, **s, **{f"{x}_vs_a": v for x, v in score_vs_baseline(s, A).items()},
                          **{f"{x}_vs_b": v for x, v in score_vs_baseline(s, B).items()}} for p, s in zip(gp, gs)])
    grid.to_csv(TABLES / "v2_trailing_grid.csv", index=False)
    print(f"grid done {time.time() - t0:.0f}s", flush=True)

    # ---- top 3: per-subset stats and paired bootstrap
    valid = tr[tr["valid"]].sort_values("dJ_vs_a", ascending=False)
    top = valid.head(10)
    top.to_csv(TABLES / "v2_trailing_top10.csv", index=False)
    srows = []
    for k in ("a_v2_pm25_0800", "fixed_close_09:31", "fixed_close_09:35", "fixed_close_09:45", "fixed_close_10:00", "fixed_close_10:15"):
        srows += stats_rows(k, results[k], trades)
    boots = []
    days = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert("America/New_York").dt.date.to_numpy()
    top_res = {}
    for r, (_, row) in enumerate(valid.head(3).iterrows(), 1):
        pol = json.loads(row["params"])
        res = simulate(ws, pol, "realistic")
        top_res[f"top{r}"] = res
        srows += stats_rows(f"top{r} trail={pol['trail']:.3f} act={pol['activate']:.3f} dl={pol['deadline']} pf={pol['peak_from']}", res, trades)
        for bname in ("a_v2_pm25_0800", "fixed_close_09:31"):
            boots.append({"candidate": f"top{r}", "params": row["params"], "vs": bname, **_paired_bootstrap(results[bname], res, days)})
    # Chronological k=5 stability, fixed configurations only; no refitting in folds.
    def write_kfold(label, result):
        frame = trades.copy()
        frame["ret"] = result["ret"].to_numpy()
        frame["exit_time"] = result["exit_time"].astype(str).to_numpy()
        frame["day"] = pd.to_datetime(frame["entry_time"].astype(str).str[:10])
        kd = pd.bdate_range(frame["day"].min(), frame["day"].max())
        kr, ks = kfold.kfold_stability(frame, kd, rank="rel_vol", pool=frame, label=f"{label}|dev")
        kfold.write_kfold(kr, ks, ROOT / "reports" / "exit", label, "dev")
    for label in ("a_v2_pm25_0800", "fixed_close_09:31", "fixed_close_09:35", "fixed_close_09:45",
                  "fixed_close_10:00", "fixed_close_10:15"):
        write_kfold(label, results[label])
    for label, result in top_res.items():
        write_kfold(label, result)
    pd.DataFrame(srows).to_csv(TABLES / "v2_top3_subset_stats.csv", index=False)
    pd.DataFrame(boots).to_csv(TABLES / "v2_top3_bootstrap.csv", index=False)
    # also bootstrap of best fixed-time vs baseline a
    bf = fx[fx["policy"].str.startswith("fixed")].sort_values("dJ_vs_a", ascending=False).iloc[0]["policy"]
    fb = [{"candidate": bf, "vs": "a_v2_pm25_0800", **_paired_bootstrap(results["a_v2_pm25_0800"], results[bf], days)}]
    for k in ("fixed_close_09:35",):
        fb.append({"candidate": k, "vs": "a_v2_pm25_0800", **_paired_bootstrap(results["a_v2_pm25_0800"], results[k], days)})
        fb.append({"candidate": k, "vs": "fixed_close_09:31", **_paired_bootstrap(results["fixed_close_09:31"], results[k], days)})
    pd.DataFrame(fb).to_csv(TABLES / "v2_fixed_bootstrap.csv", index=False)


    # ---- figures
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, pf in zip(axes, ("open", "entry")):
        g = grid[grid["peak_from"] == pf].pivot(index="trail", columns="activate", values="dJ_vs_a").sort_index(ascending=False)
        v = np.nanmax(np.abs(g.to_numpy()))
        im = ax.imshow(g.to_numpy() * 100, aspect="auto", cmap="RdBu", vmin=-v * 100, vmax=v * 100)
        ax.set_xticks(range(len(g.columns)), [f"{c:.0%}" for c in g.columns])
        ax.set_yticks(range(len(g.index)), [f"{c:.0%}" for c in g.index])
        ax.set_xlabel("activate (peak gain required)")
        ax.set_ylabel("trail (drop from peak)")
        ax.set_title(f"dev: delta J vs baseline (a), peak_from={pf}, deadline 10:15 [x100]", fontsize=9)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ax.text(j, i, f"{g.to_numpy()[i, j] * 100:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGS / "v2_trailing_grid_heatmap.png", dpi=130)
    plt.close(fig)

    f = fx[fx["policy"].str.startswith("fixed")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [p.replace("fixed_close_", "") for p in f["policy"]]
    for ax, col, title in ((axes[0], "m_base", "mean net return (all trades)"), (axes[1], "hit_base", "hit rate (all trades)")):
        ax.bar(labels, f[col] * 100)
        ax.axhline(A[col] * 100, color="tab:red", ls="--", label="baseline (a) v2/pm25 08:00")
        ax.axhline(B[col] * 100, color="tab:green", ls=":", label="(b) close of 09:30 bar (09:31)")
        ax.set_title(f"fixed-time exit, dev, realistic: {title}", fontsize=9)
        ax.set_xlabel("decision at close of bar ending at (ET)")
        ax.set_ylabel("%")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "v2_fixed_time_bars.png", dpi=130)
    plt.close(fig)
    print(f"all done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
