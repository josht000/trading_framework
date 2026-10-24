"""Dev-only analysis of the hindsight oracle table (results/exit/oracle.parquet).

Writes reports/exit/tables/oracle_*.csv and reports/exit/figures/oracle_*.png.  Test-period rows are never loaded
into any aggregate (load_dev keeps only dev trades; asserted).

  C:/data/envs/trading_framework/Scripts/python.exe scripts/analyze_exit_oracle.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exit_campaign import _masks
from exit_rerun_v2 import FIGS, TABLES, load_dev
from src.backtest import exit_oracle as O
from src.backtest.exit_sim import _clock, _info, baseline_policy, fixed_time_policy, simulate
from src.data.pit_integrity import ORACLE, oracle_staleness

H = "hindsight_"
MEASURES = {"baseline_a": "ret_baseline", "baseline_legacy": "ret_baseline_legacy",
            **{f"fixed_{c}": f"ret_at_{c.replace(':', '')}" for c in O.CLOSES},
            "exec_oracle_open": "oracle_exec_ret_open", "exec_oracle_pm": "oracle_exec_ret_pm",
            "gross_peak_open": "peak_close_ret_open", "gross_peak_pm": "peak_close_ret_pm", "rw_gross_peak_open": "rw_peak_ret_open",
            "rw_gross_peak_open_firstbar_shuffled": "rw_peak_ret_open_firstbar_shuffled"}


def stats(x):
    x = pd.Series(x).dropna()
    return {"n": len(x), "mean": x.mean(), "median": x.median(), "hit": (x > 0).mean()}


def main():
    assert not oracle_staleness(), "oracle stale: rerun scripts/build_exit_oracle.py"
    ws, trades = load_dev()
    m = _masks(trades)
    groups = {"all_dev": m["base"], "S1": m["S1"], "H3": m["H3"]}
    ora = pd.read_parquet(ORACLE).set_index("trade_id").loc[[w.trade_id for w in ws]].reset_index(drop=True)
    assert (ora["period"] == "dev").all() and len(ora) == len(trades)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    col = lambda k: ora[H + MEASURES[k]].to_numpy(float)
    base = col("baseline_a")

    rows = []
    for g, mask in groups.items():
        for k in MEASURES:
            v = col(k)[mask]
            rows.append({"subset": g, "measure": k, **stats(v), "mean_minus_baseline": np.nanmean(v - base[mask]),
                         "baseline_mean": np.nanmean(base[mask])})
    head = pd.DataFrame(rows)
    head.to_csv(TABLES / "oracle_headroom.csv", index=False)

    top = pd.read_csv(TABLES / "v2_trailing_top10.csv").head(3)
    pols = {"baseline_a": None, "fixed_09:35": fixed_time_policy("09:35"), "fixed_09:45": fixed_time_policy("09:45"),
            "fixed_10:15": fixed_time_policy("10:15"), **{f"top{i + 1}_trailing": json.loads(r["params"]) for i, r in top.iterrows()}}
    res = {k: (base if p is None else simulate(ws, p, "realistic")["ret"].to_numpy(float)) for k, p in pols.items()}
    assert np.allclose(res["fixed_09:35"], col("fixed_09:35"), equal_nan=True)
    rows = []
    for g, mask in groups.items():
        for k, r in res.items():
            for o in ("exec_oracle_open", "exec_oracle_pm"):
                ok = mask & np.isfinite(r) & np.isfinite(col(o)) & np.isfinite(base)
                p, b, orc = r[ok].mean(), base[ok].mean(), col(o)[ok].mean()
                rows.append({"subset": g, "policy": k, "oracle": o, "n": int(ok.sum()), "policy_mean": p, "baseline_mean": b,
                             "oracle_mean": orc, "capture_ratio": p / orc, "gain_captured": (p - b) / (orc - b)})
    cap = pd.DataFrame(rows)
    cap.to_csv(TABLES / "oracle_capture.csv", index=False)

    gap = O.gap_bucket(ora[H + "open_vs_entry"]).astype(str).to_numpy()
    sets = {"all_dev": m["base"], **{f"gap {b}": gap == b for b in O.GAP_LABELS}, "H3": m["H3"], "not H3": ~m["H3"], "S1": m["S1"], "not S1": ~m["S1"]}
    off = ora[H + "peak_minute_open"].to_numpy(float)
    rows = []
    for g, mask in sets.items():
        rows.append({"group": g, "n": int(mask.sum()), "median_peak_minute": np.median(off[mask]),
                     **{f"share_peak_in_first_{k}min": (off[mask] < k).mean() for k in (1, 5, 15, 30, 45)},
                     "share_peak_at_10:15": (off[mask] >= 45).mean()})
    peak = pd.DataFrame(rows)
    peak.to_csv(TABLES / "oracle_peak_time.csv", index=False)

    ext = {k: np.full(len(ws), np.nan) for k in (5, 15, 30)}
    extop = {k: np.full(len(ws), np.nan) for k in (5, 15, 30)}
    for i, w in enumerate(ws):
        inf, (_, mins) = _info(w), _clock(w)
        nx = inf["next_idx"]
        if inf["first_next"] is None:
            continue
        o = w.opens[inf["first_next"]]
        for k in ext:
            j = nx[mins[nx] == 570 + k - 1]
            if len(j):
                ext[k][i], extop[k][i] = w.closes[j[0]] / w.entry_price - 1, w.closes[j[0]] / o - 1
    rows = []
    for g, mask in {"all_dev": m["base"], **{f"gap {b}": gap == b for b in O.GAP_LABELS}}.items():
        for k in ext:
            v, e = extop[k][mask], ext[k][mask]
            ok = np.isfinite(v)
            rows.append({"group": g, "minutes_after_open": k, "n": int(ok.sum()),
                         "up_vs_open": (v[ok] > 0).mean(), "down_vs_open": (v[ok] < 0).mean(), "flat_vs_open": (v[ok] == 0).mean(),
                         "above_entry": (e[ok] > 0).mean(), "below_entry": (e[ok] < 0).mean(), "mean_move_vs_open": v[ok].mean()})
    pd.DataFrame(rows).to_csv(TABLES / "oracle_early_moves.csv", index=False)

    rows = []
    for g, mask in {**groups, **{f"gap {b}": gap == b for b in O.GAP_LABELS}}.items():
        gp, rw, ex = col("gross_peak_open")[mask], col("rw_gross_peak_open")[mask], col("exec_oracle_open")[mask]
        rows.append({"group": g, "n": int(mask.sum()), "gross_peak_open": gp.mean(), "rw_gross_peak_open": rw.mean(),
                     "excess_hindsight_gross": (gp - rw).mean(), "excess_se": (gp - rw).std() / np.sqrt(mask.sum()),
                     "rw_firstbar_shuffled": col("rw_gross_peak_open_firstbar_shuffled")[mask].mean(),
                     "exec_oracle_open": ex.mean(), "exec_minus_rw": (ex - rw).mean(), "baseline_a": np.nanmean(base[mask])})
    rwt = pd.DataFrame(rows)
    rwt.to_csv(TABLES / "oracle_random_walk.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    bins = np.arange(0, 47)
    for b in ["all_dev"] + [f"gap {x}" for x in O.GAP_LABELS]:
        ax[0].hist(off[sets[b]], bins=bins, histtype="step", density=True, label=f"{b} (n={int(sets[b].sum())})", lw=2 if b == "all_dev" else 1.2)
    ax[0].set(xlabel="minutes after 09:30 of the peak close (bar label)", ylabel="share of trades per minute",
              title="dev: time of the 09:30-10:15 peak close, by open-vs-entry gap")
    ax[0].legend(fontsize=7)
    ax[1].bar(peak["group"], peak["share_peak_in_first_5min"], label="first 5 min", alpha=.8)
    ax[1].bar(peak["group"], peak["share_peak_in_first_15min"] - peak["share_peak_in_first_5min"], bottom=peak["share_peak_in_first_5min"], label="5-15", alpha=.8)
    ax[1].bar(peak["group"], peak["share_peak_in_first_30min"] - peak["share_peak_in_first_15min"], bottom=peak["share_peak_in_first_15min"], label="15-30", alpha=.8)
    ax[1].set(ylabel="cumulative share of trades", title="dev: share with peak in first 5/15/30 min"); ax[1].legend(fontsize=7)
    ax[1].tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "oracle_peak_time.png", dpi=130); plt.close(fig)

    keys = ["baseline_a", "fixed_09:35", "fixed_10:15", "top1_trailing", "exec_oracle_open", "gross_peak_open", "rw_gross_peak_open"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, k in enumerate(keys):
        vals = [res[k][mk].mean() if k in res else col(k)[mk].mean() for mk in groups.values()]
        vals = [np.nanmean(res[k][mk]) if k in res else np.nanmean(col(k)[mk]) for mk in groups.values()]
        ax.bar(np.arange(3) + i * .12, np.array(vals) * 100, .12, label=k)
    ax.set_xticks(np.arange(3) + .36, [f"{g} (n={int(mk.sum())})" for g, mk in groups.items()])
    ax.set(ylabel="mean return per trade, % (net unless 'gross')", title="dev: hindsight headroom vs realistic policies"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "oracle_headroom.png", dpi=130); plt.close(fig)


    pd.set_option("display.width", 250, "display.float_format", lambda x: f"{x:.4f}")
    for n, d in (("headroom", head), ("capture", cap), ("peak", peak), ("rw", rwt)):
        print(f"--- {n}\n{d.to_string(index=False)}")
    print(pd.read_csv(TABLES / "oracle_early_moves.csv").to_string(index=False))


if __name__ == "__main__":
    main()
