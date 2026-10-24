"""
Locked-test scoring, executed-trade exports, portfolio plots and trade charts for the ranked strategies.

  python rank_final.py --test     # one look at the locked test period
  python rank_final.py --charts   # export executed trades, portfolio plots, trade charts and a winning-trades report for R3

Strategies (all pm_target25_v2 exits, 13-month atr_pivot breakouts, OTC / pink-sheet tickers excluded):
  base      rank by relative volume, no filters
  R2        default-weight score >= 2, top 5 by score
  R3        default-weight score >= 3, top 5 by score
  RIDGE     walk-forward learned weights (ridge, fit on all dev), predicted return >= 0.5%
  S1, H3    the earlier filter strategies, rank by relative volume
  S1+R2     S1 filters and score >= 2
"""

import argparse
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from rank_test import STRATS, apply_keys, build_table, hygiene, md_table, run, table, walk_forward_ridge, with_score
from src.backtest import ranking as rk
from src.backtest.pit_metrics import select_trades, simulate_account
from src.utils import timing

START = 5000.0


def strategies(T):
    """name -> (function(table) -> selected candidate rows, rank column, ridge model or None)."""
    dev = T[T["period"] == "dev"]
    dev = dev[hygiene(dev, 1)]
    _, ridge = walk_forward_ridge(dev)

    def prep(x, ridge_model=None):
        x = with_score(x)
        if ridge_model is not None:
            x = x.assign(pred=ridge_model.predict(rk.components(x)))
        return x

    defs = {
        "base": (lambda x: x, "rel_vol"),
        "base+OTC": (lambda x: x[hygiene(x, 1)], "rel_vol"),
        "R2": (lambda x: (lambda y: y[y["score"] >= 2])(x[hygiene(x, 1)]), "rank_key"),
        "R3": (lambda x: (lambda y: y[y["score"] >= 3])(x[hygiene(x, 1)]), "rank_key"),
        "RIDGE": (lambda x: (lambda y: y[y["pred"] >= 0.005])(x[hygiene(x, 1)]), "pred"),
        "S1": (lambda x: apply_keys(x, STRATS["S1"]), "rel_vol"),
        "H3": (lambda x: apply_keys(x, STRATS["H3"]), "rel_vol"),
        "S1+R2": (lambda x: (lambda y: y[y["score"] >= 2])(apply_keys(x[hygiene(x, 1)], STRATS["S1"])), "rank_key"),
    }
    return defs, prep, ridge


def selected(T, name, period, defs, prep, ridge):
    x = prep(T[T["period"] == period], ridge)
    f, rank = defs[name]
    return f(x), rank


def test(T):
    defs, prep, ridge = strategies(T)
    rows = []
    for name in defs:
        s, rank = selected(T, name, "test", defs, prep, ridge)
        days = pd.bdate_range(T[T["period"] == "test"]["day"].min(), T[T["period"] == "test"]["day"].max())
        st = run(s, days, rank=rank)
        rows.append({"setup": name, **(st or {"trades": 0})})
    out = md_table(table(rows))
    print("## Locked test period (one look)\n\n" + out)
    return rows


def plot_curves(curves, path, test_start, title):
    fig, ax = plt.subplots(4, 1, figsize=(14, 13), gridspec_kw={"height_ratios": [3, 1.3, 1.5, 1.3], "hspace": 0.38})
    colors = {"R3": "#7b2cbf", "R2": "#b07cd8", "RIDGE": "#e0457b", "base+OTC": "#808080", "S1": "#26a69a", "H3": "#ff8c00"}
    for name, (eq, dr) in curves.items():
        if name in ("SPY", "IWM"):
            ax[0].plot(eq.index, eq.values, color={"SPY": "#1f6fe5", "IWM": "#d4a017"}[name], lw=1.1, alpha=0.8, label=f"{name} buy & hold")
        else:
            ax[0].plot(eq.index, eq.values, color=colors.get(name, "#444444"), lw=2.2 if name == "R3" else 1.4, label=name)
            e0 = pd.concat([pd.Series([START]), eq.reset_index(drop=True)])
            ax[1].plot(eq.index, 100 * (eq.values / e0.cummax().iloc[1:].values - 1), color=colors.get(name, "#444444"), lw=1.2, label=name)
    ax[0].axhline(START, color="#bbbbbb", ls="--", lw=0.8)
    ax[0].set_ylabel(r"equity (\$)")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    ax[0].set_title(title.replace("$", r"\$"), fontsize=10, loc="left")
    ax[1].set_ylabel("drawdown %")
    ax[1].legend(frameon=False, fontsize=8, loc="lower left")
    main = curves.get("R3") or next(iter(curves.values()))
    eq, dr = main
    pnl = eq - eq.shift(1).fillna(START)
    ax[2].bar(pnl.index, pnl.values, color=np.where(pnl.values >= 0, "#26a69a", "#ef5350"), width=1.0)
    ax[2].set_ylabel(r"R3 daily P&L (\$)")
    ax[3].plot(dr.index, 100 * (dr > 0).astype(float).rolling(60, min_periods=20).mean(), color="#7b2cbf")
    ax[3].set_ylabel("R3 rolling 60-day\n% up days")
    for a in ax:
        a.grid(color="#eeeeee")
        if test_start is not None:
            a.axvspan(test_start, eq.index[-1], color="#fff3c4", alpha=0.5, zorder=0)
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def charts(T, periods=("dev", "test")):
    defs, prep, ridge = strategies(T)
    T = T[T["period"].isin(periods)]
    all_days = pd.bdate_range(T["day"].min(), T["day"].max())
    test_start = T[T["period"] == "test"]["day"].min() if "test" in periods else None
    curves, stats = {}, {}
    for name in ("R3", "R2", "RIDGE", "base+OTC", "S1", "H3"):
        parts = []
        for period in periods:
            s, rank = selected(T, name, period, defs, prep, ridge)
            parts.append(s)
        s = pd.concat(parts)
        st, eq, dr = simulate_account(s, max_pos=5, rank=defs[name][1], all_days=all_days)
        curves[name] = (eq, dr)
        stats[name] = st
    for sym in ("SPY", "IWM"):
        d = pd.read_parquet(os.path.join("data", "pit", "market", f"{sym}_daily.parquet")).set_index("date")["close"]
        d = d.reindex(all_days.union(d.index)).ffill().reindex(all_days)
        curves[sym] = (START * d / d.iloc[0], None)
    os.makedirs(os.path.join("data", "charts", "portfolio"), exist_ok=True)
    plot_curves({k: v for k, v in curves.items()}, os.path.join("data", "charts", "portfolio", "ranked_R3_with_test.png" if "test" in periods else "ranked_R3_dev.png"), test_start,
                f"Ranked strategies, ${START:,.0f} start, " + ("dev + locked test (shaded): " if "test" in periods else "dev only: ") + ", ".join(f"{k} ${stats[k]['final$']:,.0f}/DD {stats[k]['maxDD%']:.0f}%" for k in ("R3", "base+OTC", "S1")))
    print(pd.DataFrame(stats).T[["trades", "win%", "mean%", "PF", "total%", "maxDD%", "min$", "sharpe"]].to_string(float_format=lambda x: f"{x:.2f}"))
    # executed trades for R3 -> trade charts (score shown in the title)
    parts = []
    for period in periods:
        s, rank = selected(T, "R3", period, defs, prep, ridge)
        parts.append(s)
    sel = select_trades(pd.concat(parts), 5, rank).drop(columns=["day"], errors="ignore")
    path = os.path.join("results", "ranked_R3.csv")
    sel.to_csv(path, index=False)
    out = os.path.join("data", "charts", "trades", "R3")
    subprocess.run([sys.executable, "chart_trades.py", "--csv", path, "--out", out], check=False, capture_output=True)
    print("charts ->", out)
    return defs, prep, ridge


def winners_report(T, defs, prep, ridge, periods=("dev", "test")):
    import analyze_winners as aw
    parts = []
    for period in periods:
        s, rank = selected(T, "R3", period, defs, prep, ridge)
        parts.append(select_trades(s, 5, rank))
    R = pd.concat(parts, ignore_index=True)
    R["winner"], R["big"], R["monster"] = R["ret"] > 0, R["ret"] >= 0.10, R["ret"] >= 0.25
    aw.build_report(R, out_dir=os.path.join("reports", "ranked_R3"), title="Winning-trade analysis: ranked strategy R3 (score >= 3, OTC excluded)")


def summary_cases():
    """label -> (filter fn, rank column, tuned weights or None, score threshold or None) of the summary tables."""
    import json
    cfg = json.load(open(os.path.join("results", "ranking_weights.json")))
    w, th = cfg["weights"], cfg["threshold"]
    d1 = lambda x: x[hygiene(x, 1)]
    cases = {}
    for name in ("base", "S1", "H1", "H3", "H2", "base<=100M", "S1<=100M", "H3<=100M", "F30-H20", "F30-M200-H20", "M200", "M200-H20"):
        cases[name + " (rel. volume rank, no hygiene)"] = (lambda x, k=STRATS[name]: apply_keys(x, k), "rel_vol", None, None)
    cases["base + hygiene, default-weight score >= 1"] = (lambda x: d1(x), "rank_key", None, 1.0)
    cases["H3 + hygiene, default-weight score >= 1"] = (lambda x: apply_keys(d1(x), STRATS["H3"]), "rank_key", None, 1.0)
    cases["base + hygiene, tuned score"] = (lambda x: d1(x), "rank_key", w, th)
    cases["H3 + hygiene, tuned score"] = (lambda x: apply_keys(d1(x), STRATS["H3"]), "rank_key", w, th)
    cases["S1 + hygiene, tuned score"] = (lambda x: apply_keys(d1(x), STRATS["S1"]), "rank_key", w, th)
    return cases


def case_areas(lab):
    """Search areas behind a configuration (partition bias B) and the area covered by the nested procedure (bias A)."""
    areas = ["signal", "exit"]
    if any(k in lab for k in ("S1", "H1", "H2", "H3", "F30", "100M", "M200")):
        areas.append("entry_filters")
    if "score" in lab:
        areas.append("ranking")
    return areas, (("ranking",) if "tuned" in lab else ())


def case_selection(T, lab, period, cases, return_pools=False):
    f, rank, wt, thr = cases[lab]
    x = T[T["period"] == period]
    days = pd.bdate_range(x["day"].min(), x["day"].max())
    sel = f(with_score(x, wt))
    if thr is not None:
        sel = sel[sel["score"] >= thr]
    if return_pools:
        universe = x[hygiene(x, 1)]
        return sel, days, rank, universe
    return sel, days, rank


def nested_bias_A(T, lab, cases, cache_path=os.path.join("results", "partition_bias_A.json")):
    """Walk-forward bias A of the weight-tuning procedure for a tuned-score case (dev only; cached on disk, 4 nested random searches)."""
    import json
    from rank_test import nested_weight_bias
    key = f"{lab}|{int((T['period'] == 'dev').sum())}"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    if key not in cache:
        dev = T[T["period"] == "dev"]
        dev = dev[hygiene(dev, 1)]
        filt = None if lab.startswith("base") else (lambda x, k=STRATS[lab.split(" ")[0]]: apply_keys(x, k))
        r = nested_weight_bias(dev, filt, 700, _TUNE_CACHE)
        cache[key] = {"bias": r["bias"], "folds": r["folds"], "is_per_day": r.get("is_per_day"), "oos_per_day": r.get("oos_per_day")}
        json.dump(cache, open(cache_path, "w"), indent=1)
    return cache[key]


_TUNE_CACHE = {}


def partition_cases(T, labs=None, cases=None):
    """(label, period) -> partition result for the summary cases; dev uses the bias rule, test is out-of-sample (bias 0)."""
    from src.backtest import partition as P
    cases = cases or summary_cases()
    out = {}
    for lab in labs or cases:
        areas, a_areas = case_areas(lab)
        A = nested_bias_A(T, lab, cases)["bias"] if a_areas else None
        for period in ("dev", "test"):
            sel, days, rank, universe = case_selection(T, lab, period, cases, return_pools=True)
            out[(lab, period)] = P.run_partition(sel, days, rank, pool=sel, universe=universe,
                                                  areas=areas, A=A, A_areas=a_areas, oos=period == "test")
    return out


def summary_table(T):
    """The dev-summary rows on dev and on the locked test period; rewrites that section of reports/ranking_report.md."""
    from src.backtest import partition as P
    from src.backtest import kfold as K
    import re
    cases = summary_cases()
    parts = partition_cases(T, cases=cases)
    rows = []
    krows = {}
    for lab, (f, rank, wt, thr) in cases.items():
        for period in ("dev", "test"):
            x = T[T["period"] == period]
            days = pd.bdate_range(x["day"].min(), x["day"].max())
            sel = f(with_score(x, wt))
            if thr is not None:
                sel = sel[sel["score"] >= thr]
            st = run(sel, days, rank=rank)
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", lab)
            kp = os.path.join("reports", "partition", f"kfold_{safe}_{period}_summary.csv")
            if os.path.exists(kp):
                ks = pd.read_csv(kp).iloc[0].to_dict()
            else:
                kr, ks = K.kfold_stability(sel, days, rank=rank, pool=sel, label=f"{lab}|{period}",
                                            tuned="tuned" in lab)
                K.write_kfold(kr, ks, "reports/partition", safe, period)
            krows[(lab, period)] = ks
            rows.append({"setup": lab, "period": period, **(st or {"trades": 0}), **P.short_cols(parts[(lab, period)]),
                         "folds_profit": ks.get("folds_profit"), "folds_beat_random": ks.get("folds_beat_random"),
                         "fold_t": ks.get("fold_t"), "best_fold_share": ks.get("best_fold_share"), "kfold_p": ks.get("kfold_p")})
    df = pd.DataFrame(rows).set_index("setup")
    for c in ("total_log", "trend_iwm", "trend_spy", "random_pick", "random_pick_universe", "random_pick_pool",
              "skill", "skill_vs_universe", "skill_vs_pool", "bias", "bias_signed", "bias_A", "bias_B", "bias_sd", "skill_pct"):
        df[c] = 100 * df[c]
    cols = ["period", "trades", "win%", "mean%", "PF", "total%", "maxDD%", "min$", "worst_day%", "loss_streak_days", "sharpe",
            "total_log", "trend_iwm", "trend_spy", "random_pick", "random_pick_universe", "random_pick_pool",
            "random_p_universe", "random_p_pool", "skill", "skill_vs_universe", "skill_vs_pool",
            "bias", "bias_signed", "bias_A", "bias_B", "bias_sd", "bias_N_eff", "bias_unreliable", "bias_src", "skill_pct", "skill_p",
            "days_no_choice_pool", "days_no_choice_universe",
            "folds_profit", "folds_beat_random", "fold_t", "best_fold_share", "kfold_p"]
    nl = chr(10)
    cols = [c for c in cols if c in df.columns and c != "period"]
    section = ("### Summary: previous top strategies vs ranked versions" + nl + nl +
               "Tuned weights come from the dev-only random search (results/ranking_weights.json); the test rows are out-of-sample." + nl + nl +
               "Partition test (src/backtest/partition.py; log-return percent points): total_log = trend_iwm + skill + bias, where skill is a residual. "
               "random_pick_universe draws from the hard-screen/hygiene universe; random_pick_pool draws from strategy-filtered rows before ranking. "
               "Both use 2,000 seeded draws and report no-choice days and p-values. Bias A is nested walk-forward; bias B is one pooled comparable-trial haircut with N_eff and sd. "
               "Bias is clamped for subtraction and flagged unreliable when larger than total; test bias is 0 by construction." + nl + nl +
               "Skill is not a standalone causal estimate: its meaning depends on the selected trend and random baselines, and it can retain luck from a few large trades." + nl + nl +
               "K-fold stability: five contiguous chronological trading-day blocks; folds_profit and folds_beat_random are counts out of 5, fold_t is the t-statistic of fold log returns, best_fold_share is the single-best-fold concentration, and kfold_p is the exact Poisson-binomial luck tail from 2,000 random-pick draws per fold." + nl + nl +
               "#### Dev" + nl + nl + md_table(df[df["period"] == "dev"][cols]) + nl + nl +
               "#### Locked test (last 4 months)" + nl + nl + md_table(df[df["period"] == "test"][cols]) + nl)
    path = os.path.join("reports", "ranking_report.md")
    txt = open(path, encoding="utf-8").read()
    i = txt.find("### Dev summary")
    if i < 0:
        i = txt.find("### Summary: previous top strategies")
    open(path, "w", encoding="utf-8").write((txt[:i] if i >= 0 else txt + nl) + section)
    print(section)


def main():
    timing.install("rank_final")
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--charts", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    with timing.stage("build_table"):
        T = build_table()
    if a.test:
        with timing.stage("test"):
            test(T)
    if a.summary:
        with timing.stage("summary"):
            summary_table(T)
    if a.charts:
        with timing.stage("charts"):
            defs, prep, ridge = charts(T)
        with timing.stage("winners_report"):
            winners_report(T, defs, prep, ridge)


if __name__ == "__main__":
    main()
