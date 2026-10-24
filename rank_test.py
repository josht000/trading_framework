"""
Trade-ranking experiments (dev set; the locked test set is read only with --final).

  python rank_test.py                # build the table, run the dev experiments, tune weights, write reports/ranking_report.md
  python rank_test.py --rebuild      # rebuild the feature table first
  python rank_test.py --final        # one look at the locked test period for the strategies chosen on dev

Account: $5,000, <= 5 overnight positions/day, 20% of equity each (25% for boosted trades). Candidates are ranked by `score`
(src/backtest/ranking.py); the day's top 5 are taken, optionally only if score >= a threshold. Hygiene filters drop OTC / pink-sheet
tickers and names with EDGAR distress or dilution filings (3.01 delisting notice, 1.03, 4.02, late-filing forms, offerings, 3.02).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from analyze_winners import prepare
from explore import FILTERS
from src.backtest import features_extra as fx
from src.backtest import ranking as rk
from src.backtest.pit_metrics import (
    prepare_account_data,
    select_rows_for_scores,
    simulate_account_selected_py,
    select_trades,
    simulate_account,
)
from src.backtest.numba_kernels import numba_enabled
from src.data.market_cap import merge_market_cap
from src.utils import timing

OUT = os.path.join("reports", "ranking")
TABLE = os.path.join("reports", "ranked_table.parquet")
RNG = np.random.default_rng(11)

STRATS = {  # existing top strategies: entry filters on the base pm_target25_v2 trades
    "base": ["all"],
    "S1": ["above_sma200", "iwm_day>=0", "spy_day>=0"],
    "H1": ["headroom>=20%"],
    "H3": ["headroom>=20%", "above_sma200"],
    "H2": ["headroom>=20%", "above_sma200", "iwm_day>=0", "spy_day>=0"],
    "base<=100M": ["shares<=100M"],
    "S1<=100M": ["above_sma200", "iwm_day>=0", "spy_day>=0", "shares<=100M"],
    "H3<=100M": ["headroom>=20%", "above_sma200", "shares<=100M"],
    "F30-H20": ["shares<30M", "headroom>=20%"],
    "F30-M200-H20": ["shares<30M", "mcap<200M", "headroom>=20%"],
    "M200": ["mcap<200M"],
    "M200-H20": ["mcap<200M", "headroom>=20%"],
}


def build_table(rebuild=False):
    if os.path.exists(TABLE) and not rebuild:
        with timing.stage("load ranked table"):
            T = pd.read_parquet(TABLE)
        with timing.stage("merge_market_cap"):
            return merge_market_cap(T)
    with timing.stage("prepare ranking inputs"):
        T = prepare()
    with timing.stage("load candidates"):
        cand = pd.read_parquet(os.path.join("data", "pit", "candidates.parquet"))
    for name, fn in (
        ("features_extra.add_edgar", fx.add_edgar),
        ("features_extra.add_news_since_close", fx.add_news_since_close),
        ("features_extra.add_movers", fx.add_movers),
        ("features_extra.add_theme_heat", lambda frame: fx.add_theme_heat(frame, cand)),
        ("features_extra.add_otc", fx.add_otc),
    ):
        with timing.stage(name):
            T = fn(T)
    with timing.stage("write ranked table"):
        T.to_parquet(TABLE, index=False)
    with timing.stage("merge_market_cap"):
        return merge_market_cap(T)


def hygiene(T, level):
    """level 0 none, 1 no OTC, 2 + no EDGAR distress, 3 + no recent offering/dilution."""
    m = pd.Series(True, index=T.index)
    if level >= 1:
        m &= ~T["is_otc"]
    if level >= 2:
        m &= (T["e_distress_30d"].fillna(0) == 0) & (T["e_delist_notice_90d"].fillna(0) == 0)
    if level >= 3:
        m &= (T["e_offering_10d"].fillna(0) == 0) & (T["e_item_302_10d"].fillna(0) == 0)
    return m


def apply_keys(T, keys):
    m = np.ones(len(T), bool)
    for k in keys:
        m &= FILTERS[k](T).fillna(False).to_numpy()
    return T[m]


def run(t, days, top_n=5, rank="rank_key", boost=None):
    if len(t) == 0:
        return None
    st, eq, dr = simulate_account(t, max_pos=top_n, rank=rank, boost=boost, all_days=days)
    return st


COLS = ["trades", "win%", "mean%", "PF", "total%", "maxDD%", "min$", "worst_day%", "loss_streak_days", "sharpe"]


def table(rows):
    df = pd.DataFrame(rows).set_index("setup")
    return df[[c for c in COLS if c in df.columns]]


def with_score(T, weights=None, scale="raw"):
    s = rk.score(T, weights, scale)
    return T.assign(score=s, rank_key=s + 1e-6 * T["rel_vol"].clip(upper=100).fillna(0) / 100.0)


def day_split(days, frac=0.6):
    return days[: int(len(days) * frac)], days[int(len(days) * frac):]


def md_table(df):
    from winners_analysis import md_table as mt
    return mt(df, floatfmt="{:.2f}")


def dev_experiments(T, md):
    dev = T[T["period"] == "dev"]
    days = pd.bdate_range(dev["day"].min(), dev["day"].max())
    res = {}

    # E0: hygiene filters on the base strategy (selection by relative volume as before)
    rows = []
    for lvl, lab in ((0, "no hygiene"), (1, "+ no OTC / pink sheets"), (2, "+ no EDGAR distress (3.01/1.03/4.02/NT)"), (3, "+ no recent offering / 3.02")):
        s = dev[hygiene(dev, lvl)]
        st = run(with_score(s), days, rank="rel_vol")
        rows.append({"setup": lab, **st})
    md.append("### Hygiene filters on the base strategy (rank by relative volume, as before)\n\n" + md_table(table(rows)) + "\n")
    res["hygiene"] = rows

    # E1: ranking vs relative volume, thresholds, top-N; hygiene level 3
    base = with_score(dev[hygiene(dev, 1)])
    rows = [{"setup": "rank by rel. volume, top 5", **run(base, days, rank="rel_vol")}]
    rows.append({"setup": "rank by score (default weights), top 5", **run(base, days)})
    for th in (0, 1, 2, 3):
        s = base[base["score"] >= th]
        st = run(s, days)
        if st:
            rows.append({"setup": f"score >= {th}, top 5", **st})
    for n in (3, 2):
        rows.append({"setup": f"score >= 1, top {n}", **run(base[base["score"] >= 1], days, top_n=n)})
    md.append("### Ranking with the default weights (base strategy, OTC excluded)\n\n" + md_table(table(rows)) + "\n")
    res["rank_default"] = rows

    # E2: the existing top strategies with ranking
    rows = []
    for name, keys in STRATS.items():
        s0 = apply_keys(base, keys)
        for lab, rank, th in (("rel_vol", "rel_vol", None), ("score", "rank_key", None), ("score>=1", "rank_key", 1), ("score>=2", "rank_key", 2)):
            s = s0 if th is None else s0[s0["score"] >= th]
            st = run(s, days, rank=rank)
            if st:
                rows.append({"setup": f"{name} / {lab}", **st})
    md.append("### Existing top strategies with ranking (OTC excluded; EDGAR distress/dilution flags are score components)\n\n" + md_table(table(rows)) + "\n")
    res["strats"] = rows

    # E3: component ablation with threshold 1
    rows = []
    w0 = dict(rk.DEFAULT_WEIGHTS)
    full = with_score(dev[hygiene(dev, 1)], w0)
    rows.append({"setup": "all components", **run(full[full["score"] >= 1], days)})
    for k in rk.COMPONENTS:
        w = dict(w0)
        w[k] = 0.0
        s = with_score(dev[hygiene(dev, 1)], w)
        st = run(s[s["score"] >= 1], days)
        rows.append({"setup": f"without {k}", **st})
    md.append("### Component ablation (score >= 1, top 5): what does each component contribute?\n\n" + md_table(table(rows)) + "\n")
    res["ablation"] = rows

    # E4: scales
    rows = []
    for scale in ("raw", "integer", "normalized"):
        s = with_score(dev[hygiene(dev, 1)], None, scale)
        th = {"raw": 1.0, "integer": 1.0, "normalized": 0.5 + 1.0 / (2 * sum(rk.DEFAULT_WEIGHTS.values()))}[scale]
        st = run(s[s["score"] >= th], days)
        rows.append({"setup": f"{scale} (threshold {th:.2f})", **st})
    md.append("### Score scalings (same weights): raw float, integer -5..5, normalized 0..1\n\n" + md_table(table(rows)) + "\n")

    # E5: boost sizing
    rows = []
    s = base[base["score"] >= 1]
    for lab, b in (("no boost", None), ("25% when score >= 3", lambda x: x["score"] >= 3), ("25% when score >= 4", lambda x: x["score"] >= 4)):
        st = run(s, days, boost=b)
        rows.append({"setup": lab, **st})
    md.append("### Position sizing by score (score >= 1, top 5): 20% base, 25% boost\n\n" + md_table(table(rows)) + "\n")
    return res


def _weight_search_py(dev, n_iter, rng):
    """Random search of score weights x threshold on `dev` (train = first 60% of its days, validation = rest). Returns the draws table."""
    days = pd.bdate_range(dev["day"].min(), dev["day"].max())
    d_tr, d_va = day_split(days)
    tr, va = dev[dev["day"].isin(d_tr)], dev[dev["day"].isin(d_va)]
    grid = {k: [0.0, 0.5, 1.0, 1.5, 2.0] for k in rk.COMPONENTS}
    grid["delist_notice"] = [1.0, 2.0, 3.0]
    grid["late_filing"] = [1.0, 1.5, 2.0]
    ths = [0.0, 1.0, 2.0, 3.0]
    comp_tr, comp_va = rk.components(tr), rk.components(va)
    rows = []
    for i in range(n_iter):
        w = {k: float(rng.choice(v)) for k, v in grid.items()}
        th = float(rng.choice(ths))
        s_tr = sum(w[k] * comp_tr[k] for k in rk.COMPONENTS)
        sel = tr[s_tr >= th].assign(rank_key=s_tr[s_tr >= th])
        if len(sel) < 150:
            continue
        a = run(sel, d_tr, rank="rank_key")
        s_va = sum(w[k] * comp_va[k] for k in rk.COMPONENTS)
        selv = va[s_va >= th].assign(rank_key=s_va[s_va >= th])
        b = run(selv, d_va, rank="rank_key") if len(selv) >= 40 else None
        rows.append({"weights": json.dumps(w), "threshold": th, "train_trades": a["trades"], "train_sharpe": a["sharpe"], "train_total%": a["total%"],
                     "train_maxDD%": a["maxDD%"], "val_trades": b["trades"] if b else 0, "val_sharpe": b["sharpe"] if b else np.nan,
                     "val_total%": b["total%"] if b else np.nan, "val_maxDD%": b["maxDD%"] if b else np.nan, "val_min$": b["min$"] if b else np.nan})
    return pd.DataFrame(rows)


def _weight_search_numba(dev, n_iter, rng):
    days = pd.bdate_range(dev["day"].min(), dev["day"].max())
    d_tr, d_va = day_split(days)
    tr, va = dev[dev["day"].isin(d_tr)], dev[dev["day"].isin(d_va)]
    grid = {k: [0.0, 0.5, 1.0, 1.5, 2.0] for k in rk.COMPONENTS}
    grid["delist_notice"] = [1.0, 2.0, 3.0]
    grid["late_filing"] = [1.0, 1.5, 2.0]
    ths = [0.0, 1.0, 2.0, 3.0]
    comp_tr, comp_va = rk.components(tr), rk.components(va)
    tr_data = prepare_account_data(tr, d_tr)
    va_data = prepare_account_data(va, d_va)
    rows = []
    for i in range(n_iter):
        w = {k: float(rng.choice(v)) for k, v in grid.items()}
        th = float(rng.choice(ths))
        s_tr = np.zeros(len(tr), dtype=float)
        s_va = np.zeros(len(va), dtype=float)
        for k in rk.COMPONENTS:
            s_tr += w[k] * comp_tr[k].to_numpy(dtype=float)
            s_va += w[k] * comp_va[k].to_numpy(dtype=float)
        train_count = int((s_tr >= th).sum())
        if train_count < 150:
            continue
        a = simulate_account_selected_py(select_rows_for_scores(tr_data, s_tr, th), d_tr)
        val_count = int((s_va >= th).sum())
        b = simulate_account_selected_py(select_rows_for_scores(va_data, s_va, th), d_va) if val_count >= 40 else None
        rows.append({
            "weights": json.dumps(w), "threshold": th,
            "train_trades": a["trades"], "train_sharpe": a["sharpe"], "train_total%": a["total%"],
            "train_maxDD%": a["maxDD%"],
            "val_trades": b["trades"] if b else 0, "val_sharpe": b["sharpe"] if b else np.nan,
            "val_total%": b["total%"] if b else np.nan, "val_maxDD%": b["maxDD%"] if b else np.nan,
            "val_min$": b["min$"] if b else np.nan,
        })
    return pd.DataFrame(rows)


def weight_search(dev, n_iter, rng):
    if numba_enabled():
        return _weight_search_numba(dev, n_iter, rng)
    return _weight_search_py(dev, n_iter, rng)


def pick_weights(df):
    """Selection rule: best validation Sharpe among the top 15 draws by train Sharpe."""
    cand = df.sort_values("train_sharpe", ascending=False).head(15).sort_values("val_sharpe", ascending=False).iloc[0]
    return json.loads(cand["weights"]), float(cand["threshold"])


def nested_weight_bias(dev, filt=None, n_iter=700, cache=None):
    """Walk-forward bias (partition.walk_forward_bias) of the weight-tuning procedure; filt(x) optionally restricts the evaluated strategy (e.g. H3)."""
    from src.backtest.partition import walk_forward_bias
    cache = {} if cache is None else cache

    def tune(tr):
        key = (len(tr), str(tr["day"].max()))
        if key not in cache:
            df = weight_search(tr, n_iter, np.random.default_rng(5000 + len(tr)))
            cache[key] = pick_weights(df) if len(df) and df["val_sharpe"].notna().any() else None
        return cache[key]

    def ev(x, params, days):
        w, th = params
        sel = with_score(x, w)
        sel = sel[sel["score"] >= th]
        sel = filt(sel) if filt is not None else sel
        if len(sel) == 0:
            return 0.0
        return float(np.log1p(simulate_account(sel, max_pos=5, rank="rank_key", all_days=days)[2]).sum())

    return walk_forward_bias(dev, tune, ev, folds=5, embargo_days=2, n_days_period=len(pd.bdate_range(dev["day"].min(), dev["day"].max())))


def tune_weights(T, md, n_iter=700):
    dev = T[(T["period"] == "dev")]
    dev = dev[hygiene(dev, 1)]
    days = pd.bdate_range(dev["day"].min(), dev["day"].max())
    df = weight_search(dev, n_iter, RNG)
    df.to_csv(os.path.join(OUT, "weight_search.csv"), index=False)
    from src.backtest.partition import log_trials
    log_trials("ranking", n_iter, np.log1p(df["train_total%"] / 100), days=int(0.6 * len(days)), source="rank_test.tune_weights",
               notes=f"weight x threshold random search, {len(df)} valid draws; per-trial = train-segment total log return")
    top = df.sort_values("train_sharpe", ascending=False).head(15)
    corr = df[["train_sharpe", "val_sharpe"]].dropna().corr(method="spearman").iloc[0, 1]
    md.append(f"### Weight tuning (random search, {len(df)} valid draws of weights x threshold; train = first 60% of dev days, validation = last 40%)\n\n"
              f"Rank correlation between train and validation Sharpe across all draws: {corr:+.2f} "
              "(positive = tuning transfers, near zero = tuning is mostly noise). Top 15 draws by train Sharpe:\n\n" +
              md_table(top.drop(columns="weights").assign(weights=top["weights"].apply(lambda s: " ".join(f"{k[:4]}={v:g}" for k, v in json.loads(s).items()))).set_index("weights")) + "\n")
    cand = top.sort_values("val_sharpe", ascending=False).iloc[0]
    best_w, best_th = json.loads(cand["weights"]), float(cand["threshold"])
    json.dump({"weights": best_w, "threshold": best_th, "note": "picked: best validation Sharpe among top-15 train draws"},
              open(os.path.join("results", "ranking_weights.json"), "w"), indent=1)
    md.append(f"Selected (best validation Sharpe among the top 15 by train): weights {best_w}, threshold {best_th:g}; "
              f"validation Sharpe {cand['val_sharpe']:.2f}, total {cand['val_total%']:.0f}%, max DD {cand['val_maxDD%']:.0f}%, {int(cand['val_trades'])} trades.\n")
    nb = nested_weight_bias(dev, None, n_iter)
    md.append(f"Walk-forward (nested) bias of this tuning procedure: {100 * nb['bias']:+.1f} log-return points over {len(days)} dev days "
              f"({nb['folds']} folds; in-sample {100 * nb['is_per_day']:.3f}%/day vs out-of-sample {100 * nb['oos_per_day']:.3f}%/day).\n")
    return best_w, best_th, corr


def walk_forward_ridge(dev, alpha=50.0, clip=0.30, min_months=6, embargo_bdays=5):
    """Learn component weights month by month (ridge on winsorized returns, training rows end before the test month): no look-ahead."""
    from sklearn.linear_model import Ridge
    comp = rk.components(dev)
    y = dev["ret"].clip(-clip, clip)
    pred = pd.Series(np.nan, index=dev.index)
    for m in sorted(dev["month"].unique())[min_months:]:
        tr = dev["day"] < (m - pd.offsets.BDay(embargo_bdays))
        te = dev["month"] == m
        if tr.sum() < 200 or te.sum() == 0:
            continue
        pred[te] = Ridge(alpha=alpha).fit(comp[tr], y[tr]).predict(comp[te])
    final = Ridge(alpha=alpha).fit(comp, y)
    return pred, final


def ridge_experiment(T, md):
    dev = T[(T["period"] == "dev")]
    dev = dev[hygiene(dev, 1)].copy()
    pred, final = walk_forward_ridge(dev)
    ok = pred.notna()
    oos = dev[ok].copy()
    days = pd.bdate_range(oos["day"].min(), oos["day"].max())
    oos["wf"] = pred[ok]
    oos["default"] = rk.score(oos)
    oos["rank_wf"] = oos["wf"]
    oos["rank_def"] = oos["default"] + 1e-6 * oos["rel_vol"].clip(upper=100).fillna(0) / 100
    rows = [{"setup": "rel. volume rank, top 5 (same months)", **run(oos, days, rank="rel_vol")},
            {"setup": "default score >= 2", **run(oos[oos["default"] >= 2], days, rank="rank_def")},
            {"setup": "default score >= 3", **run(oos[oos["default"] >= 3], days, rank="rank_def")}]
    for th in (0.0, 0.005, 0.01, 0.02):
        st = run(oos[oos["wf"] >= th], days, rank="rank_wf")
        if st:
            rows.append({"setup": f"walk-forward ridge, predicted return >= {100 * th:.1f}%", **st})
    coef = pd.Series(final.coef_, index=rk.COMPONENTS)
    nl = chr(10)
    md.append("### Learned weights: monthly walk-forward ridge on the score components (out-of-sample months only)" + nl + nl +
              md_table(table(rows)) + nl + nl +
              "Ridge coefficients on all dev trades (expected trade return per unit of each component, winsorized at 30%): " +
              ", ".join(f"{k} {100 * v:+.2f}%" for k, v in coef.items()) + nl)
    return final, rows


def final_dev_table(T, best_w, best_th, md):
    dev = T[T["period"] == "dev"]
    days = pd.bdate_range(dev["day"].min(), dev["day"].max())
    rows = []
    cases = {}
    d3 = lambda x: x[hygiene(x, 1)]
    for name, keys in STRATS.items():
        cases[f"{name} (as before: rel. volume rank, no hygiene)"] = (lambda x, k=keys: apply_keys(x, k), "rel_vol", None, None)
    cases["base + hygiene, default-weight score >= 1"] = (lambda x: apply_keys(d3(x), ["all"]), "rank_key", None, 1.0)
    cases["H3 + hygiene, default-weight score >= 1"] = (lambda x: apply_keys(d3(x), STRATS["H3"]), "rank_key", None, 1.0)
    cases["base + hygiene, tuned score"] = (lambda x: apply_keys(d3(x), ["all"]), "rank_key", best_w, best_th)
    cases["H3 + hygiene, tuned score"] = (lambda x: apply_keys(d3(x), STRATS["H3"]), "rank_key", best_w, best_th)
    cases["S1 + hygiene, tuned score"] = (lambda x: apply_keys(d3(x), STRATS["S1"]), "rank_key", best_w, best_th)
    for lab, (f, rank, w, th) in cases.items():
        s = with_score(dev, w)
        s = f(s)
        if th is not None:
            s = s[s["score"] >= th]
        st = run(s, days, rank=rank)
        rows.append({"setup": lab, **st})
    md.append("### Dev summary: previous top strategies vs ranked versions\n\n" + md_table(table(rows)) + "\n")
    return rows, cases


def final_test(T, best_w, best_th, md):
    test = T[T["period"] == "test"]
    days = pd.bdate_range(test["day"].min(), test["day"].max())
    rows = []
    d3 = lambda x: x[hygiene(x, 1)]
    cases = {
        "base (rel. volume rank, no hygiene)": (lambda x: x, "rel_vol", None, None),
        "H3 (rel. volume rank, no hygiene)": (lambda x: apply_keys(x, STRATS["H3"]), "rel_vol", None, None),
        "base + hygiene, default score >= 1": (lambda x: d3(x), "rank_key", None, 1.0),
        "H3 + hygiene, default score >= 1": (lambda x: apply_keys(d3(x), STRATS["H3"]), "rank_key", None, 1.0),
        "base + hygiene, tuned score": (lambda x: d3(x), "rank_key", best_w, best_th),
        "H3 + hygiene, tuned score": (lambda x: apply_keys(d3(x), STRATS["H3"]), "rank_key", best_w, best_th),
    }
    for lab, (f, rank, w, th) in cases.items():
        s = f(with_score(test, w))
        if th is not None:
            s = s[s["score"] >= th]
        st = run(s, days, rank=rank)
        rows.append({"setup": lab, **(st or {"trades": 0})})
    md.append("## Locked test period (one look)\n\n" + md_table(table(rows)) + "\n")
    return rows


def run_experiments(T, iters=700, final=False):
    md = ["# Trade ranking experiments (dev set)\n"]
    wpath = os.path.join("results", "ranking_weights.json")
    if final:
        cfg = json.load(open(wpath))
        with timing.stage("final_test"):
            final_test(T, cfg["weights"], cfg["threshold"], md)
        with timing.stage("write report"):
            from src.framework import report
            report.section("Locked test report")
            report.table(pd.DataFrame([{"test trades": len(T[T["period"] == "test"]), "report": "console-only"}]))
        return
    md.append(f"{len(T[T['period'] == 'dev'])} dev trades (pm_target25_v2 exits, 13-month atr_pivot); score components: {', '.join(rk.COMPONENTS)}; "
              f"default weights {rk.DEFAULT_WEIGHTS}. OTC tickers flagged: {int(T['is_otc'].sum())} trades; trades with EDGAR data: {100 * T['e_has_data'].mean():.0f}%.\n")
    with timing.stage("dev_experiments"):
        dev_experiments(T, md)
    with timing.stage("tune_weights"):
        best_w, best_th, corr = tune_weights(T, md, iters)
    with timing.stage("ridge_experiment"):
        ridge_experiment(T, md)
    with timing.stage("final_dev_table"):
        final_dev_table(T, best_w, best_th, md)
    path = os.path.join("reports", "ranking_report.md")
    with timing.stage("write report"):
        open(path, "w", encoding="utf-8").write("\n".join(md))
        from src.framework import report
        report.section("Ranking report")
        report.table(pd.DataFrame([{"dev trades": len(T[T["period"] == "dev"]), "report": path}]))
        print("wrote", path)
    return T


def main():
    timing.install("rank_test")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--iters", type=int, default=700)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with timing.stage("build_table"):
        T = build_table(a.rebuild)
    run_experiments(T, a.iters, a.final)


if __name__ == "__main__":
    main()
