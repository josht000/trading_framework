"""Continuous-target XGBoost study on the base trades (pm_target25_v2, lb13): regression (Huber), quantile, position potential.

  python scripts/ml_return_targets.py            # dev only: purged walk-forward OOF metrics (no locked test touched)
  python scripts/ml_return_targets.py --test     # also fit on all dev once, score the locked test period, policy sims + partition tests

Targets come from results/exit/windows.npz (minute bars 15:55 -> next day 11:00), relative to the entry fill price `entry`:
  ret_1000/1030/1100 = last RTH close <= cutoff on the next session / entry - 1;  gap = next RTH open / entry - 1;
  open_to_1030 = close_1030 / next open - 1;  long_pot = max high through 10:30 / entry - 1 and short_pot = 1 - min low / entry
  (HINDSIGHT potentials, not tradeable P&L);  absret_1030 = |ret_1030|.   Realized, tradeable P&L is the strategy `ret` column.
Targets never enter the feature set.  Models: target / atr_pct, clipped at the TRAIN-fold 1st/99th percentiles.
Writes reports/ml_targets/{report.md,*.csv} and results/ml_targets_{dev_oof,test_pred}.parquet.
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from src.backtest.exit_sim import load_windows
from src.backtest.pit_metrics import simulate_account
from winners_models import FEATURES as GB_FEATURES, walk_forward as gb_walk_forward

OUT = Path("reports") / "ml_targets"
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
RANK_COLS = ["atr_pct", "sig_day_ret", "rel_vol", "dvol20", "last60", "from_high", "range_pct", "headroom"]
EXTRA = ["float_now", "short_pct_float_now", "earn_next_0_3d"]  # snapshot / publish-date features: look-ahead risk, opt-in only
LEAK = {"ret", "exit", "exit_time", "exit_reason", "open_ret", "pm_last", "ah45_min", "ah45_last", "spy_gap_next", "iwm_gap_next",
        "qqq_gap_next", "spy_cc_next", "iwm_cc_next", "qqq_cc_next", "winner", "big", "monster", "post_n", "post_specific", "post_any_news"}
HUBER_TARGETS = ["ret_1000", "ret_1030", "ret_1100", "gap", "open_to_1030", "long_pot", "short_pot", "absret_1030"]
BASE = dict(tree_method="hist", n_estimators=2000, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.8,
            colsample_bytree=0.7, reg_lambda=5.0, n_jobs=-1, random_state=0)


def window_targets(w):
    raw = np.asarray(w.times, dtype=np.int64)
    if raw.size and np.abs(raw).max() < 10**17:
        raw = raw * 1000
    ts = pd.to_datetime(raw, unit="ns", utc=True).tz_convert("America/New_York")
    ref, d0 = float(w.entry_price), ts[0].date()
    dates = np.array([x.date() for x in ts])
    tm = np.array([x.hour * 60 + x.minute for x in ts])
    nxt = dates > d0
    if not nxt.any() or not np.isfinite(ref) or ref <= 0:
        return {}
    day1 = dates[nxt][0]
    on = dates == day1
    rth = on & (tm >= 570)
    out = {}
    if rth.any():
        out["gap"] = w.opens[rth][0] / ref - 1
    for name, cut in (("ret_1000", 600), ("ret_1030", 630), ("ret_1100", 660)):
        m = rth & (tm <= cut)
        if m.any():
            out[name] = w.closes[m][-1] / ref - 1
    if "ret_1030" in out:
        out["absret_1030"] = abs(out["ret_1030"])
        if "gap" in out:
            out["open_to_1030"] = (out["ret_1030"] + 1) / (out["gap"] + 1) - 1
    pm = (np.arange(len(ts)) > 0) & ((dates < day1) | (on & (tm <= 630)))
    if pm.any() and (w.lows[pm] > 0).all():
        out["long_pot"] = w.highs[pm].max() / ref - 1
        out["short_pot"] = 1 - w.lows[pm].min() / ref
    return out


def load_table():
    T = pd.read_parquet("reports/trades_enriched.parquet").reset_index(drop=True)
    ws = {w.trade_id: w for w in load_windows("results/exit/windows.npz")}
    idx = pd.read_parquet("results/exit/windows_index.parquet")
    key = {(s, t): i for s, t, i in zip(idx["symbol"], pd.to_datetime(idx["entry_time"], utc=True), idx["trade_id"])}
    et = pd.to_datetime(T["entry_time"].astype(str), utc=True)
    tg = pd.DataFrame([window_targets(ws[key[(s, t)]]) if (s, t) in key else {} for s, t in zip(T["symbol"], et)])
    T = pd.concat([T, tg], axis=1)
    for c in RANK_COLS:
        T[f"{c}_rk"] = T.groupby("day")[c].rank(pct=True)
    T["n_day"] = T.groupby("day")["symbol"].transform("size")
    return T


def prep_y(y, atr, bounds=None):
    z = (y / atr.clip(lower=0.01)).replace([np.inf, -np.inf], np.nan)
    if bounds is None:
        bounds = tuple(z.dropna().quantile([0.01, 0.99]))
    return z.clip(*bounds), bounds


def split_val(tr):
    tr = tr.sort_values("day")
    cut = int(len(tr) * 0.85)
    val = tr.iloc[cut:]
    trn = tr[tr["day"] < val["day"].min() - pd.offsets.BDay(2)]
    return trn, val


def new_model(kind, n_est=None, early=True):
    p = dict(BASE)
    if n_est is not None:
        p["n_estimators"] = n_est
    elif early:
        p["early_stopping_rounds"] = 100
    if kind == "huber":
        return xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=1.0, **p)
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(QS), **p)


def fit_fold(tr, te, feats, specs):
    """Fit every spec on `tr` (early stopping on its last 15%), predict `te`; returns predictions and best-iteration counts."""
    trn, val = split_val(tr)
    pred, iters = {}, {}
    for name, (kind, tcol) in specs.items():
        ytr, b = prep_y(trn[tcol], trn["atr_pct"])
        yva, _ = prep_y(val[tcol], val["atr_pct"], b)
        a, v = ytr.notna(), yva.notna()
        m = new_model(kind).fit(trn.loc[a, feats], ytr[a], eval_set=[(val.loc[v, feats], yva[v])], verbose=False)
        p = m.predict(te[feats])
        iters[name] = m.best_iteration + 1
        if kind == "quantile":
            for i, q in enumerate(QS):
                pred[f"q{int(q * 100)}"] = p[:, i]
        else:
            pred[f"p_{tcol}"] = p
    return pred, iters


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m])[0]) if m.sum() > 10 else np.nan


def eval_pred(d, pred, ycol, label):
    p = np.asarray(pred, float)
    ok = np.isfinite(p) & d[ycol].notna().to_numpy() & d["ret"].notna().to_numpy()
    dd = d[ok].assign(_p=p[ok])
    out = {"model": label, "target": ycol, "n": len(dd), "IC": spearman(dd["_p"], dd[ycol]), "IC_vs_strategy_ret": spearman(dd["_p"], dd["ret"])}
    mic = pd.Series({m: spearman(g["_p"], g[ycol]) for m, g in dd.groupby("month") if len(g) > 20}).dropna()
    out["monthIC_mean"] = mic.mean()
    out["monthIC_pos%"] = 100 * (mic > 0).mean()
    out["monthIC_t"] = mic.mean() / (mic.std(ddof=1) / np.sqrt(len(mic))) if len(mic) > 2 else np.nan
    dic = pd.Series({k: spearman(g["_p"], g[ycol]) for k, g in dd.groupby("day") if len(g) >= 3}).dropna()
    out["dayIC_mean"], out["dayIC_days"] = dic.mean(), len(dic)
    dec = pd.qcut(dd["_p"].rank(method="first"), 10, labels=False)
    gm = dd.groupby(dec)
    y, r = gm[ycol].mean(), gm["ret"].mean()
    out.update({"top_dec_y%": 100 * y.iloc[-1], "bot_dec_y%": 100 * y.iloc[0], "top_dec_ret%": 100 * r.iloc[-1], "bot_dec_ret%": 100 * r.iloc[0],
                "top_minus_bot_ret%": 100 * (r.iloc[-1] - r.iloc[0])})
    return out


def policies(d):
    """name -> (score, minimum score).  Score columns are volatility-normalized model outputs (higher = better)."""
    pol = {"rel_vol (base rank)": (d["rel_vol"], None)}
    for name, col, thr in (("huber ret_1030 > 0", "p_ret_1030", 0.0), ("quantile q50 > 0", "q50", 0.0), ("long_pot rank", "p_long_pot", None),
                           ("GB big (existing)", "gb_big", None)):
        if col in d:
            pol[name] = (d[col], thr)
    if "q50" in d:
        pol["q50 > 0 and q10 > -0.5"] = (d["q50"].where(d["q10"] > -0.5), 0.0)
    return pol


def run_policies(d, days, tag):
    rows, sels = [], {}
    for name, (score, thr) in policies(d).items():
        s = d.assign(score=np.asarray(score, float))
        s = s[s["score"].notna() & ((s["score"] > thr) if thr is not None else True)]
        if s.empty:
            continue
        st, _, _ = simulate_account(s, max_pos=5, rank="score", all_days=days)
        rows.append({"policy": name, "period": tag, "candidates": len(s), **st})
        sels[name] = s
    return pd.DataFrame(rows), sels


def md(df, fmt=".3f"):
    return df.to_markdown(floatfmt=fmt) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="also score the locked test period once (fit on all dev)")
    ap.add_argument("--extra-features", action="store_true", help="add snapshot/publish-date features (look-ahead risk)")
    ap.add_argument("--min-months", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    T = load_table()
    feats = list(GB_FEATURES) + [f"{c}_rk" for c in RANK_COLS] + ["n_day"] + (EXTRA if a.extra_features else [])
    assert not LEAK & set(feats), LEAK & set(feats)
    T[feats] = T[feats].replace([np.inf, -np.inf], np.nan)
    dev = T[T["period"] == "dev"].sort_values("day").reset_index(drop=True)
    test = T[T["period"] == "test"].reset_index(drop=True)
    cov = T[HUBER_TARGETS].notna().mean().round(3)
    print("target coverage:", cov.to_dict(), "| dev", len(dev), "test", len(test), "| features", len(feats))

    specs = {t: ("huber", t) for t in HUBER_TARGETS}
    specs["quantile_ret_1030"] = ("quantile", "ret_1030")

    # ---- dev: purged expanding monthly walk-forward (same fold definition as winners_models.walk_forward) ----
    parts, iters = [], {k: [] for k in specs}
    for m in sorted(dev["month"].unique())[a.min_months:]:
        te = dev["month"] == m
        tr = dev["day"] < (m - pd.offsets.BDay(5))
        if tr.sum() < 300 or te.sum() == 0:
            continue
        pred, it = fit_fold(dev[tr], dev[te], feats, specs)
        parts.append(pd.DataFrame(pred, index=dev.index[te]))
        for k, v in it.items():
            iters[k].append(v)
        print("fold", m.date(), "train", int(tr.sum()), "test", int(te.sum()), flush=True)
    oof = dev.join(pd.concat(parts))
    for t in ("big", "winner"):  # existing classifiers on the identical folds
        oof[f"gb_{t}"], _ = gb_walk_forward(dev, GB_FEATURES, t, min_months=a.min_months)
    oof = oof[oof["p_ret_1030"].notna()].copy()
    oof.to_parquet("results/ml_targets_dev_oof.parquet", index=False)

    def metric_table(d, label):
        rows = [eval_pred(d, d[f"p_{t}"], t, f"huber {t}") for t in HUBER_TARGETS]
        rows += [eval_pred(d, d[f"p_{t}"] * d["atr_pct"], t, f"huber {t} (x atr)") for t in ("ret_1030", "gap", "open_to_1030")]
        rows += [eval_pred(d, d["q50"], "ret_1030", "quantile q50"), eval_pred(d, d["q90"] - d["q10"], "absret_1030", "quantile spread q90-q10"),
                 eval_pred(d, (d["q90"] - d["q50"]) - (d["q50"] - d["q10"]), "ret_1030", "quantile skew"),
                 eval_pred(d, d["p_long_pot"] - d["p_short_pot"], "ret_1030", "edge = p_long - p_short")]
        for t in ("big", "winner"):
            if f"gb_{t}" in d:
                rows += [eval_pred(d, d[f"gb_{t}"], "ret_1030", f"GB {t} (existing)"), eval_pred(d, d[f"gb_{t}"], "absret_1030", f"GB {t} (existing)")]
        for c in ("rel_vol", "atr_pct", "headroom"):
            rows += [eval_pred(d, d[c], "ret_1030", f"naive {c}"), eval_pred(d, d[c], "absret_1030", f"naive {c}")]
        df = pd.DataFrame(rows).set_index("model")
        df.to_csv(OUT / f"metrics_{label}.csv")
        return df

    def calibration(d):
        y = d["ret_1030"] / d["atr_pct"].clip(lower=0.01)
        return pd.DataFrame({"nominal": [f"q{int(q * 100)}" for q in QS], "share_below": [float((y < d[f"q{int(q * 100)}"]).mean()) for q in QS]}).set_index("nominal")

    days_dev = pd.bdate_range(oof["day"].min(), oof["day"].max())
    mt = metric_table(oof, "dev")
    cal = calibration(oof)
    pol_dev, sel_dev = run_policies(oof, days_dev, "dev")
    pol_dev.to_csv(OUT / "policies_dev.csv", index=False)
    lines = [f"# Continuous-target XGBoost study (dev, purged walk-forward)\n",
             f"{len(oof)} out-of-fold dev trades ({oof['day'].min().date()} to {oof['day'].max().date()}), {len(feats)} features known at 15:55. "
             "Targets relative to the entry fill, from the 15:55 -> 11:00 minute windows; models fit on target/atr_pct clipped at train 1/99 pct. "
             "Fixed hyperparameters (no tuning). IC = Spearman against the raw target; `IC_vs_strategy_ret` is against the realized strategy return.\n",
             "## Predictive metrics\n", md(mt), "## Quantile calibration (share of OOF targets below each predicted quantile)\n", md(cal),
             "## Best iteration per model (median across folds)\n", md(pd.Series({k: float(np.median(v)) for k, v in iters.items()}).to_frame("median_best_iter"), ".0f"),
             "## Policy simulation, dev out-of-fold (top 5 per day by score; NOT a complete report until --test adds partition tests)\n", md(pol_dev.set_index("policy"), ".2f")]

    if a.test:
        print("\n*** Scoring the LOCKED test period once ***", flush=True)
        tp = pd.DataFrame(index=test.index)
        for name, (kind, tcol) in specs.items():
            ytr, b = prep_y(dev[tcol], dev["atr_pct"])
            ok = ytr.notna()
            m = new_model(kind, n_est=int(np.median(iters[name]))).fit(dev.loc[ok, feats], ytr[ok], verbose=False)
            p = m.predict(test[feats])
            if kind == "quantile":
                for i, q in enumerate(QS):
                    tp[f"q{int(q * 100)}"] = p[:, i]
            else:
                tp[f"p_{tcol}"] = p
        for t in ("big", "winner"):
            from winners_models import _model
            tp[f"gb_{t}"] = _model().fit(dev[GB_FEATURES], dev[t]).predict_proba(test[GB_FEATURES])[:, 1]
        tst = test.join(tp)
        tst.to_parquet("results/ml_targets_test_pred.parquet", index=False)
        mt_t = metric_table(tst, "test")
        pol_t, sel_t = run_policies(tst, pd.bdate_range(tst["day"].min(), tst["day"].max()), "test")
        pol_t.to_csv(OUT / "policies_test.csv", index=False)
        lines += ["# Locked test period (model fit once on all dev, fixed iteration count)\n", md(mt_t), "## Quantile calibration, test\n", md(calibration(tst)),
                  "## Policy simulation, test\n", md(pol_t.set_index("policy"), ".2f")]

        from src.backtest import partition as P
        import partition_report as PR
        res = {}
        for per, sels, d, days in (("dev", sel_dev, oof, days_dev), ("test", sel_t, tst, pd.bdate_range(tst["day"].min(), tst["day"].max()))):
            for name, s in sels.items():
                label = "ML " + name.replace(" ", "_").replace(">", "gt").replace("(", "").replace(")", "")
                res[(label, per)] = P.run_partition(s, days, "score", pool=s, universe=d, areas=["signal", "exit", "ranking"], oos=per == "test")
        PR.write(res, tag="_ml")
        lines.append("\nPartition tests (trend, random-pick baselines, skill, bias, skill %) for every policy and period: reports/partition/partition_all_ml_dev.md and "
                     "partition_all_ml_test.md (figures in reports/partition/figures/). Dev bias counts a `ranking` search area; trial count is unknown (partial).\n")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "report.md").resolve())


if __name__ == "__main__":
    main()
