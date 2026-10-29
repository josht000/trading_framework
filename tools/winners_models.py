"""
Model-based sections of the winning-trade report: purged walk-forward gradient boosting with permutation importance and
remove-and-retrain checks, FDR-controlled rule mining, winner archetypes, event study, and the deflated Sharpe ratio.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import partial_dependence
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from winners_analysis import (BLUE, GOLD, GRAY, GREEN, PURPLE, RED, RNG, bh_fdr, group_stats, md_table, nw_tstat, profit_factor,
                              save_table, savefig)

FEATURES = ["sig_day_ret", "gap_open", "run5", "price", "dvol20", "rel_vol", "atr_pct", "dist_vwap", "from_high", "range_pct",
            "last60", "day_dvol", "dow", "dist_sma200", "dist_ema9", "dist_ema20", "headroom", "spy_day", "iwm_day", "qqq_day",
            "spy_ret5", "spy_ret20", "iwm_ret5", "iwm_ret20", "spy_sma50_up", "iwm_sma50_up", "iwm_sma200_up", "news_24h",
            "news_5d", "pre_specific_24h", "shares_pit", "mcap_pit"]


def _model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=2.0,
                                          min_samples_leaf=40, random_state=0)


def walk_forward(dev, feats, target, min_months=6, embargo_bdays=5):
    """Expanding-window monthly folds; training rows end `embargo_bdays` before the test month (purge)."""
    oos = pd.Series(np.nan, index=dev.index)
    models = {}
    months = sorted(dev["month"].unique())
    for m in months[min_months:]:
        te = dev["month"] == m
        tr = dev["day"] < (m - pd.offsets.BDay(embargo_bdays))
        if dev.loc[tr, target].nunique() < 2 or te.sum() == 0:
            continue
        mdl = _model().fit(dev.loc[tr, feats], dev.loc[tr, target])
        oos[te] = mdl.predict_proba(dev.loc[te, feats])[:, 1]
        models[m] = (mdl, te)
    return oos, models


def lift_top(y, score, frac=0.1):
    k = max(1, int(len(y) * frac))
    idx = np.argsort(-np.asarray(score))[:k]
    return float(np.mean(np.asarray(y)[idx]) / np.mean(y))


def sec_models(T, md):
    md.append("## 7. Can the winners be predicted? (gradient boosting, purged walk-forward)\n")
    dev = T[T["period"] == "dev"].sort_values("day").reset_index(drop=True)
    test = T[T["period"] == "test"].reset_index(drop=True)
    res = {}
    rows = []
    for target in ("big", "winner"):
        oos, models = walk_forward(dev, FEATURES, target)
        ok = oos.notna()
        y = dev.loc[ok, target].astype(int)
        auc = roc_auc_score(y, oos[ok])
        ap = average_precision_score(y, oos[ok])
        lf = lift_top(y, oos[ok])
        final = _model().fit(dev[FEATURES], dev[target])
        ps = final.predict_proba(test[FEATURES])[:, 1]
        yt = test[target].astype(int)
        res[target] = (oos, models, final)
        rows.append({"target": target, "dev_oos_AUC": auc, "dev_oos_PR_AUC": ap, "base_rate%": 100 * y.mean(),
                     "dev_oos_top10%_lift": lf, "test_AUC": roc_auc_score(yt, ps), "test_PR_AUC": average_precision_score(yt, ps),
                     "test_base_rate%": 100 * yt.mean(), "test_top20%_lift": lift_top(yt, ps, 0.2), "n_dev_oos": int(ok.sum()), "n_test": len(test)})
    tab = pd.DataFrame(rows).set_index("target")
    md.append("HistGradientBoosting (depth 3, strong regularization) on 32 point-in-time features known at 15:55; expanding monthly folds with a 5-business-day "
              "purge before each test month; final model trained on all dev and scored on the locked test period. AUC 0.5 = no skill.\n\n" +
              save_table(tab, "07_model_scores", floatfmt="{:.3f}") + "\n")

    # permutation importance on the walk-forward folds (AUC drop), target = big
    oos, models, final = res["big"]
    imp = {f: [] for f in FEATURES}
    for m, (mdl, te) in models.items():
        sub = dev.loc[te]
        if sub["big"].nunique() < 2:
            continue
        base = roc_auc_score(sub["big"], mdl.predict_proba(sub[FEATURES])[:, 1])
        for f in FEATURES:
            drops = []
            for _ in range(3):
                s2 = sub[FEATURES].copy()
                s2[f] = RNG.permutation(s2[f].to_numpy())
                drops.append(base - roc_auc_score(sub["big"], mdl.predict_proba(s2)[:, 1]))
            imp[f].append(np.mean(drops))
    pi = pd.DataFrame({"mean_AUC_drop": {f: np.mean(v) for f, v in imp.items()}, "sd_across_folds": {f: np.std(v) for f, v in imp.items()},
                       "folds_positive%": {f: 100 * np.mean(np.array(v) > 0) for f, v in imp.items()}}).sort_values("mean_AUC_drop", ascending=False)
    md.append("Permutation importance (drop in AUC when a feature is shuffled inside each walk-forward test fold; big-winner target):\n\n" +
              save_table(pi.head(15), "07_perm_importance", floatfmt="{:.4f}") + "\n")
    top3 = list(pi.index[:3])
    oos_wo, _ = walk_forward(dev, [f for f in FEATURES if f not in top3], "big")
    oos_only, _ = walk_forward(dev, top3, "big")
    ok = oos.notna() & oos_wo.notna() & oos_only.notna()
    y = dev.loc[ok, "big"].astype(int)
    md.append(f"**Remove-and-retrain check** (SHAP and permutation importance describe what a model fitted, not what generalizes): walk-forward AUC with all features "
              f"{roc_auc_score(y, oos[ok]):.3f}; without the top 3 ({', '.join(top3)}) {roc_auc_score(y, oos_wo[ok]):.3f}; with only those 3 {roc_auc_score(y, oos_only[ok]):.3f}.\n")

    # partial dependence for top features on the final model
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, f in zip(axes, pi.index[:4]):
        try:
            pdp = partial_dependence(final, dev[FEATURES], [f], grid_resolution=30)
            ax.plot(pdp["grid_values"][0], pdp["average"][0], color=PURPLE)
            ax.set_title(f, fontsize=9)
            ax.set_ylabel("P(big winner)")
            ax.grid(color="#eeeeee")
        except Exception:
            ax.axis("off")
    fig.suptitle("Partial dependence of the big-winner probability (final model, dev)", y=1.04)
    md.append(savefig(fig, "07_partial_dependence") + "\n")
    # one shallow tree for readable rules
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, random_state=0).fit(dev[FEATURES].fillna(-9), dev["big"])
    md.append("A depth-3 decision tree on dev (readable splits; unstable, shown for interpretation only):\n\n```\n" +
              export_text(tree, feature_names=FEATURES, decimals=3) + "```\n")
    return {"scores": tab, "importance": pi}


def _conditions(dev):
    cond = {}
    num = [f for f in FEATURES if dev[f].nunique() > 4]
    for f in num:
        x = dev[f].replace([np.inf, -np.inf], np.nan)
        q25, q50, q75 = x.quantile([0.25, 0.5, 0.75])
        cond[f"{f} <= {q25:.3g}"] = (x <= q25, f, "<=", q25)
        cond[f"{f} >= {q75:.3g}"] = (x >= q75, f, ">=", q75)
        cond[f"{f} >= {q50:.3g}"] = (x >= q50, f, ">=", q50)
        cond[f"{f} <= {q50:.3g}"] = (x <= q50, f, "<=", q50)
    for f in ("spy_sma50_up", "iwm_sma50_up", "iwm_sma200_up"):
        cond[f"{f} == 1"] = (dev[f] == 1, f, "==", 1)
        cond[f"{f} == 0"] = (dev[f] == 0, f, "==", 0)
    return cond


def sec_rules(T, md, min_support=60):
    md.append("## 8. Rule mining with multiple-testing control\n")
    dev = T[T["period"] == "dev"].reset_index(drop=True)
    test = T[T["period"] == "test"].reset_index(drop=True)
    cond = _conditions(dev)
    names = list(cond)
    masks = {n: cond[n][0].fillna(False).to_numpy() for n in names}
    base = dev["big"].mean()
    cand = []
    for i, a in enumerate(names):
        cand.append(((a,), masks[a]))
        for b in names[i + 1:]:
            if cond[a][1] == cond[b][1]:
                continue
            cand.append(((a, b), masks[a] & masks[b]))
    y = dev["big"].to_numpy()
    r = dev["ret"].to_numpy()
    rows = []
    for keys, m in cand:
        n = int(m.sum())
        if n < min_support:
            continue
        k = int(y[m].sum())
        p = stats.binomtest(k, n, base, alternative="greater").pvalue
        rows.append({"rule": " AND ".join(keys), "n": n, "big_rate%": 100 * k / n, "big_loss_rate%": 100 * np.mean(r[m] <= -0.10), "lift": (k / n) / base,
                     "mean_ret%": 100 * r[m].mean(), "PF": profit_factor(r[m]), "p": p})
    df = pd.DataFrame(rows)
    df["q_BH"] = bh_fdr(df["p"].to_numpy())
    md.append(f"{len(cand)} candidate rules (single and pairs of quartile/median conditions) with at least {min_support} dev trades: {len(df)} tested for a higher big-winner rate than the "
              f"{100 * base:.1f}% base (one-sided binomial). {int((df['q_BH'] < 0.10).sum())} pass BH-FDR q < 0.10 on dev.\n")
    top = df.sort_values("p").head(15).copy()
    # out-of-sample on the test period
    tm = {}
    test_loss = {}
    for _, row in top.iterrows():
        mt = np.ones(len(test), bool)
        for key in row["rule"].split(" AND "):
            col, op, val = cond[key][1], cond[key][2], cond[key][3]
            x = test[col]
            mt &= {"<=": x <= val, ">=": x >= val, "==": x == val}[op].fillna(False).to_numpy()
        n = int(mt.sum())
        tm[row["rule"]] = (n, 100 * test.loc[mt, "big"].mean() if n else np.nan, 100 * test.loc[mt, "ret"].mean() if n else np.nan)
        test_loss[row["rule"]] = 100 * (test.loc[mt, "ret"] <= -0.10).mean() if n else np.nan
    top["test_n"] = [tm[r][0] for r in top["rule"]]
    top["test_big_rate%"] = [tm[r][1] for r in top["rule"]]
    top["test_mean_ret%"] = [tm[r][2] for r in top["rule"]]
    top["test_big_loss%"] = [test_loss[r] for r in top["rule"]]
    md.append("Strongest rules by dev p-value, with their out-of-sample behaviour on the locked test period "
              f"(test base big-winner rate {100 * test['big'].mean():.1f}%):\n\n" + save_table(top.set_index("rule"), "08_rules", floatfmt="{:.3g}") + "\n")
    return {"rules": df}


def sec_archetypes(T, md):
    md.append("## 9. Winner archetypes (clustering)\n")
    W = T[T["big"]].copy()
    cols = ["sig_day_ret", "gap_open", "rel_vol", "atr_pct", "range_pct", "price", "dvol20", "run5", "headroom", "from_high"]
    X = W[cols].replace([np.inf, -np.inf], np.nan)
    X["headroom"] = X["headroom"].clip(upper=2)
    for c in ("rel_vol", "dvol20", "price", "range_pct", "sig_day_ret"):
        X[c] = np.log1p(X[c].clip(lower=0))
    X = X.fillna(X.median())
    Z = StandardScaler().fit_transform(X)
    best = (None, -1)
    for k in range(2, 7):
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
        s = silhouette_score(Z, lab)
        if s > best[1]:
            best = (k, s, lab)
    k, s, lab = best
    W["cluster"] = lab
    md.append(f"K-means on {len(W)} big winners (>= +10%) using {len(cols)} standardized features; k = {k} chosen by silhouette ({s:.2f}).\n")
    rows = []
    for c, g in W.groupby("cluster"):
        top_sec = g["sector"].value_counts().head(2)
        rows.append({"cluster": c, "n": len(g), "mean_ret%": 100 * g["ret"].mean(), "sig_day_ret%": 100 * g["sig_day_ret"].median(),
                     "gap_open%": 100 * g["gap_open"].median(), "rel_vol": g["rel_vol"].median(), "atr%": 100 * g["atr_pct"].median(),
                     "range%": 100 * g["range_pct"].median(), "price": g["price"].median(), "dvol20_$M": g["dvol20"].median() / 1e6,
                     "run5%": 100 * g["run5"].median(), "headroom": g["headroom"].clip(upper=2).median(),
                     "top_sectors": ", ".join(f"{a} ({b})" for a, b in top_sec.items()),
                     "examples": ", ".join(g.sort_values("ret", ascending=False)["symbol"].head(5))})
    tab = pd.DataFrame(rows).set_index("cluster")
    md.append(save_table(tab, "09_archetypes") + "\n")
    # how common is each archetype among ALL trades? assign via nearest centroid in the same space
    scaler = StandardScaler().fit(X)
    km = KMeans(k, n_init=10, random_state=0).fit(scaler.transform(X))
    Xa = T[cols].replace([np.inf, -np.inf], np.nan).copy()
    Xa["headroom"] = Xa["headroom"].clip(upper=2)
    for c in ("rel_vol", "dvol20", "price", "range_pct", "sig_day_ret"):
        Xa[c] = np.log1p(Xa[c].clip(lower=0))
    Xa = Xa.fillna(X.median())
    T = T.assign(arch=km.predict(scaler.transform(Xa)))
    gs = group_stats(T, "arch", min_n=10)
    md.append("Share of ALL trades falling in each archetype's region of feature space (nearest centroid) and how those trades fared:\n\n" + save_table(gs, "09_archetype_all") + "\n")
    return {"k": k}


def sec_event_study(T, md, daily):
    md.append("## 10. Event study: the path around entry\n")
    px = daily.set_index(["symbol", "date"])["close"]
    spy = pd.read_parquet(os.path.join("data", "pit", "market", "IWM_daily.parquet")).set_index("date")["close"]
    spy_idx = spy.index
    offs = list(range(-5, 4))
    rows = []
    for sym, day, ret, grp in zip(T["symbol"], T["day"], T["ret"], T["big"]):
        try:
            s = px.xs(sym, level="symbol")
        except KeyError:
            continue
        idx = s.index.searchsorted(day)
        if idx >= len(s) or s.index[idx] != day or idx < 5 or idx + 3 >= len(s):
            continue
        c0 = s.iloc[idx - 1]
        m0 = spy.get(s.index[idx - 1], np.nan)
        path = {}
        for o in offs:
            d = s.index[idx + o]
            path[o] = s.iloc[idx + o] / c0 - 1 - (spy.get(d, np.nan) / m0 - 1)
        rows.append({"ret": ret, "big": grp, **{f"d{o}": v for o, v in path.items()}})
    E = pd.DataFrame(rows)
    E["bucket"] = pd.cut(E["ret"], [-1, -0.05, 0, 0.10, 10], labels=["loser <-5%", "small loss", "winner 0-10%", "big winner >=10%"])
    tab = E.groupby("bucket", observed=True)[[f"d{o}" for o in offs]].apply(lambda g: g.clip(lower=g.quantile(0.01), upper=g.quantile(0.99), axis=1).mean()) * 100
    tab.insert(0, "n", E.groupby("bucket", observed=True).size())
    md.append("Mean cumulative return versus IWM (%, winsorized 1%) relative to the close before the entry day, by final trade outcome. Day 0 is the entry day "
              "(entry at 15:55, so day 0 shows the run-up the strategy buys into), day +1 the overnight session:\n\n" + save_table(tab, "10_event_study") + "\n")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for b, col in zip(tab.index, (RED, GRAY, BLUE, GREEN)):
        ax.plot(offs, tab.loc[b, [f"d{o}" for o in offs]].values, marker="o", color=col, label=f"{b} (n={int(tab.loc[b, 'n'])})")
    ax.axvline(0, color="#bbbbbb")
    ax.set_xlabel("trading days relative to entry day")
    ax.set_ylabel("abnormal return vs IWM (%)")
    ax.legend(frameon=False)
    ax.grid(color="#eeeeee")
    md.append(savefig(fig, "10_event_study") + "\n")
    return {}


def deflated_sharpe(sr, T_obs, skew, kurt, n_trials, var_trials=None):
    """Bailey & Lopez de Prado (2014). sr is the per-period (daily) Sharpe ratio."""
    gamma = 0.5772156649
    v = var_trials if var_trials is not None else (1 + 0.5 * sr ** 2 - skew * sr + (kurt - 3) / 4 * sr ** 2) / (T_obs - 1)
    sr0 = np.sqrt(v) * ((1 - gamma) * stats.norm.ppf(1 - 1 / n_trials) + gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return stats.norm.cdf((sr - sr0) * np.sqrt(T_obs - 1) / denom), sr0


def sec_rigor(T, md, strategies):
    md.append("## 11. Statistical rigor: is any of this skill?\n")
    rows = []
    for name, daily_ret in strategies.items():
        x = daily_ret.to_numpy()
        x = x[np.isfinite(x)]
        sr = x.mean() / x.std()
        sk, ku = stats.skew(x), stats.kurtosis(x, fisher=False)
        row = {"strategy": name, "days": len(x), "daily_SR": sr, "annual_SR": sr * np.sqrt(252), "skew": sk, "kurtosis": ku, "NW_t(mean)": nw_tstat(x)}
        for n in (1, 30, 100, 300):
            p, sr0 = deflated_sharpe(sr, len(x), sk, ku, max(n, 2)) if n > 1 else (stats.norm.cdf(sr * np.sqrt(len(x) - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)), 0)
            row[f"P(skill) N={n}"] = p
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("strategy")
    md.append("Probabilistic / deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) on dev daily portfolio returns. 'N' is the number of strategy variants tried; "
              "N=1 is the plain probabilistic Sharpe ratio, larger N deflates for selection bias. This project has tried on the order of 100 variants "
              "(exit rules, filters, sizing).\n\n" + save_table(tab, "11_dsr", floatfmt="{:.3f}") + "\n")
    return {"dsr": tab}
