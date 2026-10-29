"""
Sections of the winning-trade report (see analyze_winners.py for the data). Each section appends markdown to `md`
and writes figures to reports/figures and tables to reports/tables.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = "reports"
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")
RNG = np.random.default_rng(7)
PURPLE, GREEN, RED, GRAY, GOLD, BLUE = "#7b2cbf", "#26a69a", "#ef5350", "#808080", "#d4a017", "#1f6fe5"


# ---------------------------------------------------------------- helpers
def md_table(df, floatfmt="{:.2f}", index=True):
    df = df.copy()
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        rows.append([floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                     else ("" if (isinstance(v, float) and not np.isfinite(v)) else str(v)) for v in r])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def save_table(df, name, **kw):
    df.to_csv(os.path.join(TAB, name + ".csv"))
    return md_table(df, **kw)


def savefig(fig, name):
    path = os.path.join(FIG, name + ".png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f"![{name}](figures/{name}.png)"


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def bootstrap_ci(x, stat=np.mean, n=4000, alpha=0.05):
    x = np.asarray(x)
    idx = RNG.integers(0, len(x), (n, len(x)))
    s = np.apply_along_axis(stat, 1, x[idx])
    return np.percentile(s, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def nw_tstat(x, lags=5):
    x = np.asarray(x, float)
    m = x.mean()
    d = x - m
    n = len(x)
    s = (d * d).sum() / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * (d[l:] * d[:-l]).sum() / n
    return m / np.sqrt(max(s, 1e-18) / n)


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def profit_factor(r):
    r = np.asarray(r)
    neg = abs(r[r <= 0].sum())
    return r[r > 0].sum() / neg if neg > 0 else np.inf


def group_stats(T, by, min_n=15):
    g = T.groupby(by, observed=True)
    out = pd.DataFrame({
        "n": g.size(), "win%": 100 * g["winner"].mean(), "big%": 100 * g["big"].mean(),
        "mean%": 100 * g["ret"].mean(), "median%": 100 * g["ret"].median(),
        "PF": g["ret"].apply(profit_factor), "sum_ret": g["ret"].sum(),
    })
    out["share_of_total_profit%"] = 100 * out["sum_ret"] / T["ret"].sum()
    return out[out["n"] >= min_n]


# ---------------------------------------------------------------- A. concentration
def sec_concentration(T, md):
    md.append("## 1. How concentrated is the profit?\n")
    r = T["ret"].sort_values(ascending=False).to_numpy()
    total = r.sum()
    rows = []
    for k in (1, 3, 5, 10, 20, 50, 100):
        rows.append({"top_k_trades": k, "share_of_total_return%": 100 * r[:k].sum() / total,
                     "mean_of_rest%": 100 * r[k:].mean(), "PF_of_rest": profit_factor(r[k:])})
    tab = pd.DataFrame(rows).set_index("top_k_trades")
    md.append(f"{len(T)} trades; {int(T['winner'].sum())} winners ({100 * T['winner'].mean():.1f}%); mean trade {100 * r.mean():+.2f}%; "
              f"median {100 * np.median(r):+.2f}%. Total of per-trade returns {total:.1f} (sum of fractions).\n")
    md.append(save_table(tab, "01_topk") + "\n")
    lo, hi = bootstrap_ci(r)
    md.append(f"Bootstrap 95% CI for the mean trade: [{100 * lo:+.2f}%, {100 * hi:+.2f}%]; Newey-West t-stat of the mean: "
              f"{nw_tstat(T.sort_values('entry_time')['ret'].to_numpy()):.2f}. "
              f"Kurtosis {stats.kurtosis(r):.1f}, skew {stats.skew(r):.1f} (heavy right tail).\n")
    # Lorenz curve
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    pos = np.sort(r[r > 0])[::-1]
    ax[0].plot(np.arange(1, len(pos) + 1) / len(pos) * 100, np.cumsum(pos) / pos.sum() * 100, color=PURPLE, lw=2)
    ax[0].plot([0, 100], [0, 100], color="#bbbbbb", ls="--")
    ax[0].set_xlabel("% of winning trades (best first)")
    ax[0].set_ylabel("% of gross profit")
    ax[0].set_title("Winners: cumulative share of gross profit")
    gini = 1 - 2 * np.trapezoid(np.cumsum(np.sort(pos)) / pos.sum(), np.arange(1, len(pos) + 1) / len(pos))
    ax[0].text(5, 80, f"Gini of winner sizes {gini:.2f}", color=PURPLE)
    ax[0].grid(color="#eeeeee")
    srt = T.sort_values("entry_time")["ret"].to_numpy()
    for k, col in ((0, GRAY), (5, GOLD), (10, RED)):
        rest = srt if k == 0 else T.loc[~T.index.isin(T["ret"].nlargest(k).index)].sort_values("entry_time")["ret"].to_numpy()
        ax[1].plot(np.cumsum(rest) * 100, color=col, lw=1.6, label="all trades" if k == 0 else f"without top {k}")
    ax[1].set_title("Cumulative sum of trade returns (%) with the best trades removed")
    ax[1].set_xlabel("trade number")
    ax[1].legend(frameon=False)
    ax[1].grid(color="#eeeeee")
    md.append(savefig(fig, "01_concentration") + "\n")
    top = T.nlargest(20, "ret")[["day", "symbol", "ret", "sector", "industry", "sig_day_ret", "gap_open", "rel_vol", "exit_reason", "period"]].copy()
    top["day"] = top["day"].dt.date
    top["ret"] = 100 * top["ret"]
    top["sig_day_ret"] = 100 * top["sig_day_ret"]
    top["gap_open"] = 100 * top["gap_open"]
    md.append("Top 20 trades:\n\n" + save_table(top.set_index("day"), "01_top20", floatfmt="{:.1f}") + "\n")
    return {"gini": gini}


# ---------------------------------------------------------------- B. time structure
def sec_time(T, md, mkt):
    md.append("## 2. Timing: clusters, cycles and regimes\n")
    wk = T.groupby("week").agg(n=("ret", "size"), total=("ret", "sum"), wins=("winner", "sum"), big=("big", "sum"), monster=("monster", "sum"))
    wk["win%"] = 100 * wk["wins"] / wk["n"]
    # clustering of big winners across weeks (permutation test on label dispersion given trades per week)
    p = T["big"].mean()
    obs = ((wk["big"] - wk["n"] * p) ** 2 / (wk["n"] * p * (1 - p))).sum()
    labels = T["big"].to_numpy().copy()
    weeks = T["week"].to_numpy()
    codes, uniq = pd.factorize(weeks, sort=True)
    sims = []
    for _ in range(2000):
        RNG.shuffle(labels)
        cnt = np.bincount(codes, weights=labels, minlength=len(uniq))
        sims.append(((cnt - wk["n"].to_numpy() * p) ** 2 / (wk["n"].to_numpy() * p * (1 - p))).sum())
    pval = (np.sum(np.array(sims) >= obs) + 1) / (len(sims) + 1)
    fano = (wk["big"].var() / wk["big"].mean())
    md.append(f"**Clustering of big winners (>= +10%) across weeks.** Dispersion statistic {obs:.0f} vs {np.mean(sims):.0f} expected if "
              f"big winners were spread evenly given each week's trade count (permutation p = {pval:.4f}); Fano factor of weekly counts {fano:.2f} "
              f"(1 = Poisson). {'Big winners are significantly clustered in time.' if pval < 0.05 else 'No significant time clustering of big winners.'}\n")
    top_w = wk.sort_values("total", ascending=False).head(8)
    md.append("Best weeks by summed trade return:\n\n" + save_table(top_w[["n", "wins", "big", "monster", "total", "win%"]], "02_best_weeks") + "\n")
    share = top_w["total"].sum() / wk["total"].clip(lower=0).sum() * 100
    md.append(f"The best 8 weeks ({len(top_w) / len(wk) * 100:.0f}% of {len(wk)} weeks) hold {share:.0f}% of the positive weekly return.\n")

    # change point on trade returns in time order
    s = T.sort_values("entry_time")["ret"].to_numpy().copy()

    def best_split(x, minseg=80):
        n = len(x)
        best, k_best = 0, None
        c = np.cumsum(x)
        for k in range(minseg, n - minseg):
            m1, m2 = c[k - 1] / k, (c[-1] - c[k - 1]) / (n - k)
            v = x.var()
            t = abs(m1 - m2) / np.sqrt(v * (1 / k + 1 / (n - k)))
            if t > best:
                best, k_best = t, k
        return best, k_best

    t_obs, k_obs = best_split(s)
    null = []
    for _ in range(300):
        RNG.shuffle(s)
        null.append(best_split(s)[0])
    cp_p = (np.sum(np.array(null) >= t_obs) + 1) / (len(null) + 1)
    ordered = T.sort_values("entry_time")
    cp_date = ordered["day"].iloc[k_obs].date() if k_obs else None
    left, right = ordered["ret"].iloc[:k_obs], ordered["ret"].iloc[k_obs:]
    md.append(f"**Regime change.** The most likely single change point in mean trade return is {cp_date} (max t = {t_obs:.2f}, "
              f"permutation p = {cp_p:.3f}). Mean trade before: {100 * left.mean():+.2f}% (hit rate {100 * (left > 0).mean():.1f}%), "
              f"after: {100 * right.mean():+.2f}% (hit rate {100 * (right > 0).mean():.1f}%).\n")

    # hit rate by quarter with Wilson CI, market context
    q = T.groupby(T["day"].dt.to_period("Q")).agg(n=("ret", "size"), wins=("winner", "sum"), mean=("ret", "mean"), big=("big", "mean"))
    q["win%"] = 100 * q["wins"] / q["n"]
    ci = [wilson(w, n) for w, n in zip(q["wins"], q["n"])]
    q["win%_lo"] = [100 * c[0] for c in ci]
    q["win%_hi"] = [100 * c[1] for c in ci]
    q["mean%"] = 100 * q["mean"]
    q["big%"] = 100 * q["big"]
    iwm_q = mkt["IWM"].resample("QE").last().pct_change().to_period("Q") * 100
    spy_q = mkt["SPY"].resample("QE").last().pct_change().to_period("Q") * 100
    q["IWM_qtr%"] = iwm_q.reindex(q.index).values
    q["SPY_qtr%"] = spy_q.reindex(q.index).values
    md.append("Quarterly performance and market context (hit-rate 95% Wilson interval):\n\n" +
              save_table(q[["n", "win%", "win%_lo", "win%_hi", "big%", "mean%", "IWM_qtr%", "SPY_qtr%"]], "02_quarters") + "\n")

    # monthly P&L vs IWM, seasonality
    mo = T.groupby("month").agg(n=("ret", "size"), total=("ret", "sum"), win=("winner", "mean"))
    mo["iwm"] = mkt["IWM"].resample("MS").last().pct_change().reindex(mo.index)
    mo["spy"] = mkt["SPY"].resample("MS").last().pct_change().reindex(mo.index)
    cor_i = stats.spearmanr(mo["total"] / mo["n"], mo["iwm"], nan_policy="omit")
    cor_s = stats.spearmanr(mo["total"] / mo["n"], mo["spy"], nan_policy="omit")
    md.append(f"Monthly mean trade return vs IWM monthly return: Spearman {cor_i.statistic:+.2f} (p = {cor_i.pvalue:.2f}); vs SPY {cor_s.statistic:+.2f} "
              f"(p = {cor_s.pvalue:.2f}), n = {len(mo)} months.\n")

    # autocorrelation of weekly mean return and weekly big-winner count
    wmean = (wk["total"] / wk["n"]).to_numpy()
    ac = [pd.Series(wmean).autocorr(l) for l in range(1, 9)]
    nul = [[pd.Series(RNG.permutation(wmean)).autocorr(l) for l in range(1, 9)] for _ in range(500)]
    hi = np.percentile(np.abs(nul), 97.5, axis=0)
    md.append("Autocorrelation of weekly mean trade return, lags 1-8: " + ", ".join(f"{a:+.2f}" for a in ac) +
              f" (permutation 95% band about +-{hi.mean():.2f}). " +
              ("Some lags exceed the band, suggesting persistence." if (np.abs(ac) > hi).any() else "None exceeds the band: no detectable week-to-week persistence.") + "\n")

    # earnings-season share
    def in_season(d):
        md_ = d.month * 100 + d.day
        return (120 <= md_ <= 315) or (420 <= md_ <= 520) or (720 <= md_ <= 820) or (1020 <= md_ <= 1120)

    T = T.assign(season=T["day"].apply(in_season))
    tab = group_stats(T, "season", min_n=1)
    ct = pd.crosstab(T["season"], T["big"])
    odds, fp = stats.fisher_exact(ct.values) if ct.shape == (2, 2) else (np.nan, np.nan)
    md.append("Earnings seasons (approximate small-cap windows: Jan 20-Mar 15, Apr 20-May 20, Jul 20-Aug 20, Oct 20-Nov 20) vs other weeks:\n\n" +
              save_table(tab, "02_season") + f"\n\nFisher exact test on big-winner share: odds ratio {odds:.2f}, p = {fp:.3f}.\n")

    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    ordered = T.sort_values("entry_time")
    ax[0].plot(ordered["day"], ordered["ret"].cumsum() * 100, color=PURPLE, lw=1.8)
    ax[0].axvline(pd.Timestamp(cp_date), color=RED, ls="--")
    ax[0].set_ylabel("cumulative sum of\ntrade returns (%)")
    ax[0].set_title("Cumulative trade returns, change point (red) and weekly big-winner counts")
    ax[0].grid(color="#eeeeee")
    ax[1].bar(wk.index, wk["big"], width=5, color=GREEN, label="big winners (>= +10%)")
    ax[1].bar(wk.index, wk["monster"], width=5, color=GOLD, label="monsters (>= +25%)")
    ax[1].legend(frameon=False, loc="upper right")
    ax[1].set_ylabel("count per week")
    ax[1].grid(color="#eeeeee")
    ax[2].plot(mkt["IWM"].index, mkt["IWM"] / mkt["IWM"].loc[T["day"].min():].iloc[0], color=GOLD, label="IWM")
    ax[2].plot(mkt["SPY"].index, mkt["SPY"] / mkt["SPY"].loc[T["day"].min():].iloc[0], color=BLUE, label="SPY")
    ax[2].set_xlim(T["day"].min(), T["day"].max())
    ax[2].legend(frameon=False)
    ax[2].set_ylabel("index (start = 1)")
    ax[2].grid(color="#eeeeee")
    md.append(savefig(fig, "02_time") + "\n")
    return {"cluster_p": pval, "changepoint": cp_date, "cp_p": cp_p}


# ---------------------------------------------------------------- C. catalysts / earnings
def sec_catalysts(T, md):
    md.append("## 3. What moved them: earnings, news and themes\n")
    cats = ["earnings", "contract_partnership", "clinical_fda", "offering_dilution", "m_and_a", "analyst", "theme_hype"]
    rows = []
    groups = {"all trades": T, "winners": T[T["winner"]], "big winners (>=+10%)": T[T["big"]], "monsters (>=+25%)": T[T["monster"]],
              "losers": T[~T["winner"]], "worst 5%": T[T["ret"] <= T["ret"].quantile(0.05)]}
    for name, g in groups.items():
        row = {"group": name, "n": len(g), "any_news_after_entry%": 100 * g["post_any_news"].mean(),
               "company_specific_after%": 100 * (g["post_specific"] > 0).mean(), "mover_list_after%": 100 * (g["post_mover_list"] > 0).mean()}
        for c in cats:
            row[c + "%"] = 100 * g[f"post_{c}"].mean()
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("group")
    md.append("Share of trades with a headline between entry (15:55) and the next 10:00 ET, by type "
              "(Alpaca/Benzinga; mover-list and calendar items excluded from the category counts). Analysis only: this is "
              "information the strategy does not have at entry.\n\n" + save_table(tab, "03_post_news") + "\n")
    # lift of each category among big winners vs all with Fisher test
    lifts = []
    for c in cats + ["mover_list"]:
        col = f"post_{c}" if c != "mover_list" else "post_mover_list"
        x = (T[col] > 0) if c == "mover_list" else T[col].astype(bool)
        a = pd.crosstab(x, T["big"])
        if a.shape == (2, 2):
            o, p = stats.fisher_exact(a.values)
            lifts.append({"event": c, "share_of_trades%": 100 * x.mean(), "big_rate_with%": 100 * T.loc[x, "big"].mean(),
                          "big_rate_without%": 100 * T.loc[~x, "big"].mean(), "mean_ret_with%": 100 * T.loc[x, "ret"].mean(),
                          "odds_ratio": o, "p": p})
    lt = pd.DataFrame(lifts).set_index("event")
    lt["q_BH"] = bh_fdr(lt["p"].to_numpy())
    md.append("Does a post-entry event raise the odds of a big winner? (Fisher exact, BH-adjusted q):\n\n" + save_table(lt, "03_news_lift", floatfmt="{:.3f}") + "\n")
    # earnings via SimFin publish dates
    e = T[T["has_earn_data"] == 1]
    tab2 = group_stats(e, "earn_next_0_3d", min_n=5)
    ct = pd.crosstab(e["earn_next_0_3d"], e["big"])
    o, p = stats.fisher_exact(ct.values) if ct.shape == (2, 2) else (np.nan, np.nan)
    md.append(f"**Earnings filings (SimFin publish dates, {len(e)} trades with data)**: a quarterly filing published on the entry day or within the next 3 days "
              f"flags earnings-linked trades (the filing date can lag the press release).\n\n" + save_table(tab2, "03_earnings") +
              f"\n\nBig-winner odds ratio for earnings-linked entries {o:.2f} (Fisher p = {p:.3f}). "
              f"Share of big winners that are earnings-linked: {100 * e.loc[e['big'], 'earn_next_0_3d'].mean():.1f}% vs {100 * e['earn_next_0_3d'].mean():.1f}% of all trades.\n")
    # pre-entry news
    tab3 = group_stats(T.assign(pre_news=np.where(T["pre_specific_24h"] > 0, "company news in prior 24h", "none")), "pre_news", min_n=5)
    md.append("Pre-entry news (known at 15:55):\n\n" + save_table(tab3, "03_pre_news") + "\n")
    # where in the night do winners gain
    path = T.groupby(pd.cut(T["ret"], [-1, -0.10, -0.03, 0, 0.03, 0.10, 0.25, 10], labels=["<-10%", "-10..-3%", "-3..0", "0..3%", "3..10%", "10..25%", ">25%"]), observed=True)
    pt = path[["ah45_last", "pm_last", "open_ret"]].median() * 100
    pt.columns = ["median % at 16:45", "median % pre-market (last bar)", "median % at 09:30 open"]
    pt.insert(0, "n", path.size())
    md.append("Where the move happens (median return vs the 15:55 price at each checkpoint, by final outcome bucket):\n\n" + save_table(pt, "03_path") + "\n")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(tab.columns[3:]))
    w = 0.2
    for i, g in enumerate(["all trades", "big winners (>=+10%)", "monsters (>=+25%)", "worst 5%"]):
        ax.bar(x + (i - 1.5) * w, tab.loc[g, tab.columns[3:]].values, width=w, label=g)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("%", "").replace("_", "\n") for c in tab.columns[3:]], fontsize=8)
    ax.set_ylabel("% of trades with a post-entry headline of this type")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#eeeeee")
    md.append(savefig(fig, "03_catalysts") + "\n")
    return {"lifts": lt}


# ---------------------------------------------------------------- D. sector / industry
def sec_sector(T, md):
    md.append("## 4. Sectors, industries and themes\n")
    sec = group_stats(T, "sector", min_n=20).sort_values("share_of_total_profit%", ascending=False)
    md.append("By sector (SimFin classification, yfinance fallback; 'Unknown' = no classification, mostly delisted names):\n\n" + save_table(sec, "04_sector") + "\n")
    # enrichment of sectors among big winners: hypergeometric / Fisher with BH
    rows = []
    for s, g in T.groupby("sector"):
        if len(g) < 20:
            continue
        a = pd.crosstab(T["sector"] == s, T["big"])
        if a.shape != (2, 2):
            continue
        o, p = stats.fisher_exact(a.values)
        rows.append({"sector": s, "n": len(g), "big_rate%": 100 * g["big"].mean(), "base_rate%": 100 * T["big"].mean(), "odds_ratio": o, "p": p})
    en = pd.DataFrame(rows).set_index("sector")
    en["q_BH"] = bh_fdr(en["p"].to_numpy())
    md.append("Over/under-representation of sectors among big winners (Fisher, BH-adjusted):\n\n" + save_table(en.sort_values("p"), "04_sector_enrichment", floatfmt="{:.3f}") + "\n")
    ind = group_stats(T[T["industry"].notna()], "industry", min_n=15).sort_values("sum_ret", ascending=False)
    md.append("Top industries by summed return (min 15 trades):\n\n" + save_table(ind.head(12), "04_industry_top") +
              "\n\nWorst industries by summed return:\n\n" + save_table(ind.tail(8), "04_industry_bottom") + "\n")
    # sector-day clustering: days with >=2 big winners in the same sector, vs permutation
    big = T[T["big"] & (T["sector"] != "Unknown")]
    obs = big.groupby(["day", "sector"]).size()
    obs_pairs = int((obs * (obs - 1) / 2).sum())
    sims = []
    secs = big["sector"].to_numpy().copy()
    days = big["day"].to_numpy()
    pool_idx = np.arange(len(big))
    for _ in range(1000):
        RNG.shuffle(secs)
        c = pd.Series(1, index=pd.MultiIndex.from_arrays([days, secs])).groupby(level=[0, 1]).sum()
        sims.append(int((c * (c - 1) / 2).sum()))
    pv = (np.sum(np.array(sims) >= obs_pairs) + 1) / (len(sims) + 1)
    md.append(f"**Same-day sector sympathy.** Pairs of big winners in the same sector on the same day: {obs_pairs} observed vs {np.mean(sims):.1f} expected if sectors were "
              f"assigned at random among big winners on each day (permutation p = {pv:.4f}).\n")
    # sector-week waves
    sw = big.groupby(["week", "sector"]).size().rename("n").reset_index()
    waves = sw.sort_values("n", ascending=False).head(12)
    names = []
    for _, r in waves.iterrows():
        g = big[(big["week"] == r["week"]) & (big["sector"] == r["sector"])]
        names.append(", ".join(g.sort_values("ret", ascending=False)["symbol"].head(6)))
    waves["tickers"] = names
    waves["week"] = waves["week"].dt.date
    md.append("Strongest sector 'waves' (weeks with the most big winners in one sector):\n\n" + save_table(waves.set_index("week"), "04_waves") + "\n")
    # heatmap
    piv = T[T["big"]].pivot_table(index="sector", columns="month", values="ret", aggfunc="size", fill_value=0)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    im = ax.imshow(piv.values, aspect="auto", cmap="Purples")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_xticks(range(0, len(piv.columns), 2))
    ax.set_xticklabels([c.strftime("%y-%m") for c in piv.columns[::2]], fontsize=7, rotation=60)
    ax.set_title("Big winners (>= +10%) per sector per month")
    fig.colorbar(im, ax=ax, shrink=0.8)
    md.append(savefig(fig, "04_sector_heatmap") + "\n")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sc = sec.sort_values("mean%")
    ax.barh(sc.index, sc["mean%"], color=np.where(sc["mean%"] > 0, GREEN, RED))
    ax.set_xlabel("mean trade return (%)")
    ax.set_title("Mean trade return by sector")
    ax.grid(axis="x", color="#eeeeee")
    md.append(savefig(fig, "04_sector_bars") + "\n")
    return {"sector_cluster_p": pv}


# ---------------------------------------------------------------- E. float / size / liquidity
def sec_float(T, md):
    md.append("## 5. Float, size, liquidity and volatility\n")
    feats = {
        "shares_pit": "shares outstanding (SimFin, point-in-time; float proxy)",
        "mcap_pit": "market cap (SimFin, point-in-time)",
        "float_now": "float (yfinance current snapshot, not point-in-time)",
        "short_pct_float_now": "short % of float (current snapshot)",
        "price": "price at 15:55",
        "dvol20": "20-day avg dollar volume",
        "rel_vol": "relative volume at 15:55",
        "atr_pct": "daily ATR % of price",
        "sig_day_ret": "signal-day return to 15:55",
        "gap_open": "signal-day open gap",
        "range_pct": "signal-day range",
        "run5": "5-day run-up",
        "headroom": "headroom to next resistance (13m)",
        "dist_sma200": "distance to 200 SMA",
    }
    rows, tabs = [], {}
    for f, desc in feats.items():
        x = T[f].replace([np.inf, -np.inf], np.nan)
        if f == "headroom":
            x = x.clip(upper=2.0)
        ok = x.notna()
        if ok.sum() < 200:
            continue
        rho_ret, p_ret = stats.spearmanr(x[ok], T.loc[ok, "ret"])
        rho_big, p_big = stats.spearmanr(x[ok], T.loc[ok, "big"].astype(float))
        rows.append({"feature": f, "description": desc, "n": int(ok.sum()), "spearman_vs_ret": rho_ret, "p_ret": p_ret, "spearman_vs_big": rho_big, "p_big": p_big})
        q = pd.qcut(x[ok], 5, duplicates="drop")
        labels = [f"Q{c + 1}: {iv}" for c, iv in zip(q.cat.codes, q)]
        tabs[f] = group_stats(T.loc[ok].assign(bin=labels), "bin", min_n=10)
    corr = pd.DataFrame(rows).set_index("feature")
    corr["q_BH(ret)"] = bh_fdr(corr["p_ret"].to_numpy())
    md.append("Rank correlation of each feature with trade return and with being a big winner (BH-adjusted q on the return test):\n\n" +
              save_table(corr.drop(columns="description"), "05_feature_correlations", floatfmt="{:.3f}") + "\n")
    for f in ("shares_pit", "mcap_pit", "float_now", "short_pct_float_now", "price", "rel_vol", "atr_pct", "sig_day_ret", "dvol20", "headroom"):
        if f in tabs:
            md.append(f"**{f}** quintiles ({feats[f]}):\n\n" + save_table(tabs[f][["n", "win%", "big%", "mean%", "median%", "PF"]], f"05_quintiles_{f}") + "\n")
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, f in zip(axes.ravel(), ["shares_pit", "mcap_pit", "float_now", "price", "rel_vol", "atr_pit" if False else "atr_pct", "sig_day_ret", "headroom"]):
        if f not in tabs:
            ax.axis("off")
            continue
        g = tabs[f]
        ax.bar(range(len(g)), g["mean%"], color=np.where(g["mean%"] > 0, GREEN, RED))
        ax2 = ax.twinx()
        ax2.plot(range(len(g)), g["big%"], color=PURPLE, marker="o")
        ax.set_title(f, fontsize=9)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([f"Q{i + 1}" for i in range(len(g))], fontsize=8)
        ax.set_ylabel("mean %")
        ax2.set_ylabel("big-winner rate %", color=PURPLE)
    fig.suptitle("Mean return (bars) and big-winner rate (line) by quintile of each feature", y=1.01)
    fig.tight_layout()
    md.append(savefig(fig, "05_quintiles") + "\n")
    # low float specifically
    sh = T["shares_pit"] / 1e6
    low = T.assign(size=pd.cut(sh, [0, 10, 30, 100, 300, 1e5], labels=["<10M sh", "10-30M", "30-100M", "100-300M", ">300M"]))
    md.append("By shares outstanding (SimFin, millions):\n\n" + save_table(group_stats(low[low["size"].notna()], "size", min_n=10), "05_shares_bins") + "\n")
    return {"corr": corr}


# ---------------------------------------------------------------- F. market link
def ols_hac(y, X, lags=5):
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * res[:, None]).T @ (X * res[:, None])
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = (X[l:] * res[l:, None]).T @ (X[:-l] * res[:-l, None])
        S += w * (G + G.T)
    cov = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, beta / se, r2


def sec_market(T, md, cand):
    md.append("## 6. Link to the broader market\n")
    daily = T.groupby("day").agg(ret=("ret", "mean"), n=("ret", "size"), big=("big", "sum"),
                                 spy_gap=("spy_gap_next", "first"), iwm_gap=("iwm_gap_next", "first"), qqq_gap=("qqq_gap_next", "first"),
                                 spy_day=("spy_day", "first"), iwm_day=("iwm_day", "first"),
                                 iwm_ret5=("iwm_ret5", "first"), iwm_ret20=("iwm_ret20", "first"), spy_ret20=("spy_ret20", "first")).dropna()
    rows = []
    for c in ("spy_gap", "iwm_gap", "qqq_gap", "spy_day", "iwm_day", "iwm_ret5", "iwm_ret20", "spy_ret20"):
        rho, p = stats.spearmanr(daily[c], daily["ret"])
        rows.append({"market variable": c, "spearman_vs_daily_mean_ret": rho, "p": p})
    tab = pd.DataFrame(rows).set_index("market variable")
    md.append(f"Daily mean trade return vs market variables ({len(daily)} trading days; '*_gap' = next-session close-to-open move of the index, "
              f"known only afterwards; '*_day' = the index move from the prior close to 15:55 on entry day):\n\n" + save_table(tab, "06_market_corr", floatfmt="{:.3f}") + "\n")
    y = daily["ret"].to_numpy()
    X = daily[["iwm_gap", "spy_gap"]].to_numpy()
    beta, t, r2 = ols_hac(y, X)
    md.append(f"**Overnight beta.** OLS with Newey-West errors, daily mean trade return on IWM and SPY overnight gaps: intercept (alpha) {100 * beta[0]:+.2f}% per day "
              f"(t = {t[0]:.2f}); beta to IWM gap {beta[1]:.2f} (t = {t[1]:.2f}); beta to SPY gap {beta[2]:.2f} (t = {t[2]:.2f}); R2 = {r2:.3f}.\n")
    # market regime terciles
    rows = []
    for c, lab in (("iwm_gap_next", "IWM next-session gap"), ("iwm_day", "IWM entry-day move"), ("spy_day", "SPY entry-day move"), ("iwm_ret20", "IWM 20-day trend")):
        x = T[c]
        q = pd.qcut(x, 3, labels=["low", "mid", "high"], duplicates="drop")
        gs = group_stats(T.assign(b=q), "b", min_n=5)
        gs.insert(0, "variable", lab)
        rows.append(gs.reset_index())
    md.append("Trade outcomes by tercile of market variable:\n\n" + save_table(pd.concat(rows).set_index("variable"), "06_market_terciles") + "\n")
    # breadth / hype indicator: candidates per day
    cd = cand.groupby("date").size().rename("n_cand")
    daily = daily.join(cd, how="left")
    daily["n_cand_prev20"] = daily["n_cand"].rolling(20, min_periods=5).mean().shift(1)
    daily["hype"] = daily["n_cand"] / daily["n_cand_prev20"]
    ok = daily.dropna(subset=["hype"])
    rho, p = stats.spearmanr(ok["hype"], ok["ret"])
    md.append(f"**Breadth ('hype') indicator.** The number of candidate breakouts that day relative to its prior 20-day average: Spearman vs that day's mean trade return "
              f"{rho:+.3f} (p = {p:.3f}); vs number of big winners {stats.spearmanr(ok['hype'], ok['big']).statistic:+.3f}.\n")
    # rolling correlation figure
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    rc = daily["ret"].rolling(60).corr(daily["iwm_gap"])
    ax[0].plot(rc.index, rc, color=PURPLE)
    ax[0].axhline(0, color="#bbbbbb")
    ax[0].set_ylabel("60-day rolling corr\n(daily mean ret vs IWM gap)")
    ax[0].grid(color="#eeeeee")
    ax[1].scatter(daily["iwm_gap"] * 100, daily["ret"] * 100, s=10, alpha=0.5, color=PURPLE)
    ax[1].axhline(0, color="#bbbbbb")
    ax[1].axvline(0, color="#bbbbbb")
    ax[1].set_xlabel("IWM next-session gap (%)")
    ax[1].set_ylabel("daily mean trade return (%)")
    ax[1].grid(color="#eeeeee")
    md.append(savefig(fig, "06_market") + "\n")
    return {"beta": beta, "t": t}


def sec_hedge(T, md):
    """Market-neutral view: subtract h x IWM next-session gap from each trade (short IWM from the 15:55 close; costs and borrow ignored)."""
    md.append("### 6b. How much of the result is just the market's overnight move?\n")
    rows = []
    for h in (0.0, 0.5, 1.0):
        x = T.assign(ret_h=T["ret"] - h * T["iwm_gap_next"].fillna(0))
        d = x.groupby("day")["ret_h"].mean()
        rows.append({"IWM hedge ratio": h, "mean%": 100 * x["ret_h"].mean(), "median%": 100 * x["ret_h"].median(), "win%": 100 * (x["ret_h"] > 0).mean(),
                     "PF": profit_factor(x["ret_h"]), "daily_Sharpe_ann": d.mean() / d.std() * np.sqrt(252), "NW_t_daily": nw_tstat(d.to_numpy())})
    md.append(save_table(pd.DataFrame(rows).set_index("IWM hedge ratio"), "06b_hedged") + "\n\n"
              "A hedge ratio of 1 removes the IWM close-to-open move entirely. The trades hold some positions through the open only partly (several exit "
              "pre-market or after-hours), so this is an approximation; it shows whether the edge survives once index direction is removed.\n")


# ---------------------------------------------------------------- themes via text topics
def sec_topics(T, md):
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    md.append("## 4b. Themes found from company descriptions (TF-IDF + NMF topic model)\n")
    y = pd.read_parquet(os.path.join("data", "pit", "enrichment", "yf_info.parquet")).set_index("symbol")["longBusinessSummary"].dropna()
    syms = y.index
    vec = TfidfVectorizer(stop_words="english", min_df=8, max_df=0.35, ngram_range=(1, 2), max_features=6000)
    X = vec.fit_transform(y.values)
    k = 14
    nmf = NMF(n_components=k, random_state=0, init="nndsvda", max_iter=500)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    topic_name = {i: ", ".join(terms[np.argsort(-nmf.components_[i])[:5]]) for i in range(k)}
    topic = pd.Series(W.argmax(axis=1), index=syms)
    strong = pd.Series(W.max(axis=1) > 0.015, index=syms)
    T = T.copy()
    T["topic"] = T["symbol"].map(topic)
    T = T[T["topic"].notna() & T["symbol"].map(strong).fillna(False)]
    T["topic_label"] = T["topic"].astype(int).map(lambda i: f"T{i}: {topic_name[i]}")
    tab = group_stats(T, "topic_label", min_n=25).sort_values("sum_ret", ascending=False)
    md.append(f"Each company's business description was converted to TF-IDF features and factorized into {k} topics; each company is assigned to its dominant topic "
              f"({len(T)} trades with a confident assignment). Topics ranked by summed trade return:\n\n" + save_table(tab, "04b_topics") + "\n")
    rows = []
    for lab, g in T.groupby("topic_label"):
        if len(g) < 25:
            continue
        a = pd.crosstab(T["topic_label"] == lab, T["big"])
        if a.shape == (2, 2):
            o, p = stats.fisher_exact(a.values)
            rows.append({"topic": lab, "n": len(g), "big_rate%": 100 * g["big"].mean(), "odds_ratio": o, "p": p})
    en = pd.DataFrame(rows).set_index("topic")
    en["q_BH"] = bh_fdr(en["p"].to_numpy())
    md.append("Topic enrichment among big winners (Fisher, BH-adjusted):\n\n" + save_table(en.sort_values("p"), "04b_topic_enrichment", floatfmt="{:.3f}") + "\n")
    # themes in time: top topic-weeks of big winners
    big = T[T["big"]]
    tw = big.groupby(["week", "topic_label"]).size().rename("n").reset_index().sort_values("n", ascending=False).head(10)
    tw["tickers"] = [", ".join(big[(big["week"] == r["week"]) & (big["topic_label"] == r["topic_label"])].sort_values("ret", ascending=False)["symbol"].head(6)) for _, r in tw.iterrows()]
    tw["week"] = tw["week"].dt.date
    md.append("Strongest theme waves (weeks with the most big winners sharing a topic):\n\n" + save_table(tw.set_index("week"), "04b_theme_waves") + "\n")
    fig, ax = plt.subplots(figsize=(13, 5))
    piv = big.pivot_table(index="topic_label", columns="month", values="ret", aggfunc="size", fill_value=0)
    im = ax.imshow(piv.values, aspect="auto", cmap="Purples")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([s[:50] for s in piv.index], fontsize=7)
    ax.set_xticks(range(0, len(piv.columns), 2))
    ax.set_xticklabels([c.strftime("%y-%m") for c in piv.columns[::2]], fontsize=7, rotation=60)
    ax.set_title("Big winners (>= +10%) per theme (NMF topic) per month")
    fig.colorbar(im, ax=ax, shrink=0.8)
    md.append(savefig(fig, "04b_topic_heatmap") + "\n")


# ---------------------------------------------------------------- dev vs test robustness
def sec_robust(T, md):
    md.append("## 12. Do the main findings hold out of sample (dev vs the locked test period)?\n")
    defs = {
        "all trades": lambda t: t["ret"].notna(),
        "shares outstanding < 30M": lambda t: t["shares_pit"] < 30e6,
        "market cap < $200M": lambda t: t["mcap_pit"] < 200e6,
        "daily ATR >= 9%": lambda t: t["atr_pct"] >= 0.09,
        "signal-day move >= +26%": lambda t: t["sig_day_ret"] >= 0.26,
        "rel. volume >= 5.4": lambda t: t["rel_vol"] >= 5.4,
        "price below 200 SMA": lambda t: t["above_sma200"] == 0,
        "price above 200 SMA": lambda t: t["above_sma200"] == 1,
        "headroom >= 20%": lambda t: t["headroom"] >= 0.20,
        "market cap < $200M and headroom >= 20%": lambda t: (t["mcap_pit"] < 200e6) & (t["headroom"] >= 0.20),
        "shares < 30M and headroom >= 20%": lambda t: (t["shares_pit"] < 30e6) & (t["headroom"] >= 0.20),
        "technology sector": lambda t: t["sector"] == "Technology",
        "earnings-linked (filing 0-3d)": lambda t: t["earn_next_0_3d"] == 1,
        "company news in prior 24h": lambda t: t["pre_specific_24h"] > 0,
    }
    rows = []
    for name, f in defs.items():
        for period in ("dev", "test"):
            g = T[(T["period"] == period)]
            g = g[f(g).fillna(False)]
            if len(g) == 0:
                continue
            rows.append({"group": name, "period": period, "n": len(g), "win%": 100 * g["winner"].mean(), "big%": 100 * g["big"].mean(),
                         "big_loss%": 100 * (g["ret"] <= -0.10).mean(), "mean%": 100 * g["ret"].mean(), "PF": profit_factor(g["ret"])})
    md.append(save_table(pd.DataFrame(rows).set_index("group"), "12_robustness") + "\n")
