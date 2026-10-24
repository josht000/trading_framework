"""Chronological five-fold stability and luck tests for trading reports.

This module evaluates a fixed configuration; it never refits parameters inside
the folds.  The random comparison is generated from the same candidate pool
and sizing as the supplied strategy, then its per-fold profitability rates are
combined with an exact Poisson-binomial tail probability.
"""

from math import sqrt
import os

import numpy as np
import pandas as pd

from src.backtest import partition
from src.backtest.pit_metrics import select_trades
from src.utils import timing


def chronological_folds(days, k=5):
    """Return k contiguous, non-overlapping chronological trading-day blocks."""
    days = pd.DatetimeIndex(days).normalize().drop_duplicates().sort_values()
    if len(days) < k:
        raise ValueError(f"need at least {k} trading days, got {len(days)}")
    return [pd.DatetimeIndex(x) for x in np.array_split(days.to_numpy(), k)]


def poisson_binomial_tail(probabilities, observed):
    """Exact P(sum Bernoulli(p_i) >= observed) by dynamic programming."""
    p = np.asarray(probabilities, dtype=float)
    if len(p) == 0:
        return 1.0 if observed <= 0 else 0.0
    dp = np.zeros(len(p) + 1)
    dp[0] = 1.0
    for q in np.clip(p, 0.0, 1.0):
        dp[1:] = dp[1:] * (1 - q) + dp[:-1] * q
        dp[0] *= 1 - q
    return float(dp[max(0, int(observed)):].sum())


@timing.timed("fold metrics")
def _fold_metrics(trades, fold_days, rank, pool, max_pos, frac, n_draws, seed):
    result = partition.partition(
        trades, fold_days, pool=pool, rank=rank, bias={"value": 0.0, "src": "fixed k-fold"},
        max_pos=max_pos, frac=frac, n_draws=n_draws, n_boot=100, seed=seed,
    )
    selected = select_trades(trades, max_pos=max_pos, rank=rank)
    selected = selected[selected["day"].isin(fold_days)]
    r = selected["ret"].astype(float) if len(selected) else pd.Series(dtype=float)
    daily = pd.Series(dtype=float)
    if len(selected):
        daily = selected.assign(pnl=frac * selected["ret"]).groupby("day")["pnl"].sum()
    daily = daily.reindex(fold_days, fill_value=0.0)
    equity = (1 + daily).cumprod()
    peak = equity.cummax()
    max_dd = float((equity / peak - 1).min()) if len(equity) else 0.0
    losses = r[r <= 0]
    pf = float(r[r > 0].sum() / abs(losses.sum())) if len(losses) and losses.sum() else np.inf
    random_totals = np.asarray(result["_series"]["random_totals"], dtype=float)
    out = {
        "days": len(fold_days), "trades": len(selected),
        "total_log": float(result["total_log"]), "account_return": float(np.expm1(result["total_log"])),
        "mean_trade": float(r.mean()) if len(r) else np.nan,
        "win_rate": float((r > 0).mean()) if len(r) else np.nan,
        "pf": pf, "max_dd": max_dd,
        "trend_iwm": float(result["trend_iwm"]), "random_pick": float(result["random_pick"]),
        "skill": float(result["total_log"] - result["trend_iwm"]),
        "skill_vs_random": float(result["total_log"] - result["random_pick"]),
        "p_random_profitable": float(np.mean(random_totals > 0)),
        "profitable": bool(result["total_log"] > 0),
        "beats_random": bool(result["total_log"] > result["random_pick"]),
    }
    out.update({
        "fold_total_log": out["total_log"], "account_return_pct": 100 * out["account_return"],
        "mean_pct_trade": 100 * out["mean_trade"], "win_pct": 100 * out["win_rate"],
        "max_dd_pct": 100 * out["max_dd"], "trend": out["trend_iwm"],
        "random": out["random_pick"],
    })
    return out


def kfold_stability(trades, days, rank="rel_vol", pool=None, k=5, max_pos=5, frac=.2,
                    n_draws=2000, seed=20261102, label="", tuned=False):
    """Return fold rows and a summary for one fixed configuration and period."""
    with timing.stage("prepare kfold folds"):
        t = trades.copy()
        if "day" not in t:
            t["day"] = pd.to_datetime(t["entry_time"].astype(str).str[:10])
        else:
            t["day"] = pd.to_datetime(t["day"]).dt.normalize()
        p = pool.copy() if pool is not None else t.copy()
        if "day" not in p:
            p["day"] = pd.to_datetime(p["entry_time"].astype(str).str[:10])
        else:
            p["day"] = pd.to_datetime(p["day"]).dt.normalize()
        folds = chronological_folds(days, k)
    rows = []
    for i, fold in enumerate(folds, 1):
        m = t["day"].isin(fold)
        pm = p["day"].isin(fold)
        row = _fold_metrics(t[m], fold, rank, p[pm], max_pos, frac, n_draws, seed + i)
        row.update({"fold": i, "period": label})
        rows.append(row)
    with timing.stage("kfold summary"):
        df = pd.DataFrame(rows)
        totals = df["total_log"].to_numpy(float)
        mean = float(np.mean(totals))
        sd = float(np.std(totals, ddof=1)) if len(totals) > 1 else np.nan
        observed = int(df["profitable"].sum())
        p_tail = poisson_binomial_tail(df["p_random_profitable"], observed)
        total = float(totals.sum())
        best_i = int(np.argmax(totals))
        without_best = total - totals[best_i]
        summary = {
            "setup": label, "period": label.split("|")[-1] if "|" in label else "",
            "folds": k, "folds_profit": f"{observed}/{k}",
            "folds_beat_random": f"{int(df['beats_random'].sum())}/{k}",
            "fold_mean_log": mean, "fold_std_log": sd,
            "fold_t": mean / (sd / sqrt(k)) if np.isfinite(sd) and sd > 0 else np.nan,
            "min_fold_log": float(np.min(totals)), "max_fold_log": float(np.max(totals)),
            "worst_fold": int(np.argmin(totals) + 1),
            "best_fold_share": float(totals[best_i] / total) if total else np.nan,
            "sign_with_best": int(np.sign(total)), "sign_without_best": int(np.sign(without_best)),
            "kfold_p": p_tail, "observed_profitable": observed,
            "tuned_dev": bool(tuned),
        }
    return df, summary


def write_kfold(rows, summary, out_dir="reports/partition", name="strategy", period="dev"):
    """Write the fold table, summary, and a five-bar profitability figure."""
    out = os.path.join(out_dir)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    frame = rows.copy()
    frame.to_csv(os.path.join(out, f"kfold_{name}_{period}.csv"), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(out, f"kfold_{name}_{period}_summary.csv"), index=False)
    frame.to_markdown(os.path.join(out, f"kfold_{name}_{period}.md"), index=False, floatfmt=".6f")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = ["#2ca02c" if x else "#d62728" for x in frame["profitable"]]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(frame["fold"], frame["total_log"], color=colors)
        ax.axhline(0, color="#555", lw=.8)
        ax.set_xlabel("chronological fold")
        ax.set_ylabel("total log return")
        ax.set_title(f"k=5: {summary['folds_profit']} profitable, p={summary['kfold_p']:.3f}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "figures", f"kfold_{name}_{period}.png"), dpi=110)
        plt.close(fig)
    except ImportError:
        pass
    return frame, summary
