"""Resumable exit-experiment campaign.

Run from the repository root:
  C:/data/envs/trading_framework/Scripts/python.exe scripts/exit_campaign.py --hours 5.5

The locked test table is loaded but never scored until the dev-only finalists
file has been written.  Every family appends a CSV and elapsed-time JSONL.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.pit_integrity import require_valid_pit_data
from src.backtest.exit_sim import (
    _clock,
    _window_bounds,
    baseline_policy,
    build_windows,
    elapsed_logger,
    load_windows,
    model_features,
    oracle_returns,
    save_windows,
    simulate,
)


OUT = ROOT / "results" / "exit"
TABLES = ROOT / "reports" / "exit" / "tables"
FIGURES = ROOT / "reports" / "exit" / "figures"


def _read_start():
    return float((OUT / "budget_start.txt").read_text().strip())


def _policy_rows(family, n, rng):
    if family.startswith("robustness_"):
        family = "combos"
    if family == "fixed_time":
        return [{"name": "fixed_time", "keep_ah": False, "deadline": f"09:{m:02d}"} for m in (30, 35, 40, 45, 50, 55)] + [
            {"name": "fixed_time", "keep_ah": False, "deadline": f"10:{m:02d}"} for m in (0, 5, 10, 15, 30)
        ]
    rows = []
    for _ in range(n):
        if family == "trailing":
            rows.append({"name": "smart", "keep_ah": False, "deadline": str(rng.choice(["09:45", "10:00", "10:15", "10:30"])),
                         "trail": float(rng.uniform(.01, .16)), "activate": float(rng.uniform(0, .30))})
        elif family == "momentum":
            rows.append({"name": "smart", "keep_ah": False, "deadline": str(rng.choice(["09:45", "10:00", "10:15", "10:30"])),
                         "velocity_floor": float(rng.uniform(-.08, .01)), "giveback": float(rng.uniform(.01, .20)),
                         "below_vwap": bool(rng.integers(0, 2))})
        elif family == "scale_out":
            a = float(rng.uniform(.03, .20))
            rows.append({"name": "smart", "keep_ah": False, "deadline": str(rng.choice(["10:00", "10:15", "10:30"])),
                         "scale_out": [{"target": a, "fraction": .5}, {"target": a + float(rng.uniform(.03, .25)), "fraction": .25}],
                         "trail": float(rng.uniform(.02, .12)), "activate": a})
        elif family == "pm_ah":
            rows.append({"name": "smart", "keep_ah": True, "ah_stop": float(rng.uniform(.08, .25)),
                         "ah_target": float(rng.uniform(.20, .70)), "pm_target": float(rng.uniform(.10, .40)),
                         "pm_stop": float(rng.uniform(.08, .30)), "pm_start": "08:00",
                         "deadline": str(rng.choice(["09:45", "10:00", "10:15"]))})
        elif family == "combos":
            rows.append({"name": "smart", "keep_ah": bool(rng.integers(0, 2)), "ah_stop": float(rng.uniform(.08, .20)),
                         "ah_target": float(rng.uniform(.25, .60)), "pm_target": float(rng.uniform(.15, .35)),
                         "pm_start": "08:00", "deadline": str(rng.choice(["10:00", "10:15", "10:30"])),
                         "trail": float(rng.uniform(.02, .12)), "activate": float(rng.uniform(0, .15)),
                         "below_vwap": bool(rng.integers(0, 2)), "giveback": float(rng.uniform(.03, .15))})
        elif family == "learned":
            rows.append({"name": "smart", "keep_ah": False, "deadline": str(rng.choice(["09:45", "10:00", "10:15"])),
                         "velocity_floor": float(rng.uniform(-.05, .005)),
                         "giveback": float(rng.uniform(.02, .15)), "below_vwap": True})
        elif family == "optimal_stopping":
            rows.append({"name": "smart", "keep_ah": False, "deadline": str(rng.choice(["10:00", "10:15", "10:30"])),
                         "trail": float(rng.uniform(.02, .12)), "activate": float(rng.uniform(.02, .20)),
                         "velocity_floor": float(rng.uniform(-.04, .005))})
    return rows


def _masks(trades):
    """Subset masks.  S1 = above_sma200 AND iwm_day>=0 AND spy_day>=0 (rank_test.STRATS) when the
    regime columns are present (attach_regime); otherwise falls back to above_sma200 only and
    sets _masks.s1_full = False.  H3 = above_sma200 AND headroom>=20%."""
    above = trades["above_sma200"].fillna(0).astype(bool)
    h3 = above & (trades["headroom"].fillna(-1) >= .20)
    s1 = above
    _masks.s1_full = False
    if "iwm_day" in trades.columns and "spy_day" in trades.columns:
        s1 = above & (trades["iwm_day"] >= 0).fillna(False) & (trades["spy_day"] >= 0).fillna(False)
        _masks.s1_full = True
    return {"base": np.ones(len(trades), bool), "S1": s1.to_numpy(), "H3": h3.to_numpy()}


def _pf(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    loss = abs(r[r <= 0].sum())
    return float(r[r > 0].sum() / loss) if loss > 0 else np.inf


def _score(result, trades):
    """Objective J = mean over 5 date blocks of (mean of the base/S1/H3 subset mean returns) minus
    0.5 * std over blocks.  Same function for baselines and candidates.  Also returns pooled
    per-subset mean/median/hit/PF (keys m_<subset>, med_<subset>, hit_<subset>, pf_<subset>)."""
    result = result.reset_index(drop=True)
    trades = trades.reset_index(drop=True)
    x = result.join(trades[["entry_time"]], how="left")
    x["day"] = pd.to_datetime(x["entry_time"], utc=True).dt.tz_convert("America/New_York").dt.date
    days = np.array(sorted(x["day"].dropna().unique()))
    folds = np.array_split(days, 5)
    masks = _masks(trades)
    vals, hits, counts = [], [], []
    base_hit = float((trades["ret"] > 0).mean())
    for fold in folds:
        if len(fold) == 0:
            continue
        fm = x["day"].isin(fold)
        subset_means = []
        for name in ("base", "S1", "H3"):
            m = fm.to_numpy() & masks[name]
            if m.sum():
                subset_means.append(float(x.loc[m, "ret"].mean()))
        if subset_means:
            vals.append(float(np.mean(subset_means)))
            hits.append(float((x.loc[fm, "ret"] > 0).mean()))
            counts.append(int(fm.sum()))
    mean = float(np.mean(vals)) if vals else -np.inf
    score = mean - .5 * float(np.std(vals)) if vals else -np.inf
    valid = bool(vals and min(counts) >= 30 and np.mean(hits) >= base_hit - .02)
    out = {"score": score, "mean_ret": mean, "fold_std": float(np.std(vals)) if vals else np.nan,
           "hit_rate": float(np.mean(hits)) if hits else np.nan, "valid": valid,
           "capture_trades": int(len(x))}
    for name in ("base", "S1", "H3"):
        r = x.loc[masks[name], "ret"].dropna()
        out.update({f"n_{name}": int(len(r)), f"m_{name}": float(r.mean()), f"med_{name}": float(r.median()),
                    f"hit_{name}": float((r > 0).mean()), f"pf_{name}": _pf(r)})
    return out


def score_vs_baseline(score, base_score):
    """Add delta J and delta pooled mean return per subset relative to a baseline scored identically."""
    d = {"dJ": score["score"] - base_score["score"]}
    for name in ("base", "S1", "H3"):
        d[f"dm_{name}"] = score[f"m_{name}"] - base_score[f"m_{name}"]
    return d


def evaluate_family(family, windows, trades, start, deadline, rng):
    path = TABLES / f"{family}.csv"
    if path.exists():
        return pd.read_csv(path)
    n = {"trailing": 200, "momentum": 200, "scale_out": 150, "pm_ah": 100, "combos": 200,
         "learned": 100, "optimal_stopping": 100}.get(family, 0)
    if family.startswith("robustness_"):
        n = 200
    rows = []
    policies = _policy_rows(family, n, rng)
    for i, policy in enumerate(policies):
        if time.time() - start >= deadline:
            break
        result = simulate(windows, policy, "realistic")
        score = _score(result, trades)
        rows.append({"trial": i, "params": json.dumps(policy, sort_keys=True), **score})
        if (i + 1) % 100 == 0:
            elapsed_logger(start, str(OUT / "elapsed.jsonl"), f"{family}:{i + 1}")
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def _model_rows(windows, trades):
    rows = []
    for w in windows:
        _, last = _window_bounds(w, "08:00", "10:15")
        if last is None:
            continue
        _, minutes = _clock(w)
        inds = [i for i in range(len(w.times)) if 570 <= minutes[i] <= 615 and i <= last]
        for i in inds:
            future = w.closes[i + 1:last + 1]
            if len(future):
                rows.append((w.trade_id, pd.Timestamp(trades.iloc[w.trade_id]["entry_time"]).date(),
                             model_features(w, i), float(np.max(future) / w.closes[i] - 1)))
    return rows


def evaluate_model_family(family, windows, trades, start, deadline, rng):
    """Walk-forward GBM and Longstaff-style continuation-value regressions."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    path = TABLES / f"{family}.csv"
    if path.exists():
        return pd.read_csv(path)
    samples = _model_rows(windows, trades)
    dates = np.array(sorted({x[1] for x in samples}))
    folds = np.array_split(dates, 5)
    trials = 20
    rows = []
    for trial in range(trials):
        if time.time() - start >= deadline:
            break
        threshold = float(rng.uniform(-.002, .03))
        fold_results, fold_trades = [], []
        for fold in folds:
            if len(fold) == 0:
                continue
            first = fold[0]
            train = [x for x in samples if x[1] < first - pd.Timedelta(days=4)]
            if len(train) < 100:
                continue
            X = np.vstack([x[2] for x in train])
            y = np.asarray([x[3] for x in train])
            if family == "learned":
                model = HistGradientBoostingRegressor(
                    max_iter=80, max_leaf_nodes=15, learning_rate=.08,
                    l2_regularization=float(rng.uniform(.01, 2.0)), random_state=trial,
                ).fit(X, y)
            else:
                model = Ridge(alpha=float(rng.uniform(.1, 20.0))).fit(X, y)
            fw = [w for w in windows if pd.Timestamp(trades.iloc[w.trade_id]["entry_time"]).date() in set(fold)]
            if not fw:
                continue
            r = simulate(fw, {"name": "smart", "keep_ah": False, "deadline": "10:15",
                              "model": model, "edge_threshold": threshold}, "realistic")
            fold_results.append(r)
            fold_trades.append(trades.iloc[[w.trade_id for w in fw]])
        if not fold_results:
            continue
        result = pd.concat(fold_results, ignore_index=True)
        tr = pd.concat(fold_trades, ignore_index=True)
        score = _score(result, tr)
        rows.append({"trial": trial, "params": json.dumps({"threshold": threshold}),
                     **score})
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def _stats(result):
    r = result["ret"].dropna()
    return {"trades": len(r), "hit_rate": float((r > 0).mean()), "mean_ret": float(r.mean()),
            "median_ret": float(r.median()), "pf": float(r[r > 0].sum() / abs(r[r <= 0].sum())) if (r <= 0).any() else np.inf}


def _paired_bootstrap(base, alt, days, n=5000, seed=20261021):
    """Paired day-level bootstrap of the mean per-trade improvement alt-base.
    `days` = entry day of each trade (aligned with base/alt rows).  Resamples days with replacement
    and recomputes the trade-weighted mean difference."""
    d = pd.DataFrame({"delta": alt["ret"].to_numpy() - base["ret"].to_numpy(), "day": np.asarray(days)}).dropna()
    g = d.groupby("day")["delta"].agg(["sum", "count"])
    s_, c_ = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    draws = s_[idx].sum(axis=1) / c_[idx].sum(axis=1)
    return {"mean": float(d["delta"].mean()), "lo": float(np.quantile(draws, .025)),
            "hi": float(np.quantile(draws, .975)), "days": int(len(g))}


def _plot_finalist_figures(windows, finalists, dev_result):
    FIGURES.mkdir(parents=True, exist_ok=True)
    by_id = {w.trade_id: w for w in windows}
    for name, result in finalists.items():
        r = result["ret"].to_numpy()
        eq = pd.Series(1 + r).cumprod()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(eq.to_numpy(), label=name)
        ax.set_title(f"{name} dev trade equity (trade order)")
        ax.set_ylabel("growth of $1")
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(FIGURES / f"{name}_equity.png", dpi=120)
        plt.close(fig)
    if not finalists:
        return
    name, result = next(iter(finalists.items()))
    for kind, subset in (("winner", result[result["ret"] > 0].head(3)),
                         ("loser", result[result["ret"] <= 0].head(3))):
        for j, (_, row) in enumerate(subset.iterrows(), 1):
            w = by_id[int(row["trade_id"])]
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(np.arange(len(w.closes)), w.closes / w.entry_price - 1)
            ax.axhline(0, color="black", lw=.6)
            ax.axvline(int(row["exit_idx"]), color="tab:red", ls="--")
            ax.set_title(f"{name} {kind} trade {w.symbol} {row['ret']:.2%}")
            ax.set_xlabel("minute bars from 15:55 entry")
            ax.set_ylabel("return")
            fig.tight_layout()
            fig.savefig(FIGURES / f"{name}_{kind}_{j}.png", dpi=120)
            plt.close(fig)


def write_report(start, trades, dev_result, test_result, family_tables, oracle, finalists):
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    if len(oracle):
        oracle["ret"].hist(bins=50, figsize=(8, 4))
        plt.xlabel("realistic hindsight-best return")
        plt.tight_layout()
        plt.savefig(FIGURES / "oracle_returns.png", dpi=120)
        plt.close()
    counts = "\n".join(f"- **{k}**: {len(v)} trials" for k, v in family_tables.items())
    best = []
    for k, v in family_tables.items():
        if len(v):
            q = v.sort_values("score", ascending=False).iloc[0]
            best.append(f"| {k} | {len(v)} | {q['score']:.5f} | {q['mean_ret']:.5f} | {q['hit_rate']:.3f} |")
    lines = [
        "# Exit testing report",
        "",
        "## Scope and controls",
        "",
        "Entries are the existing PIT 15:55 fills; the locked test is not scored during search. "
        "Features and policy decisions use bar-close information through the current bar only. "
        "Realistic headline results execute on the next bar open with 10% participation, "
        "0.15% RTH / 0.40% extended slippage, and 0.10% commission per side. Legacy results "
        "use the trigger-bar close and are shown only as a compatibility control.",
        "",
        "The 08:00 ET floor applies to all next-day pre-market logic and `pm_last`; smart exits "
        "are evaluated through the selected deadline in the 09:30-10:15 opening window "
        "(the campaign also tests 09:45, 10:00, and 10:30 deadlines).",
        "",
        "## Experiment families and trial counts",
        "",
        counts,
        "",
        "| family | trials | best J | best mean return | hit rate |",
        "|---|---:|---:|---:|---:|",
        *best,
        "",
        "## Phase 0 08:00 retest",
        "",
        "The pre-change result files were copied to `results/_backup_pre08/` before regeneration. "
        "The exact comparison is in `results/exit/old_vs_new_baseline.csv`; the headline rows are "
        "included below.",
    ]
    comparison = OUT / "old_vs_new_baseline.csv"
    if comparison.exists():
        c = pd.read_csv(comparison)
        lines += ["", "| period | version | trades | mean return | PF | total | max DD |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for _, row in c.iterrows():
            lines.append(f"| {row['period']} | {row['version']} | {int(row['trades'])} | "
                         f"{row['mean%'] / 100:.4f} | {row['PF']:.2f} | {row['total%']:.1f}% | "
                         f"{row['maxDD%']:.1f}% |")
    lines += [
        "",
        "## Oracle diagnostic",
        "",
        f"Oracle rows: {len(oracle)}; mean realistic oracle return: "
        f"{oracle['ret'].mean():.4f}" if len(oracle) else "Oracle unavailable",
        "",
        "The oracle is hindsight only and is not an executable strategy. It bounds achievable "
        "capture and must not be used for selection.",
        "",
        "Equity and representative winner/loser path figures are in `reports/exit/figures/`.",
        "",
        "The `learned` and `optimal_stopping` families are causal feature-rule ablations "
        "with walk-forward-compatible parameter search; they are deliberately reported "
        "separately from a production ML/Longstaff-Schwartz model rather than overstating "
        "a model that was not fit in this compact campaign.",
        "",
        "## Finalists and locked test",
        "",
        f"Finalists were pre-registered in `reports/exit/finalists.json` before the locked test was evaluated. "
        f"Elapsed wall time: {time.time() - start:.0f} seconds.",
        "",
        "| finalist | dev mean | test mean | test hit rate | test PF |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in finalists:
        d = _stats(dev_result[name])
        t = _stats(test_result[name])
        lines.append(f"| {name} | {d['mean_ret']:.4f} | {t['mean_ret']:.4f} | {t['hit_rate']:.3f} | {t['pf']:.2f} |")
    base_dev = pd.read_csv(TABLES / "baseline_dev_realistic.csv")
    base_test = pd.read_csv(TABLES / "baseline_test_realistic.csv")
    boot = []
    for name in finalists:
        boot.append({"finalist": name, "period": "dev", **_paired_bootstrap(base_dev, dev_result[name])})
        boot.append({"finalist": name, "period": "test", **_paired_bootstrap(base_test, test_result[name])})
    pd.DataFrame(boot).to_csv(TABLES / "paired_bootstrap_improvement.csv", index=False)
    lines += ["", "Paired bootstrap (by exit day) improvement over the realistic baseline; "
              "the test interval is descriptive because the finalists were pre-registered:", "",
              "| finalist | period | mean improvement | 95% low | 95% high | days |",
              "|---|---|---:|---:|---:|---:|"]
    for row in boot:
        lines.append(f"| {row['finalist']} | {row['period']} | {row['mean']:.4f} | {row['lo']:.4f} | "
                     f"{row['hi']:.4f} | {row['days']} |")
    lines += [
        "",
        "## Limitations and sources",
        "",
        "The locked S1/H3 subsets are small and therefore weak evidence. The test entry set "
        "has been read previously, although these exit parameters were selected on dev only. "
        "Volume participation and next-bar execution assumptions can materially change results; "
        "legacy trigger-close numbers are especially optimistic under fast price movement. "
        "Conditional rules receive a multiple-testing warning because they inspect many entry features.",
        "",
        "Sources: Bailey et al., *The Probability of Backtest Overfitting* (2014); "
        "Longstaff and Schwartz, *Valuing American Options by Simulation* (2001); "
        "Kaufman, *New Trading Systems and Methods* (2005); Alpaca market-data bar documentation.",
    ]
    (ROOT / "reports" / "exit_testing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=5.5)
    a = ap.parse_args()
    start = _read_start()
    deadline = a.hours * 3600
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    require_valid_pit_data()
    kw = dict(parse_dates=["entry_time"], keep_default_na=False, na_values=[""])  # ticker "NAN" is not NaN
    dev = pd.read_csv(ROOT / "results" / "pit_trades_atr_pivot_lb13_pm_target25_v2.csv", **kw)
    test = pd.read_csv(ROOT / "results" / "holdout" / "pit_trades_atr_pivot_lb13_pm_target25_v2.csv", **kw)
    all_trades = pd.concat([dev.assign(period="dev"), test.assign(period="test")], ignore_index=True)
    cache = OUT / "windows.npz"
    windows = load_windows(cache) if cache.exists() else build_windows(all_trades, ROOT / "data" / "pit" / "alpaca_minute", cache, print)
    dev_windows = [w for w in windows if w.trade_id < len(dev)]
    test_windows = [w for w in windows if w.trade_id >= len(dev)]
    elapsed_logger(start, str(OUT / "elapsed.jsonl"), "windows_ready")
    families = {}
    rng = np.random.default_rng(20261020)
    baseline = baseline_policy()
    base_dev_legacy = simulate(dev_windows, baseline, "legacy")
    base_dev_realistic = simulate(dev_windows, baseline, "realistic")
    base_test_legacy = simulate(test_windows, baseline, "legacy")
    base_test_realistic = simulate(test_windows, baseline, "realistic")
    for label, table in (
        ("baseline_dev_legacy", base_dev_legacy), ("baseline_dev_realistic", base_dev_realistic),
        ("baseline_test_legacy", base_test_legacy), ("baseline_test_realistic", base_test_realistic),
    ):
        table.to_csv(TABLES / f"{label}.csv", index=False)
    pd.DataFrame([{"fill_model": "legacy", **_stats(base_dev_legacy)},
                  {"fill_model": "realistic", **_stats(base_dev_realistic)}]).to_csv(
                      TABLES / "baseline_fill_sensitivity_dev.csv", index=False)
    for family in ("fixed_time", "trailing", "momentum", "scale_out", "pm_ah", "combos", "learned", "optimal_stopping"):
        if time.time() - start >= deadline:
            break
        subset = dev.iloc[[w.trade_id for w in dev_windows]].reset_index(drop=True)
        if family in ("learned", "optimal_stopping"):
            families[family] = evaluate_model_family(family, dev_windows, dev, start, deadline, rng)
        else:
            families[family] = evaluate_family(family, dev_windows, subset, start, deadline, rng)
    batch = 0
    while time.time() - start < deadline:
        family = f"robustness_{batch:02d}"
        families[family] = evaluate_family(
            family, dev_windows, dev.iloc[[w.trade_id for w in dev_windows]], start, deadline, rng
        )
        batch += 1
    candidates = []
    for family, table in families.items():
        if family in ("learned", "optimal_stopping"):
            continue
        if len(table):
            for _, row in table[table["valid"]].sort_values("score", ascending=False).head(3).iterrows():
                candidates.append((f"{family}_{int(row['trial'])}", json.loads(row["params"])))
    if not candidates:
        candidates = [("baseline", baseline)]
    finalists = dict(candidates[:3])
    (ROOT / "reports" / "exit").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "exit" / "finalists.json").write_text(
        json.dumps({"timestamp_utc": pd.Timestamp.utcnow().isoformat(), "finalists": finalists}, indent=2), encoding="utf-8")
    dev_result, test_result = {}, {}
    for name, policy in finalists.items():
        dev_result[name] = simulate(dev_windows, policy, "realistic")
        test_result[name] = simulate(test_windows, policy, "realistic")
    oracle = oracle_returns(dev_windows, "realistic")
    oracle.to_csv(TABLES / "oracle_dev.csv", index=False)
    _plot_finalist_figures(dev_windows, finalists, dev_result)
    write_report(start, all_trades, dev_result, test_result, families, oracle, finalists)
    elapsed_logger(start, str(OUT / "elapsed.jsonl"), "campaign_complete")
    print(f"campaign complete; elapsed={time.time() - start:.0f}s; finalists={list(finalists)}")


if __name__ == "__main__":
    raise SystemExit("exit_campaign.main() is superseded (pre-fix simulator, invalid results). "
                     "Use scripts/exit_rerun_v2.py; helpers (_score, _masks, ...) are still imported from here.")
