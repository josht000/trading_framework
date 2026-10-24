"""
Partition test: total log return = trend + skill + bias  (skill is the residual).

Definitions (all in log-return units of the $5,000 / 5 x 20% account from pit_metrics.simulate_account):
  total   sum over trading days of log(1 + daily account return); days without a position add 0.
  trend   what market drift alone gave: sum_t fraction_invested_t * benchmark log return over the day's holding
          window. fraction_invested_t = sum of position fractions that day (gross value / equity, <= 1.0).
          The window of a trade runs from its 15:55 entry bar to its actual exit time; the benchmark price is the
          close of the last 1-minute bar at or before that time (data/pit/market/<SYM>_minute.parquet, falling back
          to daily close/open when no bar within 30 minutes). Per day the windows are weighted by position size.
          trend_spy / trend_iwm use SPY / IWM; trend_beta = beta * sum(window return) over invested days with beta
          from OLS of daily strategy log return on the daily window return (also reported with its t-stat).
          Headline trend = trend_iwm (small-cap beta).
  random  random_pick_universe and random_pick_pool are means over 2,000 seeded draws from the
          pre-filter universe and strategy-filtered pre-ranking pool respectively. Days with no real
          choice are counted. Their p-values compare each draw total to the observed total.
  bias    estimate of selection / optimisation inflation inside `total`:
          A  walk_forward_bias(): nested expanding-window procedure (exact for a tuning procedure that can be re-run):
             bias = (mean in-sample per-day log return of the params chosen on earlier blocks, minus the mean
             out-of-sample per-day return of the next block) * days in the period.
          B  expected_max_noise(sd_trial, N_eff): Bailey / Lopez de Prado expected maximum of comparable
          account-total statistics in the trial registry. One pooled sd and effective N are used across
          areas; correlated trial series reduce N_eff. It is flagged unreliable when larger than observed total.
          Headline: max(A, pooled B), floored at 0 for subtraction while signed estimate is retained; B alone if no nested procedure exists; no search => 0 with that stated; if a needed
          trial count is unknown the haircut covers the known areas only and is labelled a lower bound (n/a if none is known). On the locked test period the bias is 0 by
          trial count is unknown the haircut covers the known areas only and is labelled a lower bound (n/a if none is known). On the locked test period the bias is 0 by
          construction (configuration chosen on dev only), flagged "oos".
  skill   skill_vs_trend  = total - trend - bias (slide definition);  skill_vs_random = total - random_pick_log - bias.
          skill_pct = skill / total only when skill >= 0 and total > 0 (else n/a). Significance: 5,000-draw day-bootstrap of the
          trading days (same trend/bias estimates) for the 95% CI and a one-sided p for skill > 0. Skill is a residual:
          it also contains luck from a few large winners, so skill_ex5 recomputes it without the 5 best trades.
  cost    informational: commission 0.1%/side plus slippage constants (0.15% regular / 0.40% outside 09:30-16:00 per
          side) per trade, weighted by position fraction. Skill is net of costs (trade returns are already net).
"""

import json
import os
from math import e, sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.backtest.pit_metrics import select_trades, simulate_account
from src.data.pit_integrity import dedupe_trades
from src.utils import timing

REGISTRY = os.path.join("results", "trial_registry.jsonl")
MARKET = os.path.join("data", "pit", "market")
NY = "America/New_York"
SEED = 20261101
EULER = 0.5772156649
COMMISSION, SLIP_RTH, SLIP_EXT = 0.001, 0.0015, 0.004
_MKT = {}


# ---------------------------------------------------------------- benchmark windows
def _market(sym):
    if sym not in _MKT:
        m = pd.read_parquet(os.path.join(MARKET, f"{sym}_minute.parquet"), columns=["timestamp", "close"])
        m = m.assign(t=pd.to_datetime(m["timestamp"], utc=True).astype("int64")).sort_values("t").drop_duplicates("t")
        d = pd.read_parquet(os.path.join(MARKET, f"{sym}_daily.parquet")).set_index("date")
        d.index = pd.to_datetime(d.index)
        _MKT[sym] = (m["t"].to_numpy(), m["close"].to_numpy(float), d)
    return _MKT[sym]


def _price_at(sym, ts, kind):
    """Benchmark price at each timestamp: close of the last minute bar <= ts (within 30 min), else daily fallback."""
    t, c, d = _market(sym)
    x = ts.astype("int64").to_numpy()
    i = np.searchsorted(t, x, side="right") - 1
    ok = (i >= 0) & (x - t[np.clip(i, 0, None)] <= 30 * 60 * 10**9)
    px = np.where(ok, c[np.clip(i, 0, None)], np.nan)
    if np.isnan(px).any():
        day = ts.tz_convert(NY).tz_localize(None).normalize()
        col = "close" if kind == "entry" else "open"
        fb = d[col].reindex(day).to_numpy(float)
        px = np.where(np.isnan(px), fb, px)
    return px


def window_returns(trades, sym):
    """Benchmark log return per trade from entry_time to exit_time (0 and flagged NaN count if unavailable)."""
    en = pd.to_datetime(trades["entry_time"].astype(str), utc=True)
    ex = pd.to_datetime(trades["exit_time"].astype(str), utc=True, errors="coerce")
    ok = ex.notna().to_numpy()
    out = np.full(len(trades), np.nan)
    if ok.any():
        out[ok] = np.log(_price_at(sym, ex[ok], "exit") / _price_at(sym, en[ok], "entry"))
    return pd.Series(out, index=trades.index)


# ---------------------------------------------------------------- bias
def expected_max_noise(sd_trial, n_trials):
    """Expected maximum total log return of N zero-skill trials (Bailey & Lopez de Prado)."""
    if n_trials is None or n_trials < 2 or not np.isfinite(sd_trial):
        return 0.0
    return float(sd_trial * ((1 - EULER) * norm.ppf(1 - 1 / n_trials) + EULER * norm.ppf(1 - 1 / (n_trials * e))))


def walk_forward_bias(trades, tune_fn, evaluate_fn, folds=5, embargo_days=2, n_days_period=None, day_col="day"):
    """
    Nested expanding-window estimate of the optimisation inflation of a tuning procedure.
    tune_fn(train_trades) -> params (None if it cannot tune); evaluate_fn(trades, params, days) -> total log return
    over `days` (a DatetimeIndex of business days, zero-trade days included). Days are split into `folds`
    chronological blocks; block k>0 is tested with params tuned on blocks < k minus `embargo_days` before it.
    Returns the signed bias over `n_days_period` days (default: all days) plus per-fold detail.
    """
    d = pd.to_datetime(trades[day_col])
    days = pd.bdate_range(d.min(), d.max())
    blocks = np.array_split(np.arange(len(days)), folds)
    rows = []
    for k in range(1, folds):
        te = days[blocks[k]]
        tr = days[: blocks[k][0] - embargo_days]
        if len(tr) < 20:
            continue
        p = tune_fn(trades[d.isin(tr)])
        if p is None:
            continue
        is_ = evaluate_fn(trades[d.isin(tr)], p, tr)
        oos = evaluate_fn(trades[d.isin(te)], p, te)
        rows.append({"fold": k, "train_days": len(tr), "test_days": len(te), "is_per_day": is_ / len(tr), "oos_per_day": oos / len(te), "params": p})
    if not rows:
        return {"bias": np.nan, "per_day": np.nan, "folds": 0, "detail": pd.DataFrame()}
    df = pd.DataFrame(rows)
    per_day = float(df["is_per_day"].mean() - df["oos_per_day"].mean())
    return {"bias": per_day * (n_days_period if n_days_period is not None else len(days)), "per_day": per_day,
            "is_per_day": float(df["is_per_day"].mean()), "oos_per_day": float(df["oos_per_day"].mean()), "folds": len(df), "detail": df}


# ---------------------------------------------------------------- registry
def log_trials(project_area, n_trials, per_trial_total_log=None, notes="", estimated=False, invalid=False,
               days=None, source=None, stat_kind="account_total_log", path=REGISTRY):
    """Append one line to the trial registry. per_trial_total_log: total log return of each trial over `days` business days."""
    rec = {"ts": pd.Timestamp.now().isoformat(timespec="seconds"), "area": project_area, "n_trials": None if n_trials is None else int(n_trials),
           "per_trial_total_log": None if per_trial_total_log is None else [round(float(x), 6) for x in per_trial_total_log if np.isfinite(x)],
           "days": None if days is None else int(days), "stat_kind": stat_kind, "estimated": bool(estimated),
           "invalid": bool(invalid), "source": source, "notes": notes}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load_registry(path=REGISTRY):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ts", "area", "n_trials", "per_trial_total_log", "days", "estimated", "invalid", "source", "notes"])
    return pd.DataFrame([json.loads(x) for x in open(path, encoding="utf-8") if x.strip()])


def haircut_B(areas, n_days, sd_fallback=None, registry=None):
    """Estimate one pooled expected-maximum haircut for comparable trial totals."""
    reg = load_registry() if registry is None else registry
    values, unknown, raw_n = [], [], 0
    series = []
    by_area = {}
    for a in areas:
        r = reg[(reg["area"] == a) & (~reg["invalid"].astype(bool))] if len(reg) else reg
        if len(r) == 0 or r["n_trials"].isna().all():
            unknown.append(a)
            continue
        area_vals = []
        for x, row_days in zip(r["per_trial_total_log"], r["days"]):
            # Do not mix train-segment totals, different periods, or different
            # account horizons with the reported period.
            if isinstance(x, list) and len(x) and pd.notna(row_days) and int(row_days) == int(n_days):
                area_vals.extend(np.asarray(x, float))
        if area_vals:
            values.extend(area_vals)
            raw_n += len(area_vals)
            by_area[a] = {"N_raw": len(area_vals)}
        else:
            unknown.append(a)
    v = np.asarray(values, float)
    sd = float(np.std(v, ddof=1)) if len(v) >= 10 else (sd_fallback if sd_fallback is not None else np.nan)
    sd_src = "registry comparable trial totals" if len(v) >= 10 else "independent-trials upper bound"
    rho = np.nan
    n_eff = raw_n
    # A future registry may provide aligned per-trial series; use average pairwise
    # correlation only when there are at least two comparable series.
    if len(series) >= 2:
        rho = float(np.nanmean([np.corrcoef(series[i], series[j])[0, 1]
                                for i in range(len(series)) for j in range(i)]))
        n_eff = max(1.0, 1.0 + (raw_n - 1.0) * (1.0 - np.clip(rho, -1.0, 1.0)))
    value = expected_max_noise(sd, n_eff) if raw_n and np.isfinite(sd) else np.nan
    return {"value": value, "value_known": value, "signed": value, "N": raw_n,
            "N_raw": raw_n, "N_eff": n_eff,
            "N_eff_src": "correlation-adjusted series" if len(series) >= 2 else "raw (no aligned trial series)",
            "sd": sd, "sd_src": sd_src,
            "rho_avg": rho, "by_area": by_area, "unknown": unknown}


def bias_estimate(areas, n_days, A=None, A_areas=(), sd_fallback=None, oos=False, registry=None, total_log=None):
    """Headline bias per the reporting rule. Returns dict(value, A, B, B_known, N, src, by_area)."""
    if oos:
        return {"value": 0.0, "signed": 0.0, "A": None, "B": None, "N": None, "N_eff": None, "sd": None,
                "src": "0: out-of-sample by construction (config chosen on dev)", "by_area": {}}
    rest = [a for a in areas if a not in A_areas]
    B = haircut_B(areas, n_days, sd_fallback, registry) if areas else None
    Brest = haircut_B(rest, n_days, sd_fallback, registry) if rest else None
    a_used = max(A, 0.0) if A is not None and np.isfinite(A) else None
    b_used = Brest["value"] if Brest and np.isfinite(Brest["value"]) else None
    estimate = max(x for x in (a_used, b_used) if x is not None) if (a_used is not None or b_used is not None) else np.nan
    unreliable = bool(total_log is not None and np.isfinite(estimate) and abs(total_log) > 0 and estimate > abs(total_log))
    signed = estimate
    common = {"signed": signed, "unreliable": unreliable, "N_eff": Brest.get("N_eff") if Brest else None,
              "N_eff_src": Brest.get("N_eff_src") if Brest else None,
              "sd": Brest.get("sd") if Brest else None}
    if a_used is not None:
        if Brest is not None and Brest["unknown"]:
            src = f"max(A,B partial): unknown trial count for {','.join(Brest['unknown'])}"
        else:
            src = "max(A walk-forward, B pooled)"
        return {"value": estimate, "A": A, "B": Brest.get("value") if Brest else None,
                "N": Brest.get("N") if Brest else None, "src": src,
                "by_area": B["by_area"] if B else {}, **common}
    if B is None:
        return {"value": 0.0, "A": None, "B": None, "N": 0, "src": "0: no search behind this configuration",
                "by_area": {}, "signed": 0.0, "unreliable": False, "N_eff": 0, "sd": np.nan}
    if B["unknown"]:
        if not B["N"]:
            return {"value": np.nan, "A": None, "B": None, "N": 0, "src": f"n/a: trial count unknown for {','.join(B['unknown'])}", "by_area": {}, **common}
        return {"value": B["value"], "A": None, "B": B["value"], "N": B["N"], "src": f"B partial: trial count unknown for {','.join(B['unknown'])}", "by_area": B["by_area"], **common}
    return {"value": B["value"], "A": None, "B": B["value"], "N": B["N"], "src": "B pooled haircut", "by_area": B["by_area"], **common}


# ---------------------------------------------------------------- partition
def _ols(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    if len(x) < 5 or np.ptp(x) == 0:
        return np.nan, np.nan, np.nan
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    s2 = res @ res / (len(y) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return float(b[0]), float(b[1]), float(b[1] / se) if se > 0 else np.nan


def _cost(sel, frac):
    ex = pd.to_datetime(sel["exit_time"].astype(str), utc=True, errors="coerce").dt.tz_convert(NY)
    mins = ex.dt.hour * 60 + ex.dt.minute
    ext = ~((mins >= 570) & (mins < 960)).to_numpy()
    return float(((2 * COMMISSION + SLIP_RTH + np.where(ext, SLIP_EXT, SLIP_RTH)) * frac).sum())


def random_pick(pool, k_by_day, days, max_pos=5, frac=0.2, n_draws=2000, seed=SEED):
    """Per-day mean of the random-pick daily log returns, the n_draws totals, and the per-day draw matrix."""
    p = dedupe_trades(pool).copy()
    p["day"] = pd.to_datetime(p["entry_time"].astype(str).str[:10])
    rng = np.random.default_rng(seed)
    mat = np.zeros((n_draws, len(days)))
    pos = {d: i for i, d in enumerate(days)}
    for d, g in p.groupby("day"):
        k = int(k_by_day.get(d, 0))
        if k == 0 or d not in pos:
            continue
        r = g["ret"].to_numpy(float)
        if len(r) <= k:
            s = np.full(n_draws, r.sum())
        else:
            idx = np.argpartition(rng.random((n_draws, len(r))), k - 1, axis=1)[:, :k]
            s = r[idx].sum(1)
        mat[:, pos[d]] = np.log1p(frac * s)
    return mat


def partition(trades, days, pool=None, universe=None, rank="rel_vol", bias=None, max_pos=5, frac=0.2, n_draws=2000, n_boot=5000, seed=SEED):
    """
    Partition test for the trades table `trades` (columns ret, entry_time, exit_time, rank column) over business days `days`.
    pool: candidate rows before ranking (default: trades). bias: dict from bias_estimate (value NaN => skill n/a).
    Returns a flat dict of numbers plus `_series` (per-day arrays for plots/tests).
    """
    sel = select_trades(trades, max_pos, rank).copy()
    sel["frac"] = frac
    for s in ("SPY", "IWM"):
        sel[f"b_{s}"] = window_returns(sel, s)
    n_nan = int(sel["b_IWM"].isna().sum())
    sel[["b_SPY", "b_IWM"]] = sel[["b_SPY", "b_IWM"]].fillna(0.0)
    g = sel.groupby("day")
    r_day = np.log1p((sel["frac"] * sel["ret"]).groupby(sel["day"]).sum()).reindex(days, fill_value=0.0)
    f_day = sel["frac"].groupby(sel["day"]).sum().reindex(days, fill_value=0.0)
    tr = {s: (sel["frac"] * sel[f"b_{s}"]).groupby(sel["day"]).sum().reindex(days, fill_value=0.0)
          for s in ("SPY", "IWM")}
    total = float(r_day.sum())
    out = {"days": len(days), "trades": len(sel), "total_log": total, "avg_fraction_invested": float(f_day[f_day > 0].mean()) if (f_day > 0).any() else 0.0,
           "trend_spy": float(tr["SPY"].sum()), "trend_iwm": float(tr["IWM"].sum()), "window_missing": n_nan, "cost_est": _cost(sel, frac)}
    inv = f_day > 0
    for s in ("SPY", "IWM"):
        bw = (tr[s] / f_day.where(inv)).dropna()
        a, b, t = _ols(r_day[inv].to_numpy(), bw.reindex(r_day[inv].index).to_numpy())
        out.update({f"beta_{s.lower()}": b, f"beta_t_{s.lower()}": t, f"alpha_day_{s.lower()}": a, f"trend_beta_{s.lower()}": b * float(bw.sum())})
    k_by_day = g.size().to_dict()
    pool = trades if pool is None else pool
    universe = pool if universe is None else universe
    mat_pool = random_pick(pool, k_by_day, days, max_pos, frac, n_draws, seed)
    mat_universe = random_pick(universe, k_by_day, days, max_pos, frac, n_draws, seed + 17)
    rp_pool, rp_universe = mat_pool.sum(1), mat_universe.sum(1)
    rp_day_pool, rp_day_universe = mat_pool.mean(0), mat_universe.mean(0)
    def random_stats(x):
        return float(x.mean()), float((np.sum(x >= total) + 1) / (n_draws + 1))
    ru, pu = random_stats(rp_universe)
    rp, pp = random_stats(rp_pool)
    pool_days = pd.to_datetime(pool["entry_time"].astype(str).str[:10]).value_counts()
    universe_days = pd.to_datetime(universe["entry_time"].astype(str).str[:10]).value_counts()
    choice_days_pool = int(sum(pool_days.get(d, 0) <= k for d, k in k_by_day.items()))
    choice_days_universe = int(sum(universe_days.get(d, 0) <= k for d, k in k_by_day.items()))
    out.update({"random_pick": rp, "random_pick_pool": rp, "random_pick_universe": ru,
                "random_p": pp, "random_p_pool": pp, "random_p_universe": pu,
                "random_lo": float(np.quantile(rp_pool, .025)), "random_hi": float(np.quantile(rp_pool, .975)),
                "random_lo_universe": float(np.quantile(rp_universe, .025)), "random_hi_universe": float(np.quantile(rp_universe, .975)),
                "days_no_choice_pool": choice_days_pool, "days_no_choice_universe": choice_days_universe,
                "selection_skill": total - rp, "skill_vs_pool": total - rp,
                "skill_vs_universe": total - ru})
    bias = bias or {"value": np.nan, "src": "n/a: not computed"}
    b = bias["value"]
    out.update({"bias": b, "bias_A": bias.get("A"), "bias_B": bias.get("B"),
                "bias_N": bias.get("N"), "bias_N_eff": bias.get("N_eff"),
                "bias_sd": bias.get("sd"), "bias_signed": bias.get("signed"),
                "bias_unreliable": bias.get("unreliable", False), "bias_src": bias["src"]})
    out["skill"] = total - out["trend_iwm"] - b
    out["skill_vs_random"] = total - out["random_pick"] - b
    out["skill_pct"] = out["skill"] / total if total > 0 and out["skill"] >= 0 and np.isfinite(b) else np.nan
    out["skill_pct_random"] = out["skill_vs_pool"] / total if total > 0 and out["skill_vs_pool"] >= 0 and np.isfinite(b) else np.nan
    out["skill_pct_note"] = "" if total > 0 and out["skill"] >= 0 else "skill <= 0 or total <= 0"
    bb = 0.0 if not np.isfinite(b) else b
    rng = np.random.default_rng(seed + 1)
    ix = rng.integers(0, len(days), (n_boot, len(days)))
    for name, d in (("", (r_day - tr["IWM"]).to_numpy()), ("_vs_random", (r_day.to_numpy() - rp_day_pool))):
        sk = d[ix].sum(1) - bb
        out[f"skill{name}_lo"], out[f"skill{name}_hi"] = float(np.quantile(sk, .025)), float(np.quantile(sk, .975))
        out[f"skill{name}_p"] = float((np.sum(sk <= 0) + 1) / (n_boot + 1))
    keep = sel.drop(sel["ret"].nlargest(5).index)
    kg = keep.groupby("day")
    r5 = np.log1p((keep["frac"] * keep["ret"]).groupby(keep["day"]).sum()).reindex(days, fill_value=0.0)
    t5 = (keep["frac"] * keep["b_IWM"]).groupby(keep["day"]).sum().reindex(days, fill_value=0.0)
    out["skill_ex5"] = float(r5.sum() - t5.sum() - bb)
    out["_series"] = {"r_day": r_day, "trend_iwm_day": tr["IWM"], "random_totals": rp_pool,
                      "random_totals_pool": rp_pool, "random_totals_universe": rp_universe,
                      "random_day_pool": rp_day_pool, "random_day_universe": rp_day_universe}
    return out


def run_partition(trades, days, rank="rel_vol", pool=None, areas=(), A=None, A_areas=(), oos=False, **kw):
    """partition() with the bias headline from bias_estimate(); oos=True for the locked test period (bias 0 by construction)."""
    with timing.stage("run_partition"):
        dr = simulate_account(trades, max_pos=kw.get("max_pos", 5), rank=rank, all_days=days)[2]
        sd = float(np.log1p(dr).std() * sqrt(len(days)))
        total_log = float(np.log1p(dr).sum())
        return partition(trades, days, pool=pool, rank=rank,
                         bias=bias_estimate(list(areas), len(days), A, A_areas, sd, oos, total_log=total_log),
                         **kw)


# ---------------------------------------------------------------- tables, plots
COLS = ["total_log", "trend_iwm", "trend_spy", "trend_beta_iwm", "beta_iwm", "beta_t_iwm",
        "random_pick_universe", "random_pick_pool", "random_p_universe", "random_p_pool",
        "bias", "bias_A", "bias_B", "bias_N_eff", "bias_sd", "bias_unreliable",
        "skill", "skill_vs_universe", "skill_vs_pool", "skill_pct", "skill_p", "skill_lo", "skill_hi",
        "days_no_choice_pool", "days_no_choice_universe", "skill_ex5"]


def short_cols(res):
    """Compact columns for ranking summaries, including both random baselines."""
    keys = ("total_log", "trend_iwm", "trend_spy", "beta_iwm", "random_pick_universe",
            "random_pick_pool", "random_p_universe", "random_p_pool", "skill",
            "skill_vs_universe", "skill_vs_pool", "bias", "bias_A", "bias_B",
            "bias_N", "bias_N_eff", "bias_sd", "bias_unreliable", "skill_pct",
            "bias_signed", "bias_src", "skill_p", "days_no_choice_pool", "days_no_choice_universe")
    out = {k: res.get(k, np.nan) for k in keys}
    out["random_pick"] = res.get("random_pick_pool", np.nan)
    return out


def flat(res):
    return {k: v for k, v in res.items() if not k.startswith("_")}


def plot_partition(rows, path, labels=None):
    """Stacked bars Trend (blue) / Skill (green) / Bias (red) summing to Total (dot); one panel per row."""
    with timing.stage("plot_partition"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = list(rows)
        fig, axes = plt.subplots(1, len(rows), figsize=(3.6 * len(rows) + 1, 4.6), squeeze=False)
        for ax, r, lab in zip(axes[0], rows, labels or [""] * len(rows)):
            vals = [("Trend", r["trend_iwm"], "#4472c4"), ("Skill", r["skill"], "#70ad47"), ("TrainingBias", r["bias"], "#d9534f")]
            pos = neg = 0.0
            for name, v, c in vals:
                if not np.isfinite(v):
                    continue
                base = pos if v >= 0 else neg
                ax.bar(0, v, bottom=base, color=c, width=0.55, label=name)
                pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
            ax.plot([0], [r["total_log"]], "ko", ms=9, label="Total")
            ax.axhline(0, color="#888888", lw=0.8)
            ax.set_xticks([])
            ax.set_ylabel("log return")
            pct = r["skill_pct"]
            ax.set_title((f"Skill = {100 * pct:.0f}% of TotalReturn" if np.isfinite(pct) else "Skill = n/a % of TotalReturn") + (f"\n{lab}" if lab else ""), fontsize=10)
            ax.grid(axis="y", color="#eeeeee")
        axes[0][0].legend(frameon=True, facecolor="white", edgecolor="#999999", framealpha=1.0, fontsize=8, loc="best").set_zorder(10)
        fig.tight_layout()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)


def fmt_pct(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def print_summary(rows, title="Partition test"):
    """Console summary: rows = list of (label, result dict)."""
    print(f"\n{title}: total log = trend (IWM) + skill + bias; skill_p = day-bootstrap P(skill <= 0)")
    h = f"{'setup':34s} {'total':>8s} {'trend_iwm':>9s} {'trend_spy':>9s} {'random':>8s} {'bias':>8s} {'skill':>8s} {'skill%':>7s} {'p':>6s} {'95% CI':>17s}"
    print(h)
    for lab, r in rows:
        ci = f"[{100 * r['skill_lo']:+.1f},{100 * r['skill_hi']:+.1f}]"
        print(f"{lab[:34]:34s} {fmt_pct(r['total_log']):>8s} {fmt_pct(r['trend_iwm']):>9s} {fmt_pct(r['trend_spy']):>9s} {fmt_pct(r['random_pick']):>8s} "
              f"{fmt_pct(r['bias']):>8s} {fmt_pct(r['skill']):>8s} {fmt_pct(r['skill_pct'], 0):>7s} {r['skill_p']:6.3f} {ci:>17s}")
