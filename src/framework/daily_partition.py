"""Daily-bar partition test (trend / random-pick / bias / skill) for multi-session equal-weight books.

Definitions follow src/backtest/partition.py (total = trend + skill + bias, skill is the residual) adapted to daily bars:

  account  every open position gets weight 1 / max(open positions, slots) each session (daily mark-to-market);
           the rest is cash, so exposure is open/slots until the book is full. Trade costs are charged on the entry/exit session, weighted by position weight.
  total    sum over the period's sessions of log(1 + account return).
  trend    sum over sessions of fraction_invested * log(1 + benchmark return); the benchmark is the equal-weight
           return of the tradable universe built from the same data (no index series is required).
  random   mean total of `n_draws` seeded books where every real trade is replaced by a random name from
           (universe) the tradable names, or (pool) the strategy's pre-ranking pool, on the same signal day with the
           same entry/exit sessions.
  bias     0 on the locked test period; on dev the expected-maximum haircut for `n_trials` variants tried.
  skill    total - trend - bias (and vs. random); skill_pct = skill / total when both are >= 0.
"""

import numpy as np
import pandas as pd

from src.backtest import partition as P
from src.utils import timing

SEED = P.SEED


def expand_positions(entry_idx, exit_idx):
    """Position-session rows: (trade index, session row, is_first, is_last)."""
    entry_idx, exit_idx = np.asarray(entry_idx, np.int64), np.asarray(exit_idx, np.int64)
    lengths = exit_idx - entry_idx + 1
    trade = np.repeat(np.arange(len(entry_idx)), lengths)
    offset = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    day = entry_idx[trade] + offset
    return trade, day, offset == 0, day == exit_idx[trade]


def account_series(expansion, cols, entry_px, exit_px, close_ff, cost, n_rows, slots=1, trade_weights=None,
                   entry_cost_weights=None, exit_cost_weights=None):
    """Daily account returns; optional signed trade weights express fixed portfolio fractions."""
    trade, day, first, last = expansion
    c = cols[trade]
    count = np.bincount(day, minlength=n_rows).astype(float)
    start = np.where(first, entry_px[trade], close_ff[np.maximum(day - 1, 0), c])
    end = np.where(last, exit_px[trade], close_ff[day, c])
    ret = end / start - 1.0
    if trade_weights is None:
        start = np.where(first, start * (1 + cost), start)
        end = np.where(last, end * (1 - cost), end)
        ret = end / start - 1.0
        sums = np.bincount(day, ret, minlength=n_rows)
        r_day = sums / np.maximum(count, slots)
    else:
        weights = np.asarray(trade_weights, dtype=float)[trade]
        entry_fees = np.abs(trade_weights) if entry_cost_weights is None else np.asarray(entry_cost_weights, dtype=float)
        exit_fees = np.abs(trade_weights) if exit_cost_weights is None else np.asarray(exit_cost_weights, dtype=float)
        pnl = weights * ret - cost * (
            entry_fees[trade] * first.astype(float) + exit_fees[trade] * last.astype(float)
        )
        r_day = np.bincount(day, pnl, minlength=n_rows)
    return r_day, count


def candidate_matrix(mask):
    """Padded per-row candidate column indices of a boolean (date x symbol) mask and their counts."""
    count = mask.sum(1)
    width = max(int(count.max()), 1)
    order = np.argsort(~mask, axis=1, kind="stable")[:, :width]
    return order.astype(np.int32), count


def random_books(expansion, real_cols, signal_idx, entry_idx, exit_idx, cand, count, entry_mat, exit_mat,
                 close_ff, cost, n_rows, masks, n_draws, seed, slots=1, trade_weights=None, fixed_mask=None,
                 entry_cost_weights=None, exit_cost_weights=None):
    """Totals per draw and period, plus the mean per-session log return across draws."""
    rng = np.random.default_rng(seed)
    totals = np.zeros((n_draws, len(masks)))
    mean_log = np.zeros(n_rows)
    n = len(real_cols)
    cnt = count[signal_idx]
    fixed = np.asarray(fixed_mask, dtype=bool) if fixed_mask is not None else np.zeros(n, dtype=bool)
    for k in range(n_draws):
        cols = cand[signal_idx, (rng.random(n) * cnt).astype(int)].astype(np.int64)
        cols[fixed] = real_cols[fixed]
        for _ in range(8):
            bad = np.isnan(entry_mat[entry_idx, cols]) | np.isnan(exit_mat[exit_idx, cols])
            if not bad.any():
                break
            cols[bad] = cand[signal_idx[bad], (rng.random(int(bad.sum())) * cnt[bad]).astype(int)]
            cols[fixed] = real_cols[fixed]
        bad = np.isnan(entry_mat[entry_idx, cols]) | np.isnan(exit_mat[exit_idx, cols])
        cols[bad] = real_cols[bad]
        r_day, _ = account_series(expansion, cols, entry_mat[entry_idx, cols], exit_mat[exit_idx, cols],
                                  close_ff, cost, n_rows, slots, trade_weights=trade_weights,
                                  entry_cost_weights=entry_cost_weights, exit_cost_weights=exit_cost_weights)
        log_r = np.log1p(r_day)
        mean_log += log_r
        for m, mask in enumerate(masks):
            totals[k, m] = log_r[mask].sum()
    return totals, mean_log / n_draws


def _ols(y, x):
    if len(x) < 5 or np.ptp(x) == 0:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    se = np.sqrt(res @ res / (len(y) - 2) * np.linalg.inv(X.T @ X)[1, 1])
    return float(b[1]), float(b[1] / se) if se > 0 else np.nan


def _perf(r, trade_ret, hold):
    equity = np.cumprod(1 + r)
    dd = float((equity / np.maximum.accumulate(equity) - 1).min()) if len(r) else np.nan
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    wins, losses = trade_ret[trade_ret > 0].sum(), -trade_ret[trade_ret <= 0].sum()
    return {"win_pct": 100 * float((trade_ret > 0).mean()) if len(trade_ret) else np.nan,
            "mean_ret_pct": 100 * float(trade_ret.mean()) if len(trade_ret) else np.nan,
            "median_ret_pct": 100 * float(np.median(trade_ret)) if len(trade_ret) else np.nan,
            "pf": float(wins / losses) if losses > 0 else np.nan,
            "cagr_pct": 100 * float(np.expm1(np.log1p(r).sum() * 252 / max(len(r), 1))),
            "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd and sd > 0 else np.nan,
            "max_dd_pct": 100 * dd, "avg_hold": float(hold.mean()) if len(hold) else np.nan}


def partition_period(r_day, f_day, count, b_day, mask, trades, random, bias, r_day_ex5, n_boot, seed, entry_w, exit_w):
    """One period's partition result. `random` maps name -> (totals_for_period, mean_log_day)."""
    r, f, b, n_days = r_day[mask], f_day[mask], b_day[mask], int(mask.sum())
    log_r, log_b = np.log1p(r), np.log1p(b)
    total, trend = float(log_r.sum()), float((f * log_b).sum())
    inv = f > 0
    beta, beta_t = _ols(log_r[inv], log_b[inv])
    bias_value = bias["value"]
    out = {"days": n_days, "trades": len(trades), "total_log": total, "trend": trend,
           "trend_beta": float(beta * log_b[inv].sum()) if np.isfinite(beta) else np.nan, "beta": beta, "beta_t": beta_t,
           "avg_fraction_invested": float(f[inv].mean()) if inv.any() else 0.0,
           "avg_positions": float(count[mask][inv].mean()) if inv.any() else 0.0,
           "turnover_annual": float((entry_w[mask].sum() + exit_w[mask].sum()) * 252 / max(n_days, 1)),
           "bias": bias_value, "bias_src": bias["src"]}
    out.update(_perf(r, trades["ret"].to_numpy(float), trades["hold"].to_numpy(float)))
    for name, (totals, mean_log) in random.items():
        out[f"random_pick_{name}"] = float(totals.mean())
        out[f"random_p_{name}"] = float((np.sum(totals >= total) + 1) / (len(totals) + 1))
        out[f"skill_vs_{name}"] = total - out[f"random_pick_{name}"] - bias_value
    out["skill"] = total - trend - bias_value
    out["skill_pct"] = out["skill"] / total if total > 0 and out["skill"] >= 0 else np.nan
    out["skill_pct_random"] = out["skill_vs_pool"] / total if total > 0 and out["skill_vs_pool"] >= 0 else np.nan
    rng = np.random.default_rng(seed + 1)
    ix = rng.integers(0, n_days, (n_boot, n_days))
    sk = (log_r - f * log_b)[ix].sum(1) - bias_value
    out["skill_lo"], out["skill_hi"] = float(np.quantile(sk, .025)), float(np.quantile(sk, .975))
    out["skill_p"] = float((np.sum(sk <= 0) + 1) / (n_boot + 1))
    out["skill_ex5"] = float(np.log1p(r_day_ex5[mask]).sum() - trend - bias_value)
    return out


@timing.timed("daily partition")
def evaluate_fill(trades, book, masks, universe, pool, n_eff, cfg, seed=SEED, bias_note=""):
    """Partition results for every period in `masks` (name -> bool over rows) for one fill model.

    book: close_ff, entry_mat, exit_mat, cost, n_rows, b_day, slots.
    universe/pool: boolean (date x symbol) candidate masks for the random-pick baselines.
    Returns (results by period, per-session series dict).
    """
    n_rows, cost, slots = book["n_rows"], book["cost"], int(book.get("slots", 1))
    entry = trades["entry_idx"].to_numpy(np.int64)
    exit_ = trades["exit_idx"].to_numpy(np.int64)
    sym = trades["sym_idx"].to_numpy(np.int64)
    sig = trades["signal_idx"].to_numpy(np.int64)
    trade_weights = trades["trade_weight"].to_numpy(float) if "trade_weight" in trades else None
    fixed_mask = trades["random_fixed"].to_numpy(bool) if "random_fixed" in trades else None
    entry_cost_weights = trades["entry_cost_weight"].to_numpy(float) if "entry_cost_weight" in trades else None
    exit_cost_weights = trades["exit_cost_weight"].to_numpy(float) if "exit_cost_weight" in trades else None
    expansion = expand_positions(entry, exit_)
    with timing.stage("account"):
        r_day, count = account_series(expansion, sym, trades["entry_px"].to_numpy(float), trades["exit_px"].to_numpy(float),
                                      book["close_ff"], cost, n_rows, slots, trade_weights=trade_weights,
                                      entry_cost_weights=entry_cost_weights, exit_cost_weights=exit_cost_weights)
        if trade_weights is None:
            f_day = count / np.maximum(count, slots)
            safe = np.maximum(count, slots)
            entry_w = np.bincount(entry, 1.0 / safe[entry], minlength=n_rows)
            exit_w = np.bincount(exit_, 1.0 / safe[exit_], minlength=n_rows)
        else:
            trade, day, _, _ = expansion
            gross_exposure = np.bincount(day, np.abs(trade_weights[trade]), minlength=n_rows)
            f_day = np.asarray(book.get("trend_exposure", gross_exposure), dtype=float)
            fee_entries = np.abs(trade_weights) if entry_cost_weights is None else entry_cost_weights
            fee_exits = np.abs(trade_weights) if exit_cost_weights is None else exit_cost_weights
            entry_w = np.bincount(entry, fee_entries, minlength=n_rows)
            exit_w = np.bincount(exit_, fee_exits, minlength=n_rows)
    names = list(masks)
    mask_list = [masks[k] for k in names]
    random = {}
    for label, cands, offset in (("universe", universe, 17), ("pool", pool, 0)):
        with timing.stage(f"random picks {label}"):
            cand, cnt = candidate_matrix(cands)
            random[label] = random_books(expansion, sym, sig, entry, exit_, cand, cnt, book["entry_mat"], book["exit_mat"],
                                         book["close_ff"], cost, n_rows, mask_list, int(cfg["n_draws"]), seed + offset, slots,
                                         trade_weights=trade_weights, fixed_mask=fixed_mask,
                                         entry_cost_weights=entry_cost_weights, exit_cost_weights=exit_cost_weights)
    results = {}
    period_of = np.full(n_rows, -1)
    for m, mask in enumerate(mask_list):
        period_of[mask] = m
    trade_period = period_of[entry]
    for m, name in enumerate(names):
        sel = trade_period == m
        bias = _bias(r_day[mask_list[m]], n_eff, name == "test", bias_note)
        top5 = trades.loc[sel, "ret"].nlargest(5).index
        keep = np.ones(len(trades), bool)
        keep[top5.to_numpy()] = False
        with timing.stage("ex5 account"):
            sub = trades[keep]
            sub_weights = sub["trade_weight"].to_numpy(float) if "trade_weight" in sub else None
            r_ex5, _ = account_series(expand_positions(sub["entry_idx"], sub["exit_idx"]), sub["sym_idx"].to_numpy(np.int64),
                                      sub["entry_px"].to_numpy(float), sub["exit_px"].to_numpy(float), book["close_ff"],
                                      cost, n_rows, slots, trade_weights=sub_weights,
                                      entry_cost_weights=sub["entry_cost_weight"].to_numpy(float) if "entry_cost_weight" in sub else None,
                                      exit_cost_weights=sub["exit_cost_weight"].to_numpy(float) if "exit_cost_weight" in sub else None)
        rnd = {k: (v[0][:, m], v[1]) for k, v in random.items()}
        with timing.stage("bootstrap"):
            results[name] = partition_period(r_day, f_day, count, book["b_day"], mask_list[m], trades[sel], rnd, bias, r_ex5,
                                             int(cfg["n_boot"]), seed, entry_w, exit_w)
    series = {"r_day": r_day, "f_day": f_day, "count": count, "random_mean_log": {k: v[1] for k, v in random.items()}}
    return results, series


def _bias(r_period, n_eff, oos, note=""):
    if oos:
        return {"value": 0.0, "src": "0: out-of-sample by construction (parameters fixed from the catalog, no tuning)"}
    sd = float(np.log1p(r_period).std(ddof=1) * np.sqrt(len(r_period)))
    value = P.expected_max_noise(sd, n_eff)
    return {"value": value, "src": f"B haircut: expected max of N_eff={n_eff:.2f} comparable variants, sd={sd:.4f}{'; ' + note if note else ''}"}


def results_frame(results_by_fill):
    rows = []
    for fill, by_period in results_by_fill.items():
        for period, res in by_period.items():
            rows.append({"fill": fill, "period": period, **res})
    return pd.DataFrame(rows)
