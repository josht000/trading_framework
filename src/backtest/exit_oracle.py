"""Hindsight (oracle) exit diagnostics per trade window.  Every output column is prefixed `hindsight_`:
the table is a derived diagnostic and must never feed entry features or ranking.

Candidate decision bars (label = bar open, decision at the bar close, fill from the next bar open via
exit_sim._fill, i.e. the realistic fill model): "pm" = next-session 08:00-10:15 incl. pre-market, "open" = 09:30-10:15.
"""
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import exit_sim as X
from src.data.pit_integrity import ORACLE, ORACLE_META, WINDOWS, WINDOWS_INDEX, sha256_file

CLOSES = ["09:31", "09:35", "09:40", "09:45", "10:00", "10:15"]
START, END = "08:00", "10:15"
OPEN_MIN, PM_MIN, END_MIN = 570, 480, 615
N_PERM, SEED = 200, 20261023
GAPS = [-np.inf, 0.0, 0.03, 0.10, np.inf]
GAP_LABELS = ["open<entry", "0-3%", "3-10%", ">10%"]


def _label(m):
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"


def exec_ret(w, t, shares, horizon):
    """Realistic net return of a full sale decided at the close of bar t (== simulate_trade for one full event)."""
    px, got, _, _ = X._fill(w, t + 1, horizon, shares, "realistic")
    return px / w.entry_price - 1 - 2 * X.COMMISSION if got else np.nan


def _shares(w):
    return max(X.SIZE_USD / w.entry_signal_price * w.fill_entry if w.entry_signal_price else 1.0, 1e-9)


def _perm_peak(path, rng, n_perm):
    r = path[1:] / path[:-1] - 1
    lv = path[0] * np.cumprod(1 + rng.permuted(np.tile(r, (n_perm, 1)), axis=1), axis=1) if len(r) else path[:1, None]
    return lv.max(axis=1) if len(r) == 0 else np.maximum(lv.max(axis=1), path[0])


def rw_peak(w, reg, rng, n_perm=N_PERM):
    """Mean gross peak return over permutations of the 09:30-10:15 minute-to-minute close returns (gap and first bar
    fixed).  Second value: also shuffle the first bar's open-to-close return (sensitivity)."""
    c = w.closes[reg]
    first = float(np.mean(_perm_peak(c, rng, n_perm)) / w.entry_price - 1)
    path = np.concatenate([[w.opens[reg[0]]], c])
    return first, float(np.mean(_perm_peak(path, rng, n_perm)) / w.entry_price - 1)


def oracle_row(w, n_perm=N_PERM, seed=SEED):
    nan = {"trade_id": w.trade_id, "hindsight_reason": "ok"}
    fn = X._info(w)["first_next"]
    if fn is None:
        return {**nan, "hindsight_reason": "no_next_session"}
    _, minutes = X._clock(w)
    nxt, hz = X._info(w)["next_idx"], X._info(w)["horizon"]
    m = minutes[nxt]
    sets = {"pm": nxt[(m >= PM_MIN) & (m <= END_MIN)], "open": nxt[(m >= OPEN_MIN) & (m <= END_MIN)]}
    if not len(sets["open"]):
        return {**nan, "hindsight_reason": "no_regular_bars"}
    e, sh = w.entry_price, _shares(w)
    out = dict(nan)
    for k, idx in sets.items():
        j = idx[int(np.argmax(w.closes[idx]))]
        out[f"hindsight_peak_close_ret_{k}"] = float(w.closes[j] / e - 1)
        out[f"hindsight_peak_minute_{k}"] = int(minutes[j] - OPEN_MIN)
        out[f"hindsight_peak_clock_{k}"] = _label(minutes[j])
        r = np.array([exec_ret(w, t, sh, hz) for t in idx])
        if np.isfinite(r).any():
            b = int(np.nanargmax(r))
            out[f"hindsight_oracle_exec_ret_{k}"], out[f"hindsight_oracle_exec_bar_{k}"] = float(r[b]), _label(minutes[idx[b]])
        else:
            out[f"hindsight_oracle_exec_ret_{k}"], out[f"hindsight_oracle_exec_bar_{k}"] = np.nan, None
    reg = sets["open"]
    o = w.opens[reg[0]]
    for ref, name in ((e, "entry"), (o, "open")):
        out[f"hindsight_mfe_vs_{name}"] = float(w.closes[reg].max() / ref - 1)
        out[f"hindsight_mae_vs_{name}"] = float(w.closes[reg].min() / ref - 1)
    out["hindsight_open_vs_entry"] = float(o / e - 1)
    for c in CLOSES:
        out[f"hindsight_ret_at_{c.replace(':', '')}"] = X.simulate_trade(w, X.fixed_time_policy(c))["ret"]
    out["hindsight_ret_baseline"] = X.simulate_trade(w, X.baseline_policy())["ret"]
    out["hindsight_ret_baseline_legacy"] = X.simulate_trade(w, X.baseline_policy(), "legacy")["ret"]
    out["hindsight_rw_peak_ret_open"], out["hindsight_rw_peak_ret_open_firstbar_shuffled"] = rw_peak(w, reg, np.random.default_rng([seed, w.trade_id]), n_perm)
    return out


def build_oracle(windows, index, n_perm=N_PERM, seed=SEED):
    """One row per windows_index row (trade_id, symbol, entry_time, period + hindsight_* columns)."""
    rows = pd.DataFrame([oracle_row(w, n_perm, seed) for w in windows])
    return index[["trade_id", "symbol", "entry_time", "period"]].merge(rows, on="trade_id", how="left")


def params():
    return {"window_start": START, "window_end": END, "closes": CLOSES, "commission_per_side": X.COMMISSION,
            "participation": X.PARTICIPATION, "slippage_rth": X.SLIP_RTH, "slippage_ext": X.SLIP_EXT,
            "size_usd": X.SIZE_USD, "fill_model": "realistic (decision at bar t close, fill from t+1 open)",
            "rw_permutations": N_PERM, "rw_seed": SEED, "baseline_policy": X.baseline_policy()}


def git_commit():
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def make_meta(oracle, windows=WINDOWS, index=WINDOWS_INDEX, oracle_path=ORACLE):
    return {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "windows_sha256": sha256_file(windows), "windows_index_sha256": sha256_file(index),
            "oracle_sha256": sha256_file(oracle_path), "rows": int(len(oracle)),
            "rows_by_period": {k: int(v) for k, v in oracle["period"].value_counts().items()},
            "rows_ok": int((oracle["hindsight_reason"] == "ok").sum()),
            "code_sha256": {f: sha256_file(Path(__file__).with_name(f)) for f in ("exit_sim.py", "exit_oracle.py")},
            "git_commit": git_commit(), "params": params()}


def gap_bucket(g):
    return pd.cut(pd.Series(np.asarray(g, float)), GAPS, labels=GAP_LABELS, right=False)
