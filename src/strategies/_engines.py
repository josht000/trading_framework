"""Shared trade simulators for catalog daily-bar studies.

swing_trades: per-symbol signal -> position -> exit-condition loop (Connors/Donchian
swing systems). rebalance_trades: pick top-k by a score at each rebalance row, exit at
the next rebalance (monthly momentum/rotation). Both return the common trades schema
(sym_idx, signal_idx, entry_idx, exit_idx, entry_px, exit_px, ret, ret_gross,
exit_reason, hold, dates/symbol, fill, plus strategy extras sampled at the signal row).
"""

import numpy as np
import pandas as pd

from src.utils import timing

BASE_COLS = ["sym_idx", "signal_idx", "entry_idx", "exit_idx", "entry_px", "exit_px",
             "ret", "ret_gross", "exit_reason"]

NY = "America/New_York"


def load_field(data_dir, symbols, dates, field, suffix):
    """Extra OHLCV field (e.g. 'high','low' for ATR) aligned to a panel's date grid."""
    from pathlib import Path

    root = Path(data_dir)
    out = np.full((len(dates), len(symbols)), np.nan)
    with timing.stage(f"panel field {field}"):
        for j, symbol in enumerate(symbols):
            path = root / f"{symbol}{suffix}"
            if not path.exists():
                continue
            frame = pd.read_parquet(path, columns=[field])
            idx = frame.index.tz_convert(NY).tz_localize(None).normalize()
            rows = np.searchsorted(np.asarray(dates), idx.values)
            ok = (rows < len(dates)) & (np.asarray(dates)[np.clip(rows, 0, len(dates) - 1)] == idx.values)
            out[rows[ok], j] = frame[field].to_numpy(float)[ok]
    return out


def merge_panels(*panels):
    """Column-concatenate panels on the union of their dates (disjoint symbol sets)."""
    dates = np.unique(np.concatenate([p["dates"].values for p in panels]))
    symbols, arrays = [], {k: [] for k in ("open", "close", "volume")}
    for p in panels:
        rows = np.searchsorted(dates, p["dates"].values)
        for k in arrays:
            arr = np.full((len(dates), len(p["symbols"])), np.nan)
            arr[rows] = p[k]
            arrays[k].append(arr)
        symbols += list(p["symbols"])
    return {"dates": pd.DatetimeIndex(dates), "symbols": symbols,
            **{k: np.concatenate(v, axis=1) for k, v in arrays.items()}}


def align_ranks(panel_dates, panel_symbols, rank_dates, rank_symbols, rank):
    """Map a (dates x stock-symbols) rank matrix onto a panel grid; missing -> inf."""
    rows = np.searchsorted(np.asarray(rank_dates), np.asarray(panel_dates))
    ok_row = (rows < len(rank_dates)) & (np.asarray(rank_dates)[np.clip(rows, 0, len(rank_dates) - 1)] == np.asarray(panel_dates))
    col_of = {s: i for i, s in enumerate(rank_symbols)}
    out = np.full((len(panel_dates), len(panel_symbols)), np.inf)
    for j, s in enumerate(panel_symbols):
        i = col_of.get(s)
        if i is None:
            continue
        col = rank[:, i]
        out[ok_row, j] = col[rows[ok_row]]
    return out


def last_valid_rows(close):
    """Last row index with a finite close per column (-1 if none)."""
    return np.where(np.isfinite(close), np.arange(close.shape[0])[:, None], -1).max(0)


def _finish(frame, panel, fill, extras):
    dates, symbols = panel["dates"], panel["symbols"]
    frame["symbol"] = np.asarray(symbols, dtype=object)[frame["sym_idx"].to_numpy(int)]
    frame["signal_date"] = dates[frame["signal_idx"].to_numpy(int)]
    frame["entry_date"] = dates[frame["entry_idx"].to_numpy(int)]
    frame["exit_date"] = dates[frame["exit_idx"].to_numpy(int)]
    frame["hold"] = frame["exit_idx"] - frame["entry_idx"]
    for name, mat in extras.items():
        frame[name] = mat[frame["signal_idx"].to_numpy(int), frame["sym_idx"].to_numpy(int)]
    frame["fill"] = fill
    return frame.sort_values(["entry_idx", "symbol"], kind="stable").reset_index(drop=True)


def swing_trades(panel, signal, exit_now, cfg, fill, *, add_signal=None, max_units=1,
                 stop_atr=None, stop_mult=0.0, extras=None, delisting_return=None):
    """Per-symbol swing book: enter on `signal`, exit when `exit_now` or the stop fires.

    fill='close': signal and exit decisions on session t fill at close[t] (upper bound).
    fill='next_open': signal day t fills at open[t+1]; an exit decided on session u fills at open[u+1].
    add_signal/max_units: pyramid adds (each unit is a separate trade sharing the exit).
    stop_atr/stop_mult: disaster stop close[u] < entry_px - stop_mult*ATR[signal row]
    (level anchored on the first unit's entry price).
    delisting_return: applied to the exit price when a symbol's series ends mid-sample.
    """
    open_, close = panel["open"], panel["close"]
    dates, symbols = panel["dates"], panel["symbols"]
    n_rows = len(dates)
    cost = float(cfg["cost_bps_per_side"]) / 1e4
    is_open = fill == "next_open"
    last = last_valid_rows(close)
    rows = []
    for j in np.flatnonzero(signal.any(0)):
        cj, oj = close[:, j], open_[:, j]
        ej = exit_now[:, j]
        aj = add_signal[:, j] if add_signal is not None else None
        atrj = stop_atr[:, j] if stop_atr is not None else None
        lj = last[j]
        if lj < 0:
            continue
        t_prev = -1
        for t in np.flatnonzero(signal[:, j]):
            if t <= t_prev:
                continue
            entry0 = t + 1 if is_open else t
            if entry0 > lj or (is_open and not np.isfinite(oj[entry0])):
                continue
            e0_px = oj[entry0] if is_open else cj[t]
            stop_level = e0_px - stop_mult * atrj[t] if atrj is not None and np.isfinite(atrj[t]) else None
            reason, u_exit = "signal", None
            for u in range(t + 1, lj + 1):
                if not np.isfinite(cj[u]):
                    continue
                if stop_level is not None and cj[u] < stop_level:
                    reason, u_exit = "stop", u
                    break
                if ej[u]:
                    reason, u_exit = "signal", u
                    break
            if u_exit is None:
                u_exit = lj
                reason = "delisted" if lj < n_rows - 1 else "censored"
            legs = [t]
            if aj is not None:
                for u in range(t + 1, u_exit):
                    if len(legs) >= max_units:
                        break
                    if np.isfinite(cj[u]) and aj[u]:
                        legs.append(u)
            for t_leg in legs:
                entry = t_leg + 1 if is_open else t_leg
                if entry > lj or (is_open and not np.isfinite(oj[entry])):
                    continue
                entry_px = oj[entry] if is_open else cj[t_leg]
                exit_idx, exit_px = u_exit, cj[u_exit]
                if is_open and reason not in ("delisted", "censored") and u_exit + 1 <= lj and np.isfinite(oj[u_exit + 1]):
                    exit_idx, exit_px = u_exit + 1, oj[u_exit + 1]
                if reason == "delisted" and delisting_return is not None:
                    exit_px *= 1.0 + float(delisting_return)
                if exit_idx < entry:
                    continue
                net = exit_px * (1 - cost) / (entry_px * (1 + cost)) - 1.0
                rows.append((j, t_leg, entry, exit_idx, entry_px, exit_px, net, exit_px / entry_px - 1.0, reason))
            t_prev = u_exit
    frame = pd.DataFrame(rows, columns=BASE_COLS)
    return _finish(frame, panel, fill, extras or {})


def rebalance_trades(panel, score, tradable, rebalance_rows, k, cfg, fill, *,
                     extras=None, delisting_return=-0.30):
    """Top-k equal-weight book rebalanced at `rebalance_rows` (row indices, month-ends).

    At each rebalance row the k highest-score tradable names are bought; every position
    is closed at the next rebalance row (kept names are re-bought, i.e. weights reset).
    fill='close': decision and fill at the rebalance-session close. fill='next_open':
    decision on the rebalance close, fill at the next session's open.
    delisting_return: final-leg return applied when a held name's series ends mid-sample.
    """
    open_, close = panel["open"], panel["close"]
    dates, symbols = panel["dates"], panel["symbols"]
    n_rows = len(dates)
    cost = float(cfg["cost_bps_per_side"]) / 1e4
    is_open = fill == "next_open"
    last = last_valid_rows(close)
    rows = []
    bounds = list(rebalance_rows) + [n_rows - 1]
    for i, (t, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        elig = tradable[t] & np.isfinite(score[t]) & np.isfinite(close[t])
        if not elig.any():
            continue
        picks = np.argsort(np.where(elig, -score[t], np.inf), kind="stable")[:k]
        final_bound = i == len(bounds) - 2
        for j in picks:
            last_j = last[j]
            entry = t + 1 if is_open else t
            if entry > last_j or (is_open and not np.isfinite(open_[entry, j])):
                continue
            entry_px = open_[entry, j] if is_open else close[t, j]
            dead = last_j < b
            exit_row = min(b + 1, last_j) if is_open else min(b, last_j)
            if dead:
                reason = "delisted" if last_j < n_rows - 1 else "censored"
            else:
                reason = "censored" if final_bound else "rebalance"
            exit_px = close[exit_row, j]
            if is_open and not dead and np.isfinite(open_[exit_row, j]):
                exit_px = open_[exit_row, j]
            while exit_row > entry and not np.isfinite(exit_px):
                exit_row -= 1
                exit_px = close[exit_row, j]
            if reason == "delisted" and delisting_return is not None:
                exit_px *= 1.0 + float(delisting_return)
            if exit_row < entry:
                continue
            net = exit_px * (1 - cost) / (entry_px * (1 + cost)) - 1.0
            rows.append((j, t, entry, exit_row, entry_px, exit_px, net, exit_px / entry_px - 1.0, reason))
    frame = pd.DataFrame(rows, columns=BASE_COLS)
    return _finish(frame, panel, fill, extras or {})
