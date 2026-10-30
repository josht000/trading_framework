"""S075 signal and trade construction on a wide daily panel (all arrays are date x symbol)."""

import warnings

import numpy as np
import pandas as pd

from src.data.daily_panel import liquidity_ranks
from src.utils import timing


def daily_returns(close):
    out = np.full(close.shape, np.nan)
    out[1:] = close[1:] / close[:-1] - 1.0
    return out


def peer_returns(ret, member, groups, min_peers):
    """Leave-one-out equal-weight mean return of `member` names in the same group (NaN if < min_peers peers)."""
    out = np.full(ret.shape, np.nan)
    live = member & ~np.isnan(ret)
    values = np.where(live, ret, 0.0)
    for code in np.unique(groups[groups >= 0]):
        cols = np.flatnonzero(groups == code)
        x, m = values[:, cols], live[:, cols].astype(float)
        n = m.sum(1, keepdims=True) - m
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (x.sum(1, keepdims=True) - x) / n
        out[:, cols] = np.where(n >= min_peers, mean, np.nan)
    return out


def rolling_sum(a, window, min_periods):
    return pd.DataFrame(a).rolling(window, min_periods=min_periods).sum().to_numpy()


def rolling_beta(y, x, window, min_obs):
    """OLS slope of y on x over the trailing window ending at each row (inclusive); pairs need both observed."""
    valid = ~np.isnan(y) & ~np.isnan(x)
    yv, xv = np.where(valid, y, 0.0), np.where(valid, x, 0.0)
    n = rolling_sum(valid.astype(float), window, 1)
    sx, sy = rolling_sum(xv, window, 1), rolling_sum(yv, window, 1)
    sxx, sxy = rolling_sum(xv * xv, window, 1), rolling_sum(xv * yv, window, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        var = sxx - sx * sx / n
        beta = cov / var
    return np.where((n >= min_obs) & (var > 1e-12), beta, np.nan)


def shift_down(a, k=1):
    out = np.full(a.shape, np.nan)
    out[k:] = a[:-k]
    return out


def ew_returns(ret, tradable):
    """Equal-weight mean daily return of the tradable universe (0 where nothing is tradable)."""
    live = tradable & ~np.isnan(ret)
    return np.where(live, np.nan_to_num(ret), 0.0).sum(1) / np.maximum(live.sum(1), 1)


def regime_block(b_day, cfg):
    """Rows (sessions) where new entries are blocked by the market-stress filter; None when the filter is off.

    ew_drawdown: equal-weight universe return over `regime_window` sessions (through t) below -regime_threshold.
    ew_vol: annualised stdev of the equal-weight universe daily return over `regime_window` above regime_threshold.
    """
    mode = cfg.get("regime_mode", "none")
    if mode == "none":
        return None
    n, thr = int(cfg.get("regime_window", 20)), float(cfg["regime_threshold"])
    if mode == "ew_drawdown":
        window_ret = np.expm1(pd.Series(np.log1p(b_day)).rolling(n, min_periods=n).sum().to_numpy())
        return window_ret < -thr
    if mode == "ew_vol":
        vol = pd.Series(b_day).rolling(n, min_periods=n).std().to_numpy() * np.sqrt(252)
        return vol > thr
    raise ValueError(f"unknown regime_mode {mode!r}")


def compute_signals(panel, sector_map, cfg):
    """Return a dict of matrices: resid5, beta, tradable, pool, signal, sma_ok (all through-t information only)."""
    close, volume = panel["close"], panel["volume"]
    symbols = panel["symbols"]
    with timing.stage("liquidity ranks"):
        rank = liquidity_ranks(close, volume, min_price=float(cfg["min_price"]), min_history=int(cfg["min_history"]))
        tradable = rank <= int(cfg["universe_top_n"])
        peer_pool = rank <= int(cfg["peer_top_n"])
    with timing.stage("peer returns"):
        ret = daily_returns(close)
        industries = sector_map["Industry"].reindex(symbols)
        sectors = sector_map["Sector"].reindex(symbols)
        ind_codes = pd.factorize(industries)[0]
        sec_codes = pd.factorize(sectors)[0]
        min_peers = int(cfg["min_peers"])
        if cfg.get("peer_level", "industry") == "sector":
            peer = peer_returns(ret, peer_pool, sec_codes, min_peers)
        else:
            peer = peer_returns(ret, peer_pool, ind_codes, min_peers)
            fallback = peer_returns(ret, peer_pool, sec_codes, min_peers)
            peer = np.where(np.isnan(peer), fallback, peer)
    with timing.stage("beta + residual"):
        beta = shift_down(rolling_beta(ret, peer, int(cfg["beta_window"]), int(cfg["beta_min_obs"])))
        h = int(cfg["resid_window"])
        stock5 = np.full(close.shape, np.nan)
        stock5[h:] = close[h:] / close[:-h] - 1.0
        log_peer = np.log1p(peer)
        peer5 = np.expm1(pd.DataFrame(log_peer).rolling(h, min_periods=h).sum().to_numpy())
        resid5 = stock5 - beta * peer5
    z = None
    if cfg.get("signal_mode", "raw") == "zscore" or cfg.get("exit_mode", "resid") == "zscore":
        with timing.stage("residual z-score"):
            daily_resid = ret - beta * peer
            sigma = shift_down(pd.DataFrame(daily_resid).rolling(int(cfg.get("z_window", 60)), min_periods=40).std().to_numpy())
            with np.errstate(invalid="ignore", divide="ignore"):
                z = resid5 / (sigma * np.sqrt(h))
    score = z if cfg.get("signal_mode", "raw") == "zscore" else resid5
    with timing.stage("decile + sma filter"):
        sma_n = int(cfg["sma_window"])
        if sma_n > 0:
            sma = pd.DataFrame(close).rolling(sma_n, min_periods=sma_n).mean().to_numpy()
            sma_ok = close > sma
        else:
            sma_ok = np.ones(close.shape, bool)
        eligible = tradable & ~np.isnan(score) & ~np.isnan(close)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cut = np.nanquantile(np.where(eligible, score, np.nan), float(cfg["decile"]), axis=1)
        signal = eligible & (score <= cut[:, None]) & sma_ok
        min_abs = float(cfg.get("min_abs_resid", 0.0))
        if min_abs > 0:
            signal = signal & (resid5 <= -min_abs)
        pool = eligible & sma_ok
        skip = int(cfg.get("skip_days", 0))
        if skip > 0:
            signal = np.vstack([np.zeros((skip, signal.shape[1]), bool), signal[:-skip]]) & eligible & sma_ok
    b_day = ew_returns(ret, tradable)
    block = regime_block(b_day, cfg)
    if block is not None:
        signal = signal & ~block[:, None]
    return {"resid5": resid5, "z": z, "beta": beta, "tradable": tradable, "pool": pool, "signal": signal, "ret": ret, "b_day": b_day,
            "industry": industries.to_numpy(), "sector": sectors.to_numpy()}


def build_trades(panel, sig, cfg, fill):
    """Simulate S075 trades for one fill model ('close' or 'next_open'). Returns a trades DataFrame (net of costs)."""
    open_, close = panel["open"], panel["close"]
    resid5, beta, signal = sig["resid5"], sig["beta"], sig["signal"]
    dates, symbols = panel["dates"], panel["symbols"]
    n_rows = len(dates)
    exit_mode = cfg.get("exit_mode", "resid")
    hold_cap = int(cfg.get("fixed_hold", 5)) if exit_mode == "fixed" else int(cfg["max_hold"])
    stop, target = float(cfg.get("stop_loss", 0.0)), float(cfg.get("take_profit", 0.0))
    exit_z = float(cfg.get("exit_z", 0.0))
    zmat = sig.get("z")
    cost = float(cfg["cost_bps_per_side"]) / 1e4
    is_open = fill == "next_open"
    valid = ~np.isnan(close)
    rows = []
    for j in np.flatnonzero(signal.any(0)):
        cj, oj, rj, vj = close[:, j], open_[:, j], resid5[:, j], valid[:, j]
        zj = zmat[:, j] if zmat is not None else rj
        last = int(np.flatnonzero(vj)[-1])
        t_prev = -1
        for t in np.flatnonzero(signal[:, j]):
            if t <= t_prev:
                continue
            entry = t + 1 if is_open else t
            if entry > last or (is_open and np.isnan(oj[entry])):
                continue
            entry_px = oj[entry] if is_open else cj[t]
            reason, u = "cap", t + hold_cap
            for k in range(1, hold_cap + 1):
                u = t + k
                if u > last:
                    reason, u = ("delisted" if last < n_rows - 1 else "censored"), last
                    break
                if not vj[u]:
                    continue
                if stop and cj[u] <= entry_px * (1 - stop):
                    reason = "stop"
                    break
                if target and cj[u] >= entry_px * (1 + target):
                    reason = "target"
                    break
                if exit_mode == "resid" and rj[u] > 0:
                    reason = "resid"
                    break
                if exit_mode == "zscore" and zj[u] > exit_z:
                    reason = "resid"
                    break
            else:
                u = t + hold_cap
            while u > t and not vj[u]:
                u -= 1
            decision = u
            exit_idx, exit_px = u, cj[u]
            if is_open and reason not in ("delisted", "censored") and u + 1 <= last and not np.isnan(oj[u + 1]):
                exit_idx, exit_px = u + 1, oj[u + 1]
            t_prev = decision
            if exit_idx < entry:
                continue
            net = exit_px * (1 - cost) / (entry_px * (1 + cost)) - 1.0
            rows.append((j, t, entry, exit_idx, entry_px, exit_px, net, exit_px / entry_px - 1.0, reason, rj[t], beta[t, j]))
    cols = ["sym_idx", "signal_idx", "entry_idx", "exit_idx", "entry_px", "exit_px", "ret", "ret_gross", "exit_reason", "resid5", "beta"]
    frame = pd.DataFrame(rows, columns=cols)
    frame["symbol"] = np.asarray(symbols, dtype=object)[frame["sym_idx"].to_numpy(int)]
    frame["signal_date"] = dates[frame["signal_idx"].to_numpy(int)]
    frame["entry_date"] = dates[frame["entry_idx"].to_numpy(int)]
    frame["exit_date"] = dates[frame["exit_idx"].to_numpy(int)]
    frame["hold"] = frame["exit_idx"] - frame["entry_idx"]
    frame["industry"] = sig["industry"][frame["sym_idx"].to_numpy(int)]
    frame["sector"] = sig["sector"][frame["sym_idx"].to_numpy(int)]
    frame["fill"] = fill
    return frame.sort_values(["entry_idx", "symbol"], kind="stable").reset_index(drop=True)
