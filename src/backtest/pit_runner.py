"""
Per-symbol trade generation for the point-in-time backtest, runnable in worker processes.

Each symbol is independent, so symbols are fanned out over a process pool (PROCESSES env var,
default cpu_count - 1; 1 runs in-process). atr_pivot level tops are cached on disk per symbol and
session date (data/pit/ladder_cache/<config hash>.pkl), so repeat runs skip the detector.
A cached entry is reused only if the close series it was built from is unchanged
(hash of the first n bars), which stays valid when new bars are appended.

Fill model (cfg["realism"]): a position of size_usd is built/liquidated bar by bar, taking at most
`participation` of each 1-minute bar's volume, at that bar's close moved against us by slippage
(slip_rth in the regular session, slip_ext outside it). An entry that cannot fill min_fill of the
position by 15:59 is skipped; an exit that runs out of bars sells the remainder at the last bar.
"""

import hashlib
from dataclasses import replace
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.screening.historical import ATR_PIVOT_CONFIG
from src.strategies.humbled_bullish_overnight import HumbledBullishOvernight

CACHE_DIR = os.path.join("data", "pit", "ladder_cache")
T_OPEN, T_CLOSE = pd.Timestamp("09:30").time(), pd.Timestamp("16:00").time()


def cache_path(lookback_bars=None):
    conf = replace(ATR_PIVOT_CONFIG, lookback_bars=lookback_bars) if lookback_bars else ATR_PIVOT_CONFIG
    key = hashlib.md5(repr(conf).encode()).hexdigest()[:10]
    return os.path.join(CACHE_DIR, f"{key}.pkl")


def load_cache(lookback_bars=None):
    try:
        with open(cache_path(lookback_bars), "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return {}


def save_cache(cache, lookback_bars=None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(lookback_bars)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(cache, f)
    os.replace(path + ".tmp", path)


def _fp(daily_df, n):
    c = daily_df["close"].to_numpy(dtype=float)[:n]
    return hashlib.md5(c.tobytes()).hexdigest()


def valid_levels(entry, daily_df):
    if not entry or len(daily_df) < entry["n"] or _fp(daily_df, entry["n"]) != entry["fp"]:
        return {}
    return dict(entry["levels"])


def process_symbol(task):
    s, daily_df, minute_df, ctx, cand_dates, cfg, cached = task
    strategy = HumbledBullishOvernight(**cfg["strategy"])
    levels = valid_levels(cached, daily_df) if cfg["use_cache"] else {}
    n_before = len(levels)
    strategy.screener.ladder_cache = levels if cfg["use_cache"] else None
    try:
        sig = strategy.apply_strategy(daily_df, minute_df, ctx, entry_dates=cand_dates)
    except Exception as e:
        return s, [], 0, None, f"{s}: error {e}"
    new_entry = None
    if cfg["use_cache"] and len(levels) > n_before:
        new_entry = {"n": len(daily_df), "fp": _fp(daily_df, len(daily_df)), "levels": levels}
    trades, forced = build_trades(s, sig, daily_df, cfg, (ctx or {}).get("news_df"))
    return s, trades, forced, new_entry, None


def _atr_pct(daily_df):
    h, l, c = (daily_df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.fmax.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()
    return pd.Series(atr / c, index=daily_df.index)


def _fill(times, close, vol, shares, rz, side):
    """Sequential volume-limited fill. Returns (avg_price, filled_shares, remaining, last_index)."""
    rem, cost, got, last = shares, 0.0, 0.0, None
    for i in range(len(close)):
        if rem <= 1e-9:
            break
        cap = rz["participation"] * vol[i]
        if cap <= 0:
            continue
        take = min(rem, cap)
        slip = rz["slip_rth"] if T_OPEN <= times[i] < T_CLOSE else rz["slip_ext"]
        px = close[i] * (1 + slip) if side == "buy" else close[i] * (1 - slip)
        cost += take * px
        got += take
        rem -= take
        last = i
    return (cost / got if got else float("nan")), got, rem, last


def _exit_fill(sig, tm, trigger_ts, shares, rz):
    pos = sig.index.get_loc(trigger_ts)
    close = sig["close"].to_numpy(dtype=float)[pos:]
    vol = sig["volume"].to_numpy(dtype=float)[pos:]
    avg, got, rem, last = _fill(tm[pos:], close, vol, shares, rz, "sell")
    partial = rem > 1e-9 or got == 0
    if partial:
        lastpx = close[-1] if len(close) else float("nan")
        slip = rz["slip_ext"] if not (T_OPEN <= tm[-1] < T_CLOSE) else rz["slip_rth"]
        tail = lastpx * (1 - slip)
        avg = tail if got == 0 else (avg * got + tail * rem) / (got + rem)
    return avg, partial


def _features(sig, ts, ep, daily_df, day, atr_prior):
    d = daily_df[daily_df.index < pd.Timestamp(day)]
    prev_close = float(d["close"].iloc[-1])
    today = sig[(sig.index.date == day) & (sig.index <= ts)]
    today = today[(today.index.time >= T_OPEN) & (today.index.time < T_CLOSE)]
    row = sig.loc[ts]
    c60 = today[today.index >= ts - pd.Timedelta(minutes=60)]["close"]
    return {
        "prev_close": prev_close,
        "sig_day_ret": ep / prev_close - 1,
        "gap_open": float(today["open"].iloc[0]) / prev_close - 1,
        "run5": prev_close / float(d["close"].iloc[-6]) - 1 if len(d) >= 6 else float("nan"),
        "price": ep,
        "dvol20": float(d["volume"].tail(20).mean()) * prev_close,
        "rel_vol": float(row.get("relative_volume", float("nan"))),
        "atr_pct": atr_prior,
        "dist_vwap": ep / float(row.get("session_vwap", float("nan"))) - 1,
        "from_high": ep / float(today["high"].max()) - 1,
        "range_pct": float(today["high"].max()) / float(today["low"].min()) - 1,
        "last60": ep / float(c60.iloc[0]) - 1 if len(c60) else float("nan"),
        "day_dvol": float(today["volume"].sum()) * ep,
        "dow": ts.dayofweek,
    }


POS_WORDS = ("contract", "award", "order", "selected", "agreement", "partnership", "collaborat", "approval",
             "approves", "clearance", "fda", "breakthrough", "record", "upgrade", "raises guidance", "beats",
             "surge", "soar", "acquire", "acquisition", "launch", "wins", "expands", "patent", "milestone")
NEG_WORDS = ("offering", "private placement", "registered direct", "at-the-market", "atm ", "dilut", "shelf",
             "warrant", "reverse split", "going concern", "bankrupt", "delist", "deficiency", "downgrade",
             "lawsuit", "investigation", "misses", "cuts guidance", "resign", "halt", "default", "sec ")
DILUTION_WORDS = ("offering", "private placement", "registered direct", "at-the-market", "atm ", "dilut",
                  "shelf", "reverse split", "going concern", "warrants")


def _news_features(news, ts):
    """Headline/summary features known at `ts` (article timestamps are UTC). news: DataFrame or None."""
    keys = ("news_24h", "news_5d", "news_pos_24h", "news_neg_24h", "news_score_24h", "dilution_10d")
    out = {k: 0 for k in keys}
    out["has_news_data"] = int(news is not None and len(news) > 0)
    if news is None or len(news) == 0:
        return out
    t = ts.tz_convert("UTC")
    w24 = news[(news["created_at"] > t - pd.Timedelta(hours=24)) & (news["created_at"] <= t)]
    w5 = news[(news["created_at"] > t - pd.Timedelta(days=5)) & (news["created_at"] <= t)]
    w10 = news[(news["created_at"] > t - pd.Timedelta(days=10)) & (news["created_at"] <= t)]
    text = lambda df: (df["headline"].fillna("") + " " + df["summary"].fillna("")).str.lower()
    out["news_24h"], out["news_5d"] = len(w24), len(w5)
    if len(w24):
        tx = text(w24)
        out["news_pos_24h"] = int(sum(tx.str.contains(w, regex=False).any() for w in POS_WORDS))
        out["news_neg_24h"] = int(sum(tx.str.contains(w, regex=False).any() for w in NEG_WORDS))
        out["news_score_24h"] = out["news_pos_24h"] - out["news_neg_24h"]
    if len(w10):
        tx = text(w10)
        out["dilution_10d"] = int(sum(tx.str.contains(w, regex=False).any() for w in DILUTION_WORDS))
    return out


def _ma_features(daily_df, day, ep):
    """Moving-average position at 15:55: 200 SMA and 9/20 EMA (prior closes, rolled forward with the 15:55 price)."""
    c = daily_df.loc[daily_df.index < pd.Timestamp(day), "close"].astype(float)
    out = {}
    sma200 = float(c.tail(200).mean()) if len(c) >= 200 else float("nan")
    out["sma200"] = sma200
    out["dist_sma200"] = ep / sma200 - 1 if np.isfinite(sma200) else float("nan")
    out["above_sma200"] = float(ep > sma200) if np.isfinite(sma200) else float("nan")
    for n in (9, 20):
        if len(c) >= n:
            e = float(c.ewm(span=n, adjust=False).mean().iloc[-1])
            e = (2 / (n + 1)) * ep + (1 - 2 / (n + 1)) * e
            out[f"dist_ema{n}"] = ep / e - 1
        else:
            out[f"dist_ema{n}"] = float("nan")
    return out


def _path(sig, tm, ts, ep, day, later):
    """Where the overnight move happened: after-hours (to 16:45), pre-market, and the open."""
    out = {"ah45_min": np.nan, "ah45_last": np.nan, "pm_last": np.nan, "open_ret": np.nan}
    ah = sig[(sig.index > ts) & (sig.index.date == day) & (tm >= T_CLOSE) & (tm <= pd.Timestamp("16:45").time())]
    if len(ah):
        out["ah45_min"], out["ah45_last"] = float(ah["close"].min() / ep - 1), float(ah["close"].iloc[-1] / ep - 1)
    if len(later):
        nd = later[0].date()
        pm = sig[(sig.index.date == nd) & (tm < T_OPEN)]
        if len(pm):
            out["pm_last"] = float(pm["close"].iloc[-1] / ep - 1)
        out["open_ret"] = float(sig.at[later[0], "close"] / ep - 1)
    return out


def build_trades(s, sig, daily_df, cfg, news=None):
    """Trades for every exit variant in cfg['variants'] (name -> spec) on the same entry signals."""
    commission, rz = cfg["commission"], cfg["realism"]
    delisted = bool(pd.notna(daily_df["delisted_utc"].iloc[-1]))
    entries = sig.index[sig["entry"]]
    exits = sig.index[sig["exit"]]
    tm = sig.index.time
    t_ah = pd.Timestamp("16:00")
    atr_pct = _atr_pct(daily_df)
    close_a = sig["close"].to_numpy(dtype=float)
    vol_a = sig["volume"].to_numpy(dtype=float)
    trades, forced = [], 0

    for ts in entries:
        ep0 = float(sig.at[ts, "close"])
        day = ts.date()
        later = exits[exits > ts]
        prev = atr_pct[atr_pct.index < pd.Timestamp(day)]
        prior_atr = float(prev.iloc[-1]) if len(prev) else float("nan")

        shares = rz["size_usd"] / ep0
        pos = sig.index.get_loc(ts)
        end = pos
        while end + 1 < len(sig) and sig.index[end + 1].date() == day and tm[end + 1] < T_CLOSE:
            end += 1
        pe, got, _, _ = _fill(tm[pos:end + 1], close_a[pos:end + 1], vol_a[pos:end + 1], shares, rz, "buy")
        fill_frac = got / shares
        if fill_frac < rz["min_fill"]:
            continue
        shares = got
        feats = {**_features(sig, ts, ep0, daily_df, day, prior_atr), **_path(sig, tm, ts, ep0, day, later),
                 **_ma_features(daily_df, day, ep0), **_news_features(news, ts), "fill_entry": fill_frac,
                 "headroom": float(sig.at[ts, "headroom"]) if "headroom" in sig.columns else float("nan")}

        for name, v in cfg["variants"].items():
            hit = None
            if v.get("rules", True):
                t1 = (t_ah + pd.Timedelta(minutes=v.get("window_min", 45))).time()
                ah = sig[(sig.index > ts) & (sig.index.date == day) & (tm >= T_CLOSE) & (tm <= t1)]
                if len(ah):
                    chg = ah["close"] / pe - 1
                    stop = v.get("stop")
                    if v.get("atr_k"):
                        stop = float(np.clip(v["atr_k"] * prior_atr, v["atr_min"], v["atr_max"])) \
                            if np.isfinite(prior_atr) else v["atr_min"]
                    below = (chg <= -stop) if stop else pd.Series(False, index=chg.index)
                    n = v.get("confirm", 1)
                    if n > 1:
                        below = below.astype(int).rolling(n).sum() == n
                    above = (chg >= v["target"]) if v.get("target") else pd.Series(False, index=chg.index)
                    h = chg[below | above]
                    if len(h):
                        hit = (h.index[0], "ah_stop" if below.loc[h.index[0]] else "ah_target")
            if hit is None and len(later) and (v.get("pm_target") or v.get("pm_stop")):
                nd = later[0].date()
                pm = sig[(sig.index.date == nd) & (tm < T_OPEN)]
                if len(pm):
                    chg = pm["close"] / pe - 1
                    up = (chg >= v["pm_target"]) if v.get("pm_target") else pd.Series(False, index=chg.index)
                    dn = (chg <= -v["pm_stop"]) if v.get("pm_stop") else pd.Series(False, index=chg.index)
                    h = chg[up | dn]
                    if len(h):
                        hit = (h.index[0], "pm_target" if up.loc[h.index[0]] else "pm_stop")
            if hit is not None:
                xs, reason = hit
                px, partial = _exit_fill(sig, tm, xs, shares, rz)
                flag = "partial_exit" if partial else ""
            elif len(later):
                xs, reason = later[0], "next_open"
                px, partial = _exit_fill(sig, tm, xs, shares, rz)
                flag = "partial_exit" if partial else ""
            else:
                after = daily_df[daily_df.index > pd.Timestamp(day)]
                xp = float(after["close"].iloc[0]) if len(after) else ep0
                px = xp * (1 - rz["slip_rth"])
                xs = after.index[0] if len(after) else ts
                reason, flag = "next_open", "no_next_session_bars"
                forced += 1
            trades.append({
                "variant": name, "symbol": s, "entry_time": ts, "exit_time": xs, "entry": pe, "exit": px,
                "ret": px / pe - 1 - 2 * commission, "flag": flag, "exit_reason": reason,
                "delisted": delisted, **feats,
            })
    return trades, forced


def run_symbols(tasks, processes, log=print):
    """Run process_symbol over tasks; returns (trades, forced, cache_updates)."""
    trades, forced, updates = [], 0, {}
    total = len(tasks)

    def consume(i, res):
        nonlocal forced
        s, t, f, entry, err = res
        if err:
            log(f"  {err}")
        trades.extend(t)
        forced += f
        if entry:
            updates[s] = entry
        if i % 200 == 0:
            log(f"  processed {i}/{total} symbols, {len(trades)} trades so far")

    if processes <= 1:
        for i, task in enumerate(tasks, 1):
            consume(i, process_symbol(task))
    else:
        with ProcessPoolExecutor(max_workers=processes) as ex:
            for i, res in enumerate(ex.map(process_symbol, tasks, chunksize=8), 1):
                consume(i, res)
    return trades, forced, updates
