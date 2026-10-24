"""Fast, no-lookahead exit simulation on cached overnight minute-bar windows.

The simulator deliberately takes the entry fill from the PIT trade table.  It is
therefore suitable for comparing exits without re-running signal generation or
entry fills.  Policy callbacks receive only bars through the current close.
"""

from dataclasses import dataclass
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.pit_runner import PM_DEFAULT_START, T_CLOSE, T_OPEN
from src.data.pit_integrity import dedupe_trades


COMMISSION = 0.001
PARTICIPATION = 0.10
SLIP_RTH = 0.0015
SLIP_EXT = 0.004
SIZE_USD = 1250.0  # main.py default SIZE_USD


@dataclass
class TradeWindow:
    trade_id: int
    times: np.ndarray
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    vwap: np.ndarray
    entry_price: float
    entry_signal_price: float
    fill_entry: float
    symbol: str = ""


def _as_et(ts):
    x = pd.Timestamp(ts)
    return x.tz_localize("America/New_York") if x.tz is None else x.tz_convert("America/New_York")


def _window_rows(trade, bars):
    entry_ts = _as_et(trade["entry_time"])
    end_day = entry_ts.date()
    later = bars.index[bars.index.date > end_day]
    if len(later) == 0:
        return bars.iloc[0:0]
    next_day = later[0].date()
    end = pd.Timestamp.combine(next_day, pd.Timestamp("11:00").time()).tz_localize(entry_ts.tz)
    return bars[(bars.index >= entry_ts) & (bars.index <= end)]


def build_windows(trades, minute_root="data/pit/alpaca_minute", cache_path=None, log=print):
    """
    Build compact windows from PIT files, optionally writing one compressed NPZ (+ windows_index.parquet beside it).

    trade_id is the row position in `trades`.  Duplicate-security trades (`dedupe_trades`) get no window and
    duplicate bar timestamps (overlapping day files) are dropped; both are reported loudly.  An optional
    `period` column (dev/test) is carried into the index.
    """
    trades = trades.reset_index(drop=True).copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert("America/New_York")
    skip = set(trades.index.difference(dedupe_trades(trades).index))
    if skip:
        log(f"WARNING build_windows: skipping {len(skip)} duplicate-security trades (trade_ids {sorted(skip)[:10]}...)")
    n_dup_bars = n_dup_windows = 0
    files = {pd.Timestamp(Path(p).stem).date(): p for p in Path(minute_root).glob("*.parquet")}
    day_cache = {}
    out = []
    for i, trade in trades.iterrows():
        if i in skip:
            continue
        entry = _as_et(trade["entry_time"])
        end_date = (entry + pd.Timedelta(days=4)).date()
        dates = [d for d in files if d >= entry.date() and d <= end_date]
        frames = []
        for d in sorted(dates):
            p = files[d]
            if d not in day_cache:
                day_cache[d] = pd.read_parquet(p)
            x = day_cache[d]
            x = x[x["symbol"].eq(trade["symbol"])]
            if len(x):
                frames.append(x)
        if not frames:
            continue
        bars = pd.concat(frames, ignore_index=True)
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert("America/New_York")
        bars = bars.set_index("timestamp").sort_index()
        dup = bars.index.duplicated(keep="last")  # overlapping day files duplicate bars
        n_dup_bars, n_dup_windows = n_dup_bars + int(dup.sum()), n_dup_windows + int(dup.any())
        bars = bars[~dup]
        bars = _window_rows(trade, bars)
        if bars.empty:
            continue
        cols = lambda name, default=0.0: bars[name].to_numpy(float) if name in bars else np.full(len(bars), default)
        out.append(TradeWindow(
            i, bars.index.asi8.copy(), cols("open"), cols("high"), cols("low"),
            cols("close"), cols("volume"), cols("vwap", np.nan),
            float(trade["entry"]), float(trade.get("price", trade["entry"])),
            float(trade.get("fill_entry", 1.0)), str(trade.get("symbol", "")),
        ))
        if log and (i + 1) % 250 == 0:
            log(f"windows {i + 1}/{len(trades)}")
    if n_dup_bars:
        log(f"WARNING build_windows: dropped {n_dup_bars} duplicate bars in {n_dup_windows} windows")
    if cache_path:
        save_windows(out, cache_path)
        trades["period"] = trades["period"] if "period" in trades else ""
        idx = trades.loc[[w.trade_id for w in out], ["symbol", "entry_time", "period"]].rename_axis("trade_id").reset_index()
        idx["entry_time"] = idx["entry_time"].astype(str)
        idx.to_parquet(Path(cache_path).with_name("windows_index.parquet"), index=False)
    return out


def save_windows(windows, path):
    """Save flattened arrays and offsets so worker startup needs one file."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    offsets = [0]
    fields = {k: [] for k in ("times", "opens", "highs", "lows", "closes", "volumes", "vwap")}
    meta = []
    for w in windows:
        n = len(w.times)
        offsets.append(offsets[-1] + n)
        for k in fields:
            fields[k].append(getattr(w, k))
        meta.append([w.trade_id, w.entry_price, w.entry_signal_price, w.fill_entry, w.symbol])
    np.savez_compressed(
        path, offsets=np.asarray(offsets, dtype=np.int64),
        meta=np.asarray(meta, dtype=object),
        **{k: np.concatenate(v) if v else np.array([], dtype=float) for k, v in fields.items()},
    )


def load_windows(path):
    """Load a cache written by :func:`save_windows`."""
    z = np.load(path, allow_pickle=True)
    offsets = z["offsets"]
    meta = z["meta"]
    arrays = {k: z[k] for k in ("times", "opens", "highs", "lows", "closes", "volumes", "vwap")}
    out = []
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        t = arrays["times"][a:b]
        keep = np.ones(len(t), bool)
        if len(t) > 1:  # old caches contain duplicated bars (364 of 2687 windows): keep the first
            keep[1:] = np.diff(t) != 0
            if not keep.all():
                assert np.all(np.diff(t) >= 0)
        sl = lambda k: arrays[k][a:b][keep]
        out.append(TradeWindow(
            int(meta[i][0]), t[keep], sl("opens"), sl("highs"), sl("lows"), sl("closes"), sl("volumes"), sl("vwap"),
            float(meta[i][1]), float(meta[i][2]), float(meta[i][3]), str(meta[i][4]),
        ))
    return out


def _time(values):
    return pd.to_datetime(values).time


def _ts(w, i):
    values = getattr(w, "_times_et", None)
    if values is None:
        raw = np.asarray(w.times, dtype=np.int64)
        if raw.size and np.max(np.abs(raw)) < 10**17:
            raw = raw * 1000
        values = pd.to_datetime(raw, unit="ns", utc=True).tz_convert("America/New_York")
        w._times_et = values
    return values[i]


def _clock(w):
    values = getattr(w, "_clock_cache", None)
    if values is None:
        ts = _ts(w, 0) if len(w.times) else pd.DatetimeIndex([], tz="America/New_York")
        if len(w.times):
            all_ts = w._times_et
            values = (np.asarray(all_ts.date, dtype=object),
                      (all_ts.hour * 60 + all_ts.minute).astype(np.int16))
        else:
            values = (np.array([], dtype=object), np.array([], dtype=np.int16))
        w._clock_cache = values
    return values


def _rth(w, i):
    _, minutes = _clock(w)
    return 570 <= minutes[i] < 960


def _first_next_rth(w):
    dates, minutes = _clock(w)
    entry_day = dates[0]
    for i, d in enumerate(dates):
        if d > entry_day and minutes[i] >= 570:
            return i
    return None


def _window_bounds(w, start_time="08:00", end_time="10:15"):
    start, end = pd.Timestamp(start_time).time(), pd.Timestamp(end_time).time()
    oi = _first_next_rth(w)
    if oi is None:
        return None, None
    dates, minutes = _clock(w)
    d = dates[oi]
    inds = np.flatnonzero(dates == d)
    pre = [i for i in inds if start.hour * 60 + start.minute <= minutes[i] < 570]
    reg = [i for i in inds if 570 <= minutes[i] <= end.hour * 60 + end.minute]
    return (pre[0] if pre else None), (reg[-1] if reg else oi)


"""Conventions (read before using)
-------------------------------
* Bar label = bar OPEN time. A 1-minute bar labelled 09:30 covers 09:30:00-09:30:59 and
  closes at 09:31.  A *decision at bar t* uses information through the close of bar t (i.e. as
  of label+1min) and the earliest possible execution is the OPEN of bar t+1 (same instant).
* ``deadline`` (policy key) = label of the LAST decision bar (inclusive).  If no bar has exactly
  that label the last bar at or before it is used.  ``decision_close`` (policy key, alternative)
  gives the wall-clock time at which the decision bar closes, e.g. "09:35" = bar labelled 09:34;
  "09:31" = bar labelled 09:30 ("close of the 1st minute").
* fixed_time / open_close: sell decision exactly at the deadline bar close (nothing earlier,
  except optional AH / pre-market rules when keep_ah / pm_* are set).
* Other rules (trail, velocity_floor, giveback, below_vwap, model) are evaluated at EVERY regular
  session decision bar close from 09:30 through the deadline.  Scale-out targets likewise and the
  remainder keeps being monitored.  Scale-out ``fraction`` is a fraction of the ORIGINAL position.
* Trailing high: ``peak_from`` = "open" (default; max close of regular-session bars since 09:30
  through the current bar) or "entry" (max close since the entry bar, including after-hours and
  pre-market closes).  Drawdown = close / peak - 1.  ``activate`` compares peak/entry - 1.
* Session VWAP: cumulative sum(typical*volume)/sum(volume) over regular-session bars since 09:30,
  typical = (high+low+close)/3, through the current bar.  below_vwap = close < session VWAP.
* velocity = close / close 5 regular-session bars earlier - 1 (fewer bars if fewer available,
  undefined on the first 09:30 bar, where it never triggers).
* Fills (realistic): decision at bar t, fill starts at bar t+1 open, <=10% of each bar's volume
  (shared across tranches), slippage 0.15% RTH / 0.40% extended, remainder carried to later
  bars through the fill horizon (last cached bar of the exit day, <= 11:00) and any residue
  sold at the OPEN of that last bar.  Never at or before the decision bar.  If there is no bar after
  the decision the trade is unfilled and ret = NaN.
* Fills (legacy, compatibility with pit_runner._exit_fill): fill starts at the trigger bar's CLOSE.
"""


def _info(w):
    """Cached per-window indices (all derived from timestamps only)."""
    inf = getattr(w, "_info_cache", None)
    if inf is None:
        dates, minutes = _clock(w)
        fn = _first_next_rth(w)
        if fn is None:
            nxt = np.array([], dtype=int)
            horizon = None
        else:
            nxt = np.flatnonzero(dates == dates[fn])
            ok = nxt[minutes[nxt] <= 660]
            horizon = int(ok[-1]) if len(ok) else int(nxt[-1])
        inf = {"first_next": fn, "next_idx": nxt, "horizon": horizon}
        w._info_cache = inf
    return inf


def _label_minute(policy):
    if policy.get("decision_close") is not None:
        t = pd.Timestamp(policy["decision_close"])
        return t.hour * 60 + t.minute - 1
    t = pd.Timestamp(policy.get("deadline", "10:15"))
    return t.hour * 60 + t.minute


def close_to_label(close_hhmm):
    """Label (open time) of the 1-min bar that closes at the given wall-clock time."""
    t = pd.Timestamp(close_hhmm) - pd.Timedelta(minutes=1)
    return t.strftime("%H:%M")


def fixed_time_policy(close_hhmm, keep_ah=False):
    return {"name": "fixed_time", "keep_ah": keep_ah, "decision_close": close_hhmm}


def _deadline_idx(w, policy):
    inf = _info(w)
    if inf["first_next"] is None:
        return None
    _, minutes = _clock(w)
    nxt = inf["next_idx"]
    lm = _label_minute(policy)
    reg = nxt[(minutes[nxt] >= 570) & (minutes[nxt] <= lm)]
    return int(reg[-1]) if len(reg) else int(inf["first_next"])


def _rth_session_slice(w, i):
    """Indices of regular-session bars of bar i's date from 09:30 through i."""
    dates, minutes = _clock(w)
    return np.flatnonzero((dates[: i + 1] == dates[i]) & (minutes[: i + 1] >= 570) & (minutes[: i + 1] < 960))


def features_at(w, i, pre_window_start=PM_DEFAULT_START, peak_from="open"):
    """Causal features at close i; no array element after i is inspected."""
    c = w.closes[: i + 1]
    j = _rth_session_slice(w, i)
    t = _ts(w, i)
    oi = j[0] if len(j) else i
    v = w.volumes[: i + 1]
    ret = c[-1] / w.entry_price - 1
    if peak_from == "entry" or not len(j):
        peak = np.max(c)
    else:
        peak = np.max(w.closes[j])
    since_open = c[-1] / c[oi] - 1 if c[oi] else np.nan
    if len(j) > 1:
        n = min(5, len(j) - 1)
        velocity = w.closes[j[-1]] / w.closes[j[-1 - n]] - 1
    else:
        velocity = np.nan
    if len(j):
        tp = (w.highs[j] + w.lows[j] + w.closes[j]) / 3.0
        vv = np.maximum(w.volumes[j], 0.0)
        svwap = float((tp * vv).sum() / vv.sum()) if vv.sum() > 0 else np.nan
    else:
        svwap = np.nan
    m = np.mean(v[max(0, i - 20):i + 1])
    return {
        "return": float(ret), "since_open": float(since_open),
        "peak_return": float(peak / w.entry_price - 1),
        "drawdown": float(c[-1] / peak - 1) if peak else np.nan,
        "velocity": float(velocity), "volume_ratio": float(v[-1] / m) if m else 0.0,
        "minutes_since_open": max(0, int((t.hour * 60 + t.minute) - 570)),
        "close": float(c[-1]), "vwap": svwap,
    }


def model_features(w, i, peak_from="open"):
    """Numeric causal feature vector for learned and optimal-stopping policies."""
    f = features_at(w, i, peak_from=peak_from)
    return np.asarray([
        f["return"], f["since_open"], f["peak_return"], f["drawdown"],
        0.0 if not np.isfinite(f["velocity"]) else f["velocity"], f["volume_ratio"], f["minutes_since_open"],
    ], dtype=float)


def _events(w, policy):
    """Return [(decision_idx, fraction_of_original_position, reason), ...] using causal info only."""
    name = policy.get("name", "smart")
    ah_stop, ah_target = policy.get("ah_stop", 0.12), policy.get("ah_target", 0.50)
    pm_target, pm_stop = policy.get("pm_target"), policy.get("pm_stop")
    peak_from = policy.get("peak_from", "open")
    inf = _info(w)
    first_next = inf["first_next"]
    dl = _deadline_idx(w, policy)
    if first_next is None or dl is None:
        return []
    dates, minutes = _clock(w)
    lo = pd.Timestamp(policy.get("pm_start", "08:00")).time()
    lo = lo.hour * 60 + lo.minute
    decisions = [int(i) for i in inf["next_idx"] if minutes[i] >= lo and i <= dl]
    if not decisions:
        return []
    entry = w.entry_price
    if policy.get("keep_ah", True):
        for i in range(1, first_next):
            if dates[i] != dates[0] or minutes[i] < 960 or minutes[i] > 1005:
                continue
            chg = w.closes[i] / entry - 1
            if chg <= -ah_stop:
                return [(i, 1.0, "ah_stop")]
            if chg >= ah_target:
                return [(i, 1.0, "ah_target")]
    scale_out = list(policy.get("scale_out", ()))
    scaled = set()
    events = []
    sold = 0.0
    cummax_entry = np.maximum.accumulate(w.closes[: decisions[-1] + 1]) if peak_from == "entry" else None
    peak_open = -np.inf
    cum_pv = cum_v = 0.0
    rth_closes = []
    trail = policy.get("trail")
    activate = policy.get("activate", -1.0)
    model = policy.get("model")
    fixed = name in ("open_close", "fixed_time")

    def rest(reason, i):
        return events + [(i, max(0.0, 1.0 - sold), reason)]

    for i in decisions:
        c = float(w.closes[i])
        chg = c / entry - 1
        if minutes[i] < 570:
            if pm_target is not None and chg >= pm_target:
                return [(i, 1.0, "pm_target")]
            if pm_stop is not None and chg <= -pm_stop:
                return [(i, 1.0, "pm_stop")]
            continue
        # regular-session decision bar: update causal state
        peak_open = max(peak_open, c)
        vol = max(float(w.volumes[i]), 0.0)
        cum_pv += (float(w.highs[i]) + float(w.lows[i]) + c) / 3.0 * vol
        cum_v += vol
        rth_closes.append(c)
        for j, item in enumerate(scale_out):
            if j not in scaled and chg >= item["target"]:
                scaled.add(j)
                frac = min(float(item["fraction"]), max(0.0, 1.0 - sold))
                if frac > 0:
                    events.append((i, frac, item.get("reason", "scale_target")))
                    sold += frac
        if sold >= 1.0 - 1e-12:
            return events
        if fixed:
            if i == dl:
                return rest("fixed_time", i)
            continue
        if name == "legacy_v2":
            return rest("next_open", i)
        peak = peak_open if peak_from == "open" else float(cummax_entry[i])
        peak_ret = peak / entry - 1
        dd = c / peak - 1 if peak else np.nan
        if model is not None:
            prediction = float(model.predict(model_features(w, i, peak_from).reshape(1, -1))[0])
            if prediction <= float(policy.get("edge_threshold", 0.0)):
                return rest("model_stop", i)
        if trail is not None and peak_ret >= activate and dd <= -trail:
            return rest("trailing_stop", i)
        if policy.get("below_vwap") and cum_v > 0 and c < cum_pv / cum_v:
            return rest("vwap_loss", i)
        if len(rth_closes) > 1:
            n = min(5, len(rth_closes) - 1)
            vel = c / rth_closes[-1 - n] - 1
            if vel <= policy.get("velocity_floor", -1.0):
                return rest("momentum_loss", i)
        if dd <= -policy.get("giveback", 1.0):
            return rest("giveback", i)
        if i >= dl:
            return rest("deadline", i)
    return rest("deadline", decisions[-1])


def _fill(w, start, end, shares, model, used=None):
    """Sequential fill from bar `start` through `end`; returns (avg price, filled shares, last idx, first idx).

    Each bar supplies at most PARTICIPATION*volume (minus volume already used by earlier tranches
    in `used`).  Any residue is sold at bar `end` (open for realistic, close for legacy)."""
    if start is None or end is None or start >= len(w.times) or start > end:
        return np.nan, 0.0, None, None
    end = min(end, len(w.times) - 1)
    used = {} if used is None else used
    got = cost = 0.0
    last = first = None
    price = (lambda i: w.closes[i]) if model == "legacy" else (lambda i: w.opens[i])
    for i in range(start, end + 1):
        cap = PARTICIPATION * max(float(w.volumes[i]), 0.0) - used.get(i, 0.0)
        take = min(shares - got, cap)
        if take <= 0:
            continue
        slip = SLIP_RTH if _rth(w, i) else SLIP_EXT
        cost += take * price(i) * (1 - slip)
        used[i] = used.get(i, 0.0) + take
        got += take
        last = i
        first = i if first is None else first
        if got >= shares - 1e-12:
            break
    if got < shares - 1e-12:
        i = end
        slip = SLIP_RTH if _rth(w, i) else SLIP_EXT
        cost += (shares - got) * price(i) * (1 - slip)
        got = shares
        last = i
        first = i if first is None else first
    return cost / got if got else np.nan, got, last, first


def simulate_trade(w, policy=None, fill_model="realistic"):
    """Simulate one trade and return a serializable per-trade result."""
    policy = dict(policy or {})
    events = _events(w, policy)
    nan_row = {"trade_id": w.trade_id, "symbol": w.symbol, "ret": np.nan, "exit_time": None, "exit_price": np.nan,
               "exit_idx": -1, "decision_idx": -1, "first_fill_idx": -1, "filled_fraction": 0.0}
    if not events:
        return {**nan_row, "reason": "no_bars"}
    # sizing as pit_runner: size_usd / signal price, times the entry fill fraction
    shares = max(SIZE_USD / w.entry_signal_price * w.fill_entry if w.entry_signal_price else 1.0, 1e-9)
    horizon = _info(w)["horizon"]
    remaining = shares
    proceeds = 0.0
    reasons = []
    used = {}
    last_idx, first_fill = -1, -1
    for idx, fraction, reason in events:
        take = min(remaining, shares * min(max(fraction, 0.0), 1.0))
        if take <= 1e-12:
            continue
        start = idx if fill_model == "legacy" else idx + 1
        px, got, last, first = _fill(w, start, horizon, take, fill_model, used)
        if got:
            proceeds += got * px
            remaining -= got
            last_idx = max(last_idx, last)
            first_fill = first if first_fill < 0 else min(first_fill, first)
        reasons.append(reason)
    done = remaining <= 1e-8
    exit_px = proceeds / shares if done else np.nan
    ret = exit_px / w.entry_price - 1 - 2 * COMMISSION if done else np.nan
    return {
        "trade_id": w.trade_id, "symbol": w.symbol, "exit_time": _ts(w, last_idx) if last_idx >= 0 else None,
        "exit_price": exit_px, "ret": ret, "reason": "+".join(dict.fromkeys(reasons)),
        "exit_idx": last_idx, "decision_idx": events[-1][0], "first_fill_idx": first_fill,
        "filled_fraction": float((shares - remaining) / shares),
    }


def simulate(windows, policy=None, fill_model="realistic"):
    return pd.DataFrame([simulate_trade(w, policy, fill_model) for w in windows])



def baseline_policy():
    return {"name": "legacy_v2", "keep_ah": True, "ah_stop": 0.12, "ah_target": 0.50,
            "pm_target": 0.25, "pm_start": "08:00", "deadline": "10:15"}


def oracle_returns(windows, fill_model="realistic", deadline="10:15"):
    """Hindsight best close in the regular-session window, for an upper-bound diagnostic."""
    rows = []
    for w in windows:
        first, last = _window_bounds(w, "08:00", deadline)
        if first is None or last is None:
            continue
        inds = [i for i in range(first, last + 1) if T_OPEN <= _ts(w, i).time() <= pd.Timestamp(deadline).time()]
        if not inds:
            continue
        j = max(inds, key=lambda i: w.closes[i])
        x = simulate_trade(w, {"name": "fixed_time", "deadline": _ts(w, j).strftime("%H:%M"), "keep_ah": False}, fill_model)
        x["oracle_idx"] = j
        rows.append(x)
    return pd.DataFrame(rows)


def elapsed_logger(start_epoch, path, label):
    """Append elapsed wall time for long campaign workers."""
    elapsed = max(0.0, time.time() - float(start_epoch))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"label": label, "elapsed_seconds": elapsed}) + "\n")
    return elapsed
