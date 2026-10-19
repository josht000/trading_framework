"""
Chart the best / worst trades from a point-in-time backtest run.

  python chart_trades.py [--csv results/pit_trades_atr_pivot.csv] [--n 10] [--out data/charts/trades]

One PNG per trade: daily bars (with atr_pivot resistance levels known at the prior close) on top,
5-minute bars for the entry session and the next session with session VWAP below.
Entry/exit markers use the prices recorded in the trade log.
"""

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.screening.historical import ATR_PIVOT_CONFIG
from src.screening.resistance import ResistanceDetector

UP, DOWN, EXT = "#26a69a", "#ef5350", "#b0b0b0"
NY = "America/New_York"


def candles(ax, x, o, h, l, c, colors, width=0.7):
    ax.vlines(x, l, h, colors=colors, linewidth=0.9, zorder=3)
    body = np.maximum(np.abs(c - o), (np.nanmax(h) - np.nanmin(l)) * 0.0005)
    ax.bar(x, body, bottom=np.minimum(o, c), width=width, color=colors, zorder=4)


def five_min(symbol, signal_date):
    m = pd.read_parquet(f"data/pit/alpaca_minute/{signal_date}.parquet")
    m = m[m["symbol"] == symbol].copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True).dt.tz_convert(NY)
    m = m.set_index("timestamp").sort_index()
    five = m.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    five["day"] = five.index.date
    t = five.index.time
    five["rth"] = (t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())
    vw = []
    for _, g in five.groupby("day"):
        r = g[g["rth"]]
        tp = (r["high"] + r["low"] + r["close"]) / 3
        v = (tp * r["volume"]).cumsum() / r["volume"].cumsum()
        vw.append(v.reindex(g.index))
    five["vwap"] = pd.concat(vw)
    return five


def chart_trade(row, daily, det, path, label):
    sym = row["symbol"]
    entry_ts = pd.Timestamp(row["entry_time"])
    sig = pd.Timestamp(entry_ts.tz_localize(None).date() if entry_ts.tzinfo is None else entry_ts.tz_convert(NY).date())
    d = daily[daily["symbol"] == sym].sort_values("date").set_index("date")
    pos = d.index.searchsorted(sig)
    lo, hi = max(0, pos - 59), min(len(d), pos + 3)
    dd = d.iloc[lo:hi]
    x = np.arange(len(dd))
    sig_x = pos - lo

    prep = det.prepare(d[["open", "high", "low", "close", "volume"]])
    k = det.known_through_before(prep, sig)
    prev_close = float(d["close"].iloc[pos - 1]) if pos > 0 else float("nan")
    lad = det.ladder(prep, k, price=prev_close) if k >= 0 else pd.DataFrame(columns=["upper"])

    fig = plt.figure(figsize=(15, 11), dpi=100)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 1], hspace=0.22)
    ax = fig.add_subplot(gs[0])
    cols = np.where(dd["close"] >= dd["open"], UP, DOWN)
    candles(ax, x, dd["open"].values, dd["high"].values, dd["low"].values, dd["close"].values, cols)
    ax.axvspan(sig_x - 0.5, sig_x + 0.5, color="#fff3c4", zorder=0)
    if sig_x + 1 < len(dd):
        ax.axvspan(sig_x + 0.5, sig_x + 1.5, color="#e3f0ff", zorder=0)
    y_hi = float(dd["high"].max() * 1.05)
    y_lo = float(dd["low"].min() * 0.95)
    for u in lad["upper"] if len(lad) else []:
        if y_lo <= u <= y_hi:
            crossed = prev_close < u < float(d["close"].iloc[pos])
            ax.axhline(u, color="#ff8c00" if crossed else "#9a9a9a", linestyle="--", linewidth=1.2 if crossed else 0.8, zorder=2)
            ax.text(len(dd) - 0.4, u, f"{u:.2f}", va="center", fontsize=8, color="#ff8c00" if crossed else "#707070")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-1, len(dd) + 1.2)
    ax.set_xticks(x[::6])
    ax.set_xticklabels([dd.index[i].strftime("%Y-%m-%d") for i in range(0, len(dd), 6)], fontsize=7)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.6)
    ax.set_title("daily (yellow = signal day, blue = next day; orange line = atr_pivot level crossed)", fontsize=9, loc="left")

    f = five_min(sym, sig.date())
    days = sorted(f["day"].unique())[:2]
    f = f[f["day"].isin(days)]
    xs, ticks, labels, gap = [], [], [], 0
    pos_map = {}
    for i, dday in enumerate(days):
        g = f[f["day"] == dday]
        idx = np.arange(len(g)) + gap
        pos_map[dday] = (g.index, idx)
        xs.append(idx)
        ticks.append(idx[0])
        labels.append(str(dday))
        gap = idx[-1] + 6
    ax2 = fig.add_subplot(gs[1])
    axv = fig.add_subplot(gs[2], sharex=ax2)
    for dday in days:
        g = f[f["day"] == dday]
        _, idx = pos_map[dday]
        c = np.where(g["rth"], np.where(g["close"] >= g["open"], UP, DOWN), EXT)
        candles(ax2, idx, g["open"].values, g["high"].values, g["low"].values, g["close"].values, c, width=0.75)
        ax2.plot(idx, g["vwap"].values, color="#1f4fd8", linewidth=1.3, zorder=5, label="session VWAP (RTH)")
        axv.bar(idx, g["volume"].values, width=0.75, color=c)
        ax2.axvline(idx[0] - 3, color="#cccccc", linewidth=0.8)
    h, l = ax2.get_legend_handles_labels()
    ax2.legend(h[:1], l[:1], loc="upper left", fontsize=8, frameon=False)
    ax2.set_xticks(sorted(int(i) for dday in days for i in pos_map[dday][1][::24]))
    ent_time = entry_ts if entry_ts.tzinfo is not None else entry_ts.tz_localize(NY)
    ent_time = ent_time.tz_convert(NY)
    ex_time = pd.Timestamp(row["exit_time"])
    ex_time = ex_time.tz_convert(NY) if ex_time.tzinfo is not None else ex_time.tz_localize(NY)
    for tm, price, color, name in ((ent_time, row["entry"], "#0a7d2c", "entry"), (ex_time, row["exit"], "#c0151f", "exit")):
        if tm in f.index or tm.floor("5min") in f.index:
            t5 = tm if tm in f.index else tm.floor("5min")
            g = f[f["day"] == t5.date()]
            _, idx = pos_map[t5.date()]
            xi = idx[list(g.index).index(t5)]
            ax2.scatter([xi], [price], marker="^" if name == "entry" else "v", s=90, color=color, zorder=8, label=name)
            ax2.annotate(f"{name} {price:.2f}", (xi, price), textcoords="offset points", xytext=(6, 8 if name == "entry" else -14),
                         fontsize=8, color=color, fontweight="bold")
    labs = []
    for t in ax2.get_xticks():
        txt = ""
        for dday in days:
            gi, idx = pos_map[dday]
            if t in idx:
                txt = gi[list(idx).index(t)].strftime("%m-%d %H:%M")
        labs.append(txt)
    ax2.set_xticklabels(labs, fontsize=7, rotation=0)
    ax2.set_xlim(-3, gap)
    ax2.grid(axis="y", color="#e8e8e8", linewidth=0.6)
    ax2.set_title("5-minute (gray = extended hours)", fontsize=9, loc="left")
    plt.setp(axv.get_xticklabels(), visible=False)
    axv.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}K"))
    axv.tick_params(labelsize=7)

    nxt = d.iloc[pos + 1] if pos + 1 < len(d) else None
    gap_txt = f"  next-day open vs signal close: {nxt['open'] / d['close'].iloc[pos] - 1:+.1%}" if nxt is not None else ""
    fig.suptitle(f"{label}  {sym}  signal {sig.date()}  ret {row['ret']:+.1%}  entry {row['entry']:.2f} -> exit {row['exit']:.2f}"
                 f"{gap_txt}  exit={row.get('exit_reason', 'next_open')}"
                 f"  {row['flag'] if isinstance(row['flag'], str) else ''}", fontsize=11, y=0.96)
    fig.subplots_adjust(left=0.05, right=0.96, top=0.92, bottom=0.05)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/pit_trades_atr_pivot_v2.csv")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="data/charts/trades/v2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t = pd.read_csv(a.csv, parse_dates=["entry_time", "exit_time"])
    daily = pd.read_parquet("data/pit/daily_pit.parquet", columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    det = ResistanceDetector(ATR_PIVOT_CONFIG)
    sel = [("best", r) for _, r in t.nlargest(a.n, "ret").iterrows()] + [("worst", r) for _, r in t.nsmallest(a.n, "ret").iterrows()]
    html = ["<html><body style='font-family:sans-serif'><h2>Best / worst trades</h2>"]
    for i, (kind, r) in enumerate(sel):
        name = f"{kind}_{i % a.n + 1:02d}_{r['symbol']}_{str(r['entry_time'])[:10]}.png"
        try:
            chart_trade(r, daily, det, os.path.join(a.out, name), f"{kind.upper()} #{i % a.n + 1}")
            html.append(f"<h3>{name}</h3><img src='{name}' style='max-width:100%'>")
            print("wrote", name, f"{r['ret']:+.1%}")
        except Exception as e:
            print("FAILED", r["symbol"], r["entry_time"], repr(e))
    open(os.path.join(a.out, "index.html"), "w").write("\n".join(html) + "</body></html>")


if __name__ == "__main__":
    main()
