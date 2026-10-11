"""
Static PNG chart renderer (matplotlib, no browser needed) for the VLM benchmark.

Same content as the interactive charts a human labeled on: daily candles with
volume, price axis on both sides with gridlines every $0.25 (widened
only when that would exceed MAX_TICKS), optional horizontal level lines with price
tags placed to the right of the last bar. Moving averages are not drawn (show_ma=True
restores SMA20/50/200).

The y-range comes from the bars only, so the with-labels and without-labels versions
of a chart share an identical frame and the blank one leaks nothing about the levels.
"""

from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from src.screening.resistance import PreparedSeries

TICK_STEPS = (0.25, 0.5, 1, 2, 5, 10, 25, 50)
MAX_TICKS = 70
PAD_BARS = 12
UP, DOWN = "#26a69a", "#ef5350"
MA_COLORS = ("#1f77b4", "#ff7f0e", "#7f7f7f")


def tick_step(y_lo: float, y_hi: float) -> float:
    return next((s for s in TICK_STEPS if (y_hi - y_lo) / s <= MAX_TICKS), TICK_STEPS[-1])


def level(price: float, color: str = "#1f77b4", style: str = "--", width: float = 1.6, tag: Optional[str] = None) -> Dict:
    return {"price": float(price), "color": color, "style": style, "width": width,
            "tag": f"{price:.2f}" if tag is None else tag}


def render_chart_png(p: PreparedSeries, known_through: int, path: str, symbol: str = "",
                     lookback: int = 160, levels: Optional[List[Dict]] = None, title: Optional[str] = None,
                     size=(16, 10), dpi: int = 100, show_ma: bool = False) -> tuple:
    """Draw bars [k-lookback .. k] and save to path. Returns (width_px, height_px, n_bars, first_date)."""
    k = known_through
    lo = max(0, k - lookback)
    sl = slice(lo, k + 1)
    n = k - lo + 1
    xs = np.arange(n)
    o, h, l, c, v = p.open[sl], p.high[sl], p.low[sl], p.close[sl], p.volume[sl]
    dates = pd.to_datetime(p.dates[sl])

    fig, (ax, axv) = plt.subplots(2, 1, figsize=size, dpi=dpi, sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1], "hspace": 0.04})
    up = c >= o
    colors = np.where(up, UP, DOWN)
    ax.vlines(xs, l, h, colors=colors, linewidth=1.0, zorder=3)
    body = np.maximum(np.abs(c - o), (h.max() - l.min()) * 0.0005)
    ax.bar(xs, body, bottom=np.minimum(o, c), width=0.7, color=colors, zorder=4)

    if show_ma:
        for (period, series), color in zip(p.sma.items(), MA_COLORS):
            ax.plot(xs, series[sl], color=color, linewidth=1.0, alpha=0.85, zorder=2, label=f"SMA{period}")

    y_lo, y_hi = float(l.min() * 0.95), float(h.max() * 1.08)
    step = tick_step(y_lo, y_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-1, n + PAD_BARS)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", which="both", left=True, right=True, labelleft=True, labelright=True, labelsize=7)
    ax.grid(axis="y", color="#e2e2e2", linewidth=0.6, zorder=0)
    ax.grid(axis="x", color="#f0f0f0", linewidth=0.6, zorder=0)
    if show_ma:
        ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)

    for lv in levels or []:
        if not (y_lo <= lv["price"] <= y_hi):
            continue
        ax.hlines(lv["price"], -1, n + PAD_BARS - 1, colors=lv["color"], linestyles=lv["style"],
                  linewidth=lv["width"], zorder=5)
        ax.text(n + 0.4, lv["price"], lv["tag"], color=lv["color"], fontsize=9, fontweight="bold",
                va="center", ha="left", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.8))

    axv.bar(xs, v, width=0.7, color=colors, zorder=3)
    axv.grid(axis="y", color="#eeeeee", linewidth=0.6)
    axv.tick_params(axis="y", which="both", left=True, right=True, labelleft=True, labelright=True, labelsize=7)
    axv.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}K"))

    tick_pos = list(range(0, n, 10))
    axv.set_xticks(tick_pos)
    axv.set_xticklabels([dates[i].strftime("%Y-%m-%d") for i in tick_pos], fontsize=7)
    axv.set_xlabel("daily bars (weekends and holidays removed)", fontsize=7)

    fig.suptitle(title or f"{symbol} as of {dates[-1].date()}  price={c[-1]:.2f}  ATR14={p.atr[k]:.2f}", fontsize=11, y=0.95)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.07)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return int(size[0] * dpi), int(size[1] * dpi), n, dates[0]
