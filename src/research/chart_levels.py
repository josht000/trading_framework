"""
Interactive candlestick + volume chart with detected/labeled levels, for visual
inspection the way a day trader would read the chart.

Each level is one horizontal line with a short tag on the right ("<id> <price>").
Hover a line for status, kind, score, source and note. When a label file has rows for
this (symbol, asof) the chart draws exactly those rows, so a line and its CSV row share
the same id. Otherwise it draws the detector's current levels, numbered from the
highest price down. Use live=True to draw the detector's current levels as well.

Colors: orange = unreviewed (auto), blue dashed = keep/add (gold), grey dotted = rejected,
red = live resistance, green = live support. Bars after the as-of date are drawn faded so
you can see what happened without it affecting the levels.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.screening.resistance import LEVEL_COLUMNS, ResistanceDetector

STATUS_STYLES = {
    "auto": ("230,126,34", "solid"),
    "keep": ("31,119,180", "dash"),
    "add": ("31,119,180", "dash"),
    "reject": ("130,130,130", "dot"),
}
LIVE_STYLES = {"resistance": ("220,50,47", "solid"), "support": ("38,139,70", "solid"),
               "in_zone": ("230,160,20", "solid")}


SOURCE_COLORS = {"atr_pivot": "230,126,34", "volume_profile": "148,103,189", "manual": "31,119,180"}

TICK_STEPS = (0.25, 0.5, 1, 2, 5, 10, 25, 50)
MAX_TICKS = 70
PAD_BARS = 12


def _add_level(fig, x0, x1, y, rgb, dash, tag, hover, group, shown, width=1.4, opacity=0.9):
    """One level = one line trace whose last point carries the tag, so legend toggling hides both."""
    first = group not in shown
    shown.add(group)
    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[y, y], mode="lines+text", text=["", tag], textposition="middle right",
        textfont=dict(size=11, color=f"rgb({rgb})"), cliponaxis=False,
        line=dict(color=f"rgba({rgb},{opacity})", width=width, dash=dash),
        hoverinfo="text", hovertext=[hover, hover], name=group, legendgroup=group,
        showlegend=first), row=1, col=1)


def plot_levels(daily: pd.DataFrame, asof_date, detector=None, symbol: str = "",
                lookback: int = 160, forward: int = 15, show_ma: bool = False,
                path: Optional[str] = None, labels: Optional[pd.DataFrame] = None,
                live: bool = False, detector_name: str = "", blank: bool = False) -> go.Figure:
    """blank=True is for blind manual labeling: the detector is never run or drawn, and the title
    carries no detector information. Reviewed label rows are still drawn if given."""
    detector = detector or ResistanceDetector()
    p = detector.prepare(daily)
    k = detector.known_through_on(p, asof_date)
    if k < 0:
        raise ValueError(f"No bars on or before {asof_date}")
    levels = pd.DataFrame(columns=LEVEL_COLUMNS) if blank else detector.levels(p, k)

    lo_pos = max(0, k - lookback)
    hi_pos = min(len(p.close) - 1, k + forward)
    sl = slice(lo_pos, hi_pos + 1)
    dates = pd.to_datetime(p.dates[sl])
    n_past = k - lo_pos + 1

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03,
                        specs=[[{"secondary_y": True}], [{}]])

    def candles(a, b, up, down, name, opacity):
        fig.add_trace(go.Candlestick(
            x=dates[a:b], open=p.open[sl][a:b], high=p.high[sl][a:b], low=p.low[sl][a:b],
            close=p.close[sl][a:b], name=name, opacity=opacity, showlegend=False,
            increasing_line_color=up, decreasing_line_color=down), row=1, col=1)

    candles(0, n_past, "#26a69a", "#ef5350", "known", 1.0)
    if hi_pos > k:
        candles(n_past, len(dates), "#8fd3cb", "#f5a9a7", "after as-of", 0.6)

    vol_colors = np.where(p.close[sl] >= p.open[sl], "#26a69a", "#ef5350")
    vol_colors = np.array([c if i < n_past else "#bbbbbb" for i, c in enumerate(vol_colors)])
    fig.add_trace(go.Bar(x=dates, y=p.volume[sl], marker_color=vol_colors, name="volume", showlegend=False),
                  row=2, col=1)

    if show_ma:
        for period, color in zip(p.sma, ("#1f77b4", "#ff7f0e", "#7f7f7f")):
            fig.add_trace(go.Scatter(x=dates, y=p.sma[period][sl], mode="lines", hoverinfo="skip",
                                     line=dict(width=1, color=color), name=f"SMA{period}", showlegend=False),
                          row=1, col=1)

    x0, x_end = dates[0], dates[-1]
    x_tag = x_end + pd.offsets.BDay(1)
    x_pad = x_end + pd.offsets.BDay(PAD_BARS)
    view_lo, view_hi = p.low[sl].min() * 0.9, p.high[sl].max() * 1.15
    file_rows = labels if labels is not None and not labels.empty else None
    shown = set()

    if file_rows is not None:
        for r in file_rows.sort_values("level", ascending=False).itertuples(index=False):
            if r.level < view_lo or r.level > view_hi or r.status not in STATUS_STYLES:
                continue
            rgb, dash = STATUS_STYLES[r.status]
            if r.status == "auto":
                rgb = SOURCE_COLORS.get(r.source, rgb)
                group = f"{r.source} (unreviewed)"
            else:
                group = {"keep": "keep", "add": "add (manual)", "reject": "rejected"}[r.status]
            rid = f"{int(r.id)} " if pd.notna(r.id) else ""
            hover = (f"id {rid.strip() or '-'} | {r.level:.2f} | {r.status} | {r.kind} | score {r.score} | "
                     f"{r.source}" + (f" | {r.note}" if isinstance(r.note, str) and r.note else ""))
            _add_level(fig, x0, x_tag, r.level, rgb, dash, f"{rid}{r.level:.2f}", hover, group, shown,
                       width=1.8 if r.status in ("keep", "add") else 1.2)

    if (file_rows is None or live) and not blank:
        ordered = levels.sort_values("center", ascending=False).reset_index(drop=True)
        for i, z in enumerate(ordered.itertuples(index=False), 1):
            if z.upper < view_lo or z.lower > view_hi:
                continue
            rgb, dash = LIVE_STYLES[z.role]
            tag = f"{i} {z.upper:.2f}" if z.lower == z.upper else f"{i} {z.lower:.2f}-{z.upper:.2f}"
            hover = (f"live #{i} | {z.role} | {z.kind} | score {z.score:.1f} | pivots {z.n_pivots} | "
                     f"touches {z.touches} | {z.distance_atr:.1f} ATR away")
            _add_level(fig, max(z.first_date, x0), x_tag, z.upper, rgb, dash, tag, hover, f"live {z.role}",
                       shown, width=1 + 2.0 * z.score / max(ordered["score"].max(), 1e-9), opacity=0.8)

    fig.add_vline(x=pd.Timestamp(p.dates[k]), line=dict(color="#333", width=1, dash="dash"))
    all_days = pd.date_range(dates[0], x_pad, freq="D")
    missing = [d.strftime("%Y-%m-%d") for d in all_days.difference(dates)
               if d.weekday() >= 5 or d <= x_end]
    fig.update_xaxes(rangebreaks=[dict(values=missing)], rangeslider_visible=False,
                     range=[x0 - pd.Timedelta(days=1), x_pad])

    n_res = int((levels["role"] == "resistance").sum()) if len(levels) else 0
    shown = f"{len(file_rows)} labeled rows" if file_rows is not None else f"{len(levels)} live levels"
    y_lo, y_hi = float(p.low[sl].min() * 0.95), float(p.high[sl].max() * 1.08)
    step = next((s for s in TICK_STEPS if (y_hi - y_lo) / s <= MAX_TICKS), TICK_STEPS[-1])
    axis = dict(range=[y_lo, y_hi], dtick=step, tick0=0, tickformat=".2f", tickfont=dict(size=10))
    fig.update_yaxes(**axis, showgrid=True, gridcolor="#e6e6e6", showspikes=True, spikemode="across",
                     spikesnap="cursor", spikethickness=1, spikecolor="#888",
                     row=1, col=1, secondary_y=False)
    fig.update_yaxes(**axis, showgrid=False, matches="y", row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=[x0, x_end], y=[y_lo, y_hi], mode="markers", marker=dict(opacity=0),
                             hoverinfo="skip", showlegend=False), row=1, col=1, secondary_y=True)
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikecolor="#888")
    if blank:
        title = (f"{symbol} as of {pd.Timestamp(p.dates[k]).date()}  price={p.close[k]:.2f}  "
                 f"ATR14={p.atr[k]:.2f}" + (f"  {len(file_rows)} levels labeled" if file_rows is not None else ""))
    else:
        title = (f"{symbol} as of {pd.Timestamp(p.dates[k]).date()}  price={p.close[k]:.2f}  ATR14={p.atr[k]:.2f}  "
                 f"[{detector_name or type(detector).__name__}]  {shown}  resistance above: {n_res}"
                 f"{'  (blue sky)' if n_res == 0 else ''}")
    fig.update_layout(
        title=title,
        height=820, xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        margin=dict(l=50, r=50, t=90, b=30))

    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn")
    return fig
