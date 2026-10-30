"""One-PNG, fully labelled dashboard for daily equal-weight book studies (see daily_partition)."""

import os

import numpy as np
import pandas as pd

FILL_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
METRIC_ROWS = [("trades", "n trades", "{:.0f}"), ("total_log", "total (log)", "{:+.3f}"), ("trend", "trend (log)", "{:+.3f}"),
               ("random_pick_pool", "random pool", "{:+.3f}"), ("bias", "bias", "{:.3f}"), ("skill", "skill", "{:+.3f}"),
               ("skill_pct", "skill % of total", "{:.0%}"), ("skill_p", "p(skill<=0)", "{:.2f}"), ("sharpe", "Sharpe", "{:+.2f}"), ("pf", "profit factor", "{:.2f}"), ("win_pct", "win rate %", "{:.1f}"),
               ("cagr_pct", "CAGR %", "{:+.1f}"), ("max_dd_pct", "max DD %", "{:.1f}"), ("avg_positions", "avg positions", "{:.0f}"),
               ("avg_fraction_invested", "avg invested", "{:.0%}")]


def _fmt(fmt, value):
    return "n/a" if value is None or not np.isfinite(value) else fmt.format(value)


def metrics_text(frame):
    cols = [(r["fill"], r["period"]) for _, r in frame.iterrows()]
    head = f"{'metric':<18}" + "".join(f"{f[:9]}|{p:<5}".rjust(16) for f, p in cols)
    lines = [head]
    for key, label, fmt in METRIC_ROWS:
        lines.append(f"{label:<18}" + "".join(_fmt(fmt, r[key]).rjust(16) for _, r in frame.iterrows()))
    return "\n".join(lines)


def daily_dashboard(path, title, caption, dates, book_series, frame, trades, b_day, first_row, split_row, n_trials_note=""):
    """book_series: fill -> {r_day, f_day, count, random_mean_log: {pool, universe}}; frame: results_frame(); trades: fill -> DataFrame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fills = list(book_series)
    color = {f: FILL_COLORS[i % len(FILL_COLORS)] for i, f in enumerate(fills)}
    x = pd.DatetimeIndex(dates)[first_row:]
    split_x = x[min(split_row - first_row, len(x) - 1)]
    fig = plt.figure(figsize=(20, 22))
    gs = fig.add_gridspec(4, 3, hspace=0.32, wspace=0.22, top=0.93, bottom=0.04, left=0.05, right=0.98)
    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.text(0.5, 0.945, caption, ha="center", va="bottom", fontsize=8.5, family="monospace", color="#333333")

    ax = fig.add_subplot(gs[0, :2])
    for fill in fills:
        s = book_series[fill]
        path_strategy = np.cumsum(np.log1p(s["r_day"]))[first_row:]
        ax.plot(x, path_strategy, color=color[fill], lw=1.8, label=f"strategy [{fill}] net of costs (end {path_strategy[-1]:+.2f})")
        rnd = np.cumsum(s["random_mean_log"]["pool"])[first_row:]
        ax.plot(x, rnd, color=color[fill], lw=1.1, ls=":", label=f"random-pick pool mean [{fill}] (end {rnd[-1]:+.2f})")
    ref = book_series[fills[-1]]
    trend = np.cumsum(ref["f_day"] * np.log1p(b_day))[first_row:]
    ax.plot(x, trend, color="#555555", lw=1.4, ls="--", label=f"trend = exposure-weighted equal-weight liquid universe [{fills[-1]}] (end {trend[-1]:+.2f})")
    ax.axvspan(x[0], split_x, color="#e8f0fe", alpha=0.5, lw=0)
    ax.axvspan(split_x, x[-1], color="#fdecea", alpha=0.5, lw=0)
    ax.axvline(split_x, color="#888888", lw=1)
    ymax = ax.get_ylim()[1]
    ax.text(x[0], ymax, " DEV (selection period)", va="top", fontsize=11, color="#1a4f9c")
    ax.text(split_x, ymax, " TEST (locked out-of-sample)", va="top", fontsize=11, color="#a32620")
    ax.set_title("A. Cumulative log return of the equal-weight book vs trend and random-pick baseline")
    ax.set_ylabel("cumulative log return")
    ax.set_xlabel("session date")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(color="#dddddd")

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.set_title("B. Key metrics (log returns, research scale, net of costs)", fontsize=10)
    ax.text(0, 1, metrics_text(frame), va="top", ha="left", family="monospace", fontsize=8, transform=ax.transAxes)

    ax = fig.add_subplot(gs[1, :2])
    groups = [(r["fill"], r["period"]) for _, r in frame.iterrows()]
    for i, (_, r) in enumerate(frame.iterrows()):
        pos = neg = 0.0
        for name, value, col in (("trend", r["trend"], "#4472c4"), ("skill", r["skill"], "#70ad47"), ("bias", r["bias"], "#d9534f")):
            if not np.isfinite(value):
                continue
            base = pos if value >= 0 else neg
            ax.bar(i, value, bottom=base, color=col, width=0.6, label=name if i == 0 else None)
            if abs(value) > 0.03:
                ax.text(i, base + value / 2, f"{name}\n{value:+.2f}", ha="center", va="center", fontsize=8, color="white")
            pos, neg = (pos + value, neg) if value >= 0 else (pos, neg + value)
        ax.plot([i], [r["total_log"]], "ko", ms=9, label="total" if i == 0 else None)
        ax.text(i + 0.33, r["total_log"], f"total {r['total_log']:+.2f}", fontsize=8, va="center")
        pct = r["skill_pct"]
        ax.text(i, ax.get_ylim()[0], f"skill {_fmt('{:.0%}', pct)} of total\np={_fmt('{:.2f}', r['skill_p'])}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{f}\n{p}" for f, p in groups])
    ax.axhline(0, color="#888888", lw=0.8)
    ax.margins(y=0.18)
    ax.set_title("C. Partition test: total = trend + skill + bias (log return per period; dev bias = selection haircut, test bias = 0)")
    ax.set_ylabel("log return over period")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", color="#eeeeee")

    ax = fig.add_subplot(gs[1, 2])
    for fill in fills:
        eq = np.cumprod(1 + book_series[fill]["r_day"][first_row:])
        dd = 100 * (eq / np.maximum.accumulate(eq) - 1)
        ax.plot(x, dd, color=color[fill], lw=1.2, label=f"{fill} (max {dd.min():.1f}%)")
    ax.axvline(split_x, color="#888888", lw=1)
    ax.set_title("D. Drawdown of the strategy equity curve")
    ax.set_ylabel("drawdown from peak, %")
    ax.set_xlabel("session date")
    ax.legend(fontsize=8)
    ax.grid(color="#dddddd")

    ax = fig.add_subplot(gs[2, :2])
    years = pd.DatetimeIndex(dates).year
    uniq = sorted(set(years[first_row:]))
    width = 0.8 / max(len(fills), 1)
    for i, fill in enumerate(fills):
        r = book_series[fill]["r_day"]
        vals = [100 * np.expm1(np.log1p(r[(years == y) & (np.arange(len(r)) >= first_row)]).sum()) for y in uniq]
        bars = ax.bar(np.arange(len(uniq)) + i * width, vals, width, color=color[fill], label=fill)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax.set_xticks(np.arange(len(uniq)) + width * (len(fills) - 1) / 2)
    ax.set_xticklabels([f"{y}{' (test)' if y >= split_x.year else ''}" for y in uniq])
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_title("E. Calendar-year return of the strategy account (percent, net of costs)")
    ax.set_ylabel("return, %")
    ax.legend(fontsize=8)
    ax.grid(axis="y", color="#eeeeee")

    ax = fig.add_subplot(gs[2, 2])
    for fill in fills:
        t = trades[fill]["ret"].to_numpy(float) * 100
        ax.hist(np.clip(t, -25, 25), bins=np.linspace(-25, 25, 51), alpha=0.5, color=color[fill],
                label=f"{fill}: n={len(t)}, mean {t.mean():+.2f}%, median {np.median(t):+.2f}%, win {np.mean(t > 0):.0%}")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("F. Net return per trade (clipped to +-25%)")
    ax.set_xlabel("trade return, %")
    ax.set_ylabel("number of trades")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[3, 0])
    max_hold = int(max(trades[f]["hold"].max() for f in fills))
    for i, fill in enumerate(fills):
        counts = np.bincount(trades[fill]["hold"].to_numpy(int), minlength=max_hold + 1)
        ax.bar(np.arange(max_hold + 1) + i * width, counts, width, color=color[fill], label=f"{fill} (mean {trades[fill]['hold'].mean():.1f})")
    ax.set_title("G. Holding period")
    ax.set_xlabel("sessions held (entry to exit)")
    ax.set_ylabel("number of trades")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[3, 1])
    reasons = sorted(set().union(*[set(trades[f]["exit_reason"]) for f in fills]))
    for i, fill in enumerate(fills):
        counts = trades[fill]["exit_reason"].value_counts()
        vals = [int(counts.get(r, 0)) for r in reasons]
        bars = ax.bar(np.arange(len(reasons)) + i * width, vals, width, color=color[fill], label=fill)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(reasons)) + width * (len(fills) - 1) / 2)
    ax.set_xticklabels(reasons)
    ax.set_title("H. Exit reason (resid = signal recovery, cap = max hold, stop/target = price rules)")
    ax.set_ylabel("number of trades")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[3, 2])
    for fill in fills:
        n = pd.Series(book_series[fill]["count"][first_row:]).rolling(20, min_periods=1).mean().to_numpy()
        ax.plot(x, n, color=color[fill], lw=1.2, label=fill)
    ax.axvline(split_x, color="#888888", lw=1)
    ax.set_title("I. Open positions (20-session average)")
    ax.set_ylabel("number of open positions")
    ax.set_xlabel("session date")
    ax.legend(fontsize=8)
    ax.grid(color="#dddddd")
    if n_trials_note:
        fig.text(0.5, 0.01, n_trials_note, ha="center", fontsize=9, color="#555555")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
