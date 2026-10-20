"""
Portfolio performance plots for a named strategy from explore.py (STRATEGIES) on a $5,000 account.

  python plot_portfolio.py S1              # dev period
  python plot_portfolio.py S1 --with-test  # dev + the locked final test period (shaded)

Panels: equity ($) for the strategy, the unfiltered base strategy, and SPY / IWM buy-and-hold; drawdown;
daily P&L with trades per day; rolling 50-trade hit rate. Up to 5 overnight trades a day, 20% of equity each,
so gross exposure is at most 100%.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from explore import METHOD, STRATEGIES, apply_spec
from src.backtest.market_regime import attach_regime
from src.backtest.pit_metrics import dedupe_trades, select_trades, simulate_account

START = 5000.0


def load(variant, holdout):
    p = os.path.join("results", "holdout" if holdout else "", f"pit_trades_{METHOD}_{variant}.csv")
    t = pd.read_csv(p)
    t["entry_time"] = t["entry_time"].astype(str)
    return attach_regime(dedupe_trades(t))


def market_curve(symbol, days):
    d = pd.read_parquet(os.path.join("data", "pit", "market", f"{symbol}_daily.parquet")).set_index("date")["close"]
    d = d.reindex(days.union(d.index)).ffill().reindex(days)
    return START * d / d.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--with-test", action="store_true")
    ap.add_argument("--out", default=os.path.join("data", "charts", "portfolio"))
    a = ap.parse_args()
    spec = STRATEGIES[a.name]
    parts = [load(spec["variant"], False)]
    if a.with_test:
        parts.append(load(spec["variant"], True))
    t = pd.concat(parts, ignore_index=True)
    day = pd.to_datetime(t["entry_time"].astype(str).str[:10])
    days = pd.bdate_range(day.min(), day.max())
    test_start = pd.to_datetime(parts[1]["entry_time"].astype(str).str[:10]).min() if a.with_test else None

    strat = apply_spec(t, spec)
    base = apply_spec(t, STRATEGIES["base"])
    st, eq, dr = simulate_account(strat, start=START, all_days=days)
    bst, beq, bdr = simulate_account(base, start=START, all_days=days)
    spy, iwm = market_curve("SPY", days), market_curve("IWM", days)
    sel = select_trades(strat)

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), dpi=100, sharex=False,
                             gridspec_kw={"height_ratios": [3, 1.3, 1.5, 1.3], "hspace": 0.38})
    ax = axes[0]
    ax.plot(eq.index, eq.values, color="#7b2cbf", linewidth=2.2, label=f"{a.name} ({spec['variant']}; {' & '.join(spec['keys'])})")
    ax.plot(beq.index, beq.values, color="#808080", linewidth=1.4, label="base (no entry filters)")
    ax.plot(spy.index, spy.values, color="#1f6fe5", linewidth=1.2, alpha=0.8, label="SPY buy & hold")
    ax.plot(iwm.index, iwm.values, color="#d4a017", linewidth=1.2, alpha=0.8, label="IWM buy & hold")
    ax.axhline(START, color="#bbbbbb", linewidth=0.8, linestyle="--")
    ax.set_ylabel(r"equity (\$)")
    ax.set_title((f"{a.name}: ${START:,.0f} start -> ${st['final$']:,.0f} ({st['total%']:+.0f}%), max DD {st['maxDD%']:.1f}%, "
                  f"lowest balance ${st['min$']:,.0f}, hit rate {st['win%']:.1f}% over {st['trades']} trades "
                  f"| base: ${bst['final$']:,.0f}, max DD {bst['maxDD%']:.1f}%").replace("$", r"\$"), fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(color="#eeeeee")

    ax = axes[1]
    for e, col, lab in ((eq, "#7b2cbf", a.name), (beq, "#808080", "base")):
        peak = pd.concat([pd.Series([START]), e.reset_index(drop=True)]).cummax().iloc[1:].values
        ax.plot(e.index, 100 * (e.values / peak - 1), color=col, linewidth=1.4, label=lab)
    ax.fill_between(eq.index, 100 * (eq.values / pd.concat([pd.Series([START]), eq.reset_index(drop=True)]).cummax().iloc[1:].values - 1),
                    color="#7b2cbf", alpha=0.15)
    ax.set_ylabel("drawdown %")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.grid(color="#eeeeee")

    ax = axes[2]
    pnl = (eq - eq.shift(1).fillna(START))
    ax.bar(pnl.index, pnl.values, color=np.where(pnl.values >= 0, "#26a69a", "#ef5350"), width=1.0)
    ax.set_ylabel(r"daily P&L (\$)")
    ax2 = ax.twinx()
    cnt = sel.groupby("day").size().reindex(days, fill_value=0)
    ax2.plot(cnt.index, cnt.values, color="#555555", linewidth=0.8, drawstyle="steps-mid")
    ax2.set_ylabel("trades / day")
    ax2.set_ylim(0, 6)
    ax.grid(color="#eeeeee")

    ax = axes[3]
    win = (sel.sort_values("entry_time")["ret"] > 0).astype(float).reset_index(drop=True)
    ax.plot(sel.sort_values("entry_time")["day"].values, 100 * win.rolling(50, min_periods=20).mean().values, color="#7b2cbf")
    ax.axhline(100 * win.mean(), color="#bbbbbb", linestyle="--", linewidth=0.8)
    ax.set_ylabel("rolling 50-trade\nhit rate %")
    ax.grid(color="#eeeeee")

    if test_start is not None:
        for a_ in axes:
            a_.axvspan(test_start, days[-1], color="#fff3c4", alpha=0.5, zorder=0)
        axes[0].text(test_start, axes[0].get_ylim()[1], "  final test (locked)", va="top", fontsize=8, color="#8a6d00")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.name}{'_with_test' if a.with_test else '_dev'}.png")
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)
    print(pd.DataFrame({a.name: st, "base": bst}).T.to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
