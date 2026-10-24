"""Portfolio metrics for a table of per-trade results (one row per trade, columns ret + entry_time)."""

import numpy as np
import pandas as pd

from src.data.pit_integrity import dedupe_trades  # noqa: F401  (re-exported)


def daily_returns(t, max_pos=5, rank="rel_vol", sizing="full", all_days=None):
    """
    Portfolio return per day. Each day keeps the top `max_pos` trades by `rank` (descending), then
      sizing="full":  equal weight over the day's picks (fully invested overnight)
      sizing="slots": each pick gets 1/max_pos of capital (thin days hold cash)
      sizing="invatr": weight proportional to 1 / daily ATR% (calmer names get more), fully invested
    """
    t = dedupe_trades(t).copy()
    t["day"] = pd.to_datetime(t["entry_time"].astype(str).str[:10])
    if t.empty:
        return pd.Series(dtype=float)
    sel = t.sort_values(["day", rank], ascending=[True, False], kind="stable").groupby("day").head(max_pos)
    if sizing == "invatr":
        w = 1.0 / sel["atr_pct"].fillna(sel["atr_pct"].median()).clip(lower=0.02)
        sel = sel.assign(w=w / w.groupby(sel["day"]).transform("sum"))
        per_day = (sel["w"] * sel["ret"]).groupby(sel["day"]).sum()
    else:
        g = sel.groupby("day")["ret"]
        per_day = g.mean() if sizing == "full" else g.sum() / max_pos
    idx = all_days if all_days is not None else pd.bdate_range(t["day"].min(), t["day"].max())
    return per_day.reindex(idx, fill_value=0.0)


def select_trades(t, max_pos=5, rank="rel_vol"):
    """The trades actually taken: de-duplicated, top `max_pos` per day by `rank` (descending)."""
    t = dedupe_trades(t).copy()
    t["day"] = pd.to_datetime(t["entry_time"].astype(str).str[:10])
    return t.sort_values(["day", rank], ascending=[True, False], kind="stable").groupby("day").head(max_pos)


def simulate_account(t, start=5000.0, base_frac=0.20, boost_frac=0.25, max_pos=5, rank="rel_vol",
                     boost=None, all_days=None):
    """
    Dollar account: each day hold the top `max_pos` trades by `rank`, each sized as a fraction of
    current equity (base_frac = $1,000 on $5,000; boost_frac = $1,250 for trades where `boost(t)` is true).
    5 x 25% is 1.25x gross exposure, i.e. a little margin. Trades are overnight, so P&L is booked daily
    and compounds. Returns (stats dict, equity Series, daily-return Series).
    """
    sel = select_trades(t, max_pos, rank).copy()
    idx = all_days if all_days is not None else pd.bdate_range(sel["day"].min(), sel["day"].max())
    sel["frac"] = np.where(boost(sel) if boost is not None else False, boost_frac, base_frac)
    day_ret = (sel["frac"] * sel["ret"]).groupby(sel["day"]).sum().reindex(idx, fill_value=0.0)
    gross = sel.groupby("day")["frac"].sum().reindex(idx, fill_value=0.0)
    equity = start * (1 + day_ret).cumprod()
    eq0 = pd.concat([pd.Series([start]), equity.reset_index(drop=True)], ignore_index=True)
    dd = float((eq0 / eq0.cummax() - 1).min())
    neg = (day_ret < 0).astype(int)
    streak = neg.groupby((neg != neg.shift()).cumsum()).cumsum().max()
    sd = day_ret.std()
    r = sel["ret"]
    stats = {
        "trades": len(sel), "win%": 100 * (r > 0).mean(), "mean%": 100 * r.mean(),
        "PF": r[r > 0].sum() / abs(r[r <= 0].sum()) if (r <= 0).any() else float("inf"),
        "final$": float(equity.iloc[-1]), "total%": 100 * (equity.iloc[-1] / start - 1), "maxDD%": 100 * dd,
        "min$": float(eq0.min()), "worst_day%": 100 * float(day_ret.min()),
        "worst_trade%": 100 * float(r.min()), "loss_streak_days": int(streak),
        "avg_gross": float(gross[gross > 0].mean()) if (gross > 0).any() else 0.0,
        "sharpe": day_ret.mean() / sd * np.sqrt(252) if sd > 0 else float("nan"),
    }
    return stats, equity, day_ret


def summarize(t, max_pos=5, rank="rel_vol", sizing="full", all_days=None):
    t = dedupe_trades(t)
    r = t["ret"]
    wins, losses = r[r > 0], r[r <= 0]
    per_day = daily_returns(t, max_pos, rank, sizing, all_days)
    equity = pd.concat([pd.Series([1.0]), (1 + per_day).cumprod().reset_index(drop=True)], ignore_index=True)
    dd = float((equity / equity.cummax() - 1).min()) if len(per_day) else float("nan")
    total = float(equity.iloc[-1] - 1) if len(equity) else float("nan")
    sd = per_day.std()
    return {
        "trades": len(t),
        "win%": 100 * (r > 0).mean(),
        "mean%": 100 * r.mean(),
        "median%": 100 * r.median(),
        "PF": wins.sum() / abs(losses.sum()) if len(losses) else float("inf"),
        "p5%": 100 * r.quantile(0.05),
        "worst%": 100 * r.min(),
        "total%": 100 * total,
        "maxDD%": 100 * dd,
        "sharpe": per_day.mean() / sd * np.sqrt(252) if sd > 0 else float("nan"),
        "calmar": total / abs(dd) if dd < 0 else float("nan"),
    }
