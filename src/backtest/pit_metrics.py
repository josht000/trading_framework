"""Portfolio metrics for a table of per-trade results (one row per trade, columns ret + entry_time)."""

import numpy as np
import pandas as pd


def dedupe_trades(t):
    """
    Drop duplicate securities: a ticker change leaves the same company's bars under the old and
    new symbol, so the same trade appears twice (identical day, prior close, 15:55 price).
    """
    day = t["entry_time"].astype(str).str[:10]
    return t.loc[~pd.DataFrame({"d": day, "p": t["prev_close"].round(4), "e": t["price"].round(4)}).duplicated()]


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


def summarize(t, max_pos=5, rank="rel_vol", sizing="full", all_days=None):
    t = dedupe_trades(t)
    r = t["ret"]
    wins, losses = r[r > 0], r[r <= 0]
    per_day = daily_returns(t, max_pos, rank, sizing, all_days)
    equity = (1 + per_day).cumprod()
    dd = float((equity / equity.cummax() - 1).min()) if len(equity) else float("nan")
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
