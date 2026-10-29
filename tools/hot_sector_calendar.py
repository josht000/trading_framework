"""Hot-sector calendar: a sector is hot for week W+1 if, in week W (Mon-Sun), it was among the top 2 sectors by number of
breakout candidates (reports/ranked_table.parquet, signal days only) with >=5 candidates. One-week lag => no hindsight."""
import pandas as pd
t = pd.read_parquet("reports/ranked_table.parquet", columns=["day", "sector", "sig_day_ret"])
t = t[t.sector != "Unknown"].copy()
t["wk"] = t.day.dt.to_period("W-SUN").dt.start_time
g = t.groupby(["wk", "sector"]).agg(n=("day", "size"), avg_sig_ret=("sig_day_ret", "mean")).reset_index()
g["rank"] = g.groupby("wk")["n"].rank(ascending=False, method="first")
h = g[(g["rank"] <= 2) & (g.n >= 5)].copy()
h["hot_from"] = h.wk + pd.Timedelta(days=7)
h["hot_to"] = h.hot_from + pd.Timedelta(days=6)
h = h.sort_values(["hot_from", "rank"])
out = h[["hot_from", "hot_to", "sector", "n", "avg_sig_ret"]].rename(columns={"n": "candidates_prior_wk", "avg_sig_ret": "avg_signal_day_ret"})
out["hot_from"] = out.hot_from.dt.date; out["hot_to"] = out.hot_to.dt.date
out["avg_signal_day_ret"] = (out.avg_signal_day_ret * 100).round(1)
out.to_csv("reports/hot_sector_calendar.csv", index=False)
print(out.to_markdown(index=False))
