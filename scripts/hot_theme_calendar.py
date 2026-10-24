"""Hot theme/industry calendar. Group = keyword theme (matched on yfinance business summary) else yfinance industry.
Score = trailing-8-week candidates / group expanding-mean baseline (no look-ahead), needs >=5 candidates. Hot set = 2 groups with sticky membership (an incumbent stays while still top-4 and >=4
candidates; vacancies fill from top rank, min 6). Labels for week W+1 use data through week W only (1-week lag).
Note: descriptions are current yfinance text (classification only, not returns)."""
import re, pandas as pd
THEMES = {
 "Quantum": r"quantum", "AI / ML": r"artificial intelligence|machine learning|\bai\b|generative|large language|neural",
 "Semiconductors": r"semiconductor|silicon|wafer|chip|photonic|integrated circuit|\bgpu\b",
 "Crypto / Bitcoin": r"bitcoin|crypto|blockchain|digital asset|token|mining of digital",
 "Drones / Defense": r"drone|unmanned|defense|military|aerospace|missile|counter-uas",
 "Rare earth / Critical minerals": r"rare earth|lithium|critical mineral|uranium|graphite|cobalt|nickel|copper|gold|silver",
 "Nuclear / Energy transition": r"nuclear|small modular|fusion|hydrogen|solar|battery|energy storage|fuel cell",
 "Robotics / EV": r"robot|autonomous|electric vehicle|lidar|humanoid",
 "Biotech / Pharma": r"clinical|biopharma|therapeutic|oncology|fda|biotechnology|drug",
 "Medical devices / Diagnostics": r"medical device|diagnostic|surgical",
}
y = pd.read_parquet("data/pit/enrichment/yf_info.parquet", columns=["symbol", "industry", "longBusinessSummary"])
def grp(r):
    s = str(r.longBusinessSummary or "").lower()
    for k, p in THEMES.items():
        if re.search(p, s): return k
    return r.industry if isinstance(r.industry, str) else None
y["grp"] = y.apply(grp, axis=1)
t = pd.read_parquet("reports/ranked_table.parquet", columns=["day", "symbol", "industry"])
t["grp"] = t.symbol.map(y.set_index("symbol").grp).fillna(t["industry"])
t = t.dropna(subset=["grp"]); t["wk"] = t.day.dt.to_period("W-SUN").dt.start_time
c = t.groupby(["wk", "grp"]).size().unstack(fill_value=0)
c = c.reindex(pd.date_range(c.index.min(), c.index.max(), freq="7D"), fill_value=0)
n4 = c.rolling(8, min_periods=1).sum()
base = n4.expanding(min_periods=12).mean().shift(1).clip(lower=1.0)
s = (n4 / base).where(n4 >= 6, 0)
s = s.where(base.notna(), 0)
hot, rows = [], []
for wk, r in s.iterrows():
    rk = r.rank(ascending=False, method="first")
    keep = [g for g in hot if rk[g] <= 4 and r[g] >= 1.5]
    for g in r.sort_values(ascending=False).index:
        if len(keep) >= 2: break
        if g not in keep and r[g] >= 2.5: keep.append(g)
    hot = keep
    for g in hot: rows.append((wk + pd.Timedelta(days=7), g, round(r[g],1)))
d = pd.DataFrame(rows, columns=["hot_from", "group", "spike"])
spans = []
for g, x in d.groupby("group"):
    x = x.sort_values("hot_from"); st = pr = None; pk = 0
    for _, q in x.iterrows():
        if pr is not None and (q.hot_from - pr).days == 7: pr = q.hot_from; pk = max(pk, q.spike); continue
        if st is not None: spans.append((st, pr, g, pk))
        st = pr = q.hot_from; pk = q.spike
    spans.append((st, pr, g, pk))
o = pd.DataFrame(spans, columns=["hot_from", "hot_to", "group", "peak_spike_x"]).sort_values("hot_from")
o["hot_to"] = (o.hot_to + pd.Timedelta(days=6)).dt.date; o["hot_from"] = o.hot_from.dt.date
o.to_csv("reports/hot_theme_calendar.csv", index=False); print(o.to_markdown(index=False))
