"""Point-in-time market-cap fallback merge."""

from pathlib import Path

import numpy as np
import pandas as pd

STALE_DAYS = 7
SIMFIN_PRICES = Path("data/simfin/us-shareprices-daily.csv")

def _suspect(frame):
    a = frame["weighted_shares"].abs()
    b = frame["share_class_shares"].abs()
    return a.gt(0) & b.gt(0) & ((a / b > 5) | (b / a > 5))


def simfin_stale_mask(frame, shareprices=None, max_age_days=STALE_DAYS):
    """True where the last SimFin share-price row is older than max_age_days."""
    days = pd.to_datetime(frame["day"]).dt.normalize()
    if shareprices is None:
        if not SIMFIN_PRICES.exists():
            return pd.Series(False, index=frame.index)
        sp = pd.read_csv(SIMFIN_PRICES, sep=";")
        if "Ticker" not in sp or "Date" not in sp:
            sp = pd.read_csv(SIMFIN_PRICES)
        sp = sp[["Ticker", "Date"]]
        sp["Date"] = pd.to_datetime(sp["Date"])
        sp = sp.rename(columns={"Ticker": "symbol", "Date": "price_date"})
    else:
        if isinstance(shareprices.index, pd.MultiIndex):
            sp = shareprices.reset_index()[["Ticker", "Date"]].rename(
                columns={"Ticker": "symbol", "Date": "price_date"})
        else:
            sp = shareprices.reset_index()[["symbol", "date"]].rename(columns={"date": "price_date"})
    sp["price_date"] = pd.to_datetime(sp["price_date"]).dt.normalize()
    left = pd.DataFrame({"symbol": frame["symbol"].to_numpy(), "day": days.to_numpy(),
                         "_index": frame.index}).sort_values(["symbol", "day"])
    left = left.sort_values(["day", "symbol"])
    right = sp.sort_values(["price_date", "symbol"])
    matched = pd.merge_asof(left, right, left_on="day", right_on="price_date",
                            by="symbol", direction="backward")
    age = (matched["day"] - matched["price_date"]).dt.days
    stale = age.gt(max_age_days).fillna(False)
    return pd.Series(stale.to_numpy(), index=matched["_index"]).reindex(frame.index, fill_value=False).astype(bool)


def merge_market_cap(trades, enrichment_path="data/pit/enrichment/massive_mcap.parquet"):
    """Add mcap_final, mcap_src, and mcap_suspect without modifying mcap_pit."""
    out = trades.copy()
    if "mcap_pit" not in out:
        out["mcap_pit"] = np.nan
    out["mcap_stale"] = simfin_stale_mask(out)
    stale = out["mcap_stale"] & (out["mcap_pit"].notna() | out.get("shares_pit", pd.Series(index=out.index)).notna())
    out.loc[stale, ["mcap_pit", "shares_pit"]] = np.nan
    out["mcap_final"] = out["mcap_pit"]
    out["mcap_src"] = np.where(out["mcap_pit"].notna(), "simfin", "missing")
    out["mcap_suspect"] = False
    path = Path(enrichment_path)
    if not path.exists() or "symbol" not in out or "day" not in out:
        return out
    e = pd.read_parquet(path)
    if not len(e):
        return out
    e = e.copy()
    if "validation_only" in e:
        e = e[~e["validation_only"].fillna(False)].copy()
    e["day"] = pd.to_datetime(e["day"]).dt.normalize()
    out["day"] = pd.to_datetime(out["day"]).dt.normalize()
    e["mcap_suspect"] = _suspect(e)
    cols = ["symbol", "day", "market_cap", "weighted_shares", "share_class_shares", "mcap_suspect"]
    e = e[cols].drop_duplicates(["symbol", "day"], keep="last")
    out = out.drop(columns=["mcap_suspect"], errors="ignore")
    out = out.merge(e, on=["symbol", "day"], how="left")
    use = out["mcap_pit"].isna() & out["market_cap"].notna()
    shares = pd.to_numeric(out["weighted_shares"], errors="coerce")
    fallback = shares.fillna(pd.to_numeric(out["share_class_shares"], errors="coerce"))
    if "price" in out:
        price = pd.to_numeric(out["price"], errors="coerce")
    elif "entry" in out:
        price = pd.to_numeric(out["entry"], errors="coerce")
    else:
        price = pd.Series(np.nan, index=out.index)
    use_shares = out["mcap_pit"].isna() & fallback.notna() & price.notna()
    out.loc[use_shares, "mcap_final"] = fallback[use_shares] * price[use_shares]
    out.loc[use_shares, "mcap_src"] = "massive_sharesxprice"
    out.loc[use & ~use_shares, "mcap_final"] = out.loc[use & ~use_shares, "market_cap"]
    out.loc[use & ~use_shares, "mcap_src"] = "massive"
    out["mcap_suspect"] = out["mcap_suspect"].fillna(False).astype(bool)
    return out.drop(columns=["market_cap", "weighted_shares", "share_class_shares"])
