"""Point-in-time market-cap fallback merge."""

from pathlib import Path

import numpy as np
import pandas as pd


def _suspect(frame):
    a = frame["weighted_shares"].abs()
    b = frame["share_class_shares"].abs()
    return a.gt(0) & b.gt(0) & ((a / b > 5) | (b / a > 5))


def merge_market_cap(trades, enrichment_path="data/pit/enrichment/massive_mcap.parquet"):
    """Add mcap_final, mcap_src, and mcap_suspect without modifying mcap_pit."""
    out = trades.copy()
    if "mcap_pit" not in out:
        out["mcap_pit"] = np.nan
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
    cols = ["symbol", "day", "market_cap", "mcap_suspect"]
    e = e[cols].drop_duplicates(["symbol", "day"], keep="last")
    out = out.drop(columns=["mcap_suspect"], errors="ignore")
    out = out.merge(e, on=["symbol", "day"], how="left")
    use = out["mcap_pit"].isna() & out["market_cap"].notna()
    out.loc[use, "mcap_final"] = out.loc[use, "market_cap"]
    out.loc[use, "mcap_src"] = "massive"
    out["mcap_suspect"] = out["mcap_suspect"].fillna(False).astype(bool)
    return out.drop(columns=["market_cap"])
