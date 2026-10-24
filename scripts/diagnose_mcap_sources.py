"""Three-source diagnostics for a 50-row Massive versus SimFin market-cap sample."""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.pit_dataset import MASSIVE, MassiveClient

BASE = ROOT / "data" / "pit" / "enrichment"
RANKED = ROOT / "reports" / "ranked_table.parquet"


def _get(client, url, params):
    for attempt in range(4):
        try:
            r = client.get(url, **params)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            return r
        except requests.RequestException:
            if attempt == 3:
                return None
            time.sleep(2 ** attempt)
    return None


def _fetch_rows(rows, cache_path, query_col, result_name):
    cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    terminal = {"fetched", "not_found"}
    cache_done = cache[cache["status"].isin(terminal)] if len(cache) and "status" in cache else cache
    done = set(zip(cache_done.get("symbol", []), cache_done.get("day", []))) if len(cache_done) else set()
    todo = rows[~rows.apply(lambda r: (r.symbol, r.day) in done, axis=1)].reset_index(drop=True)
    futures = {}
    out = cache
    with ThreadPoolExecutor(max_workers=4) as pool:
        next_start = time.monotonic()
        for i, r in todo.iterrows():
            wait = next_start - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            next_start += 3.0
            c = MassiveClient(interval=0)
            futures[pool.submit(_get, c, f"{MASSIVE}/v3/reference/tickers/{r.symbol}",
                                {"date": str(pd.Timestamp(r[query_col]).date())})] = r
            if len(futures) >= 4:
                f = next(as_completed(futures))
                rr = futures.pop(f)
                response = f.result()
                result = (response.json() or {}).get("results", {}) if response is not None and response.ok else {}
                status = "fetched" if result.get("market_cap") is not None else (
                    "not_found" if response is not None and response.status_code == 404 else
                    "rate_limited" if response is not None and response.status_code == 429 else "no_data")
                rec = {"symbol": rr.symbol, "day": rr.day, "query_date": rr[query_col],
                       "market_cap": result.get("market_cap"), "weighted_shares": result.get("weighted_shares_outstanding"),
                       "share_class_shares": result.get("share_class_shares_outstanding"), "status": status}
                out = pd.concat([out, pd.DataFrame([rec])], ignore_index=True).drop_duplicates(["symbol", "day"], keep="last")
                out.to_parquet(cache_path, index=False)
        for f in as_completed(futures):
            rr = futures[f]
            response = f.result()
            result = (response.json() or {}).get("results", {}) if response is not None and response.ok else {}
            status = "fetched" if result.get("market_cap") is not None else (
                "not_found" if response is not None and response.status_code == 404 else
                "rate_limited" if response is not None and response.status_code == 429 else "no_data")
            rec = {"symbol": rr.symbol, "day": rr.day, "query_date": rr[query_col],
                   "market_cap": result.get("market_cap"), "weighted_shares": result.get("weighted_shares_outstanding"),
                   "share_class_shares": result.get("share_class_shares_outstanding"), "status": status}
            out = pd.concat([out, pd.DataFrame([rec])], ignore_index=True).drop_duplicates(["symbol", "day"], keep="last")
            out.to_parquet(cache_path, index=False)
    return out


def _fetch_splits(symbols, path):
    cache = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    done = set(cache.get("symbol", [])) if len(cache) else set()
    todo = [s for s in symbols if s not in done]
    futures = {}
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        next_start = time.monotonic()
        for symbol in todo:
            wait = next_start - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            next_start += 3
            c = MassiveClient(interval=0)
            futures[pool.submit(_get, c, f"{MASSIVE}/v3/reference/splits",
                                {"ticker": symbol, "limit": 100})] = symbol
            if len(futures) >= 4:
                f = next(as_completed(futures))
                sym = futures.pop(f)
                r = f.result()
                vals = (r.json() or {}).get("results", []) if r is not None and r.ok else []
                rows.extend([{"symbol": sym, **x} for x in vals] or [{"symbol": sym, "status": "none"}])
        for f in as_completed(futures):
            sym = futures[f]
            r = f.result()
            vals = (r.json() or {}).get("results", []) if r is not None and r.ok else []
            rows.extend([{"symbol": sym, **x} for x in vals] or [{"symbol": sym, "status": "none"}])
    out = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
    out.to_parquet(path, index=False)
    return out


def _stats(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    return f"n={len(x)}, p10={x.quantile(.1):.3f}, p50={x.median():.3f}, p90={x.quantile(.9):.3f}, within10={(x.between(.9,1.1)).mean():.1%}"


def main():
    load_dotenv(ROOT / ".env")
    t = pd.read_parquet(RANKED)
    e = pd.read_parquet(BASE / "massive_mcap.parquet")
    sample = e[(e["validation_only"]) & (e["status"] == "fetched")].sample(50, random_state=20261104)
    sample["day"] = pd.to_datetime(sample["day"]).dt.normalize()
    sample["query_date"] = pd.to_datetime(sample["query_date"]).dt.normalize()
    ranked = t[["symbol", "day", "mcap_pit", "shares_pit", "price"]].copy()
    ranked["day"] = pd.to_datetime(ranked["day"]).dt.normalize()
    sample = sample[["symbol", "day", "query_date"]].merge(ranked, on=["symbol", "day"], how="left")
    sample["signal_query_date"] = sample["day"]
    prior = pd.read_parquet(BASE / "mcap_diag_prior.parquet") if (BASE / "mcap_diag_prior.parquet").exists() else pd.DataFrame()
    source_prior = e[e["validation_only"]][["symbol", "day", "query_date", "market_cap", "weighted_shares",
                                           "share_class_shares", "status"]].copy()
    source_prior = source_prior.merge(sample[["symbol", "day"]], on=["symbol", "day"], how="inner")
    source_prior.to_parquet(BASE / "mcap_diag_prior.parquet", index=False)
    prior = source_prior
    signal = _fetch_rows(sample, BASE / "mcap_diag_signal.parquet", "signal_query_date", "signal")
    splits = _fetch_splits(sample["symbol"].unique(), BASE / "mcap_diag_splits.parquet")
    p = prior[["symbol","day","market_cap","weighted_shares"]].rename(columns={"market_cap":"massive_prior","weighted_shares":"shares_prior"})
    s = signal[["symbol","day","market_cap","weighted_shares"]].rename(columns={"market_cap":"massive_signal","weighted_shares":"shares_signal"})
    d = sample.merge(p,on=["symbol","day"]).merge(s,on=["symbol","day"])
    d["ratio_prior"] = d.massive_prior / d.mcap_pit
    d["ratio_signal"] = d.massive_signal / d.mcap_pit
    d["ratio_signal_prior"] = d.massive_signal / d.massive_prior
    d["shares_ratio"] = d.shares_prior / d.shares_pit
    d["price_prior"] = d.massive_prior / d.shares_prior
    d["price_signal"] = d.massive_signal / d.shares_signal
    d["price_simfin"] = d.mcap_pit / d.shares_pit
    # Alpaca daily is explicitly fetched with split adjustment in pit_dataset.py.
    bars = pd.read_parquet(ROOT / "data" / "pit" / "daily_pit.parquet",
                           columns=["symbol","date","close"])
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    d = d.merge(bars.rename(columns={"date":"day","close":"close_signal"}), on=["symbol","day"], how="left")
    d = d.merge(bars.rename(columns={"date":"query_date","close":"close_prior"}), on=["symbol","query_date"], how="left")
    d["signal_move"] = d.close_signal / d.close_prior - 1
    d["price_prior_vs_close"] = d.price_prior / d.close_prior
    d["price_signal_vs_close"] = d.price_signal / d.close_signal
    d["price_simfin_vs_close"] = d.price_simfin / d.close_signal
    split_dates = pd.to_datetime(splits.get("execution_date", pd.Series(dtype="datetime64[ns]")), errors="coerce")
    splits = splits.assign(_execution=split_dates)
    dates = splits.groupby("symbol")["_execution"].agg(list).to_dict()
    d["split_in_window"] = d.symbol.map(lambda x: any(pd.notna(z) and pd.Timestamp("2023-09-01") <= z <= pd.Timestamp("2026-10-23") for z in dates.get(x, [])))
    d["split_since_signal"] = d.apply(lambda r: any(pd.notna(z) and z >= r.day for z in dates.get(r.symbol, [])), axis=1)
    d.to_parquet(BASE / "mcap_diag_joined.parquet", index=False)
    lines = ["# Massive versus SimFin market-cap diagnosis (n=50)", "",
             "All API calls use a 3-second start interval and at most four workers. Query dates are explicit. "
             "No SEC/EDGAR data was used.", "",
             "## Baseline and date offset", "",
             f"Prior-day Massive / SimFin: {_stats(d.ratio_prior)}",
             f"Signal-day Massive / SimFin: {_stats(d.ratio_signal)}",
             f"Signal/prior Massive cap ratio: {_stats(d.ratio_signal_prior)}; signal-day move median={d.signal_move.median():.3%}.",
             f"Share-count ratio (Massive weighted / SimFin shares_pit): {_stats(d.shares_ratio)}", "",
             "## Splits", "",
             f"Symbols with a split in 2023-09-01..2026-10-23: {int(d.split_in_window.sum())}/50; "
             f"with split on/after signal day: {int(d.split_since_signal.sum())}/50."]
    for label, mask in [("split flagged", d.split_in_window), ("not flagged", ~d.split_in_window)]:
        lines += [f"- {label}: prior mcap ratio {_stats(d.loc[mask,'ratio_prior'])}; price-leg ratio {_stats(d.loc[mask,'price_prior_vs_close'])}."]
    lines += ["", "## Which price agrees with bars?", "",
              "The available PIT daily bars were fetched with Alpaca Adjustment.SPLIT, so they are split-adjusted rather than as-traded historical closes.",
              f"Massive prior implied price versus prior close: median abs={100*d.price_prior_vs_close.sub(1).abs().median():.2f}%, within2%={(d.price_prior_vs_close.sub(1).abs()<.02).mean():.1%}.",
              f"Massive signal implied price versus signal close: median abs={100*d.price_signal_vs_close.sub(1).abs().median():.2f}%, within2%={(d.price_signal_vs_close.sub(1).abs()<.02).mean():.1%}.",
              f"SimFin implied price versus signal close: median abs={100*d.price_simfin_vs_close.sub(1).abs().median():.2f}%, within2%={(d.price_simfin_vs_close.sub(1).abs()<.02).mean():.1%}.",
              f"Split-flagged versus not: Massive prior price-leg within2%="
              f"{(d.loc[d.split_in_window,'price_prior_vs_close'].sub(1).abs()<.02).mean():.1%}/"
              f"{(d.loc[~d.split_in_window,'price_prior_vs_close'].sub(1).abs()<.02).mean():.1%}; "
              f"SimFin signal price-leg within2%="
              f"{(d.loc[d.split_in_window,'price_simfin_vs_close'].sub(1).abs()<.02).mean():.1%}/"
              f"{(d.loc[~d.split_in_window,'price_simfin_vs_close'].sub(1).abs()<.02).mean():.1%}.",
              "", "## Reading", "",
              "This is a 50-row diagnostic. Date offset and split adjustment are evaluated separately from share-count differences; "
              "the price-leg and share-leg columns show which component drives disagreement. Split flags are based on the Massive splits endpoint, "
              "and a flagged row is not automatically proof of causality."]
    (ROOT / "reports" / "mcap_source_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
