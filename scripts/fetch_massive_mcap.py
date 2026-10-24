"""Resumable historical Massive market-cap enrichment for ranked PIT rows.

The query date is the previous business day before each signal day, avoiding
look-ahead at the 15:55 decision.  No API credentials are printed.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.pit_dataset import MASSIVE, MassiveClient, _atomic_write
from src.data.market_cap import merge_market_cap

TARGET = ROOT / "reports" / "ranked_table.parquet"
CACHE = ROOT / "data" / "pit" / "enrichment" / "massive_mcap.parquet"


def previous_business_day(day):
    day = pd.Timestamp(day).normalize()
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    day -= pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day.date()


def targets(sample_size=150, seed=20261103):
    t = pd.read_parquet(TARGET)
    t["day"] = pd.to_datetime(t["day"]).dt.normalize()
    missing = t[t["mcap_pit"].isna()][["symbol", "day"]].drop_duplicates()
    present = t[t["mcap_pit"].notna()][["symbol", "day"]].drop_duplicates()
    sample = present.sample(min(sample_size, len(present)), random_state=seed)
    x = pd.concat([missing.assign(validation_only=False), sample.assign(validation_only=True)], ignore_index=True)
    x["query_date"] = x["day"].map(previous_business_day)
    return x.drop_duplicates(["symbol", "day"]).reset_index(drop=True)


def _row(symbol, day, query_date, status, result=None, validation_only=False):
    result = result or {}
    weighted, share_class = result.get("weighted_shares_outstanding"), result.get("share_class_shares_outstanding")
    suspect = bool(weighted and share_class and ((weighted / share_class > 5) or (share_class / weighted > 5)))
    return {
        "symbol": symbol, "day": pd.Timestamp(day), "query_date": pd.Timestamp(query_date),
        "market_cap": result.get("market_cap"), "weighted_shares": result.get("weighted_shares_outstanding"),
        "share_class_shares": result.get("share_class_shares_outstanding"), "type": result.get("type"),
        "active": result.get("active"), "mcap_suspect": suspect,
        "validation_only": bool(validation_only), "status": status,
        "fetched_at": pd.Timestamp.now(tz="UTC"),
    }


def fetch_one(client, symbol, day, query_date, validation_only=False, retries=3):
    url = f"{MASSIVE}/v3/reference/tickers/{symbol}"
    for attempt in range(retries + 1):
        try:
            response = client.get(url, date=str(query_date))
            if response.status_code == 404:
                return _row(symbol, day, query_date, "not_found", validation_only=validation_only)
            if response.status_code == 429:
                if attempt >= retries:
                    return _row(symbol, day, query_date, "rate_limited", validation_only=validation_only)
                time.sleep(30 * (attempt + 1))
                continue
            if not response.ok:
                return _row(symbol, day, query_date, f"http_{response.status_code}", validation_only=validation_only)
            result = (response.json() or {}).get("results") or {}
            return _row(symbol, day, query_date,
                        "fetched" if result.get("market_cap") is not None else "no_data",
                        result, validation_only)
        except requests.RequestException:
            if attempt >= retries:
                return _row(symbol, day, query_date, "network_error", validation_only=validation_only)
            time.sleep(2 ** attempt)
    return _row(symbol, day, query_date, "error", validation_only=validation_only)


def fetch_parallel(todo, interval, workers, done):
    """Start at most one request per interval while overlapping network latency."""
    if not len(todo):
        return done
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        next_start = time.monotonic()
        cursor = 0
        while cursor < len(todo) or futures:
            while cursor < len(todo) and len(futures) < workers and time.monotonic() >= next_start:
                r = todo.iloc[cursor]
                client = MassiveClient(interval=0)
                f = pool.submit(fetch_one, client, r["symbol"], r["day"], r["query_date"],
                                bool(r["validation_only"]))
                futures[f] = cursor
                cursor += 1
                next_start += interval
            if futures:
                completed = next(as_completed(futures))
                out = completed.result()
                del futures[completed]
                done = pd.concat([done, pd.DataFrame([out])], ignore_index=True).drop_duplicates(
                    ["symbol", "day"], keep="last")
                _atomic_write(done, CACHE)
                if len(done) % 10 == 0:
                    print(f"{len(done)} cache rows; {cursor}/{len(todo)} submitted", flush=True)
            else:
                time.sleep(max(0.01, next_start - time.monotonic()))
    return done


def write_coverage(cache):
    source = pd.read_parquet(TARGET)
    source["day"] = pd.to_datetime(source["day"]).dt.normalize()
    merged = merge_market_cap(source, CACHE)
    before = source["mcap_pit"].notna()
    after = merged["mcap_final"].notna()
    before_filter = source["mcap_pit"] < 200e6
    after_filter = merged["mcap_final"] < 200e6
    sample = source[source["mcap_pit"].notna()][["symbol", "day", "mcap_pit"]].drop_duplicates()
    sample = sample.sample(min(150, len(sample)), random_state=20261103)
    checked = sample.merge(cache[cache["validation_only"]][["symbol", "day", "market_cap"]], on=["symbol", "day"], how="inner")
    ratio = checked["market_cap"] / checked["mcap_pit"]
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    a = pd.to_numeric(cache["weighted_shares"], errors="coerce").abs()
    b = pd.to_numeric(cache["share_class_shares"], errors="coerce").abs()
    suspect = a.gt(0) & b.gt(0) & ((a / b > 5) | (b / a > 5))
    status_kinds = {}
    for status in ("not_found", "no_data"):
        rows = cache[cache["status"] == status]
        status_kinds[status] = {
            "rows": len(rows),
            "types": rows["type"].fillna("unknown").value_counts().to_dict(),
            "active": rows["active"].value_counts(dropna=False).to_dict(),
        }
    lines = [
        "# Massive market-cap enrichment coverage",
        "",
        "Query date is the previous business day before each signal day, strictly before the 15:55 decision.",
        "",
        f"- Rows in ranked table: {len(source)}; missing SimFin mcap rows: {int((~before).sum())}.",
        f"- Final mcap coverage: {int(after.sum())}/{len(after)} ({after.mean():.1%}); source counts: {merged['mcap_src'].value_counts().to_dict()}.",
        f"- Massive statuses: {cache['status'].value_counts().to_dict()}.",
        f"- Not-found/no-data kinds: {status_kinds}.",
        f"- Suspect share-count rows (>5x weighted/share-class): {int(suspect.sum())}; market_cap is retained as returned.",
        f"- mcap<200M before/after: {int(before_filter.sum())}/{int(after_filter.sum())}; moved in={int((~before_filter & after_filter).sum())}, moved out={int((before_filter & ~after_filter).sum())}.",
        "",
        "## Quarterly coverage",
        "",
        "| quarter | rows | SimFin before | mcap_final after |",
        "|---|---:|---:|---:|",
    ]
    q = source.assign(quarter=source["day"].dt.to_period("Q")).groupby("quarter")
    for quarter, frame in q:
        idx = frame.index
        lines.append(f"| {quarter} | {len(frame)} | {source.loc[idx, 'mcap_pit'].notna().mean():.1%} | {merged.loc[idx, 'mcap_final'].notna().mean():.1%} |")
    lines += [
        "",
        "## SimFin validation sample",
        "",
        f"Rows with both values: {len(ratio)}; median Massive/SimFin ratio: {ratio.median():.3f}; "
        f"p10={ratio.quantile(.1):.3f}; p90={ratio.quantile(.9):.3f}; within 10%: {(ratio.between(.9, 1.1)).mean():.1%}.",
        "",
        "The shares_pit/float_now filter remains unchanged; mcap_final only supplies the market-cap fallback.",
    ]
    (ROOT / "reports" / "massive_mcap_coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between calls; 12.5 is the safe fallback")
    ap.add_argument("--sample-size", type=int, default=150)
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env")
    todo = targets(a.sample_size)
    done = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame()
    if len(done) and "validation_only" not in done:
        done["validation_only"] = False
    terminal = {"fetched", "not_found", "no_data"}
    terminal_done = done[done["status"].isin(terminal)] if len(done) else done
    done_keys = set(zip(terminal_done.get("symbol", []), pd.to_datetime(terminal_done.get("day", [])).dt.normalize() if len(terminal_done) else [],
                       terminal_done.get("validation_only", [])))
    todo = todo[~todo.apply(lambda r: (r["symbol"], r["day"], r["validation_only"]) in done_keys, axis=1)].reset_index(drop=True)
    client = MassiveClient(interval=a.interval)
    if not a.no_probe and len(todo):
        probe = todo.head(min(20, len(todo)))
        probe_status = []
        for _, r in probe.iterrows():
            out = fetch_one(client, r["symbol"], r["day"], r["query_date"],
                            bool(r["validation_only"]), retries=0)
            probe_status.append(out["status"])
            done = pd.concat([done, pd.DataFrame([out])], ignore_index=True)
        if any(s == "rate_limited" for s in probe_status):
            client.interval = 12.5
            print("rate probe saw 429; using safe interval 12.5 seconds", flush=True)
        else:
            print(f"rate probe: {len(probe_status)} calls at {a.interval:.2f}s without 429", flush=True)
        _atomic_write(done.drop_duplicates(["symbol", "day"], keep="last"), CACHE)
        todo = todo.iloc[len(probe):].reset_index(drop=True)
    started = time.time()
    done = fetch_parallel(todo, client.interval, 4, done)
    cache = done.drop_duplicates(["symbol", "day"], keep="last")
    write_coverage(cache)
    print(f"complete: {len(done)} cache rows; runtime {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
