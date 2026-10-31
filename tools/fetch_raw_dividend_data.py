"""Fetch raw daily bars and corporate actions for overnight studies.

Examples:
  python tools/fetch_raw_dividend_data.py --symbols SPY --start 2016-01-04 --end 2026-10-24
  python tools/fetch_raw_dividend_data.py --ever-top-n 500 --include-symbols SPY
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.alpaca_history import START, fetch_bars_batch
from src.data.liquid_universe import load_liquid_ranks

OUT_DIR = ROOT / "data" / "cache" / "daily_10years_raw"
ACTION_URL = "https://data.alpaca.markets/v1/corporate-actions"
REQUEST_INTERVAL = 0.31


def formation_symbols(top_n, frequency):
    dates, symbols, ranks = load_liquid_ranks()
    formation_rows = []
    for year in sorted(set(dates.year)):
        year_rows = np.flatnonzero(dates.year == year)
        if not len(year_rows):
            continue
        periods = sorted(set(dates[year_rows].to_period("M")))
        for period in periods:
            if period == dates[-1].to_period("M") and dates[-1].day < period.days_in_month - 3:
                continue
            if frequency == "annual" and period.month != 12:
                continue
            if frequency == "quarterly" and period.month not in (3, 6, 9, 12):
                continue
            rows = year_rows[dates[year_rows].to_period("M") == period]
            if len(rows) and dates[rows[-1]].month == 12:
                formation_rows.append(rows[-1])
            elif len(rows) and frequency != "annual":
                formation_rows.append(rows[-1])
    selected = set()
    for row in formation_rows:
        selected.update(symbols[i] for i in (ranks[row] <= top_n).nonzero()[0])
    return sorted(selected)


def wait_for_slot(next_request):
    delay = next_request[0] - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    next_request[0] = time.monotonic() + REQUEST_INTERVAL


def fetch_actions(session, symbols, start, end, next_request):
    params = {
        "symbols": ",".join(symbols),
        "types": "cash_dividend,forward_split,reverse_split,unit_split",
        "start": start,
        "end": end,
        "limit": 1000,
        "sort": "asc",
        "region": "us",
    }
    rows = []
    while True:
        for attempt in range(6):
            wait_for_slot(next_request)
            response = session.get(ACTION_URL, params=params, timeout=90)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 5:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else min(2 ** attempt, 30))
                continue
            response.raise_for_status()
            payload = response.json()
            actions = payload.get("corporate_actions", {})
            for action_type, key in (
                ("cash_dividend", "cash_dividends"),
                ("forward_split", "forward_splits"),
                ("reverse_split", "reverse_splits"),
                ("unit_split", "unit_splits"),
            ):
                rows.extend(dict(row, action_type=action_type) for row in actions.get(key, []))
            token = payload.get("next_page_token")
            if token:
                params["page_token"] = token
            else:
                return rows
    return rows


def batch_marker(state_dir, kind, symbols, start, end):
    digest = hashlib.sha1(",".join(symbols).encode("utf-8")).hexdigest()[:12]
    return state_dir / f"{kind}_{start}_{end}_{digest}.done"


def merge_actions(path, rows):
    if not rows:
        return 0
    incoming = pd.DataFrame(rows)
    path = Path(path)
    if path.exists():
        old = pd.read_parquet(path)
        incoming = pd.concat([old, incoming], ignore_index=True)
    incoming = incoming.drop_duplicates("id", keep="last")
    incoming["ex_date"] = pd.to_datetime(incoming["ex_date"], errors="coerce")
    incoming = incoming.sort_values(["symbol", "ex_date", "id"])
    incoming.to_parquet(path, index=False)
    return len(incoming)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--include-symbols", nargs="*", default=[])
    parser.add_argument("--ever-top-n", type=int, default=0,
                        help="include names that ranked in the annual formation-date top N")
    parser.add_argument("--formation-frequency", choices=("annual", "quarterly", "monthly"), default="annual")
    parser.add_argument("--start", default=str(START.date()))
    parser.add_argument("--end", default=None, help="exclusive daily-bar end date")
    parser.add_argument("--actions-start", default="2015-01-01")
    parser.add_argument("--actions-end", default=None, help="inclusive corporate-action process-date end")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--force", action="store_true", help="refetch batches with completion markers")
    args = parser.parse_args()

    symbols = set(args.symbols) | set(args.include_symbols)
    if args.ever_top_n:
        symbols.update(formation_symbols(args.ever_top_n, args.formation_frequency))
    symbols = sorted(symbols)
    if not symbols:
        parser.error("pass --symbols, --include-symbols, or --ever-top-n")

    out_dir = Path(args.out_dir)
    end = args.end
    if end is None:
        sample = pd.read_parquet(ROOT / "data" / "cache" / "daily_10years" / "SPY_1Day_split.parquet")
        end = str((sample.index.max() + pd.Timedelta(days=1)).date())
    actions_end = args.actions_end or end
    print(f"{len(symbols)} symbols; raw bars {args.start}..{end}; actions through {actions_end}; output={out_dir}")
    if args.plan:
        print("Plan only; no network request made.")
        return

    load_dotenv(ROOT / ".env")
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET are required for data access")
    state_dir = out_dir / "_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    next_request = [time.monotonic()]
    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(end)
    failed = []

    for offset in range(0, len(symbols), 5):
        batch = symbols[offset:offset + 5]
        marker = batch_marker(state_dir, "raw", batch, args.start, end)
        if marker.exists() and not args.force:
            continue
        if not args.force:
            batch = [symbol for symbol in batch
                     if not (out_dir / f"{symbol.replace('/', '_')}_1Day_raw.parquet").exists()]
        if not batch:
            continue
        try:
            fetch_bars_batch(
                session, batch, next_request, start_ts, end_ts,
                adjustment="raw", out_dir=out_dir, suffix="_1Day_raw.parquet",
            )
            marker.write_text("\n".join(batch) + "\n", encoding="utf-8")
            print(f"raw bars {min(offset + 5, len(symbols))}/{len(symbols)}", flush=True)
        except Exception as exc:
            failed.append((batch, str(exc)))
            print(f"FAILED raw {batch}: {exc}", flush=True)

    for offset in range(0, len(symbols), 50):
        batch = symbols[offset:offset + 50]
        marker = batch_marker(state_dir, "corp", batch, args.actions_start, actions_end)
        if marker.exists() and not args.force:
            continue
        try:
            rows = fetch_actions(session, batch, args.actions_start, actions_end, next_request)
            merge_actions(out_dir / "cash_dividends.parquet", rows)
            marker.write_text(f"{len(rows)} actions\n", encoding="utf-8")
            print(f"corporate actions {min(offset + 50, len(symbols))}/{len(symbols)} ({len(rows)} rows)", flush=True)
        except Exception as exc:
            failed.append((batch, str(exc)))
            print(f"FAILED actions {batch}: {exc}", flush=True)

    if failed:
        print(f"{len(failed)} batches failed; rerun to retry.")
        raise SystemExit(1)
    print("Complete.")


if __name__ == "__main__":
    main()
