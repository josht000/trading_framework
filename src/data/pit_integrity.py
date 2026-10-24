"""
Data-integrity helpers and validation for the PIT dataset.

Two kinds of duplicates exist:
  * bars: the same (symbol, timestamp) / (symbol, date) twice.  Minute files are self-contained (signal day + next
    session), so a symbol that is a candidate on consecutive days has the same bars in two files; loaders must
    drop duplicate timestamps (`load_minute`, `dedupe_on`).
  * securities: a ticker change leaves the same company's bars under old and new symbol.  Canonical rule: of the
    symbols with an identical bar on the same date, the alphabetically first *candidate* is kept (what
    `dedupe_trades` does, since trade tables are symbol-sorted).  "Identical" = same date and open/high/low/close
    with volume within 2% (late prints differ slightly between tickers) and at least MIN_ALIAS_VOLUME shares.
"""

import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PIT_DIR = ROOT / "data" / "pit"
PRICE = ["open", "high", "low", "close"]
MIN_ALIAS_VOLUME = 10_000  # penny/SPAC names print identical 100-share bars by coincidence
DEV_TRADES = ROOT / "results" / "pit_trades_atr_pivot_lb13_pm_target25_v2.csv"
TEST_TRADES = ROOT / "results" / "holdout" / "pit_trades_atr_pivot_lb13_pm_target25_v2.csv"
WINDOWS = ROOT / "results" / "exit" / "windows.npz"
WINDOWS_INDEX = WINDOWS.with_name("windows_index.parquet")
ORACLE = WINDOWS.with_name("oracle.parquet")
ORACLE_META = WINDOWS.with_name("oracle_meta.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def oracle_staleness(oracle=ORACLE, meta=ORACLE_META, windows=WINDOWS, index=WINDOWS_INDEX):
    """Reasons the hindsight oracle table is inconsistent with the current window cache (empty list = fresh)."""
    if not Path(meta).exists():
        return ["oracle_meta.json missing"]
    m = json.loads(Path(meta).read_text())
    bad = [f"{n} changed since the oracle was built" for n, p in (("windows.npz", windows), ("windows_index.parquet", index), ("oracle.parquet", oracle))
           if sha256_file(p) != m.get(f"{n.split('.')[0]}_sha256")]
    ids = set(np.load(windows, allow_pickle=True)["meta"][:, 0].astype(int).tolist())
    o = set(pd.read_parquet(oracle, columns=["trade_id"])["trade_id"].astype(int).tolist())
    if o != ids:
        bad.append(f"trade_id set differs from windows cache ({len(o - ids)} extra, {len(ids - o)} missing)")
    return bad


def dedupe_trades(t):
    """
    Drop duplicate securities: a ticker change leaves the same company's bars under the old and
    new symbol, so the same trade appears twice (identical day, prior close, 15:55 price).
    """
    day = t["entry_time"].astype(str).str[:10]
    return t.loc[~pd.DataFrame({"d": day, "p": t["prev_close"].round(4), "e": t["price"].round(4)}).duplicated()]


def dedupe_on(df, keys):
    return df.drop_duplicates(keys, keep="last")


def alias_mask(bars):
    """True for rows (date, symbol, OHLCV) repeating an alphabetically earlier symbol's bar on the same date."""
    b = bars[bars["volume"] >= MIN_ALIAS_VOLUME].sort_values("symbol", kind="stable")
    v = b.groupby(["date", *PRICE])["volume"]
    dup = b.duplicated(["date", *PRICE]) & (v.transform("max") <= 1.02 * v.transform("min"))
    return bars.index.isin(dup[dup].index)


def traded_pairs():
    """(signal date, symbol) of the de-duplicated trades in the dev and holdout tables (structure columns only)."""
    ts = [t for t in (_read_trades(DEV_TRADES), _read_trades(TEST_TRADES)) if t is not None]
    if not ts:
        return set()
    k = dedupe_trades(pd.concat(ts, ignore_index=True))
    return set(zip(pd.to_datetime(k["entry_time"].astype(str).str[:10]), k["symbol"]))


def drop_daily_aliases(cands, daily, keep=()):
    """
    Remove candidate (date, symbol) rows that are alias duplicates of another candidate's bar, except the
    (date, symbol) pairs in `keep` (twins that are the traded security in existing results).
    """
    c = cands.merge(daily[["date", "symbol", *PRICE, "volume"]], on=["date", "symbol"], how="left")
    kept = pd.MultiIndex.from_frame(c[["date", "symbol"]]).isin(list(keep)) if keep else np.zeros(len(c), bool)
    return cands[~(alias_mask(c) & ~kept)].reset_index(drop=True)


def clean_minute(m):
    """Drop duplicate (symbol, timestamp) bars; sorted by symbol, time."""
    m = dedupe_on(m, ["symbol", "timestamp"])
    return m.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)


def load_minute(root=PIT_DIR / "alpaca_minute", symbols=None):
    """All minute files concatenated, duplicate (symbol, timestamp) bars (overlapping day files) dropped."""
    m = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(Path(root) / "*.parquet")))], ignore_index=True)
    if symbols is not None:
        m = m[m["symbol"].isin(set(symbols))]
    return dedupe_on(m, ["symbol", "timestamp"]).reset_index(drop=True)


def clean_daily(df):
    """Drop duplicate (symbol, date) rows and unusable bars (NaN / non-positive price, high < low, negative volume)."""
    df = dedupe_on(df, ["symbol", "date"])
    ok = df[PRICE].notna().all(axis=1) & (df[PRICE] > 0).all(axis=1) & (df["high"] >= df["low"]) & (df["volume"] >= 0)
    return df[ok]


def _bad_prices(x):
    p = x[PRICE]
    return {
        "nan_price": int(x[[*PRICE, "volume"]].isna().any(axis=1).sum()),
        "nonpositive_price": int((p <= 0).any(axis=1).sum()),
        "high_below_low": int((x["high"] < x["low"]).sum()),
        "negative_volume": int((x["volume"] < 0).sum()),
    }


def _read_trades(path):
    cols = ["symbol", "entry_time", "prev_close", "price"]
    return pd.read_csv(path, usecols=cols, keep_default_na=False, na_values=[""]) if Path(path).exists() else None


def validate_pit_data(pit_dir=PIT_DIR, trades=True, windows=True, log=print, session=("04:00", "20:00")):
    """
    Integrity checks over the PIT files; returns {"ok", "errors", "warnings", "counts"} where errors/warnings map
    check name -> {file or table: offending count}.  `trades` also checks the dev/holdout trade tables (structure
    columns only, no returns) and `windows` the exit window cache against them.
    """
    pit_dir = Path(pit_dir)
    errors, warnings, counts = {}, {}, {}
    add = lambda store, check, where, n: n and store.setdefault(check, {}).__setitem__(where, int(n))
    lo, hi = (pd.Timestamp(s).hour * 60 + pd.Timestamp(s).minute for s in session)

    parts = []
    files = sorted((pit_dir / "alpaca_minute").glob("*.parquet"))
    for f in files:
        x = pd.read_parquet(f)
        if not len(x):
            continue
        ts = pd.to_datetime(x["timestamp"], utc=True).dt.tz_convert("America/New_York")
        add(errors, "duplicate (symbol, timestamp) in minute file", f.name, x.duplicated(["symbol", "timestamp"]).sum())
        add(errors, "timestamps not monotonic per symbol", f.name, (ts.groupby(x["symbol"].to_numpy()).diff() < pd.Timedelta(0)).sum())
        tm = ts.dt.hour * 60 + ts.dt.minute
        day = pd.Timestamp(f.stem).date()
        add(errors, "timestamps outside session range / file date window", f.name,
            ((tm < lo) | (tm >= hi) | (ts.dt.date < day) | (ts.dt.date > day + pd.Timedelta(days=5))).sum())
        for k, n in _bad_prices(x).items():
            add(errors, f"bad minute bars: {k}", f.name, n)
        parts.append(x[["symbol", "timestamp", "close", "volume"]].assign(timestamp=ts.to_numpy()))
    if parts:
        a = pd.concat(parts, ignore_index=True)
        dup = a.duplicated(["symbol", "timestamp"], keep=False)
        counts["minute_bars"], counts["minute_overlap_bars"] = len(a), int(a.duplicated(["symbol", "timestamp"]).sum())
        conflict = a[dup].groupby(["symbol", "timestamp"])[["close", "volume"]].nunique().max(axis=1) > 1
        add(errors, "same (symbol, timestamp) with different values across minute files", "alpaca_minute", conflict.sum())

    daily = pd.read_parquet(pit_dir / "daily_pit.parquet") if (pit_dir / "daily_pit.parquet").exists() else None
    if daily is not None:
        add(errors, "duplicate (symbol, date)", "daily_pit.parquet", daily.duplicated(["symbol", "date"]).sum())
        for k, n in _bad_prices(daily).items():
            add(errors, f"bad daily bars: {k}", "daily_pit.parquet", n)
        counts["daily_alias_rows"] = int(alias_mask(daily).sum())
    cp = pit_dir / "candidates.parquet"
    if cp.exists():
        c = pd.read_parquet(cp)
        add(errors, "duplicate (symbol, date)", "candidates.parquet", c.duplicated(["symbol", "date"]).sum())
        if daily is not None:
            add(errors, "alias securities among candidates", "candidates.parquet", len(c) - len(drop_daily_aliases(c, daily, traded_pairs())))
    for name, keys in (("news/news.parquet", ["symbol", "created_at", "headline"]), ("edgar/filings.parquet", ["symbol", "accessionNumber"])):
        if (pit_dir / name).exists():
            add(errors, "duplicate rows", name, pd.read_parquet(pit_dir / name).duplicated(keys).sum())

    if trades:
        tt = {n: _read_trades(p) for n, p in (("dev", DEV_TRADES), ("test", TEST_TRADES))}
        for n, t in tt.items():
            if t is not None:
                counts[f"trades_{n}"] = len(t)
                add(warnings, "duplicate securities in trade table (dedupe_trades drops them)", f"{n} trades", len(t) - len(dedupe_trades(t)))
        if windows and WINDOWS.exists() and all(t is not None for t in tt.values()):
            _check_windows(pd.concat([tt["dev"], tt["test"]], ignore_index=True), WINDOWS, errors, warnings, counts, add)
        if windows and WINDOWS.exists() and ORACLE.exists():
            for why in oracle_staleness():
                add(warnings, "exit oracle table stale (rerun scripts/build_exit_oracle.py)", why, 1)

    ranked = ROOT / "reports" / "ranked_table.parquet"
    enrichment = pit_dir / "enrichment" / "massive_mcap.parquet"
    if ranked.exists() and enrichment.exists():
        from src.data.market_cap import merge_market_cap
        merged = merge_market_cap(pd.read_parquet(ranked), enrichment)
        coverage = float(merged["mcap_final"].notna().mean()) if len(merged) else 1.0
        counts["mcap_final_coverage"] = coverage
        if coverage < 0.90:
            add(warnings, "mcap_final coverage below 90%", "ranked_table.parquet", int(round((1 - coverage) * len(merged))))

    return {"ok": not errors, "errors": errors, "warnings": warnings, "counts": counts}


def _check_windows(all_t, path, errors, warnings, counts, add):
    z = np.load(path, allow_pickle=True)
    off, meta, times = z["offsets"], z["meta"], z["times"]
    ids = meta[:, 0].astype(int)
    counts["windows"] = len(ids)
    add(errors, "trades with more than one window", "windows", pd.Series(ids).duplicated().sum())
    add(errors, "window trade_id outside trade table", "windows", (ids >= len(all_t)).sum())
    keep = set(dedupe_trades(all_t).index)
    add(errors, "windows for duplicate-security trades", "windows", sum(i not in keep for i in ids))
    missing = keep - set(ids.tolist())
    add(warnings, "trades without a window (no next-session minute bars)", "windows", len(missing))
    entry = pd.to_datetime(all_t["entry_time"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
    ns = lambda v: int(v) * (1000 if abs(int(v)) < 10**17 else 1)
    et = lambda v: pd.Timestamp(ns(v), tz="UTC").tz_convert("America/New_York").date()
    bad_dup = bad_order = bad_entry = bad_sym = no_next = 0
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        t = times[a:b]
        d = np.diff(t)
        bad_dup += bool((d == 0).any())
        bad_order += bool((d < 0).any())
        if ids[i] < len(all_t):
            bad_entry += not (len(t) and ns(t[0]) == entry[ids[i]])
            bad_sym += all_t["symbol"].iat[ids[i]] != str(meta[i][4])
        no_next += bool(len(t)) and et(t[-1]) == et(t[0])
    add(errors, "windows with duplicate timestamps", "windows", bad_dup)
    add(errors, "windows with non-increasing timestamps", "windows", bad_order)
    add(errors, "windows whose first bar is not the entry bar", "windows", bad_entry)
    add(errors, "windows whose symbol differs from the trade", "windows", bad_sym)
    add(warnings, "windows without next-session bars", "windows", no_next)


def format_report(rep):
    lines = [f"PIT data integrity: {'OK' if rep['ok'] else 'FAILED'}  counts={rep['counts']}"]
    for title, store in (("ERROR", rep["errors"]), ("WARNING", rep["warnings"])):
        for check, where in store.items():
            lines.append(f"{title}: {check}: {sum(where.values())} in {len(where)} place(s)")
            lines += [f"    {k}: {v}" for k, v in list(where.items())[:10]]
            if len(where) > 10:
                lines.append(f"    ... {len(where) - 10} more")
    return "\n".join(lines)


def require_valid_pit_data(**kw):
    """Run validate_pit_data, print the summary and exit non-zero on errors."""
    rep = validate_pit_data(**kw)
    print(format_report(rep), flush=True)
    if not rep["ok"]:
        raise SystemExit(1)
    return rep
