"""
Point-in-time (PIT) daily dataset built from free sources.

Sources
- Massive (ex-Polygon) free plan: reference tickers (active + inactive, with
  delisted_utc) and grouped daily bars (every ticker that traded each day,
  including names that were later delisted). Free window is ~2 years.
- MarketParquet free sample: one daily parquet per date, last 365 days.
- yfinance: independent cross-check of bar values (does not carry delisted names).

Layout under data/pit/
  reference.parquet            Massive ticker reference (all stocks, active + inactive)
  massive_daily/DATE.parquet   one file per trading date (grouped daily, adjusted)
  mp_daily/DATE.parquet        MarketParquet daily files
  daily_pit.parquet            merged bars, one row per (date, symbol)
  report.txt                   coverage / validation report
"""

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.pit_integrity import clean_daily, clean_minute, dedupe_on, drop_daily_aliases, empty_minute_frame  # noqa: F401

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
PIT_DIR = ROOT / "data" / "pit"
MASSIVE = "https://api.massive.com"
MP = "https://marketparquet.com/api/v1"
MIN_INTERVAL = 12.5  # free plan: 5 calls/minute


class MassiveClient:
    def __init__(self, api_key=None, interval=None):
        self.key = api_key or os.getenv("MASSIVE_API_KEY")
        if not self.key:
            raise ValueError("MASSIVE_API_KEY not set")
        self._last = 0.0
        self.interval = MIN_INTERVAL if interval is None else float(interval)

    def get(self, url, **params):
        if url.startswith("/"):
            url = MASSIVE + url
        for attempt in range(6):
            wait = self.interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            r = requests.get(url, params={**params, "apiKey": self.key}, timeout=60)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            return r
        return r


def _atomic_write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def fetch_reference(client=None, log=print):
    """Page through all stock tickers (active and inactive) and cache them."""
    client = client or MassiveClient()
    frames = []
    for active in ("true", "false"):
        url, params, pages = "/v3/reference/tickers", dict(market="stocks", active=active, limit=1000), 0
        while url:
            r = client.get(url, **params)
            if not r.ok:
                raise RuntimeError(f"reference {active}: {r.status_code} {r.text[:200]}")
            j = r.json()
            frames.append(pd.DataFrame(j.get("results", [])).assign(_active_query=active))
            pages += 1
            url, params = j.get("next_url"), {}
            if pages % 5 == 0:
                log(f"  reference active={active}: {pages} pages")
    ref = pd.concat(frames, ignore_index=True)
    ref = ref.drop_duplicates(subset=["ticker", "name", "delisted_utc"], keep="first")
    keep = [c for c in ["ticker", "name", "type", "primary_exchange", "active", "currency_name",
                        "cik", "composite_figi", "list_date", "delisted_utc", "last_updated_utc"]
            if c in ref.columns]
    ref = ref[keep]
    if "delisted_utc" in ref:
        ref["delisted_utc"] = pd.to_datetime(ref["delisted_utc"], utc=True, errors="coerce")
    _atomic_write(ref, PIT_DIR / "reference.parquet")
    log(f"reference: {len(ref)} rows, {int((~ref['active'].astype(bool)).sum())} inactive")
    return ref


def trading_dates(start, end):
    d, out = start, []
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def fetch_massive_daily(start, end, client=None, log=print, max_days=None):
    """Download grouped daily bars one date at a time. Resumable."""
    client = client or MassiveClient()
    out_dir = PIT_DIR / "massive_daily"
    todo = [d for d in trading_dates(start, end) if not (out_dir / f"{d}.parquet").exists()]
    if max_days:
        todo = todo[:max_days]
    log(f"massive daily: {len(todo)} dates to fetch")
    cols = {"T": "symbol", "o": "open", "h": "high", "l": "low", "c": "close",
            "v": "volume", "vw": "vwap", "n": "trades"}
    for i, d in enumerate(todo, 1):
        r = client.get(f"/v2/aggs/grouped/locale/us/market/stocks/{d}", adjusted="true", include_otc="true")
        if r.status_code == 403:
            log(f"  {d}: outside plan window, skipping")
            continue
        if not r.ok:
            log(f"  {d}: HTTP {r.status_code} {r.text[:100]}")
            continue
        res = r.json().get("results") or []
        df = pd.DataFrame(res).rename(columns=cols)
        df = df[[c for c in cols.values() if c in df.columns]] if len(df) else pd.DataFrame(columns=list(cols.values()))
        df.insert(0, "date", pd.Timestamp(d))
        _atomic_write(df, out_dir / f"{d}.parquet")
        if i % 20 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)} (last {d}, {len(df)} rows)")


def fetch_marketparquet(api_key=None, log=print):
    """Download the free last-365-days daily stock files."""
    key = api_key or os.getenv("MARKETPARQUET_API_KEY")
    if not key:
        raise ValueError("MARKETPARQUET_API_KEY not set")
    H = {"Authorization": f"Bearer {key}"}
    end = date.today()
    start = end - timedelta(days=364)
    m = requests.get(f"{MP}/manifest/stock_daily", params={"start": str(start), "end": str(end)},
                     headers=H, timeout=60).json()
    files = m.get("files", [])
    out_dir = PIT_DIR / "mp_daily"
    out_dir.mkdir(parents=True, exist_ok=True)

    def get(f):
        name = f["filename"].replace("stock_daily_", "")
        p = out_dir / name
        if not p.exists():
            p.write_bytes(requests.get(f["download_url"], timeout=120).content)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(get, files))
    log(f"marketparquet: {len(files)} files")


DAILY_START = date(2023, 9, 1)  # ~200 trading days of warm-up before the 2024-10 start, for the 200 SMA
MARKET_SYMBOLS = ("SPY", "QQQ", "IWM")


def fetch_market(start=DAILY_START, end=None, log=print):
    """SPY / QQQ / IWM daily bars (from `start`) and 1-minute bars (from 2024-10-01), including extended hours."""
    from alpaca.data.timeframe import TimeFrame
    end = end or date.today()
    client = _alpaca()
    out = PIT_DIR / "market"
    out.mkdir(parents=True, exist_ok=True)
    for sym in MARKET_SYMBOLS:
        d = _alpaca_bars(client, [sym], TimeFrame.Day, pd.Timestamp(start, tz=NY),
                         min(pd.Timestamp(end, tz=NY), pd.Timestamp.now(tz=NY) - pd.Timedelta(days=1)))
        d["date"] = d["timestamp"].dt.tz_localize(None).dt.normalize()
        d[["date", "open", "high", "low", "close", "volume"]].to_parquet(out / f"{sym}_daily.parquet", index=False)
        frames, cur = [], pd.Timestamp("2024-10-01", tz=NY)
        limit = pd.Timestamp.now(tz=NY) - pd.Timedelta(minutes=20)  # free plan blocks the latest 15 min of SIP
        while cur < min(pd.Timestamp(end, tz=NY) + pd.Timedelta(days=1), limit):
            nxt = min(cur + pd.DateOffset(months=2), limit)
            m = _alpaca_bars(client, [sym], TimeFrame.Minute, cur, nxt)
            if len(m):
                frames.append(m[["timestamp", "open", "high", "low", "close", "volume"]])
            cur = nxt
        dedupe_on(pd.concat(frames, ignore_index=True), ["timestamp"]).to_parquet(out / f"{sym}_minute.parquet", index=False)
        log(f"{sym}: {len(d)} daily bars, {sum(len(f) for f in frames)} minute bars")


def fetch_news(symbols, start=DAILY_START, end=None, workers=4, log=print):
    """Alpaca (Benzinga) news per symbol over the whole window -> news/news.parquet (one row per symbol x article)."""
    from src.data.news_client import AlpacaNewsClient
    end = end or date.today() + timedelta(days=1)
    client = AlpacaNewsClient()
    syms = sorted(set(symbols))

    def one(sym):
        try:
            df = client.fetch_news([sym], start=str(start), end=str(end))
        except Exception:
            return sym, None
        if df is None or df.empty:
            return sym, pd.DataFrame()
        df = df[[sym in list(s) for s in df["symbols"]]]
        keep = [c for c in ("created_at", "headline", "summary", "source") if c in df.columns]
        df = df[keep].copy()
        df.insert(0, "symbol", sym)
        return sym, df

    frames, failed = [], []
    with ThreadPoolExecutor(workers) as ex:
        for i, (sym, df) in enumerate(ex.map(one, syms), 1):
            (frames.append(df) if df is not None else failed.append(sym))
            if i % 200 == 0:
                log(f"  news {i}/{len(syms)} symbols")
    news = dedupe_on(pd.concat([f for f in frames if len(f)], ignore_index=True), ["symbol", "created_at", "headline"])
    news["created_at"] = pd.to_datetime(news["created_at"], utc=True)
    out = PIT_DIR / "news"
    out.mkdir(parents=True, exist_ok=True)
    news.to_parquet(out / "news.parquet", index=False)
    log(f"news: {len(news)} rows, {news['symbol'].nunique()} of {len(syms)} symbols have articles, {len(failed)} failed")


def fetch_otc_reference(client=None, log=print):
    """Massive reference for OTC / pink-sheet tickers (market=otc, active and inactive) -> otc_reference.parquet."""
    client = client or MassiveClient()
    frames = []
    for active in ("true", "false"):
        url, params, pages = "/v3/reference/tickers", dict(market="otc", active=active, limit=1000), 0
        while url:
            r = client.get(url, **params)
            if not r.ok:
                raise RuntimeError(f"otc reference {active}: {r.status_code} {r.text[:200]}")
            j = r.json()
            frames.append(pd.DataFrame(j.get("results", [])).assign(_active_query=active))
            pages += 1
            url, params = j.get("next_url"), {}
    otc = pd.concat(frames, ignore_index=True)
    keep = [c for c in ["ticker", "name", "type", "market", "active", "cik", "delisted_utc"] if c in otc.columns]
    otc = otc[keep].drop_duplicates(subset=["ticker", "name"])
    _atomic_write(otc, PIT_DIR / "otc_reference.parquet")
    log(f"otc reference: {len(otc)} rows")


EDGAR_FORMS = ("8-K", "8-K/A", "S-1", "S-1/A", "S-3", "S-3/A", "F-1", "F-3", "424B1", "424B2", "424B3", "424B4", "424B5", "424B7",
               "NT 10-K", "NT 10-Q", "10-K", "10-Q", "SC 13D", "SC 13G", "SCHEDULE 13D", "SCHEDULE 13G")


def fetch_edgar(symbols, start=DAILY_START, workers=6, log=print):
    """
    SEC EDGAR filings (8-K with item codes, offering and late-filing forms, 10-K/10-Q dates) per symbol using the CIK in the
    Massive reference -> edgar/filings.parquet. SEC asks for a descriptive User-Agent: set EDGAR_USER_AGENT="Name email".
    Rate limited to about 8 requests per second.
    """
    import threading

    ua = os.getenv("EDGAR_USER_AGENT", "TradingFramework research (set EDGAR_USER_AGENT)")
    ref = pd.read_parquet(PIT_DIR / "reference.parquet").sort_values("active", ascending=False).drop_duplicates("ticker").set_index("ticker")
    ciks = {s: str(ref.at[s, "cik"]).zfill(10) for s in sorted(set(symbols)) if s in ref.index and pd.notna(ref.at[s, "cik"])}
    lock, state = threading.Lock(), {"t": 0.0}

    def get(url):
        for attempt in range(5):
            with lock:
                wait = 0.125 - (time.time() - state["t"])
                if wait > 0:
                    time.sleep(wait)
                state["t"] = time.time()
            r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=60)
            if r.status_code in (429, 503):
                time.sleep(5 * (attempt + 1))
                continue
            return r
        return r

    def rows_from(block, sym, cik):
        df = pd.DataFrame(block)
        if df.empty:
            return df
        df = df[df["form"].isin(EDGAR_FORMS)].copy()
        if df.empty:
            return df
        df["acceptance_utc"] = pd.to_datetime(df["acceptanceDateTime"], utc=True, errors="coerce")
        df = df[df["acceptance_utc"] >= pd.Timestamp(start, tz="UTC")]
        df["symbol"], df["cik"] = sym, cik
        df["items"] = df.get("items", "")
        return df[["symbol", "cik", "form", "items", "acceptance_utc", "accessionNumber", "primaryDocument"]]

    def one(item):
        sym, cik = item
        try:
            r = get(f"https://data.sec.gov/submissions/CIK{cik}.json")
            if not r.ok:
                return sym, None
            j = r.json()
            out = [rows_from(j["filings"]["recent"], sym, cik)]
            for f in j["filings"].get("files", []):
                if pd.Timestamp(f.get("filingTo", "2000-01-01")) < pd.Timestamp(start):
                    continue
                rr = get(f"https://data.sec.gov/submissions/{f['name']}")
                if rr.ok:
                    out.append(rows_from(rr.json(), sym, cik))
            return sym, pd.concat(out, ignore_index=True)
        except Exception:
            return sym, None

    frames, failed = [], []
    with ThreadPoolExecutor(workers) as ex:
        for i, (sym, df) in enumerate(ex.map(one, ciks.items()), 1):
            (frames.append(df) if df is not None else failed.append(sym))
    filings = dedupe_on(pd.concat([f for f in frames if len(f)], ignore_index=True), ["symbol", "accessionNumber"])
    out = PIT_DIR / "edgar"
    out.mkdir(parents=True, exist_ok=True)
    filings.to_parquet(out / "filings.parquet", index=False)
    log(f"edgar: {len(filings)} filings for {filings['symbol'].nunique()} of {len(ciks)} symbols with a CIK ({len(failed)} failed)")


SKIP_TYPES = {"WARRANT", "RIGHT", "UNIT", "SP", "ETF", "FUND", "PFD", "ETS", "ETN", "INDEX"}
NY = "America/New_York"


def _alpaca():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"))


def universe_symbols(window_start):
    """Tickers that were listed at any point on/after window_start (active, or delisted after it)."""
    ref = pd.read_parquet(PIT_DIR / "reference.parquet")
    ref = ref[~ref["type"].isin(SKIP_TYPES)]
    alive = ref["active"].astype(bool) | (ref["delisted_utc"] >= pd.Timestamp(window_start, tz="UTC"))
    return sorted(ref.loc[alive, "ticker"].dropna().unique())


def _alpaca_bars(client, symbols, tf, start, end):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.enums import DataFeed, Adjustment
    req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=tf, start=start, end=end,
                           feed=DataFeed.SIP, adjustment=Adjustment.SPLIT)
    df = client.get_stock_bars(req).df
    if df.empty:
        return df
    df = df.reset_index()
    df["timestamp"] = df["timestamp"].dt.tz_convert(NY)
    return df


def fetch_alpaca_daily(start, end, batch=100, log=print):
    """Daily split-adjusted SIP bars for the whole PIT universe. Resumable per batch."""
    from alpaca.data.timeframe import TimeFrame
    client = _alpaca()
    syms = universe_symbols(start)
    out_dir = PIT_DIR / "alpaca_daily"
    log(f"alpaca daily: {len(syms)} symbols, {start}..{end}")
    s, e = pd.Timestamp(start, tz=NY), pd.Timestamp(end, tz=NY) + pd.Timedelta(days=1)
    for bi, i in enumerate(range(0, len(syms), batch)):
        path = out_dir / f"part_{bi:04d}.parquet"
        if path.exists():
            continue
        chunk = syms[i:i + batch]
        try:
            df = _alpaca_bars(client, chunk, TimeFrame.Day, s, e)
        except Exception:
            frames = []
            for sym in chunk:
                try:
                    f = _alpaca_bars(client, [sym], TimeFrame.Day, s, e)
                    if len(f):
                        frames.append(f)
                except Exception:
                    pass
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(df):
            df["date"] = df["timestamp"].dt.tz_localize(None).dt.normalize()
            df = df[["date", "symbol", "open", "high", "low", "close", "volume", "trade_count", "vwap"]]
        else:
            df = pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume", "trade_count", "vwap"])
        _atomic_write(df, path)
        if bi % 10 == 0:
            log(f"  batch {bi + 1}/{-(-len(syms) // batch)}: {len(df)} rows")


def daily_candidates(daily, min_price=2.0, max_price=20.0, min_avg_volume=500_000, min_rel_volume=2.0,
                     lookback=20, use_atr_pivot=True, lookback_bars=None):
    """
    Daily-only prefilter for the overnight strategy. Loose on purpose: full-day volume stands in for
    15:55 volume, and a breakout is either a new rolling-20 high or a close through an atr_pivot
    resistance level, so the set covers both resistance methods.
    """
    d = daily.sort_values(["symbol", "date"]).copy()
    g = d.groupby("symbol", sort=False)
    d["prev_close"] = g["close"].shift(1)
    d["avg_vol"] = g["volume"].transform(lambda v: v.shift(1).rolling(lookback).mean())
    d["prior_high"] = g["high"].transform(lambda v: v.shift(1).rolling(lookback).max())
    base = (
        d["close"].between(min_price, max_price)
        & (d["avg_vol"] >= min_avg_volume)
        & (d["volume"] >= min_rel_volume * d["avg_vol"])
        & (d["close"] > d["prev_close"])
        & (d["open"] > d["prev_close"])
    )
    ok = base & (d["close"] > d["prior_high"])
    if use_atr_pivot:
        from src.screening.historical import ATR_PIVOT_CONFIG
        from src.screening.resistance import ResistanceDetector
        from dataclasses import replace
        det = ResistanceDetector(replace(ATR_PIVOT_CONFIG, lookback_bars=lookback_bars) if lookback_bars else ATR_PIVOT_CONFIG)
        for sym, rows in d[base & ~ok].groupby("symbol"):
            p = det.prepare(d.loc[d["symbol"] == sym, ["date", "open", "high", "low", "close", "volume"]]
                            .set_index("date"))
            for idx, r in rows.iterrows():
                k = det.known_through_before(p, r["date"])
                if k < 0:
                    continue
                lad = det.ladder(p, k, price=r["prev_close"])
                if ((lad["upper"] > r["prev_close"]) & (lad["upper"] < r["close"])).any():
                    ok.at[idx] = True
    return dedupe_on(drop_daily_aliases(d.loc[ok, ["date", "symbol"]], d), ["symbol", "date"]).reset_index(drop=True)  # one security per bar


def fetch_alpaca_minute(cands, log=print):
    """1-minute bars for candidate symbol-days: the signal date and the next session. One request per date."""
    from alpaca.data.timeframe import TimeFrame
    client = _alpaca()
    out_dir = PIT_DIR / "alpaca_minute"
    dates = sorted(cands["date"].unique())
    log(f"alpaca minute: {len(cands)} symbol-days over {len(dates)} dates")
    for i, d in enumerate(dates, 1):
        path = out_dir / f"{pd.Timestamp(d).date()}.parquet"
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        have = set(old["symbol"]) if len(old) else set()
        syms = sorted(set(cands.loc[cands["date"] == d, "symbol"]) - have)
        if not syms:
            continue
        s = pd.Timestamp(d, tz=NY)
        e = s + pd.Timedelta(days=5)
        frames = []
        for j in range(0, len(syms), 100):
            try:
                f = _alpaca_bars(client, syms[j:j + 100], TimeFrame.Minute, s, e)
            except Exception:
                f = pd.concat([x for x in (_safe_bars(client, sym, s, e) for sym in syms[j:j + 100]) if len(x)] or [pd.DataFrame()])
            if len(f):
                frames.append(f)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(df):
            day = df["timestamp"].dt.normalize()
            first2 = df.assign(day=day).groupby("symbol")["day"].transform(
                lambda s: s.map({v: i for i, v in enumerate(sorted(s.unique()))}))
            df = df[first2.values < 2]
            df = df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "trade_count", "vwap"]]
        if len(old):
            df = pd.concat([old, df], ignore_index=True)
        df = clean_minute(df) if len(df) else df
        _atomic_write(df if len(df) else empty_minute_frame(), path)
        if i % 20 == 0 or i == len(dates):
            log(f"  {i}/{len(dates)} dates (last {pd.Timestamp(d).date()}, {len(df)} rows)")


def _safe_bars(client, sym, s, e):
    from alpaca.data.timeframe import TimeFrame
    try:
        return _alpaca_bars(client, [sym], TimeFrame.Minute, s, e)
    except Exception:
        return pd.DataFrame()


def _read_mp(path):
    x = pd.read_parquet(path)
    x["date"] = pd.to_datetime(x["date"] if "date" in x else x["timestamp"]).dt.normalize()
    return x[["date", "symbol", "open", "high", "low", "close", "volume"]]


def load_source(name):
    d = PIT_DIR / name
    files = sorted(d.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    read = _read_mp if name == "mp_daily" else pd.read_parquet
    return pd.concat([read(f) for f in files], ignore_index=True)


def build_daily(log=print):
    """Merge Massive (primary) with MarketParquet (fills gaps) into daily_pit.parquet."""
    parts, seen = [], set()
    universe = set(universe_symbols(pd.Timestamp(DAILY_START)))
    for name, label in (("alpaca_daily", "alpaca"), ("massive_daily", "massive"), ("mp_daily", "marketparquet")):
        src = load_source(name)
        if not len(src):
            continue
        src = src[src["symbol"].isin(universe)].assign(src=label)
        if seen:
            src = src[[k not in seen for k in zip(src["date"], src["symbol"])]]
        seen |= set(zip(src["date"], src["symbol"]))
        parts.append(src)
        log(f"{label}: {len(src)} rows used")
    df = clean_daily(pd.concat(parts, ignore_index=True)).sort_values(["symbol", "date"]).reset_index(drop=True)
    ref_path = PIT_DIR / "reference.parquet"
    if ref_path.exists():
        ref = pd.read_parquet(ref_path)
        ref = ref.sort_values("active", ascending=False).drop_duplicates("ticker", keep="first")
        df = df.merge(ref[["ticker", "type", "primary_exchange", "delisted_utc"]]
                      .rename(columns={"ticker": "symbol"}), on="symbol", how="left")
    _atomic_write(df, PIT_DIR / "daily_pit.parquet")
    log(f"daily_pit: {len(df)} rows, {df['symbol'].nunique()} symbols, "
        f"{df['date'].min().date()}..{df['date'].max().date()}")
    return df


def validate(sample=60, log=print):
    """Integrity checks (exit 1 on errors), then coverage, source agreement, delisting coverage and a yfinance cross-check."""
    import numpy as np
    from src.data.pit_integrity import require_valid_pit_data
    require_valid_pit_data(log=log)
    lines = []
    out = lambda s="": (lines.append(s), log(s))
    df = pd.read_parquet(PIT_DIR / "daily_pit.parquet")
    mas, mp = load_source("massive_daily"), load_source("mp_daily")
    out(f"merged: {len(df)} rows, {df['symbol'].nunique()} symbols, {df['date'].min().date()}..{df['date'].max().date()}")
    out(f"rows by source: {df['src'].value_counts().to_dict()}")
    per_day = df.groupby("date").size()
    out(f"symbols/day: median {int(per_day.median())}, min {int(per_day.min())}, max {int(per_day.max())}")

    both = mas.merge(mp, on=["date", "symbol"], suffixes=("_ma", "_mp"))
    if len(both):
        rel = (both["close_ma"] / both["close_mp"] - 1).abs()
        out(f"massive vs marketparquet on {len(both)} overlapping rows: "
            f"median |close diff| {rel.median():.2e}, >1%: {(rel > 0.01).mean():.2%}, >10%: {(rel > 0.10).mean():.3%}")
        bad = both.loc[rel > 0.10].groupby("symbol").size().sort_values(ascending=False).head(10)
        if len(bad):
            out(f"  symbols with most >10% disagreements: {bad.to_dict()}")
        only_ma = set(mas["symbol"]) - set(mp["symbol"])
        only_mp = set(mp["symbol"]) - set(mas["symbol"])
        out(f"symbols only in massive: {len(only_ma)}, only in marketparquet: {len(only_mp)}")

    last = df.groupby("symbol")["date"].max()
    end = df["date"].max()
    ended = last[last < end - pd.Timedelta(days=5)]
    out(f"symbols whose last bar is >5 days before dataset end: {len(ended)}")
    if "delisted_utc" in df:
        d = df.drop_duplicates("symbol").set_index("symbol")["delisted_utc"]
        known = d.reindex(ended.index).notna().mean() if len(ended) else float("nan")
        out(f"  of those, share with a Massive delisted_utc: {known:.1%}")

    known_dead = {"K": "2025-12-11", "DAY": "2026-02-04", "CMA": "2026-02-01",
                  "HOLX": "2026-04-07", "EA": "2026-08-04"}
    out("known delistings (symbol: last bar vs merger date):")
    for s, m in known_dead.items():
        out(f"  {s}: last bar {last[s].date() if s in last.index else 'ABSENT'} (merger {m})")

    try:
        import yfinance as yf
        rng = np.random.default_rng(0)
        pool = df[df["date"] >= end - pd.Timedelta(days=60)]
        pool = pool[(pool["close"] > 2) & (pool["volume"] > 500_000)]["symbol"].unique()
        syms = list(rng.choice(pool, min(sample, len(pool)), replace=False)) + list(known_dead)
        px = yf.download(syms, start=str(end - pd.Timedelta(days=60))[:10], end=str(end + pd.Timedelta(days=1))[:10],
                         auto_adjust=False, progress=False, group_by="ticker", threads=True)
        have, diffs = 0, []
        for s in syms:
            try:
                c = px[s]["Close"].dropna()
            except KeyError:
                continue
            if c.empty:
                continue
            have += 1
            mine = df[df["symbol"] == s].set_index("date")["close"]
            j = pd.concat([c.rename("y"), mine.rename("m")], axis=1).dropna()
            if len(j):
                diffs.append(float(((j["m"] / j["y"]) - 1).abs().median()))
        out(f"yfinance returned data for {have}/{len(syms)} symbols "
            f"({len([s for s in known_dead if s in px.columns.get_level_values(0) and not px[s]['Close'].dropna().empty])}/{len(known_dead)} known-delisted)")
        if diffs:
            out(f"yfinance vs dataset median |close diff|: {np.median(diffs):.2e}; symbols >1%: {(np.array(diffs) > 0.01).sum()}")
    except Exception as e:
        out(f"yfinance check failed: {e}")

    (PIT_DIR / "report.txt").write_text("\n".join(lines), encoding="utf-8")
