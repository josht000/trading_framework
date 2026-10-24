"""Persistent, fingerprinted data loaders for the study inputs."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from src.utils import timing


ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / "data" / "pit" / "cache"
MINUTE_CACHE = CACHE_ROOT / "minute"
SIMFIN_CACHE = CACHE_ROOT / "simfin_market_cap"
MINUTE_VERSION = "minute-v2-dedupe-et-symbol-partition"
SIMFIN_VERSION = "simfin-market-cap-v1"


def _source_fingerprint(root):
    files = []
    for path in sorted(Path(root).glob("*.parquet")):
        stat = path.stat()
        files.append({"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return files


def _key(version, sources, extra=()):
    payload = json.dumps({"version": version, "sources": sources, "extra": list(extra)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_enabled():
    return os.getenv("DATA_CACHE", "1") != "0"


def _rebuild_requested(rebuild=False):
    return rebuild or os.getenv("DATA_CACHE_REBUILD", "0") == "1"


def load_candidates_daily(start, end):
    candidates = pd.read_parquet(ROOT / "data" / "pit" / "candidates.parquet")
    candidates = candidates[(candidates["date"] >= start) & (candidates["date"] <= end)]
    daily = pd.read_parquet(
        ROOT / "data" / "pit" / "daily_pit.parquet",
        columns=["date", "symbol", "open", "high", "low", "close", "volume", "delisted_utc"],
    )
    return candidates, daily


def _minute_uncached(root, symbols):
    from src.data.pit_integrity import load_minute as _load_minute

    with timing.stage("minute uncached load"):
        minute = _load_minute(root=root, symbols=set(symbols))
    with timing.stage("minute tz convert"):
        if str(getattr(minute["timestamp"].dtype, "tz", None)) != "America/New_York":
            minute["timestamp"] = pd.to_datetime(minute["timestamp"], utc=True).dt.tz_convert("America/New_York")
    return minute


def _write_minute_cache(minute, key, manifest):
    MINUTE_CACHE.mkdir(parents=True, exist_ok=True)
    final = MINUTE_CACHE / key
    if final.exists():
        shutil.rmtree(final)
    final.mkdir()
    with timing.stage("minute cache sort/write"):
        minute = minute.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)
        minute.to_parquet(final / "dataset", index=False, partition_cols=["symbol"])
        (final / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return final / "dataset"


def _minute_cached(minute_symbols, root, rebuild=False):
    symbols = tuple(sorted(set(minute_symbols)))
    sources = _source_fingerprint(root)
    key = _key(MINUTE_VERSION, sources, symbols)
    cache_dir = MINUTE_CACHE / key
    manifest_path = cache_dir / "manifest.json"
    dataset = cache_dir / "dataset"
    valid = (
        _cache_enabled()
        and not _rebuild_requested(rebuild)
        and manifest_path.exists()
        and dataset.exists()
    )
    if valid:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid = manifest.get("key") == key and manifest.get("version") == MINUTE_VERSION
        except (OSError, ValueError):
            valid = False
    if valid:
        with timing.stage("minute cache read"):
            return pd.read_parquet(dataset, filters=[("symbol", "in", list(symbols))])
    minute = _minute_uncached(root, symbols)
    manifest = {
        "key": key,
        "version": MINUTE_VERSION,
        "symbols": list(symbols),
        "sources": sources,
        "columns": list(minute.columns),
    }
    _write_minute_cache(minute, key, manifest)
    return minute


def load_minute(symbols, root=None, by_symbol=False, rebuild=False):
    """Load normalized minute bars, using a source-fingerprinted symbol dataset."""
    root = Path(root or ROOT / "data" / "pit" / "alpaca_minute")
    minute = _minute_cached(symbols, root, rebuild=rebuild) if _cache_enabled() else _minute_uncached(root, symbols)
    if not by_symbol:
        return minute
    with timing.stage("minute cache group"):
        grouped = {
            symbol: frame.set_index("timestamp").sort_index()
            for symbol, frame in minute.groupby("symbol", sort=False)
        }
    return minute, grouped


def _simfin_sources():
    return [
        {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in sorted((ROOT / "data" / "simfin").rglob("*"))
        if path.is_file()
    ]


def load_simfin_contexts(symbols):
    contexts = {symbol: {} for symbol in symbols}
    if not os.getenv("SIMFIN_API_KEY") or not _cache_enabled():
        return _load_simfin_uncached(contexts, symbols)
    sources = _simfin_sources()
    symbols = tuple(sorted(set(symbols)))
    key = _key(SIMFIN_VERSION, sources, symbols)
    path = SIMFIN_CACHE / f"{key}.parquet"
    if path.exists() and not _rebuild_requested():
        with timing.stage("simfin cache read"):
            cached = pd.read_parquet(path)
        return _contexts_from_frame(contexts, cached)
    out = _load_simfin_uncached(contexts, symbols)
    rows = []
    for symbol, context in out.items():
        series = context.get("market_cap")
        if series is not None:
            rows.extend({"symbol": symbol, "date": date, "market_cap": value} for date, value in series.items())
    SIMFIN_CACHE.mkdir(parents=True, exist_ok=True)
    with timing.stage("simfin cache write"):
        pd.DataFrame(rows, columns=["symbol", "date", "market_cap"]).to_parquet(path, index=False)
    return out


def _contexts_from_frame(contexts, frame):
    for symbol, group in frame.groupby("symbol"):
        series = group.set_index("date")["market_cap"].sort_index()
        series.index = pd.to_datetime(series.index).normalize()
        contexts[symbol]["market_cap"] = series.rename("market_cap")
    return contexts


def _load_simfin_uncached(contexts, symbols):
    from src.data.simfin_store import SimFinStore

    with timing.stage("simfin uncached load"):
        store = SimFinStore()
        for symbol in symbols:
            cap = store.market_cap_series(symbol)
            if len(cap) > 0:
                contexts[symbol]["market_cap"] = cap
    return contexts


def load_contexts(symbols, start, end, require_news):
    if require_news:
        from main import build_symbol_contexts

        return build_symbol_contexts(symbols, start, end)
    return load_simfin_contexts(symbols)


def load_news(contexts):
    path = ROOT / "data" / "pit" / "news" / "news.parquet"
    if not path.exists():
        return None
    news = pd.read_parquet(path)
    for symbol, group in news.groupby("symbol"):
        if symbol in contexts:
            contexts[symbol]["news_df"] = group.sort_values("created_at").reset_index(drop=True)
    return news
