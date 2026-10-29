"""Wide daily-bar panel (date x symbol) built from a folder of per-symbol parquet files."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.framework.data_cache import CACHE_ROOT, ROOT, _cache_enabled, _key, _rebuild_requested, _source_fingerprint
from src.utils import timing

PANEL_CACHE = CACHE_ROOT / "daily_panel"
PANEL_VERSION = "daily-panel-v1"
FIELDS = ("open", "close", "volume")
SUFFIX = "_1Day_split.parquet"
NY = "America/New_York"


def load_sector_map(simfin_dir=None):
    """SimFin ticker -> Industry/Sector (current snapshot, not point-in-time)."""
    simfin_dir = Path(simfin_dir or ROOT / "data" / "simfin")
    companies = pd.read_csv(simfin_dir / "us-companies.csv", sep=";", usecols=["Ticker", "IndustryId"])
    industries = pd.read_csv(simfin_dir / "industries.csv", sep=";")
    frame = companies.dropna(subset=["Ticker", "IndustryId"]).merge(industries, on="IndustryId", how="inner")
    frame = frame.drop_duplicates("Ticker", keep="first").set_index("Ticker")
    return frame[["Industry", "Sector"]]


def _read_symbol(path):
    frame = pd.read_parquet(path, columns=list(FIELDS))
    dates = frame.index.tz_convert(NY).tz_localize(None).normalize()
    frame.index = dates
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def _build(root, symbols):
    frames = {}
    with timing.stage("panel read files"):
        for symbol in symbols:
            frames[symbol] = _read_symbol(root / f"{symbol}{SUFFIX}")
    with timing.stage("panel assemble"):
        dates = np.unique(np.concatenate([f.index.values for f in frames.values()]))
        arrays = {k: np.full((len(dates), len(symbols)), np.nan) for k in FIELDS}
        for j, symbol in enumerate(symbols):
            rows = np.searchsorted(dates, frames[symbol].index.values)
            for k in FIELDS:
                arrays[k][rows, j] = frames[symbol][k].to_numpy(float)
    return dates, arrays


def load_panel(data_dir, symbols=None, rebuild=False):
    """Load open/close/volume matrices for `symbols` (default: every file) from `data_dir`, with a fingerprinted cache."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data dir not found: {root}")
    available = sorted(p.name[: -len(SUFFIX)] for p in root.glob(f"*{SUFFIX}"))
    wanted = available if symbols is None else sorted(set(available) & set(symbols))
    if not wanted:
        raise ValueError(f"no *{SUFFIX} files in {root} match the requested symbols")
    key = _key(PANEL_VERSION, _source_fingerprint(root), wanted)
    cache_dir = PANEL_CACHE / key
    if _cache_enabled() and not _rebuild_requested(rebuild) and (cache_dir / "manifest.json").exists():
        with timing.stage("panel cache read"):
            dates = np.load(cache_dir / "dates.npy")
            arrays = {k: np.load(cache_dir / f"{k}.npy") for k in FIELDS}
        return {"dates": pd.DatetimeIndex(dates), "symbols": wanted, **arrays}
    dates, arrays = _build(root, wanted)
    if _cache_enabled():
        with timing.stage("panel cache write"):
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_dir / "dates.npy", dates)
            for k in FIELDS:
                np.save(cache_dir / f"{k}.npy", arrays[k])
            (cache_dir / "manifest.json").write_text(json.dumps({"key": key, "version": PANEL_VERSION, "symbols": len(wanted)}), encoding="utf-8")
    return {"dates": pd.DatetimeIndex(dates), "symbols": wanted, **arrays}


def liquidity_ranks(close, volume, window=60, min_obs=40, min_price=5.0, min_history=250):
    """Rank (1 = most liquid) of trailing median dollar volume per day, using data through the prior session only.

    Names need a prior close >= min_price and >= min_history prior valid bars; others get NaN.
    """
    close_df, volume_df = pd.DataFrame(close), pd.DataFrame(volume)
    median_dv = (close_df * volume_df).rolling(window, min_periods=min_obs).median().shift(1)
    history = close_df.notna().cumsum().shift(1)
    ok = (close_df.shift(1) >= min_price) & (history >= min_history)
    return median_dv.where(ok).rank(axis=1, ascending=False, method="first").to_numpy()
