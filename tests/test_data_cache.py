import os
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.testing import assert_frame_equal

from src.framework import data_cache
from src.data.pit_dataset import _atomic_write
from src.data.pit_integrity import empty_minute_frame, load_minute


def _minute_frame():
    return pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB"],
        "timestamp": pd.to_datetime(["2025-01-02 15:55", "2025-01-02 15:56", "2025-01-02 15:55"]),
        "open": [10.0, 10.1, 20.0],
        "high": [10.1, 10.2, 20.1],
        "low": [9.9, 10.0, 19.9],
        "close": [10.05, 10.15, 20.05],
        "volume": [100, 120, 200],
    })


def test_minute_cache_hit_matches_uncached_and_invalidates(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _minute_frame().to_parquet(source / "2025-01-02.parquet", index=False)
    cache = tmp_path / "cache"
    monkeypatch.setattr(data_cache, "MINUTE_CACHE", cache)
    monkeypatch.setenv("DATA_CACHE", "0")
    uncached = data_cache.load_minute(["AAA", "BBB"], root=source)
    monkeypatch.setenv("DATA_CACHE", "1")
    cached = data_cache.load_minute(["AAA", "BBB"], root=source)
    assert_frame_equal(uncached.reset_index(drop=True), cached.reset_index(drop=True), check_like=True)

    before = sorted(cache.glob("*"))
    old_stat = (source / "2025-01-02.parquet").stat()
    os.utime(source / "2025-01-02.parquet", ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 2_000_000_000))
    rebuilt = data_cache.load_minute(["AAA", "BBB"], root=source)
    assert_frame_equal(uncached.reset_index(drop=True), rebuilt.reset_index(drop=True), check_like=True)
    assert len(list(cache.glob("*"))) > len(before)


def test_data_cache_zero_bypasses_cache(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _minute_frame().to_parquet(source / "bars.parquet", index=False)
    monkeypatch.setattr(data_cache, "MINUTE_CACHE", tmp_path / "cache")
    monkeypatch.setenv("DATA_CACHE", "1")
    data_cache.load_minute(["AAA"], root=source)
    monkeypatch.setenv("DATA_CACHE", "0")
    uncached = data_cache.load_minute(["AAA"], root=source)
    assert len(uncached) == 2


def test_load_minute_drops_empty_files_and_all_empty_is_typed(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    normal = _minute_frame()
    normal["timestamp"] = pd.to_datetime(normal["timestamp"], utc=True).dt.tz_convert("America/New_York")
    normal.to_parquet(source / "normal.parquet", index=False)
    empty_minute_frame().iloc[:0].to_parquet(source / "empty.parquet", index=False)
    actual = load_minute(source)
    expected = normal.drop_duplicates(["symbol", "timestamp"], keep="last").reset_index(drop=True)
    assert str(actual["timestamp"].dtype.tz) == "America/New_York"
    assert_frame_equal(actual.reset_index(drop=True), expected, check_like=True)
    all_empty = tmp_path / "all_empty"
    all_empty.mkdir()
    empty_minute_frame().iloc[:0].to_parquet(all_empty / "a.parquet", index=False)
    result = load_minute(all_empty)
    assert result.empty
    assert str(result["timestamp"].dtype.tz) == "America/New_York"


def test_empty_minute_writer_keeps_typed_schema(tmp_path):
    path = tmp_path / "empty.parquet"
    _atomic_write(empty_minute_frame(), path)
    written = pd.read_parquet(path)
    assert written.empty
    assert str(written["timestamp"].dtype.tz) == "America/New_York"
