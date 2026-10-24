"""Synthetic tests for src/data/pit_integrity.py: dedupe helpers and every validate_pit_data check."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.exit_sim import TradeWindow, build_windows, save_windows
from src.data import pit_integrity as pi

NY = "America/New_York"


def minute(sym, day="2025-01-02", start="15:00", n=30, px=10.0):
    ts = pd.date_range(f"{day} {start}", periods=n, freq="min", tz=NY)
    c = px + np.arange(n) * 0.01
    return pd.DataFrame({"timestamp": ts, "symbol": sym, "open": c, "high": c + 0.02, "low": c - 0.02, "close": c,
                         "volume": 1000.0, "trade_count": 10.0, "vwap": c})


def daily(sym, day="2025-01-02", close=5.0, vol=200_000.0):
    return pd.DataFrame({"date": [pd.Timestamp(day)], "symbol": sym, "open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": vol})


class PitDir:
    """A throw-away data/pit tree; `minute_files` maps file date -> frame."""

    def __init__(self, minute_files, daily_df=None, cands=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        (self.path / "alpaca_minute").mkdir()
        for d, df in minute_files.items():
            df.to_parquet(self.path / "alpaca_minute" / f"{d}.parquet", index=False)
        if daily_df is not None:
            daily_df.to_parquet(self.path / "daily_pit.parquet", index=False)
        if cands is not None:
            cands.to_parquet(self.path / "candidates.parquet", index=False)

    def validate(self):
        return pi.validate_pit_data(self.path, trades=False, windows=False, log=lambda *_: None)


def errors_of(rep):
    return " | ".join(rep["errors"])


class DedupeTests(unittest.TestCase):
    def test_dedupe_on_keeps_last(self):
        df = pd.DataFrame({"symbol": ["A", "A", "B"], "timestamp": [1, 1, 1], "v": [1, 2, 3]})
        out = pi.dedupe_on(df, ["symbol", "timestamp"])
        self.assertEqual(out["v"].tolist(), [2, 3])

    def test_clean_minute_drops_duplicate_bars_and_sorts(self):
        m = pd.concat([minute("A"), minute("A").iloc[:10], minute("B")], ignore_index=True)
        out = pi.clean_minute(m)
        self.assertEqual(len(out), 60)
        self.assertFalse(out.duplicated(["symbol", "timestamp"]).any())
        self.assertTrue(out["symbol"].is_monotonic_increasing)

    def test_dedupe_trades_drops_ticker_twin_keeps_first(self):
        t = pd.DataFrame({"symbol": ["AAA", "AAB", "CCC"], "entry_time": ["2025-01-02 15:55:00-05:00"] * 3,
                          "prev_close": [2.0, 2.0, 3.0], "price": [2.5, 2.5, 3.5]})
        self.assertEqual(pi.dedupe_trades(t)["symbol"].tolist(), ["AAA", "CCC"])

    def test_alias_candidates_keep_alphabetical_first_except_traded(self):
        d = pd.concat([daily("OLD"), daily("NEW", vol=200_100.0), daily("OTHER", close=7.0)], ignore_index=True)
        c = d[["date", "symbol"]]
        self.assertEqual(sorted(pi.drop_daily_aliases(c, d)["symbol"]), ["NEW", "OTHER"])
        keep = {(pd.Timestamp("2025-01-02"), "OLD")}
        self.assertEqual(sorted(pi.drop_daily_aliases(c, d, keep)["symbol"]), ["NEW", "OLD", "OTHER"])

    def test_tiny_volume_coincidence_is_not_an_alias(self):
        d = pd.concat([daily("AAA", vol=100.0), daily("BBB", vol=100.0)], ignore_index=True)
        self.assertEqual(len(pi.drop_daily_aliases(d[["date", "symbol"]], d)), 2)

    def test_clean_daily_drops_unusable_rows(self):
        d = pd.concat([daily("A"), daily("A"), daily("B", close=0.0), daily("C")], ignore_index=True)
        d.loc[3, ["open", "high", "low", "close"]] = [1.0, 2.0, 0.0, 0.0]
        self.assertEqual(pi.clean_daily(d)["symbol"].tolist(), ["A"])


class ValidateTests(unittest.TestCase):
    def test_clean_data_passes(self):
        p = PitDir({"2025-01-02": pd.concat([minute("A"), minute("B", px=20)], ignore_index=True)},
                   pd.concat([daily("A"), daily("B", close=8.0)], ignore_index=True),
                   pd.DataFrame({"date": [pd.Timestamp("2025-01-02")] * 2, "symbol": ["A", "B"]}))
        rep = p.validate()
        self.assertTrue(rep["ok"], errors_of(rep))

    def test_detects_duplicate_minute_bars(self):
        m = pd.concat([minute("A"), minute("A").iloc[5:8]], ignore_index=True)
        rep = PitDir({"2025-01-02": m}).validate()
        self.assertIn("duplicate (symbol, timestamp) in minute file", rep["errors"])
        self.assertEqual(rep["errors"]["duplicate (symbol, timestamp) in minute file"], {"2025-01-02.parquet": 3})

    def test_detects_conflicting_overlap_across_files_but_not_identical_overlap(self):
        a, b = minute("A"), minute("A", day="2025-01-03")
        ok = PitDir({"2025-01-02": pd.concat([a, minute("A", day="2025-01-03")]), "2025-01-03": b}).validate()
        self.assertTrue(ok["ok"], errors_of(ok))
        self.assertEqual(ok["counts"]["minute_overlap_bars"], 30)
        b2 = b.assign(close=b["close"] + 1, open=b["open"] + 1, high=b["high"] + 1, low=b["low"] + 1)
        rep = PitDir({"2025-01-02": pd.concat([a, minute("A", day="2025-01-03")]), "2025-01-03": b2}).validate()
        self.assertIn("same (symbol, timestamp) with different values across minute files", rep["errors"])

    def test_detects_non_monotonic_timestamps(self):
        m = minute("A")
        m = pd.concat([m.iloc[10:], m.iloc[:10]], ignore_index=True)
        self.assertIn("timestamps not monotonic per symbol", PitDir({"2025-01-02": m}).validate()["errors"])

    def test_detects_timestamps_outside_session(self):
        m = minute("A", start="02:00", n=10)
        self.assertIn("timestamps outside session range / file date window", PitDir({"2025-01-02": m}).validate()["errors"])
        m = minute("A", day="2025-01-20")
        self.assertIn("timestamps outside session range / file date window", PitDir({"2025-01-02": m}).validate()["errors"])

    def test_detects_bad_prices(self):
        for col, val, key in (("close", np.nan, "nan_price"), ("open", 0.0, "nonpositive_price"), ("low", -1.0, "nonpositive_price"),
                              ("volume", -5.0, "negative_volume")):
            m = minute("A")
            m.loc[3, col] = val
            self.assertIn(f"bad minute bars: {key}", PitDir({"2025-01-02": m}).validate()["errors"], (col, val))
        m = minute("A")
        m.loc[4, ["high", "low"]] = [9.0, 11.0]
        self.assertIn("bad minute bars: high_below_low", PitDir({"2025-01-02": m}).validate()["errors"])

    def test_detects_daily_and_candidate_duplicates_and_aliases(self):
        d = pd.concat([daily("A"), daily("A"), daily("OLD", close=9.0), daily("NEW", close=9.0)], ignore_index=True)
        c = pd.DataFrame({"date": [pd.Timestamp("2025-01-02")] * 3, "symbol": ["OLD", "NEW", "NEW"]})
        errs = PitDir({"2025-01-02": minute("A")}, d, c).validate()["errors"]
        self.assertEqual(errs["duplicate (symbol, date)"], {"daily_pit.parquet": 1, "candidates.parquet": 1})
        self.assertEqual(errs["alias securities among candidates"], {"candidates.parquet": 2})

    def test_detects_bad_daily_prices(self):
        d = daily("A", close=0.0)
        self.assertIn("bad daily bars: nonpositive_price", PitDir({"2025-01-02": minute("A")}, d).validate()["errors"])


class WindowTests(unittest.TestCase):
    def _trades(self):
        return pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "entry_time": ["2025-01-02 15:55:00-05:00"] * 3,
                             "prev_close": [2.0, 2.0, 3.0], "price": [2.5, 2.5, 3.5]})

    def _window(self, tid, sym, dup=False, shift=0):
        t = pd.date_range("2025-01-02 15:55", periods=6, freq="min", tz=NY).append(
            pd.date_range("2025-01-03 09:30", periods=5, freq="min", tz=NY))
        times = t.asi8 + shift
        if dup:
            times = np.insert(times, 3, times[3])
        n = len(times)
        z = np.ones(n)
        return TradeWindow(tid, times, z, z, z, z, z, z, 1.0, 1.0, 1.0, sym)

    def _run(self, windows):
        errors, warnings, counts = {}, {}, {}
        add = lambda store, check, where, n: n and store.setdefault(check, {}).__setitem__(where, int(n))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "w.npz"
            save_windows(windows, path)
            pi._check_windows(self._trades(), path, errors, warnings, counts, add)
        return errors, warnings

    def test_clean_windows_pass_and_twin_trade_is_flagged(self):
        errors, warnings = self._run([self._window(0, "AAA"), self._window(2, "CCC")])
        self.assertEqual((errors, warnings), ({}, {}))  # BBB is a twin of AAA: no window expected
        _, warnings = self._run([self._window(0, "AAA")])
        self.assertIn("trades without a window (no next-session minute bars)", warnings)
        errors, _ = self._run([self._window(0, "AAA"), self._window(1, "BBB"), self._window(2, "CCC")])
        self.assertEqual(errors["windows for duplicate-security trades"], {"windows": 1})

    def test_detects_duplicate_timestamps_wrong_entry_and_duplicate_window(self):
        errors, _ = self._run([self._window(0, "AAA", dup=True), self._window(2, "CCC")])
        self.assertIn("windows with duplicate timestamps", errors)
        errors, _ = self._run([self._window(0, "AAA", shift=60 * 10**9), self._window(2, "CCC")])
        self.assertIn("windows whose first bar is not the entry bar", errors)
        errors, _ = self._run([self._window(0, "AAA"), self._window(0, "AAA"), self._window(2, "CCC")])
        self.assertIn("trades with more than one window", errors)
        errors, _ = self._run([self._window(0, "ZZZ"), self._window(2, "CCC")])
        self.assertIn("windows whose symbol differs from the trade", errors)

    def test_build_windows_drops_duplicate_bars_and_twin_trades_and_warns(self):
        m1 = pd.concat([minute("AAA", start="15:30", n=30), minute("BBB", start="15:30", n=30)], ignore_index=True)
        nxt = pd.concat([minute("AAA", day="2025-01-03", start="09:30", n=40), minute("BBB", day="2025-01-03", start="09:30", n=40)])
        m2 = pd.concat([m1, nxt, nxt.iloc[:10]], ignore_index=True)  # day file holds repeated bars
        t = self._trades().iloc[:2].assign(entry=2.5, fill_entry=1.0, period=["dev", "test"])
        log = []
        with tempfile.TemporaryDirectory() as d:
            m2.to_parquet(Path(d) / "2025-01-02.parquet", index=False)
            nxt.to_parquet(Path(d) / "2025-01-03.parquet", index=False)
            ws = build_windows(t, d, Path(d) / "w.npz", log.append)
            idx = pd.read_parquet(Path(d) / "windows_index.parquet")
        self.assertEqual([w.symbol for w in ws], ["AAA"])
        self.assertTrue(all(np.all(np.diff(w.times) > 0) for w in ws))
        self.assertEqual(idx["trade_id"].tolist(), [0])
        self.assertEqual(idx["period"].tolist(), ["dev"])
        self.assertTrue(any("duplicate-security" in x for x in log) and any("duplicate bars" in x for x in log))


if __name__ == "__main__":
    unittest.main()
