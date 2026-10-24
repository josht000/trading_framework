import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import exit_oracle as O
from src.backtest.exit_sim import fixed_time_policy, load_windows, simulate_trade, _info, _clock
from src.data.pit_integrity import WINDOWS, oracle_staleness
from tests.test_exit_sim import make_window


class OracleTests(unittest.TestCase):
    def test_known_peak_and_offset(self):
        w = make_window({0: 10.2, 5: 11.0, 12: 10.5}, pm=[10.1, 12.0, 10.0])
        r = O.oracle_row(w, n_perm=20)
        self.assertAlmostEqual(r["hindsight_peak_close_ret_open"], 0.10)
        self.assertEqual(r["hindsight_peak_minute_open"], 5)
        self.assertEqual(r["hindsight_peak_clock_open"], "09:35")
        self.assertAlmostEqual(r["hindsight_peak_close_ret_pm"], 0.20)
        self.assertEqual(r["hindsight_peak_minute_pm"], -89)
        self.assertEqual(r["hindsight_peak_clock_pm"], "08:01")
        self.assertAlmostEqual(r["hindsight_mfe_vs_entry"], 0.10)
        self.assertAlmostEqual(r["hindsight_mae_vs_entry"], 0.02)
        self.assertEqual(r["hindsight_oracle_exec_bar_open"], "09:35")
        self.assertLess(r["hindsight_oracle_exec_ret_open"], r["hindsight_peak_close_ret_open"])

    def _check_equal(self, w):
        _, minutes = _clock(w)
        inf = _info(w)
        sh, hz = O._shares(w), inf["horizon"]
        n = 0
        for t in inf["next_idx"]:
            if 570 <= minutes[t] <= 615:
                lab = f"{minutes[t] // 60:02d}:{minutes[t] % 60:02d}"
                a = simulate_trade(w, {"name": "fixed_time", "keep_ah": False, "deadline": lab})["ret"]
                b = O.exec_ret(w, t, sh, hz)
                self.assertTrue((np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9)
                n += 1
        for t in inf["next_idx"]:
            if 480 <= minutes[t] < 570:
                lab = f"{minutes[t] // 60:02d}:{minutes[t] % 60:02d}"
                a = simulate_trade(w, {"name": "smart", "keep_ah": False, "pm_start": lab, "pm_target": -10, "deadline": "10:15"})["ret"]
                self.assertAlmostEqual(a, O.exec_ret(w, t, sh, hz), places=9)
        return n

    def test_exec_matches_simulate_synthetic(self):
        w = make_window({0: 10.2, 5: 11.0, 20: 9.0}, vol=500, pm=[10.1, 12.0, 10.0])
        self.assertGreater(self._check_equal(w), 40)

    def test_exec_matches_simulate_real_windows(self):
        if not WINDOWS.exists():
            self.skipTest("no windows cache")
        ws = load_windows(WINDOWS)[::40][:60]
        self.assertGreaterEqual(len(ws), 50)
        for w in ws:
            if _info(w)["first_next"] is not None:
                self._check_equal(w)

    def test_staleness_detects_modified_cache(self):
        if not WINDOWS.exists():
            self.skipTest("no windows cache")
        w = load_windows(WINDOWS)[:3]
        idx = pd.DataFrame({"trade_id": [x.trade_id for x in w], "symbol": [x.symbol for x in w], "entry_time": "", "period": "dev"})
        d = Path(tempfile.mkdtemp())
        try:
            from src.backtest.exit_sim import save_windows
            save_windows(w, d / "windows.npz")
            idx.to_parquet(d / "windows_index.parquet")
            orc = O.build_oracle(w, idx, n_perm=5)
            orc.to_parquet(d / "oracle.parquet")
            (d / "oracle_meta.json").write_text(json.dumps(O.make_meta(orc, d / "windows.npz", d / "windows_index.parquet", d / "oracle.parquet")))
            args = dict(oracle=d / "oracle.parquet", meta=d / "oracle_meta.json", windows=d / "windows.npz", index=d / "windows_index.parquet")
            self.assertEqual(oracle_staleness(**args), [])
            save_windows(w[:2], d / "windows.npz")
            bad = oracle_staleness(**args)
            self.assertTrue(any("windows.npz changed" in b for b in bad) and any("trade_id set" in b for b in bad))
            self.assertTrue(all(c.startswith("hindsight_") for c in orc.columns if c not in ("trade_id", "symbol", "entry_time", "period")))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_prefix_on_real_oracle(self):
        p = WINDOWS.with_name("oracle.parquet")
        if not p.exists():
            self.skipTest("no oracle")
        cols = pd.read_parquet(p).columns
        self.assertTrue(all(c.startswith("hindsight_") for c in cols if c not in ("trade_id", "symbol", "entry_time", "period")))


if __name__ == "__main__":
    unittest.main()
