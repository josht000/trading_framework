import datetime as dt
import unittest

import numpy as np
import pandas as pd

from src.backtest.pit_metrics import dedupe_trades, daily_returns, summarize
from src.backtest.pit_runner import _fill

RZ = {"participation": 0.10, "slip_rth": 0.001, "slip_ext": 0.004}
RTH, EXT = dt.time(10, 0), dt.time(17, 0)


class FillTests(unittest.TestCase):
    def test_participation_limits_each_bar_and_spreads_over_bars(self):
        avg, got, rem, last = _fill([RTH] * 3, np.array([10.0, 11.0, 12.0]), np.array([100.0, 100.0, 100.0]), 25, RZ, "buy")
        self.assertEqual((got, rem, last), (25, 0, 2))
        self.assertAlmostEqual(avg, (10 * 10.01 + 10 * 11.011 + 5 * 12.012) / 25)

    def test_unfilled_remainder_reported_and_sell_slips_down(self):
        avg, got, rem, _ = _fill([EXIT := EXT], np.array([10.0]), np.array([50.0]), 100, RZ, "sell")
        self.assertEqual((got, rem), (5, 95))
        self.assertAlmostEqual(avg, 10 * (1 - 0.004))


class MetricsTests(unittest.TestCase):
    def _trades(self):
        return pd.DataFrame({
            "entry_time": ["2025-01-02 15:55", "2025-01-02 15:55", "2025-01-02 15:55", "2025-01-03 15:55"],
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "prev_close": [5.0, 5.0, 6.0, 7.0], "price": [6.0, 6.0, 7.0, 8.0],
            "ret": [0.10, 0.10, -0.20, 0.05], "rel_vol": [3.0, 3.0, 2.0, 4.0], "atr_pct": [0.05] * 4,
        })

    def test_ticker_change_duplicates_are_dropped(self):
        self.assertEqual(list(dedupe_trades(self._trades())["symbol"]), ["AAA", "CCC", "DDD"])

    def test_full_vs_slots_sizing(self):
        t = self._trades()
        full = daily_returns(t, max_pos=5, sizing="full")
        slots = daily_returns(t, max_pos=5, sizing="slots")
        self.assertAlmostEqual(full.iloc[0], (0.10 - 0.20) / 2)
        self.assertAlmostEqual(slots.iloc[0], (0.10 - 0.20) / 5)

    def test_summarize_reports_drawdown(self):
        s = summarize(self._trades(), max_pos=5)
        self.assertLess(s["maxDD%"], 0)
        self.assertEqual(s["trades"], 3)


if __name__ == "__main__":
    unittest.main()
