import unittest

from src.screening.pivot_resistance import (
    PivotResistanceConfig, PivotResistanceDetector, detect_resistance, merge_top, swing_highs,
)

HIGHS = [10, 11, 12, 15, 12, 11, 10, 11, 13, 14.05, 13, 12, 11, 12, 14, 12, 11, 10, 9, 9.5]


class PivotResistanceTests(unittest.TestCase):
    def test_swing_highs_find_local_maxima(self):
        got = [v for _, v in swing_highs(HIGHS, pivot_bars=2, edge_pivots=False)]
        self.assertEqual(got, [15, 14.05, 14])

    def test_merge_keeps_top_within_tolerance(self):
        self.assertEqual(merge_top([5.0, 5.08, 5.5, 5.55, 6.0], 0.10), [5.08, 5.55, 6.0])

    def test_levels_are_above_last_close_and_ascending(self):
        levels = detect_resistance(HIGHS, last_close=13.5, pivot_bars=2, edge_pivots=False)
        self.assertEqual(levels, [14.05, 15])
        self.assertTrue(all(v > 13.5 for v in levels))

    def test_blue_sky_returns_empty(self):
        self.assertEqual(detect_resistance(HIGHS, last_close=20.0), [])

    def test_edge_pivots_toggle(self):
        highs = [10, 9, 8, 9, 10, 11, 12]
        self.assertIn(12, detect_resistance(highs, 5.0, pivot_bars=3, edge_pivots=True))
        self.assertNotIn(12, detect_resistance(highs, 5.0, pivot_bars=3, edge_pivots=False))

    def test_max_levels_and_bars_input(self):
        det = PivotResistanceDetector(PivotResistanceConfig(pivot_bars=2, edge_pivots=False, max_levels=1))
        bars = {"high": HIGHS, "close": [9.0] * len(HIGHS)}
        self.assertEqual(det.detect_bars(bars), [14.05])


if __name__ == "__main__":
    unittest.main()
