import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.research.config_io import dump_config, load_config
from src.research.labels import export_levels, load_labels, match_levels, save_labels, score_labels
from src.research.pick_charts import PickConfig, eligible_dates, pick_dates
from src.research.resistance_eval import (
    EvalConfig, run_evaluation, summarize_breakouts, summarize_reactions, wilson,
)
from src.screening.resistance import ResistanceConfig, ResistanceDetector
from src.screening.volume_profile import VolumeProfileConfig, VolumeProfileDetector, _triangular_cdf

PEAK = 12.0
TROUGH = 10.0


def make_cycles(cycles=3, up=8, down=8, tail=0, start="2023-01-02", warmup=20):
    path = [TROUGH] * (warmup + 1)
    for _ in range(cycles):
        path += list(np.linspace(TROUGH, PEAK, up + 1)[1:])
        path += list(np.linspace(PEAK, TROUGH, down + 1)[1:])
    path += [TROUGH] * tail
    close = np.array(path)
    idx = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close,
         "volume": 1_000_000.0},
        index=idx,
    )


def peak_positions(df):
    return [i for i in range(1, len(df) - 1)
            if df["close"].iloc[i] == PEAK]


class ResistanceDetectorTests(unittest.TestCase):
    def setUp(self):
        self.det = ResistanceDetector(ResistanceConfig(include_swing_lows=False, include_ranges=False))
        self.df = make_cycles(cycles=3, tail=6)
        self.p = self.det.prepare(self.df)
        self.peaks = peak_positions(self.df)

    def test_repeated_peaks_form_one_resistance_zone_above_price(self):
        k = len(self.df) - 1
        lv = self.det.levels(self.p, k)
        res = lv[lv["role"] == "resistance"]
        self.assertEqual(len(res), 1)
        zone = res.iloc[0]
        self.assertEqual(zone["n_pivots"], 3)
        self.assertGreaterEqual(zone["touches"], 3)
        self.assertAlmostEqual(zone["center"], PEAK + 0.05, delta=0.15)
        self.assertGreater(zone["lower"], self.p.close[k])

    def test_zone_collapses_to_highest_price_by_default(self):
        lv = self.det.levels(self.p, len(self.df) - 1)
        res = lv[lv["role"] == "resistance"].iloc[0]
        self.assertEqual(res["lower"], res["upper"])
        self.assertAlmostEqual(res["upper"], PEAK + 0.1, places=9)

    def test_band_mode_keeps_zone_width(self):
        det = ResistanceDetector(ResistanceConfig(include_swing_lows=False, include_ranges=False,
                                                  collapse_to_high=False))
        p = det.prepare(self.df)
        lv = det.levels(p, len(self.df) - 1)
        res = lv[lv["role"] == "resistance"].iloc[0]
        self.assertGreater(res["upper"], res["lower"])
        self.assertAlmostEqual(res["upper"], PEAK + 0.1, places=9)

    def test_pivot_usable_only_after_confirmation_lag(self):
        right = self.det.config.pivot_right
        last_peak = self.peaks[-1]
        before = self.det.levels(self.p, last_peak + right - 1)
        after = self.det.levels(self.p, last_peak + right)
        self.assertEqual(before[before["role"] != "support"]["n_pivots"].max(), 2)
        self.assertEqual(after[after["role"] != "support"]["n_pivots"].max(), 3)

    def test_levels_are_causal(self):
        k = self.peaks[1] + 10
        base = self.det.levels(self.p, k)

        tampered = self.df.copy()
        tampered.iloc[k + 1:, tampered.columns.get_indexer(["high", "low", "close", "open"])] = 99.0
        tampered_p = self.det.prepare(tampered)
        changed = self.det.levels(tampered_p, k)

        cols = ["lower", "upper", "role", "n_pivots", "touches", "score"]
        pd.testing.assert_frame_equal(base[cols], changed[cols])

    def test_role_follows_price_and_flips_to_support(self):
        k = len(self.df) - 1
        lv = self.det.levels(self.p, k)
        z = lv.iloc[0]
        above = self.det.levels(self.p, k, price=z["upper"] + 1.0)
        inside = self.det.levels(self.p, k, price=z["center"])
        below = self.det.levels(self.p, k, price=z["lower"] - 1.0)
        self.assertEqual(above.iloc[0]["role"], "support")
        self.assertEqual(inside.iloc[0]["role"], "in_zone")
        self.assertEqual(below.iloc[0]["role"], "resistance")

    def test_ladder_sorted_nearest_first_and_excludes_support(self):
        ladder = self.det.ladder(self.p, len(self.df) - 1, price=TROUGH)
        self.assertTrue((ladder["role"] != "support").all())
        self.assertTrue(ladder["lower"].is_monotonic_increasing)

    def test_blue_sky_has_empty_ladder(self):
        ladder = self.det.ladder(self.p, len(self.df) - 1, price=PEAK + 5)
        self.assertTrue(ladder.empty)

    def test_known_through_helpers(self):
        session = self.df.index[20]
        self.assertEqual(self.det.known_through_before(self.p, session), 19)
        self.assertEqual(self.det.known_through_on(self.p, session), 20)

    def test_moving_average_levels_optional(self):
        det = ResistanceDetector(ResistanceConfig(include_moving_averages=True, ma_periods=(5,)))
        p = det.prepare(self.df)
        lv = det.levels(p, len(self.df) - 1)
        self.assertIn("sma5", set(lv["kind"]))
        self.assertNotIn("sma5", set(self.det.levels(self.p, len(self.df) - 1)["kind"]))


def make_range_series():
    warm = [10.0] * 21
    ramp = list(np.linspace(10.0, 12.0, 11)[1:])
    rng = [12.0, 12.5] * 6
    close = np.array(warm + ramp + rng)
    idx = pd.bdate_range("2023-01-02", periods=len(close))
    return pd.DataFrame(
        {"open": close, "high": close + 0.05, "low": close - 0.05, "close": close,
         "volume": 1_000_000.0},
        index=idx,
    )


class ConsolidationRangeTests(unittest.TestCase):
    def setUp(self):
        self.df = make_range_series()
        self.k = len(self.df) - 1

    def _levels(self, include_ranges):
        det = ResistanceDetector(ResistanceConfig(
            include_swing_lows=False, include_ranges=include_ranges, min_prominence_atr=5.0))
        return det.levels(det.prepare(self.df), self.k)

    def test_range_high_and_low_both_become_levels(self):
        lv = self._levels(True)
        high_hit = lv[(lv["upper"] - 12.55).abs() < 0.03]
        low_hit = lv[(lv["upper"] - 11.95).abs() < 0.03]
        self.assertFalse(high_hit.empty)
        self.assertFalse(low_hit.empty)
        self.assertIn("range", high_hit.iloc[0]["kind"])

    def test_no_range_levels_when_disabled(self):
        lv = self._levels(False)
        self.assertTrue(lv[(lv["upper"] - 12.55).abs() < 0.03].empty)

    def test_range_not_known_before_window_completes(self):
        det = ResistanceDetector(ResistanceConfig(
            include_swing_lows=False, min_prominence_atr=2.0, range_min_bars=5))
        p = det.prepare(self.df)
        first_range_bar = 21 + 10
        early = det.levels(p, first_range_bar + 1)
        self.assertTrue(early[(early["upper"] - 12.55).abs() < 0.03].empty)

    def test_trending_move_is_not_a_range(self):
        close = np.linspace(10, 20, 80)
        df = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1,
                           "close": close, "volume": 1e6},
                          index=pd.bdate_range("2023-01-02", periods=80))
        det = ResistanceDetector(ResistanceConfig(include_swing_lows=False, min_prominence_atr=50.0))
        lv = det.levels(det.prepare(df), 79)
        self.assertTrue(lv.empty)


class LabelAndConfigTests(unittest.TestCase):
    def setUp(self):
        self.df = make_cycles(cycles=3, tail=6)
        self.asof = self.df.index[-1]
        self.det = ResistanceDetector(ResistanceConfig(include_swing_lows=False, include_ranges=False))

    def _labels(self, rows):
        return pd.DataFrame(
            [{"symbol": "SYN", "asof": self.asof, "level": lvl, "status": st, "source": "manual",
              "score": np.nan, "kind": "", "note": ""} for lvl, st in rows])

    def test_match_levels_is_one_to_one_nearest(self):
        pairs, gold_left, det_left = match_levels(np.array([10.0, 10.2]), np.array([10.1]), tol=0.2)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(gold_left), 1)
        self.assertEqual(det_left, [])

    def test_score_counts_hits_and_misses(self):
        labels = self._labels([(PEAK + 0.1, "keep"), (11.0, "add")])
        res = score_labels(labels, {"SYN": self.df}, self.det)
        o = res["overall"]
        self.assertEqual((o["tp"], o["fn"]), (1, 1))
        self.assertAlmostEqual(o["recall"], 0.5)
        self.assertAlmostEqual(o["precision_all"], 1.0)
        self.assertEqual(len(res["missed"]), 1)

    def test_rejected_level_still_detected_is_false_positive(self):
        labels = self._labels([(PEAK + 0.1, "reject"), (11.0, "keep")])
        o = score_labels(labels, {"SYN": self.df}, self.det)["overall"]
        self.assertEqual(o["fp_rejected"], 1)
        self.assertEqual(o["tp"], 0)

    def test_auto_rows_are_not_scored(self):
        res = score_labels(self._labels([(PEAK + 0.1, "auto")]), {"SYN": self.df}, self.det)
        self.assertTrue(res["per_asof"].empty)

    def test_export_then_reexport_adds_only_new_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "labels.csv")
            first = export_levels(self.det, self.df, "SYN", self.asof, path)
            again = export_levels(self.det, self.df, "SYN", self.asof, path)
            self.assertGreater(first, 0)
            self.assertEqual(again, 0)
            df = load_labels(path)
            self.assertTrue((df["status"] == "auto").all())

    def test_edited_labels_round_trip_and_invalid_status_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "labels.csv")
            export_levels(self.det, self.df, "SYN", self.asof, path)
            df = load_labels(path)
            df.loc[df.index[0], "status"] = "keep"
            save_labels(df, path)
            self.assertIn("keep", set(load_labels(path)["status"]))
            raw = pd.read_csv(path)
            raw.loc[0, "status"] = "maybe"
            raw.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_labels(path)

    def test_config_json_round_trip_and_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = dump_config(os.path.join(tmp, "cfg.json"), ResistanceConfig(pivot_right=4))
            cfg = load_config(path, merge_atr=0.8, pivot_left=None)
            self.assertEqual(cfg.pivot_right, 4)
            self.assertEqual(cfg.merge_atr, 0.8)
            self.assertEqual(cfg.pivot_left, 5)
            self.assertIsInstance(cfg.ma_periods, tuple)

    def test_config_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.json")
            with open(path, "w") as f:
                f.write('{"not_a_setting": 1}')
            with self.assertRaises(ValueError):
                load_config(path)


def make_two_node_series():
    n1, n2 = 150, 150
    i = np.arange(n1 + n2)
    close = np.where(i < n1, 12.0, 10.0) + 0.15 * np.sin(i / 3.0)
    volume = np.where(i < n1, 5e6, 2e6).astype(float)
    return pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": volume},
        index=pd.bdate_range("2023-01-02", periods=n1 + n2))


class VolumeProfileDetectorTests(unittest.TestCase):
    def setUp(self):
        self.df = make_two_node_series()
        self.det = VolumeProfileDetector(VolumeProfileConfig())
        self.p = self.det.prepare(self.df)
        self.k = len(self.df) - 1

    def test_heavy_volume_price_becomes_resistance_above(self):
        lv = self.det.levels(self.p, self.k)
        res = lv[lv["role"] == "resistance"]
        self.assertTrue(((res["upper"] - 12.0).abs() < 0.3).any())
        self.assertTrue((lv["lower"] == lv["upper"]).all())
        self.assertTrue((lv["kind"] == "hvn").all())

    def test_node_below_price_is_support_and_roles_flip_with_price(self):
        lv = self.det.levels(self.p, self.k, price=13.0)
        near12 = lv[(lv["upper"] - 12.0).abs() < 0.3]
        self.assertFalse(near12.empty)
        self.assertEqual(near12.iloc[0]["role"], "support")

    def test_levels_are_causal(self):
        k = 220
        base = self.det.levels(self.p, k)
        tampered = self.df.copy()
        tampered.iloc[k + 1:, tampered.columns.get_indexer(["high", "low", "close", "open", "volume"])] = 99.0
        changed = self.det.levels(self.det.prepare(tampered), k)
        cols = ["upper", "role", "score", "touches"]
        pd.testing.assert_frame_equal(base[cols], changed[cols])

    def test_caps_on_levels_each_side(self):
        det = VolumeProfileDetector(VolumeProfileConfig(max_above=1, max_below=1))
        lv = det.levels(det.prepare(self.df), self.k)
        self.assertLessEqual((lv["role"] == "resistance").sum(), 1)
        self.assertLessEqual((lv["role"] == "support").sum(), 1)

    def test_triangular_cdf_is_a_distribution(self):
        x = np.linspace(9, 13, 401)[None, :]
        cdf = _triangular_cdf(x, np.array([[10.0]]), np.array([[12.0]]), np.array([[11.0]]))
        self.assertAlmostEqual(float(cdf[0, 0]), 0.0)
        self.assertAlmostEqual(float(cdf[0, -1]), 1.0)
        self.assertTrue(np.all(np.diff(cdf[0]) >= -1e-12))

    def test_works_with_evaluation_harness(self):
        res = run_evaluation({"SYN": make_cycles(cycles=15)}, eval_config=EvalConfig(min_history=60),
                             detector=VolumeProfileDetector(VolumeProfileConfig()),
                             sources=["atr_pivot", "shifted"])
        self.assertFalse(res["reactions"].empty)


def make_decline_then_recovery(n_up=100, n_down=100, n_rec=100):
    close = np.concatenate([np.linspace(10, 20, n_up), np.linspace(20, 10, n_down), np.linspace(10, 13, n_rec)])
    return pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1_000_000.0},
        index=pd.bdate_range("2022-01-03", periods=len(close)))


class ChartPickerTests(unittest.TestCase):
    def test_recovery_below_prior_high_is_eligible_and_rules_hold(self):
        df = make_decline_then_recovery()
        el = eligible_dates(df, PickConfig())
        self.assertFalse(el.empty)
        self.assertTrue(((el["ratio_to_252d_high"] >= 0.55) & (el["ratio_to_252d_high"] <= 0.90)).all())
        self.assertTrue((el["ret20"] >= 0.05).all())
        self.assertTrue((el["pos"] <= len(df) - 1 - PickConfig().forward_buffer).all())

    def test_blue_sky_series_is_never_eligible(self):
        close = np.linspace(5, 19, 320)
        df = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1, "close": close,
                           "volume": 1e6}, index=pd.bdate_range("2022-01-03", periods=320))
        self.assertTrue(eligible_dates(df, PickConfig()).empty)

    def test_pick_is_reproducible_and_respects_spacing(self):
        daily = {"AAA": make_decline_then_recovery(), "BBB": make_decline_then_recovery()}
        a = pick_dates(daily, "2022-01-03", "2030-01-01", PickConfig(per_symbol=2, min_spacing=5, seed=3))
        b = pick_dates(daily, "2022-01-03", "2030-01-01", PickConfig(per_symbol=2, min_spacing=5, seed=3))
        pd.testing.assert_frame_equal(a, b)
        for _, g in a.groupby("symbol"):
            days = sorted(g["asof"])
            if len(days) > 1:
                self.assertGreaterEqual(len(pd.bdate_range(days[0], days[1])) - 1, 5)


class ManualLabelTests(unittest.TestCase):
    def test_three_column_file_loads_with_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "manual.csv")
            with open(path, "w") as f:
                f.write("symbol,asof,level\nsofi,2023-06-01,8.52\nSOFI,2023-06-01,7.70\n")
            df = load_labels(path)
            self.assertEqual(set(df["status"]), {"add"})
            self.assertEqual(set(df["source"]), {"manual"})
            self.assertEqual(set(df["symbol"]), {"SOFI"})

    def test_blank_status_cells_default_to_add_and_header_only_file_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "manual.csv")
            with open(path, "w") as f:
                f.write("symbol,asof,id,level,status,source,score,kind,note\nSOFI,2023-06-01,,8.52,,,,,\n")
            self.assertEqual(load_labels(path).iloc[0]["status"], "add")
            with open(path, "w") as f:
                f.write("symbol,asof,id,level,status,source,score,kind,note\n")
            self.assertTrue(load_labels(path).empty)

    def test_blank_level_rows_are_placeholders_and_typos_are_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "manual.csv")
            with open(path, "w") as f:
                f.write("symbol,asof,level,note\nBB,2024-10-21,,chart 01\nBB,2024-10-21,3.10,chart 01\n")
            df = load_labels(path)
            self.assertEqual(len(df), 1)
            self.assertAlmostEqual(df.iloc[0]["level"], 3.10)
            with open(path, "w") as f:
                f.write('symbol,asof,level,note\nBB,2024-10-21,"3,10",chart 01\n')
            with self.assertRaises(ValueError):
                load_labels(path)

    def test_manual_labels_score_detector_recall(self):
        df = make_cycles(cycles=3, tail=6)
        det = ResistanceDetector(ResistanceConfig(include_swing_lows=False, include_ranges=False))
        labels = pd.DataFrame([{"symbol": "SYN", "asof": df.index[-1], "level": PEAK + 0.1, "status": "add",
                                "source": "manual", "score": np.nan, "kind": "", "note": "", "id": np.nan}])
        o = score_labels(labels, {"SYN": df}, det)["overall"]
        self.assertEqual(o["recall"], 1.0)

    def _y_axes(self, fig):
        ys = [fig.layout[k] for k in fig.layout if k.startswith("yaxis")]
        return fig.layout.yaxis, [y for y in ys if y.side == "right"]

    def test_chart_has_quarter_dollar_ticks_on_both_sides_for_low_priced_stock(self):
        from src.research.chart_levels import plot_levels
        df = make_cycles(cycles=3, tail=6)
        fig = plot_levels(df, df.index[-1], symbol="SYN", forward=0, blank=True, lookback=60)
        primary, right = self._y_axes(fig)
        self.assertEqual(primary.dtick, 0.25)
        self.assertEqual(len(right), 1)
        self.assertEqual(tuple(right[0].range), tuple(primary.range))
        self.assertEqual(right[0].dtick, 0.25)
        self.assertEqual(right[0].matches, "y")

    def test_tick_step_widens_when_quarter_dollar_would_be_unreadable(self):
        from src.research.chart_levels import MAX_TICKS, plot_levels
        close = np.linspace(100, 300, 200)
        df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1e6},
                          index=pd.bdate_range("2022-01-03", periods=200))
        fig = plot_levels(df, df.index[-1], symbol="BIG", forward=0, blank=True, lookback=190)
        primary, _ = self._y_axes(fig)
        lo, hi = primary.range
        self.assertGreater(primary.dtick, 0.25)
        self.assertLessEqual((hi - lo) / primary.dtick, MAX_TICKS)

    def test_level_tags_sit_in_padding_right_of_last_bar(self):
        from src.research.chart_levels import plot_levels
        df = make_cycles(cycles=3, tail=6)
        labels = pd.DataFrame([{"symbol": "SYN", "asof": df.index[-1], "level": 11.0, "status": "add",
                                "source": "manual", "score": np.nan, "kind": "", "note": "", "id": np.nan}])
        fig = plot_levels(df, df.index[-1], symbol="SYN", forward=0, blank=True, labels=labels, lookback=60)
        line = [t for t in fig.data if t.type == "scatter" and t.mode == "lines+text"][0]
        self.assertGreater(pd.Timestamp(line.x[-1]), df.index[-1])
        self.assertGreater(pd.Timestamp(fig.layout.xaxis.range[1]), pd.Timestamp(line.x[-1]))

    def test_blank_chart_hides_detector_and_future(self):
        from src.research.chart_levels import plot_levels
        df = make_cycles(cycles=3, tail=6)
        fig = plot_levels(df, df.index[-10], symbol="SYN", forward=0, blank=True, lookback=40)
        title = fig.layout.title.text
        self.assertNotIn("resistance above", title)
        self.assertNotIn("live", " ".join(str(t.name) for t in fig.data))
        last_candle_x = max(max(t.x) for t in fig.data if t.type == "candlestick")
        self.assertLessEqual(pd.Timestamp(last_candle_x), df.index[-10])


class PickerExclusionTests(unittest.TestCase):
    def test_excluded_dates_and_neighbors_are_not_picked(self):
        daily = {"AAA": make_decline_then_recovery()}
        base = pick_dates(daily, "2022-01-03", "2030-01-01", PickConfig(per_symbol=1, seed=5))
        used = pd.DataFrame({"symbol": ["AAA"], "asof": [base.iloc[0]["asof"]]})
        again = pick_dates(daily, "2022-01-03", "2030-01-01", PickConfig(per_symbol=1, seed=5, min_spacing=10),
                           exclude=used)
        if not again.empty:
            lo, hi = sorted([base.iloc[0]["asof"], again.iloc[0]["asof"]])
            self.assertGreaterEqual(len(pd.bdate_range(lo, hi)) - 1, 10)
            self.assertNotEqual(base.iloc[0]["asof"], again.iloc[0]["asof"])

    def test_max_total_caps_the_set_reproducibly(self):
        daily = {s: make_decline_then_recovery() for s in "ABCDEF"}
        a = pick_dates(daily, "2022-01-03", "2030-01-01", PickConfig(max_total=3, seed=2))
        b = pick_dates(daily, "2022-01-03", "2030-01-01", PickConfig(max_total=3, seed=2))
        self.assertEqual(len(a), 3)
        pd.testing.assert_frame_equal(a, b)


class LabelIdTests(unittest.TestCase):
    def test_ids_number_levels_from_highest_price_and_continue_on_reexport(self):
        df = make_cycles(cycles=3, tail=6)
        det = ResistanceDetector(ResistanceConfig(include_swing_lows=True, include_ranges=False))
        asof = df.index[-1]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "labels.csv")
            n = export_levels(det, df, "SYN", asof, path, source="atr_pivot")
            self.assertGreater(n, 0)
            first = load_labels(path)
            self.assertEqual(list(first["id"].astype(int)), list(range(1, n + 1)))
            self.assertTrue(first["level"].is_monotonic_decreasing)
            self.assertEqual(set(first["source"]), {"atr_pivot"})

            det2 = ResistanceDetector(ResistanceConfig(include_swing_lows=True, include_ranges=True))
            added = export_levels(det2, df, "SYN", asof, path, source="atr_pivot")
            if added:
                again = load_labels(path)
                self.assertEqual(again["id"].nunique(), len(again))


class EvaluationHarnessTests(unittest.TestCase):
    def setUp(self):
        self.daily = {"SYN": make_cycles(cycles=40, tail=0)}
        self.cfg = EvalConfig(min_history=60)
        self.results = run_evaluation(
            self.daily, ResistanceConfig(include_swing_lows=False), self.cfg)

    def test_frames_have_expected_columns(self):
        r, b = self.results["reactions"], self.results["breakouts"]
        for col in ("source", "outcome", "distance_atr", "score", "date"):
            self.assertIn(col, r.columns)
        if not b.empty:
            for col in ("source", "gap_ret", "ret5", "room_atr", "false_break", "vol_ok"):
                self.assertIn(col, b.columns)
        self.assertEqual(set(r["source"]),
                         {"atr_pivot", "shifted", "rolling20", "rolling10", "round", "random"})

    def test_shifted_placebo_preserves_zone_widths(self):
        r = self.results["reactions"]
        a = r[r["source"] == "atr_pivot"]["width_atr"].mean()
        s = r[r["source"] == "shifted"]["width_atr"].mean()
        self.assertAlmostEqual(a, s, delta=0.35)

    def test_detector_rejects_more_than_random_on_cyclic_series(self):
        s = summarize_reactions(self.results["reactions"]).set_index("source")
        self.assertGreater(s.loc["atr_pivot", "touched"], 5)
        self.assertGreater(s.loc["atr_pivot", "reject_rate"], s.loc["random", "reject_rate"])

    def test_outcomes_use_only_defined_labels(self):
        allowed = {"no_touch", "reject", "break", "neutral"}
        self.assertTrue(set(self.results["reactions"]["outcome"]) <= allowed)

    def test_breakout_summary_includes_unconditional_rows(self):
        s = summarize_breakouts(self.results["breakouts"], self.results["uncond"])
        self.assertIn("ALL_BARS", set(s["source"]))

    def test_wilson_interval_brackets_proportion(self):
        lo, hi = wilson(30, 100)
        self.assertLess(lo, 0.3)
        self.assertGreater(hi, 0.3)
        self.assertTrue(np.isnan(wilson(0, 0)[0]))


if __name__ == "__main__":
    unittest.main()
