"""Synthetic-window tests for src/backtest/exit_sim.py (fast, no data files except the optional
legacy-equivalence test).  Bar label = bar open time; decision at bar close; fill from next bar open."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.exit_sim import (
    SLIP_RTH, TradeWindow, _clock, _info, baseline_policy, close_to_label, fixed_time_policy,
    load_windows, simulate, simulate_trade,
)

NY = "America/New_York"
ROOT = Path(__file__).resolve().parents[1]


def make_window(path=None, vol=1e6, pm=None, n_rth=91):
    """Entry 2025-01-02 15:55 at 10.0.  RTH next day 09:30.. (n_rth one-minute bars, to 11:00).

    path: {minute_since_0930: close}; unspecified minutes carry the previous close.
    pm: list of 3 pre-market closes at 08:00..08:02.  opens == previous close unless overridden later.
    """
    t = pd.date_range("2025-01-02 15:55", periods=6, freq="min", tz=NY)
    t = t.append(pd.date_range("2025-01-03 08:00", periods=3, freq="min", tz=NY))
    t = t.append(pd.date_range("2025-01-03 09:30", periods=n_rth, freq="min", tz=NY))
    c = np.full(len(t), 10.0)
    if pm:
        c[6:9] = pm
    else:
        c[6:9] = 10.0
    last = c[8]
    for m in range(n_rth):
        last = (path or {}).get(m, last)
        c[9 + m] = last
    o = np.concatenate([[c[0]], c[:-1]]).copy()
    v = np.full(len(t), float(vol))
    return TradeWindow(0, t.asi8, o, np.maximum(o, c), np.minimum(o, c), c, v, c.copy(), 10.0, 10.0, 1.0, "AAA")


def idx_of(w, hhmm):
    ts = pd.Timestamp(f"2025-01-03 {hhmm}", tz=NY).value
    t = np.asarray(w.times, dtype=np.int64)
    if np.max(np.abs(t)) < 10**17:
        ts //= 1000
    return int(np.flatnonzero(t == ts)[0])


def label_of(w, idx):
    return pd.Timestamp(simulate_trade.__globals__["_ts"](w, idx)).strftime("%H:%M")


def up_then_drop_path():
    """Ramp 10.2 -> 12.0 over minutes 0..10, flat 12 to minute 17, sharp drop to 11.0 at 09:48 (m=18)."""
    p = {m: 10.2 + 0.18 * m for m in range(11)}
    p[11] = 12.0
    p[18] = 11.0
    return p


class ExitSimulatorTests(unittest.TestCase):
    # ---- carried-over tests -------------------------------------------------------------
    def test_08_floor_excludes_early_pre_market(self):
        w = make_window()
        r = simulate_trade(w, {"name": "smart", "pm_target": 0.05, "keep_ah": False}, "legacy")
        self.assertGreaterEqual(pd.Timestamp(r["exit_time"]).hour, 8)

    def test_baseline_policy_has_pm_floor(self):
        self.assertEqual(baseline_policy()["pm_start"], "08:00")

    def test_pm_target_fires_on_premarket_bar_with_floor(self):
        w = make_window(pm=[13.0, 13.0, 13.0])
        r = simulate_trade(w, {"name": "smart", "pm_target": 0.25, "keep_ah": False, "deadline": "10:15"}, "realistic")
        self.assertEqual(r["reason"], "pm_target")
        self.assertEqual(r["decision_idx"], 6)

    # ---- defect 1: later bars -----------------------------------------------------------
    def _later_bar_case(self, policy):
        w = make_window(up_then_drop_path())
        r = simulate_trade(w, {"name": "smart", "keep_ah": False, "deadline": "10:15", **policy}, "realistic")
        self.assertEqual(label_of(w, r["decision_idx"]), "09:48", r)
        self.assertGreater(r["exit_idx"], r["decision_idx"])
        return r

    def test_trailing_fires_on_later_bar(self):
        r = self._later_bar_case({"trail": 0.05, "activate": 0.0})
        self.assertEqual(r["reason"], "trailing_stop")

    def test_velocity_floor_fires_on_later_bar(self):
        r = self._later_bar_case({"velocity_floor": -0.05})
        self.assertEqual(r["reason"], "momentum_loss")

    def test_giveback_fires_on_later_bar(self):
        r = self._later_bar_case({"giveback": 0.05})
        self.assertEqual(r["reason"], "giveback")

    def test_session_vwap_loss_fires_on_later_bar(self):
        r = self._later_bar_case({"below_vwap": True})
        self.assertEqual(r["reason"], "vwap_loss")

    def test_session_vwap_uses_cumulative_typical_price_not_bar_vwap_field(self):
        w = make_window(up_then_drop_path())
        w.vwap[:] = 1e9  # per-bar vwap field is garbage: must be ignored
        r = simulate_trade(w, {"name": "smart", "keep_ah": False, "below_vwap": True}, "realistic")
        self.assertEqual(label_of(w, r["decision_idx"]), "09:48")

    def test_peak_from_open_vs_entry(self):
        # pre-market spike to 13 then regular session sits at 11.5: entry-peak trail fires at 09:30, open-peak never
        w = make_window(path={m: 11.5 for m in range(91)}, pm=[13.0, 13.0, 13.0])
        base = {"name": "smart", "keep_ah": False, "trail": 0.05, "activate": 0.0, "deadline": "10:15"}
        e = simulate_trade(w, {**base, "peak_from": "entry"}, "realistic")
        o = simulate_trade(w, {**base, "peak_from": "open"}, "realistic")
        self.assertEqual(e["reason"], "trailing_stop")
        self.assertEqual(label_of(w, e["decision_idx"]), "09:30")
        self.assertEqual(o["reason"], "deadline")
        self.assertEqual(label_of(w, o["decision_idx"]), "10:15")

    # ---- defect 2: fixed time -----------------------------------------------------------
    def test_fixed_time_decision_bars_and_prices_differ(self):
        w = make_window({m: 10.0 + 0.01 * m for m in range(91)})
        closes = ["09:31", "09:35", "09:40", "09:45", "10:00", "10:15", "10:30"]
        labels = {"09:31": "09:30", "09:35": "09:34", "09:40": "09:39", "09:45": "09:44",
                  "10:00": "09:59", "10:15": "10:14", "10:30": "10:29"}
        px = []
        for c in closes:
            self.assertEqual(close_to_label(c), labels[c])
            r = simulate_trade(w, fixed_time_policy(c), "realistic")
            self.assertEqual(label_of(w, r["decision_idx"]), labels[c])
            # decision bar closes at c; first fill bar is labelled c (opens at c)
            self.assertEqual(label_of(w, r["first_fill_idx"]), c)
            px.append(r["exit_price"])
        self.assertTrue(np.all(np.diff(px) > 0), px)

    def test_deadline_key_is_label_of_decision_bar(self):
        w = make_window({m: 10.0 + 0.01 * m for m in range(91)})
        r = simulate_trade(w, {"name": "fixed_time", "keep_ah": False, "deadline": "09:30"}, "realistic")
        self.assertEqual(label_of(w, r["decision_idx"]), "09:30")
        r = simulate_trade(w, {"name": "fixed_time", "keep_ah": False, "deadline": "09:35"}, "realistic")
        self.assertEqual(label_of(w, r["decision_idx"]), "09:35")

    # ---- defect 3: fills ----------------------------------------------------------------
    def test_realistic_exit_price_comes_from_next_bar_open(self):
        w = make_window({m: 10.0 + 0.01 * m for m in range(91)})
        t = idx_of(w, "09:44")
        w.opens[t] = 1.0          # sentinels on the decision bar and the one before
        w.opens[t - 1] = 2.0
        w.opens[t + 1] = 11.0
        w.closes[t] = 10.44       # decision bar close (not used for the fill)
        r = simulate_trade(w, fixed_time_policy("09:45"), "realistic")
        self.assertEqual(r["decision_idx"], t)
        self.assertAlmostEqual(r["exit_price"], 11.0 * (1 - SLIP_RTH), places=9)
        leg = simulate_trade(w, fixed_time_policy("09:45"), "legacy")
        self.assertAlmostEqual(leg["exit_price"], 10.44 * (1 - SLIP_RTH), places=9)

    def test_no_fill_at_or_before_decision_and_volume_limited(self):
        # 125 shares, cap = 10% * 100 = 10 per bar -> 13 bars
        w = make_window({m: 10.0 + 0.01 * m for m in range(91)}, vol=100.0)
        for c in ("09:31", "09:45", "10:15"):
            r = simulate_trade(w, fixed_time_policy(c), "realistic")
            self.assertGreater(r["first_fill_idx"], r["decision_idx"])
            self.assertEqual(r["exit_idx"], r["decision_idx"] + 13)
            self.assertEqual(r["filled_fraction"], 1.0)

    def test_residue_after_window_priced_at_last_bar_open(self):
        w = make_window({m: 10.0 + 0.01 * m for m in range(91)}, vol=1.0)  # 0.1 share/bar: never completes
        r = simulate_trade(w, fixed_time_policy("10:30"), "realistic")
        last = len(w.times) - 1
        self.assertGreater(r["decision_idx"], 0)
        self.assertEqual(r["exit_idx"], last)
        self.assertGreater(r["first_fill_idx"], r["decision_idx"])

    def test_decision_on_last_bar_is_unfilled_nan_not_lookahead(self):
        w = make_window({m: 10.0 for m in range(91)}, n_rth=46)  # last bar 10:15
        r = simulate_trade(w, fixed_time_policy("10:16"), "realistic")
        self.assertTrue(np.isnan(r["ret"]))

    # ---- defect 4: scale-out remainder trailed -------------------------------------------
    def test_scale_out_then_trailing_on_remainder(self):
        w = make_window(up_then_drop_path())
        policy = {"name": "smart", "keep_ah": False, "deadline": "10:15", "trail": 0.05, "activate": 0.10,
                  "scale_out": [{"target": 0.10, "fraction": 0.5}]}
        r = simulate_trade(w, policy, "realistic")
        self.assertIn("scale_target", r["reason"])
        self.assertIn("trailing_stop", r["reason"])
        self.assertEqual(label_of(w, r["decision_idx"]), "09:48")
        self.assertEqual(r["filled_fraction"], 1.0)

    # ---- no lookahead --------------------------------------------------------------------
    def test_no_lookahead_future_perturbation_does_not_change_decision(self):
        w = make_window(up_then_drop_path())
        for policy in ({"trail": 0.05, "activate": 0.0}, {"below_vwap": True}, {"giveback": 0.05},
                       {"velocity_floor": -0.05}):
            p = {"name": "smart", "keep_ah": False, "deadline": "10:15", **policy}
            a = simulate_trade(w, p, "realistic")
            ch = TradeWindow(w.trade_id, w.times, w.opens.copy(), w.highs.copy(), w.lows.copy(), w.closes.copy(),
                             w.volumes.copy(), w.vwap.copy(), w.entry_price, w.entry_signal_price, w.fill_entry, "AAA")
            d = a["decision_idx"]
            ch.closes[d + 1:] *= 3.0
            ch.highs[d + 1:] *= 3.0
            ch.lows[d + 1:] *= 0.1
            ch.volumes[d + 1:] *= 0.5
            b = simulate_trade(ch, p, "realistic")
            self.assertEqual(a["decision_idx"], b["decision_idx"])
            self.assertEqual(a["reason"], b["reason"])

    # ---- legacy equivalence with stored trades --------------------------------------------
    def test_legacy_baseline_matches_stored_trades(self):
        npz = ROOT / "results" / "exit" / "windows.npz"
        csv = ROOT / "results" / "pit_trades_atr_pivot_lb13_pm_target25_v2.csv"
        if not (npz.exists() and csv.exists()):
            self.skipTest("cached windows / trade table not available")
        dev = pd.read_csv(csv)
        ws = [w for w in load_windows(npz) if w.trade_id < len(dev)]
        rng = np.random.default_rng(1)
        sample = [ws[i] for i in rng.choice(len(ws), 20, replace=False)]
        res = simulate(sample, baseline_policy(), "legacy")
        diff = res["ret"].to_numpy() - dev.iloc[res["trade_id"]]["ret"].to_numpy()
        self.assertLess(np.max(np.abs(diff)), 1e-9)

    def test_loaded_windows_have_unique_timestamps(self):
        npz = ROOT / "results" / "exit" / "windows.npz"
        if not npz.exists():
            self.skipTest("no cache")
        for w in load_windows(npz)[:300]:
            self.assertTrue(np.all(np.diff(w.times) > 0))


if __name__ == "__main__":
    unittest.main()
