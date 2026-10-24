import os
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from live import clock
from live.config import load_risk, load_strategy
from live.core.models import ExitReason, PosState, Position, Quote, TradingState
from live.core.store import Store
from live.risk import guard
from live.risk.sleeve import Sleeve
from live.strategies import overnight_h3 as st
from src.backtest import ranking as rk

CAL = clock.SessionCalendar()
CFG, RISK = load_strategy(), load_risk()
MON = datetime(2026, 10, 19).date()


def ts(day, hhmm):
    return clock.at(day, hhmm)


def features(rows):
    base = dict(symbol="AAA", headroom=0.6, above_sma200=1, iwm_day=0.01, spy_day=0.01, is_otc=False, rel_vol=5.0,
                n_top_gainer_days=0, n_mover_5d=0, e_delist_notice_90d=0, e_late_30d=0, theme_heat=3, sector_heat=0, shares_pit=20e6)
    return pd.DataFrame([{**base, **r} for r in rows])


class StoreTests(unittest.TestCase):
    def test_state_survives_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "live.db")
            s = Store(path)
            p = s.add_position(Position("overnight_h3", "2026-10-19", "AAA", 10, PosState.OPEN, entry_px=5.0, meta={"oid": "x"}))
            s.event("overnight_h3", "entry", symbol="AAA")
            s.set("last_day", "2026-10-19")
            s.db.close()
            s2 = Store(path)
            held = s2.open_positions("overnight_h3")
            self.assertEqual([(h.id, h.symbol, h.state, h.meta) for h in held], [(p.id, "AAA", PosState.OPEN, {"oid": "x"})])
            self.assertEqual(s2.get("last_day"), "2026-10-19")
            self.assertEqual(len(s2.events(kind="entry")), 1)
            s2.db.close()

    def test_one_position_per_symbol_per_day(self):
        s = Store(":memory:")
        s.add_position(Position("a", "2026-10-19", "AAA", 1))
        with self.assertRaises(Exception):
            s.add_position(Position("a", "2026-10-19", "AAA", 1))


class SleeveTests(unittest.TestCase):
    def setUp(self):
        self.s = Store(":memory:")
        self.sl = Sleeve(self.s, "a", 5000.0)

    def test_start_sizing_is_20_percent_whole_shares(self):
        self.assertEqual(self.sl.slot_shares(3.0, 0.20), 333)
        self.assertEqual(self.sl.slot_shares(2000.0, 0.20), 0)

    def test_equity_compounds_realized_and_marks_open(self):
        p = self.s.add_position(Position("a", "2026-10-16", "AAA", 100, PosState.CLOSED, entry_px=5.0, exit_px=6.0, exit_ts="2026-10-19T09:30", pnl=100.0))
        self.s.add_position(Position("a", "2026-10-19", "BBB", 100, PosState.OPEN, entry_px=10.0))
        self.assertAlmostEqual(self.sl.cash(), 5000 + 100 - 1000)
        self.assertAlmostEqual(self.sl.equity({"BBB": 11.0}), 5100 + 100)
        self.assertEqual(self.sl.slot_shares(10.0, 0.20, {"BBB": 11.0}), 104)


class GuardTests(unittest.TestCase):
    now = ts(MON, "15:55")
    q = Quote("AAA", 4.99, 5.01, 5.0, now)

    def check(self, **kw):
        a = dict(side="buy", qty=200, limit_px=5.0, quote=self.q, now=self.now, equity=5000, cash=5000, held_n=0, held_cost=0)
        return guard.pre_trade(RISK, **{**a, **kw})

    def test_clean_order_passes(self):
        self.assertEqual(self.check(), [])

    def test_violations(self):
        self.assertTrue(self.check(qty=300))
        self.assertTrue(self.check(held_n=5))
        self.assertTrue(self.check(limit_px=5.5))
        self.assertTrue(self.check(quote=Quote("AAA", 4.0, 5.0, 5.0, self.now)))
        self.assertTrue(self.check(quote=Quote("AAA", 4.99, 5.01, 5.0, self.now - timedelta(minutes=5))))
        self.assertTrue(self.check(bar_volume=1000))
        self.assertTrue(self.check(cash=500))
        self.assertTrue(self.check(tradable=False))

    def test_breakers(self):
        g = guard.Guard(RISK)
        self.assertEqual(g.state(5000, 5000, 5000)[0], TradingState.ACTIVE)
        self.assertEqual(g.state(4700, 5000, 5000)[0], TradingState.REDUCING)
        self.assertEqual(g.state(4200, 4300, 5000)[0], TradingState.REDUCING)
        g.mismatch = True
        self.assertEqual(g.state(5000, 5000, 5000)[0], TradingState.HALTED)
        g = guard.Guard(RISK)
        for _ in range(RISK.max_order_rejects):
            g.note_reject()
        self.assertEqual(g.state(5000, 5000, 5000)[0], TradingState.HALTED)


class SelectionTests(unittest.TestCase):
    def test_weights_match_research_selection(self):
        self.assertEqual(CFG.weights["delist_notice"], 3.0)
        self.assertEqual(CFG.threshold, 2.0)
        self.assertEqual(set(CFG.weights), set(rk.COMPONENTS))

    def test_filters_hygiene_threshold_and_rank(self):
        f = features([
            dict(symbol="GOOD", theme_heat=3),
            dict(symbol="OTC", is_otc=True),
            dict(symbol="LOWHEAD", headroom=0.10),
            dict(symbol="BELOW200", above_sma200=0),
            dict(symbol="DELIST", e_delist_notice_90d=1),
            dict(symbol="BEST", theme_heat=3, n_top_gainer_days=3),
        ])
        out = st.eligible(f, CFG)
        self.assertEqual(list(out["symbol"]), ["BEST", "GOOD"])
        self.assertTrue((out["score"] >= 2).all())

    def test_score_matches_ranking_module(self):
        f = features([dict(symbol="X")])
        self.assertAlmostEqual(st.score_candidates(f, CFG)["score"].iloc[0], rk.score(f, CFG.weights).iloc[0])

    def test_plan_entries_sizes_top_five_at_20_percent(self):
        store = Store(":memory:")
        sl = Sleeve(store, CFG.name, 5000.0)
        now = ts(MON, "15:55")
        f = features([dict(symbol=f"S{i}", n_top_gainer_days=i % 4) for i in range(8)])
        quotes = {f"S{i}": Quote(f"S{i}", 4.99, 5.01, 5.0, now) for i in range(8)}
        intents, skipped = st.plan_entries(CFG, RISK, sl, f, quotes, now)
        self.assertEqual(len(intents), 5)
        self.assertTrue(all(i.qty * i.limit_px <= 1000 * 1.001 for i in intents))
        self.assertLessEqual(sum(i.qty * i.limit_px for i in intents), 5000)
        self.assertEqual(skipped, [])

    def test_plan_entries_respects_held_and_missing_quote(self):
        store = Store(":memory:")
        sl = Sleeve(store, CFG.name, 5000.0)
        now = ts(MON, "15:55")
        f = features([dict(symbol="A"), dict(symbol="B")])
        intents, skipped = st.plan_entries(CFG, RISK, sl, f, {"A": Quote("A", 4.99, 5.01, 5.0, now)}, now, held_n=4)
        self.assertEqual([i.symbol for i in intents] + [s for s, _ in skipped], ["A"] if len(intents) == 1 else ["A"])
        self.assertLessEqual(len(intents), 1)

    def test_min_fill(self):
        self.assertTrue(st.entry_filled_enough(25, 100, CFG))
        self.assertFalse(st.entry_filled_enough(24, 100, CFG))


class ExitTests(unittest.TestCase):
    def pos(self):
        return Position(CFG.name, "2026-10-19", "AAA", 100, PosState.OPEN, entry_px=10.0)

    def sig(self, day, hhmm, px):
        return st.exit_signal(self.pos(), ts(day, hhmm), px, CAL, CFG.exits)

    def test_after_hours_stop_and_target_use_bar_close(self):
        self.assertEqual(self.sig(MON, "16:10", 8.79).reason, ExitReason.AH_STOP)
        self.assertIsNone(self.sig(MON, "16:10", 8.81))
        self.assertEqual(self.sig(MON, "16:30", 15.0).reason, ExitReason.AH_TARGET)
        self.assertIsNone(self.sig(MON, "16:46", 5.0))
        self.assertIsNone(self.sig(MON, "15:58", 5.0))

    def test_premarket_target_then_open(self):
        tue = MON + timedelta(days=1)
        self.assertEqual(self.sig(tue, "05:00", 12.5).reason, ExitReason.PM_TARGET)
        self.assertIsNone(self.sig(tue, "05:00", 12.4))
        s = self.sig(tue, "09:30", 11.0)
        self.assertEqual(s.reason, ExitReason.OPEN)
        self.assertFalse(s.extended)
        self.assertLess(s.limit_px, 11.0)

    def test_restart_after_open_still_exits_but_not_on_closed_days(self):
        tue = MON + timedelta(days=1)
        self.assertEqual(self.sig(tue, "11:00", 9.0).reason, ExitReason.OPEN)
        sat = MON + timedelta(days=5)
        self.assertIsNone(self.sig(sat, "11:00", 9.0))
        self.assertIsNone(self.sig(MON, "20:30", 9.0))


class ClockTests(unittest.TestCase):
    def test_phases(self):
        self.assertEqual(clock.phase(ts(MON, "03:59"), CAL), clock.Phase.CLOSED)
        self.assertEqual(clock.phase(ts(MON, "04:00"), CAL), clock.Phase.PRE)
        self.assertEqual(clock.phase(ts(MON, "12:00"), CAL), clock.Phase.RTH)
        self.assertEqual(clock.phase(ts(MON, "16:00"), CAL), clock.Phase.AFTER)
        self.assertEqual(clock.phase(ts(MON + timedelta(days=5), "12:00"), CAL), clock.Phase.CLOSED)

    def test_next_session_skips_weekend_and_holidays(self):
        self.assertEqual(CAL.next_session(MON + timedelta(days=4)), MON + timedelta(days=7))
        hol = clock.SessionCalendar([MON, MON + timedelta(days=2)])
        self.assertEqual(hol.next_session(MON), MON + timedelta(days=2))


if __name__ == "__main__":
    unittest.main()
