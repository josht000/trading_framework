"""
H3 + hygiene + tuned score, pm_target25_v2 exits. Pure decision logic: no broker, clock or I/O access, so it is testable and replayable.

Entry (15:55 ET): hygiene (no OTC), headroom >= 20%, above SMA200, tuned score >= threshold, top N by rank_key.
Exit: after-hours 16:00-16:45 stop/target on bar closes; then next-day pre-market target; otherwise the first regular-session bar.
"""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from live import clock
from live.config import ExitConfig, RiskConfig, StrategyConfig
from live.core.models import ExitReason, Position, Quote
from live.risk import guard
from live.risk.sleeve import Sleeve
from src.backtest import ranking as rk

SLIP_RTH, SLIP_EXT = 0.0015, 0.004


@dataclass(frozen=True)
class EntryIntent:
    symbol: str
    qty: int
    limit_px: float
    score: float
    meta: dict


@dataclass(frozen=True)
class ExitIntent:
    reason: ExitReason
    limit_px: float
    extended: bool


def score_candidates(features: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    s = rk.score(features, cfg.weights)
    rel = features["rel_vol"].clip(upper=100).fillna(0) / 100.0 if "rel_vol" in features else 0.0
    return features.assign(score=s, rank_key=s + 1e-6 * rel)


def eligible(features: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    m = pd.Series(True, index=features.index)
    if cfg.hygiene_level >= 1:
        m &= ~features["is_otc"].fillna(False).astype(bool)
    m &= features["headroom"] >= cfg.headroom_min
    if cfg.above_sma200:
        m &= features["above_sma200"] == 1
    s = score_candidates(features[m.fillna(False)], cfg)
    return s[s["score"] >= cfg.threshold].sort_values("rank_key", ascending=False, kind="stable")


def entry_limit(price: float) -> float:
    return round(price * (1 + SLIP_RTH), 2)


def plan_entries(cfg: StrategyConfig, risk: RiskConfig, sleeve: Sleeve, features: pd.DataFrame, quotes: dict, now: datetime,
                 held_n: int = 0, bar_volume: dict | None = None, tradable: dict | None = None):
    """Returns (intents, skipped) where skipped is a list of (symbol, reasons)."""
    picks = eligible(features, cfg).head(max(0, cfg.sizing.max_positions - held_n))
    intents, skipped = [], []
    equity, cash, cost, n = sleeve.equity(), sleeve.cash(), sleeve.cost_basis(), held_n
    for _, r in picks.iterrows():
        sym, q = r["symbol"], quotes.get(r["symbol"])
        if q is None:
            skipped.append((sym, ["no quote"]))
            continue
        px = entry_limit(q.last)
        qty = sleeve.slot_shares(px, cfg.sizing.position_frac)
        qty = min(qty, int(cash // px))
        bad = guard.pre_trade(risk, side="buy", qty=qty, limit_px=px, quote=q, now=now, equity=equity, cash=cash, held_n=n,
                              held_cost=cost, bar_volume=(bar_volume or {}).get(sym), participation=cfg.sizing.participation,
                              tradable=(tradable or {}).get(sym, True))
        if bad:
            skipped.append((sym, bad))
            continue
        intents.append(EntryIntent(sym, qty, px, float(r["score"]), {"rank_key": float(r["rank_key"])}))
        cash, cost, n = cash - qty * px, cost + qty * px, n + 1
    return intents, skipped


def entry_filled_enough(filled: int, target: int, cfg: StrategyConfig) -> bool:
    return target > 0 and filled / target >= cfg.sizing.min_fill


def marketable_limit(side: str, ref: float, extended: bool) -> float:
    slip = SLIP_EXT if extended else SLIP_RTH
    return round(ref * (1 - slip if side == "sell" else 1 + slip), 2)


def exit_signal(pos: Position, now: datetime, last_px: float, cal: clock.SessionCalendar, cfg: ExitConfig) -> ExitIntent | None:
    """What (if anything) a held position should do right now. last_px is the latest 1-min close.
    Also covers restarts: if the app was down through the pre-market or the open, the open exit fires immediately."""
    n = clock.ny(now)
    entry_day = pd.Timestamp(pos.day).date()
    t = n.time()
    entry = pos.entry_px
    if n.date() == entry_day:
        end = clock.at(entry_day, "16:00") + pd.Timedelta(minutes=cfg.ah_window_min)
        if clock.at(entry_day, "16:00") <= n < end:
            if last_px <= entry * (1 - cfg.ah_stop):
                return ExitIntent(ExitReason.AH_STOP, marketable_limit("sell", last_px, True), True)
            if last_px >= entry * (1 + cfg.ah_target):
                return ExitIntent(ExitReason.AH_TARGET, marketable_limit("sell", last_px, True), True)
        return None
    if n.date() < entry_day or not cal.is_session(n.date()):
        return None
    if clock.OPEN <= t:
        return ExitIntent(ExitReason.OPEN, marketable_limit("sell", last_px, False), False)
    if clock.PRE_OPEN <= t < clock.OPEN and last_px >= entry * (1 + cfg.pm_target):
        return ExitIntent(ExitReason.PM_TARGET, marketable_limit("sell", last_px, True), True)
    return None
