import os
from datetime import datetime, timedelta

from live.config import ROOT, RiskConfig
from live.core.models import Quote, TradingState


def pre_trade(cfg: RiskConfig, *, side, qty, limit_px, quote: Quote, now: datetime, equity, cash, held_n, held_cost,
              bar_volume=None, participation=0.10, tradable=True):
    """Order-level checks. Returns a list of violations; an empty list means the order may be sent."""
    v = []
    if not tradable:
        v.append("not tradable")
    if qty <= 0:
        v.append("qty <= 0")
    if side == "sell":
        return v
    notional = qty * limit_px
    if notional > cfg.max_position_frac * equity * 1.001:
        v.append(f"position {notional:.0f} > {cfg.max_position_frac:.0%} of sleeve equity {equity:.0f}")
    if notional > cash + 1e-6:
        v.append(f"notional {notional:.0f} > sleeve cash {cash:.0f}")
    if held_n >= cfg.max_positions:
        v.append(f"already {held_n} positions")
    if held_cost + notional > cfg.max_gross_frac * equity * 1.001:
        v.append("gross exposure above sleeve limit")
    if quote.last <= 0 or abs(limit_px / quote.last - 1) > cfg.max_limit_deviation:
        v.append(f"limit {limit_px} too far from last {quote.last}")
    if quote.bid and quote.ask and (quote.ask - quote.bid) / quote.mid > cfg.max_spread_frac:
        v.append("spread too wide")
    if quote.ts is not None and now - quote.ts > timedelta(seconds=cfg.max_quote_age_sec):
        v.append("stale quote")
    if bar_volume is not None and qty > participation * bar_volume:
        v.append(f"qty {qty} > {participation:.0%} of recent bar volume {bar_volume:.0f}")
    return v


class Guard:
    """Portfolio-level circuit breakers. Loss breakers stop new entries but allow exits (REDUCING);
    kill file, order rejects and broker/local mismatches halt everything and need a human."""

    def __init__(self, cfg: RiskConfig):
        self.cfg, self.rejects, self.mismatch = cfg, 0, False

    @property
    def kill_path(self):
        p = self.cfg.kill_file
        return p if os.path.isabs(p) else str(ROOT / p)

    def killed(self):
        return os.path.exists(self.kill_path)

    def note_reject(self):
        self.rejects += 1

    def note_ok(self):
        self.rejects = 0

    def state(self, equity, day_start_equity, peak_equity):
        reasons = []
        if self.killed():
            reasons.append("kill file present")
        if self.rejects >= self.cfg.max_order_rejects:
            reasons.append(f"{self.rejects} consecutive order rejects")
        if self.mismatch:
            reasons.append("broker/local mismatch")
        if reasons:
            return TradingState.HALTED, reasons
        if day_start_equity and equity / day_start_equity - 1 <= self.cfg.daily_loss_halt:
            reasons.append(f"daily loss {equity / day_start_equity - 1:.1%}")
        if peak_equity and equity / peak_equity - 1 <= self.cfg.drawdown_halt:
            reasons.append(f"drawdown {equity / peak_equity - 1:.1%}")
        return (TradingState.REDUCING, reasons) if reasons else (TradingState.ACTIVE, [])
