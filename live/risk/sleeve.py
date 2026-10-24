import math

from live.core.models import PosState
from live.core.store import Store


class Sleeve:
    """Internal capital sleeve: equity = start + realized PnL + unrealized PnL of this strategy's positions. Broker equity is never read."""

    def __init__(self, store: Store, strategy: str, start: float):
        self.store, self.strategy, self.start = store, strategy, start

    def _held(self):
        return [p for p in self.store.open_positions(self.strategy) if p.state in (PosState.OPEN, PosState.EXIT_PENDING) and p.entry_px]

    def cost_basis(self):
        return sum(p.entry_px * p.qty for p in self._held())

    def cash(self):
        return self.start + self.store.realized_pnl(self.strategy) - self.cost_basis()

    def equity(self, marks=None):
        marks = marks or {}
        mv = sum(marks.get(p.symbol, p.entry_px) * p.qty for p in self._held())
        return self.cash() + mv

    def slot_shares(self, price, frac, marks=None):
        """Whole shares for one position: frac of sleeve equity, never more than sleeve cash."""
        if price <= 0:
            return 0
        budget = min(frac * self.equity(marks), self.cash())
        return max(0, math.floor(budget / price))
