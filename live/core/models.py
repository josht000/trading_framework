from dataclasses import dataclass, field
from enum import Enum


class PosState(str, Enum):
    ENTRY_PENDING = "ENTRY_PENDING"
    OPEN = "OPEN"
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


class TradingState(str, Enum):
    ACTIVE = "ACTIVE"
    REDUCING = "REDUCING"
    HALTED = "HALTED"


class ExitReason(str, Enum):
    AH_STOP = "ah_stop"
    AH_TARGET = "ah_target"
    PM_TARGET = "pm_target"
    OPEN = "open"
    KILL = "kill"


@dataclass
class Position:
    strategy: str
    day: str
    symbol: str
    qty: int
    state: PosState = PosState.ENTRY_PENDING
    limit_px: float = 0.0
    entry_px: float = 0.0
    entry_ts: str = ""
    exit_px: float = 0.0
    exit_ts: str = ""
    exit_reason: str = ""
    pnl: float = 0.0
    meta: dict = field(default_factory=dict)
    id: int = 0


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    last: float
    ts: object

    @property
    def mid(self):
        return (self.bid + self.ask) / 2 if self.bid and self.ask else self.last
