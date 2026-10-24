from datetime import date, datetime, time, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
PRE_OPEN, OPEN, CLOSE, POST_CLOSE = time(4, 0), time(9, 30), time(16, 0), time(20, 0)


class Phase(str, Enum):
    CLOSED = "closed"
    PRE = "pre"
    RTH = "rth"
    AFTER = "after"


class SessionCalendar:
    """Trading days. Weekdays by default; pass explicit session dates (e.g. from Alpaca's calendar) to honour holidays."""

    def __init__(self, sessions=None):
        self.sessions = sorted(sessions) if sessions else None

    def is_session(self, d: date) -> bool:
        return d in set(self.sessions) if self.sessions else d.weekday() < 5

    def next_session(self, d: date) -> date:
        n = d + timedelta(days=1)
        while not self.is_session(n):
            n += timedelta(days=1)
        return n


def ny(now: datetime) -> datetime:
    return now.astimezone(NY) if now.tzinfo else now.replace(tzinfo=NY)


def phase(now: datetime, cal: SessionCalendar) -> Phase:
    n = ny(now)
    if not cal.is_session(n.date()):
        return Phase.CLOSED
    t = n.time()
    if PRE_OPEN <= t < OPEN:
        return Phase.PRE
    if OPEN <= t < CLOSE:
        return Phase.RTH
    if CLOSE <= t < POST_CLOSE:
        return Phase.AFTER
    return Phase.CLOSED


def at(day: date, hhmm: str) -> datetime:
    h, m = map(int, hhmm.split(":"))
    return datetime.combine(day, time(h, m), tzinfo=NY)
