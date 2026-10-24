import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from live.core.models import PosState, Position

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, strategy TEXT, day TEXT, symbol TEXT, qty INTEGER, state TEXT,
  limit_px REAL, entry_px REAL, entry_ts TEXT, exit_px REAL, exit_ts TEXT, exit_reason TEXT, pnl REAL, meta TEXT,
  UNIQUE (strategy, day, symbol));
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, strategy TEXT, kind TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS snapshots (ts TEXT, strategy TEXT, equity REAL, cash REAL, open_n INTEGER, PRIMARY KEY (ts, strategy));
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""
OPEN_STATES = (PosState.ENTRY_PENDING.value, PosState.OPEN.value, PosState.EXIT_PENDING.value)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def _pos(self, r):
        d = dict(r)
        d["state"], d["meta"] = PosState(d["state"]), json.loads(d["meta"] or "{}")
        return Position(**d)

    def add_position(self, p: Position) -> Position:
        d = asdict(p)
        d.pop("id")
        d["state"], d["meta"] = p.state.value, json.dumps(p.meta)
        cols = ",".join(d)
        cur = self.db.execute(f"INSERT INTO positions ({cols}) VALUES ({','.join('?' * len(d))})", list(d.values()))
        p.id = cur.lastrowid
        return p

    def update_position(self, p: Position):
        d = asdict(p)
        pid = d.pop("id")
        d["state"], d["meta"] = p.state.value, json.dumps(p.meta)
        self.db.execute(f"UPDATE positions SET {','.join(f'{k}=?' for k in d)} WHERE id=?", [*d.values(), pid])

    def positions(self, strategy=None, states=None, day=None):
        q, args = "SELECT * FROM positions WHERE 1=1", []
        if strategy:
            q, args = q + " AND strategy=?", args + [strategy]
        if states:
            q, args = q + f" AND state IN ({','.join('?' * len(states))})", args + [s.value if hasattr(s, "value") else s for s in states]
        if day:
            q, args = q + " AND day=?", args + [day]
        return [self._pos(r) for r in self.db.execute(q + " ORDER BY id", args)]

    def open_positions(self, strategy=None):
        return self.positions(strategy, OPEN_STATES)

    def realized_pnl(self, strategy, day=None):
        q, args = "SELECT COALESCE(SUM(pnl),0) FROM positions WHERE strategy=? AND state=?", [strategy, PosState.CLOSED.value]
        if day:
            q, args = q + " AND exit_ts LIKE ?", args + [f"{day}%"]
        return float(self.db.execute(q, args).fetchone()[0])

    def event(self, strategy, kind, **payload):
        self.db.execute("INSERT INTO events (ts, strategy, kind, payload) VALUES (?,?,?,?)", (utcnow(), strategy, kind, json.dumps(payload, default=str)))

    def events(self, strategy=None, kind=None):
        q, args = "SELECT * FROM events WHERE 1=1", []
        if strategy:
            q, args = q + " AND strategy=?", args + [strategy]
        if kind:
            q, args = q + " AND kind=?", args + [kind]
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in self.db.execute(q + " ORDER BY id", args)]

    def snapshot(self, strategy, equity, cash, open_n, ts=None):
        self.db.execute("INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?)", (ts or utcnow(), strategy, equity, cash, open_n))

    def snapshots(self, strategy):
        return [dict(r) for r in self.db.execute("SELECT * FROM snapshots WHERE strategy=? ORDER BY ts", [strategy])]

    def get(self, k, default=None):
        r = self.db.execute("SELECT v FROM kv WHERE k=?", [k]).fetchone()
        return json.loads(r["v"]) if r else default

    def set(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, json.dumps(v)))
