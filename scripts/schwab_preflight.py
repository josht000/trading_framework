"""
Read-only Schwab market data check: token age, quotes, price history (minute and daily), market hours, movers, streamer info and a
short live stream. Places no orders (the client cannot reach order endpoints).

  python scripts/schwab_preflight.py
  python scripts/schwab_preflight.py --symbols AAPL,SPY,IWM --stream-seconds 20
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.data.schwab import SchwabAuth, SchwabRest, SchwabStreamer, load_settings


def check(label, fn):
    try:
        out = fn()
        print(f"[ok]   {label}: {out}")
        return out
    except Exception as e:
        print(f"[FAIL] {label}: {type(e).__name__}: {str(e)[:300]}")


async def stream(rest, auth, symbols, seconds):
    got = {"quotes": 0, "bars": 0}
    s = SchwabStreamer(rest, auth, on_quote=lambda q: got.__setitem__("quotes", got["quotes"] + 1),
                       on_bar=lambda b: got.__setitem__("bars", got["bars"] + 1))
    await s.subscribe(symbols, symbols)
    task = asyncio.create_task(s.run())
    try:
        await asyncio.wait_for(s.connected.wait(), 20)
        await asyncio.sleep(seconds)
    finally:
        await s.stop()
        task.cancel()
    return {**got, "delayed": {k: s.book.delayed(k) for k in symbols}, "last": {k: (q.last if (q := s.book.quote(k)) else None) for k in symbols}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="AAPL,SPY,IWM")
    ap.add_argument("--stream-seconds", type=int, default=15)
    a = ap.parse_args()
    syms = a.symbols.split(",")
    auth = SchwabAuth(load_settings())
    rest = SchwabRest(auth)
    check("token age (days)", lambda: round(auth.refresh_age_days(), 2))
    check("access token", lambda: bool(auth.access_token()))
    check("quotes", lambda: {k: (v.bid, v.ask, v.last, str(v.ts)) for k, v in rest.quotes(syms).items()})
    check("minute bars (1d, extended)", lambda: (lambda d: (len(d), str(d.index[0]), str(d.index[-1]), int(d["volume"].sum())))(rest.minute_bars(syms[0], 1)))
    check("daily bars (1y)", lambda: (lambda d: (len(d), str(d.index[0]), str(d.index[-1])))(rest.daily_bars(syms[0], 1)))
    check("market hours", lambda: rest.market_hours("equity"))
    check("movers EQUITY_ALL up", lambda: [(m["symbol"], m.get("change")) for m in rest.movers()][:5])
    info = check("streamer info (needs /trader/v1/userPreference)", lambda: {k: v for k, v in rest.streamer_info().items() if "Id" not in k and "Correl" not in k})
    if info and a.stream_seconds:
        check("stream", lambda: asyncio.run(stream(rest, auth, syms, a.stream_seconds)))
    else:
        print("skipped stream (no streamer info)")


if __name__ == "__main__":
    main()
