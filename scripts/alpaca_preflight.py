"""
Alpaca paper preflight: auth, account, clock, calendar, asset flags, data feeds, and (with --order) a resting far-from-market limit
order that is confirmed and then cancelled. Paper only; refuses to run against a live endpoint. Leaves no orders or positions behind.

  python scripts/alpaca_preflight.py
  python scripts/alpaca_preflight.py --order --symbol AAPL
"""

import argparse
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetCalendarRequest, GetOrdersRequest, LimitOrderRequest

load_dotenv()


def check(label, fn):
    try:
        out = fn()
        print(f"[ok]   {label}: {out}")
        return out
    except Exception as e:
        print(f"[FAIL] {label}: {type(e).__name__}: {str(e)[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--order", action="store_true", help="place and cancel a far-from-market limit order")
    a = ap.parse_args()

    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
    url = os.getenv("ALPACA_BASE_URL", "")
    if not key or not secret:
        sys.exit("ALPACA_API_KEY / ALPACA_API_SECRET missing")
    if "paper" not in url:
        sys.exit(f"refusing to run: ALPACA_BASE_URL is not a paper endpoint ({url!r})")

    tc = TradingClient(key, secret, paper=True)
    dc = StockHistoricalDataClient(key, secret)

    acct = check("account", lambda: (lambda x: f"status={x.status} equity={x.equity} cash={x.cash} buying_power={x.buying_power} "
                                              f"pdt={x.pattern_day_trader} shorting={x.shorting_enabled}")(tc.get_account()))
    clock = check("clock", lambda: (lambda c: f"open={c.is_open} next_open={c.next_open} next_close={c.next_close}")(tc.get_clock()))
    check("calendar", lambda: [(d.date.isoformat(), d.open.strftime("%H:%M"), d.close.strftime("%H:%M"))
                               for d in tc.get_calendar(GetCalendarRequest(start=datetime.now().date(), end=datetime.now().date() + timedelta(days=7)))])
    asset = check("asset", lambda: (lambda s: f"{s.symbol} exchange={s.exchange} tradable={s.tradable} marginable={s.marginable} "
                                              f"shortable={s.shortable} fractionable={s.fractionable}")(tc.get_asset(a.symbol)))
    check("open orders", lambda: len(tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))))
    check("positions", lambda: [(p.symbol, p.qty) for p in tc.get_all_positions()])

    last = check("latest trade (iex)", lambda: dc.get_stock_latest_trade(
        StockLatestTradeRequest(symbol_or_symbols=a.symbol, feed=DataFeed.IEX))[a.symbol].price)
    check("latest trade (sip, expect failure on free plan)", lambda: dc.get_stock_latest_trade(
        StockLatestTradeRequest(symbol_or_symbols=a.symbol, feed=DataFeed.SIP))[a.symbol].price)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    check("historical sip bars (older than 15 min)", lambda: len(dc.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=a.symbol, timeframe=TimeFrame.Minute, start=end - timedelta(days=4), end=end, feed=DataFeed.SIP)).data.get(a.symbol, [])))
    check("historical sip bars (last 5 min, expect failure on free plan)", lambda: len(dc.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=a.symbol, timeframe=TimeFrame.Minute, start=datetime.now(timezone.utc) - timedelta(days=4),
        end=datetime.now(timezone.utc), feed=DataFeed.SIP)).data.get(a.symbol, [])))

    if not a.order:
        print("skipped order test (pass --order)")
        return
    if not (acct and asset and last):
        sys.exit("cannot run order test: account/asset/price checks failed")

    px = round(float(last) * 0.5, 2)
    cid = f"preflight-{uuid.uuid4().hex[:12]}"
    print(f"placing LIMIT BUY 1 {a.symbol} @ {px} (50% of last), DAY, client_order_id={cid}")
    order = tc.submit_order(LimitOrderRequest(symbol=a.symbol, qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                                              limit_price=px, client_order_id=cid))
    try:
        time.sleep(2)
        o = tc.get_order_by_client_id(cid)
        print(f"[ok]   resting: status={o.status} id={o.id} extended_hours={o.extended_hours} filled_qty={o.filled_qty}")
    finally:
        try:
            tc.cancel_order_by_id(order.id)
        except APIError as e:
            print(f"cancel error: {e}")
        time.sleep(2)
        o = tc.get_order_by_client_id(cid)
        print(f"[ok]   after cancel: status={o.status}")
    left = [x for x in tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)) if x.client_order_id == cid]
    print("[ok]   no leftover preflight order" if not left else f"[FAIL] leftover order {left[0].id}")


if __name__ == "__main__":
    main()
