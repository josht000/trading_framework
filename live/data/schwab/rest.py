import collections
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from live.core.models import Quote
from live.data.schwab.auth import SchwabAuth

BASE = "https://api.schwabapi.com"
ALLOWED_PREFIXES = ("/marketdata/v1/",)
ALLOWED_EXACT = ("/trader/v1/userPreference",)  # read-only; the streamer needs its connection details
NY = "America/New_York"


class DataOnlyError(RuntimeError):
    pass


def _check(path):
    if not (path.startswith(ALLOWED_PREFIXES) or path in ALLOWED_EXACT):
        raise DataOnlyError(f"{path} is not a market-data endpoint; this client is data only")


def _ms(v):
    return datetime.fromtimestamp(v / 1000, tz=timezone.utc) if v else None


def quote_from_rest(sym, d) -> Quote:
    q = d.get("quote", {})
    ts = _ms(q.get("quoteTime") or q.get("tradeTime"))
    return Quote(sym, float(q.get("bidPrice") or 0), float(q.get("askPrice") or 0), float(q.get("lastPrice") or 0), ts)


class SchwabRest:
    """GET-only market data. The only non-market-data path allowed is /trader/v1/userPreference (streamer connection info)."""

    def __init__(self, auth: SchwabAuth, session=None, per_minute=100, sleep=time.sleep, clock=time.monotonic):
        self.auth, self.http, self.per_minute, self._sleep, self._clock = auth, session or requests.Session(), per_minute, sleep, clock
        self._hits = collections.deque()

    def _throttle(self):
        now = self._clock()
        while self._hits and now - self._hits[0] > 60:
            self._hits.popleft()
        if len(self._hits) >= self.per_minute:
            self._sleep(60 - (now - self._hits[0]) + 0.05)
        self._hits.append(self._clock())

    def get(self, path, params=None, retries=3):
        _check(path)
        for i in range(retries + 1):
            self._throttle()
            r = self.http.get(BASE + path, params=params, timeout=20,
                              headers={"Authorization": f"Bearer {self.auth.access_token()}", "Accept": "application/json"})
            if r.status_code in (429, 500, 502, 503, 504) and i < retries:
                self._sleep(2 ** i)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"Schwab GET {path} -> {r.status_code}: {r.text[:200]}")
            return r.json()

    def quotes(self, symbols, fields="quote,reference") -> dict:
        out = {}
        for i in range(0, len(symbols), 100):
            data = self.get("/marketdata/v1/quotes", {"symbols": ",".join(symbols[i:i + 100]), "fields": fields})
            out.update({s: quote_from_rest(s, d) for s, d in data.items() if isinstance(d, dict) and "quote" in d})
        return out

    def raw_quotes(self, symbols, fields=None) -> dict:
        params = {"symbols": ",".join(symbols)}
        if fields:
            params["fields"] = fields
        return self.get("/marketdata/v1/quotes", params)

    def price_history(self, symbol, *, period_type="day", period=None, frequency_type=None, frequency=None,
                      start=None, end=None, extended=True, previous_close=False) -> pd.DataFrame:
        epoch_ms = lambda x: None if x is None else int(pd.Timestamp(x).timestamp() * 1000)
        p = {"symbol": symbol, "periodType": period_type, "period": period, "frequencyType": frequency_type, "frequency": frequency,
             "startDate": epoch_ms(start), "endDate": epoch_ms(end),
             "needExtendedHoursData": str(extended).lower(), "needPreviousClose": str(previous_close).lower()}
        data = self.get("/marketdata/v1/pricehistory", {k: v for k, v in p.items() if v is not None})
        df = pd.DataFrame(data.get("candles", []))
        if df.empty:
            return df
        df.index = pd.to_datetime(df.pop("datetime"), unit="ms", utc=True).dt.tz_convert(NY)
        return df[["open", "high", "low", "close", "volume"]]

    def minute_bars(self, symbol, days=1, extended=True) -> pd.DataFrame:
        return self.price_history(symbol, period_type="day", period=days, frequency_type="minute", frequency=1, extended=extended)

    def daily_bars(self, symbol, years=1) -> pd.DataFrame:
        return self.price_history(symbol, period_type="year", period=years, frequency_type="daily", frequency=1, extended=False)

    def movers(self, index="EQUITY_ALL", sort="PERCENT_CHANGE_UP", frequency=0) -> list:
        return self.get(f"/marketdata/v1/movers/{index}", {"sort": sort, "frequency": frequency}).get("screeners", [])

    def market_hours(self, markets="equity", date=None) -> dict:
        params = {"markets": markets}
        if date:
            params["date"] = date
        return self.get("/marketdata/v1/markets", params)

    def user_preference(self) -> dict:
        return self.get("/trader/v1/userPreference")

    def streamer_info(self) -> dict:
        return self.user_preference()["streamerInfo"][0]
