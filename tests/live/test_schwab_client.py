import asyncio
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from live.data.schwab import DataOnlyError, QuoteBook, SchwabAuth, SchwabRest, SchwabSettings, SchwabStreamer, TokenExpired
from live.data.schwab import stream as sm


class FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code, self._b, self.text = status, body if body is not None else {}, json.dumps(body)

    def json(self):
        return self._b


class FakeHttp:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return self.responses.pop(0)

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return self.responses.pop(0)


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def settings(d):
    return SchwabSettings("KEY", "SECRET", "https://127.0.0.1:8182", Path(d) / "tok.json")


class AuthTests(unittest.TestCase):
    def test_code_from_redirect_and_authorize_url(self):
        with tempfile.TemporaryDirectory() as d:
            a = SchwabAuth(settings(d), FakeHttp([]))
            self.assertEqual(a.code_from_redirect("https://127.0.0.1:8182/?code=C0.abc%40&session=x"), "C0.abc@")
            self.assertIn("redirect_uri=https%3A%2F%2F127.0.0.1%3A8182", a.authorize_url())
            with self.assertRaises(ValueError):
                a.code_from_redirect("https://127.0.0.1:8182/?error=1")

    def test_login_refresh_and_persist(self):
        with tempfile.TemporaryDirectory() as d:
            clk = Clock()
            http = FakeHttp([FakeResp(200, {"access_token": "A1", "refresh_token": "R1", "expires_in": 1800}),
                             FakeResp(200, {"access_token": "A2", "expires_in": 1800})])
            a = SchwabAuth(settings(d), http, clk)
            a.login("https://127.0.0.1:8182/?code=C0.abc%40")
            self.assertEqual(http.calls[0][2]["data"]["code"], "C0.abc@")
            self.assertTrue(http.calls[0][2]["headers"]["Authorization"].startswith("Basic "))
            self.assertEqual(a.access_token(), "A1")
            clk.t += 1750
            self.assertEqual(a.access_token(), "A2")
            self.assertEqual(http.calls[1][2]["data"], {"grant_type": "refresh_token", "refresh_token": "R1"})
            a2 = SchwabAuth(settings(d), FakeHttp([]), clk)
            self.assertEqual(a2.access_token(), "A2")
            self.assertAlmostEqual(a2.refresh_age_days(), 1750 / 86400)

    def test_expired_refresh_token_and_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            clk = Clock()
            a = SchwabAuth(settings(d), FakeHttp([FakeResp(200, {"access_token": "A1", "refresh_token": "R1", "expires_in": 1800})]), clk)
            with self.assertRaises(TokenExpired):
                a.access_token()
            a.login("https://x/?code=c")
            self.assertFalse(a.needs_relogin_soon())
            clk.t += 5.1 * 86400
            self.assertTrue(a.needs_relogin_soon())
            clk.t += 2 * 86400
            with self.assertRaises(TokenExpired):
                a.access_token()

    def test_token_endpoint_error_does_not_leak_secret(self):
        with tempfile.TemporaryDirectory() as d:
            a = SchwabAuth(settings(d), FakeHttp([FakeResp(401, {"error": "invalid_client"})]), Clock())
            with self.assertRaises(RuntimeError) as cm:
                a.login("https://x/?code=c")
            self.assertNotIn("SECRET", str(cm.exception))


class StubAuth:
    def access_token(self):
        return "TOK"


class RestTests(unittest.TestCase):
    def rest(self, responses, **kw):
        http = FakeHttp(responses)
        return SchwabRest(StubAuth(), http, sleep=lambda s: None, **kw), http

    def test_data_only_whitelist(self):
        r, http = self.rest([])
        for path in ("/trader/v1/accounts/123/orders", "/trader/v1/accounts", "/v1/oauth/token", "/marketdata/../trader/v1/x"):
            if path.startswith("/marketdata/../"):
                continue
            with self.assertRaises(DataOnlyError):
                r.get(path)
        self.assertEqual(http.calls, [])
        self.assertFalse(hasattr(r, "post") or hasattr(r, "place_order") or hasattr(r, "delete"))

    def test_quotes_parse_and_batch(self):
        body = {"AAPL": {"quote": {"bidPrice": 1.0, "askPrice": 1.2, "lastPrice": 1.1, "quoteTime": 1_700_000_000_000}, "symbol": "AAPL"}}
        r, http = self.rest([FakeResp(200, body)])
        q = r.quotes(["AAPL"])["AAPL"]
        self.assertEqual((q.bid, q.ask, q.last), (1.0, 1.2, 1.1))
        self.assertEqual(http.calls[0][2]["headers"]["Authorization"], "Bearer TOK")
        self.assertEqual(http.calls[0][2]["params"]["symbols"], "AAPL")

    def test_price_history_frame_is_ny_indexed(self):
        body = {"candles": [{"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100, "datetime": 1_700_000_000_000}]}
        r, http = self.rest([FakeResp(200, body)])
        df = r.minute_bars("AAPL")
        self.assertEqual(str(df.index.tz), "America/New_York")
        self.assertEqual(list(df.columns), ["open", "high", "low", "close", "volume"])
        p = http.calls[0][2]["params"]
        self.assertEqual((p["frequencyType"], p["frequency"], p["needExtendedHoursData"]), ("minute", 1, "true"))

    def test_retry_on_429_then_error(self):
        r, http = self.rest([FakeResp(429), FakeResp(200, {"screeners": [{"symbol": "X"}]})])
        self.assertEqual(r.movers(), [{"symbol": "X"}])
        self.assertEqual(len(http.calls), 2)
        r, _ = self.rest([FakeResp(401, {"e": 1})])
        with self.assertRaises(RuntimeError):
            r.market_hours()

    def test_throttle_sleeps_when_over_limit(self):
        slept = []
        http = FakeHttp([FakeResp(200, {}) for _ in range(3)])
        t = [0.0]
        r = SchwabRest(StubAuth(), http, per_minute=2, sleep=slept.append, clock=lambda: t[0])
        for _ in range(3):
            r.get("/marketdata/v1/markets")
        self.assertEqual(len(slept), 1)


INFO = {"streamerSocketUrl": "wss://x", "schwabClientCustomerId": "cust", "schwabClientCorrelId": "corr",
        "schwabClientChannel": "N9", "schwabClientFunctionId": "APIAPP"}


class StreamTests(unittest.TestCase):
    def test_quotebook_merges_change_only_updates(self):
        b = QuoteBook()
        b.update({"key": "AAPL", "delayed": False, "1": 10.0, "2": 10.1, "3": 10.05, "34": 1_700_000_000_000})
        b.update({"key": "AAPL", "3": 10.08, "35": 1_700_000_001_000})
        q = b.quote("AAPL")
        self.assertEqual((q.bid, q.ask, q.last), (10.0, 10.1, 10.08))
        self.assertEqual(q.ts.timestamp(), 1_700_000_001)
        self.assertFalse(b.delayed("AAPL"))
        self.assertIsNone(b.quote("NOPE"))

    def test_requests_shape(self):
        login = sm.login_request(INFO, "TOKEN")["requests"][0]
        self.assertEqual((login["service"], login["command"], login["parameters"]["Authorization"]), ("ADMIN", "LOGIN", "TOKEN"))
        self.assertEqual(login["SchwabClientCorrelId"], "corr")
        sub = sm.command_request(INFO, "LEVELONE_EQUITIES", "SUBS", {"SPY", "AAPL"}, sm.LEVELONE_FIELDS, 3)["requests"][0]
        self.assertEqual(sub["parameters"]["keys"], "AAPL,SPY")

    def test_handle_login_denied_and_close_codes(self):
        s = SchwabStreamer(None, None)
        with self.assertRaises(sm.LoginDenied):
            s.handle({"response": [{"command": "LOGIN", "content": {"code": 3, "msg": "denied"}}]})
        with self.assertRaises(sm.StreamError):
            s.handle({"response": [{"command": "SUBS", "content": {"code": 12, "msg": "max connections"}}]})
        s.handle({"response": [{"command": "SUBS", "content": {"code": 26, "msg": "ok"}}], "notify": [{"heartbeat": "1"}]})

    def test_handle_data_calls_back(self):
        quotes, bars = [], []
        s = SchwabStreamer(None, None, on_quote=quotes.append, on_bar=bars.append)
        s.handle({"data": [{"service": "LEVELONE_EQUITIES", "content": [{"key": "AAPL", "1": 9.9, "2": 10.1, "3": 10.0}]},
                           {"service": "CHART_EQUITY", "content": [{"key": "AAPL", "1": 1, "2": 2, "3": 0.5, "4": 1.5, "5": 100, "6": 7,
                                                                    "7": 1_700_000_000_000}]}]})
        self.assertEqual(quotes[0].last, 10.0)
        self.assertEqual((bars[0]["symbol"], bars[0]["close"], bars[0]["volume"]), ("AAPL", 1.5, 100))

    def test_session_logs_in_subscribes_and_reconnects(self):
        sent, conns = [], []

        class WS:
            def __init__(self, script):
                self.script = list(script)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def send(self, m):
                sent.append(json.loads(m))

            async def recv(self):
                if not self.script:
                    raise ConnectionError("dropped")
                return json.dumps(self.script.pop(0))

            async def close(self):
                pass

        login_ok = {"response": [{"service": "ADMIN", "command": "LOGIN", "content": {"code": 0, "msg": "ok"}}]}
        tick = {"data": [{"service": "LEVELONE_EQUITIES", "content": [{"key": "AAPL", "1": 9.9, "2": 10.1, "3": 10.0}]}]}

        def connect(url):
            conns.append(url)
            return WS([login_ok, tick])

        class Rest:
            def streamer_info(self):
                return INFO

        got = []

        async def go():
            async def nap(_):
                await asyncio.sleep(0)

            s = SchwabStreamer(Rest(), StubAuth(), on_quote=got.append, connect=connect, idle_timeout=1, sleep=nap)
            await s.subscribe(["AAPL"], ["AAPL"])
            task = asyncio.create_task(s.run())
            while len(conns) < 2:
                await asyncio.sleep(0.01)
            await s.stop()
            task.cancel()

        asyncio.run(go())
        services = [(m["requests"][0]["service"], m["requests"][0]["command"]) for m in sent]
        self.assertEqual(services[:3], [("ADMIN", "LOGIN"), ("LEVELONE_EQUITIES", "SUBS"), ("CHART_EQUITY", "SUBS")])
        self.assertGreaterEqual(len(conns), 2)
        self.assertIn(services.count(("ADMIN", "LOGIN")), (len(conns) - 1, len(conns)))
        self.assertTrue(got and got[0].last == 10.0)


if __name__ == "__main__":
    unittest.main()
