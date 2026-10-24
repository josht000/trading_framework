import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from live.core.models import Quote

log = logging.getLogger("schwab.stream")

LEVELONE_FIELDS = "0,1,2,3,4,5,8,9,10,11,12,17,18,32,34,35"
CHART_FIELDS = "0,1,2,3,4,5,6,7"
QUOTE_FIELDS = {"1": "bid", "2": "ask", "3": "last", "4": "bid_size", "5": "ask_size", "8": "volume", "9": "last_size", "10": "high",
                "11": "low", "12": "prev_close", "17": "open", "18": "net_change", "32": "status", "34": "quote_ms", "35": "trade_ms"}
BAR_FIELDS = {"1": "open", "2": "high", "3": "low", "4": "close", "5": "volume", "6": "seq", "7": "time_ms"}


class StreamError(RuntimeError):
    pass


class LoginDenied(StreamError):
    pass


def _ms(v):
    return datetime.fromtimestamp(v / 1000, tz=timezone.utc) if v else None


class QuoteBook:
    """LEVELONE_EQUITIES is change-only: each message carries just the changed fields, so state is merged per symbol."""

    def __init__(self):
        self.state = {}

    def update(self, item: dict):
        d = self.state.setdefault(item["key"], {})
        for num, name in QUOTE_FIELDS.items():
            if num in item:
                d[name] = item[num]
        if "delayed" in item:
            d["delayed"] = item["delayed"]
        return d

    def quote(self, symbol) -> Quote | None:
        d = self.state.get(symbol)
        if not d or not d.get("last"):
            return None
        return Quote(symbol, float(d.get("bid") or 0), float(d.get("ask") or 0), float(d["last"]), _ms(max(d.get("quote_ms") or 0, d.get("trade_ms") or 0)))

    def delayed(self, symbol):
        return self.state.get(symbol, {}).get("delayed")


def parse_bar(item: dict) -> dict:
    b = {name: item[num] for num, name in BAR_FIELDS.items() if num in item}
    return {"symbol": item["key"], **b, "time": _ms(b.get("time_ms"))}


def login_request(info: dict, token: str, request_id=1) -> dict:
    return {"requests": [{"requestid": str(request_id), "service": "ADMIN", "command": "LOGIN",
                          "SchwabClientCustomerId": info["schwabClientCustomerId"], "SchwabClientCorrelId": info["schwabClientCorrelId"],
                          "parameters": {"Authorization": token, "SchwabClientChannel": info["schwabClientChannel"],
                                         "SchwabClientFunctionId": info["schwabClientFunctionId"]}}]}


def command_request(info: dict, service, command, keys, fields, request_id) -> dict:
    return {"requests": [{"requestid": str(request_id), "service": service, "command": command,
                          "SchwabClientCustomerId": info["schwabClientCustomerId"], "SchwabClientCorrelId": info["schwabClientCorrelId"],
                          "parameters": {"keys": ",".join(sorted(keys)), "fields": fields}}]}


class SchwabStreamer:
    """Streams level-one equity quotes and 1-minute chart bars. One connection per user is allowed by Schwab, so run a single instance.
    Reconnects with a fresh token and re-subscribes after any failure."""

    def __init__(self, rest, auth, on_quote=None, on_bar=None, connect=None, idle_timeout=45.0, sleep=asyncio.sleep):
        self.rest, self.auth, self.on_quote, self.on_bar, self.idle_timeout = rest, auth, on_quote, on_bar, idle_timeout
        self.connect, self._sleep = connect or self._default_connect, sleep
        self.book, self.quote_symbols, self.chart_symbols = QuoteBook(), set(), set()
        self.bars = {}
        self.info, self.ws, self._req, self._stop = None, None, 1, False
        self.last_message = None
        self.connected = asyncio.Event()

    @staticmethod
    def _default_connect(url):
        import websockets
        return websockets.connect(url, ping_interval=20, max_size=2 ** 22)

    def _next(self):
        self._req += 1
        return self._req

    async def _send(self, msg):
        await self.ws.send(json.dumps(msg))

    async def subscribe(self, quotes=(), charts=()):
        new_q, new_c = set(quotes) - self.quote_symbols, set(charts) - self.chart_symbols
        self.quote_symbols |= new_q
        self.chart_symbols |= new_c
        if self.ws is not None and self.connected.is_set():
            if new_q:
                await self._send(command_request(self.info, "LEVELONE_EQUITIES", "ADD", new_q, LEVELONE_FIELDS, self._next()))
            if new_c:
                await self._send(command_request(self.info, "CHART_EQUITY", "ADD", new_c, CHART_FIELDS, self._next()))

    async def unsubscribe(self, quotes=(), charts=()):
        gone_q, gone_c = set(quotes) & self.quote_symbols, set(charts) & self.chart_symbols
        self.quote_symbols -= gone_q
        self.chart_symbols -= gone_c
        if self.ws is not None and self.connected.is_set():
            if gone_q:
                await self._send(command_request(self.info, "LEVELONE_EQUITIES", "UNSUBS", gone_q, LEVELONE_FIELDS, self._next()))
            if gone_c:
                await self._send(command_request(self.info, "CHART_EQUITY", "UNSUBS", gone_c, CHART_FIELDS, self._next()))

    def handle(self, msg: dict):
        """Process one decoded streamer message."""
        for r in msg.get("response", []):
            code = r.get("content", {}).get("code")
            if r.get("command") == "LOGIN" and code != 0:
                raise LoginDenied(r["content"].get("msg", "login denied"))
            if code in (12, 30):
                raise StreamError(f"streamer closed the session: code {code} {r['content'].get('msg')}")
            if code == 19:
                log.warning("streamer symbol limit reached: %s", r["content"].get("msg"))
            elif code not in (0, 26, 27, 28, 29, None):
                log.warning("streamer response code %s: %s", code, r["content"].get("msg"))
        for d in msg.get("data", []):
            svc = d.get("service")
            for item in d.get("content", []):
                if svc == "LEVELONE_EQUITIES":
                    self.book.update(item)
                    if self.on_quote and (q := self.book.quote(item["key"])):
                        self.on_quote(q)
                elif svc == "CHART_EQUITY":
                    bar = parse_bar(item)
                    self.bars.setdefault(bar["symbol"], {})[bar.get("time_ms")] = bar
                    if self.on_bar:
                        self.on_bar(bar)

    async def _session(self):
        self.info = await asyncio.to_thread(self.rest.streamer_info)
        async with self.connect(self.info["streamerSocketUrl"]) as ws:
            self.ws = ws
            await self._send(login_request(self.info, await asyncio.to_thread(self.auth.access_token), self._next()))
            while True:
                raw = await asyncio.wait_for(ws.recv(), self.idle_timeout)
                self.last_message = datetime.now(timezone.utc)
                msg = json.loads(raw)
                self.handle(msg)
                if not self.connected.is_set() and any(r.get("command") == "LOGIN" for r in msg.get("response", [])):
                    self.connected.set()
                    if self.quote_symbols:
                        await self._send(command_request(self.info, "LEVELONE_EQUITIES", "SUBS", self.quote_symbols, LEVELONE_FIELDS, self._next()))
                    if self.chart_symbols:
                        await self._send(command_request(self.info, "CHART_EQUITY", "SUBS", self.chart_symbols, CHART_FIELDS, self._next()))

    async def run(self, max_backoff=60.0):
        delay = 1.0
        while not self._stop:
            try:
                await self._session()
            except asyncio.CancelledError:
                raise
            except LoginDenied as e:
                log.error("streamer login denied: %s", e)
            except Exception as e:
                log.warning("streamer session ended: %s: %s", type(e).__name__, e)
            finally:
                if self.connected.is_set():
                    delay = 1.0
                self.connected.clear()
                self.ws = None
            if self._stop:
                break
            await self._sleep(delay)
            delay = min(max_backoff, delay * 2)

    async def stop(self):
        self._stop = True
        if self.ws is not None:
            await self.ws.close()
