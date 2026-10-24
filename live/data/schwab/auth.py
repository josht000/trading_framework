import base64
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import requests
from dotenv import load_dotenv

from live.config import DATA_DIR, ROOT

AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
REFRESH_LIFETIME = 7 * 86400
REFRESH_MARGIN = 3600
WARN_AGE = 5 * 86400
ACCESS_MARGIN = 60


class TokenExpired(RuntimeError):
    pass


@dataclass(frozen=True)
class SchwabSettings:
    client_id: str
    client_secret: str
    callback_url: str = "https://127.0.0.1:8182"
    token_path: Path = DATA_DIR / "schwab_token.json"


def load_settings() -> SchwabSettings:
    load_dotenv(ROOT / ".env")
    cid = os.getenv("SCHWAB_CLIENT_ID") or os.getenv("SCHWAB_APP_KEY") or os.getenv("SCHWAB_CLIENT_ED")
    secret = os.getenv("SCHWAB_CLIENT_SECRET") or os.getenv("SCHWAB_APP_SECRET")
    if not cid or not secret:
        raise RuntimeError("Schwab app key/secret missing: set SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET in .env")
    return SchwabSettings(cid, secret, os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182"),
                          Path(os.getenv("SCHWAB_TOKEN_PATH", str(DATA_DIR / "schwab_token.json"))))


class SchwabAuth:
    """OAuth2 authorization-code flow. Access tokens last 30 minutes and are refreshed automatically; the refresh token dies 7 days
    after the browser login, after which `scripts/schwab_login.py` must be run again."""

    def __init__(self, settings: SchwabSettings, session=None, clock=time.time):
        self.s, self.http, self.clock = settings, session or requests.Session(), clock
        self._lock, self._tok = threading.RLock(), None

    def authorize_url(self) -> str:
        return f"{AUTH_URL}?client_id={self.s.client_id}&redirect_uri={quote(self.s.callback_url, safe='')}"

    @staticmethod
    def code_from_redirect(url: str) -> str:
        q = parse_qs(urlparse(url.strip()).query)
        if "code" not in q:
            raise ValueError("no 'code' parameter in the pasted URL")
        return q["code"][0]

    def _post(self, data):
        basic = base64.b64encode(f"{self.s.client_id}:{self.s.client_secret}".encode()).decode()
        r = self.http.post(TOKEN_URL, data=data, timeout=30,
                           headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"})
        if r.status_code != 200:
            raise RuntimeError(f"Schwab token endpoint {r.status_code}: {r.text[:200]}")
        return r.json()

    def _stamp(self, tok):
        return {**tok, "expires_at": self.clock() + float(tok.get("expires_in", 1800))}

    def login(self, redirected_url: str):
        tok = self._post({"grant_type": "authorization_code", "code": self.code_from_redirect(redirected_url),
                          "redirect_uri": self.s.callback_url})
        with self._lock:
            self._tok = {"creation_timestamp": self.clock(), "token": self._stamp(tok)}
            self._save()

    def _save(self):
        p = Path(self.s.token_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._tok), encoding="utf-8")
        os.replace(tmp, p)

    def _load(self):
        p = Path(self.s.token_path)
        if not p.exists():
            raise TokenExpired(f"no Schwab token at {p}; run scripts/schwab_login.py")
        self._tok = json.loads(p.read_text(encoding="utf-8"))

    def refresh_age_days(self) -> float:
        with self._lock:
            if self._tok is None:
                self._load()
            return (self.clock() - self._tok["creation_timestamp"]) / 86400

    def needs_relogin_soon(self) -> bool:
        return self.refresh_age_days() * 86400 >= WARN_AGE

    def access_token(self) -> str:
        with self._lock:
            if self._tok is None:
                self._load()
            if self.clock() - self._tok["creation_timestamp"] >= REFRESH_LIFETIME - REFRESH_MARGIN:
                raise TokenExpired("Schwab refresh token is expired or about to expire; run scripts/schwab_login.py")
            tok = self._tok["token"]
            if tok["expires_at"] - self.clock() < ACCESS_MARGIN:
                new = self._post({"grant_type": "refresh_token", "refresh_token": tok["refresh_token"]})
                new.setdefault("refresh_token", tok["refresh_token"])
                self._tok["token"] = self._stamp(new)
                self._save()
            return self._tok["token"]["access_token"]
