"""Download every URL cited in the strategy catalog JSON as plain text for offline claim checking.

Usage: python tools/fetch_catalog_sources.py <out_dir>
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_catalog_urls import HEADERS, urls


def slug(u):
    return re.sub(r"[^A-Za-z0-9]+", "_", u.split("://", 1)[1])[:120]


def fetch(args):
    u, out = args
    for attempt in range(3):
        try:
            r = requests.get(u, headers=HEADERS, timeout=(15, 60))
            if "pdf" in r.headers.get("content-type", ""):
                (out / f"{slug(u)}.pdf").write_bytes(r.content)
                return u, r.status_code, "pdf"
            soup = BeautifulSoup(r.text, "html.parser")
            for t in soup(["script", "style", "nav", "footer", "noscript"]):
                t.decompose()
            text = re.sub(r"\n\s*\n+", "\n", soup.get_text("\n"))
            (out / f"{slug(u)}.txt").write_text(text, encoding="utf-8")
            return u, r.status_code, len(text)
        except Exception as exc:
            err = type(exc).__name__
    return u, "ERR", err


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as ex:
        for row in ex.map(fetch, [(u, out) for u in urls()]):
            print(*row)
