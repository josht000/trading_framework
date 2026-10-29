"""Extract every URL cited in the strategy catalog JSON and report HTTP status and redirects."""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs" / "strategies" / "Trading Strategy Catalog (JSON for coding agent).json"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}


def urls():
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    found = {}
    for s in data["strategies"]:
        for u in re.findall(r"\((https?://[^)\s]+)\)", s.get("sources") or ""):
            found.setdefault(u, []).append(s["id"])
    return found


def check(u):
    try:
        r = requests.get(u, headers=HEADERS, timeout=25, allow_redirects=True)
        return u, r.status_code, r.url if r.url != u else "", len(r.text)
    except Exception as exc:
        return u, "ERR", type(exc).__name__, 0


if __name__ == "__main__":
    found = urls()
    with ThreadPoolExecutor(8) as ex:
        for u, code, final, n in sorted(ex.map(check, found), key=lambda x: str(x[1])):
            print(f"{code}\t{n}\t{','.join(found[u])}\t{u}\t{final}")
    sys.exit(0)
