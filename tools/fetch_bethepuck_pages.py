"""Download selected Be The Puck play pages as plain text for cross-checking catalog claims.

Usage: python tools/fetch_bethepuck_pages.py <out_dir> <slug> [<slug> ...]
"""

import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}

if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for slug in sys.argv[2:]:
        r = requests.get(f"https://bethepuck.com/strategies/{slug}/", headers=HEADERS, timeout=60)
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "noscript"]):
            t.decompose()
        text = re.sub(r"\n\s*\n+", "\n", soup.get_text("\n"))
        (out / f"btp_{slug}.txt").write_text(text, encoding="utf-8")
        print(slug, r.status_code, len(text))
