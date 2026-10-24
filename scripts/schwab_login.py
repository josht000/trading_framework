"""
One-time (then weekly) Schwab OAuth login. The refresh token expires 7 days after login, so re-run this when `schwab_preflight.py`
or the app reports an expired token.

  python scripts/schwab_login.py

Prints the authorize URL; open it, log in and approve. The browser then fails to load the https://127.0.0.1:8182 callback (that is
expected): copy the full URL from the address bar and paste it here. Token is saved to data/live/schwab_token.json (gitignored).
"""

import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.data.schwab import SchwabAuth, load_settings


def main():
    auth = SchwabAuth(load_settings())
    url = auth.authorize_url()
    print("Open this URL, log in, approve, then paste the full redirected URL below:\n" + url + "\n")
    webbrowser.open(url)
    auth.login(input("Redirected URL: ").strip())
    print(f"saved token to {auth.s.token_path}; refresh token valid for 7 days")


if __name__ == "__main__":
    main()
