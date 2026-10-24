# Connecting to Charles Schwab (later broker / data source)

Sourced from Schwab developer docs, schwab-py docs and Schwab's extended-hours page, 2026-10-19. Verify the points marked (verify) during a spike before committing to Schwab.

## What exists
- **Schwab Trader API** (the TD Ameritrade API replacement): REST for accounts, positions, orders, transactions, plus quotes, price history and options chains, and a websocket streamer. Base: `https://api.schwabapi.com`.
- Developer portal products: "Accounts and Trading Production" and "Market Data". The schwab-py community recommends registering the first, which seems to cover everything schwab-py supports.
- **Individual vs Commercial**: we are Individual (own account, own app). Throttles: 120 non-order requests per minute per app; order requests (place/replace/cancel) 0-120 per minute per account, set at app registration. GET order requests are unthrottled. Use 120.
- **Data only**: per the repo rule in `AGENTS.md`, we never place orders through Schwab. Execution stays on Alpaca.

## Setup steps (manual, one time)
1. Create a developer account at developer.schwab.com and register an app (API product, order limit 120, callback URL such as `https://127.0.0.1:8182`). The callback must match exactly, including the trailing slash.
2. **Wait for Schwab approval** (days according to schwab-py docs). Start early.
3. Receive an app key and secret. Store them in `.env` (`SCHWAB_APP_KEY`, `SCHWAB_APP_SECRET`, `SCHWAB_CALLBACK_URL`, `SCHWAB_TOKEN_PATH`). Never commit them or the token file.
4. Run the OAuth login once in a browser (`schwab.auth.client_from_login_flow`) to create the token file.

## Authentication model and its operational cost
- OAuth2 three-legged flow. **Access token lasts 30 minutes** and is refreshed automatically. **The refresh token expires after 7 days** with no way to extend it. After 7 days you must redo the browser login.
- Consequences for an always-on bot:
  - A weekly re-login. Plan it for Sunday, and have the app alert when the token age exceeds about 5 days (`Client` exposes token age) and refuse to start a trading day on an expired token.
  - On a headless Raspberry Pi, do the login on the PC and copy the token file over, or tunnel the callback.
  - Single token file, single writer: never run two instances on the same token (refresh would invalidate one of them).
- Library: **schwab-py** (unofficial, community, async capable, supports quotes, price history, streaming and orders). Compatibility with Python 3.14 is unverified, so test it in the spike. A fallback is a thin client of our own using `requests-oauthlib`, which is what Lumibot does.

## Fit with the strategy and Nautilus
- No Nautilus adapter exists for Schwab either, so it would be a custom **data-only** adapter behind the same boundary as the Alpaca data client.
- **Data**: Schwab streams real-time quotes and trades for account holders (exchange agreements needed, verify) and serves daily and intraday price history. That would replace the free IEX limitation: full consolidated quotes and volume at 15:55. Check whether intraday history volume is consolidated and how far back minute bars go.
- **Sessions differ from Alpaca**: orders carry `session` (`NORMAL`, `AM`, `PM`, `SEAMLESS`). Schwab pre-market is 7:00 to 9:25 ET and after-hours is 4:05 to 8:00 PM ET (the page lists 4:05, so there is a gap right after the close). The backtest assumed 04:00 pre-market bars and 16:00 to 16:45 after-hours exits.
  - Entry at 15:55 as a regular-session limit is unaffected.
  - After-hours stop/target watching can only act from 16:05.
  - The pre-market +25% target can only rest from 07:00 (versus 04:00), so some gap-ups captured in 04:00 to 07:00 would be missed. Quantify from the minute data in `data/pit` before choosing Schwab as the execution venue.
- Extended-hours orders are limit orders only (as at Alpaca).
- Throttles: 120 non-order requests per minute is enough for a shortlist and a few held symbols, but the full-universe daily scan should keep using other data (Alpaca SIP history or bulk files).

## Plan
- **Schwab is data only. No orders, ever** (rule in `AGENTS.md`). Alpaca (paper) remains the only execution venue.
- Enforcement: the Schwab adapter implements the `DataSource` interface only, with no `Broker` implementation. The client wrapper exposes only quote, price-history, movers and streamer calls and never imports order-builder code. A test asserts that no Schwab order endpoint is reachable from our wrapper. Register only the Market Data product, which also avoids granting order scope.
- Spike (`scripts/schwab_preflight.py`, after approval): login, token refresh, quote, price history for a few symbols, streamer login.

## Status: client built (data only), not yet run against Schwab
Code: `live/data/schwab/` (`auth.py`, `rest.py`, `stream.py`), scripts `scripts/schwab_login.py` and `scripts/schwab_preflight.py`, tests `tests/live/test_schwab_client.py` (offline, fake HTTP and websocket).
- **Auth**: manual paste flow (no local HTTPS listener). Token in `data/live/schwab_token.json`. Auto refresh of the 30-minute access token; `TokenExpired` after about 7 days; `needs_relogin_soon()` at 5 days.
- **REST** (GET only, path whitelist, 100 requests per minute throttle, retry on 429/5xx): quotes, price history (minute and daily, extended hours), movers (top 10 per index/exchange), market hours.
- **Streamer**: LEVELONE_EQUITIES (change-only fields merged in `QuoteBook`) and CHART_EQUITY 1-minute bars, auto reconnect with fresh token and re-subscribe, idle watchdog.
- **Gap found in the docs**: the streamer takes its socket URL, customer ID, correlation ID, channel and function ID from `GET /trader/v1/userPreference`, which is part of the Accounts and Trading product, not Market Data Production (the market data spec has no such endpoint). The client allows only that single read-only path outside `/marketdata/v1/`. If the Market Data token is refused there, the streamer cannot start; REST quotes and minute bars still work. Next step is to run `scripts/schwab_preflight.py` after the first login and see.
- **Env names**: `.env` currently has `SCHWAB_CLIENT_ED` (typo?). The loader reads `SCHWAB_CLIENT_ID` first and falls back to `SCHWAB_CLIENT_ED`. Rename it when convenient.
