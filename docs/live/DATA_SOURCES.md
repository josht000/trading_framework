# Live data sources for the tuned score

The tuned score only uses these components (others have weight 0): `headroom` 0.5, `trend200` 1, `market` 1, `movers` 0.5, `delist_notice` 3, `late_filing` 1.5, `theme` 1.5. Plus the OTC hygiene filter and the H3 filters (`headroom >= 20%`, `above_sma200`). So `float`, `news`, `edgar_pos` and `dilution` need no live feed for this strategy.

Key finding from reading `src/backtest/features_extra.py`: the research "movers" feature does **not** use Benzinga's Movers API. It is built from:
1. `n_top_gainer_days`: days in the previous 5 sessions on which the stock was in the top 20 daily gainers, computed from **our own daily bars** (price >= $1, dollar volume >= $0.5M, whole listed universe including delisted names).
2. `n_mover_5d`: Alpaca news articles (Benzinga-sourced headlines) in the last 5 days whose headline matches mover-list phrases ("stocks moving", "movers", "trading higher", "mid-day", "52-week" and so on).

So we can reproduce it live without buying the Benzinga Movers API.

## EDGAR (delist notice 90d, late filing 30d)
- Research: per-symbol `data.sec.gov/submissions/CIK##########.json`, filtered to 8-K item codes and forms (`src/data/pit_dataset.fetch_edgar`, acceptance time in UTC).
- Live: same endpoint. It needs **no key**, is updated in real time as filings are disseminated, and each filing has `acceptanceDateTime`, `form` and `items`. Flags: 8-K item `3.01` within 90 days (delisting notice), forms `NT 10-K` / `NT 10-Q` within 30 days (late filing). Also 1.03 / 4.02 / offerings if the dilution weight is ever turned on.
- Symbol to CIK: `https://www.sec.gov/files/company_tickers.json` (refresh daily). Alpaca asset records are the second check on tickers.
- Rate and etiquette: SEC fair access is 10 requests per second across all machines and requires a declared `User-Agent` (name and email). We already read `EDGAR_USER_AGENT` from the environment. Use 5 per second max with retry and backoff.
- Query volume: only the post-filter shortlist (H3 + hygiene survivors, typically a few dozen names) at 15:55, with a per-CIK cache (a few minutes TTL). A nightly job can also pre-fetch the candidates seen that day. This is far below the rate limit.
- Fallback signal: the real-time Atom feed (`cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom`) is available if we want a push-like watchlist of fresh 3.01 / NT filings. It is not required for v1.
- Caveats: filings accepted after 17:30 ET are disseminated the next business day (not relevant at 15:55). Ticker changes and CIK reuse need the CIK mapping to be as-of-date aware. Items on an 8-K come from the `items` field and can be blank for older filings.

## Movers (top gainers + mover headlines)
- **Top-20 gainers history** (a daily nightly job, not at 15:55): fetch daily bars for the whole US equity universe for the latest completed sessions from Alpaca historical SIP (allowed on the free plan once data is more than 15 minutes old). Multi-symbol requests of about 1,000 symbols per call keep us inside 200 calls per minute. Apply the same filters as research (`prev close >= 1`, `close * volume >= 500k`) and store the top 20 per day in SQLite. At 15:55 the feature is a lookup of the previous 5 stored sessions. This also removes the need for any same-day volume.
- **Mover-list headlines**: Alpaca news endpoint (Benzinga source), already used by `src/data/news_client.py`. Query the shortlist for the last 5 days and apply the same `MOVERS` phrase list from `features_extra.py` (import it, do not copy it).
- **Benzinga Movers API** (optional, paid): `GET https://api.benzinga.com/api/v1/market/movers` with a token. It returns gainers, losers and most active with session filters (PRE_MARKET, REGULAR, AFTER_MARKET), price, change, volume, market cap, sector and float. It is a real-time alternative but is a different definition from the one used in the research (their ranking, their filters), so using it would require re-validating the weight. Not recommended for v1.
- Check: log both the live-built feature and a recomputed one from complete data the next day, and compare.

## Theme heat (weight 1.5, the largest soft weight)
- Research: count of other **breakout candidates** in the same NMF topic over the last 7 days. The topic comes from yfinance business descriptions, a TF-IDF + NMF model with 14 topics, plus the candidate list (`data/pit/candidates.parquet`) from the screener.
- Live needs two things we don't have yet:
  1. **Frozen topic model and symbol-to-topic map**: persist the fitted vectorizer and NMF (or just the map) and refit rarely (for example quarterly) with a version tag. New symbols without a description get topic -1 (no heat), same as research.
  2. **Daily candidate log**: every day, store all screener candidates (not only trades) in SQLite so the last 7 days are available. The screener must run daily even on days we don't trade. Bootstrap the first week by running the screener historically.
- Risk: yfinance is an unofficial scrape and can break or rate-limit. Descriptions only need refreshing for new symbols, so cache them. Alternatives: Alpaca asset data has no description, SimFin has industry classification (already in the repo).

## Other inputs
- `above_sma200`, `headroom`, atr_pivot ladder: daily history from Alpaca SIP (prior sessions), current price from the 15:55 quote. Uses `src/screening/resistance.py` directly.
- `iwm_day`, `spy_day` (market component): IEX last price vs previous close at 15:55. Acceptable on IEX because both are very liquid.
- `is_otc`: research used Massive's OTC reference. Live: Alpaca `assets` endpoint `exchange` field (OTC) and tradable flag, also drop symbols Alpaca doesn't list.
- Today's volume, relative volume and VWAP are required by the screener (not by the score). See PLAN section 2 for the IEX vs SIP plan.

## Parity test
A `scripts/replay_day.py` replays a historical day through the live feature code (reading cached data in place of the APIs) and diffs `components` and `score` against `reports/ranked_table.parquet`. Gate before paper trading.
