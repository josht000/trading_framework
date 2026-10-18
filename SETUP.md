# Setup Instructions

## Python setup in the Pytorch 2.8 container from RunPod

```bash
cd /workspace
git clone git@github.com:josht000/trading_framework.git && cd trading_framework

python -m virtualenv --system-site-packages "$HOME/.venvs/trading"
source "$HOME/.venvs/trading/bin/activate"
python -m pip install -r /workspace/trading_framework/requirements.txt
```


## Alpaca API Setup

1. Sign up for a free Alpaca account at https://alpaca.markets/
2. Get your API key and secret from the dashboard
3. Create a `.env` file in the project root with the following content:

```
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here
ALPACA_BASE_URL=https://data.alpaca.markets

# Point-in-time context sources
SIMFIN_API_KEY=your_simfin_key_here   # SimFin — as-of market cap / fundamentals
FMP_API_KEY=your_fmp_key_here         # optional — FMP social sentiment bonus flag
```

Note: The `.env` file is already in `.gitignore` to protect your credentials. The Alpaca keys are also used for the Alpaca news feed (news catalyst). `SIMFIN_API_KEY` comes from a free SimFin account (https://simfin.com/); `FMP_API_KEY` is optional — social sentiment fails soft without it.

## Configuration

Backtest settings are read from environment variables (set them in `.env`):

- `BACKTEST_SYMBOLS`: Comma-separated broad universe (default: `SOFI`)
- `BACKTEST_START_DATE`: Backtest start date (default: `2024-01-01`)
- `BACKTEST_END_DATE`: Backtest end date (default: yesterday)
- `BACKTEST_INITIAL_CASH`: Starting capital (default: `100000`)
- `BACKTEST_COMMISSION`: Commission per trade (default: `0.001`)

Point-in-time context sources and screening:

- `SIMFIN_API_KEY`: SimFin API key — enables as-of market-cap screening (`src/data/simfin_store.py`)
- `FMP_API_KEY`: Financial Modeling Prep API key (optional) — enables the social-sentiment bonus flag (`src/data/sentiment_client.py`); fails soft without it
- `REQUIRE_NEWS`: Require a news catalyst to enter (default: `1`; set `0` to disable)
- `REQUIRE_MARKET_CAP`: Require market-cap data to be present and <= $2B (default: `0`; missing data passes unless set to `1`)
- `NEWS_LOOKBACK_DAYS`: Days to look back for news articles (default: `3`)
- `MIN_NEWS_ARTICLES`: Minimum article count within the lookback window (default: `1`)

The default is intentionally a single symbol for initial validation. Larger universes can be comma-separated (e.g. `BACKTEST_SYMBOLS=SOFI,PLTR,F`) but will download much more 1-minute data.

**Note**: The first run downloads the SimFin bulk zip (~hundreds of MB) into `data/simfin/` and caches it for subsequent runs.

## Running the Application

Once you have your Alpaca credentials set up, run:

```bat
C:\data\envs\trading_framework\Scripts\python.exe main.py
```

The application will:
1. Build the backtest universe from `BACKTEST_SYMBOLS` (comma-separated, default `SOFI`)
2. Download split-adjusted SIP daily and 1-minute bars from Alpaca, Alpaca news article counts, SimFin bulk share prices (cached in `data/simfin/`), and — if `FMP_API_KEY` is set — FMP social sentiment
3. Screen each date point-in-time at 15:55 America/New_York, using only prior daily bars, that session's minute bars through 15:55, as-of market cap (<= $2B), and the news catalyst; social surge is recorded as a non-blocking bonus flag
4. Backtest the signals with vectorbt (entry on the 15:55 bar; exit on the first regular-session bar of the next trading date)
5. Display performance metrics

## Important Note on Universe Selection

**Survivorship Bias Warning**: Survivorship bias is not solved. `BACKTEST_SYMBOLS` is a plain configured symbol list — each symbol is screened point-in-time at each date, but the list itself still needs point-in-time membership and delisting data for a production-grade universe. For production backtesting you should:

1. Use a historical universe database (e.g., from vendors like FirstRate, Kibot, or MarketParquet)
2. Or build your own historical universe by tracking which stocks met your criteria at each date
3. The strategy itself filters stocks based on historical price/volume data, but the initial universe selection should also be historically accurate

For initial exploration and validation, the default single-symbol universe is acceptable.
