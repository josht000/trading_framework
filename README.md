# Trading Framework

A Python-based automated trading framework for backtesting, simulating, and live trading strategies.

## Features

- **Data Ingestion**: Fetch split-adjusted SIP daily and 1-minute bars from Alpaca (free multi-year history)
- **Point-in-Time Context Sources**: Alpaca news article counts (news catalyst), SimFin fundamentals/market cap, and FMP social sentiment (non-blocking bonus flag)
- **Universe Selection**: Configurable symbol list (`BACKTEST_SYMBOLS`, default `SOFI`) as the broad backtest universe
- **Point-in-Time Screening**: Each date is screened at 15:55 ET using only prior daily bars, that session's minute bars through 15:55, and as-of context data
- **Strategy Implementation**: Humbled Trader Bullish Overnight strategy
- **Backtesting**: Vectorbt-based backtesting engine for fast strategy evaluation
- **Multi-Symbol Support**: Backtest multiple symbols simultaneously
- **Resistance Research**: Interactive detector tooling (`research_resistance.py`), a hand-labeled chart dataset (`data/resistance_dataset/`), and a browser labeling server (`label_charts.py`)
- **VLM Benchmark**: Harness for scoring vision-language models on resistance-level reading (`vlm_benchmark.py`)
- **Point-in-time dataset**: Survivorship-aware 2-year daily + candidate-day 1-minute dataset built from free sources (`build_pit_dataset.py`, `data/pit/`) — delisted names included
- **Realistic PIT backtest**: Parallel per-symbol screening with a cached `atr_pivot` ladder, volume-limited fills, slippage, exit-rule variants, and a locked final test set (`main.py`, `iterate.py`)

## Architecture

```
trading_framework/
├── main.py                   # Backtest entry point (point-in-time mode by default when data/pit exists)
├── build_pit_dataset.py      # Build data/pit: reference, daily bars, candidates, candidate-day minute bars
├── iterate.py                # Offline strategy iteration on per-trade tables (dev set) + one-shot test-set scoring
├── chart_trades.py           # Best/worst trade charts: daily + 5-min with VWAP, atr_pivot levels, entry/exit
├── research_resistance.py    # Detector research CLI (ladder/chart/eval/labels/pick/config)
├── eval_pivot_resistance.py  # Score a detector on a resistance_dataset split
├── sweep_resistance.py       # Param-sweep detectors on train/val, save best scores
├── label_charts.py           # Browser labeling server for dataset charts
├── vlm_benchmark.py          # VLM benchmark CLI (prepare/models/run/report)
├── src/
│   ├── data/                 # Data ingestion modules
│   │   ├── alpaca_client.py    # Alpaca API client (daily/minute bars)
│   │   ├── news_client.py      # Alpaca news client (per-symbol daily article counts)
│   │   ├── simfin_store.py     # SimFin bulk store (as-of market cap / fundamentals)
│   │   ├── pit_dataset.py      # Point-in-time dataset builders (Massive / MarketParquet / Alpaca)
│   │   ├── sentiment_client.py # FMP social sentiment client (bonus flag, fails soft)
│   │   └── finviz_screener.py  # Finviz screener (not used by historical backtests)
│   ├── screening/            # Screening and level detectors
│   │   ├── historical.py       # Point-in-time historical screening
│   │   ├── resistance.py       # ATR-scaled pivot zone detector (support/resistance)
│   │   ├── pivot_resistance.py # Simple swing-high detector for the labeled dataset
│   │   ├── cluster_levels.py   # ZigZag / rolling-extreme pivots + agglomerative clustering
│   │   ├── sr_finder.py        # Ports of albertsl's finder: deriv extrema, MeanShift, pivots
│   │   └── volume_profile.py   # Volume-at-price high-volume-node detector
│   ├── strategies/           # Trading strategies
│   │   └── humbled_bullish_overnight.py
│   ├── backtest/             # Backtesting
│   │   ├── engine.py           # vectorbt engine (symbol-list mode)
│   │   ├── pit_runner.py       # PIT per-symbol worker: signals, fill model, exit variants, features; ladder cache
│   │   ├── exit_variants.py    # Named exit-rule specs (after-hours stop/target, pre-market target, ...)
│   │   └── pit_metrics.py      # Portfolio metrics, de-duplication of renamed tickers, sizing modes
│   ├── research/             # Research tooling behind research_resistance.py
│   │   ├── chart_levels.py     # Interactive plotly level charts
│   │   ├── chart_png.py        # Static matplotlib PNG renderer (VLM packets)
│   │   ├── config_io.py        # Detector configs as editable JSON
│   │   ├── data.py             # Cached daily-bar loader (data/cache/daily/)
│   │   ├── labels.py           # Manual label CSV store + scoring
│   │   ├── pick_charts.py      # Detector-independent chart-date picker
│   │   └── resistance_eval.py  # Outcome-based detector evaluation vs baselines
│   ├── vlm/                  # VLM benchmark internals
│   │   ├── packets.py          # Packet building from labeled charts
│   │   ├── models.py           # Model registry (data/vlm/models.json)
│   │   ├── providers.py        # HTTP adapters (Anthropic/OpenAI/OpenRouter/mock)
│   │   ├── runner.py           # Cached runner with spend cap
│   │   ├── prompt.py           # Prompt variants + answer parsing
│   │   ├── fewshot.py          # Worked examples from labeled splits
│   │   ├── scoring.py          # ATR-tolerance level matching
│   │   ├── estimate.py         # Token/cost estimates
│   │   └── report.py           # Leaderboard + HTML review page
│   └── utils/                # Utility functions
│       └── helpers.py
├── tests/                    # unittest suite
├── data/
│   ├── pit/                  # Point-in-time dataset (gitignored; see "Point-in-time dataset")
│   ├── charts/trades/<variant>/  # Best/worst trade charts per exit variant (index.html in each)
│   ├── cache/daily/          # Cached daily bars (gitignored)
│   ├── resistance_dataset/   # Hand-labeled chart dataset (train/val/test)
│   ├── simfin/               # SimFin bulk data cache (gitignored)
│   └── vlm/                  # VLM packets, model registry, run results
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.11 or higher (tested with Python 3.14)
- Anaconda or Miniconda

### Setup

1. Clone the repository
2. Create a conda environment:
```bash
conda create -n trading_framework python=3.14 -y
conda activate trading_framework
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
This includes `simfin==1.0.2`, used for as-of market cap and fundamentals. On first run the SimFin bulk zip (~hundreds of MB) is downloaded into `data/simfin/` and cached.

4. Create a `.env` file with your Alpaca API credentials (see [SETUP.md](SETUP.md)). Optionally add `SIMFIN_API_KEY` (market-cap screening) and `FMP_API_KEY` (social-sentiment bonus flag).

## Alpaca API Setup

1. Sign up for a free Alpaca account at https://alpaca.markets/
2. Get your API key and secret from the dashboard
3. Add them to your `.env` file:
```
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here
ALPACA_BASE_URL=https://data.alpaca.markets
```

**Note**: Alpaca's free plan provides:
- Historical 1-minute bars back to ~2016
- Real-time IEX data (limited to a small slice of volume)
- Full SIP data is blocked for the last 15 minutes

## Usage

### Quick Start

Run the backtest application:

```bat
C:\data\envs\trading_framework\Scripts\python.exe main.py
```

This builds the universe from `BACKTEST_SYMBOLS`, downloads split-adjusted SIP daily and 1-minute bars from Alpaca, screens each date point-in-time, backtests the resulting signals with vectorbt, and prints performance metrics.

### Configuration

Backtest settings are read from environment variables (set them in `.env`):

- `BACKTEST_SYMBOLS`: Comma-separated universe (default: `SOFI`)
- `BACKTEST_START_DATE`: Backtest start date (default: `2024-01-01`)
- `BACKTEST_END_DATE`: Backtest end date (default: yesterday)
- `BACKTEST_INITIAL_CASH`: Starting capital (default: `100000`)
- `BACKTEST_COMMISSION`: Commission per trade (default: `0.001`)

### Basic Usage

```python
from src.data.alpaca_client import AlpacaDataClient
from src.strategies.humbled_bullish_overnight import HumbledBullishOvernight
from src.backtest.engine import MultiSymbolBacktest

# Initialize Alpaca client
client = AlpacaDataClient()

# Fetch historical data (split-adjusted SIP bars)
daily_data = client.fetch_daily_bars(
    symbols=['SOFI'],
    start='2024-01-01',
    end='2024-01-31',
)
minute_data = client.fetch_minute_bars(
    symbols=['SOFI'],
    start='2024-01-01',
    end='2024-01-31',
)

# Initialize strategy
strategy = HumbledBullishOvernight()

# Generate point-in-time signals (entry on the 15:55 bar,
# exit on the first regular-session bar of the next trading date)
signals = strategy.apply_strategy(daily_data['SOFI'], minute_data['SOFI'])

# Backtest
backtest = MultiSymbolBacktest(initial_cash=100_000, commission=0.001)
portfolio = backtest.run_multi_symbol_backtest(
    price_data=minute_data,
    entry_signals={'SOFI': signals['entry']},
    exit_signals={'SOFI': signals['exit']},
    freq='1min',
)

# View results
print(portfolio.stats())
```

### Universe Selection

Backtests start from `BACKTEST_SYMBOLS` — a comma-separated list in `.env` (default: `SOFI`) — as the broad universe. The default is intentionally a single symbol for initial validation; larger universes can be comma-separated (e.g. `BACKTEST_SYMBOLS=SOFI,PLTR,F`) but will download much more 1-minute data.

The historical backtest does not screen against Finviz (which only shows current data and would bias historical results) and does not use a fixed curated universe — each configured symbol is screened independently at each date.

**Important**: Survivorship bias is not solved. Configured symbols still need point-in-time membership and delisting data for a production-grade universe. For accurate backtesting, use a historical universe database (e.g., from FirstRate, Kibot, or MarketParquet) to know which stocks existed and met your criteria at each historical date.

## Humbled Trader Bullish Overnight Strategy

This strategy identifies stocks with bullish momentum for overnight holds.

### Implemented Entry Criteria

All criteria are evaluated once per date at 15:55 America/New_York, using only prior daily bars and that session's minute bars through 15:55:

- Price between $2 and $20
- Prior 20-day average volume >= 500,000 shares
- Current cumulative relative volume >= 2x the prior 20-day average
- Close higher than the prior close
- Session open above the prior close
- Breakout above resistance (prior 20-day high, or an `atr_pivot` level; see below)
- Close above the session VWAP
- Market cap <= $2B as-of the screen date (SimFin: Close * Shares Outstanding). Controlled by `REQUIRE_MARKET_CAP` (0/1); missing data passes unless required
- News catalyst: >= `MIN_NEWS_ARTICLES` articles within `NEWS_LOOKBACK_DAYS` (defaults: 1 article / 3 days), from Alpaca news per-day counts. `REQUIRE_NEWS=0` disables
- Social surge is recorded as a bonus flag only — it never blocks an entry

### Not Yet Enforced

Historical NASDAQ membership and sector hype are not available from the current data sources and are explicitly not yet enforced. Social sentiment is implemented as a non-blocking bonus flag (`social_surge`) rather than a hard entry criterion.

Breakout resistance has two methods (`RESISTANCE_METHOD`, default `atr_pivot` in `main.py`):

- `rolling`: close above the prior 20-day high.
- `atr_pivot`: the 15:55 price is above an `atr_pivot` level (`src/screening/resistance.py`, sweep-tuned config `ATR_PIVOT_CONFIG` in `historical.py`) that sat above the prior close. Levels come only from daily bars through the prior session, with pivots usable `pivot_right` bars after they form, so nothing leaks from the future. `MIN_BREAKOUT_ATR` adds an optional buffer above the level.

### Entry Timing
- Enter on the 15:55 ET bar when all criteria pass

### Exit Timing
- Default: exit on the first regular-session bar of the next trading date
- PIT mode adds named exit variants (`src/backtest/exit_variants.py`), all evaluated on the same entries:
  - after-hours stop/target: during 16:00 to 16:00+`window_min` ET, exit at the first 1-minute close that is `stop` below or `target` above the entry price (v2: 45 min, -12% / +50%)
  - pre-market target (`pm_target`): if no after-hours exit fired, take profit at the first next-day pre-market bar `pm_target` above entry (pm_target25_v2: +25%)
  - pre-market stop (`pm_stop`), ATR-scaled stop, N-bar stop confirmation (all tested; see lessons)

## Point-in-time dataset (`data/pit/`, gitignored)

A fixed ticker list is survivor-biased, so PIT mode screens every symbol that was tradable on each date, including ones later delisted. Built by `build_pit_dataset.py` (all steps resumable):

| step | command | output |
|---|---|---|
| ticker reference | `python build_pit_dataset.py reference` | `reference.parquet`: Massive reference for every stock ticker, active and inactive, with `delisted_utc`, `type`, `list_date`. Free plan; ~12.5 s per call (5 calls/min) |
| daily bars | `... alpaca-daily` | `alpaca_daily/part_*.parquet`: split-adjusted SIP daily bars for every ticker listed any time since 2024-09-01 (types such as warrants, rights, units, ETFs, funds, preferred dropped). Alpaca returns bars for delisted tickers up to their last trading day |
| other sources | `... massive`, `... mp` | `massive_daily/DATE.parquet` (grouped daily; free window ~2 years, slow) and `mp_daily/DATE.parquet` (MarketParquet free sample, last 365 days). Used only to fill gaps / cross-check |
| merge | `... build` | `daily_pit.parquet`: `date, symbol, open, high, low, close, volume, trade_count, vwap, src, type, primary_exchange, delisted_utc` (Alpaca first, then Massive, then MarketParquet) |
| prefilter | `... candidates` | `candidates.parquet` (`date, symbol`): daily-only screen (price $2-$20, 20d avg volume >= 500k, volume >= 2x average, close and open above prior close, rolling-20 high **or** `atr_pivot` breakout) |
| minute bars | `... alpaca-minute` | `alpaca_minute/DATE.parquet`: 1-minute SIP bars (incl. extended hours) for the candidate symbols of that date, signal session plus next session. `timestamp` is America/New_York |
| checks | `... validate` | `report.txt`: coverage, cross-source agreement, known delistings, yfinance cross-check |

Notes and caveats:

- Delisted names: Massive's free reference list carries delisting dates for old and new tickers; Alpaca serves bars for them. MarketParquet's free sample was **missing** names acquired inside its window (K, EA, CMA, HOLX, DAY), so it is only a gap-filler. Older Massive records have no `type`; the universe is built without a `type=CS` filter.
- Ticker changes leave the same company under two symbols with identical bars. `pit_metrics.dedupe_trades` drops these (about 9% of trades before de-duplication).
- Rows are 2024-09 to now (Massive's free window limits grouped bars; the Alpaca pull itself is not limited and could go back further).
- Not covered: news history for delisted names (Alpaca news is nearly empty for them), delisting returns, historical float and borrow data.

## Running the PIT backtest (`main.py`)

`python main.py` uses `data/pit` automatically (`BACKTEST_MODE=symbols` forces the old symbol-list mode). Relevant environment variables:

| variable | default | meaning |
|---|---|---|
| `RESISTANCE_METHOD` | `atr_pivot` | `atr_pivot` or `rolling` |
| `EXIT_VARIANTS` | `v2` | comma list of names from `exit_variants.py`, or `all` |
| `PROCESSES` | cpu count - 1 | worker processes (`1` runs in-process) |
| `LADDER_CACHE` | `1` | cache `atr_pivot` level tops per symbol/date in `data/pit/ladder_cache/` |
| `SIZE_USD` / `PARTICIPATION` | 20000 / 0.10 | position size per trade; max share of each 1-minute bar's volume |
| `SLIP_RTH` / `SLIP_EXT` | 0.0015 / 0.004 | slippage per side, regular / extended hours (estimates, no quote data) |
| `MIN_FILL` | 0.25 | entries filling less than this fraction by 15:59 are skipped |
| `BACKTEST_COMMISSION` | 0.001 | per side |
| `BACKTEST_SET` | `dev` | `dev`, `test`, or `all` (see evaluation protocol) |
| `MAX_POSITIONS_PER_DAY` | 5 | portfolio keeps the top N trades per day by relative volume |

Runtime: about 100 s for the full run (startup ~55 s, the rest parallel screening). Lazy evaluation skips the `atr_pivot` ladder on days that already fail the cheap criteria.

Fill model: entry builds the position over 15:55-15:59 and exits sell bar by bar, each taking at most `PARTICIPATION` of that minute's volume at the bar close moved against us by the slippage. Portfolio: equal weight over the day's top N trades, fully invested overnight, compounding. Per-trade CSVs go to `results/pit_trades_atr_pivot_<variant>.csv` (with entry-time features such as `sig_day_ret`, `atr_pct`, `range_pct` and path columns `ah45_*`, `pm_last`, `open_ret`).

### Evaluation protocol: dev set and locked final test set

- The last 3 months of data (`TEST_MONTHS`, currently 2026-07-16 onward) are a final test set. `main.py` writes those trades to `results/holdout/` and never summarizes them unless `BACKTEST_SET=test|all`. All tuning uses the dev set only.
- `python iterate.py` runs 12 iterations per base strategy on dev; `python iterate.py --test NAME,...` scores chosen specs on the test set. Benchmark the test set rarely and only with candidates already picked on dev.
- This is standard practice (a chronological holdout, not a random split, because returns are time-ordered). With ~250 trades over 61 days the test set is small and noisy, so treat it as a sanity check, not a precise estimate.
- Caveat: v1, v2 and pm_target25 were first compared on the full two years before the holdout existed, so the test months are not pristine for those three. Anything built after the split is clean.
- Dev set also gets a split-half check (`sharpe_h1/h2`, `dd_h1/h2` in `results/iterations_dev.csv`) to flag results that rely on one half of the data.

## Backtest results and lessons (atr_pivot, point-in-time universe)

Dev set 2024-10-18 to 2026-07-15, realistic fills (0.15% / 0.4% slippage, 0.1% commission per side, volume-limited), top 5/day by relative volume:

| strategy | trades | total | max DD | Sharpe |
|---|--:|--:|--:|--:|
| v1: exit at next open | 2,202 | +56% | -61% | 0.69 |
| v2: after-hours stop -12% / target +50% (45 min) | 2,202 | +86% | -47% | 0.79 |
| pm_target25_v2: v2 + next-day pre-market target +25% | 2,202 | +153% | -43% | 0.99 |
| pm_target25_v2 + stop -15% | 2,202 | +170% | -43% | 1.03 |
| pm_target25_v2 + stop -15% + 1/5-capital slots | 2,202 | +94% | -36% | 0.90 |

Final test set (last 3 months, 254 trades): v2 -52% (max DD -62%, Sharpe -2.6), pm_target25_v2 -50% (-61%, -2.6), pm_target25_v2 + stop -15% + slots -24% (-29%, -2.1). No tested variant made money there.

Lessons:

- **Costs decide everything.** Before costs the strategy earned about +0.2% per trade. Total return for v2 / pm_target25_v2 at slippage 0 / 0.1%-0.3% / 0.15%-0.4% / 0.3%-0.6% (regular/extended): +465% / +144% / +61% / -55% and +723% / +243% / +124% / -38%. Earlier results without slippage and volume limits were far too optimistic.
- **Data hygiene:** renamed tickers double-counted ~9% of trades; the MarketParquet free sample drops acquired names; a `type=CS` filter silently drops older delisted names; a first-day drawdown was missed until the equity curve started from 1.0.
- **After-hours/overnight news dominates tails:** the worst trades are mostly after-close offerings or overnight gap-downs (e.g. SIDU after a +96% contract spike and a same-night $25M offering); the best are overnight gap-ups on news (INTZ, BATL, ZENA, LASE, GRML). A 12% after-hours stop trims the tail but sells ~half of positions that later recover.
- **Losers look "overheated":** top-20 losers had median signal-day move +53% (all trades +16%), daily ATR 12% of price (vs 7%), intraday range 46% (vs 16%), 4x the day's dollar volume. Hard filters on those features did **not** help (they cut winners as much as losers; every filter lowered Sharpe).
- **Pre-market stops hurt:** a -15% pre-market stop destroyed returns (thin books, recoveries after the open); the pre-market *target* helped.
- **Drawdown reduction is mostly de-risking:** the only things that lowered max DD were smaller positions (1/5 slots, inverse-ATR weighting), which also cut return; Sharpe did not improve. The strategy's edge is small and appears to have faded in the final months.
- **Premature optimization of rules:** nine exit rule sets on the same two years is enough to overfit; prefer the dev/test protocol above.

## Data Sources

### Alpaca (Primary)
- **Cost**: Free
- **Coverage**: US stocks/ETFs back to ~2016 (1-minute bars)
- **Requests**: Split-adjusted SIP bars (daily and 1-minute)
- **Limitations**: Real-time is IEX only; last 15 minutes of SIP blocked
- **Best for**: Historical backtesting

### Alpaca News (Catalyst)
- **Module**: `src/data/news_client.py` — `AlpacaNewsClient` wraps the alpaca-py `NewsClient`
- **Auth**: Same `ALPACA_API_KEY` / `ALPACA_API_SECRET` as bars
- **API**: `fetch_news(symbols, start, end)` with pagination; `daily_counts()` returns per-symbol per-date article counts used as the news catalyst
- **Best for**: Point-in-time news catalyst screening

### SimFin (Fundamentals / Market Cap)
- **Module**: `src/data/simfin_store.py` — `SimFinStore` wraps the `simfin` package
- **Auth**: Set `SIMFIN_API_KEY` in `.env`
- **Data**: Bulk-downloads US `shareprices-daily` (cached in `data/simfin/` after first run)
- **API**: `market_cap_series(ticker)` = Close * Shares Outstanding as-of each date; `fundamentals_as_of()` filters income statements by Publish Date
- **Caveat**: SimFin bulk files are restated snapshots, not a true historical database — publish-date filtering mitigates but does not eliminate look-ahead bias
- **Best for**: As-of market-cap screening

### FMP Social Sentiment (Bonus Flag)
- **Module**: `src/data/sentiment_client.py` — `FMPSentimentClient`
- **Auth**: `FMP_API_KEY` (optional)
- **API**: FMP's **legacy** v4 endpoint `https://financialmodelingprep.com/api/v4/historical/social-sentiment`
- **Behavior**: Fails soft, returning empty data on errors; the social surge is a bonus flag and never blocks an entry
- **Alternatives**: Apewisdom free aggregate API (limited history); Stocktwits public stream (last ~30 messages only)

### Finviz (Screener)
- **Cost**: Free
- **Coverage**: Current market data only
- **Best for**: Exploring current candidates — not used by historical backtests (current constituents would introduce survivorship/look-ahead bias)

### Future Additions
- MarketParquet (one-time purchase for full historical archive)
- yfinance (free, limited lookback)
- Schwab API (for live trading)

## Backtesting with Vectorbt

The framework uses vectorbt for fast vectorized backtesting:

- Supports multiple symbols simultaneously
- Efficient parameter sweeps
- Built-in performance metrics
- Easy equity curve visualization

## Resistance Detector Research

`src/screening/resistance.py` is an ATR-scaled pivot zone detector for support/resistance on daily bars. Pivots are clustered into zones (candle body top to wick high for swing highs, wick low to body bottom for swing lows), and each zone is reported as a single level: the highest price in it (`--band` keeps the full range instead). A pivot is usable only `pivot_right` bars after it forms, so levels at bar k never depend on later bars. Role is relative to price: zones above are resistance, zones below are support, and a zone price is inside is `in_zone`. When nothing is above price the ladder is empty (blue sky).

`src/screening/pivot_resistance.py` is a simpler, dependency-free alternative built from the `data/resistance_dataset` few-shot experiments: swing highs (+-4 bars) above the last close, merged within $0.10 keeping the exact top. About 0.67 F1 (+-$0.10 hit) on both train and test splits. Score it with `python eval_pivot_resistance.py data/resistance_dataset/test` (writes `predictions_pivot.csv`, then prints TP/FP/FN, precision, recall, F1, accuracy).

`src/screening/cluster_levels.py` is a pure-numpy port of day0market/support_resistance (`pricelevels`): `zigzag` runs ZigZag `peak_valley_pivots` over the highs (or closes), `raw_price` takes bars equal to the centered rolling max, and both cluster the candidate prices with AgglomerativeClustering into levels.

`src/screening/sr_finder.py` ports the implemented parts of albertsl/support-resistance_trading-bot's `support_resistance_finder.py` (upstream is a 2018 sketch — `_pivot_points` and the trendline methods are `pass`; trendlines are skipped since the dataset labels horizontal levels): `deriv_extrema` finds local maxima where the (optionally Butterworth-smoothed) series' gradient crosses, `volume_levels` takes low-slope + high-volume bars, `meanshift` clusters the price series with `estimate_bandwidth` and emits band edges, and `floor_pivots` computes classic/camarilla/fibonacci pivot-point formulas over a lookback.

`eval_pivot_resistance.py` scores any of the detectors on a split: `--detector pivot|atr_pivot|zigzag|raw_price|volume_profile|deriv_extrema|volume_levels|meanshift|floor_pivots`, with config fields overridable via repeated `--set key=value` (e.g. `--detector zigzag --set peak_percent_delta=2 --set series=close`).

`sweep_resistance.py` grid-searches all five detector families over the train+val splits (`--tune`), prints a per-family leaderboard ranked by combined F1, writes the full table to `data/resistance_dataset/sweep_results.csv`, and saves each family's best config the usual way: `predictions_<detector>.csv` + `eval_results_<detector>.txt` (score.py output) in every split. Last sweep (3318 configs, 9 families): pivot kept its defaults (pivot_bars=4, merge_tol=0.10) at 0.710 train+val F1; zigzag (delta=3%, $0.10 merge) 0.694, raw_price (5-bar rolling max, $0.10 merge) 0.685 and deriv_extrema (unsmoothed sign-cross maxima, $0.10 merge) 0.667 are close behind — everything that works prefers a ~$0.10 merge, matching the scoring tolerance; atr_pivot's best (`pivot_left=3 pivot_right=3 min_prominence_atr=0.5 merge_atr=0.3 include_ranges=false`) reached 0.632 and improved test F1 from 0.574 to 0.656; meanshift landed mid-pack (0.575), while volume_levels (0.530), floor_pivots (0.362) and volume_profile (0.123) trail — formulaic and volume-node levels don't match the labelled swing-high style.

Research CLI (run with the `trading_framework` interpreter):

```bat
python research_resistance.py ladder SOFI 2024-03-05
python research_resistance.py chart  SOFI 2024-03-05 --forward 12
python research_resistance.py eval   SOFI,PLUG,OPEN --start 2023-01-01 --end 2025-06-30 --split 2024-07-01
```

- `chart` writes an interactive candlestick + volume HTML to `data/charts/` with each level drawn as a line from where it formed (red resistance, green support; thicker means a higher strength score). Bars after the as-of date are drawn faded so you can see what happened without it affecting the levels.
- Consolidation ranges: when price holds a tight range for 5-10 bars (height at most 2 ATR, net drift at most 0.75 ATR), the range high and the range low are both levels. Tune with `range_*` keys in the config; `--no-ranges` turns it off.
- A second detector, `--detector volume_profile` (`src/screening/volume_profile.py`), finds high-volume nodes: each daily bar's volume is spread over its high-low range into a volume-at-price histogram, smoothed, and the strongest peaks above/below price become levels (at most `max_above` / `max_below`). It returns few levels by design and only finds resistance where a lot of volume actually traded. All commands (`ladder`, `chart`, `eval`, `labels`, `config dump`) take `--detector`.
- Chart dates must have overhead resistance or they cannot test a detector. `pick SYMBOLS --start --end` chooses dates by detector-independent rules (price $1-$20, 20-day volume at least 500k, close 55-90% of the prior 252-bar high, 20-day return at least +5%, 20+ bars above price in the last year), sampled at random with a fixed seed and minimum spacing, and writes `data/labels/chart_set.csv`. `labels build-set` then exports candidates from several detectors for each date into the one label file and writes `data/charts/<SYMBOL>_<date>_set.html`. Review the union once (`keep` / `reject` / `add`), then score each detector against the same gold labels.
- Blind manual labeling: `pick ... --max-total 20 --exclude <earlier csvs> --out data/labels/chart_set_manual_1.csv` selects fresh dates, then `labels manual-set --set data/labels/chart_set_manual_1.csv` writes `data/charts/manual/NN_SYMBOL_date.html` plus an `index.html`. These charts never run or draw a detector and show no bars after the as-of date, so labels are made without hindsight or detector bias. Label by adding rows `symbol,asof,level` (status defaults to add, source to manual) to `data/labels/manual_labels.csv`; re-running `manual-set` draws your labels as dashed lines. Later `labels score --labels data/labels/manual_labels.csv --detector atr_pivot` (or `volume_profile`) scores a detector against them.
- Charts color unreviewed levels by detector (orange = atr_pivot, purple = volume_profile) and the legend entries toggle each group on and off.
- The label file has an `id` column per (symbol, asof), numbered from the highest price down, and a `source` column naming the detector. Charts draw the label rows with the same `id` as a tag next to each line and show details on hover. Earlier labels from the pivot detector are kept in `data/labels/archive/`.
- Detector settings can be edited in a JSON file: `config dump` writes `data/labels/detector_config.json`; pass it with `--config`.
- `eval` scores the detector by outcomes against baselines (rolling 20/10-bar high, round numbers, random bands, and a `shifted` placebo that keeps the detector's zone widths and scores but moves them). It reports touch/reject/break rates, breakout forward returns, and whether room to the next resistance predicts follow-through, on a tune and a holdout period.

### Labeled chart dataset (`data/resistance_dataset/`)

A hand-labeled dataset of daily charts for benchmarking resistance detection, split into `train/`, `val/`, and `test/`. Each split contains:

- `charts/NN_SYMBOL_ASOF.png` — daily candles + volume; the last bar is the as-of date
- `price_data/NN_SYMBOL_ASOF.csv` — `date,open,high,low,close,volume` for exactly the bars on the chart (~161 bars)
- `labels.csv` — `symbol,asof,level` gold resistance levels
- `explainations/` (train/val only) — per-chart analyst reasoning, used as few-shot worked examples
- `predictions*.csv` / `eval_results*.txt` — saved outputs from past detector and VLM runs
- `charts_html/` (test only) — interactive HTML renders of the test charts

Dataset helpers (run from the dataset directory or the repo root):

- `data/resistance_dataset/score.py` — score a predictions CSV against a split's labels: `python score.py <split> <predictions.csv>`. A prediction is a hit within ±$0.10 of a label, or ±1% of the label when the last close is ≥ $50 (large caps)
- `data/resistance_dataset/highs.py` — pivot-high price lookup helper: `python highs.py <price_csv> [window]` prints the last close and every bar high above it that is the max of ±`window` bars; it does not choose levels
- `data/resistance_dataset/prompt.txt` — the few-shot task description the dataset was built for
- `data/resistance_dataset/val/evaluate.py` — earlier quick scorer hardcoded to `test/labels.csv`; superseded by `score.py`
- `eval_pivot_resistance.py` — run a detector on a split and score it: `python eval_pivot_resistance.py data/resistance_dataset/test --detector atr_pivot` (writes `predictions_<detector>.csv`, then prints TP/FP/FN, precision, recall, F1, accuracy)
- `sweep_resistance.py` — grid-search detector params on train+val and save the best configs' scores: `python sweep_resistance.py` (`--detector` for one family, `--no-save` for leaderboard only)
- `label_charts.py` — interactive browser labeling server for a chart folder: `python label_charts.py --dir data/resistance_dataset/<split> --port 8765`. Left-click adds a level line, right-click deletes the nearest, arrow keys switch charts, and Save writes `<dir>/labels.csv`

## VLM Benchmark (can a vision model read resistance levels?)

`vlm_benchmark.py` benchmarks vision-language models against the human-labeled charts. Each packet is the blind chart PNG (no labels, no future bars) plus a CSV of the same daily OHLCV bars; your manual levels are the gold answer and are never sent to the model.

```bat
python vlm_benchmark.py prepare                 # PNGs with/without labels, OHLCV csvs, manifest -> data/vlm/
python vlm_benchmark.py models                  # table of VLMs: prices, vision signals, est. cost
python vlm_benchmark.py models --write          # also data/vlm/vlm_candidates.csv / .md
python vlm_benchmark.py run --dry-run           # cost preview only; no keys, no network
python vlm_benchmark.py run --first-pass --repeats 3 --budget 5
python vlm_benchmark.py run --split val --shots train --prompt learned   # dev: val charts, train worked examples
python vlm_benchmark.py run --split test --shots train,val --prompt learned  # final: test charts, train+val examples
python vlm_benchmark.py report                  # leaderboard.csv + results/review/index.html
python vlm_benchmark.py report --split val,test # score dataset splits instead of the manifest
```

- **Keys:** `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` in `.env`. `--via openrouter` routes every model through OpenRouter so one key covers all vendors.
- **Safety:** every call is cached in `data/vlm/results/runs.jsonl` keyed by (model, chart, prompt, image bytes, repeat); reruns skip completed calls and never pay twice. `--budget` caps spend per invocation; a call is skipped if its estimated cost would exceed it. Always `--dry-run` first.
- **Offline check:** `run --mock oracle,noisy,empty,garbage` exercises the whole pipeline with fake models (no keys, no network).
- **Scoring:** a predicted level matches a gold level within `--tol` ATR14 (headline 0.5; also reported at 0.25 and 1.0). `report` writes `leaderboard.csv`, `per_run_scores.csv`, and `results/review/index.html` where each model's levels are overlaid next to the gold chart (blue dashed = gold, green = matched, red = unmatched, orange = missed).
- **Model registry:** `data/vlm/models.json` holds prices, context sizes, vision benchmark signals and per-model assumptions, with a `notes` section naming sources and caveats. `models --scenario low|base|high` varies the assumed hidden-reasoning size.
- **Dataset splits:** `run --split train|val|test` (and `report --split`) evaluate the `data/resistance_dataset` splits directly instead of the legacy `data/vlm` manifest. `--shots train,val` prepends worked examples from labeled splits (user turn: chart + OHLCV; assistant turn: explanation + gold JSON) — the `test` split is never allowed as shots.
- **Prompts:** `--prompt rubric` encodes the labeling rules (default without shots); `plain` is a bare request for an ablation; `fewshot` and `learned` are used with `--shots` (`learned` is the default when shots are given). `--user-text-file` appends extra notes to the shared system prompt.

## Benchmark results (`data/resistance_dataset`)

Unless noted, scores come from `score.py`: a prediction hits a label within ±$0.10,
or ±1% of the label when the chart's last close ≥ $50 (large caps — the metric shown).
`accuracy` = TP/(TP+FP+FN). Per-chart details live in each split's `eval_results*.txt`;
the full parameter table is `data/resistance_dataset/sweep_results.csv`.

### Rule-based detectors — test split

Each row is the config picked by the train+val sweep (`sweep_resistance.py`,
3,318 configs across 9 families), sorted by F1.

| detector | TP | FP | FN | precision | recall | F1 | accuracy |
|---|--:|--:|--:|--:|--:|--:|--:|
| atr_pivot | 63 | 39 | 27 | 0.618 | 0.700 | 0.656 | 0.488 |
| pivot | 65 | 48 | 25 | 0.575 | 0.722 | 0.640 | 0.471 |
| zigzag | 71 | 61 | 19 | 0.538 | 0.789 | 0.640 | 0.470 |
| raw_price | 73 | 74 | 17 | 0.497 | 0.811 | 0.616 | 0.445 |
| meanshift | 62 | 58 | 28 | 0.517 | 0.689 | 0.590 | 0.419 |
| deriv_extrema | 67 | 87 | 23 | 0.435 | 0.744 | 0.549 | 0.379 |
| volume_levels | 68 | 183 | 22 | 0.271 | 0.756 | 0.399 | 0.249 |
| floor_pivots | 27 | 53 | 63 | 0.338 | 0.300 | 0.318 | 0.189 |
| volume_profile | 14 | 32 | 76 | 0.304 | 0.156 | 0.206 | 0.115 |

### AI / VLM-based — model reads the blind chart + OHLCV CSV, test split

Agent/VLM chart-reading runs scored by `score.py` (`predictions_*.csv` written
before labels are read; `eval_results*.txt` holds the detail). Earlier val-split
iterations are kept in `val/eval_results*.txt`.

| run | TP | FP | FN | precision | recall | F1 | accuracy |
|---|--:|--:|--:|--:|--:|--:|--:|
| vlm_audit_20261009 | 72 | 37 | 18 | 0.661 | 0.800 | 0.724 | 0.567 |
| vlm2 | 72 | 46 | 18 | 0.610 | 0.800 | 0.692 | 0.529 |
| skill, rescored | 70 | 44 | 20 | 0.614 | 0.778 | 0.686 | 0.522 |
| v5 (skill) | 75 | 67 | 15 | 0.528 | 0.833 | 0.647 | 0.478 |

The `vlm_benchmark.py` leaderboard uses its own metric — greedy matching within
a multiple of ATR14, not $0.10 — so it is kept separate (full table:
`data/vlm/results/leaderboard_*.csv`; `runs` = charts scored). Test split so far:

| model | runs | precision@0.5ATR | recall@0.5ATR | F1@0.5ATR | F1@1ATR |
|---|--:|--:|--:|--:|--:|
| gpt-6-luna | 20 | 0.723 | 0.783 | 0.726 | 0.763 |

## Roadmap

- [ ] Add more strategies (momentum, mean reversion, etc.)
- [ ] Implement paper trading mode
- [ ] Add Schwab API integration for live trading
- [ ] Implement risk management and position sizing
- [ ] Add performance attribution
- [ ] Create web dashboard for strategy monitoring

## Notes

- This framework is for research and educational purposes
- Always validate strategies with out-of-sample testing
- Past performance does not guarantee future results
- Use proper risk management when trading with real money

## Testing

The suite uses `unittest` (no pytest required):

```bat
C:\data\envs\trading_framework\Scripts\python.exe -m unittest discover -s tests
```

Covers the backtest engine, historical strategy screening, the resistance detectors (atr_pivot, volume_profile, pivot_resistance), the PIT fill model and metrics (volume-limited fills, ticker-change de-duplication, sizing modes), and the VLM benchmark internals (scoring, providers, runner, reports) — 93 tests.

## Utility scripts

- `python -m http.server 8765 -d C:\data\trading\trading_framework\data\charts` — serve the generated HTML charts locally
- `python label_charts.py --dir data/resistance_dataset/<split> --port 8765` — interactive chart labeling server (see above)
- `data/vlm/gpt_6_astral/` — one-off VLM audit artifact: `thoughts.md` (detector rules + protocol notes) and `evaluate_run.py` (scores a predictions CSV with the dataset's matching rules)