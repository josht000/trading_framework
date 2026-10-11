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

## Architecture

```
trading_framework/
├── main.py                   # Backtest application entry point
├── research_resistance.py    # Detector research CLI (ladder/chart/eval/labels/pick/config)
├── eval_pivot_resistance.py  # Score the pivot detector on a resistance_dataset split
├── label_charts.py           # Browser labeling server for dataset charts
├── vlm_benchmark.py          # VLM benchmark CLI (prepare/models/run/report)
├── src/
│   ├── data/                 # Data ingestion modules
│   │   ├── alpaca_client.py    # Alpaca API client (daily/minute bars)
│   │   ├── news_client.py      # Alpaca news client (per-symbol daily article counts)
│   │   ├── simfin_store.py     # SimFin bulk store (as-of market cap / fundamentals)
│   │   ├── sentiment_client.py # FMP social sentiment client (bonus flag, fails soft)
│   │   └── finviz_screener.py  # Finviz screener (not used by historical backtests)
│   ├── screening/            # Screening and level detectors
│   │   ├── historical.py       # Point-in-time historical screening
│   │   ├── resistance.py       # ATR-scaled pivot zone detector (support/resistance)
│   │   ├── pivot_resistance.py # Simple swing-high detector for the labeled dataset
│   │   └── volume_profile.py   # Volume-at-price high-volume-node detector
│   ├── strategies/           # Trading strategies
│   │   └── humbled_bullish_overnight.py
│   ├── backtest/             # Backtesting engine
│   │   └── engine.py
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
- Breakout above the prior 20-day high
- Close above the session VWAP
- Market cap <= $2B as-of the screen date (SimFin: Close * Shares Outstanding). Controlled by `REQUIRE_MARKET_CAP` (0/1); missing data passes unless required
- News catalyst: >= `MIN_NEWS_ARTICLES` articles within `NEWS_LOOKBACK_DAYS` (defaults: 1 article / 3 days), from Alpaca news per-day counts. `REQUIRE_NEWS=0` disables
- Social surge is recorded as a bonus flag only — it never blocks an entry

### Not Yet Enforced

Historical NASDAQ membership and sector hype are not available from the current data sources and are explicitly not yet enforced. Social sentiment is implemented as a non-blocking bonus flag (`social_surge`) rather than a hard entry criterion.

### Entry Timing
- Enter on the 15:55 ET bar when all criteria pass

### Exit Timing
- Exit on the first regular-session bar of the next trading date

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
- `eval_pivot_resistance.py` — run the `pivot_resistance` detector on a split and score it: `python eval_pivot_resistance.py data/resistance_dataset/test` (writes `predictions_pivot.csv`, then prints TP/FP/FN, precision, recall, F1, accuracy)
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

Covers the backtest engine, historical strategy screening, the resistance detectors (atr_pivot, volume_profile, pivot_resistance), and the VLM benchmark internals (scoring, providers, runner, reports) — 88 tests.

## Utility scripts

- `python -m http.server 8765 -d C:\data\trading\trading_framework\data\charts` — serve the generated HTML charts locally
- `python label_charts.py --dir data/resistance_dataset/<split> --port 8765` — interactive chart labeling server (see above)
- `data/vlm/gpt_6_astral/` — one-off VLM audit artifact: `thoughts.md` (detector rules + protocol notes) and `evaluate_run.py` (scores a predictions CSV with the dataset's matching rules)