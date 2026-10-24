# Live trading system plan (Alpaca paper first, NautilusTrader core)

First strategy: **H3 + hygiene + tuned score** (bullish overnight, `pm_target25_v2` exits). See `FRAMEWORK_COMPARISON.md` for the decision to adopt NautilusTrader. A dedicated `docs/live/` set of documents is planned: this plan, the comparison, and (when built) `RISK.md` and `OPERATIONS.md`.

## 1. Strategy spec (from existing research)
Sources: `rank_test.py`, `src/backtest/ranking.py`, `src/backtest/exit_variants.py`, `src/backtest/pit_runner.py`.

**Entry (15:55 ET)**
1. Candidates from the screener (price/volume filters, 13-month `atr_pivot` breakout, higher close and open, above VWAP, volume surge).
2. Hygiene level 1: drop OTC and pink-sheet names.
3. H3 filters: `headroom >= 20%` and `above_sma200`.
4. Tuned score >= 2. Weights are frozen from the `rank_test.tune_weights` output into `live/config/strategies/overnight_h3.yaml`:
   headroom 0.5, trend200 1, market 1, movers 0.5, delist_notice 3, late_filing 1.5, theme 1.5. All others are 0.
5. Top 5 by `rank_key`, each sized at 20% of **sleeve equity**.
6. Buy with a marketable limit near the 15:55 price, with size capped as a fraction of recent volume (the backtest fill model is volume-limited with slippage).

**Sizing: paper trading uses exactly the research account (`pit_metrics.simulate_account`)**
- Sleeve starts at $5,000 and compounds with realized PnL. Each of at most 5 positions is 20% of current sleeve equity (about $1,000 at the start). No boost (the 25% boost is not used by the H3 tuned run), so max gross is 100% and never levered.
- Fill realism parameters from `main.py`, reused for the live sanity checks and the live-vs-backtest comparison: fills take at most 10% of each 1-min bar's volume, slippage 0.15% regular session and 0.40% extended hours per side, an entry filling less than 25% of the target is skipped (`size_usd` in the fill model is $1,250, a bit more conservative than the $1,000 account sizing), and commission per `COMMISSION` in `main.py`.
- Whole shares only. Round down, and skip a name if one share exceeds its slot.
- Paper account equity at Alpaca (about $100k) is never used for sizing.

**Exits (`pm_target25_v2`)**
- 16:00 to 16:45: sell if a 1-min close is <= entry -12% (stop) or >= entry +50% (target).
- Otherwise, next-day pre-market: sell at the first bar >= entry +25%.
- Otherwise: sell at the first bar of the next regular session.

**Features to be rebuilt for live time (parity with `features_extra.py`):** headroom (`atr_pivot` ladder), above_sma200, iwm/spy day, top-gainer and mover history, EDGAR delist and late-filing flags, theme heat, OTC flag. Each may use only data available at 15:55. Live sources for each are in `DATA_SOURCES.md`. Short version: EDGAR from the free `data.sec.gov` submissions API; movers from our own nightly top-20-gainers job and Alpaca news headlines (no Benzinga Movers API needed); theme heat from a frozen topic model plus a daily candidate log.

**Later data source (data only, never orders):** Charles Schwab, researched in `SCHWAB.md` (7-day refresh token, different extended-hours windows). The Schwab app is the Market Data product, callback `https://127.0.0.1:8182`.

## 2. Data constraints with Alpaca free (IEX) data
Verified from Alpaca docs:
- Free Basic plan: live data is **IEX only** (about 2.5% of market volume), 30 websocket symbols, 200 historical calls per minute.
- Historical SIP data is available since 2016 but **not for the latest 15 minutes**. So at 15:55 we have SIP minute bars up to about 15:40 and IEX up to now.
- Consequences for this strategy:
  - The research used SIP volume. Same-day volume, relative volume and "top gainers by dollar volume" computed from IEX will be far lower and noisy, so any thresholds based on them are wrong unless we scale or use SIP.
  - Daily history (prior sessions) can be fetched from SIP for free, so `atr_pivot`, SMA200 and the movers history are fine.
  - Same-day price, VWAP and volume surge are the weak spot. Options: (a) use SIP data up to 15:40 and extrapolate volume, with IEX only for the last price; (b) screen at 15:40 with SIP-delayed data and refresh prices from IEX at 15:55; (c) pay for Algo Trader Plus ($99 per month) or move to Schwab data later.
- Plan: build the feed behind a `DataSource` interface, run (b) first, and log a **parity report** after each session comparing candidates from the live path with SIP-complete bars fetched the next day. That gives a measured cost of using the free data.
- Websocket limit of 30 symbols means we stream only held positions and the shortlist. Everything else is REST snapshots.

## 3. Architecture
```
live/
  node.py               # builds the Nautilus LiveNode from config; CLI entry (run | status | flatten | report | replay)
  config/               # pydantic settings, strategies/*.yaml, risk.yaml; secrets only from .env
  adapters/alpaca/      # custom Nautilus adapter (no official one exists)
    config.py factories.py
    instruments.py      # InstrumentProvider (equities, lot/tick rules, tradable flags)
    data.py             # DataClient (REST bars/snapshots, IEX websocket for a small symbol set)
    execution.py        # ExecutionClient (orders, order-update stream, reconcile reports, extended_hours param)
  data/features/        # live builders: pivots, sma200, market, movers, edgar, theme, otc
  strategies/
    overnight_h3.py     # Nautilus Strategy: thin; calls pure scoring functions from src/backtest/ranking.py
  risk/
    sleeve.py           # sleeve accounting (section 4)
    guard.py            # Actor: portfolio-level risk checks and kill switch
  reporting/
    ledger.py           # sleeve equity snapshots and trade journal
    stats.py            # statistics (section 5)
    plots.py            # plots (section 5)
scripts/                # preflight, replay_day, flatten, report, setup_live_env
tests/live/
```
Principles:
- **Resume** = Nautilus reconciliation on start (open orders, fills and positions pulled from Alpaca and matched to the cache) + our SQLite (strategy position-state machine: `ENTRY_PENDING -> OPEN -> AH_WATCH -> PM_WATCH -> EXIT_PENDING -> CLOSED`, and the sleeve ledger). On start the strategy derives "what should be happening now" from the clock and state, then continues. Cache persistence in Redis is optional and decided in the spike.
- **Idempotency**: deterministic `ClientOrderId` per strategy, day, symbol and leg.
- **Adapter boundary**: strategies only use Nautilus types. A Schwab client later replaces `adapters/alpaca` without touching strategies.
- **Same code for backtest and replay**: Nautilus `BacktestEngine` with a simulated venue (extended hours and the fill model approximated) for plumbing, plus the existing vectorbt research for strategy truth. Do not expect these two to match exactly.
- **Many strategies**: one `LiveNode` (Nautilus supports many strategies per node), each with its own `StrategyId`, config file and sleeve. Split into separate processes later for isolation.
- Logging: Nautilus structured logs plus a human console. Minimal comments.

## 4. Risk management engine
Layers, from innermost to outermost:

1. **Nautilus RiskEngine (pre-trade, built in).** Configure `max_notional_per_order` (per instrument; set from the sleeve cap when the shortlist is known), `max_order_submit_rate`, and keep `bypass=False`. Trading state ACTIVE / REDUCING / HALTED is the master switch: a HALTED state blocks new orders, REDUCING allows only position-reducing ones.
2. **Sleeve gate (our code).** Nautilus checks the broker account (about $100k paper), not our $5k sleeve, so we enforce it:
   - Sleeve equity = $5,000 + realized PnL + unrealized PnL of this strategy's positions (from `StrategyId`-tagged positions). Never reads the broker's account equity.
   - Each position <= 20% of sleeve equity, <= 5 positions, total gross <= 100% of sleeve (no leverage), whole shares only, buying power check against the sleeve cash balance.
3. **Order sanity checks (our code):** limit price within X% of last, spread cap, size <= a fraction of recent volume, symbol allowlist and tradable flag, no duplicate entry for the same symbol/day, data-staleness guard (no entry if the last price is older than N seconds), orders only inside configured session windows.
4. **Portfolio guard Actor (our code, runs on a timer):**
   - Daily loss limit (for example -5% of sleeve) and drawdown circuit breaker (for example -15% from peak): set trading state to REDUCING or HALTED and alert.
   - Mismatch monitor: broker vs local positions or orders differ beyond tolerance -> HALT.
   - "Unprotected position" monitor: any open position without a live watch or exit plan -> alarm and flatten per policy.
   - Consecutive-rejection and API-error breaker.
   - **Kill switch**: a `KILL` file or `flatten` command cancels all orders and closes positions (limit orders in extended hours).
5. **Operational:** paper/live allowlist on the base URL, explicit live flag, max daily orders, heartbeat file for an external watchdog.

Thresholds live in `risk.yaml`. Each breaker is unit-tested with the simulated venue. Initial numbers come from the research distribution (the `reports/ranking_report.md` max drawdown and worst-day figures) and are reviewed after the first paper weeks. `RISK.md` will list each rule, its default and its test.

## 5. Portfolio statistics and plots
The sleeve is the unit of measurement: all returns are against the initial **$5,000**, computed from our ledger (not the broker account).

- **Ledger** (SQLite): trade journal (entry/exit time and price, fees, slippage vs signal price, exit reason, score and components), and sleeve equity snapshots at session end (and optionally intraday).
- **Statistics**: use Nautilus `PortfolioAnalyzer` (Sharpe, Sortino, profit factor, win rate, expectancy, returns volatility and so on) on the sleeve return series, and add our own so live matches the research tables from `pit_metrics.simulate_account`: trades, win%, mean%, PF, total%, maxDD%, min$, worst_day%, loss-streak days, Sharpe. Also execution-quality metrics: slippage vs backtest fill model, fill rate, partial fills, time to fill. Also exit-reason mix and the score-bucket calibration (expected vs realized return per score).
- **Plots** (static PNG plus a self-contained HTML report, in `reports/live/<date>/`): equity curve vs the backtest expectation, drawdown, daily PnL bars, monthly heatmap, trade return histogram, per-trade scatter (score vs return), rolling Sharpe, exposure over time, slippage distribution, exit-reason breakdown. Reuse `plot_portfolio.py` conventions. Nautilus' `create_tearsheet` is backtest-oriented, so it is used for BacktestEngine replays, while the live report uses our generator over the same chart set.
- A daily summary (markdown) after the close and a weekly report. `scripts/report.py` regenerates everything from the ledger.

## 6. Where it runs
- **This Windows machine** to start. Nautilus lists Windows x86_64 as supported (CI uses Windows Server 2022, so Windows 10/11 should work but is not officially listed). Run via Task Scheduler or NSSM with auto-restart.
- **Raspberry Pi 5**: a good fit for the live app. Nautilus publishes Linux ARM64 wheels (glibc 2.35+), the strategy is light (a handful of symbols, one decision per day), and it makes an always-on box with low power draw. Requirements: a 64-bit OS (Raspberry Pi OS Bookworm or Ubuntu 22.04+), 8 GB RAM, **an SSD or high-endurance storage rather than an SD card** for SQLite/Redis, a small UPS, NTP time sync (timestamps matter), systemd for restart and watchdog. Keep research and backtests on the PC. Treat the Pi as optional: validate by installing the wheel and running the preflight on it early rather than assuming. 12 strategies of this type should be fine, while heavy intraday data processing would not be.
- Because the state lives in SQLite (and optionally Redis) plus the broker, moving between PC and Pi means copying one data directory. Never run both against the same account at once. A lock check on the broker side (`status` command) is part of the plan.

## 7. Environment
- Python **3.14.8** is installed at `C:\Users\jtham\AppData\Local\Programs\Python\Python314\python.exe`. The bare `python` in this shell is still 2.7.17, so always use an explicit interpreter or an activated venv. The conda env `trading_framework` is 3.11 and stays for research.
- Live app: separate venv `.venv-live` (gitignored) built from the 3.14.8 interpreter with `python -m venv` (uv is not installed and is optional). Pin **nautilus_trader 1.231.0** (stable; PyPI lists 3.12-3.14 wheels, to be confirmed by the install in milestone 0). Revisit when 2.0 final ships.
- Packages (each needs an install check on 3.14 in the spike): `nautilus_trader[visualization]==1.231.0`, `alpaca-py`, `pandas`, `numpy`, `pydantic`, `pyyaml`, `python-dotenv`, `tenacity`, `requests`, `pytest`. `scikit-learn` and `yfinance` only if the theme topic model is rebuilt inside the live env (otherwise ship the frozen map). `schwab-py` later.
- Files: `requirements-live.txt` (pinned) and `scripts/setup_live_env.ps1` (plus `.sh` for the Pi). Keep `requirements.txt` (research) separate. Both must match real imports (AGENTS.md); I will flag when either needs updating.
- `.env` already has `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_BASE_URL`. Add `ALPACA_PAPER=true`, `EDGAR_USER_AGENT="Name email"` (SEC requirement), and later the `SCHWAB_*` keys.
- Terminal note: `C:\data\trading_framework\Scripts\activate` does not exist. Conda envs are under `C:\Users\jtham\Anaconda3\envs\`.

## 8. Milestones
0. **Spike and preflight** (`scripts/alpaca_preflight.py`, `scripts/nautilus_spike.py`): Alpaca auth, account, clock, calendar, tradable flags, data feeds. Place a far-from-market limit buy, confirm it rests, cancel it (overnight-safe). Install Nautilus in `.venv-live`, run a minimal `LiveNode` with a stub adapter, kill and restart it, and check what reconciliation gives us. Decide Redis or not, and the Nautilus version.
1. Adapter v1: instruments and execution (submit, cancel, order stream, reconciliation reports, extended-hours params). Test with resting far-from-market limits during off hours.
2. Data client and feature builders, with parity replay against `reports/ranked_table.parquet` (`scripts/replay_day.py`). Gate: picks match on replayed days, or differences are explained. Include the IEX vs SIP parity report.
3. Sleeve, ledger, risk guard and tests (kill and restart mid-position, simulated breaker triggers).
4. `overnight_h3` strategy in dry-run mode (logs intents only) for several sessions.
5. Paper trading at the research sizing from day one (5 x 20% of the $5,000 sleeve). Daily stats and plots from section 5.
6. Second strategy to validate the abstractions. Candidates: base + hygiene tuned, S1 + hygiene tuned.

## 9. Open questions
1. For the 15:55 screen, accept option (b) in section 2 (SIP up to 15:40 plus IEX last price) for the first paper runs? Default: yes, with the parity report.
2. Risk defaults: daily loss halt -5% and drawdown halt -15% of the sleeve, OK as starting points?
3. The theme-heat feature needs a frozen topic model and a daily candidate log (see `DATA_SOURCES.md`). Fine to spend milestone 2 on that, since theme has the biggest soft weight?
4. When should we start the Schwab developer registration? Approval takes days, so it can run in parallel.
5. Where should the app run first: this PC (Task Scheduler) or the Pi 5? Default: PC.
