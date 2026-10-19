"""
Trading Framework - Main Application

Humbled Trader Bullish Overnight Strategy
"""

import sys
import os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(__file__))

from src.data.alpaca_client import AlpacaDataClient
from src.data.news_client import AlpacaNewsClient
from src.data.simfin_store import SimFinStore
from src.data.sentiment_client import FMPSentimentClient
from src.strategies.humbled_bullish_overnight import HumbledBullishOvernight
from src.backtest.engine import BacktestEngine, MultiSymbolBacktest


def get_backtest_universe():
    """
    Get a fixed universe of stocks for backtesting.

    Note: This uses a fixed set of stocks that existed during the backtest period.
    For production backtesting, you should use a historical universe database
    to avoid survivorship bias. Finviz only shows current data.

    Returns:
        List of ticker symbols
    """
    print("=" * 60)
    print("STEP 1: Selecting Backtest Universe")
    print("=" * 60)
    print("\nUsing configured symbols as the broad historical universe.")
    print("Price and volume criteria are evaluated independently at each historical date.")

    tickers = [
        symbol.strip().upper()
        for symbol in os.getenv("BACKTEST_SYMBOLS", "SOFI").split(",")
        if symbol.strip()
    ]

    print(f"\nSelected {len(tickers)} stock(s) for backtest:")
    for i, ticker in enumerate(tickers, 1):
        print(f"  {i}. {ticker}")

    return tickers


def fetch_historical_data(
    symbols,
    start_date,
    end_date,
    client,
):
    """
    Fetch historical data from Alpaca

    Args:
        symbols: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        client: AlpacaDataClient instance

    Returns:
        Dictionary of symbol -> DataFrame
    """
    print("\n" + "=" * 60)
    print("STEP 2: Fetching Historical Data from Alpaca")
    print("=" * 60)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Symbols: {len(symbols)}")

    try:
        # Fetch daily data for signal generation
        print("\nFetching daily bars...")
        daily_data = client.fetch_daily_bars(
            symbols=symbols,
            start=start_date,
            end=end_date,
        )

        print(f"Daily data fetched for {len(daily_data)} symbols")

        # Fetch 1-minute data for backtesting
        print("\nFetching 1-minute bars...")
        minute_data = client.fetch_minute_bars(
            symbols=symbols,
            start=start_date,
            end=end_date,
            feed="sip",  # Use SIP for full data
        )

        print(f"Minute data fetched for {len(minute_data)} symbols")

        return daily_data, minute_data

    except Exception as e:
        print(f"\nError fetching data: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure ALPACA_API_KEY and ALPACA_API_SECRET are set in .env file")
        return None, None


def build_symbol_contexts(symbols, start_date, end_date):
    """
    Build point-in-time context inputs for each symbol:
      - market_cap: SimFin daily close * shares outstanding
      - news_count: Alpaca news articles per day
      - social_activity: FMP social activity per day (optional/bonus)

    Returns dict of symbol -> context dict.
    """
    contexts = {s: {} for s in symbols}

    # --- SimFin market cap (needs SIMFIN_API_KEY) ---
    if os.getenv("SIMFIN_API_KEY"):
        try:
            print("Loading SimFin bulk data (share prices)...")
            store = SimFinStore()
            for symbol in symbols:
                cap = store.market_cap_series(symbol)
                if len(cap) > 0:
                    contexts[symbol]["market_cap"] = cap
                else:
                    print(f"  {symbol}: no SimFin market cap data")
        except Exception as e:
            print(f"  SimFin load failed: {e}")
    else:
        print("SIMFIN_API_KEY not set — market cap check will pass-through "
              "(set REQUIRE_MARKET_CAP=1 to require it)")

    # --- Alpaca news (uses same Alpaca keys; no extra setup) ---
    try:
        news_client = AlpacaNewsClient()
        print("Fetching Alpaca news articles...")
        news_df = news_client.fetch_news(symbols, start=start_date, end=end_date)
        print(f"  Fetched {len(news_df)} articles")
        counts = news_client.daily_counts(news_df, symbols)
        for symbol in symbols:
            if symbol in counts.columns:
                contexts[symbol]["news_count"] = counts[symbol]
    except Exception as e:
        print(f"  News fetch failed: {e}")

    # --- FMP social sentiment (optional; legacy endpoint) ---
    if os.getenv("FMP_API_KEY"):
        try:
            print("Fetching FMP social sentiment...")
            sentiment_client = FMPSentimentClient()
            for symbol in symbols:
                activity = sentiment_client.daily_activity(
                    symbol, start=start_date, end=end_date
                )
                if len(activity) > 0:
                    contexts[symbol]["social_activity"] = activity
                else:
                    print(f"  {symbol}: no FMP sentiment data "
                          "(legacy endpoint may be unavailable)")
        except Exception as e:
            print(f"  Sentiment fetch failed: {e}")
    else:
        print("FMP_API_KEY not set — social sentiment (bonus) disabled")

    return contexts


def apply_strategy_to_symbols(
    daily_data,
    minute_data,
    strategy,
    contexts=None,
):
    """
    Apply strategy to all symbols

    Args:
        daily_data: Dictionary of daily data
        minute_data: Dictionary of minute data
        strategy: Strategy instance

    Returns:
        Dictionary of symbol -> DataFrame with signals
    """
    print("\n" + "=" * 60)
    print("STEP 4: Applying Strategy")
    print("=" * 60)

    signals = {}
    total_entries = 0
    total_exits = 0

    for symbol in daily_data.keys():
        if symbol not in minute_data:
            print(f"  {symbol}: Skipping (no minute data)")
            continue

        try:
            daily_df = daily_data[symbol]
            minute_df = minute_data[symbol]

            # Apply strategy
            context = (contexts or {}).get(symbol)
            with_signals = strategy.apply_strategy(daily_df, minute_df, context)
            signals[symbol] = with_signals

            entries = with_signals['entry'].sum()
            exits = with_signals['exit'].sum()
            total_entries += entries
            total_exits += exits

            print(f"  {symbol}: {entries} entry signals, {exits} exit signals")
            candidate_rows = with_signals["relative_volume"].notna()
            candidate_count = int(candidate_rows.sum())
            if candidate_count:
                criteria = [
                    column
                    for column in with_signals.columns
                    if column.startswith("screen_")
                ]
                passed = ", ".join(
                    f"{column.removeprefix('screen_')}="
                    f"{int(with_signals.loc[candidate_rows, column].sum())}/{candidate_count}"
                    for column in criteria
                )
                print(f"    Historical screen pass counts: {passed}")

        except Exception as e:
            print(f"  {symbol}: Error - {e}")

    print(f"\nTotal signals: {total_entries} entries, {total_exits} exits")

    return signals


def run_backtest(
    minute_data: dict,
    signals: dict,
    initial_cash: float = 100_000,
    commission: float = 0.001,
) -> object:
    """
    Run backtest on all symbols

    Args:
        minute_data: Dictionary of minute data
        signals: Dictionary of signals
        initial_cash: Starting capital
        commission: Commission rate

    Returns:
        Portfolio object
    """
    print("\n" + "=" * 60)
    print("STEP 4: Running Backtest")
    print("=" * 60)

    backtest = MultiSymbolBacktest(initial_cash=initial_cash, commission=commission)

    # Prepare data for backtest
    entry_signals = {}
    exit_signals = {}

    for symbol in signals.keys():
        entry_signals[symbol] = signals[symbol]['entry']
        exit_signals[symbol] = signals[symbol]['exit']

    portfolio = backtest.run_multi_symbol_backtest(
        price_data=minute_data,
        entry_signals=entry_signals,
        exit_signals=exit_signals,
        freq='1min',
    )

    return portfolio


def display_results(portfolio):
    """
    Display backtest results

    Args:
        portfolio: Portfolio object
    """
    print("\n" + "=" * 60)
    print("STEP 5: Backtest Results")
    print("=" * 60)

    stats = portfolio.stats()
    print(stats)

    # Calculate custom metrics
    backtest = BacktestEngine()
    metrics = backtest.calculate_metrics(portfolio)

    print("\n" + "=" * 60)
    print("KEY METRICS")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")

    # Trade details
    trades = portfolio.trades.records_readable
    if len(trades) > 0:
        print("\n" + "=" * 60)
        print("RECENT TRADES")
        print("=" * 60)
        print(trades.head(10))


def pit_available():
    return os.path.exists(os.path.join("data", "pit", "candidates.parquet"))


def run_pit_backtest():
    """
    Backtest on the point-in-time dataset in data/pit (built by build_pit_dataset.py):
    delisted names included, daily screen already narrowed to candidate symbol-days,
    1-minute bars only for those days. Trades are computed directly (entry = close of
    the 15:55 bar, exit = close of the first bar of the next session).
    """
    import glob
    import numpy as np
    import pandas as pd

    print("\n" + "=" * 60)
    print("POINT-IN-TIME BACKTEST (data/pit)")
    print("=" * 60)
    COMMISSION = float(os.getenv("BACKTEST_COMMISSION", "0.001"))
    METHOD = os.getenv("RESISTANCE_METHOD", "atr_pivot")
    MAX_POS = int(os.getenv("MAX_POSITIONS_PER_DAY", "5"))
    REQUIRE_NEWS = os.getenv("REQUIRE_NEWS", "0") == "1"
    from src.backtest.exit_variants import VARIANTS

    sel = os.getenv("EXIT_VARIANTS", "v2")
    variants = list(VARIANTS) if sel == "all" else [v.strip() for v in sel.split(",") if v.strip()]
    START = os.getenv("BACKTEST_START_DATE", "2024-10-18")
    END = os.getenv("BACKTEST_END_DATE", "2100-01-01")

    cands = pd.read_parquet("data/pit/candidates.parquet")
    cands = cands[(cands["date"] >= START) & (cands["date"] <= END)]
    daily = pd.read_parquet(
        "data/pit/daily_pit.parquet",
        columns=["date", "symbol", "open", "high", "low", "close", "volume", "delisted_utc"],
    )
    minute = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob("data/pit/alpaca_minute/*.parquet"))],
        ignore_index=True,
    )
    minute = minute[minute["symbol"].isin(set(cands["symbol"]))]
    if str(getattr(minute["timestamp"].dtype, "tz", None)) != "America/New_York":
        minute["timestamp"] = pd.to_datetime(minute["timestamp"], utc=True).dt.tz_convert("America/New_York")
    print(f"candidate symbol-days: {len(cands)}, symbols: {cands['symbol'].nunique()}, "
          f"minute rows: {len(minute)}")

    symbols = sorted(cands["symbol"].unique())
    contexts = build_symbol_contexts_pit(symbols, START, END, REQUIRE_NEWS)

    from src.backtest import pit_runner

    PROCESSES = int(os.getenv("PROCESSES", str(max(1, (os.cpu_count() or 2) - 1))))
    USE_CACHE = os.getenv("LADDER_CACHE", "1") == "1" and METHOD == "atr_pivot"
    cfg = {
        "strategy": dict(
            min_volume=500_000, min_price=2.0, max_price=20.0, max_market_cap=2_000_000_000,
            min_relative_volume=2.0,
            news_lookback_days=int(os.getenv("NEWS_LOOKBACK_DAYS", "3")),
            min_news_articles=int(os.getenv("MIN_NEWS_ARTICLES", "1")),
            require_market_cap=os.getenv("REQUIRE_MARKET_CAP", "0") == "1",
            require_news_catalyst=REQUIRE_NEWS,
            resistance_method=METHOD,
            min_breakout_atr=float(os.getenv("MIN_BREAKOUT_ATR", "0")),
            lazy_resistance=True,
        ),
        "commission": COMMISSION, "variants": {n: VARIANTS[n] for n in variants}, "use_cache": USE_CACHE,
        "realism": {
            "size_usd": float(os.getenv("SIZE_USD", "20000")),
            "participation": float(os.getenv("PARTICIPATION", "0.10")),
            "slip_rth": float(os.getenv("SLIP_RTH", "0.0015")),
            "slip_ext": float(os.getenv("SLIP_EXT", "0.004")),
            "min_fill": float(os.getenv("MIN_FILL", "0.25")),
        },
    }

    daily_by = {s: g.set_index("date").sort_index() for s, g in daily.groupby("symbol")}
    minute_by = {s: g.set_index("timestamp").sort_index() for s, g in minute.groupby("symbol")}
    cand_dates = {s: set(pd.DatetimeIndex(g["date"]).date) for s, g in cands.groupby("symbol")}

    cache = pit_runner.load_cache() if USE_CACHE else {}
    tasks = [
        (s, daily_by[s], minute_by[s], contexts.get(s), cand_dates[s], cfg, cache.get(s))
        for s in symbols if s in daily_by and s in minute_by
    ]
    print(f"processes: {PROCESSES}, ladder cache: {'on, ' + str(len(cache)) + ' symbols cached' if USE_CACHE else 'off'}")
    trades, forced, updates = pit_runner.run_symbols(tasks, PROCESSES)
    if USE_CACHE and updates:
        cache.update(updates)
        pit_runner.save_cache(cache)
        print(f"ladder cache updated for {len(updates)} symbols")

    all_t = pd.DataFrame(trades)
    print("\n" + "=" * 60)
    print(f"RESULTS (resistance={METHOD}, news_required={REQUIRE_NEWS}, commission={COMMISSION})")
    print("=" * 60)
    if all_t.empty:
        print("No trades.")
        return
    from src.backtest.pit_metrics import summarize

    last_day = pd.Timestamp(daily["date"].max())
    test_start = last_day - pd.DateOffset(months=int(os.getenv("TEST_MONTHS", "3")))
    which = os.getenv("BACKTEST_SET", "dev")
    entry_day = pd.to_datetime(all_t["entry_time"].astype(str).str[:10])
    is_test = entry_day >= test_start
    os.makedirs(os.path.join("results", "holdout"), exist_ok=True)
    for name in variants:
        v = all_t[all_t["variant"] == name].drop(columns="variant")
        v[~is_test[v.index]].to_csv(os.path.join("results", f"pit_trades_{METHOD}_{name}.csv"), index=False)
        v[is_test[v.index]].to_csv(os.path.join("results", "holdout", f"pit_trades_{METHOD}_{name}.csv"), index=False)
    shown = all_t[~is_test] if which == "dev" else all_t[is_test] if which == "test" else all_t
    print(f"set shown: {which} (holdout starts {test_start.date()}; {int(is_test.sum() / len(variants))} "
          f"holdout trades kept out of the dev files and not summarized unless BACKTEST_SET=test|all)")
    rows = []
    for name in variants:
        t = shown[shown["variant"] == name]
        row = summarize(t, MAX_POS)
        row["exits"] = ", ".join(f"{k}={n}" for k, n in t["exit_reason"].value_counts().items())
        row["partial_exit"] = int((t["flag"] == "partial_exit").sum())
        rows.append({"variant": name, **row})
    summary = pd.DataFrame(rows).set_index("variant")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print(summary)
    summary.to_csv(os.path.join("results", f"pit_summary_{METHOD}_{which}.csv"))
    r = cfg["realism"]
    print(f"portfolio: top {MAX_POS}/day by relative volume, equal weight, fully invested, compounding")
    print(f"fills: ${r['size_usd']:,.0f}/trade, <= {r['participation']:.0%} of bar volume, slippage "
          f"{r['slip_rth']:.2%} regular / {r['slip_ext']:.2%} extended per side, commission {COMMISSION:.2%} per side")
    first = shown[shown["variant"] == variants[0]]
    print(f"trades in later-delisted names: {int(first['delisted'].sum())}; skipped for unfilled entry: see counts; "
          f"entries without next-session bars: {forced // len(variants)}")


def build_symbol_contexts_pit(symbols, start, end, with_news):
    """Market cap from SimFin (pass-through when missing); Alpaca news only when required."""
    if with_news:
        return build_symbol_contexts(symbols, start, end)
    contexts = {s: {} for s in symbols}
    if os.getenv("SIMFIN_API_KEY"):
        try:
            store = SimFinStore()
            for s in symbols:
                cap = store.market_cap_series(s)
                if len(cap) > 0:
                    contexts[s]["market_cap"] = cap
        except Exception as e:
            print(f"  SimFin load failed: {e}")
    return contexts


def main():
    """Main application entry point"""
    print("\n" + "=" * 60)
    print("HUMBLED TRADER BULLISH OVERNIGHT STRATEGY")
    print("=" * 60)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if os.getenv("BACKTEST_MODE", "pit" if pit_available() else "symbols") == "pit":
        return run_pit_backtest()

    START_DATE = os.getenv("BACKTEST_START_DATE", "2024-01-01")
    END_DATE = os.getenv(
        "BACKTEST_END_DATE",
        (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    INITIAL_CASH = float(os.getenv("BACKTEST_INITIAL_CASH", "100000"))
    COMMISSION = float(os.getenv("BACKTEST_COMMISSION", "0.001"))

    # Step 1: Get backtest universe
    tickers = get_backtest_universe()

    if not tickers:
        print("No tickers found. Exiting.")
        return

    # Step 2: Initialize Alpaca client and fetch data
    try:
        client = AlpacaDataClient()
        daily_data, minute_data = fetch_historical_data(
            symbols=tickers,
            start_date=START_DATE,
            end_date=END_DATE,
            client=client,
        )

        if daily_data is None or minute_data is None:
            print("Failed to fetch data. Exiting.")
            return

    except ValueError as e:
        print(f"Error: {e}")
        print("Please set ALPACA_API_KEY and ALPACA_API_SECRET in .env file")
        return

    # Step 3: Build point-in-time context (market cap, news, sentiment)
    print("\n" + "=" * 60)
    print("STEP 3: Building Point-in-Time Context")
    print("=" * 60)
    contexts = build_symbol_contexts(tickers, START_DATE, END_DATE)

    # Step 4: Initialize strategy and apply it
    strategy = HumbledBullishOvernight(
        min_volume=500_000,
        min_price=2.0,
        max_price=20.0,
        max_market_cap=2_000_000_000,
        min_relative_volume=2.0,
        news_lookback_days=int(os.getenv("NEWS_LOOKBACK_DAYS", "3")),
        min_news_articles=int(os.getenv("MIN_NEWS_ARTICLES", "1")),
        require_market_cap=os.getenv("REQUIRE_MARKET_CAP", "0") == "1",
        require_news_catalyst=os.getenv("REQUIRE_NEWS", "1") == "1",
        resistance_method=os.getenv("RESISTANCE_METHOD", "atr_pivot"),
        min_breakout_atr=float(os.getenv("MIN_BREAKOUT_ATR", "0")),
    )

    # Step 4: Apply strategy
    signals = apply_strategy_to_symbols(daily_data, minute_data, strategy, contexts)

    if not signals:
        print("No signals generated. Exiting.")
        return

    # Check if there are any entry signals
    total_entries = sum(signals[s]['entry'].sum() for s in signals)
    if total_entries == 0:
        print("\nNo entry signals generated. Strategy criteria may be too strict.")
        print("Consider adjusting strategy parameters or using different symbols.")
        return

    # Step 5: Run backtest
    portfolio = run_backtest(
        minute_data=minute_data,
        signals=signals,
        initial_cash=INITIAL_CASH,
        commission=COMMISSION,
    )

    # Step 6: Display results
    display_results(portfolio)

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
