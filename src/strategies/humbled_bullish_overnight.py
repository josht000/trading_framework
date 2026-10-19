"""
Humbled Trader Bullish Overnight Strategy

Entry Criteria:
- Daily candle closes higher than previous candle
- Open is higher than previous close
- Price breaks out over previous resistance on high volume
- Closing high above VWAP
- Significant increase in volume in past few days
- Volume > 500k shares daily
- Fundamental catalyst (positive news)
- (Bonus) Sector sympathy and sector hype

Entry:
- Buy near EOD during power hour (last hour of trading)

Exit:
- Sell for pre-market gap up OR early morning spike (first 30 min)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from src.screening import HistoricalPriceVolumeScreener


class HumbledBullishOvernight:
    """
    Humbled Trader Bullish Overnight strategy implementation
    """

    def __init__(
        self,
        min_volume: int = 500_000,
        min_price: float = 1.0,
        max_price: float = 20.0,
        max_market_cap: float = 2_000_000_000,  # $2B
        power_hour_start: str = "15:00",
        power_hour_end: str = "16:00",
        entry_time: str = "15:55",
        early_morning_end: str = "10:00",
        min_relative_volume: float = 2.0,
        news_lookback_days: int = 3,
        min_news_articles: int = 1,
        require_market_cap: bool = False,
        require_news_catalyst: bool = True,
        social_surge_factor: float = 2.0,
        resistance_method: str = "rolling",
        min_breakout_atr: float = 0.0,
        lazy_resistance: bool = False,
    ):
        """
        Initialize strategy parameters

        Args:
            min_volume: Minimum daily volume (shares)
            min_price: Minimum stock price
            max_price: Maximum stock price
            max_market_cap: Maximum market cap (in USD)
            power_hour_start: Start of power hour (HH:MM)
            power_hour_end: End of power hour (HH:MM)
            early_morning_end: End of early morning exit window (HH:MM)
        """
        self.min_volume = min_volume
        self.min_price = min_price
        self.max_price = max_price
        self.max_market_cap = max_market_cap
        self.power_hour_start = power_hour_start
        self.power_hour_end = power_hour_end
        self.entry_time = entry_time
        self.early_morning_end = early_morning_end
        self.min_relative_volume = min_relative_volume
        self.screener = HistoricalPriceVolumeScreener(
            min_price=min_price,
            max_price=max_price,
            min_average_volume=min_volume,
            min_relative_volume=min_relative_volume,
            max_market_cap=max_market_cap,
            entry_time=entry_time,
            news_lookback_days=news_lookback_days,
            min_news_articles=min_news_articles,
            require_market_cap=require_market_cap,
            require_news_catalyst=require_news_catalyst,
            social_surge_factor=social_surge_factor,
            resistance_method=resistance_method,
            min_breakout_atr=min_breakout_atr,
            lazy_resistance=lazy_resistance,
        )

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price

        Args:
            df: DataFrame with OHLCV data

        Returns:
            VWAP series
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return vwap

    def detect_resistance_breakout(
        self, df: pd.DataFrame, lookback: int = 20
    ) -> pd.Series:
        """
        Detect when price breaks above previous resistance

        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to look back for resistance

        Returns:
            Boolean series indicating breakout
        """
        # Calculate rolling high as resistance
        resistance = df["high"].rolling(window=lookback).max().shift(1)

        # Breakout when current high exceeds previous resistance
        breakout = df["high"] > resistance

        return breakout

    def check_volume_surge(
        self, df: pd.DataFrame, lookback: int = 5, threshold: float = 1.5
    ) -> pd.Series:
        """
        Check for significant volume increase

        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to compare
            threshold: Volume multiplier threshold

        Returns:
            Boolean series indicating volume surge
        """
        avg_volume = df["volume"].rolling(window=lookback).mean().shift(1)
        volume_surge = df["volume"] > (avg_volume * threshold)

        return volume_surge

    def generate_daily_signals(
        self, daily_df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry signals based on daily data

        Args:
            daily_df: Daily OHLCV DataFrame

        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        df = daily_df.copy()

        # Price filters
        price_filter = (df["close"] >= self.min_price) & (df["close"] <= self.max_price)

        # Volume filter
        volume_filter = df["volume"] >= self.min_volume

        # Daily candle criteria
        # Close higher than previous close
        higher_close = df["close"] > df["close"].shift(1)

        # Open higher than previous close
        higher_open = df["open"] > df["close"].shift(1)

        # VWAP calculation
        vwap = self.calculate_vwap(df)
        above_vwap = df["close"] > vwap

        # Resistance breakout on high volume
        breakout = self.detect_resistance_breakout(df, lookback=20)
        volume_surge = self.check_volume_surge(df, lookback=5, threshold=1.5)

        # Combine all entry criteria
        entry_signals = (
            price_filter
            & volume_filter
            & higher_close
            & higher_open
            & above_vwap
            & breakout
            & volume_surge
        )

        # Exit signals: default to next day close (will be refined with intraday data)
        exit_signals = entry_signals.shift(1).fillna(False)

        return entry_signals, exit_signals

    def generate_intraday_entry(
        self, intraday_df: pd.DataFrame, daily_signal: bool
    ) -> pd.Series:
        """
        Generate intraday entry signals during power hour

        Args:
            intraday_df: Intraday (1-min) OHLCV DataFrame
            daily_signal: Whether daily criteria were met

        Returns:
            Boolean series indicating entry during power hour
        """
        if not daily_signal:
            return pd.Series(False, index=intraday_df.index)

        # Filter to power hour
        start_time = pd.to_datetime(self.power_hour_start).time()
        end_time = pd.to_datetime(self.power_hour_end).time()

        times = pd.Series(intraday_df.index.time)
        power_hour = (times >= start_time) & (times <= end_time)

        return power_hour

    def generate_intraday_exit(self, intraday_df: pd.DataFrame) -> pd.Series:
        """
        Generate intraday exit signals during early morning

        Args:
            intraday_df: Intraday (1-min) OHLCV DataFrame

        Returns:
            Boolean series indicating exit during early morning
        """
        # Filter to early morning (first 30 minutes or until 10:00)
        start_time = pd.to_datetime("09:30").time()
        end_time = pd.to_datetime(self.early_morning_end).time()

        times = pd.Series(intraday_df.index.time)
        early_morning = (times >= start_time) & (times <= end_time)

        return early_morning

    def apply_strategy(
        self,
        daily_df: pd.DataFrame,
        intraday_df: pd.DataFrame,
        context: Optional[Dict] = None,
        entry_dates: Optional[set] = None,
    ) -> pd.DataFrame:
        """
        Args:
            entry_dates: optional set of session dates; entries are only allowed on these dates,
                and exits must land on the next trading day of daily_df
            daily_df: Daily OHLCV DataFrame
            intraday_df: Intraday (1-min) OHLCV DataFrame
            context: Optional dict of point-in-time Series keyed by date:
                "market_cap" (SimFin close*shares outstanding),
                "news_count" (Alpaca news articles per day),
                "social_activity" (FMP social activity per day, optional)
        """
        """
        Apply the full strategy combining daily and intraday data

        Args:
            daily_df: Daily OHLCV DataFrame
            intraday_df: Intraday (1-min) OHLCV DataFrame

        Returns:
            DataFrame with entry and exit signals
        """
        screen = self.screener.scan(daily_df, intraday_df, context=context)
        intraday_with_signals = intraday_df.copy().sort_index()
        if intraday_with_signals.index.tz is None:
            intraday_with_signals.index = intraday_with_signals.index.tz_localize(
                self.screener.timezone
            )
        else:
            intraday_with_signals.index = intraday_with_signals.index.tz_convert(
                self.screener.timezone
            )
        intraday_with_signals = intraday_with_signals[
            ~intraday_with_signals.index.duplicated(keep="last")
        ]
        entry_signals = screen["eligible"].reindex(
            intraday_with_signals.index, fill_value=False
        )
        if entry_dates is not None:
            entry_signals = entry_signals & pd.Series(
                [d in entry_dates for d in entry_signals.index.date], index=entry_signals.index
            )
        exit_signals = pd.Series(False, index=intraday_with_signals.index)
        session_dates = pd.Index(intraday_with_signals.index.date)
        available_dates = pd.Index(session_dates.unique()).sort_values()
        daily_dates = pd.Index(pd.DatetimeIndex(daily_df.index).date) if entry_dates is not None else None
        for entry_index in entry_signals[entry_signals].index:
            later_dates = available_dates[available_dates > entry_index.date()]
            if later_dates.empty:
                continue
            next_date = later_dates[0]
            if daily_dates is not None:
                next_daily = daily_dates[daily_dates > entry_index.date()]
                if not next_daily.empty and next_daily[0] != next_date:
                    entry_signals.at[entry_index] = False
                    continue
            next_session = intraday_with_signals.loc[session_dates == next_date]
            regular_session = next_session.between_time(
                "09:30", self.early_morning_end, inclusive="both"
            )
            if not regular_session.empty:
                exit_signals.at[regular_session.index[0]] = True
        intraday_with_signals["entry"] = entry_signals.astype(bool)
        intraday_with_signals["exit"] = exit_signals
        intraday_with_signals["relative_volume"] = screen[
            "relative_volume_value"
        ].reindex(intraday_with_signals.index)
        intraday_with_signals["session_vwap"] = screen["session_vwap"].reindex(
            intraday_with_signals.index
        )
        for column in (
            "price",
            "average_volume",
            "relative_volume",
            "higher_close",
            "higher_open",
            "resistance_breakout",
            "above_vwap",
            "market_cap",
            "news_catalyst",
            "social_surge",
        ):
            intraday_with_signals[f"screen_{column}"] = screen[column].reindex(
                intraday_with_signals.index, fill_value=False
            )
        return intraday_with_signals

    def get_strategy_params(self) -> Dict:
        """Return strategy parameters for logging/config"""
        return {
            "min_volume": self.min_volume,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "max_market_cap": self.max_market_cap,
            "power_hour_start": self.power_hour_start,
            "power_hour_end": self.power_hour_end,
            "entry_time": self.entry_time,
            "early_morning_end": self.early_morning_end,
            "min_relative_volume": self.min_relative_volume,
        }
