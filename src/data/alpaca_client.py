"""
Alpaca data client for historical stock data
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
import pytz
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

load_dotenv()


class AlpacaDataClient:
    """Client for fetching historical data from Alpaca"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize Alpaca data client

        Args:
            api_key: Alpaca API key (defaults to env var ALPACA_API_KEY)
            api_secret: Alpaca API secret (defaults to env var ALPACA_API_SECRET)
            base_url: Alpaca base URL (defaults to env var ALPACA_BASE_URL)
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        self.base_url = base_url or os.getenv("ALPACA_BASE_URL", "https://data.alpaca.markets")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca API credentials not found. "
                "Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables."
            )

        # Use the new Alpaca data SDK
        self.client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
        )

        self.ny_tz = pytz.timezone("America/New_York")

    def fetch_bars(
        self,
        symbols: List[str],
        start: str,
        end: str,
        timeframe: str = "1Min",
        feed: str = "sip",
        adjustment: str = "raw",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical bars for multiple symbols

        Args:
            symbols: List of ticker symbols
            start: Start date (YYYY-MM-DD or datetime)
            end: End date (YYYY-MM-DD or datetime)
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            feed: Data feed ('sip' for full SIP, 'iex' for IEX)
            adjustment: Adjustment type ('raw', 'split', 'dividend')

        Returns:
            Dictionary mapping symbol to DataFrame with OHLCV data
        """
        # Convert timeframe string to TimeFrame enum
        timeframe_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrame.Minute),
            "15Min": TimeFrame(15, TimeFrame.Minute),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }

        tf = timeframe_map.get(timeframe, TimeFrame.Minute)

        # Convert dates
        if isinstance(start, str):
            start = pd.to_datetime(start).tz_localize(self.ny_tz)
        if isinstance(end, str):
            end = pd.to_datetime(end).tz_localize(self.ny_tz)

        # Fetch data
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start,
            end=end,
            feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX,
            adjustment=adjustment,
        )

        bars = self.client.get_stock_bars(request_params)

        # Convert to dictionary of DataFrames
        result = {}

        # Handle different response structures
        if hasattr(bars, 'data'):
            # New API structure
            for symbol in symbols:
                if symbol in bars.data:
                    symbol_data = bars.data[symbol]
                    if symbol_data:
                        df = pd.DataFrame([bar.model_dump() for bar in symbol_data])
                        if not df.empty:
                            df = df.set_index('timestamp')
                            # Ensure timezone is set
                            if df.index.tz is None:
                                df.index = df.index.tz_localize(self.ny_tz)
                            result[symbol] = df
        else:
            # Legacy structure
            for symbol in symbols:
                try:
                    symbol_bars = bars[symbol]
                    if symbol_bars:
                        df = symbol_bars.df
                        # Ensure timezone is set
                        if df.index.tz is None:
                            df.index = df.index.tz_localize(self.ny_tz)
                        result[symbol] = df
                except (KeyError, AttributeError):
                    continue

        return result

    def fetch_daily_bars(
        self,
        symbols: List[str],
        start: str,
        end: str,
        adjustment: str = "split",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch daily bars for universe selection"""
        return self.fetch_bars(
            symbols=symbols,
            start=start,
            end=end,
            timeframe="1Day",
            feed="sip",
            adjustment=adjustment,
        )

    def fetch_minute_bars(
        self,
        symbols: List[str],
        start: str,
        end: str,
        feed: str = "sip",
        adjustment: str = "split",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch 1-minute bars for intraday backtesting"""
        return self.fetch_bars(
            symbols=symbols,
            start=start,
            end=end,
            timeframe="1Min",
            feed=feed,
            adjustment=adjustment,
        )

    def save_to_parquet(
        self,
        data: Dict[str, pd.DataFrame],
        output_dir: str,
        timeframe: str = "1Min",
    ):
        """
        Save fetched data to parquet files

        Args:
            data: Dictionary of symbol -> DataFrame
            output_dir: Directory to save parquet files
            timeframe: Timeframe for folder naming
        """
        os.makedirs(output_dir, exist_ok=True)

        for symbol, df in data.items():
            # Clean symbol name for filename
            clean_symbol = symbol.replace("/", "_")
            filepath = os.path.join(output_dir, f"{clean_symbol}_{timeframe}.parquet")
            df.to_parquet(filepath)
            print(f"Saved {symbol} to {filepath}")

    def load_from_parquet(
        self,
        symbols: List[str],
        data_dir: str,
        timeframe: str = "1Min",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data from parquet files

        Args:
            symbols: List of ticker symbols
            data_dir: Directory containing parquet files
            timeframe: Timeframe for file naming

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        result = {}
        for symbol in symbols:
            clean_symbol = symbol.replace("/", "_")
            filepath = os.path.join(data_dir, f"{clean_symbol}_{timeframe}.parquet")

            if os.path.exists(filepath):
                result[symbol] = pd.read_parquet(filepath)
            else:
                print(f"Warning: {filepath} not found")

        return result
