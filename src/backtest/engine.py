"""
Backtesting engine using vectorbt
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


class BacktestEngine:
    """
    Vectorbt-based backtesting engine
    """

    def __init__(self, initial_cash: float = 100_000, commission: float = 0.0):
        """
        Initialize backtesting engine

        Args:
            initial_cash: Starting capital
            commission: Commission per trade (as decimal, e.g., 0.001 for 0.1%)
        """
        self.initial_cash = initial_cash
        self.commission = commission

    def run_backtest(
        self,
        price_data: pd.DataFrame,
        entry_signals: pd.DataFrame,
        exit_signals: pd.DataFrame,
        freq: str = "1min",
    ) -> vbt.Portfolio:
        """
        Run a backtest using vectorbt

        Args:
            price_data: DataFrame with price data (close column or OHLCV)
            entry_signals: DataFrame with entry signals (boolean)
            exit_signals: DataFrame with exit signals (boolean)
            freq: Frequency of the data (e.g., '1min', '5min', '1h', '1d')

        Returns:
            vectorbt Portfolio object
        """
        # Ensure we have close prices
        if isinstance(price_data, pd.DataFrame):
            if "close" in price_data.columns:
                close = price_data["close"]
            else:
                close = price_data.iloc[:, 0]  # Use first column
        else:
            close = price_data

        # Create portfolio from signals
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entry_signals,
            exits=exit_signals,
            init_cash=self.initial_cash,
            fees=self.commission,
            freq=freq,
        )

        return portfolio

    def run_backtest_with_orders(
        self,
        price_data: pd.DataFrame,
        entry_signals: pd.DataFrame,
        exit_signals: pd.DataFrame,
        size: Optional[float] = None,
        size_type: str = "amount",
        freq: str = "1min",
    ) -> vbt.Portfolio:
        """
        Run backtest with custom order sizing

        Args:
            price_data: DataFrame with price data
            entry_signals: DataFrame with entry signals
            exit_signals: DataFrame with exit signals
            size: Order size (if None, uses default sizing)
            size_type: 'amount' (cash) or 'shares' or 'percent'
            freq: Frequency of the data

        Returns:
            vectorbt Portfolio object
        """
        if isinstance(price_data, pd.DataFrame):
            if "close" in price_data.columns:
                close = price_data["close"]
            else:
                close = price_data.iloc[:, 0]
        else:
            close = price_data

        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entry_signals,
            exits=exit_signals,
            init_cash=self.initial_cash,
            fees=self.commission,
            size=size,
            size_type=size_type,
            freq=freq,
        )

        return portfolio

    def calculate_metrics(self, portfolio: vbt.Portfolio) -> Dict:
        """
        Calculate performance metrics from portfolio

        Args:
            portfolio: vectorbt Portfolio object

        Returns:
            Dictionary of performance metrics
        """
        stats = portfolio.stats()

        # Handle different key names in vectorbt versions
        metrics = {
            "total_return": stats.get("Total Return [%]", stats.get("Total Return", 0)),
            "annual_return": stats.get("Annual Return [%]", stats.get("Annual Return", 0)),
            "sharpe_ratio": stats.get("Sharpe Ratio", 0),
            "sortino_ratio": stats.get("Sortino Ratio", 0),
            "max_drawdown": stats.get("Max Drawdown [%]", stats.get("Max Drawdown", 0)),
            "win_rate": stats.get("Win Rate [%]", stats.get("Win Rate", 0)),
            "total_trades": stats.get("# Trades", stats.get("Total Trades", 0)),
            "avg_trade": stats.get("Avg Trade [%]", stats.get("Avg Trade", 0)),
            "best_trade": stats.get("Best Trade [%]", stats.get("Best Trade", 0)),
            "worst_trade": stats.get("Worst Trade [%]", stats.get("Worst Trade", 0)),
            "profit_factor": stats.get("Profit Factor", 0),
        }

        return metrics

    def print_metrics(self, portfolio: vbt.Portfolio):
        """
        Print performance metrics

        Args:
            portfolio: vectorbt Portfolio object
        """
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)

        stats = portfolio.stats()
        print(stats)

        print("\n" + "=" * 50)
        print("TRADES")
        print("=" * 50)
        trades = portfolio.trades.records_readable
        print(trades)

    def plot_equity_curve(self, portfolio: vbt.Portfolio, save_path: Optional[str] = None):
        """
        Plot equity curve

        Args:
            portfolio: vectorbt Portfolio object
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt

        equity = portfolio.value()

        plt.figure(figsize=(12, 6))
        equity.plot()
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Equity curve saved to {save_path}")
        else:
            plt.show()

    def plot_drawdown(self, portfolio: vbt.Portfolio, save_path: Optional[str] = None):
        """
        Plot drawdown

        Args:
            portfolio: vectorbt Portfolio object
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt

        drawdown = portfolio.drawdown()

        plt.figure(figsize=(12, 6))
        drawdown.plot()
        plt.title("Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown %")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Drawdown plot saved to {save_path}")
        else:
            plt.show()

    def get_trade_records(self, portfolio: vbt.Portfolio) -> pd.DataFrame:
        """
        Get detailed trade records

        Args:
            portfolio: vectorbt Portfolio object

        Returns:
            DataFrame with trade details
        """
        return portfolio.trades.records_readable


class MultiSymbolBacktest:
    """
    Backtest multiple symbols simultaneously
    """

    def __init__(self, initial_cash: float = 100_000, commission: float = 0.0):
        """
        Initialize multi-symbol backtester

        Args:
            initial_cash: Starting capital
            commission: Commission per trade
        """
        self.initial_cash = initial_cash
        self.commission = commission

    def run_multi_symbol_backtest(
        self,
        price_data: Dict[str, pd.DataFrame],
        entry_signals: Dict[str, pd.Series],
        exit_signals: Dict[str, pd.Series],
        freq: str = "1min",
    ) -> vbt.Portfolio:
        """
        Run backtest across multiple symbols

        Args:
            price_data: Dictionary of symbol -> DataFrame with OHLCV
            entry_signals: Dictionary of symbol -> Series with entry signals
            exit_signals: Dictionary of symbol -> Series with exit signals
            freq: Frequency of the data

        Returns:
            vectorbt Portfolio object
        """
        # Align all data to common index
        # For now, assume data is already aligned

        # Build multi-symbol DataFrames
        def _clean(s):
            s = s[~s.index.duplicated(keep="last")].sort_index()
            if s.index.tz is not None:
                s.index = s.index.tz_convert("UTC")
            return s

        closes = {}
        for symbol, df in price_data.items():
            col = df["close"] if "close" in df.columns else df.iloc[:, 0]
            closes[symbol] = _clean(col.astype(float))
        close_prices = pd.DataFrame(closes).sort_index().ffill()

        def _signals(signals):
            out = {}
            for symbol in close_prices.columns:
                s = signals.get(symbol)
                out[symbol] = (
                    _clean(s.astype(bool)).reindex(close_prices.index, fill_value=False)
                    if s is not None
                    else pd.Series(False, index=close_prices.index)
                )
            return pd.DataFrame(out)

        entries = _signals(entry_signals)
        exits = _signals(exit_signals)

        # Create portfolio
        portfolio = vbt.Portfolio.from_signals(
            close=close_prices,
            entries=entries,
            exits=exits,
            init_cash=self.initial_cash,
            fees=self.commission,
            freq=freq,
        )

        return portfolio
