"""
Utility functions for the trading framework
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os
import pytz


def ensure_timezone(df: pd.DataFrame, timezone: str = "America/New_York") -> pd.DataFrame:
    """
    Ensure DataFrame has timezone-aware DatetimeIndex

    Args:
        df: DataFrame with DatetimeIndex
        timezone: Timezone to localize to

    Returns:
        DataFrame with timezone-aware index
    """
    tz = pytz.timezone(timezone)

    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    elif df.index.tz != tz:
        df.index = df.index.tz_convert(tz)

    return df


def filter_trading_hours(
    df: pd.DataFrame,
    start: str = "09:30",
    end: str = "16:00",
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """
    Filter DataFrame to trading hours

    Args:
        df: DataFrame with DatetimeIndex
        start: Start time (HH:MM)
        end: End time (HH:MM)
        timezone: Timezone

    Returns:
        Filtered DataFrame
    """
    tz = pytz.timezone(timezone)

    start_time = pd.to_datetime(start).time()
    end_time = pd.to_datetime(end).time()

    mask = df.index.time.between(start_time, end_time)
    return df[mask]


def resample_to_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    agg_dict: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Resample DataFrame to different timeframe

    Args:
        df: DataFrame with OHLCV data
        timeframe: Resampling rule (e.g., '5min', '1h', '1d')
        agg_dict: Custom aggregation dictionary

    Returns:
        Resampled DataFrame
    """
    if agg_dict is None:
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

    return df.resample(timeframe).agg(agg_dict).dropna()


def align_dataframes(
    dfs: List[pd.DataFrame],
    method: str = "inner",
) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames to common index

    Args:
        dfs: List of DataFrames to align
        method: Alignment method ('inner', 'outer', 'left', 'right')

    Returns:
        List of aligned DataFrames
    """
    if not dfs:
        return []

    # Get common index
    if method == "inner":
        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.intersection(df.index)
    elif method == "outer":
        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.union(df.index)
    else:
        common_index = dfs[0].index

    # Reindex all DataFrames
    aligned = []
    for df in dfs:
        aligned_df = df.reindex(common_index)
        aligned.append(aligned_df)

    return aligned


def calculate_returns(
    prices: pd.Series,
    periods: int = 1,
    log_returns: bool = False,
) -> pd.Series:
    """
    Calculate returns from price series

    Args:
        prices: Price series
        periods: Number of periods
        log_returns: Whether to calculate log returns

    Returns:
        Returns series
    """
    if log_returns:
        return np.log(prices / prices.shift(periods))
    else:
        return prices.pct_change(periods)


def calculate_rolling_metrics(
    prices: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """
    Calculate rolling metrics

    Args:
        prices: Price series
        window: Rolling window size

    Returns:
        DataFrame with rolling metrics
    """
    df = pd.DataFrame(index=prices.index)
    df["returns"] = calculate_returns(prices)
    df["rolling_mean"] = prices.rolling(window).mean()
    df["rolling_std"] = prices.rolling(window).std()
    df["rolling_volatility"] = df["returns"].rolling(window).std() * np.sqrt(252)
    df["rolling_sharpe"] = df["returns"].rolling(window).mean() / df["returns"].rolling(window).std() * np.sqrt(252)

    return df


def save_results(
    results: Dict,
    output_dir: str,
    filename: str,
) -> None:
    """
    Save results to file

    Args:
        results: Dictionary of results
        output_dir: Output directory
        filename: Filename
    """
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    if isinstance(results, pd.DataFrame):
        results.to_csv(filepath)
    elif isinstance(results, dict):
        pd.DataFrame([results]).to_csv(filepath, index=False)
    else:
        with open(filepath, "w") as f:
            f.write(str(results))

    print(f"Results saved to {filepath}")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    import json

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def split_train_test(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
) -> tuple:
    """
    Split DataFrame into train and test sets

    Args:
        df: DataFrame to split
        train_ratio: Ratio of training data

    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    return train_df, test_df
