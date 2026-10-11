import unittest

import numpy as np
import pandas as pd

from src.screening import HistoricalPriceVolumeScreener
from src.strategies.humbled_bullish_overnight import HumbledBullishOvernight


def make_daily_data():
    index = pd.date_range(
        "2024-01-02", periods=25, freq="B", tz="America/New_York"
    )
    close = np.linspace(8.0, 10.0, len(index))
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.25,
            "low": close - 0.25,
            "close": close,
            "volume": 600_000,
        },
        index=index,
    )


def make_minute_data(volume=4_000):
    signal_day = pd.date_range(
        "2024-02-06 09:30",
        "2024-02-06 15:59",
        freq="min",
        tz="America/New_York",
    )
    exit_day = pd.date_range(
        "2024-02-07 09:30",
        "2024-02-07 10:00",
        freq="min",
        tz="America/New_York",
    )
    index = signal_day.append(exit_day)
    close = np.concatenate(
        [
            np.linspace(11.0, 12.0, len(signal_day)),
            np.linspace(12.5, 12.2, len(exit_day)),
        ]
    )
    return pd.DataFrame(
        {
            "open": close - 0.01,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


NEWS_CONTEXT = {
    "news_count": pd.Series([1.0], index=pd.DatetimeIndex(["2024-02-06"]))
}


class HistoricalPriceVolumeScreenerTests(unittest.TestCase):
    def test_scans_at_entry_time_using_only_available_data(self):
        daily = make_daily_data()
        minute = make_minute_data()
        screener = HistoricalPriceVolumeScreener()

        result = screener.scan(daily, minute, context=NEWS_CONTEXT)
        entry_index = pd.Timestamp("2024-02-06 15:55", tz="America/New_York")

        self.assertTrue(result.at[entry_index, "eligible"])
        self.assertEqual(int(result["eligible"].sum()), 1)

        changed = minute.copy()
        changed.loc[changed.index > entry_index, ["high", "low", "close", "volume"]] = [
            1_000.0,
            0.01,
            0.01,
            100_000_000,
        ]
        changed_result = screener.scan(daily, changed, context=NEWS_CONTEXT)
        self.assertEqual(
            result.loc[entry_index].to_dict(),
            changed_result.loc[entry_index].to_dict(),
        )

    def test_rejects_insufficient_relative_volume(self):
        result = HistoricalPriceVolumeScreener().scan(
            make_daily_data(), make_minute_data(volume=500)
        )
        self.assertFalse(result["eligible"].any())

    def test_news_catalyst_blocks_when_required_and_absent(self):
        screener = HistoricalPriceVolumeScreener(require_news_catalyst=True)
        result = screener.scan(
            make_daily_data(), make_minute_data(),
            context={"news_count": pd.Series(dtype=float)},
        )
        entry_index = pd.Timestamp("2024-02-06 15:55", tz="America/New_York")
        self.assertFalse(result.at[entry_index, "news_catalyst"])
        self.assertFalse(result.at[entry_index, "eligible"])

    def test_news_catalyst_passes_with_recent_articles(self):
        news_count = pd.Series(
            [1.0],
            index=pd.DatetimeIndex(["2024-02-06"]),
        )
        screener = HistoricalPriceVolumeScreener(require_news_catalyst=True)
        result = screener.scan(
            make_daily_data(), make_minute_data(),
            context={"news_count": news_count},
        )
        entry_index = pd.Timestamp("2024-02-06 15:55", tz="America/New_York")
        self.assertTrue(result.at[entry_index, "news_catalyst"])
        self.assertTrue(result.at[entry_index, "eligible"])

    def test_market_cap_blocks_over_limit_when_required(self):
        market_cap = pd.Series(
            [5e9], index=pd.DatetimeIndex(["2024-02-05"])
        )
        screener = HistoricalPriceVolumeScreener(require_market_cap=True)
        result = screener.scan(
            make_daily_data(), make_minute_data(),
            context={"market_cap": market_cap},
        )
        entry_index = pd.Timestamp("2024-02-06 15:55", tz="America/New_York")
        self.assertFalse(result.at[entry_index, "market_cap"])
        self.assertFalse(result.at[entry_index, "eligible"])

    def test_market_cap_passes_under_limit(self):
        market_cap = pd.Series(
            [1e9], index=pd.DatetimeIndex(["2024-02-05"])
        )
        screener = HistoricalPriceVolumeScreener(require_market_cap=True)
        context = {"market_cap": market_cap, **NEWS_CONTEXT}
        result = screener.scan(
            make_daily_data(), make_minute_data(), context=context
        )
        entry_index = pd.Timestamp("2024-02-06 15:55", tz="America/New_York")
        self.assertTrue(result.at[entry_index, "market_cap"])
        self.assertTrue(result.at[entry_index, "eligible"])


class HumbledBullishOvernightTests(unittest.TestCase):
    def test_enters_once_and_exits_next_regular_session(self):
        result = HumbledBullishOvernight().apply_strategy(
            make_daily_data(), make_minute_data(), context=NEWS_CONTEXT
        )

        self.assertEqual(
            result.index[result["entry"]].tolist(),
            [pd.Timestamp("2024-02-06 15:55", tz="America/New_York")],
        )
        self.assertEqual(
            result.index[result["exit"]].tolist(),
            [pd.Timestamp("2024-02-07 09:30", tz="America/New_York")],
        )


if __name__ == "__main__":
    unittest.main()
