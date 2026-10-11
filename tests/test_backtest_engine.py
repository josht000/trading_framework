import unittest

import pandas as pd

from src.backtest.engine import MultiSymbolBacktest


def make_minute_index():
    return pd.date_range(
        "2024-02-06 09:30", periods=4, freq="min", tz="America/New_York"
    )


def make_price_data(index):
    return {
        "AAA": pd.DataFrame(
            {"close": [10.0, 10.1, 10.2, 10.3]}, index=index, dtype=float
        ),
        "BBB": pd.DataFrame(
            {"close": [20.0, 20.1, 20.2, 20.3]}, index=index, dtype=float
        ),
    }


class MultiSymbolBacktestTests(unittest.TestCase):
    def test_two_symbols_each_complete_one_round_trip(self):
        index = make_minute_index()
        price_data = make_price_data(index)
        entry_signals = {
            symbol: pd.Series([True, False, False, False], index=index, dtype=bool)
            for symbol in price_data
        }
        exit_signals = {
            symbol: pd.Series([False, False, True, False], index=index, dtype=bool)
            for symbol in price_data
        }

        portfolio = MultiSymbolBacktest(
            initial_cash=100_000, commission=0
        ).run_multi_symbol_backtest(price_data, entry_signals, exit_signals)

        orders = portfolio.orders.records_readable
        trades = portfolio.trades.records_readable

        self.assertEqual(len(orders), 4)
        self.assertEqual(
            sorted(orders["Side"].tolist()), ["Buy", "Buy", "Sell", "Sell"]
        )
        self.assertEqual(set(orders["Column"]), {"AAA", "BBB"})

        self.assertEqual(len(trades), 2)
        self.assertEqual(set(trades["Column"]), {"AAA", "BBB"})
        self.assertTrue((trades["Status"] == "Closed").all())


if __name__ == "__main__":
    unittest.main()
