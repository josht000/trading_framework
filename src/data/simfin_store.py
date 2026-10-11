"""
SimFin bulk fundamentals store.

Downloads SimFin bulk datasets once and builds a local parquet-backed store
for point-in-time lookups (market cap from daily share prices * shares
outstanding, and quarterly fundamentals keyed on publish date).

Note on point-in-time integrity: SimFin bulk files are periodically
restated snapshots, not a historical database of what was known when.
Fundamentals are filtered by 'Publish Date' where available to avoid the
worst look-ahead, but market cap derived from 'Shares Outstanding' reflects
SimFin's data for that date row, not a contemporaneous filing table.
"""

import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATA_DIR = os.path.join("data", "simfin")


class SimFinStore:
    """Bulk SimFin data with point-in-time-style lookups."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        data_dir: str = DEFAULT_DATA_DIR,
        market: str = "us",
    ):
        self.api_key = api_key or os.getenv("SIMFIN_API_KEY")
        if not self.api_key:
            raise ValueError("SIMFIN_API_KEY not set in environment/.env")
        self.data_dir = data_dir
        self.market = market
        os.makedirs(data_dir, exist_ok=True)

        self._shareprices = None
        self._companies = None
        self._income = None

    def _load(self):
        import simfin as sf

        sf.set_api_key(self.api_key)
        sf.set_data_dir(self.data_dir)
        return sf

    def load_shareprices(self) -> pd.DataFrame:
        """Daily share prices incl. Shares Outstanding. MultiIndex (Ticker, Date)."""
        if self._shareprices is None:
            sf = self._load()
            self._shareprices = sf.load_shareprices(
                variant="daily", market=self.market
            )
        return self._shareprices

    def load_companies(self) -> pd.DataFrame:
        if self._companies is None:
            sf = self._load()
            self._companies = sf.load_companies(market=self.market)
        return self._companies

    def load_income(self, variant="quarterly") -> pd.DataFrame:
        if self._income is None:
            sf = self._load()
            self._income = sf.load_income(variant=variant, market=self.market)
        return self._income

    def market_cap_series(self, ticker: str) -> pd.Series:
        """
        Daily market cap series for a ticker: Close * Shares Outstanding.
        Returns a Series indexed by tz-naive date. Empty if ticker unknown.
        """
        sp = self.load_shareprices()
        try:
            df = sp.xs(ticker, level="Ticker")
        except KeyError:
            return pd.Series(dtype=float, name="market_cap")

        if "Shares Outstanding" not in df.columns or "Close" not in df.columns:
            return pd.Series(dtype=float, name="market_cap")

        cap = df["Close"] * df["Shares Outstanding"]
        cap.index = pd.to_datetime(cap.index).normalize()
        return cap.rename("market_cap")

    def fundamentals_as_of(self, ticker: str, as_of, variant="quarterly") -> pd.DataFrame:
        """
        Income statement rows for a ticker whose 'Publish Date' <= as_of.
        Uses publish date (not report date) to avoid look-ahead.
        """
        income = self.load_income(variant=variant)
        try:
            df = income.xs(ticker, level="Ticker")
        except KeyError:
            return pd.DataFrame()

        publish_col = "Publish Date" if "Publish Date" in df.columns else None
        if publish_col is None:
            return df

        as_of_ts = pd.to_datetime(as_of).normalize()
        publish = pd.to_datetime(df[publish_col])
        return df[publish <= as_of_ts]
