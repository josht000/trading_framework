"""
Social sentiment client.

Primary provider: Financial Modeling Prep's historical social sentiment
endpoint (legacy v4):
    https://financialmodelingprep.com/api/v4/historical/social-sentiment?symbol=SYM&page=N

Caveats:
- This is a *legacy* FMP endpoint; availability depends on plan tier and it
  may be removed. The client fails soft (returns empty DataFrame) so a
  missing/locked endpoint never breaks a backtest.
- Fields observed: date, symbol, stocktwitsPosts, twitterPosts,
  stocktwitsComments, twitterComments, stocktwitsLikes, twitterLikes,
  stocktwitsImpressions, twitterImpressions, stocktwitsSentiment,
  twitterSentiment.

Alternatives if FMP legacy is unavailable on your plan:
- Apewisdom (free, no key): https://apewisdom.io/api/v1.0/filter/{filter}
  aggregated Reddit/Stocktwits/Twitter mentions — current + limited history.
- Stocktwits public stream: https://api.stocktwits.com/api/2/streams/symbol/{sym}.json
  (no key, but only the last ~30 messages — useless for historical scans).
- Reddit Pushshift-style archives or stocktwits bulk dumps — gray area,
  verify terms before using in production.
"""

import os
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

FMP_BASE = "https://financialmodelingprep.com/api/v4/historical/social-sentiment"
APEWISDOM_BASE = "https://apewisdom.io/api/v1.0/filter"


class FMPSentimentClient:
    """Historical social sentiment from FMP's legacy v4 endpoint."""

    def __init__(self, api_key: Optional[str] = None, delay: float = 0.25):
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        self.delay = delay
        self.session = requests.Session()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def fetch_historical(
        self,
        symbol: str,
        start=None,
        end=None,
        max_pages: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch paginated historical sentiment for one symbol.

        Returns:
            DataFrame indexed by tz-naive date with sentiment/post columns.
            Empty DataFrame if no key, endpoint locked, or no data.
        """
        if not self.api_key:
            return pd.DataFrame()

        rows = []
        for page in range(max_pages):
            try:
                resp = self.session.get(
                    FMP_BASE,
                    params={
                        "symbol": symbol.upper(),
                        "page": page,
                        "apikey": self.api_key,
                    },
                    timeout=30,
                )
                if resp.status_code in (401, 402, 403, 404, 429):
                    break
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                break

            if not isinstance(data, list) or not data:
                break
            rows.extend(data)

            if start is not None and data:
                oldest = pd.to_datetime(data[-1].get("date"), errors="coerce")
                if pd.notna(oldest) and oldest < pd.to_datetime(start):
                    break

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"]).set_index("date").sort_index()

        if start is not None:
            df = df[df.index >= pd.to_datetime(start)]
        if end is not None:
            df = df[df.index <= pd.to_datetime(end)]

        return df

    def daily_activity(self, symbol: str, start=None, end=None) -> pd.Series:
        """
        Total social activity per day (posts + comments + likes across sources).
        Useful as a 'user engagement increase' proxy.
        """
        df = self.fetch_historical(symbol, start=start, end=end)
        if df.empty:
            return pd.Series(dtype=float, name="social_activity")

        count_cols = [
            c for c in df.columns
            if c.endswith(("Posts", "Comments", "Likes"))
        ]
        if not count_cols:
            return pd.Series(dtype=float, name="social_activity")

        return df[count_cols].sum(axis=1).rename("social_activity")
