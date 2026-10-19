"""
Alpaca historical news client.

Fetches Benzinga news from Alpaca's news endpoint for a symbol list and
date range, and converts articles into per-symbol per-date counts used as
a point-in-time "news catalyst" input for screening.
"""

import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

load_dotenv()


class AlpacaNewsClient:
    """Fetch historical news from Alpaca's news endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        self.client = NewsClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
        )

    def fetch_news(
        self,
        symbols,
        start,
        end,
        limit: int = 50,
        max_pages: int = 200,
    ) -> pd.DataFrame:
        """
        Fetch news articles for symbols over [start, end] with pagination.

        Args:
            symbols: Single symbol string or list of symbols
            start: Start datetime/date
            end: End datetime/date
            limit: Page size (API max is 50)
            max_pages: Safety cap on pagination

        Returns:
            DataFrame with columns [id, headline, summary, source, symbols,
            created_at, updated_at] indexed by created_at.
        """
        if isinstance(symbols, str):
            symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        else:
            symbol_list = list(symbols)
        symbols_param = ",".join(symbol_list)

        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        frames = []
        page_token = None
        pages = 0

        while pages < max_pages:
            request = NewsRequest(
                symbols=symbols_param,
                start=start.to_pydatetime() if hasattr(start, "to_pydatetime") else start,
                end=end.to_pydatetime() if hasattr(end, "to_pydatetime") else end,
                limit=limit,
                sort="asc",
                include_content=False,
                exclude_contentless=True,
                page_token=page_token,
            )
            news_set = self.client.get_news(request)

            try:
                df = news_set.df
            except (KeyError, AttributeError):
                df = None
            if df is not None and not df.empty:
                frames.append(df)

            page_token = getattr(news_set, "next_page_token", None)
            pages += 1
            if not page_token:
                break

        if not frames:
            return pd.DataFrame(
                columns=["headline", "summary", "source", "symbols",
                         "created_at", "updated_at"]
            )

        df = pd.concat(frames)

        # alpaca-py returns a MultiIndex (symbol, timestamp); normalize
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index().rename(
                columns={"timestamp": "created_at", "symbol": "symbol_row"}
            )

        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

        return df

    def daily_counts(
        self,
        news_df: pd.DataFrame,
        symbols: List[str],
        ny_tz: str = "America/New_York",
    ) -> pd.DataFrame:
        """
        Convert a news DataFrame to per-symbol per-date article counts
        (dates localized to America/New_York).

        Returns:
            DataFrame indexed by normalized ET date with one column per symbol.
        """
        counts = pd.DataFrame(columns=symbols, dtype=float)

        if news_df.empty or "created_at" not in news_df.columns:
            return counts

        df = news_df.copy()
        df["date"] = (
            df["created_at"].dt.tz_convert(ny_tz).dt.normalize().dt.tz_localize(None)
        )

        symbol_col = None
        if "symbol_row" in df.columns:
            symbol_col = "symbol_row"
        elif "symbols" in df.columns:
            symbol_col = "symbols"

        if symbol_col is None:
            return counts

        exploded = df.explode(symbol_col)
        exploded = exploded[exploded[symbol_col].isin(symbols)]

        counts = (
            exploded.groupby(["date", symbol_col])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=symbols, fill_value=0)
        )
        counts.index = pd.to_datetime(counts.index)
        return counts
