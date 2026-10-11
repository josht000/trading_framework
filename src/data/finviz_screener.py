"""
Finviz screener integration for universe selection
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict, Optional
import time
import random


class FinvizScreener:
    """
    Fetch stock universe from Finviz screener
    """

    BASE_URL = "https://finviz.com/screener.ashx"

    # Pre-configured screener URL for Humbled Trader strategy
    HUMBLED_TRADER_URL = "https://finviz.com/screener?v=111&f=cap_midover%2Cexch_nasd%2Csh_avgvol_o500%2Csh_price_o2%2Csh_relvol_o2&c=1"

    def __init__(self, delay: float = 1.0):
        """
        Initialize Finviz screener

        Args:
            delay: Delay between requests in seconds (to be respectful)
        """
        self.delay = delay
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def build_url(
        self,
        exchange: str = "NASDAQ",
        min_price: float = 1.0,
        max_price: float = 20.0,
        min_volume: Optional[int] = None,
        min_rel_vol: Optional[float] = None,
        max_market_cap: Optional[float] = None,
        custom_filters: Optional[str] = None,
    ) -> str:
        """
        Build Finviz screener URL with filters

        Args:
            exchange: Exchange filter (NASDAQ, NYSE, AMEX)
            min_price: Minimum stock price
            max_price: Maximum stock price
            min_volume: Minimum average volume
            min_rel_vol: Minimum relative volume
            max_market_cap: Maximum market cap (in billions)
            custom_filters: Custom filter string if needed

        Returns:
            Finviz screener URL
        """
        filters = []

        # Exchange
        exchange_map = {"NASDAQ": "exch_nasd", "NYSE": "exch_nyse", "AMEX": "exch_amex"}
        if exchange in exchange_map:
            filters.append(exchange_map[exchange])

        # Price
        filters.append(f"sh_price_o{min_price}")
        filters.append(f"sh_price_u{max_price}")

        # Volume
        if min_volume:
            filters.append(f"sh_avgvol_o{min_volume}")

        # Relative volume
        if min_rel_vol:
            filters.append(f"sh_relvol_o{min_rel_vol}")

        # Market cap
        if max_market_cap:
            filters.append(f"sh_mktcap_u{max_market_cap}")

        filter_string = ",".join(filters)

        url = f"{self.BASE_URL}?v=111&f={filter_string}"
        return url

    def fetch_tickers_from_url(
        self,
        url: str,
        max_results: int = 1000,
    ) -> List[str]:
        """
        Fetch tickers from a specific Finviz screener URL

        Args:
            url: Full Finviz screener URL
            max_results: Maximum number of results to fetch

        Returns:
            List of ticker symbols
        """
        tickers = []
        page = 1

        while len(tickers) < max_results:
            # Add page parameter if not present
            if "&c=" in url:
                paginated_url = url.rsplit("&c=", 1)[0] + f"&c={page}"
            else:
                paginated_url = f"{url}&c={page}"

            try:
                # Add random delay to avoid blocking
                time.sleep(random.uniform(1, 3))

                response = self.session.get(paginated_url, timeout=30)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, "html.parser")

                # Find the table with screener results
                table = soup.find("table", {"class": "table-light"})

                if not table:
                    break

                rows = table.find_all("tr")[1:]  # Skip header row

                if not rows:
                    break

                for row in rows:
                    cells = row.find_all("td")
                    if cells:
                        ticker = cells[1].text.strip()
                        if ticker and ticker not in tickers:
                            tickers.append(ticker)

                page += 1

                # Check if we've exhausted results
                if len(rows) < 20:  # Finviz typically shows 20 per page
                    break

            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                break

        return tickers[:max_results]

    def fetch_tickers(
        self,
        exchange: str = "NASDAQ",
        min_price: float = 1.0,
        max_price: float = 20.0,
        min_volume: Optional[int] = None,
        min_rel_vol: Optional[float] = None,
        max_market_cap: Optional[float] = None,
        max_results: int = 1000,
    ) -> List[str]:
        """
        Fetch tickers from Finviz screener

        Args:
            exchange: Exchange filter
            min_price: Minimum stock price
            max_price: Maximum stock price
            min_volume: Minimum average volume
            min_rel_vol: Minimum relative volume
            max_market_cap: Maximum market cap (in billions)
            max_results: Maximum number of results to fetch

        Returns:
            List of ticker symbols
        """
        url = self.build_url(
            exchange=exchange,
            min_price=min_price,
            max_price=max_price,
            min_volume=min_volume,
            min_rel_vol=min_rel_vol,
            max_market_cap=max_market_cap,
        )

        return self.fetch_tickers_from_url(url, max_results)

    def fetch_stock_data(
        self,
        exchange: str = "NASDAQ",
        min_price: float = 1.0,
        max_price: float = 20.0,
        min_volume: Optional[int] = None,
        min_rel_vol: Optional[float] = None,
        max_market_cap: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Fetch detailed stock data from Finviz

        Args:
            exchange: Exchange filter
            min_price: Minimum stock price
            max_price: Maximum stock price
            min_volume: Minimum average volume
            min_rel_vol: Minimum relative volume
            max_market_cap: Maximum market cap (in billions)

        Returns:
            DataFrame with stock data
        """
        url = self.build_url(
            exchange=exchange,
            min_price=min_price,
            max_price=max_price,
            min_volume=min_volume,
            min_rel_vol=min_rel_vol,
            max_market_cap=max_market_cap,
        )

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            # Use pandas to read the HTML table
            tables = pd.read_html(response.content)
            df = tables[0]  # First table is the screener results

            # Clean up column names
            df.columns = [
                "No.",
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Country",
                "Market Cap",
                "P/E",
                "Price",
                "Change",
                "Volume",
                "Change",
            ]

            return df

        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return pd.DataFrame()


# Predefined screener configurations
SCREENER_CONFIGS = {
    "humbled_bullish_overnight": {
        "exchange": "NASDAQ",
        "min_price": 1.0,
        "max_price": 20.0,
        "min_volume": 500_000,
        "min_rel_vol": 1.0,
        "max_market_cap": 2.0,  # $2B
    }
}


def get_humbled_bullish_universe(max_results: int = 100) -> List[str]:
    """
    Get universe for Humbled Trader Bullish Overnight strategy

    Uses the pre-configured Finviz URL:
    https://finviz.com/screener?v=111&f=cap_midover%2Cexch_nasd%2Csh_avgvol_o500%2Csh_price_o2%2Csh_relvol_o2&c=1

    Filters:
    - cap_midover: Mid cap and over
    - exch_nasd: NASDAQ
    - sh_avgvol_o500: Average volume over 500k
    - sh_price_o2: Price over $2
    - sh_relvol_o2: Relative volume over 2

    Args:
        max_results: Maximum number of tickers to return

    Returns:
        List of ticker symbols
    """
    screener = FinvizScreener()

    # Use the pre-configured URL
    tickers = screener.fetch_tickers_from_url(
        url=FinvizScreener.HUMBLED_TRADER_URL,
        max_results=max_results,
    )

    return tickers
