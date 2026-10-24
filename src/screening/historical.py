from dataclasses import replace
from datetime import time

import numpy as np
import pandas as pd

from src.screening.resistance import ResistanceConfig, ResistanceDetector
from src.utils import timing

ATR_PIVOT_CONFIG = ResistanceConfig(
    pivot_left=3,
    pivot_right=3,
    min_prominence_atr=0.5,
    merge_atr=0.3,
    include_ranges=False,
)


class HistoricalPriceVolumeScreener:
    def __init__(
        self,
        min_price=2.0,
        max_price=20.0,
        min_average_volume=500_000,
        min_relative_volume=2.0,
        max_market_cap=2_000_000_000,
        resistance_lookback=20,
        volume_lookback=20,
        entry_time="15:55",
        timezone="America/New_York",
        news_lookback_days=3,
        min_news_articles=1,
        require_market_cap=False,
        require_news_catalyst=True,
        social_surge_factor=2.0,
        resistance_method="rolling",
        resistance_config=None,
        min_breakout_atr=0.0,
        lazy_resistance=False,
        lookback_bars=None,
    ):
        self.lazy_resistance = lazy_resistance
        self.ladder_cache = None
        self.resistance_method = resistance_method
        self.min_breakout_atr = min_breakout_atr
        self.detector = (
            ResistanceDetector(resistance_config or (replace(ATR_PIVOT_CONFIG, lookback_bars=lookback_bars) if lookback_bars else ATR_PIVOT_CONFIG))
            if resistance_method == "atr_pivot"
            else None
        )
        self.min_price = min_price
        self.max_price = max_price
        self.min_average_volume = min_average_volume
        self.min_relative_volume = min_relative_volume
        self.max_market_cap = max_market_cap
        self.resistance_lookback = resistance_lookback
        self.volume_lookback = volume_lookback
        self.entry_time = time.fromisoformat(entry_time)
        self.timezone = timezone
        self.news_lookback_days = news_lookback_days
        self.min_news_articles = min_news_articles
        self.require_market_cap = require_market_cap
        self.require_news_catalyst = require_news_catalyst
        self.social_surge_factor = social_surge_factor

    def _level_uppers(self, prepared, session_date):
        """ATR and the top of every atr_pivot zone known before session_date (cached per date)."""
        with timing.wstage("screen ladder/levels"):
            cache = self.ladder_cache
            if cache is not None and session_date in cache:
                return cache[session_date]
            k = self.detector.known_through_before(prepared, session_date)
            atr = float(prepared.atr[k]) if k >= 0 else np.nan
            lv = self.detector.levels(prepared, k)
            out = (atr, lv["upper"].to_numpy(dtype=float))
            if cache is not None:
                cache[session_date] = out
            return out

    def scan(self, daily_df, minute_df, context=None):
        daily = self._prepare(daily_df)
        minute = self._prepare(minute_df)
        minute = minute.between_time("09:30", "16:00", inclusive="left")
        context = self._prepare_context(context, minute)
        result = pd.DataFrame(index=minute.index)
        criteria = [
            "price",
            "average_volume",
            "relative_volume",
            "higher_close",
            "higher_open",
            "resistance_breakout",
            "above_vwap",
            "market_cap",
            "news_catalyst",
        ]
        for name in criteria:
            result[name] = False
        result["relative_volume_value"] = np.nan
        result["session_vwap"] = np.nan
        result["social_surge"] = False
        result["headroom"] = np.nan
        result["eligible"] = False

        daily_dates = pd.Index(daily.index.date)
        prepared = self.detector.prepare(daily) if self.detector else None
        for session_date, session in minute.groupby(minute.index.date, sort=True):
            candidates = session[session.index.time == self.entry_time]
            if candidates.empty:
                continue
            prior = daily.loc[daily_dates < session_date]
            required = max(self.resistance_lookback, self.volume_lookback)
            if len(prior) < required:
                continue
            candidate_index = candidates.index[0]
            available = session.loc[:candidate_index]
            candidate = available.iloc[-1]
            previous_close = float(prior["close"].iloc[-1])
            session_open = float(available["open"].iloc[0])
            cumulative_volume = float(available["volume"].sum())
            average_volume = float(prior["volume"].tail(self.volume_lookback).mean())
            resistance = float(prior["high"].tail(self.resistance_lookback).max())
            typical_price = (
                available["high"] + available["low"] + available["close"]
            ) / 3
            session_vwap = float(
                (typical_price * available["volume"]).sum() / cumulative_volume
            )
            relative_volume = (
                cumulative_volume / average_volume if average_volume > 0 else np.nan
            )
            headroom = np.nan
            if prepared is not None:
                with timing.wstage("screen cheap criteria"):
                    cheap_pass = (
                        self.min_price <= candidate["close"] <= self.max_price
                        and average_volume >= self.min_average_volume
                        and relative_volume >= self.min_relative_volume
                        and candidate["close"] > previous_close
                        and session_open > previous_close
                    )
                    if self.lazy_resistance and not cheap_pass:
                        resistance = float("inf")
                    else:
                        atr, uppers = self._level_uppers(prepared, session_date)
                        crossed = uppers[
                            (uppers > previous_close)
                            & (uppers + self.min_breakout_atr * atr < candidate["close"])
                        ]
                        resistance = float(crossed.max()) if len(crossed) else float("inf")
                        above = uppers[uppers > candidate["close"]]
                        headroom = float(above.min() / candidate["close"] - 1) if len(above) else 9.99
            date_key = pd.Timestamp(session_date)
            caps = context["market_cap"]
            if caps.empty:
                market_cap = np.nan
            else:
                available = caps.loc[:date_key]
                market_cap = available.iloc[-1] if len(available) > 0 else np.nan
            if pd.notna(market_cap):
                market_cap_ok = market_cap <= self.max_market_cap
            else:
                market_cap_ok = not self.require_market_cap

            news_count = float(
                context["news_count"]
                .loc[:date_key]
                .tail(self.news_lookback_days)
                .sum()
            )
            news_catalyst = news_count >= self.min_news_articles
            if context["news_count"].empty:
                news_catalyst = not self.require_news_catalyst

            social_surge = False
            social = context["social_activity"]
            if not social.empty and date_key in social.index:
                history = social.loc[:date_key].tail(30)
                if len(history) > 1:
                    baseline = history.iloc[:-1].mean()
                    social_surge = (
                        baseline > 0
                        and history.iloc[-1] >= baseline * self.social_surge_factor
                    )

            values = {
                "price": self.min_price <= candidate["close"] <= self.max_price,
                "average_volume": average_volume >= self.min_average_volume,
                "relative_volume": relative_volume >= self.min_relative_volume,
                "higher_close": candidate["close"] > previous_close,
                "higher_open": session_open > previous_close,
                "resistance_breakout": candidate["close"] > resistance,
                "above_vwap": candidate["close"] > session_vwap,
                "market_cap": market_cap_ok,
                "news_catalyst": news_catalyst,
            }
            for name, value in values.items():
                result.at[candidate_index, name] = bool(value)
            result.at[candidate_index, "relative_volume_value"] = relative_volume
            result.at[candidate_index, "session_vwap"] = session_vwap
            if prepared is not None:
                result.at[candidate_index, "headroom"] = headroom
            result.at[candidate_index, "social_surge"] = bool(social_surge)
            result.at[candidate_index, "eligible"] = all(values.values())
        return result

    def _prepare_context(self, context, minute):
        """Normalize optional context series to tz-naive date indices."""
        context = context or {}
        market_cap = context.get("market_cap")
        news_count = context.get("news_count")
        social_activity = context.get("social_activity")

        def normalize(series):
            if series is None or len(series) == 0:
                return pd.Series(dtype=float)
            s = series.copy()
            idx = pd.to_datetime(s.index)
            if idx.tz is not None:
                idx = idx.tz_convert(self.timezone).tz_localize(None)
            s.index = idx.normalize()
            return s.sort_index()

        return {
            "market_cap": normalize(market_cap),
            "news_count": normalize(news_count),
            "social_activity": normalize(social_activity),
        }

    def _prepare(self, df):
        result = df.copy().sort_index()
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.to_datetime(result.index)
        if result.index.tz is None:
            result.index = result.index.tz_localize(self.timezone)
        else:
            result.index = result.index.tz_convert(self.timezone)
        return result[~result.index.duplicated(keep="last")]
