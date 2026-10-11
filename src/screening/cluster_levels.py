"""
Cluster-based support/resistance levels, a faithful port of day0market/support_resistance
(github.com/day0market/support_resistance, package `pricelevels`) for benchmarking on
data/resistance_dataset.

Two detectors, both ending in AgglomerativeClustering over candidate prices:

- ZigZagClusterLevels: candidate prices are zigzag pivots from jbn/ZigZag's
  `peak_valley_pivots` (peak_percent_delta is a relative min-move, e.g. 3 means +-3%).
  `peaks` picks High pivots, Low pivots, or both; `min_bars_between_peaks` drops pivots
  closer than N bars (greedy, first one wins).
- RawPriceClusterLevels: candidate prices are bars that equal the rolling max (or min)
  over an odd `bars_for_peak` window, shifted so the window is centered on the bar.

Merging: pivots within `merge_distance` dollars (or `merge_percent`% of the series mean)
are clustered; the level is the cluster's mean or median price (`level_selector`).

Upstream is Cython + pinned to ancient deps (sklearn 0.21, pandas 0.25), so this file is a
pure-numpy port of `peak_valley_pivots` (limit_to_finalized_segments=True,
eager switching off) and of pricelevels/cluster.py. `fit` is generalized to take any
1-D series so the detector can run on highs as well as closes.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

PEAK, VALLEY = 1, -1


def _identify_initial_pivot(X, up_thresh, down_thresh):
    x_0 = X[0]
    max_x = min_x = x_0
    max_t = min_t = 0
    up_thresh += 1
    down_thresh += 1
    for t in range(1, len(X)):
        x_t = X[t]
        if x_t / min_x >= up_thresh:
            return VALLEY if min_t == 0 else PEAK
        if x_t / max_x <= down_thresh:
            return PEAK if max_t == 0 else VALLEY
        if x_t > max_x:
            max_x, max_t = x_t, t
        if x_t < min_x:
            min_x, min_t = x_t, t
    return VALLEY if x_0 < X[-1] else PEAK


def peak_valley_pivots(X, up_thresh, down_thresh):
    """Pure-numpy port of ZigZag's peak_valley_pivots_detailed(X, up, down, True, False).
    Returns an int array: 1 = peak, -1 = valley, 0 = no pivot. First and last
    elements are always annotated (finalized-segments mode)."""
    X = np.asarray(X, dtype=np.float64)
    if down_thresh > 0:
        raise ValueError("The down_thresh must be negative.")

    initial_pivot = _identify_initial_pivot(X, up_thresh, down_thresh)
    t_n = len(X)
    pivots = np.zeros(t_n, dtype=np.intp)
    trend = -initial_pivot
    last_pivot_t = 0
    last_pivot_x = X[0]
    pivots[0] = initial_pivot

    up_thresh += 1
    down_thresh += 1
    for t in range(1, t_n):
        x = X[t]
        r = x / last_pivot_x
        if trend == -1:
            if r >= up_thresh:
                pivots[last_pivot_t] = trend
                trend = PEAK
                last_pivot_x, last_pivot_t = x, t
            elif x < last_pivot_x:
                last_pivot_x, last_pivot_t = x, t
        else:
            if r <= down_thresh:
                pivots[last_pivot_t] = trend
                trend = VALLEY
                last_pivot_x, last_pivot_t = x, t
            elif x > last_pivot_x:
                last_pivot_x, last_pivot_t = x, t

    if last_pivot_t == t_n - 1:
        pivots[last_pivot_t] = trend
    elif pivots[t_n - 1] == 0:
        pivots[t_n - 1] = -trend
    return pivots


def cluster_prices_to_levels(prices, distance, level_selector="median"):
    """AgglomerativeClustering merge; returns [{'price': level, 'peak_count': n}, ...]."""
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) == 0:
        return []
    clustering = AgglomerativeClustering(distance_threshold=distance, n_clusters=None)
    try:
        clustering.fit(prices.reshape(-1, 1))
    except ValueError:
        return []

    df = pd.DataFrame({"price": prices, "cluster": clustering.labels_, "peak_count": 1})
    grouped = df.groupby("cluster").agg({"price": level_selector, "peak_count": "sum"}).reset_index()
    return grouped.to_dict("records")


def _pivot_indexes(pivots, peaks, min_bars_between_peaks):
    if peaks == "All":
        indexes = np.where(np.abs(pivots) == 1)[0]
    elif peaks == "High":
        indexes = np.where(pivots == 1)[0]
    elif peaks == "Low":
        indexes = np.where(pivots == -1)[0]
    else:
        raise ValueError("peaks must be one of: All, High, Low")
    if min_bars_between_peaks == 0 or len(indexes) == 0:
        return indexes
    selected = [indexes[0]]
    pre = indexes[0]
    for i in indexes[1:]:
        if i - pre >= min_bars_between_peaks:
            selected.append(i)
            pre = i
    return np.asarray(selected)


def _merge_distance(X, merge_distance, merge_percent):
    if merge_distance:
        return merge_distance
    return merge_percent * float(np.mean(X)) / 100.0


@dataclass(frozen=True)
class ZigZagLevelsConfig:
    peak_percent_delta: float = 3.0   # zigzag min move, percent
    merge_distance: Optional[float] = None   # dollars
    merge_percent: Optional[float] = 1.0     # percent of series mean (used if merge_distance is None)
    min_bars_between_peaks: int = 0
    peaks: str = "High"                # All | High | Low
    level_selector: str = "median"     # mean | median


@dataclass(frozen=True)
class RawPriceLevelsConfig:
    merge_distance: Optional[float] = None
    merge_percent: Optional[float] = 1.0
    level_selector: str = "median"
    use_maximums: bool = True          # True: rolling-max bars; False: rolling-min bars
    bars_for_peak: int = 21            # must be odd


def zigzag_cluster_levels(series: Sequence[float], cfg: ZigZagLevelsConfig) -> list:
    """Sorted level prices (dicts with price/peak_count) from zigzag pivots of `series`."""
    X = np.asarray(series, dtype=np.float64)
    d = cfg.peak_percent_delta / 100.0
    pivots = peak_valley_pivots(X, d, -d)
    idx = _pivot_indexes(pivots, cfg.peaks, cfg.min_bars_between_peaks)
    dist = _merge_distance(X, cfg.merge_distance, cfg.merge_percent)
    return cluster_prices_to_levels(X[idx], dist, cfg.level_selector)


def raw_price_cluster_levels(series: Sequence[float], cfg: RawPriceLevelsConfig) -> list:
    X = np.asarray(series, dtype=np.float64)
    if cfg.bars_for_peak % 2 == 0:
        raise ValueError("bars_for_peak must be odd")
    s = pd.Series(X)
    shift = (cfg.bars_for_peak - 1) // 2
    roll = s.rolling(window=cfg.bars_for_peak)
    F = (roll.max() if cfg.use_maximums else roll.min()).shift(-shift)
    prices = pd.unique(s[F == s])
    dist = _merge_distance(X, cfg.merge_distance, cfg.merge_percent)
    return cluster_prices_to_levels(prices, dist, cfg.level_selector)


class ClusterLevelsDetector:
    """Dataset-eval adapter: run on highs or closes, keep levels above the last close."""

    def __init__(self, kind: str = "zigzag", series: str = "high", config=None):
        self.kind = kind
        self.series = series
        self.config = config or (ZigZagLevelsConfig() if kind == "zigzag" else RawPriceLevelsConfig())

    def detect_bars(self, bars) -> list:
        """`bars` is a DataFrame or mapping with at least 'close' and the chosen series column."""
        close = bars["close"]
        last_close = float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])
        x = bars[self.series]
        x = x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x, dtype=np.float64)
        fn = zigzag_cluster_levels if self.kind == "zigzag" else raw_price_cluster_levels
        levels = [lv["price"] for lv in fn(x, self.config)]
        return sorted(v for v in levels if v > last_close)
