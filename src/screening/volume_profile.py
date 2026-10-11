"""
Volume-profile (high-volume node) level detector on daily bars.

Idea: price tends to stall where it previously traded a lot of volume. Each daily
bar's volume is spread over its high-low range (triangular, peaking at the typical
price), summed into a volume-at-price histogram over the lookback window, smoothed,
and the local peaks are the high-volume nodes (HVNs). A node above price is resistance,
below is support, by the same role rule as the pivot detector.

Compared with the pivot detector it returns few levels by construction: only the top
`max_above` / `max_below` nodes by relative volume within `max_dist_atr` of price.

This is a daily-bar approximation of true volume-at-price (minute bars would be exact).
Causal: only bars <= known_through are used.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.screening.resistance import LEVEL_COLUMNS, PreparedSeries, ResistanceConfig, ResistanceDetector


@dataclass(frozen=True)
class VolumeProfileConfig:
    atr_period: int = 14
    lookback_bars: int = 252
    bin_atr: float = 0.10
    smooth_sigma_bins: float = 2.0
    recency_halflife_bars: int = 126
    min_prominence_frac: float = 0.05
    min_gap_atr: float = 0.75
    max_above: int = 5
    max_below: int = 3
    max_dist_atr: float = 15.0
    touch_gap_bars: int = 3
    touch_tol_atr: float = 0.15
    round_tolerance_atr: float = 0.25
    ma_periods: tuple = (20, 50, 200)


def _triangular_cdf(x, a, b, c):
    span = np.maximum(b - a, 1e-9)
    b = a + span
    c = np.clip(c, a + 1e-6 * span, b - 1e-6 * span)
    with np.errstate(divide="ignore", invalid="ignore"):
        left = (x - a) ** 2 / (span * (c - a))
        right = 1.0 - (b - x) ** 2 / (span * (b - c))
    return np.where(x <= a, 0.0, np.where(x <= c, left, np.where(x < b, right, 1.0)))


class VolumeProfileDetector:
    def __init__(self, config: Optional[VolumeProfileConfig] = None):
        self.config = config or VolumeProfileConfig()
        self._base = ResistanceDetector(ResistanceConfig(
            atr_period=self.config.atr_period, ma_periods=self.config.ma_periods,
            include_swing_lows=False, include_ranges=False))

    def prepare(self, daily: pd.DataFrame) -> PreparedSeries:
        return self._base.prepare(daily)

    known_through_before = staticmethod(ResistanceDetector.known_through_before)
    known_through_on = staticmethod(ResistanceDetector.known_through_on)

    def profile(self, p: PreparedSeries, known_through: int):
        """Smoothed volume-at-price: (bin_centers, volume, bin_width). None if unavailable."""
        cfg = self.config
        k = known_through
        if k < 1 or not np.isfinite(p.atr[k]) or p.atr[k] <= 0:
            return None
        lo = max(0, k - cfg.lookback_bars + 1)
        h, l, c, v = p.high[lo:k + 1], p.low[lo:k + 1], p.close[lo:k + 1], p.volume[lo:k + 1]
        price = float(p.close[k])
        width = max(cfg.bin_atr * float(p.atr[k]), price * 0.0005)
        pmin, pmax = float(l.min()), float(h.max())
        n_bins = int(min(np.ceil((pmax - pmin) / width) + 1, 3000))
        edges = pmin + width * np.arange(n_bins + 1)
        centers = edges[:-1] + width / 2

        age = (k - np.arange(lo, k + 1)).astype(float)
        weights = v * 0.5 ** (age / cfg.recency_halflife_bars)
        tp = (h + l + c) / 3
        cdf = _triangular_cdf(edges[None, :], l[:, None], h[:, None], tp[:, None])
        profile = (np.diff(cdf, axis=1) * weights[:, None]).sum(axis=0)
        return centers, gaussian_filter1d(profile, cfg.smooth_sigma_bins, mode="constant"), width

    def levels(self, p: PreparedSeries, known_through: int, price: Optional[float] = None) -> pd.DataFrame:
        cfg = self.config
        k = known_through
        prof = self.profile(p, k)
        if prof is None:
            return pd.DataFrame(columns=LEVEL_COLUMNS)
        centers, smooth, width = prof
        atr = float(p.atr[k])
        price = float(p.close[k] if price is None else price)
        if smooth.max() <= 0:
            return pd.DataFrame(columns=LEVEL_COLUMNS)

        peaks, _ = find_peaks(smooth, prominence=cfg.min_prominence_frac * smooth.max(),
                              distance=max(1, int(round(cfg.min_gap_atr * atr / width))))
        cand = []
        for i in peaks:
            level = float(centers[i])
            dist = abs(level - price) / atr
            if dist <= cfg.max_dist_atr:
                cand.append((level, 10.0 * float(smooth[i] / smooth.max())))

        above = sorted((c for c in cand if c[0] > price), key=lambda t: -t[1])[:cfg.max_above]
        below = sorted((c for c in cand if c[0] <= price), key=lambda t: -t[1])[:cfg.max_below]
        lo = max(0, k - cfg.lookback_bars + 1)
        tol = cfg.touch_tol_atr * atr
        round_tol = cfg.round_tolerance_atr * atr

        rows = []
        for level, score in above + below:
            hit = np.flatnonzero((p.high[lo:k + 1] >= level - tol) & (p.low[lo:k + 1] <= level + tol))
            touches = 0 if len(hit) == 0 else 1 + int(np.sum(np.diff(hit) > cfg.touch_gap_bars))
            first_pos = lo + int(hit[0]) if len(hit) else lo
            last_pos = lo + int(hit[-1]) if len(hit) else lo
            role, dist = ResistanceDetector._role(level, level, price, atr)
            is_round = ResistanceDetector._contains_round(level - round_tol, level + round_tol, level)
            ma_conf = any(np.isfinite(s[k]) and abs(s[k] - level) <= round_tol for s in p.sma.values())
            rows.append(ResistanceDetector._row(
                p, level, level, role, "hvn", dist, price, 0, touches, np.nan, is_round, ma_conf,
                first_pos, last_pos, k, score, atr))
        if not rows:
            return pd.DataFrame(columns=LEVEL_COLUMNS)
        return pd.DataFrame(rows, columns=LEVEL_COLUMNS).sort_values("center").reset_index(drop=True)

    def ladder(self, p: PreparedSeries, known_through: int, price: Optional[float] = None) -> pd.DataFrame:
        lv = self.levels(p, known_through, price)
        lv = lv[lv["role"].isin(["resistance", "in_zone"])]
        return lv.sort_values("lower").reset_index(drop=True)
