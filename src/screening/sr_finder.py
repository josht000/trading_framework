"""
Ports of the implemented methods in albertsl/support-resistance_trading-bot's
support_resistance_finder.py (May 2018), adapted to the resistance_dataset eval
shape: each detector's detect_bars(df) returns sorted horizontal resistance
levels strictly above the last close.

Upstream is a sketch: `_elementary_method` is unfinished ("Attention, not
finished yet!"), and `_pivot_points` / `_weight` / `_so` / `_trendline` are
`pass`. What is actually implemented there:

- MeanShift over the 1-D price series (estimate_bandwidth(quantile=q,
  n_samples=100), bin_seeding), emitting the min and max member of each
  cluster as band edges.
- The "volume" method: bars where |gradient(price)| <= threshold and
  volume >= np.percentile(volume, 50) contribute their price as a level.
- np.gradient as the derivative, and an 8th-order Butterworth low-pass
  (signal.butter(8, 0.5), filtfilt padlen=30) as the preprocessor
  (their Signal_Processor._filter).

Ports:

- deriv_extrema  completes the elementary idea: local maxima are where the
                 gradient crosses +->- (mode="cross") or flat runs where
                 |gradient| <= threshold (mode="flat"). Emits the raw or
                 smoothed price at each extremum bar, merged within merge_tol
                 dollars keeping the highest.
- volume_levels  _volume_method plus a `relative` option (gradient of log
                 price, so the absolute threshold is comparable across $2 and
                 $200 names) and merge_tol to collapse the many qualifying
                 bars into distinct levels.
- meanshift      faithful port; `emit` adds max/center/mean alternatives to
                 the upstream band edges, min_cluster_size drops tiny clusters.
- floor_pivots   fills in the `pass`ed method with standard floor-trader
                 pivot points over the last `lookback` bars: classic
                 (PP/R1-R3/S1-S3), camarilla (R1-R4/S1-S4) and fibonacci
                 variants; `which` selects the level set.

Trendline methods are not ported: they emit diagonal lines and this dataset
labels horizontal levels.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal
from sklearn.cluster import MeanShift, estimate_bandwidth

from src.screening.pivot_resistance import merge_top


def _butter(x, order=8, wn=0.5, padlen=30):
    """Upstream Signal_Processor._filter: 8th-order Butterworth low-pass, filtfilt."""
    x = np.asarray(x, dtype=float)
    if len(x) <= padlen:
        return x
    b, a = signal.butter(order, wn)
    return signal.filtfilt(b, a, x, padlen=padlen)


def _series(df, name, smooth):
    x = df[name].to_numpy(dtype=float)
    return _butter(x) if smooth else x


def _gradient(x, relative):
    return np.gradient(np.log(x)) if relative else np.gradient(x)


@dataclass(frozen=True)
class DerivExtremaConfig:
    series: str = "high"        # high | close
    smooth: bool = True         # Butterworth low-pass before the derivative
    mode: str = "cross"         # cross = sign changes; flat = |d| <= threshold runs
    threshold: float = 0.005    # |gradient| cutoff for flat mode
    relative: bool = False      # gradient of log price (scale-free threshold)
    emit: str = "raw"           # raw | smoothed price at the extremum bar
    merge_tol: float = 0.10     # merge levels within $tol, keeping the highest


@dataclass(frozen=True)
class VolumeLevelsConfig:
    series: str = "high"
    smooth: bool = True
    threshold: float = 0.01     # upstream default
    relative: bool = False
    percentile: float = 50.0    # upstream 'prcentile' = 50
    emit: str = "raw"
    merge_tol: float = 0.10


@dataclass(frozen=True)
class MeanShiftConfig:
    series: str = "high"
    smooth: bool = False        # upstream test clustered raw highs
    quantile: float = 0.1       # estimate_bandwidth quantile (upstream default)
    n_samples: int = 100
    emit: str = "edges"         # edges (faithful) | max | center | mean
    min_cluster_size: int = 1


@dataclass(frozen=True)
class FloorPivotsConfig:
    lookback: int = 20          # bars in the reference period (1 = classic daily)
    variant: str = "classic"    # classic | camarilla | fibonacci
    which: str = "r123"         # r12 | r123 | pp_r123 | all


def _extremum_indexes(xs, mode, threshold, relative):
    """Indexes of local maxima of `xs`: gradient sign crossings or flat runs."""
    d = _gradient(xs, relative)
    if mode == "flat":
        out, i = [], 0
        while i < len(d):
            if abs(d[i]) > threshold:
                i += 1
                continue
            j = i
            while j + 1 < len(d) and abs(d[j + 1]) <= threshold:
                j += 1
            out.append(i + int(np.argmax(xs[i:j + 1])))
            i = j + 1
        return out
    return [i - 1 if xs[i - 1] >= xs[i] else i
            for i in range(1, len(d)) if d[i - 1] > 0 >= d[i]]


class _FinderDetector:
    cfg_cls = None

    def __init__(self, config: Optional[object] = None):
        self.config = config or self.cfg_cls()

    @staticmethod
    def _above(df, levels):
        last_close = float(df["close"].iloc[-1])
        return sorted(v for v in dict.fromkeys(levels) if v > last_close)


class DerivExtremaDetector(_FinderDetector):
    cfg_cls = DerivExtremaConfig

    def detect_bars(self, df):
        cfg = self.config
        raw = df[cfg.series].to_numpy(dtype=float)
        xs = _butter(raw) if cfg.smooth else raw
        idx = _extremum_indexes(xs, cfg.mode, cfg.threshold, cfg.relative)
        src = xs if cfg.emit == "smooth" else raw
        return self._above(df, merge_top([src[i] for i in idx], cfg.merge_tol))


class VolumeLevelsDetector(_FinderDetector):
    cfg_cls = VolumeLevelsConfig

    def detect_bars(self, df):
        cfg = self.config
        raw = df[cfg.series].to_numpy(dtype=float)
        xs = _butter(raw) if cfg.smooth else raw
        d = _gradient(xs, cfg.relative)
        vol = df["volume"].to_numpy(dtype=float)
        qvol = np.percentile(vol, cfg.percentile)
        idx = [i for i in range(len(raw)) if abs(d[i]) <= cfg.threshold and vol[i] >= qvol]
        src = xs if cfg.emit == "smooth" else raw
        return self._above(df, merge_top([src[i] for i in idx], cfg.merge_tol))


class MeanShiftDetector(_FinderDetector):
    cfg_cls = MeanShiftConfig

    def detect_bars(self, df):
        cfg = self.config
        xs = _series(df, cfg.series, cfg.smooth)
        X = xs.reshape(-1, 1)
        bw = estimate_bandwidth(X, quantile=cfg.quantile,
                                n_samples=min(cfg.n_samples, len(xs)))
        if not np.isfinite(bw) or bw <= 0:
            return []
        ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(X)
        levels = []
        for k in np.unique(ms.labels_):
            vals = xs[ms.labels_ == k]
            if len(vals) < cfg.min_cluster_size:
                continue
            if cfg.emit == "edges":
                levels += [float(vals.min()), float(vals.max())]
            elif cfg.emit == "max":
                levels.append(float(vals.max()))
            elif cfg.emit == "center":
                levels.append(float(ms.cluster_centers_[k][0]))
            else:
                levels.append(float(vals.mean()))
        return self._above(df, levels)


def _pivot_level_set(variant, H, L, C):
    R = H - L
    pp = (H + L + C) / 3
    if variant == "classic":
        return {"PP": pp, "R1": 2 * pp - L, "R2": pp + R, "R3": H + 2 * (pp - L),
                "S1": 2 * pp - H, "S2": pp - R, "S3": L - 2 * (H - pp)}
    if variant == "camarilla":
        return {"R1": C + R * 1.1 / 12, "R2": C + R * 1.1 / 6, "R3": C + R * 1.1 / 4,
                "R4": C + R * 1.1 / 2,
                "S1": C - R * 1.1 / 12, "S2": C - R * 1.1 / 6, "S3": C - R * 1.1 / 4,
                "S4": C - R * 1.1 / 2}
    return {"PP": pp, "R1": pp + 0.382 * R, "R2": pp + 0.618 * R, "R3": pp + R,
            "S1": pp - 0.382 * R, "S2": pp - 0.618 * R, "S3": pp - R}


class FloorPivotsDetector(_FinderDetector):
    cfg_cls = FloorPivotsConfig

    def detect_bars(self, df):
        cfg = self.config
        lb = min(cfg.lookback, len(df))
        H = float(df["high"].iloc[-lb:].max())
        L = float(df["low"].iloc[-lb:].min())
        C = float(df["close"].iloc[-1])
        levels = _pivot_level_set(cfg.variant, H, L, C)
        names = {"r12": ("R1", "R2"), "r123": ("R1", "R2", "R3"),
                 "pp_r123": ("PP", "R1", "R2", "R3")}.get(cfg.which)
        out = [v for k, v in levels.items() if names is None or k in names]
        return sorted(v for v in out if v > C)


CONFIGS = {
    "deriv_extrema": DerivExtremaConfig,
    "volume_levels": VolumeLevelsConfig,
    "meanshift": MeanShiftConfig,
    "floor_pivots": FloorPivotsConfig,
}

DETECTORS = {
    "deriv_extrema": DerivExtremaDetector,
    "volume_levels": VolumeLevelsDetector,
    "meanshift": MeanShiftDetector,
    "floor_pivots": FloorPivotsDetector,
}
