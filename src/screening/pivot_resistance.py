"""
Pivot-high resistance ladder for daily bars (the v4 detector from the resistance_dataset experiments).

Rule: every bar whose high is the maximum of the +-`pivot_bars` window is a swing high. Swing highs
above the last close are sorted and merged: highs within `merge_tol` dollars of the previous level
collapse into one level, kept at the highest (exact pivot-bar high, not a zone midpoint).

Choices that were validated on the train split (and rejected alternatives that scored worse there):
  - exact pivot-bar highs, merged keeping the top      (keep-bottom / both-edges were worse)
  - pivot_bars=4                                       (3 overfit, 5 equal, 6+ lose recall)
  - no interior fill / round-number / swing-low / record-high additions (each added more false
    positives than true positives)

Known limit: the labeller also marks interior "staircase" highs and breakdown-candle prices that are
not pivots; a mechanical OHLCV rule plateaued at ~0.66 F1 (+-$0.10 hit) on train and test.

Windows are truncated at the right edge (`edge_pivots=True`), so recent highs in the last
`pivot_bars` bars can qualify as levels. Use `edge_pivots=False` for fully confirmed pivots only.

Pure Python: accepts plain sequences or anything with `.tolist()` (pandas Series, numpy arrays).
"""

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class PivotResistanceConfig:
    pivot_bars: int = 4
    merge_tol: float = 0.10
    edge_pivots: bool = True
    max_levels: Optional[int] = None


def _as_list(x) -> list:
    return [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


def swing_highs(highs: Sequence[float], pivot_bars: int = 4, edge_pivots: bool = True) -> list:
    """(index, high) of bars whose high >= every high within +-pivot_bars."""
    h = _as_list(highs)
    n = len(h)
    out = []
    for i in range(n):
        if not edge_pivots and i + pivot_bars >= n:
            break
        if h[i] >= max(h[max(0, i - pivot_bars):i + pivot_bars + 1]):
            out.append((i, h[i]))
    return out


def merge_top(values: Sequence[float], tol: float = 0.10) -> list:
    """Sort ascending and merge values within `tol` of the previous merged level, keeping the higher."""
    out = []
    for v in sorted(values):
        if out and v - out[-1] <= tol:
            out[-1] = v
        else:
            out.append(v)
    return out


class PivotResistanceDetector:
    def __init__(self, config: Optional[PivotResistanceConfig] = None):
        self.config = config or PivotResistanceConfig()

    def detect(self, highs: Sequence[float], last_close: float) -> list:
        """Resistance levels above `last_close`, ascending. Empty when price is above every pivot (blue sky)."""
        cfg = self.config
        pivots = [v for _, v in swing_highs(highs, cfg.pivot_bars, cfg.edge_pivots) if v > last_close]
        levels = merge_top(pivots, cfg.merge_tol)
        if cfg.max_levels is not None:
            levels = levels[:cfg.max_levels]
        return levels

    def detect_bars(self, bars) -> list:
        """`bars` is a DataFrame or mapping with 'high' and 'close' columns, oldest bar first."""
        close = _as_list(bars["close"])
        return self.detect(bars["high"], close[-1])


def detect_resistance(highs: Sequence[float], last_close: float, **overrides) -> list:
    return PivotResistanceDetector(PivotResistanceConfig(**overrides)).detect(highs, last_close)
