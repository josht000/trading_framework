"""
ATR-scaled pivot zone detector for support/resistance levels on daily bars.

Design notes
- Levels are zones (bands), not lines: a swing high contributes the band from
  its candle body top to its wick high; a swing low contributes wick low to
  body bottom.
- Causal by construction: a pivot at bar i is defined using bars up to
  i + pivot_right only, and is usable from bar i + pivot_right onward. Nothing
  after the as-of bar influences the returned levels.
- Role is relative to the reference price: zones entirely above price are
  resistance, zones entirely below are support, zones containing price are
  in_zone. A level that price trades through flips role automatically.
- Zones are ranked by an additive strength score whose weights live in
  ResistanceConfig so they can be tuned against the evaluation harness.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

NY_TZ = "America/New_York"

LEVEL_COLUMNS = [
    "lower", "upper", "center", "role", "kind", "distance_atr", "distance_pct",
    "n_pivots", "touches", "volume_ratio", "is_round", "ma_confluence",
    "first_pos", "last_touch_pos", "first_date", "last_touch_date",
    "last_touch_age", "score", "atr",
]


@dataclass(frozen=True)
class ResistanceConfig:
    atr_period: int = 14
    pivot_left: int = 5
    pivot_right: int = 3
    min_prominence_atr: float = 1.0
    lookback_bars: int = 252
    merge_atr: float = 0.5
    min_zone_atr: float = 0.3
    max_zone_atr: float = 1.5
    touch_gap_bars: int = 3
    round_tolerance_atr: float = 0.25
    include_swing_lows: bool = True
    include_ranges: bool = True
    range_min_bars: int = 5
    range_max_bars: int = 10
    range_max_atr: float = 2.0
    range_drift_atr: float = 0.75
    collapse_to_high: bool = True
    include_moving_averages: bool = False
    ma_periods: tuple = (20, 50, 200)
    ma_half_width_atr: float = 0.15
    recency_halflife_bars: int = 120
    w_pivots: float = 1.0
    w_touches: float = 0.5
    w_volume: float = 0.5
    w_round: float = 1.0
    w_recency: float = 1.0
    w_ma: float = 0.5


@dataclass
class PreparedSeries:
    dates: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    atr: np.ndarray
    vol_avg: np.ndarray
    sma: dict
    pivots: pd.DataFrame


def _wilder_atr(high, low, close, period):
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.fmax.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    tr[0] = high[0] - low[0]
    return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().to_numpy()


class ResistanceDetector:
    def __init__(self, config: Optional[ResistanceConfig] = None):
        self.config = config or ResistanceConfig()

    def prepare(self, daily: pd.DataFrame) -> PreparedSeries:
        cfg = self.config
        df = daily.copy()
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert(NY_TZ).tz_localize(None)
        df.index = idx.normalize()
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        o = df["open"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        v = df["volume"].to_numpy(dtype=float)

        atr = _wilder_atr(h, l, c, cfg.atr_period)
        vol_avg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        sma = {p: pd.Series(c).rolling(p).mean().to_numpy() for p in cfg.ma_periods}
        pivots = self._detect_pivots(o, h, l, c, v, atr, vol_avg)

        return PreparedSeries(
            dates=df.index.to_numpy(dtype="datetime64[D]"),
            open=o, high=h, low=l, close=c, volume=v,
            atr=atr, vol_avg=vol_avg, sma=sma, pivots=pivots,
        )

    def _detect_pivots(self, o, h, l, c, v, atr, vol_avg):
        cfg = self.config
        L, R = cfg.pivot_left, cfg.pivot_right
        rows = []
        for i in range(L, len(h) - R):
            a = atr[i]
            if not np.isfinite(a) or a <= 0:
                continue
            vr = self._vol_ratio(v, vol_avg, i)
            left_h, right_h = h[i - L:i], h[i + 1:i + R + 1]
            left_l, right_l = l[i - L:i], l[i + 1:i + R + 1]
            if h[i] > left_h.max() and h[i] >= right_h.max():
                prom = min(h[i] - left_l.min(), h[i] - right_l.min())
                if prom >= cfg.min_prominence_atr * a:
                    rows.append((i, i + R, "high", max(o[i], c[i]), h[i], vr, "swing", h[i]))
            if cfg.include_swing_lows and l[i] < left_l.min() and l[i] <= right_l.min():
                prom = min(left_h.max() - l[i], right_h.max() - l[i])
                if prom >= cfg.min_prominence_atr * a:
                    rows.append((i, i + R, "low", l[i], min(o[i], c[i]), vr, "swing", min(o[i], c[i])))
        if cfg.include_ranges:
            rows.extend(self._detect_ranges(o, h, l, c, v, atr, vol_avg))
        return pd.DataFrame(
            rows, columns=["pos", "confirmed_pos", "kind", "lower", "upper", "volume_ratio", "src", "level"]
        )

    @staticmethod
    def _vol_ratio(v, vol_avg, i):
        return v[i] / vol_avg[i] if np.isfinite(vol_avg[i]) and vol_avg[i] > 0 else np.nan

    def _detect_ranges(self, o, h, l, c, v, atr, vol_avg):
        """
        Consolidation ranges: a window of range_min_bars..range_max_bars bars whose
        total high-low height is at most range_max_atr ATR and whose net close drift is at
        most range_drift_atr ATR. Both the range high and the range low become levels.
        A range is known at the window's last bar, so nothing later affects it.
        """
        cfg = self.config
        seen, rows = set(), []
        for e in range(cfg.range_min_bars - 1, len(h)):
            a = atr[e]
            if not np.isfinite(a) or a <= 0:
                continue
            for n in range(cfg.range_max_bars, cfg.range_min_bars - 1, -1):
                s = e - n + 1
                if s < 0:
                    continue
                wh, wl = h[s:e + 1], l[s:e + 1]
                if wh.max() - wl.min() <= cfg.range_max_atr * a and abs(c[e] - c[s]) <= cfg.range_drift_atr * a:
                    for pos, kind in ((s + int(wh.argmax()), "high"), (s + int(wl.argmin()), "low")):
                        if (pos, kind) in seen:
                            continue
                        seen.add((pos, kind))
                        vr = self._vol_ratio(v, vol_avg, pos)
                        if kind == "high":
                            rows.append((pos, e, kind, max(o[pos], c[pos]), h[pos], vr, "range", h[pos]))
                        else:
                            rows.append((pos, e, kind, l[pos], min(o[pos], c[pos]), vr, "range", l[pos]))
                    break
        return rows

    @staticmethod
    def known_through_before(p: PreparedSeries, session_date) -> int:
        return int(np.searchsorted(p.dates, np.datetime64(pd.Timestamp(session_date).date(), "D"), side="left")) - 1

    @staticmethod
    def known_through_on(p: PreparedSeries, date) -> int:
        return int(np.searchsorted(p.dates, np.datetime64(pd.Timestamp(date).date(), "D"), side="right")) - 1

    def levels(self, p: PreparedSeries, known_through: int, price: Optional[float] = None) -> pd.DataFrame:
        cfg = self.config
        k = known_through
        if k < 0 or k >= len(p.close) or not np.isfinite(p.atr[k]) or p.atr[k] <= 0:
            return pd.DataFrame(columns=LEVEL_COLUMNS)
        atr = float(p.atr[k])
        price = float(p.close[k] if price is None else price)

        piv = p.pivots
        if len(piv):
            piv = piv[(piv["confirmed_pos"] <= k) & (piv["pos"] >= k - cfg.lookback_bars)]
        zones = self._cluster(piv, atr) if len(piv) else []

        rows = [self._zone_row(p, z, k, atr, price) for z in zones]
        if cfg.include_moving_averages:
            rows.extend(self._ma_rows(p, k, atr, price))
        if not rows:
            return pd.DataFrame(columns=LEVEL_COLUMNS)

        out = pd.DataFrame(rows, columns=LEVEL_COLUMNS)
        return out.sort_values("center").reset_index(drop=True)

    def ladder(self, p: PreparedSeries, known_through: int, price: Optional[float] = None) -> pd.DataFrame:
        lv = self.levels(p, known_through, price)
        lv = lv[lv["role"].isin(["resistance", "in_zone"])]
        return lv.sort_values("lower").reset_index(drop=True)

    def _cluster(self, piv: pd.DataFrame, atr: float):
        cfg = self.config
        members = piv.assign(center=(piv["lower"] + piv["upper"]) / 2).sort_values("center")
        zones, cur = [], None
        for m in members.itertuples(index=False):
            if cur is None:
                cur = {"lower": m.lower, "upper": m.upper, "members": [m]}
                continue
            new_lower, new_upper = min(cur["lower"], m.lower), max(cur["upper"], m.upper)
            gap = m.lower - cur["upper"]
            if gap <= cfg.merge_atr * atr and (new_upper - new_lower) <= cfg.max_zone_atr * atr:
                cur.update(lower=new_lower, upper=new_upper)
                cur["members"].append(m)
            else:
                zones.append(cur)
                cur = {"lower": m.lower, "upper": m.upper, "members": [m]}
        if cur is not None:
            zones.append(cur)
        return zones

    @staticmethod
    def _role(lower, upper, price, atr):
        if upper < price:
            return "support", (price - upper) / atr
        if lower > price:
            return "resistance", (lower - price) / atr
        return "in_zone", 0.0

    def _zone_row(self, p, z, k, atr, price):
        cfg = self.config
        lower, upper = z["lower"], z["upper"]
        level = max(m.level for m in z["members"])
        if upper - lower < cfg.min_zone_atr * atr:
            mid, half = (lower + upper) / 2, cfg.min_zone_atr * atr / 2
            lower, upper = mid - half, mid + half

        members = z["members"]
        first_pos = int(min(m.pos for m in members))
        srcs = {m.src for m in members}
        n_pivots = len({m.pos for m in members if m.src == "swing"}) + int("range" in srcs)
        kind = "+".join(sorted(srcs, reverse=True))
        vr = np.nanmean([m.volume_ratio for m in members]) if any(
            np.isfinite(m.volume_ratio) for m in members) else np.nan

        seg_h, seg_l = p.high[first_pos:k + 1], p.low[first_pos:k + 1]
        hit = np.flatnonzero((seg_h >= lower) & (seg_l <= upper))
        touches = 0 if len(hit) == 0 else 1 + int(np.sum(np.diff(hit) > cfg.touch_gap_bars))
        last_touch_pos = first_pos + int(hit[-1]) if len(hit) else first_pos

        if cfg.collapse_to_high:
            lower = upper = level

        tol = cfg.round_tolerance_atr * atr
        center = (lower + upper) / 2
        is_round = self._contains_round(lower - tol, upper + tol, center)
        ma_conf = any(
            np.isfinite(s[k]) and lower - tol <= s[k] <= upper + tol for s in p.sma.values()
        )

        recency = 0.5 ** ((k - last_touch_pos) / cfg.recency_halflife_bars)
        score = (
            cfg.w_pivots * min(n_pivots, 5)
            + cfg.w_touches * min(touches, 8)
            + cfg.w_volume * float(np.clip(vr if np.isfinite(vr) else 1.0, 0, 3))
            + cfg.w_round * float(is_round)
            + cfg.w_recency * recency
            + cfg.w_ma * float(ma_conf)
        )
        role, dist = self._role(lower, upper, price, atr)
        return self._row(p, lower, upper, role, kind, dist, price, n_pivots, touches,
                         vr, is_round, ma_conf, first_pos, last_touch_pos, k, score, atr)

    def _ma_rows(self, p, k, atr, price):
        cfg = self.config
        rows = []
        for period, series in p.sma.items():
            if not np.isfinite(series[k]):
                continue
            lower = series[k] - cfg.ma_half_width_atr * atr
            upper = series[k] + cfg.ma_half_width_atr * atr
            role, dist = self._role(lower, upper, price, atr)
            score = cfg.w_ma * (1 if period < 50 else 2)
            rows.append(self._row(p, lower, upper, role, f"sma{period}", dist, price, 0, 0,
                                  np.nan, False, False, k, k, k, score, atr))
        return rows

    @staticmethod
    def _contains_round(lo, hi, center):
        step = 1.0 if center < 20 else 5.0 if center < 100 else 10.0
        return np.ceil(lo / step) * step <= hi

    @staticmethod
    def _row(p, lower, upper, role, kind, dist, price, n_pivots, touches, vr, is_round,
             ma_conf, first_pos, last_touch_pos, k, score, atr):
        return {
            "lower": lower, "upper": upper, "center": (lower + upper) / 2, "role": role,
            "kind": kind, "distance_atr": dist,
            "distance_pct": abs((lower if role == "resistance" else upper) - price) / price
            if role != "in_zone" else 0.0,
            "n_pivots": n_pivots, "touches": touches, "volume_ratio": vr,
            "is_round": bool(is_round), "ma_confluence": bool(ma_conf),
            "first_pos": first_pos, "last_touch_pos": last_touch_pos,
            "first_date": pd.Timestamp(p.dates[first_pos]),
            "last_touch_date": pd.Timestamp(p.dates[last_touch_pos]),
            "last_touch_age": k - last_touch_pos, "score": score, "atr": atr,
        }
