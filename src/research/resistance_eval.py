"""
Outcome-based evaluation harness for resistance-level detectors.

There is no ground-truth labeled dataset for equity support/resistance, so each
level source is judged by what price does afterwards, against baselines:

  atr_pivot  the detector under test
  rolling20  highest high of the prior 20 bars (current rule of thumb)
  rolling10  highest high of the prior 10 bars (FinLLM-B definition)
  round      next whole-number price above (psychological level)
  random     random band above price (null)

Two tests share one causal pass over each symbol (levels at bar k use bars
<= k; outcomes use bars > k):

  Reaction  For the nearest resistance zone above price (within max_distance_atr),
            did price touch it within horizon_bars, and then reject or break it?
  Breakout  When bar k+1 closes above a zone, what are forward returns, the
            overnight gap, false-breakout rate, and does the room to the next
            zone above predict the follow-through?
"""

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.screening.resistance import PreparedSeries, ResistanceConfig, ResistanceDetector

SOURCES = ["atr_pivot", "shifted", "rolling20", "rolling10", "round", "random"]


@dataclass(frozen=True)
class EvalConfig:
    horizon_bars: int = 10
    react_bars: int = 5
    max_distance_atr: float = 3.0
    break_buffer_atr: float = 0.25
    reject_atr: float = 1.0
    volume_confirm: float = 1.5
    min_history: int = 120
    fwd_horizon: int = 5
    false_break_bars: int = 2
    max_room_atr: float = 10.0
    random_width_atr: float = 0.0
    seed: int = 7


def _zone(lower, upper, **extra):
    return {"lower": float(lower), "upper": float(upper), "score": np.nan,
            "n_pivots": 0, "touches": 0, "is_round": False, "kind": "", **extra}


def _detector_zones(detector, p, k, price, roles=None):
    lv = detector.levels(p, k, price)
    if roles is not None:
        lv = lv[lv["role"].isin(roles)]
    return [
        _zone(r.lower, r.upper, score=r.score, n_pivots=r.n_pivots,
              touches=r.touches, is_round=r.is_round, kind=r.kind, role=r.role)
        for r in lv.itertuples(index=False)
    ]


def detector_source(detector: ResistanceDetector) -> Callable:
    def source(p, k, price, atr, rng):
        return _detector_zones(detector, p, k, price, ("resistance", "in_zone"))
    return source


def shifted_source(detector: ResistanceDetector, min_shift_atr: float = 0.75,
                   max_shift_atr: float = 2.5) -> Callable:
    """Placebo: the detector's own zones (same widths and scores) moved by one common
    random offset, so only the location is destroyed."""
    def source(p, k, price, atr, rng):
        zones = _detector_zones(detector, p, k, price)
        offset = rng.choice([-1.0, 1.0]) * rng.uniform(min_shift_atr, max_shift_atr) * atr
        for z in zones:
            z["lower"] += offset
            z["upper"] += offset
            z["role"] = "resistance"
        return zones
    return source


def rolling_high_source(n: int) -> Callable:
    def source(p, k, price, atr, rng):
        if k + 1 < n:
            return []
        level = p.high[k - n + 1:k + 1].max()
        return [_zone(level, level, role="resistance")] if level > price else []
    return source


def round_source(p, k, price, atr, rng):
    step = 1.0 if price < 20 else 5.0 if price < 100 else 10.0
    level = (np.floor(price / step) + 1) * step
    return [_zone(level, level, is_round=True, role="resistance")]


def make_random_source(cfg: EvalConfig, max_distance_atr: float) -> Callable:
    def source(p, k, price, atr, rng):
        lower = price + rng.uniform(0.25, max_distance_atr) * atr
        return [_zone(lower, lower + cfg.random_width_atr * atr, role="resistance")]
    return source


def build_sources(detector: ResistanceDetector, cfg: EvalConfig, names=None) -> Dict[str, Callable]:
    sources = {
        "atr_pivot": detector_source(detector),
        "shifted": shifted_source(detector),
        "rolling20": rolling_high_source(20),
        "rolling10": rolling_high_source(10),
        "round": round_source,
        "random": make_random_source(cfg, cfg.max_distance_atr),
    }
    return {n: sources[n] for n in (names or SOURCES)}


def _reaction(p: PreparedSeries, k: int, zone: dict, atr: float, cfg: EvalConfig):
    lo, up = zone["lower"], zone["upper"]
    touch = None
    for j in range(k + 1, k + 1 + cfg.horizon_bars):
        if p.high[j] >= lo:
            touch = j
            break
    if touch is None:
        return "no_touch"
    buf = cfg.break_buffer_atr * atr
    for m in range(touch, touch + cfg.react_bars + 1):
        if p.close[m] > up + buf:
            return "break"
        if p.close[m] < lo and p.low[m] <= lo - cfg.reject_atr * atr:
            return "reject"
    return "neutral"


def _breakout(p: PreparedSeries, k: int, zones: List[dict], price: float, atr: float, cfg: EvalConfig):
    b = k + 1
    buf = cfg.break_buffer_atr * atr
    crossed = [z for z in zones if p.close[b] > z["upper"] + buf]
    if not crossed:
        return None
    broken = max(crossed, key=lambda z: z["upper"])
    above = [z["lower"] for z in zones if z["lower"] > broken["upper"]]
    blue_sky = not above
    room = cfg.max_room_atr if blue_sky else min((min(above) - broken["upper"]) / atr, cfg.max_room_atr)
    vol_ratio = p.volume[b] / p.vol_avg[b] if np.isfinite(p.vol_avg[b]) and p.vol_avg[b] > 0 else np.nan
    closes_after = p.close[b + 1:b + 1 + cfg.false_break_bars]
    return {
        "date": pd.Timestamp(p.dates[b]), "lower": broken["lower"], "upper": broken["upper"],
        "score": broken["score"], "n_pivots": broken["n_pivots"], "touches": broken["touches"],
        "is_round": broken["is_round"], "volume_ratio": vol_ratio,
        "vol_ok": bool(np.isfinite(vol_ratio) and vol_ratio >= cfg.volume_confirm),
        "room_atr": room, "blue_sky": blue_sky,
        **_forward(p, b, atr, cfg),
        "false_break": bool(len(closes_after) and closes_after.min() < broken["upper"]),
    }


def _forward(p: PreparedSeries, b: int, atr: float, cfg: EvalConfig) -> dict:
    fh = cfg.fwd_horizon
    return {
        "gap_ret": p.open[b + 1] / p.close[b] - 1,
        "ret1": p.close[b + 1] / p.close[b] - 1,
        "ret5": p.close[b + fh] / p.close[b] - 1,
        "mfe5_atr": (p.high[b + 1:b + 1 + fh].max() - p.close[b]) / atr,
    }


def evaluate_symbol(symbol: str, p: PreparedSeries, detector: ResistanceDetector,
                    cfg: EvalConfig, sources: Dict[str, Callable]):
    n = len(p.close)
    k_max = n - 2 - max(cfg.horizon_bars + cfg.react_bars, cfg.fwd_horizon + cfg.false_break_bars + 1)
    reactions, breakouts, uncond = [], [], []
    rngs = {name: np.random.default_rng(cfg.seed + i) for i, name in enumerate(sources)}

    for k in range(cfg.min_history, k_max + 1):
        atr = p.atr[k]
        if not np.isfinite(atr) or atr <= 0:
            continue
        price = float(p.close[k])
        b = k + 1
        vr_b = p.volume[b] / p.vol_avg[b] if np.isfinite(p.vol_avg[b]) and p.vol_avg[b] > 0 else np.nan
        uncond.append({"symbol": symbol, "date": pd.Timestamp(p.dates[b]),
                       "volume_ratio": vr_b, **_forward(p, b, atr, cfg)})

        for name, source in sources.items():
            zones = [z for z in source(p, k, price, atr, rngs[name]) if z["upper"] > price]
            above = sorted((z for z in zones if z["lower"] > price), key=lambda z: z["lower"])
            near = above[0] if above and (above[0]["lower"] - price) <= cfg.max_distance_atr * atr else None
            if near is not None:
                reactions.append({
                    "symbol": symbol, "source": name, "date": pd.Timestamp(p.dates[k]),
                    "price": price, "atr": atr, "lower": near["lower"], "upper": near["upper"],
                    "distance_atr": (near["lower"] - price) / atr,
                    "width_atr": (near["upper"] - near["lower"]) / atr,
                    "score": near["score"], "n_pivots": near["n_pivots"],
                    "touches": near["touches"], "is_round": near["is_round"],
                    "kind": near["kind"], "outcome": _reaction(p, k, near, atr, cfg),
                })
            ev = _breakout(p, k, zones, price, atr, cfg) if zones else None
            if ev is not None:
                breakouts.append({"symbol": symbol, "source": name, **ev})

    return pd.DataFrame(reactions), pd.DataFrame(breakouts), pd.DataFrame(uncond)


def run_evaluation(daily_by_symbol: Dict[str, pd.DataFrame],
                   detector_config: Optional[ResistanceConfig] = None,
                   eval_config: Optional[EvalConfig] = None,
                   sources: Optional[List[str]] = None,
                   detector=None) -> dict:
    detector = detector or ResistanceDetector(detector_config)
    cfg = eval_config or EvalConfig()
    src = build_sources(detector, cfg, sources)
    parts = {"reactions": [], "breakouts": [], "uncond": []}
    for symbol, daily in daily_by_symbol.items():
        p = detector.prepare(daily)
        r, b, u = evaluate_symbol(symbol, p, detector, cfg, src)
        parts["reactions"].append(r)
        parts["breakouts"].append(b)
        parts["uncond"].append(u)
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in parts.items()}


def split_by_date(results: dict, split_date) -> Dict[str, dict]:
    split = pd.Timestamp(split_date)
    out = {"tune": {}, "holdout": {}}
    for key, df in results.items():
        if df.empty:
            out["tune"][key] = df
            out["holdout"][key] = df
            continue
        out["tune"][key] = df[df["date"] < split].reset_index(drop=True)
        out["holdout"][key] = df[df["date"] >= split].reset_index(drop=True)
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def summarize_reactions(reactions: pd.DataFrame, by=("source",)) -> pd.DataFrame:
    if reactions.empty:
        return pd.DataFrame()
    rows = []
    for key, g in reactions.groupby(list(by), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        touched = g[g["outcome"] != "no_touch"]
        nt = len(touched)
        rej, brk = int((touched["outcome"] == "reject").sum()), int((touched["outcome"] == "break").sum())
        lo, hi = wilson(rej, nt)
        rows.append({
            **dict(zip(by, key)), "levels": len(g), "touched": nt,
            "width_atr": g["width_atr"].mean(),
            "touch_rate": nt / len(g), "reject_rate": rej / nt if nt else np.nan,
            "reject_ci_lo": lo, "reject_ci_hi": hi,
            "break_rate": brk / nt if nt else np.nan,
            "neutral_rate": 1 - (rej + brk) / nt if nt else np.nan,
        })
    return pd.DataFrame(rows)


def reactions_by_distance(reactions: pd.DataFrame) -> pd.DataFrame:
    if reactions.empty:
        return pd.DataFrame()
    r = reactions.assign(dist_bucket=pd.cut(reactions["distance_atr"], [0, 0.75, 1.5, 3.0]))
    return summarize_reactions(r, by=("source", "dist_bucket"))


def reactions_by_score(reactions: pd.DataFrame, buckets: int = 4) -> pd.DataFrame:
    r = reactions[reactions["source"] == "atr_pivot"].dropna(subset=["score"])
    if len(r) < buckets * 5:
        return pd.DataFrame()
    r = r.assign(score_bucket=pd.qcut(r["score"], buckets, duplicates="drop"))
    return summarize_reactions(r, by=("score_bucket",))


def _return_stats(df: pd.DataFrame, label: dict) -> dict:
    out = {**label, "n": len(df)}
    for col in ("gap_ret", "ret1", "ret5"):
        s = df[col].dropna()
        out[f"{col}_mean"] = s.mean() if len(s) else np.nan
        out[f"{col}_t"] = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 and s.std(ddof=1) > 0 else np.nan
    out["gap_win"] = (df["gap_ret"] > 0).mean() if len(df) else np.nan
    out["mfe5_atr"] = df["mfe5_atr"].mean() if len(df) else np.nan
    if "false_break" in df:
        out["false_break_rate"] = df["false_break"].mean() if len(df) else np.nan
    return out


def summarize_breakouts(breakouts: pd.DataFrame, uncond: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not uncond.empty:
        rows.append(_return_stats(uncond, {"source": "ALL_BARS", "vol_ok": "-"}))
        vol = uncond[uncond["volume_ratio"] >= 1.5]
        rows.append(_return_stats(vol, {"source": "ALL_BARS_vol>=1.5x", "vol_ok": "-"}))
    if not breakouts.empty:
        for (src, vol_ok), g in breakouts.groupby(["source", "vol_ok"]):
            rows.append(_return_stats(g, {"source": src, "vol_ok": vol_ok}))
    return pd.DataFrame(rows)


def room_analysis(breakouts: pd.DataFrame, source: str = "atr_pivot") -> pd.DataFrame:
    b = breakouts[breakouts["source"] == source]
    if len(b) < 12:
        return pd.DataFrame()
    b = b.assign(room_bucket=pd.cut(b["room_atr"], [0, 1, 2, 4, 10.0001], include_lowest=True))
    rows = [_return_stats(g, {"room_bucket": str(k)}) for k, g in b.groupby("room_bucket", observed=True)]
    out = pd.DataFrame(rows)
    out.attrs["spearman_room_vs_mfe5"] = b["room_atr"].corr(b["mfe5_atr"], method="spearman")
    out.attrs["spearman_room_vs_ret5"] = b["room_atr"].corr(b["ret5"], method="spearman")
    return out


def format_report(results: dict, title: str = "") -> str:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pct = lambda x: f"{x:.1%}" if pd.notna(x) else "nan"
    parts = [f"=== {title} ===" if title else "==="]

    rs = summarize_reactions(results["reactions"])
    if not rs.empty:
        parts.append("\n-- Reaction at nearest resistance zone (within max distance) --")
        parts.append(rs.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        parts.append("\n-- By distance bucket (compare sources within a bucket) --")
        parts.append(reactions_by_distance(results["reactions"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        sc = reactions_by_score(results["reactions"])
        if not sc.empty:
            parts.append("\n-- atr_pivot reject rate by strength-score bucket --")
            parts.append(sc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    bs = summarize_breakouts(results["breakouts"], results["uncond"])
    if not bs.empty:
        parts.append("\n-- Breakout forward returns (events overlap; t-stats are optimistic) --")
        parts.append(bs.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    room = room_analysis(results["breakouts"])
    if not room.empty:
        parts.append("\n-- atr_pivot: room to next resistance vs follow-through --")
        parts.append(room.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        parts.append(f"spearman(room, mfe5)={room.attrs['spearman_room_vs_mfe5']:.3f}  "
                     f"spearman(room, ret5)={room.attrs['spearman_room_vs_ret5']:.3f}")
    return "\n".join(parts)
