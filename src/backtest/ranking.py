"""
Trade ranking: every candidate gets a score from components known at the 15:55 entry; higher is better.

Components are each scaled to [-1, 1] and combined with weights, so the score is a weighted sum. The weights are themselves a
tunable part of the strategy (see rank_test.py). Three scalings are provided:
  raw         weighted sum (unnormalized float)
  integer     round(raw) clipped to [-5, 5]
  normalized  (raw - lo) / (hi - lo) in [0, 1] using the theoretical bounds of the weights
"""

import numpy as np
import pandas as pd

COMPONENTS = ["headroom", "float", "trend200", "market", "news", "movers", "edgar_pos", "delist_notice", "late_filing", "dilution", "theme"]

DEFAULT_WEIGHTS = {"headroom": 1.0, "float": 1.0, "trend200": 1.0, "market": 1.0, "news": 1.0, "movers": 0.5, "edgar_pos": 1.0,
                   "delist_notice": 2.0, "late_filing": 1.5, "dilution": 0.75, "theme": 1.0}


def _num(t, col, default=np.nan):
    return t[col].astype(float) if col in t.columns else pd.Series(default, index=t.index)


def _log_count(n, per_doubling=0.5):
    """Count-aware score in [0, 1]: 0 for none, 0.5 for one, about 0.8 for two, 1.0 for three or more."""
    return np.minimum(1.0, per_doubling * np.log2(1.0 + np.asarray(n, dtype=float)))


def components(t):
    """Component matrix (columns in COMPONENTS), each in [-1, 1]; unknown inputs score 0.
    News (Benzinga) and EDGAR filings are separate components; news and top-mover signals are count-aware."""
    c = pd.DataFrame(index=t.index)
    h = _num(t, "headroom")
    c["headroom"] = np.select([h < 0.05, h < 0.10, h < 0.20, h < 0.50, h >= 0.50], [-1.0, -0.5, 0.0, 0.5, 1.0], default=0.0)

    sh = _num(t, "shares_pit").fillna(_num(t, "float_now"))
    c["float"] = np.select([sh < 30e6, sh < 100e6, sh < 300e6], [1.0, 0.0, -0.5], default=0.0)

    a = _num(t, "above_sma200")
    c["trend200"] = np.where(a == 1, 1.0, np.where(a == 0, -1.0, 0.0))

    iw, sp = _num(t, "iwm_day"), _num(t, "spy_day")
    c["market"] = 0.5 * np.where(iw.isna(), 0.0, np.where(iw >= 0, 1.0, -1.0)) + 0.5 * np.where(sp.isna(), 0.0, np.where(sp >= 0, 1.0, -1.0))

    # news: positive = event headlines (count-aware) plus a quarter point for other company-specific headlines;
    # negative = headlines with negative keywords (count-aware); the two net against each other
    ev, sp_n, ng = _num(t, "n_event_sc", 0).fillna(0), _num(t, "n_specific_sc", 0).fillna(0), _num(t, "n_neg_sc", 0).fillna(0)
    pos = np.minimum(1.0, _log_count(ev) + 0.25 * ((sp_n - ev) > 0))
    c["news"] = np.clip(pos - _log_count(ng), -1.0, 1.0)

    # recent top-mover appearances: days in the previous 5 sessions among the top-20 daily gainers plus Benzinga mover-list mentions
    tg, mv = _num(t, "n_top_gainer_days", 0).fillna(0), _num(t, "n_mover_5d", 0).fillna(0)
    c["movers"] = _log_count(tg + np.minimum(mv, 5.0))

    # EDGAR: positive filings since the prior close (separate from news)
    e101, e201, e701 = (_num(t, k, 0).fillna(0) for k in ("e_item_101", "e_item_201", "e_item_701_801"))
    c["edgar_pos"] = np.where((e101 > 0) | (e201 > 0), 1.0, np.where(e701 > 0, 0.5, 0.0))
    # EDGAR penalties, each its own component: exchange delisting notice (8-K 3.01) in 90 days, late-filing notice (NT 10-K/10-Q)
    # in 30 days, and dilution (offering forms or item 3.02 in 10 days; item 1.03 / 4.02 in 30 days also counts as distress here)
    c["delist_notice"] = -(_num(t, "e_delist_notice_90d", 0).fillna(0) > 0).astype(float)
    c["late_filing"] = -(_num(t, "e_late_30d", 0).fillna(0) > 0).astype(float)
    dil = (_num(t, "e_offering_10d", 0).fillna(0) > 0) | (_num(t, "e_item_302_10d", 0).fillna(0) > 0) | (_num(t, "e_other_distress_30d", 0).fillna(0) > 0)
    c["dilution"] = -dil.astype(float)

    th, sh_ = _num(t, "theme_heat", 0).fillna(0), _num(t, "sector_heat", 0).fillna(0)
    c["theme"] = np.clip(np.where(th >= 3, 1.0, np.where(th >= 1, 0.5, 0.0)) + np.where(sh_ >= 3, 0.5, 0.0), 0.0, 1.0)
    return c[COMPONENTS]


def score(t, weights=None, scale="raw"):
    w = weights or DEFAULT_WEIGHTS
    c = components(t)
    raw = sum(w.get(k, 0.0) * c[k] for k in COMPONENTS)
    if scale == "raw":
        return raw
    if scale == "integer":
        return raw.round().clip(-5, 5)
    if scale == "normalized":
        hi = sum(abs(w.get(k, 0.0)) for k in COMPONENTS)
        lo = -hi
        return (raw - lo) / (hi - lo) if hi > 0 else raw * 0
    raise ValueError(scale)
