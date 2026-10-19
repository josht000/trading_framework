"""
Exit-rule variants for the overnight strategy, evaluated on the same entry signals.

Spec keys (all optional except rules):
  rules        False -> no after-hours rule (hold to next session's first bar)
  window_min   minutes after 16:00 ET to watch (default 45)
  stop         fixed stop: exit when a bar closes this far below entry (fraction)
  atr_k, atr_min, atr_max   volatility-scaled stop: clip(atr_k * prior ATR14 / close, atr_min, atr_max)
  target       exit when a bar closes this far above entry
  confirm      consecutive 1-min closes beyond the stop needed to trigger it (default 1)
  pm_target    if no after-hours exit, take profit at the first next-day pre-market bar this far above entry
  pm_stop      if no after-hours exit, stop at the first next-day pre-market bar this far below entry
Triggers use bar closes; fills follow the volume-limited, slippage-adjusted model in pit_runner.
"""

VARIANTS = {
    "v1": dict(rules=False),
    "v2": dict(window_min=45, stop=0.12, target=0.50),
    "w30_stop20": dict(window_min=30, stop=0.20, target=0.50),
    "w30_tgt20": dict(window_min=30, stop=0.12, target=0.20),
    "w30_20_20": dict(window_min=30, stop=0.20, target=0.20),
    "ah_full": dict(window_min=240, stop=0.12, target=0.50),
    "pm_target25_v2": dict(window_min=45, stop=0.12, target=0.50, pm_target=0.25),
    "atr_stop": dict(window_min=45, atr_k=0.6, atr_min=0.08, atr_max=0.25, target=0.50),
    "confirm2": dict(window_min=45, stop=0.12, target=0.50, confirm=2),
    # round-2 exit knobs, off v2 and pm_target25_v2
    "v2_pmstop15": dict(window_min=45, stop=0.12, target=0.50, pm_stop=0.15),
    "v2_pmstop25": dict(window_min=45, stop=0.12, target=0.50, pm_stop=0.25),
    "v2_stop15": dict(window_min=45, stop=0.15, target=0.50),
    "v2_stop10": dict(window_min=45, stop=0.10, target=0.50),
    "pm25_pmstop15": dict(window_min=45, stop=0.12, target=0.50, pm_target=0.25, pm_stop=0.15),
    "pm25_pmstop25": dict(window_min=45, stop=0.12, target=0.50, pm_target=0.25, pm_stop=0.25),
    "pm25_stop15": dict(window_min=45, stop=0.15, target=0.50, pm_target=0.25),
    "pm25_stop10": dict(window_min=45, stop=0.10, target=0.50, pm_target=0.25),
    "pm15_v2": dict(window_min=45, stop=0.12, target=0.50, pm_target=0.15),
    "pm35_v2": dict(window_min=45, stop=0.12, target=0.50, pm_target=0.35),
}
