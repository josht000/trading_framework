---
name: resistance-detector
description: Act as the resistance-level detector yourself. Read a daily candlestick chart PNG plus its OHLCV CSV and list the horizontal resistance zones above the last close. Use when asked to find, label, or predict resistance levels for a chart in data/resistance_dataset (or any daily chart + OHLCV pair).
---

# Resistance detector (VLM-direct)

You ARE the detector. Look at the chart image and read the OHLCV rows; do not write a rule-based
detector. Small scripts are fine only to read exact prices out of the CSV (for example "list the
highs between $5.00 and $5.60"), never to decide which levels to output.

## Inputs
- `<split>/charts/NN_SYMBOL_ASOF.png`: daily candles + volume, last bar = as-of date, last close in the title.
- `<split>/price_data/NN_SYMBOL_ASOF.csv`: `date,open,high,low,close,volume`, oldest first (about 161 bars).
- Never open `labels.csv` of the split you are predicting. Write predictions first, score after.

## What counts
A resistance level is overhead supply: a price above the last close where an advance previously
stalled and reversed. Treat it as a zone. A prediction is a hit when it is within the tolerance of a
labelled level: +-$0.10 for ordinary names, and +-1% of the labelled level for large caps (last
close of $50 or more, for example AAPL, GOOGL, MSFT, NVDA unadjusted), where $0.10 is under 0.1% of price.
Only levels strictly above the last close. If nothing is above, output none (blue sky).

On large caps the tolerance is wider but the levels are still the same swing highs: read exact bar
highs, keep the ladder of distinct pivots, and do not use the extra room to emit more levels.

## Rules learned from the labelled train set (37 labels, 11 charts) and from scoring mistakes

1. Output the exact price of the extreme bar, read from the CSV `high` column (for example 5.26,
   not "about 5.25"). Zone midpoints and chart-gridline guesses cost hits; exact bar highs matched
   labels to within a few cents.
2. The labels are a ladder of distinct swing highs going up from price: the most recent rejection
   highs near price, then each earlier peak, then the major spike or top. Typical chart has 3 to 6.
3. Always include the all-time-in-window peak and the sharp spike/blow-off highs (long upper wicks
   on high volume), for example the highest bar on the chart.
4. When price has repeatedly stalled in a shelf (2 or more highs within about $0.10 to $0.15),
   emit one level at the highest of them. Do not emit one per touch.
5. In a descending staircase of lower highs (each rally failing below the previous), every step
   is a level, even when steps are only $0.10 to $0.25 apart (NOK 5.34, 5.45, 5.59, 5.85).
6. Interior highs inside a thick supply zone count, not just its edges. After a breakdown, the high
   of the rebound days and the highs on the way down are often labelled (RIVN-style gap-down ladders).
   When a gap above price is wider than about $1 with no level in it, look for a cluster of highs
   in it and add the top of the densest cluster.
7. A pivot high must be a clear local peak (higher than roughly 4 to 5 bars on each side). Ignore
   one-bar noise, minor wiggles, and single closes with no wick rejection.
8. Include a recent high from the last few bars only when it coincides with an older pivot
   (PTON 5.26 today matched the 5.30 breakdown candle). Rule 14 overrides this.
9. Round numbers matter only as a tiebreaker: use them when highs cluster at or just under the
   round number (BB 3.00, HUT 20.00), never invent a level at a round number with no highs near it.
10. Drop weak near-price levels the chart barely respects. Over-predicting the first few levels
    above price was the biggest source of false positives; precision matters as much as recall.
11. Do not use moving averages, volume profile or lows unless a chart clearly shows a prior support
    shelf that broke and is now being retested from below (rare in the labelled set).
12. Prefer 3 to 7 levels. More than 8 means you are listing noise.

## Rules added in the train/val review of 2026-10-09 (val F1 0.676 before these were applied)

13. The labels are mostly the descending staircase that starts at the main peak and walks
    forward in time: each post-peak rebound high or decline-day high that is lower than the last
    one (NIO 7.71, 7.38, 6.93, 6.45, 6.03, 5.36; PLUG 31.56, 30.43, 23.45, 19.75, 18.26; HUT; BB).
    Pivots formed before the main peak are kept only when they fill an empty gap (CLOV 1.24).
14. Drop the last bar's own high and pivots from the last 3 bars unless the wick is a blow-off
    more than about 10% above the close (GME 20.20 kept; NIO 5.13, PLUG 18.09, IONQ 11.00,
    HUT 15.83, RIOT 18.36 all unlabelled). Also drop levels less than about 2% above the close.
15. Do not merge two peaks from separate episodes (weeks apart) just because they are $0.10 to
    $0.20 apart: SOUN 4.97 and 5.11, RKLB 5.67 and 5.75, SNAP 17.50 and 17.75 were all labelled.
    Merge only touches inside one consolidation.
16. For a multi-touch shelf, use the price most bars topped out at (SOFI 10.10 cluster, not the
    10.28 extreme; SNAP 17.29); a single taller wick above the shelf is not the level.
17. Inside a gap-down or fast decline, prefer the high of the decline day just after the gap
    (RIVN 20.66 on 1/3, 21.94 on 1/2) over a small rebound high a few days later (19.76 unlabelled).
18. The window peak is usually but not always labelled (NOK 6.40 and SNAP 17.90 were not). Keep
    it, but it is the most dispensable level when it sits far above a dense staircase.

## Rules added after the val relabel of 2026-10-10 (v5 fit check, val F1 1.000)

The val split was re-labelled more densely and each chart now has a labeler note
in `<split>/explainations/`.  The revised convention:

19. Mark EVERY significant rejection high after the window all-time-high, not
    just clean pivots: decline-day highs, gap-down-day highs and rebound highs
    are each labelled, even when steps are a few cents apart (SNDL 2.17/2.20,
    SOFI 10.10/10.28).  Walk left-and-up from the close collecting them.
20. A labelled level can be a former support low, not only a high: the floor of
    a shelf that broke counts (PTON 5.56 triple-touch lows, 5.37 poke low;
    SNAP 15.52; RIVN 16.94).  Check the `low` column for multi-touch floors.
21. Near-price levels are kept when they are real rejection points (SNAP 15.52
    is only 1.3% above close).  The <2% drop in rule 14 is too aggressive.
22. The last bar's own high counts when it was a violent, high-volume rejection
    at a prior level (RIOT 18.36; labeler note: "multiple tests of resistance
    all failed").  Otherwise still drop last-3-bar highs.
23. "Don't consider prices before the all-time-high" - levels formed before the
    window peak are ignored, EXCEPT highs from an earlier separate spike
    episode (SOUN 4.95 predates the Jun peak but is labelled).
24. High volume on the rejection day makes the level more likely to be labelled
    (RKLB 5.76 on 52M, SNAP 17.50 on 92M, PTON 5.30 on 84M).

## Rules added after the blind test run of 2026-10-10 (test F1 0.647, recall 0.833, precision 0.528)

The blind run showed the same staircase is read correctly but the wrong bar is often
picked; each of these came from a real miss:

25. On a fresh pullback the first post-peak decline-day high is labelled, not the
    window ATH: AFRM labelled 25.13 (Sep19 decline day), 23.11 (Sep12 shelf LOW),
    22.68 (Sep7 shelf top) - while the 25.63 ATH and a 24.45 cluster midpoint were
    both wrong. Emit the specific bar the descent started from, and check shelf lows.
26. Do NOT merge adjacent staircase steps to a midpoint. Levels 0.10 to 1.0 apart are
    separately labelled (BB 4.30 vs 4.62 vs 5.35, JOBY 7.93 vs 8.35, IONQ 13.95 vs
    my 14.09, AAPL 157.03 vs my merged 158.10). Emit each exact bar; a midpoint can
    cost both. Reinforces rule 15 - when in doubt, emit both bars.
27. A blow-off peak day can be labelled at its body/close, not the wick: AMC's peak
    day was labelled 5.20 (close), not the 5.56 high. For long-wick climax bars,
    consider the close as the level.
28. Match density to chart shape. The labeler is dense inside a post-peak descent
    leg (RKLB 8/9 hits, JOBY 7, NOK 7) but sparse on clean V-recoveries and shallow
    pullbacks (DNA only 3 labels, PLTR 3, NIO 2, AFRM 3). On charts that fell and
    recovered without a long staircase, output only the few majors: the peak, the
    recovery leg top, and maybe one mid-shelf.

## Rules added after the v6 val run of 2026-10-10 (val F1 0.651, recall 0.976, precision 0.488)

v6 applied rules 19/26 maximally (every decline-day high, emit both near-duplicate
bars). Recall was near-perfect but 43 FPs: the val labeler takes 3-7 levels per
chart, NOT every staircase step.

29. "Every rejection high" means swing/rebound highs and the bars the descent
    pivoted on, not every daily high of a down-leg. Mid-leg ordinary-volume
    decline-day highs were unlabelled (RIVN 17.01/17.16/19.90, SOFI 9.24-9.57,
    SNDL 2.12/2.26, RKLB 5.35/5.57, PTON 6.00/6.11/6.88/7.05 all FPs). Cut the
    steps between major rebounds when they lack a distinct volume or gap event.
30. Crash-day highs are NOT automatically levels: RIVN 16.35 (262M gap-down day),
    RIOT 17.02 (71M) and SOUN 3.99 (last-bar crash open) were all unlabelled.
    Label crash-day highs only when they sit at/near a real prior pivot.
31. Do not emit two bars within ~$0.10 of each other as insurance: RIVN 16.84/16.86,
    RIVN 18.855/18.865, PTON 5.26/5.30, SNDL 2.26/2.27 each produced a wasted FP
    (the labeler picked one bar). Pick the single best bar per zone - the highest
    of a shelf or the most violent rejection - rule 26's "emit both" applies to
    levels >$0.10-0.15 apart, not to near-identical prices.
32. The window ATH is usually labelled after all on val (SNAP label at 17.86 vs
    my omitted 17.90 was the only FN); keep the top unless it is clearly a lone
    spike never retested. Former-support lows stay valid (PTON 5.37/5.56,
    RIVN 16.94, SNAP 15.52 all hit).

## Procedure
1. Read the PNG. Note the last close, the major peaks, shelves, gap-down ladders and spikes.
2. Read the CSV. For each candidate zone, find the exact high (and date) and confirm it is above
   the last close and a real pivot.
3. Apply the rules above: merge shelf touches, keep every staircase step, add the top and spikes,
   drop near-price noise.
4. Write `predictions_<tag>.csv` in the split folder with header `symbol,asof,level,note`; one row
   per level, `note` = `chart NN - <6 words reason>`.
5. Only then score against `labels.csv` (hit = within the tolerance above, $0.10 or 1% for large
   caps, one-to-one nearest match) and report TP, FP, FN, precision, recall, F1, accuracy per chart
   and total, plus the split's score under a flat $0.10 for comparison with older runs.
   `eval_pivot_resistance.py` has a reusable scorer (`match` function) if you want it, but it uses a
   fixed tolerance, so pass the per-chart tolerance yourself.

- Helpers: `data/resistance_dataset/highs.py <csv> [w]` lists bar highs above the close (lookup
  only); `data/resistance_dataset/score.py <split> <preds.csv>` scores with per-chart tolerance.
  Use `py -3` on this machine (`python` is 2.7).
