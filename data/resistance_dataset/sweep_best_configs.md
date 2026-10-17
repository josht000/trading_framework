# Best config per detector - resistance sweep

Summary of `sweep_results.csv` (3,318 configs, 9 detectors, tuned on train+val,
held-out test split scored by `score.py`). Generated 2026-10-11.
For every detector the accuracy-best and F1-best config turned out to be the
same, so 'best' below is unambiguous.

## Bottom line

- **pivot** wins the tuning data: accuracy 0.551 on train+val (F1 0.710).
- On the **unseen test split** **atr_pivot** edges it out, accuracy 0.488 vs 0.471 - effectively a tie.
- Best VLM chart-reader baseline (vlm_audit_20261009) still beats every detector on
  test accuracy: 0.567 (see the last table).

## Scoring, in plain English

A predicted level hits a label when it lands within +-$0.10 of it (or +-1% of the
label when the stock's last close is >= $50). One-to-one nearest matching. Then:

- **accuracy** = TP / (TP + FP + FN): of all levels involved, predicted or labeled, the
  share the detector matched. Penalizes false alarms and missed levels alike.
- **precision**: of the levels the detector predicted, the share that hit a label.
- **recall**: of the labeled levels, the share the detector found.
- **F1**: the precision/recall balance; what the sweep originally ranked by.

## Best of each detector, ranked by test accuracy

| # | Detector | Test accuracy | Test F1 | Test precision | Test recall | Train+val accuracy | Configs swept |
|---|----------|---------------|---------|----------------|-------------|--------------------|---------------|
| 1 | atr_pivot | 0.488 | 0.656 | 0.618 | 0.700 | 0.462 | 864 |
| 2 | pivot | 0.471 | 0.640 | 0.575 | 0.722 | 0.550 | 180 |
| 3 | zigzag | 0.470 | 0.640 | 0.538 | 0.789 | 0.532 | 1,344 |
| 4 | raw_price | 0.445 | 0.616 | 0.497 | 0.811 | 0.521 | 192 |
| 5 | meanshift | 0.419 | 0.590 | 0.517 | 0.689 | 0.403 | 128 |
| 6 | deriv_extrema | 0.379 | 0.549 | 0.435 | 0.744 | 0.500 | 112 |
| 7 | volume_levels | 0.249 | 0.399 | 0.271 | 0.756 | 0.360 | 288 |
| 8 | floor_pivots | 0.189 | 0.318 | 0.338 | 0.300 | 0.221 | 48 |
| 9 | volume_profile | 0.115 | 0.206 | 0.304 | 0.156 | 0.065 | 162 |

## Winning settings

### 1. atr_pivot

`src/screening/resistance.py` - best of 864 configs by train+val F1 (and by accuracy).

- pivot_left = 3
- pivot_right = 3
- min_prominence_atr = 0.5
- merge_atr = 0.3
- max_zone_atr = 1.5
- lookback_bars = 252
- include_ranges = False
- include_swing_lows = True

| split | accuracy | F1 | precision | recall | TP / FP / FN |
|-------|----------|----|-----------|--------|--------------|
| train | 0.500 | 0.667 | 0.625 | 0.714 | 30 / 18 / 12 |
| val | 0.413 | 0.585 | 0.826 | 0.452 | 19 / 4 / 23 |
| test | 0.488 | 0.656 | 0.618 | 0.700 | 63 / 39 / 27 |

### 2. pivot

`src/screening/pivot_resistance.py` - best of 180 configs by train+val F1 (and by accuracy).

- pivot_bars = 4
- merge_tol = 0.1
- edge_pivots = True
- max_levels = null (no cap)

| split | accuracy | F1 | precision | recall | TP / FP / FN |
|-------|----------|----|-----------|--------|--------------|
| train | 0.542 | 0.703 | 0.653 | 0.762 | 32 / 17 / 10 |
| val | 0.560 | 0.718 | 0.778 | 0.667 | 28 / 8 / 14 |
| test | 0.471 | 0.640 | 0.575 | 0.722 | 65 / 48 / 25 |

### 3. zigzag

`src/screening/cluster_levels.py` - best of 1,344 configs by train+val F1 (and by accuracy).

- series = high
- peak_percent_delta = 3
- peaks = High
- min_bars_between_peaks = 0
- level_selector = mean
- merge_distance = 0.1

| split | accuracy | F1 | precision | recall | TP / FP / FN |
|-------|----------|----|-----------|--------|--------------|
| train | 0.522 | 0.686 | 0.571 | 0.857 | 36 / 27 / 6 |
| val | 0.544 | 0.705 | 0.674 | 0.738 | 31 / 15 / 11 |
| test | 0.470 | 0.640 | 0.538 | 0.789 | 71 / 61 / 19 |

### 4. raw_price

`src/screening/cluster_levels.py` - best of 192 configs by train+val F1 (and by accuracy).

- series = high
- bars_for_peak = 5
- level_selector = mean
- merge_distance = 0.1

| split | accuracy | F1 | precision | recall | TP / FP / FN |
|-------|----------|----|-----------|--------|--------------|
| train | 0.508 | 0.673 | 0.589 | 0.786 | 33 / 23 / 9 |
| val | 0.537 | 0.699 | 0.707 | 0.691 | 29 / 12 / 13 |
| test | 0.445 | 0.616 | 0.497 | 0.811 | 73 / 74 / 17 |

### 5. meanshift

`src/screening/sr_finder.py` - best of 128 configs by train+val F1 (and by accuracy).

- series = high
- smooth = False
- quantile = 0.1
- emit = max
- min_cluster_size = 1

| split | accuracy | F1 | precision | recall | TP / FP / FN |
|-------|----------|----|-----------|--------|--------------|
| train | 0.425 | 0.596 | 0.500 | 0.738 | 31 / 31 / 11 |
| val | 0.375 | 0.545 | 0.600 | 0.500 | 21 / 14 / 21 |
| test | 0.419 | 0.590 | 0.517 | 0.689 | 62 / 58 / 28 |

### 6. deriv_extrema

`src/screening/sr_finder.py` - best of 112 configs by train+val F1 (and by accuracy).

- series = high
- smooth = False
- mode = cross
- emit = raw
- merge_tol = 0.1

| split | accuracy | F1 | precision | recall | TP / FP / FN |
|-------|----------|----|-----------|--------|--------------|
| train | 0.500 | 0.667 | 0.593 | 0.762 | 32 / 22 / 10 |
| val | 0.500 | 0.667 | 0.644 | 0.691 | 29 / 16 / 13 |
| test | 0.379 | 0.549 | 0.435 | 0.744 | 67 / 87 / 23 |

### 7. volume_levels

`src/screening/sr_finder.py` - best of 288 configs by train+val F1 (and by accuracy).

- series = high
- smooth = True
- threshold = 0.02
- relative = True
- percentile = 50
- emit = raw
- merge_tol = 0.05

| split | accuracy | F1 | precision | recall | TP / FP / FN |
|-------|----------|----|-----------|--------|--------------|
| train | 0.326 | 0.492 | 0.375 | 0.714 | 30 / 50 / 12 |
| val | 0.406 | 0.577 | 0.509 | 0.667 | 28 / 27 / 14 |
| test | 0.249 | 0.399 | 0.271 | 0.756 | 68 / 183 / 22 |

### 8. floor_pivots

`src/screening/sr_finder.py` - best of 48 configs by train+val F1 (and by accuracy).

- lookback = 60
- variant = camarilla
- which = all

| split | accuracy | F1 | precision | recall | TP / FP / FN |
|-------|----------|----|-----------|--------|--------------|
| train | 0.178 | 0.302 | 0.295 | 0.309 | 13 / 31 / 29 |
| val | 0.276 | 0.432 | 0.500 | 0.381 | 16 / 16 / 26 |
| test | 0.189 | 0.318 | 0.338 | 0.300 | 27 / 53 / 63 |

### 9. volume_profile

`src/screening/volume_profile.py` - best of 162 configs by train+val F1 (and by accuracy).

- max_above = 5
- min_gap_atr = 0.5
- bin_atr = 0.1
- min_prominence_frac = 0.03
- recency_halflife_bars = 252

| split | accuracy | F1 | precision | recall | TP / FP / FN |
|-------|----------|----|-----------|--------|--------------|
| train | 0.107 | 0.194 | 0.300 | 0.143 | 6 / 14 / 36 |
| val | 0.020 | 0.038 | 0.100 | 0.024 | 1 / 9 / 41 |
| test | 0.115 | 0.206 | 0.304 | 0.156 | 14 / 32 / 76 |

## VLM baselines (not part of the sweep)

Vision-language models reading the blind chart PNG + OHLCV CSV (`vlm_benchmark.py`);
no tunable settings - predictions come straight from the model. Same scorer, test split:

| Run | Test accuracy | Test F1 | Test precision | Test recall |
|-----|---------------|---------|----------------|-------------|
| vlm_audit_20261009 | 0.567 | 0.724 | 0.661 | 0.800 |
| vlm2 | 0.529 | 0.692 | 0.610 | 0.800 |

## Where the full data lives

- Every config and its scores: `sweep_results.csv` (this file summarizes it)
- Winner predictions per split: `<split>/predictions_<detector>.csv`
- Winner eval output per split: `<split>/eval_results_<detector>.txt`
- Matching rule and accuracy definition: `score.py`
