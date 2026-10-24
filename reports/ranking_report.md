# Trade ranking experiments (dev set)

2273 dev trades (pm_target25_v2 exits, 13-month atr_pivot); score components: headroom, float, trend200, market, news, movers, edgar_pos, delist_notice, late_filing, dilution, theme; default weights {'headroom': 1.0, 'float': 1.0, 'trend200': 1.0, 'market': 1.0, 'news': 1.0, 'movers': 0.5, 'edgar_pos': 1.0, 'delist_notice': 2.0, 'late_filing': 1.5, 'dilution': 0.75, 'theme': 1.0}. OTC tickers flagged: 7 trades; trades with EDGAR data: 99%.

### Hygiene filters on the base strategy (rank by relative volume, as before)

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| no hygiene | 1547 | 40.59 | 0.32 | 1.14 | 97.66 | -44.23 | 4336.48 | -10.08 | 9 | 0.93 |
| + no OTC / pink sheets | 1547 | 40.66 | 0.35 | 1.15 | 116.55 | -41.74 | 4336.48 | -10.08 | 9 | 1.02 |
| + no EDGAR distress (3.01/1.03/4.02/NT) | 1506 | 40.57 | 0.22 | 1.10 | 49.02 | -46.19 | 4241.58 | -10.08 | 9 | 0.69 |
| + no recent offering / 3.02 | 1440 | 40.21 | 0.06 | 1.03 | -4.15 | -52.96 | 3480.46 | -10.05 | 9 | 0.20 |

### Ranking with the default weights (base strategy, OTC excluded)

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| rank by rel. volume, top 5 | 1547 | 40.66 | 0.35 | 1.15 | 116.55 | -41.74 | 4336.48 | -10.08 | 9 | 1.02 |
| rank by score (default weights), top 5 | 1547 | 41.05 | 0.28 | 1.12 | 75.03 | -42.10 | 4626.71 | -10.05 | 9 | 0.83 |
| score >= 0, top 5 | 1329 | 41.46 | 0.37 | 1.16 | 101.44 | -40.52 | 4413.28 | -10.05 | 9 | 0.98 |
| score >= 1, top 5 | 1070 | 42.34 | 0.46 | 1.19 | 109.98 | -37.28 | 4424.88 | -8.86 | 8 | 1.05 |
| score >= 2, top 5 | 765 | 42.75 | 0.70 | 1.31 | 140.15 | -40.54 | 4875.91 | -8.75 | 8 | 1.28 |
| score >= 3, top 5 | 364 | 40.38 | 0.98 | 1.41 | 86.10 | -20.10 | 5000.00 | -8.75 | 4 | 1.25 |
| score >= 1, top 3 | 809 | 41.78 | 0.37 | 1.15 | 53.99 | -36.45 | 4854.19 | -6.94 | 9 | 0.76 |
| score >= 1, top 2 | 608 | 41.12 | 0.36 | 1.14 | 37.05 | -38.31 | 5000.00 | -6.53 | 11 | 0.66 |

### Existing top strategies with ranking (OTC excluded; EDGAR distress/dilution flags are score components)

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| base / rel_vol | 1547 | 40.66 | 0.35 | 1.15 | 116.55 | -41.74 | 4336.48 | -10.08 | 9 | 1.02 |
| base / score | 1547 | 41.05 | 0.28 | 1.12 | 75.03 | -42.10 | 4626.71 | -10.05 | 9 | 0.83 |
| base / score>=1 | 1070 | 42.34 | 0.46 | 1.19 | 109.98 | -37.28 | 4424.88 | -8.86 | 8 | 1.05 |
| base / score>=2 | 765 | 42.75 | 0.70 | 1.31 | 140.15 | -40.54 | 4875.91 | -8.75 | 8 | 1.28 |
| S1 / rel_vol | 660 | 43.94 | 0.92 | 1.51 | 195.50 | -18.16 | 4921.56 | -7.69 | 3 | 1.76 |
| S1 / score | 660 | 43.64 | 0.84 | 1.48 | 171.83 | -18.29 | 4904.59 | -8.75 | 3 | 1.74 |
| S1 / score>=1 | 639 | 43.51 | 0.84 | 1.48 | 162.18 | -18.46 | 4853.47 | -8.75 | 3 | 1.68 |
| S1 / score>=2 | 538 | 44.42 | 0.98 | 1.55 | 158.29 | -18.81 | 5000.00 | -8.75 | 3 | 1.74 |
| H1 / rel_vol | 479 | 42.17 | 1.16 | 1.42 | 155.17 | -31.47 | 4904.93 | -8.10 | 8 | 1.39 |
| H1 / score | 479 | 42.17 | 1.17 | 1.43 | 159.10 | -30.01 | 4904.93 | -8.10 | 8 | 1.41 |
| H1 / score>=1 | 446 | 41.70 | 0.70 | 1.26 | 63.17 | -32.35 | 4904.93 | -8.10 | 8 | 0.88 |
| H1 / score>=2 | 368 | 41.03 | 0.73 | 1.29 | 52.70 | -27.83 | 4904.93 | -7.58 | 7 | 0.82 |
| H3 / rel_vol | 394 | 44.16 | 1.41 | 1.60 | 164.86 | -23.72 | 4911.98 | -7.58 | 7 | 1.54 |
| H3 / score | 394 | 44.16 | 1.41 | 1.60 | 165.10 | -23.72 | 4911.98 | -7.58 | 7 | 1.54 |
| H3 / score>=1 | 378 | 43.92 | 1.11 | 1.48 | 106.20 | -22.05 | 4911.98 | -7.58 | 7 | 1.27 |
| H3 / score>=2 | 333 | 42.94 | 1.12 | 1.49 | 90.35 | -22.05 | 4911.98 | -7.58 | 7 | 1.19 |
| H2 / rel_vol | 204 | 43.14 | 1.26 | 1.62 | 59.41 | -19.20 | 4966.27 | -7.58 | 2 | 1.23 |
| H2 / score | 204 | 43.14 | 1.26 | 1.63 | 59.55 | -19.20 | 4966.27 | -7.58 | 2 | 1.24 |
| H2 / score>=1 | 200 | 43.00 | 1.22 | 1.61 | 55.71 | -16.98 | 4966.27 | -7.58 | 2 | 1.23 |
| H2 / score>=2 | 195 | 42.05 | 1.01 | 1.50 | 42.91 | -16.10 | 4966.27 | -7.58 | 2 | 1.07 |
| base<=100M / rel_vol | 1093 | 42.09 | 0.51 | 1.20 | 128.55 | -44.89 | 4610.53 | -10.01 | 8 | 1.09 |
| base<=100M / score | 1093 | 42.54 | 0.60 | 1.24 | 173.92 | -44.70 | 4655.57 | -9.23 | 8 | 1.25 |
| base<=100M / score>=1 | 758 | 42.35 | 0.53 | 1.20 | 82.23 | -38.12 | 4506.14 | -7.58 | 8 | 0.93 |
| base<=100M / score>=2 | 554 | 43.32 | 0.97 | 1.41 | 146.27 | -29.83 | 4893.16 | -6.94 | 7 | 1.35 |
| S1<=100M / rel_vol | 424 | 46.23 | 1.31 | 1.72 | 174.34 | -14.74 | 4813.91 | -7.49 | 3 | 1.85 |
| S1<=100M / score | 424 | 45.75 | 1.26 | 1.71 | 166.34 | -14.21 | 4794.73 | -6.94 | 3 | 1.86 |
| S1<=100M / score>=1 | 414 | 45.89 | 1.29 | 1.73 | 165.73 | -14.21 | 4794.73 | -6.94 | 3 | 1.86 |
| S1<=100M / score>=2 | 381 | 46.46 | 1.45 | 1.85 | 176.77 | -13.77 | 4925.72 | -6.94 | 3 | 1.96 |
| H3<=100M / rel_vol | 198 | 43.43 | 2.48 | 1.94 | 138.43 | -17.10 | 5000.00 | -5.20 | 5 | 1.50 |
| H3<=100M / score | 198 | 43.43 | 2.48 | 1.94 | 138.43 | -17.10 | 5000.00 | -5.20 | 5 | 1.50 |
| H3<=100M / score>=1 | 191 | 42.41 | 1.88 | 1.71 | 87.53 | -17.10 | 5000.00 | -5.20 | 5 | 1.23 |
| H3<=100M / score>=2 | 185 | 43.24 | 1.93 | 1.74 | 87.14 | -17.10 | 5000.00 | -5.20 | 5 | 1.23 |
| F30-H20 / rel_vol | 88 | 44.32 | 5.78 | 2.28 | 147.31 | -13.21 | 5000.00 | -4.28 | 3 | 1.56 |
| F30-H20 / score | 88 | 44.32 | 5.78 | 2.28 | 147.31 | -13.21 | 5000.00 | -4.28 | 3 | 1.56 |
| F30-H20 / score>=1 | 78 | 42.31 | 3.81 | 1.83 | 66.57 | -20.13 | 5000.00 | -4.28 | 3 | 1.05 |
| F30-H20 / score>=2 | 69 | 42.03 | 3.89 | 1.85 | 58.13 | -21.83 | 5000.00 | -4.28 | 2 | 0.97 |
| F30-M200-H20 / rel_vol | 70 | 44.29 | 5.93 | 2.15 | 106.67 | -13.69 | 5000.00 | -3.96 | 3 | 1.31 |
| F30-M200-H20 / score | 70 | 44.29 | 5.93 | 2.15 | 106.67 | -13.69 | 5000.00 | -3.96 | 3 | 1.31 |
| F30-M200-H20 / score>=1 | 61 | 42.62 | 3.50 | 1.66 | 41.94 | -22.41 | 4644.82 | -3.96 | 3 | 0.78 |
| F30-M200-H20 / score>=2 | 53 | 43.40 | 4.14 | 1.78 | 43.92 | -19.80 | 4800.93 | -3.96 | 2 | 0.81 |
| M200 / rel_vol | 508 | 44.88 | 1.61 | 1.44 | 291.48 | -30.22 | 5000.00 | -9.87 | 6 | 1.65 |
| M200 / score | 508 | 44.69 | 1.56 | 1.42 | 273.93 | -30.22 | 5000.00 | -9.87 | 6 | 1.61 |
| M200 / score>=1 | 350 | 44.29 | 1.16 | 1.31 | 88.79 | -31.86 | 5000.00 | -7.72 | 6 | 1.01 |
| M200 / score>=2 | 245 | 46.94 | 2.12 | 1.63 | 144.17 | -21.43 | 5000.00 | -7.72 | 4 | 1.41 |
| M200-H20 / rel_vol | 129 | 44.19 | 3.06 | 1.64 | 94.40 | -27.66 | 5000.00 | -7.01 | 3 | 1.15 |
| M200-H20 / score | 129 | 44.19 | 3.06 | 1.64 | 94.40 | -27.66 | 5000.00 | -7.01 | 3 | 1.15 |
| M200-H20 / score>=1 | 114 | 43.86 | 2.11 | 1.47 | 47.22 | -30.00 | 4654.14 | -7.01 | 3 | 0.80 |
| M200-H20 / score>=2 | 96 | 44.79 | 2.26 | 1.50 | 41.74 | -24.65 | 4840.72 | -7.01 | 3 | 0.76 |

### Component ablation (score >= 1, top 5): what does each component contribute?

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| all components | 1070 | 42.34 | 0.46 | 1.19 | 109.98 | -37.28 | 4424.88 | -8.86 | 8 | 1.05 |
| without headroom | 1228 | 41.45 | 0.29 | 1.13 | 60.43 | -38.55 | 4398.21 | -8.40 | 12 | 0.76 |
| without float | 1068 | 42.42 | 0.44 | 1.21 | 109.62 | -38.61 | 4444.18 | -8.61 | 10 | 1.09 |
| without trend200 | 927 | 40.67 | 0.40 | 1.18 | 73.54 | -40.36 | 4755.42 | -8.02 | 5 | 0.89 |
| without market | 1075 | 42.70 | 0.52 | 1.22 | 140.03 | -46.90 | 4544.21 | -10.05 | 16 | 1.18 |
| without news | 1056 | 41.95 | 0.40 | 1.17 | 84.60 | -37.28 | 4348.70 | -10.43 | 7 | 0.91 |
| without movers | 1053 | 42.45 | 0.44 | 1.19 | 101.88 | -37.14 | 4413.74 | -8.86 | 7 | 1.02 |
| without edgar_pos | 1050 | 42.19 | 0.48 | 1.20 | 117.40 | -36.96 | 4424.88 | -8.86 | 8 | 1.09 |
| without delist_notice | 1094 | 42.14 | 0.47 | 1.19 | 114.70 | -37.28 | 4523.17 | -8.86 | 8 | 1.06 |
| without late_filing | 1074 | 42.18 | 0.42 | 1.18 | 92.78 | -39.69 | 4424.88 | -8.86 | 8 | 0.95 |
| without dilution | 1094 | 42.14 | 0.40 | 1.17 | 87.64 | -37.99 | 4467.37 | -9.90 | 8 | 0.92 |
| without theme | 827 | 41.96 | 0.44 | 1.19 | 69.43 | -40.00 | 4740.22 | -8.75 | 6 | 0.85 |

### Score scalings (same weights): raw float, integer -5..5, normalized 0..1

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| raw (threshold 1.00) | 1070 | 42.34 | 0.46 | 1.19 | 109.98 | -37.28 | 4424.88 | -8.86 | 8 | 1.05 |
| integer (threshold 1.00) | 1110 | 42.43 | 0.37 | 1.15 | 77.89 | -38.45 | 4392.51 | -10.43 | 8 | 0.87 |
| normalized (threshold 0.54) | 1070 | 42.34 | 0.46 | 1.19 | 109.98 | -37.28 | 4424.88 | -8.86 | 8 | 1.05 |

### Position sizing by score (score >= 1, top 5): 20% base, 25% boost

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| no boost | 1070 | 42.34 | 0.46 | 1.19 | 109.98 | -37.28 | 4424.88 | -8.86 | 8 | 1.05 |
| 25% when score >= 3 | 1070 | 42.34 | 0.46 | 1.19 | 137.41 | -40.95 | 4444.61 | -10.94 | 8 | 1.12 |
| 25% when score >= 4 | 1070 | 42.34 | 0.46 | 1.19 | 116.44 | -38.50 | 4420.53 | -10.05 | 8 | 1.06 |

### Weight tuning (random search, 643 valid draws of weights x threshold; train = first 60% of dev days, validation = last 40%)

Rank correlation between train and validation Sharpe across all draws: -0.19 (positive = tuning transfers, near zero = tuning is mostly noise). Top 15 draws by train Sharpe:

| weights | threshold | train_trades | train_sharpe | train_total% | train_maxDD% | val_trades | val_sharpe | val_total% | val_maxDD% | val_min$ |
|---|---|---|---|---|---|---|---|---|---|---|
| head=0.5 floa=1 tren=2 mark=0 news=2 move=0 edga=1 deli=1 late=2 dilu=1.5 them=2 | 3.00 | 613 | 2.84 | 217.63 | -14.46 | 454 | -0.32 | -22.61 | -47.02 | 2725.15 |
| head=1.5 floa=0.5 tren=2 mark=0.5 news=2 move=0 edga=1 deli=3 late=1.5 dilu=1 them=1 | 1.00 | 676 | 2.76 | 250.58 | -16.47 | 494 | -0.63 | -31.94 | -43.83 | 2812.96 |
| head=1.5 floa=0.5 tren=2 mark=1.5 news=1 move=0 edga=1.5 deli=1 late=1.5 dilu=0.5 them=2 | 3.00 | 509 | 2.75 | 214.85 | -16.20 | 383 | 0.17 | -3.47 | -38.61 | 3908.06 |
| head=2 floa=0.5 tren=2 mark=0 news=0 move=0.5 edga=1 deli=2 late=1.5 dilu=2 them=2 | 3.00 | 359 | 2.69 | 173.00 | -22.44 | 278 | -0.06 | -10.53 | -29.98 | 3536.92 |
| head=2 floa=0 tren=0.5 mark=0.5 news=0.5 move=0 edga=2 deli=1 late=1.5 dilu=1 them=1.5 | 0.00 | 702 | 2.69 | 272.86 | -24.48 | 505 | 0.06 | -9.78 | -38.37 | 3452.66 |
| head=1.5 floa=1.5 tren=1 mark=1 news=2 move=0.5 edga=1 deli=1 late=2 dilu=0.5 them=1.5 | 3.00 | 282 | 2.68 | 176.43 | -15.91 | 240 | 0.47 | 8.92 | -39.25 | 4226.69 |
| head=1.5 floa=0.5 tren=1.5 mark=0 news=1.5 move=0.5 edga=1 deli=3 late=1.5 dilu=0.5 them=1.5 | 2.00 | 478 | 2.68 | 204.91 | -22.49 | 369 | -0.59 | -29.22 | -41.46 | 2927.11 |
| head=1.5 floa=1 tren=1.5 mark=1.5 news=2 move=0 edga=0 deli=2 late=2 dilu=0.5 them=1.5 | 2.00 | 512 | 2.65 | 203.92 | -21.14 | 376 | 0.02 | -9.06 | -43.71 | 3955.40 |
| head=1.5 floa=0.5 tren=2 mark=1 news=0 move=1.5 edga=1 deli=2 late=1 dilu=0 them=2 | 2.00 | 653 | 2.64 | 246.95 | -24.96 | 478 | -0.24 | -20.68 | -38.15 | 3097.31 |
| head=2 floa=1 tren=0.5 mark=0.5 news=0 move=2 edga=2 deli=2 late=1 dilu=1.5 them=0 | 0.00 | 533 | 2.63 | 228.68 | -21.00 | 387 | -0.32 | -21.61 | -44.56 | 3547.66 |
| head=2 floa=2 tren=2 mark=0 news=2 move=1 edga=1.5 deli=2 late=2 dilu=0.5 them=0.5 | 2.00 | 393 | 2.63 | 235.53 | -19.70 | 309 | -0.33 | -21.74 | -40.87 | 3215.03 |
| head=0.5 floa=0 tren=1 mark=1 news=0 move=0.5 edga=0 deli=3 late=1.5 dilu=0 them=1.5 | 2.00 | 527 | 2.63 | 177.23 | -15.33 | 383 | 0.46 | 7.88 | -23.39 | 4458.51 |
| head=2 floa=1 tren=2 mark=0.5 news=2 move=1.5 edga=1 deli=2 late=1 dilu=0.5 them=2 | 3.00 | 473 | 2.60 | 208.02 | -26.15 | 362 | -0.25 | -18.98 | -39.34 | 3396.35 |
| head=2 floa=0.5 tren=1.5 mark=1 news=1 move=0.5 edga=0 deli=3 late=2 dilu=1 them=2 | 1.00 | 644 | 2.58 | 229.14 | -21.63 | 481 | -0.25 | -20.32 | -38.49 | 3460.50 |
| head=1.5 floa=1 tren=1 mark=0.5 news=1 move=2 edga=0.5 deli=2 late=1 dilu=0.5 them=0 | 0.00 | 623 | 2.57 | 237.64 | -23.63 | 439 | -0.63 | -32.02 | -48.53 | 3399.10 |

Selected (best validation Sharpe among the top 15 by train): weights {'headroom': 1.5, 'float': 1.5, 'trend200': 1.0, 'market': 1.0, 'news': 2.0, 'movers': 0.5, 'edgar_pos': 1.0, 'delist_notice': 1.0, 'late_filing': 2.0, 'dilution': 0.5, 'theme': 1.5}, threshold 3; validation Sharpe 0.47, total 9%, max DD -39%, 240 trades.

Walk-forward (nested) bias of this tuning procedure: +329.1 log-return points over 432 dev days (4 folds; in-sample 0.574%/day vs out-of-sample -0.188%/day).

### Learned weights: monthly walk-forward ridge on the score components (out-of-sample months only)

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| rel. volume rank, top 5 (same months) | 1174 | 40.03 | 0.09 | 1.03 | -1.38 | -41.74 | 3189.45 | -10.08 | 8 | 0.27 |
| default score >= 2 | 623 | 41.25 | 0.23 | 1.10 | 14.72 | -40.54 | 3899.07 | -8.75 | 8 | 0.45 |
| default score >= 3 | 301 | 37.54 | 0.39 | 1.15 | 17.49 | -20.10 | 4637.73 | -8.75 | 4 | 0.53 |
| walk-forward ridge, predicted return >= 0.0% | 895 | 42.12 | 0.34 | 1.13 | 46.40 | -34.24 | 3780.88 | -9.90 | 8 | 0.78 |
| walk-forward ridge, predicted return >= 0.5% | 614 | 42.18 | 0.37 | 1.12 | 29.56 | -36.58 | 3256.55 | -9.87 | 11 | 0.62 |
| walk-forward ridge, predicted return >= 1.0% | 403 | 43.42 | -0.14 | 0.96 | -19.11 | -43.41 | 3034.47 | -9.87 | 4 | -0.22 |
| walk-forward ridge, predicted return >= 2.0% | 155 | 38.06 | -0.45 | 0.90 | -17.74 | -31.88 | 3919.07 | -6.49 | 3 | -0.37 |

Ridge coefficients on all dev trades (expected trade return per unit of each component, winsorized at 30%): headroom +0.21%, float +0.51%, trend200 +0.27%, market +0.14%, news -0.15%, movers -0.20%, edgar_pos -0.40%, delist_notice -1.46%, late_filing +0.16%, dilution -1.46%, theme +0.83%

### Dev summary: previous top strategies vs ranked versions

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| base (as before: rel. volume rank, no hygiene) | 1547 | 40.59 | 0.32 | 1.14 | 97.66 | -44.23 | 4336.48 | -10.08 | 9 | 0.93 |
| S1 (as before: rel. volume rank, no hygiene) | 660 | 43.94 | 0.90 | 1.49 | 187.72 | -18.16 | 4921.56 | -7.69 | 3 | 1.71 |
| H1 (as before: rel. volume rank, no hygiene) | 480 | 42.08 | 1.19 | 1.43 | 162.14 | -31.93 | 4904.93 | -8.10 | 8 | 1.41 |
| H3 (as before: rel. volume rank, no hygiene) | 394 | 44.16 | 1.38 | 1.58 | 157.52 | -25.84 | 4911.98 | -7.58 | 7 | 1.49 |
| H2 (as before: rel. volume rank, no hygiene) | 204 | 43.14 | 1.20 | 1.57 | 54.99 | -21.44 | 4966.27 | -7.58 | 2 | 1.15 |
| base<=100M (as before: rel. volume rank, no hygiene) | 1095 | 41.92 | 0.49 | 1.19 | 117.58 | -44.89 | 4610.53 | -10.01 | 8 | 1.03 |
| S1<=100M (as before: rel. volume rank, no hygiene) | 424 | 46.23 | 1.28 | 1.70 | 168.15 | -14.74 | 4813.91 | -7.49 | 3 | 1.80 |
| H3<=100M (as before: rel. volume rank, no hygiene) | 199 | 43.22 | 2.39 | 1.89 | 131.30 | -17.10 | 5000.00 | -5.20 | 5 | 1.45 |
| F30-H20 (as before: rel. volume rank, no hygiene) | 91 | 43.96 | 5.63 | 2.21 | 148.33 | -14.93 | 5000.00 | -4.28 | 3 | 1.55 |
| F30-M200-H20 (as before: rel. volume rank, no hygiene) | 73 | 43.84 | 5.74 | 2.07 | 107.51 | -14.41 | 5000.00 | -3.96 | 3 | 1.30 |
| M200 (as before: rel. volume rank, no hygiene) | 513 | 44.64 | 1.55 | 1.41 | 269.68 | -33.09 | 5000.00 | -9.87 | 6 | 1.56 |
| M200-H20 (as before: rel. volume rank, no hygiene) | 132 | 43.94 | 3.02 | 1.62 | 95.06 | -28.57 | 5000.00 | -7.01 | 3 | 1.15 |
| base + hygiene, default-weight score >= 1 | 1070 | 42.34 | 0.46 | 1.19 | 109.98 | -37.28 | 4424.88 | -8.86 | 8 | 1.05 |
| H3 + hygiene, default-weight score >= 1 | 378 | 43.92 | 1.11 | 1.48 | 106.20 | -22.05 | 4911.98 | -7.58 | 7 | 1.27 |
| base + hygiene, tuned score | 522 | 41.76 | 1.20 | 1.46 | 193.77 | -39.25 | 4865.22 | -7.76 | 7 | 1.56 |
| H3 + hygiene, tuned score | 328 | 43.90 | 1.47 | 1.64 | 135.08 | -22.17 | 4985.68 | -7.58 | 7 | 1.49 |
| S1 + hygiene, tuned score | 322 | 44.10 | 1.60 | 1.77 | 156.78 | -16.50 | 4894.04 | -6.96 | 4 | 1.84 |
