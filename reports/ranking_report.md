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
| rank by score (default weights), top 5 | 1547 | 41.05 | 0.28 | 1.12 | 73.68 | -42.10 | 4626.71 | -10.05 | 9 | 0.82 |
| score >= 0, top 5 | 1329 | 41.46 | 0.37 | 1.16 | 99.88 | -40.52 | 4413.28 | -10.05 | 9 | 0.97 |
| score >= 1, top 5 | 1072 | 42.26 | 0.45 | 1.19 | 103.95 | -37.28 | 4424.88 | -8.86 | 8 | 1.02 |
| score >= 2, top 5 | 767 | 42.63 | 0.67 | 1.30 | 132.52 | -40.54 | 4875.91 | -8.75 | 8 | 1.24 |
| score >= 3, top 5 | 367 | 40.60 | 1.03 | 1.43 | 95.13 | -20.10 | 5000.00 | -8.75 | 4 | 1.33 |
| score >= 1, top 3 | 810 | 41.48 | 0.35 | 1.14 | 47.58 | -37.86 | 4854.19 | -6.94 | 9 | 0.71 |
| score >= 1, top 2 | 609 | 41.05 | 0.34 | 1.13 | 34.63 | -39.40 | 5000.00 | -6.53 | 11 | 0.63 |

### Existing top strategies with ranking (OTC excluded; EDGAR distress/dilution flags are score components)

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| base / rel_vol | 1547 | 40.66 | 0.35 | 1.15 | 116.55 | -41.74 | 4336.48 | -10.08 | 9 | 1.02 |
| base / score | 1547 | 41.05 | 0.28 | 1.12 | 73.68 | -42.10 | 4626.71 | -10.05 | 9 | 0.82 |
| base / score>=1 | 1072 | 42.26 | 0.45 | 1.19 | 103.95 | -37.28 | 4424.88 | -8.86 | 8 | 1.02 |
| base / score>=2 | 767 | 42.63 | 0.67 | 1.30 | 132.52 | -40.54 | 4875.91 | -8.75 | 8 | 1.24 |
| S1 / rel_vol | 660 | 43.94 | 0.92 | 1.51 | 195.50 | -18.16 | 4921.56 | -7.69 | 3 | 1.76 |
| S1 / score | 660 | 43.64 | 0.84 | 1.48 | 171.83 | -18.29 | 4904.59 | -8.75 | 3 | 1.74 |
| S1 / score>=1 | 640 | 43.44 | 0.83 | 1.47 | 157.53 | -18.46 | 4853.47 | -8.75 | 3 | 1.65 |
| S1 / score>=2 | 539 | 44.34 | 0.97 | 1.54 | 157.12 | -18.81 | 5000.00 | -8.75 | 3 | 1.73 |
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
| base<=100M / score>=1 | 760 | 42.24 | 0.51 | 1.19 | 76.98 | -39.90 | 4506.14 | -7.58 | 8 | 0.90 |
| base<=100M / score>=2 | 556 | 43.17 | 0.93 | 1.39 | 138.52 | -29.83 | 4893.16 | -6.94 | 7 | 1.31 |
| S1<=100M / rel_vol | 424 | 46.23 | 1.31 | 1.72 | 174.34 | -14.74 | 4813.91 | -7.49 | 3 | 1.85 |
| S1<=100M / score | 424 | 45.75 | 1.26 | 1.71 | 166.34 | -14.21 | 4794.73 | -6.94 | 3 | 1.86 |
| S1<=100M / score>=1 | 415 | 45.78 | 1.27 | 1.71 | 161.01 | -14.21 | 4794.73 | -6.94 | 3 | 1.82 |
| S1<=100M / score>=2 | 382 | 46.34 | 1.44 | 1.84 | 175.51 | -13.77 | 4925.72 | -6.94 | 3 | 1.95 |
| H3<=100M / rel_vol | 198 | 43.43 | 2.48 | 1.94 | 138.43 | -17.10 | 5000.00 | -5.20 | 5 | 1.50 |
| H3<=100M / score | 198 | 43.43 | 2.48 | 1.94 | 138.43 | -17.10 | 5000.00 | -5.20 | 5 | 1.50 |
| H3<=100M / score>=1 | 191 | 42.41 | 1.88 | 1.71 | 87.53 | -17.10 | 5000.00 | -5.20 | 5 | 1.23 |
| H3<=100M / score>=2 | 185 | 43.24 | 1.93 | 1.74 | 87.14 | -17.10 | 5000.00 | -5.20 | 5 | 1.23 |

### Component ablation (score >= 1, top 5): what does each component contribute?

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| all components | 1072 | 42.26 | 0.45 | 1.19 | 103.95 | -37.28 | 4424.88 | -8.86 | 8 | 1.02 |
| without headroom | 1230 | 41.46 | 0.30 | 1.13 | 63.37 | -38.55 | 4398.21 | -8.40 | 12 | 0.78 |
| without float | 1069 | 42.38 | 0.44 | 1.20 | 105.90 | -38.61 | 4444.18 | -8.61 | 10 | 1.07 |
| without trend200 | 928 | 40.52 | 0.39 | 1.17 | 71.62 | -40.36 | 4755.42 | -8.02 | 5 | 0.88 |
| without market | 1078 | 42.58 | 0.50 | 1.21 | 130.52 | -46.90 | 4544.21 | -10.05 | 16 | 1.14 |
| without news | 1058 | 41.87 | 0.39 | 1.16 | 79.30 | -37.28 | 4348.70 | -10.43 | 7 | 0.88 |
| without movers | 1056 | 42.42 | 0.44 | 1.19 | 100.16 | -37.14 | 4413.74 | -8.86 | 7 | 1.01 |
| without edgar_pos | 1052 | 42.11 | 0.47 | 1.20 | 111.17 | -36.96 | 4424.88 | -8.86 | 8 | 1.06 |
| without delist_notice | 1095 | 42.10 | 0.46 | 1.19 | 110.90 | -37.28 | 4523.17 | -8.86 | 8 | 1.04 |
| without late_filing | 1076 | 42.10 | 0.41 | 1.17 | 87.25 | -39.69 | 4424.88 | -8.86 | 8 | 0.92 |
| without dilution | 1095 | 42.10 | 0.40 | 1.17 | 85.54 | -37.99 | 4467.37 | -9.90 | 8 | 0.91 |
| without theme | 827 | 41.96 | 0.44 | 1.19 | 69.43 | -40.00 | 4740.22 | -8.75 | 6 | 0.85 |

### Score scalings (same weights): raw float, integer -5..5, normalized 0..1

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| raw (threshold 1.00) | 1072 | 42.26 | 0.45 | 1.19 | 103.95 | -37.28 | 4424.88 | -8.86 | 8 | 1.02 |
| integer (threshold 1.00) | 1111 | 42.39 | 0.37 | 1.15 | 75.90 | -38.45 | 4392.51 | -10.43 | 8 | 0.86 |
| normalized (threshold 0.54) | 1072 | 42.26 | 0.45 | 1.19 | 103.95 | -37.28 | 4424.88 | -8.86 | 8 | 1.02 |

### Position sizing by score (score >= 1, top 5): 20% base, 25% boost

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| no boost | 1072 | 42.26 | 0.45 | 1.19 | 103.95 | -37.28 | 4424.88 | -8.86 | 8 | 1.02 |
| 25% when score >= 3 | 1072 | 42.26 | 0.45 | 1.19 | 133.32 | -40.95 | 4444.61 | -10.94 | 8 | 1.10 |
| 25% when score >= 4 | 1072 | 42.26 | 0.45 | 1.19 | 110.23 | -38.50 | 4420.53 | -10.05 | 8 | 1.03 |

### Weight tuning (random search, 643 valid draws of weights x threshold; train = first 60% of dev days, validation = last 40%)

Rank correlation between train and validation Sharpe across all draws: -0.17 (positive = tuning transfers, near zero = tuning is mostly noise). Top 15 draws by train Sharpe:

| weights | threshold | train_trades | train_sharpe | train_total% | train_maxDD% | val_trades | val_sharpe | val_total% | val_maxDD% | val_min$ |
|---|---|---|---|---|---|---|---|---|---|---|
| head=0.5 floa=1 tren=2 mark=0 news=2 move=0 edga=1 deli=1 late=2 dilu=1.5 them=2 | 3.00 | 614 | 2.79 | 211.66 | -14.46 | 455 | -0.26 | -20.76 | -45.92 | 2781.74 |
| head=2 floa=0 tren=0.5 mark=0.5 news=0.5 move=0 edga=2 deli=1 late=1.5 dilu=1 them=1.5 | 0.00 | 702 | 2.68 | 271.53 | -24.48 | 506 | 0.07 | -9.28 | -38.37 | 3452.66 |
| head=2 floa=0.5 tren=2 mark=0 news=0 move=0.5 edga=1 deli=2 late=1.5 dilu=2 them=2 | 3.00 | 360 | 2.67 | 171.23 | -22.94 | 279 | -0.05 | -10.03 | -29.98 | 3536.92 |
| head=1.5 floa=0.5 tren=2 mark=0.5 news=2 move=0 edga=1 deli=3 late=1.5 dilu=1 them=1 | 1.00 | 678 | 2.66 | 234.95 | -18.84 | 494 | -0.63 | -31.94 | -43.83 | 2812.96 |
| head=1.5 floa=0.5 tren=1.5 mark=0 news=1.5 move=0.5 edga=1 deli=3 late=1.5 dilu=0.5 them=1.5 | 2.00 | 479 | 2.66 | 202.93 | -23.00 | 370 | -0.57 | -28.83 | -41.46 | 2927.11 |
| head=0.5 floa=0 tren=1 mark=1 news=0 move=0.5 edga=0 deli=3 late=1.5 dilu=0 them=1.5 | 2.00 | 528 | 2.63 | 180.02 | -15.33 | 384 | 0.53 | 10.12 | -23.39 | 4532.33 |
| head=2 floa=1 tren=0.5 mark=0.5 news=0 move=2 edga=2 deli=2 late=1 dilu=1.5 them=0 | 0.00 | 533 | 2.63 | 228.68 | -21.00 | 387 | -0.32 | -21.61 | -44.56 | 3547.66 |
| head=2 floa=2 tren=2 mark=0 news=2 move=1 edga=1.5 deli=2 late=2 dilu=0.5 them=0.5 | 2.00 | 393 | 2.63 | 235.53 | -19.70 | 309 | -0.33 | -21.74 | -40.87 | 3215.03 |
| head=1.5 floa=1.5 tren=1 mark=1 news=2 move=0.5 edga=1 deli=1 late=2 dilu=0.5 them=1.5 | 3.00 | 285 | 2.59 | 165.80 | -16.85 | 241 | 0.53 | 11.27 | -39.25 | 4226.69 |
| head=1.5 floa=1 tren=1.5 mark=1.5 news=2 move=0 edga=0 deli=2 late=2 dilu=0.5 them=1.5 | 2.00 | 514 | 2.59 | 195.18 | -23.40 | 376 | 0.02 | -9.06 | -43.71 | 3955.40 |
| head=1.5 floa=0.5 tren=2 mark=1.5 news=1 move=0 edga=1.5 deli=1 late=1.5 dilu=0.5 them=2 | 3.00 | 512 | 2.59 | 193.17 | -16.70 | 383 | 0.17 | -3.47 | -38.61 | 3908.06 |
| head=1.5 floa=1 tren=1 mark=0.5 news=1 move=2 edga=0.5 deli=2 late=1 dilu=0.5 them=0 | 0.00 | 623 | 2.57 | 237.64 | -23.63 | 439 | -0.63 | -32.02 | -48.53 | 3399.10 |
| head=2 floa=1 tren=2 mark=0.5 news=2 move=1.5 edga=1 deli=2 late=1 dilu=0.5 them=2 | 3.00 | 475 | 2.56 | 202.58 | -27.45 | 362 | -0.25 | -18.98 | -39.34 | 3396.35 |
| head=1.5 floa=0.5 tren=2 mark=1 news=0 move=1.5 edga=1 deli=2 late=1 dilu=0 them=2 | 2.00 | 654 | 2.54 | 229.20 | -26.03 | 479 | -0.23 | -20.24 | -38.15 | 3097.31 |
| head=2 floa=0.5 tren=1.5 mark=1 news=1 move=0.5 edga=0 deli=3 late=2 dilu=1 them=2 | 1.00 | 645 | 2.54 | 223.31 | -21.63 | 482 | -0.23 | -19.88 | -38.49 | 3460.50 |

Selected (best validation Sharpe among the top 15 by train): weights {'headroom': 0.5, 'float': 0.0, 'trend200': 1.0, 'market': 1.0, 'news': 0.0, 'movers': 0.5, 'edgar_pos': 0.0, 'delist_notice': 3.0, 'late_filing': 1.5, 'dilution': 0.0, 'theme': 1.5}, threshold 2; validation Sharpe 0.53, total 10%, max DD -23%, 384 trades.

### Learned weights: monthly walk-forward ridge on the score components (out-of-sample months only)

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| rel. volume rank, top 5 (same months) | 1174 | 40.03 | 0.09 | 1.03 | -1.38 | -41.74 | 3189.45 | -10.08 | 8 | 0.27 |
| default score >= 2 | 625 | 41.12 | 0.20 | 1.09 | 11.08 | -40.54 | 3775.22 | -8.75 | 8 | 0.40 |
| default score >= 3 | 304 | 37.83 | 0.47 | 1.18 | 23.19 | -20.10 | 4760.13 | -8.75 | 4 | 0.64 |
| walk-forward ridge, predicted return >= 0.0% | 896 | 42.08 | 0.29 | 1.11 | 33.77 | -38.69 | 3584.22 | -9.90 | 10 | 0.67 |
| walk-forward ridge, predicted return >= 0.5% | 619 | 42.49 | 0.47 | 1.16 | 46.22 | -35.32 | 3321.69 | -9.87 | 11 | 0.79 |
| walk-forward ridge, predicted return >= 1.0% | 407 | 43.00 | -0.15 | 0.95 | -20.23 | -43.54 | 2990.10 | -9.87 | 4 | -0.24 |
| walk-forward ridge, predicted return >= 2.0% | 154 | 38.31 | -0.42 | 0.91 | -16.88 | -31.17 | 3960.14 | -6.49 | 3 | -0.34 |

Ridge coefficients on all dev trades (expected trade return per unit of each component, winsorized at 30%): headroom +0.21%, float +0.51%, trend200 +0.27%, market +0.14%, news -0.15%, movers -0.20%, edgar_pos -0.40%, delist_notice -1.46%, late_filing +0.16%, dilution -1.45%, theme +0.87%

### Summary: previous top strategies vs ranked versions

Tuned weights come from the dev-only random search (results/ranking_weights.json); the test rows are out-of-sample.

#### Dev

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| base (rel. volume rank, no hygiene) | 1547 | 40.59 | 0.32 | 1.14 | 97.66 | -44.23 | 4336.48 | -10.08 | 9 | 0.93 |
| S1 (rel. volume rank, no hygiene) | 660 | 43.94 | 0.90 | 1.49 | 187.72 | -18.16 | 4921.56 | -7.69 | 3 | 1.71 |
| H3 (rel. volume rank, no hygiene) | 394 | 44.16 | 1.38 | 1.58 | 157.52 | -25.84 | 4911.98 | -7.58 | 7 | 1.49 |
| H2 (rel. volume rank, no hygiene) | 204 | 43.14 | 1.20 | 1.57 | 54.99 | -21.44 | 4966.27 | -7.58 | 2 | 1.15 |
| base<=100M (rel. volume rank, no hygiene) | 1095 | 41.92 | 0.49 | 1.19 | 117.58 | -44.89 | 4610.53 | -10.01 | 8 | 1.03 |
| S1<=100M (rel. volume rank, no hygiene) | 424 | 46.23 | 1.28 | 1.70 | 168.15 | -14.74 | 4813.91 | -7.49 | 3 | 1.80 |
| H3<=100M (rel. volume rank, no hygiene) | 199 | 43.22 | 2.39 | 1.89 | 131.30 | -17.10 | 5000.00 | -5.20 | 5 | 1.45 |
| base + hygiene, default-weight score >= 1 | 1072 | 42.26 | 0.45 | 1.19 | 103.95 | -37.28 | 4424.88 | -8.86 | 8 | 1.02 |
| H3 + hygiene, default-weight score >= 1 | 378 | 43.92 | 1.11 | 1.48 | 106.20 | -22.05 | 4911.98 | -7.58 | 7 | 1.27 |
| base + hygiene, tuned score | 912 | 44.41 | 0.67 | 1.34 | 191.34 | -25.25 | 4506.89 | -10.08 | 10 | 1.65 |
| H3 + hygiene, tuned score | 339 | 43.95 | 0.72 | 1.31 | 52.63 | -17.19 | 4903.04 | -7.58 | 7 | 1.03 |
| S1 + hygiene, tuned score | 613 | 44.21 | 1.01 | 1.61 | 209.91 | -17.05 | 5000.00 | -5.90 | 3 | 2.02 |

#### Locked test (last 4 months)

| setup | trades | win% | mean% | PF | total% | maxDD% | min$ | worst_day% | loss_streak_days | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| base (rel. volume rank, no hygiene) | 257 | 36.96 | -0.68 | 0.75 | -31.79 | -36.13 | 3213.08 | -4.34 | 7 | -2.39 |
| S1 (rel. volume rank, no hygiene) | 75 | 38.67 | 0.23 | 1.14 | 2.56 | -11.29 | 4927.77 | -2.70 | 2 | 0.42 |
| H3 (rel. volume rank, no hygiene) | 34 | 47.06 | 1.34 | 1.79 | 8.78 | -5.98 | 4914.48 | -2.13 | 2 | 1.26 |
| H2 (rel. volume rank, no hygiene) | 17 | 41.18 | 2.66 | 2.93 | 8.77 | -2.43 | 4949.90 | -1.27 | 1 | 1.33 |
| base<=100M (rel. volume rank, no hygiene) | 184 | 39.67 | -0.77 | 0.75 | -26.69 | -33.27 | 3411.64 | -4.53 | 9 | -2.11 |
| S1<=100M (rel. volume rank, no hygiene) | 44 | 45.45 | 1.36 | 1.88 | 11.75 | -6.27 | 5000.00 | -2.70 | 2 | 1.54 |
| H3<=100M (rel. volume rank, no hygiene) | 20 | 35.00 | 1.34 | 1.57 | 4.80 | -5.54 | 4919.93 | -2.13 | 2 | 0.76 |
| base + hygiene, default-weight score >= 1 | 135 | 40.74 | -0.07 | 0.96 | -3.26 | -18.49 | 4584.97 | -3.96 | 6 | -0.20 |
| H3 + hygiene, default-weight score >= 1 | 34 | 47.06 | 1.34 | 1.79 | 8.78 | -5.98 | 4914.48 | -2.13 | 2 | 1.26 |
| base + hygiene, tuned score | 109 | 36.70 | -0.13 | 0.93 | -3.89 | -13.52 | 4653.31 | -3.97 | 4 | -0.31 |
| H3 + hygiene, tuned score | 28 | 53.57 | 2.66 | 3.71 | 15.30 | -1.59 | 4973.42 | -1.27 | 2 | 2.14 |
| S1 + hygiene, tuned score | 70 | 35.71 | 0.13 | 1.07 | 0.85 | -12.74 | 4852.13 | -2.70 | 2 | 0.22 |
