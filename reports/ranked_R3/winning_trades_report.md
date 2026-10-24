# Winning-trade analysis: ranked strategy R3 (score >= 3, OTC excluded)

Generated from 396 de-duplicated trades, 2024-10-18 to 2026-10-09 (364 dev, 32 test), 301 symbols, 160 winners (40.4%), 34 big winners (>= +10%), 20 monsters (>= +25%). Returns are per trade after 0.1%/side commission and slippage (0.15% regular / 0.4% extended hours); entry = 15:55 ET, exit rules as in README.

## 1. How concentrated is the profit?

396 trades; 160 winners (40.4%); mean trade +0.95%; median -0.56%. Total of per-trade returns 3.8 (sum of fractions).

| top_k_trades | share_of_total_return% | mean_of_rest% | PF_of_rest |
|---|---|---|---|
| 1.00 | 22.48 | 0.74 | 1.30 |
| 3.00 | 53.47 | 0.44 | 1.18 |
| 5.00 | 75.59 | 0.23 | 1.09 |
| 10.00 | 114.55 | -0.14 | 0.94 |
| 20.00 | 182.83 | -0.83 | 0.68 |
| 50.00 | 289.67 | -2.06 | 0.27 |
| 100.00 | 344.75 | -3.11 | 0.06 |

Bootstrap 95% CI for the mean trade: [-0.08%, +1.98%]; Newey-West t-stat of the mean: 1.70. Kurtosis 14.6, skew 2.8 (heavy right tail).

![01_concentration](figures/01_concentration.png)

Top 20 trades:

| day | symbol | ret | sector | industry | sig_day_ret | gap_open | rel_vol | exit_reason | period |
|---|---|---|---|---|---|---|---|---|---|
| 2025-10-06 | BMEA | 84.4 | Healthcare | Biotechnology | 23.0 | 0.9 | 6.2 | ah_target | dev |
| 2026-03-02 | BATL | 59.5 | Energy | Oil & Gas - E&P | 116.1 | 93.8 | 6.3 | pm_target | dev |
| 2026-07-09 | JLHL | 56.9 | Industrials | Building Products & Equipment | 325.5 | 21.9 | 126.3 | ah_target | test |
| 2026-01-12 | AZIO | 49.4 | Technology | Software - Infrastructure | 392.3 | 148.3 | 20.9 | ah_target | dev |
| 2025-10-13 | GWH | 33.7 | Industrials | Industrial Products | 125.7 | 39.4 | 8.3 | pm_target | dev |
| 2024-11-25 | ZENA | 32.0 | Technology | Software - Infrastructure | 285.7 | 5.6 | 125.9 | pm_target | dev |
| 2026-10-02 | SDEV | 31.9 | Financial Services | Asset Management | 98.6 | 38.3 | 5.7 | pm_target | test |
| 2025-04-01 | HERE | 28.3 | Consumer Cyclical | Leisure | 59.2 | 8.5 | 9.9 | pm_target | dev |
| 2024-12-13 | WQEY | 27.2 | Technology | Software - Application | 141.9 | 0.6 | 117.8 | pm_target | dev |
| 2024-12-06 | BBAI | 26.8 | Technology | Application Software | 18.5 | 11.2 | 3.4 | pm_target | dev |
| 2024-12-26 | SRFM | 26.7 | Industrials | Airlines | 56.1 | 7.1 | 4.5 | pm_target | dev |
| 2025-07-25 | LIDR | 26.4 | Consumer Cyclical | Autos | 52.6 | 18.7 | 13.9 | pm_target | dev |
| 2024-12-16 | QUBT | 25.9 | Technology | Computer Hardware | 65.5 | 5.9 | 2.4 | pm_target | dev |
| 2025-10-13 | NVTS | 25.6 | Technology | Semiconductors | 21.4 | 4.7 | 3.0 | pm_target | dev |
| 2026-04-13 | RMSG | 25.5 | Technology | Software - Application | 481.5 | 101.6 | 21.7 | pm_target | dev |
| 2026-05-22 | CPSH | 25.4 | Technology | Computer Hardware | 28.1 | 2.1 | 5.9 | pm_target | dev |
| 2025-04-14 | TMC | 25.3 | Basic Materials | Metals & Mining | 46.2 | 23.1 | 7.4 | pm_target | dev |
| 2024-11-22 | PDYN | 25.3 | Technology | Software - Infrastructure | 123.0 | 11.0 | 103.7 | pm_target | dev |
| 2026-03-02 | TMDE | 25.2 | Energy | Oil & Gas Midstream | 228.6 | 224.2 | 55.8 | pm_target | dev |
| 2025-01-03 | ARBE | 25.2 | Technology | Application Software | 27.9 | 6.4 | 12.3 | pm_target | dev |

## 2. Timing: clusters, cycles and regimes

**Clustering of big winners (>= +10%) across weeks.** Dispersion statistic 114 vs 94 expected if big winners were spread evenly given each week's trade count (permutation p = 0.1024); Fano factor of weekly counts 1.01 (1 = Poisson). No significant time clustering of big winners.

Best weeks by summed trade return:

| week | n | wins | big | monster | total | win% |
|---|---|---|---|---|---|---|
| 2026-02-28 00:00:00 | 6 | 5 | 2 | 2 | 0.96 | 83.33 |
| 2025-10-04 00:00:00 | 9 | 6 | 1 | 1 | 0.94 | 66.67 |
| 2024-11-16 00:00:00 | 9 | 8 | 2 | 1 | 0.57 | 88.89 |
| 2026-07-04 00:00:00 | 2 | 1 | 1 | 1 | 0.55 | 50.00 |
| 2024-11-23 00:00:00 | 7 | 3 | 2 | 1 | 0.47 | 42.86 |
| 2026-04-11 00:00:00 | 10 | 6 | 2 | 1 | 0.41 | 60.00 |
| 2026-01-10 00:00:00 | 7 | 3 | 1 | 1 | 0.37 | 42.86 |
| 2025-06-21 00:00:00 | 5 | 3 | 2 | 0 | 0.35 | 60.00 |

The best 8 weeks (8% of 95 weeks) hold 51% of the positive weekly return.

**Regime change.** The most likely single change point in mean trade return is 2026-04-15 (max t = 2.00, permutation p = 0.322). Mean trade before: +1.57% (hit rate 43.1%), after: -0.87% (hit rate 32.7%).

Quarterly performance and market context (hit-rate 95% Wilson interval):

| day | n | win% | win%_lo | win%_hi | big% | mean% | IWM_qtr% | SPY_qtr% |
|---|---|---|---|---|---|---|---|---|
| 2024Q4 | 40 | 55.00 | 39.83 | 69.29 | 22.50 | 4.80 | 0.03 | 2.15 |
| 2025Q1 | 23 | 52.17 | 32.96 | 70.76 | 8.70 | 2.02 | -9.72 | -4.55 |
| 2025Q2 | 47 | 42.55 | 29.51 | 56.72 | 8.51 | 0.41 | 8.17 | 10.45 |
| 2025Q3 | 56 | 33.93 | 22.92 | 47.00 | 5.36 | -0.83 | 12.13 | 7.82 |
| 2025Q4 | 65 | 38.46 | 27.59 | 50.62 | 6.15 | 1.05 | 1.74 | 2.36 |
| 2026Q1 | 57 | 42.11 | 30.19 | 55.02 | 10.53 | 2.22 | 0.75 | -4.63 |
| 2026Q2 | 81 | 33.33 | 24.03 | 44.15 | 4.94 | -0.60 | 21.15 | 14.83 |
| 2026Q3 | 22 | 40.91 | 23.26 | 61.27 | 4.55 | -0.19 | -7.51 | 2.12 |
| 2026Q4 | 5 | 40.00 | 11.76 | 76.93 | 20.00 | 4.52 | -2.68 | 0.08 |

Monthly mean trade return vs IWM monthly return: Spearman -0.11 (p = 0.60); vs SPY -0.05 (p = 0.80), n = 25 months.

Autocorrelation of weekly mean trade return, lags 1-8: +0.01, -0.02, -0.10, -0.01, +0.11, +0.09, +0.06, -0.13 (permutation 95% band about +-0.23). None exceeds the band: no detectable week-to-week persistence.

Earnings seasons (approximate small-cap windows: Jan 20-Mar 15, Apr 20-May 20, Jul 20-Aug 20, Oct 20-Nov 20) vs other weeks:

| season | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| False | 220 | 44.55 | 12.27 | 1.96 | -0.50 | 1.74 | 4.31 | 114.87 |
| True | 176 | 35.23 | 3.98 | -0.32 | -0.72 | 0.86 | -0.56 | -14.87 |

Fisher exact test on big-winner share: odds ratio 0.30, p = 0.004.

![02_time](figures/02_time.png)

## 3. What moved them: earnings, news and themes

Share of trades with a headline between entry (15:55) and the next 10:00 ET, by type (Alpaca/Benzinga; mover-list and calendar items excluded from the category counts). Analysis only: this is information the strategy does not have at entry.

| group | n | any_news_after_entry% | company_specific_after% | mover_list_after% | earnings% | contract_partnership% | clinical_fda% | offering_dilution% | m_and_a% | analyst% | theme_hype% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all trades | 396 | 14.90 | 6.57 | 11.11 | 1.52 | 1.52 | 0.25 | 0.51 | 0.00 | 1.77 | 2.78 |
| winners | 160 | 16.88 | 7.50 | 12.50 | 1.25 | 1.88 | 0.00 | 0.00 | 0.00 | 1.88 | 3.12 |
| big winners (>=+10%) | 34 | 26.47 | 17.65 | 17.65 | 0.00 | 8.82 | 0.00 | 0.00 | 0.00 | 2.94 | 11.76 |
| monsters (>=+25%) | 20 | 30.00 | 20.00 | 25.00 | 0.00 | 15.00 | 0.00 | 0.00 | 0.00 | 0.00 | 15.00 |
| losers | 236 | 13.56 | 5.93 | 10.17 | 1.69 | 1.27 | 0.42 | 0.85 | 0.00 | 1.69 | 2.54 |
| worst 5% | 20 | 50.00 | 20.00 | 50.00 | 10.00 | 5.00 | 0.00 | 10.00 | 0.00 | 5.00 | 10.00 |

Does a post-entry event raise the odds of a big winner? (Fisher exact, BH-adjusted q):

| event | share_of_trades% | big_rate_with% | big_rate_without% | mean_ret_with% | odds_ratio | p | q_BH |
|---|---|---|---|---|---|---|---|
| earnings | 1.515 | 0.000 | 8.718 | -7.473 | 0.000 | 1.000 | 1.000 |
| contract_partnership | 1.515 | 50.000 | 7.949 | 13.666 | 11.581 | 0.010 | 0.034 |
| clinical_fda | 0.253 | 0.000 | 8.608 | -0.550 | 0.000 | 1.000 | 1.000 |
| offering_dilution | 0.505 | 0.000 | 8.629 | -20.731 | 0.000 | 1.000 | 1.000 |
| analyst | 1.768 | 14.286 | 8.483 | -0.781 | 1.798 | 0.469 | 0.821 |
| theme_hype | 2.778 | 36.364 | 7.792 | 6.101 | 6.762 | 0.010 | 0.034 |
| mover_list | 11.111 | 13.636 | 7.955 | 0.790 | 1.827 | 0.247 | 0.576 |

**Earnings filings (SimFin publish dates, 172 trades with data)**: a quarterly filing published on the entry day or within the next 3 days flags earnings-linked trades (the filing date can lag the press release).

| earn_next_0_3d | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 162.00 | 41.36 | 9.26 | 1.57 | -0.50 | 1.84 | 2.54 | 97.18 |
| 1.00 | 10.00 | 50.00 | 0.00 | 0.74 | -0.04 | 2.47 | 0.07 | 2.82 |

Big-winner odds ratio for earnings-linked entries 0.00 (Fisher p = 0.603). Share of big winners that are earnings-linked: 0.0% vs 5.8% of all trades.

Pre-entry news (known at 15:55):

| pre_news | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| company news in prior 24h | 64 | 31.25 | 7.81 | -0.82 | -0.76 | 0.77 | -0.53 | -14.04 |
| none | 332 | 42.17 | 8.73 | 1.29 | -0.50 | 1.57 | 4.28 | 114.04 |

Where the move happens (median return vs the 15:55 price at each checkpoint, by final outcome bucket):

| ret | n | median % at 16:45 | median % pre-market (last bar) | median % at 09:30 open |
|---|---|---|---|---|
| <-10% | 33 | -3.41 | -10.95 | -10.86 |
| -10..-3% | 66 | -0.37 | -2.68 | -4.49 |
| -3..0 | 137 | 0.00 | -0.03 | -0.72 |
| 0..3% | 71 | 0.21 | 1.55 | 1.40 |
| 3..10% | 55 | 0.76 | 4.91 | 5.81 |
| 10..25% | 14 | 0.51 | 14.90 | 16.84 |
| >25% | 20 | 2.69 | 20.70 | 20.75 |

![03_catalysts](figures/03_catalysts.png)

## 4. Sectors, industries and themes

By sector (SimFin classification, yfinance fallback; 'Unknown' = no classification, mostly delisted names):

| sector | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| Technology | 113 | 50.44 | 12.39 | 2.03 | 0.00 | 1.86 | 2.30 | 61.12 |
| Industrials | 58 | 39.66 | 12.07 | 1.88 | -0.52 | 1.75 | 1.09 | 29.09 |
| Healthcare | 104 | 38.46 | 4.81 | 0.55 | -0.77 | 1.27 | 0.57 | 15.28 |
| Consumer Cyclical | 31 | 38.71 | 9.68 | -0.07 | -0.76 | 0.98 | -0.02 | -0.54 |
| Basic Materials | 31 | 32.26 | 3.23 | -1.09 | -1.22 | 0.63 | -0.34 | -8.97 |
| Financial Services | 21 | 28.57 | 4.76 | -1.84 | -1.74 | 0.47 | -0.39 | -10.29 |

Over/under-representation of sectors among big winners (Fisher, BH-adjusted):

| sector | n | big_rate% | base_rate% | odds_ratio | p | q_BH |
|---|---|---|---|---|---|---|
| Technology | 113 | 12.389 | 8.586 | 1.860 | 0.111 | 0.456 |
| Healthcare | 104 | 4.808 | 8.586 | 0.458 | 0.152 | 0.456 |
| Industrials | 58 | 12.069 | 8.586 | 1.581 | 0.311 | 0.622 |
| Basic Materials | 31 | 3.226 | 8.586 | 0.335 | 0.500 | 0.749 |
| Consumer Cyclical | 31 | 9.677 | 8.586 | 1.154 | 0.740 | 0.888 |
| Financial Services | 21 | 4.762 | 8.586 | 0.518 | 1.000 | 1.000 |

Top industries by summed return (min 15 trades):

| industry | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| Computer Hardware | 25 | 68.00 | 16.00 | 3.72 | 2.77 | 3.26 | 0.93 | 23.39 |
| Biotechnology | 81 | 38.27 | 4.94 | 0.92 | -0.75 | 1.54 | 0.74 | 18.66 |
| Application Software | 25 | 52.00 | 8.00 | 1.50 | 0.08 | 1.72 | 0.37 | 9.43 |
| Industrial Products | 20 | 40.00 | 10.00 | 1.80 | -0.91 | 1.95 | 0.36 | 9.04 |
| Metals & Mining | 15 | 46.67 | 6.67 | 0.42 | -0.68 | 1.17 | 0.06 | 1.58 |
| Semiconductors | 19 | 36.84 | 5.26 | 0.11 | -1.44 | 1.04 | 0.02 | 0.54 |

Worst industries by summed return:

| industry | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| Computer Hardware | 25 | 68.00 | 16.00 | 3.72 | 2.77 | 3.26 | 0.93 | 23.39 |
| Biotechnology | 81 | 38.27 | 4.94 | 0.92 | -0.75 | 1.54 | 0.74 | 18.66 |
| Application Software | 25 | 52.00 | 8.00 | 1.50 | 0.08 | 1.72 | 0.37 | 9.43 |
| Industrial Products | 20 | 40.00 | 10.00 | 1.80 | -0.91 | 1.95 | 0.36 | 9.04 |
| Metals & Mining | 15 | 46.67 | 6.67 | 0.42 | -0.68 | 1.17 | 0.06 | 1.58 |
| Semiconductors | 19 | 36.84 | 5.26 | 0.11 | -1.44 | 1.04 | 0.02 | 0.54 |

**Same-day sector sympathy.** Pairs of big winners in the same sector on the same day: 1 observed vs 0.4 expected if sectors were assigned at random among big winners on each day (permutation p = 0.4016).

Strongest sector 'waves' (weeks with the most big winners in one sector):

| week | sector | n | tickers |
|---|---|---|---|
| 2025-06-21 | Industrials | 2 | SRFM, SRFM |
| 2026-04-11 | Technology | 2 | RMSG, XNDU |
| 2026-02-28 | Energy | 2 | BATL, TMDE |
| 2024-11-23 | Energy | 1 | OPTT |
| 2024-11-23 | Technology | 1 | ZENA |
| 2024-11-16 | Healthcare | 1 | ACRS |
| 2024-10-12 | Industrials | 1 | LTBR |
| 2024-12-07 | Technology | 1 | WQEY |
| 2024-12-14 | Technology | 1 | QUBT |
| 2024-12-28 | Technology | 1 | ARBE |
| 2024-12-21 | Industrials | 1 | SRFM |
| 2025-01-11 | Technology | 1 | BTCT |

![04_sector_heatmap](figures/04_sector_heatmap.png)

![04_sector_bars](figures/04_sector_bars.png)

## 4b. Themes found from company descriptions (TF-IDF + NMF topic model)

Each company's business description was converted to TF-IDF features and factorized into 14 topics; each company is assigned to its dominant topic (374 trades with a confident assignment). Topics ranked by summed trade return:

| topic_label | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| T13: systems, solutions, equipment, segment, industrial | 81 | 45.68 | 12.35 | 2.11 | -0.33 | 1.89 | 1.71 | 41.08 |
| T12: ai, data, cloud, software, solutions | 62 | 45.16 | 12.90 | 2.18 | -0.50 | 1.84 | 1.35 | 32.37 |
| T1: treatment, clinical, phase, therapeutics, trial | 82 | 42.68 | 6.10 | 1.02 | -0.67 | 1.52 | 0.84 | 20.12 |
| T4: gold, silver, located, canada, exploration | 31 | 41.94 | 6.45 | 0.96 | -0.63 | 1.39 | 0.30 | 7.17 |
| T9: energy, power, solar, storage, energy storage | 34 | 41.18 | 8.82 | 0.70 | -0.64 | 1.26 | 0.24 | 5.67 |

Topic enrichment among big winners (Fisher, BH-adjusted):

| topic | n | big_rate% | odds_ratio | p | q_BH |
|---|---|---|---|---|---|
| T12: ai, data, cloud, software, solutions | 62 | 12.903 | 1.701 | 0.222 | 0.643 |
| T13: systems, solutions, equipment, segment, industrial | 81 | 12.346 | 1.653 | 0.266 | 0.643 |
| T1: treatment, clinical, phase, therapeutics, trial | 82 | 6.098 | 0.612 | 0.386 | 0.643 |
| T4: gold, silver, located, canada, exploration | 31 | 6.452 | 0.694 | 1.000 | 1.000 |
| T9: energy, power, solar, storage, energy storage | 34 | 8.824 | 1.000 | 1.000 | 1.000 |

Strongest theme waves (weeks with the most big winners sharing a topic):

| week | topic_label | n | tickers |
|---|---|---|---|
| 2025-06-21 | T13: systems, solutions, equipment, segment, industrial | 2 | SRFM, SRFM |
| 2026-02-28 | T3: oil, gas, natural gas, natural, crude | 2 | BATL, TMDE |
| 2024-11-16 | T12: ai, data, cloud, software, solutions | 1 | PDYN |
| 2024-11-23 | T12: ai, data, cloud, software, solutions | 1 | ZENA |
| 2024-11-23 | T13: systems, solutions, equipment, segment, industrial | 1 | OPTT |
| 2024-11-30 | T12: ai, data, cloud, software, solutions | 1 | BBAI |
| 2024-10-12 | T9: energy, power, solar, storage, energy storage | 1 | LTBR |
| 2024-12-07 | T12: ai, data, cloud, software, solutions | 1 | WQEY |
| 2024-12-14 | T12: ai, data, cloud, software, solutions | 1 | QUBT |
| 2024-12-28 | T13: systems, solutions, equipment, segment, industrial | 1 | ARBE |

![04b_topic_heatmap](figures/04b_topic_heatmap.png)

## 5. Float, size, liquidity and volatility

Rank correlation of each feature with trade return and with being a big winner (BH-adjusted q on the return test):

| feature | n | spearman_vs_ret | p_ret | spearman_vs_big | p_big | q_BH(ret) |
|---|---|---|---|---|---|---|
| shares_pit | 271 | -0.044 | 0.469 | -0.186 | 0.002 | 0.791 |
| mcap_pit | 271 | -0.068 | 0.265 | -0.230 | 0.000 | 0.623 |
| float_now | 371 | 0.063 | 0.223 | -0.089 | 0.087 | 0.623 |
| short_pct_float_now | 366 | 0.136 | 0.009 | 0.019 | 0.723 | 0.125 |
| price | 396 | -0.021 | 0.680 | -0.126 | 0.012 | 0.791 |
| dvol20 | 396 | 0.017 | 0.734 | -0.055 | 0.274 | 0.791 |
| rel_vol | 396 | -0.056 | 0.267 | 0.238 | 0.000 | 0.623 |
| atr_pct | 396 | 0.021 | 0.672 | 0.332 | 0.000 | 0.791 |
| sig_day_ret | 396 | -0.030 | 0.557 | 0.338 | 0.000 | 0.791 |
| gap_open | 396 | -0.064 | 0.204 | 0.148 | 0.003 | 0.623 |
| range_pct | 396 | -0.017 | 0.732 | 0.367 | 0.000 | 0.791 |
| run5 | 396 | -0.084 | 0.095 | 0.126 | 0.012 | 0.623 |
| headroom | 396 | -0.007 | 0.886 | -0.031 | 0.537 | 0.886 |
| dist_sma200 | 381 | 0.049 | 0.338 | 0.177 | 0.001 | 0.676 |

**shares_pit** quintiles (shares outstanding (SimFin, point-in-time; float proxy)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (527096.999, 26525068.0] | 56 | 51.79 | 21.43 | 4.28 | 0.27 | 2.60 |
| Q2: (26525068.0, 53785369.0] | 53 | 26.42 | 0.00 | -2.17 | -1.77 | 0.28 |
| Q3: (53785369.0, 85484525.0] | 54 | 50.00 | 9.26 | 2.70 | -0.00 | 2.67 |
| Q4: (85484525.0, 132263584.0] | 54 | 35.19 | 1.85 | -0.22 | -0.62 | 0.85 |
| Q5: (132263584.0, 574592599.0] | 54 | 37.04 | 3.70 | 0.25 | -0.73 | 1.13 |

**mcap_pit** quintiles (market cap (SimFin, point-in-time)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (3531549.8989999997, 124673754.9] | 55 | 54.55 | 21.82 | 4.38 | 0.75 | 2.58 |
| Q2: (124673754.9, 240180908.44] | 54 | 44.44 | 9.26 | 1.89 | -0.46 | 1.82 |
| Q3: (240180908.44, 541497414.84] | 54 | 35.19 | 0.00 | -1.01 | -0.72 | 0.54 |
| Q4: (541497414.84, 926624338.38] | 54 | 24.07 | 0.00 | -1.35 | -0.94 | 0.29 |
| Q5: (926624338.38, 1946667522.96] | 54 | 42.59 | 5.56 | 1.05 | -0.56 | 1.71 |

**float_now** quintiles (float (yfinance current snapshot, not point-in-time)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (-0.001, 23487775.0] | 75 | 41.33 | 14.67 | 0.78 | -0.87 | 1.18 |
| Q2: (23487775.0, 43818337.0] | 75 | 40.00 | 9.33 | 0.94 | -0.55 | 1.38 |
| Q3: (43818337.0, 77447074.0] | 73 | 39.73 | 2.74 | 1.17 | -0.64 | 1.69 |
| Q4: (77447074.0, 125521574.0] | 74 | 47.30 | 12.16 | 2.14 | -0.30 | 2.23 |
| Q5: (125521574.0, 7475818960.0] | 74 | 40.54 | 5.41 | 0.54 | -0.73 | 1.26 |

**short_pct_float_now** quintiles (short % of float (current snapshot)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (-0.0004000000000000001, 0.0479] | 74 | 33.78 | 5.41 | -1.46 | -0.93 | 0.62 |
| Q2: (0.0479, 0.083] | 73 | 38.36 | 9.59 | 1.30 | -0.82 | 1.54 |
| Q3: (0.083, 0.127] | 75 | 41.33 | 13.33 | 2.35 | -0.50 | 2.26 |
| Q4: (0.127, 0.183] | 72 | 47.22 | 8.33 | 1.60 | -0.26 | 1.79 |
| Q5: (0.183, 0.745] | 72 | 50.00 | 6.94 | 1.80 | -0.02 | 1.84 |

**price** quintiles (price at 15:55):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (2.011, 3.71] | 80 | 48.75 | 16.25 | 2.86 | -0.26 | 2.05 |
| Q2: (3.71, 5.92] | 79 | 35.44 | 6.33 | -0.14 | -1.09 | 0.95 |
| Q3: (5.92, 8.685] | 79 | 43.04 | 10.13 | 0.64 | -0.33 | 1.20 |
| Q4: (8.685, 12.741] | 79 | 39.24 | 6.33 | 1.43 | -0.55 | 1.85 |
| Q5: (12.741, 19.85] | 79 | 35.44 | 3.80 | -0.08 | -0.58 | 0.96 |

**rel_vol** quintiles (relative volume at 15:55):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (2.001, 2.375] | 80 | 40.00 | 3.75 | 0.60 | -0.51 | 1.46 |
| Q2: (2.375, 3.013] | 79 | 35.44 | 2.53 | -0.34 | -0.50 | 0.81 |
| Q3: (3.013, 4.48] | 79 | 46.84 | 5.06 | 0.46 | -0.43 | 1.23 |
| Q4: (4.48, 7.858] | 79 | 40.51 | 10.13 | 1.57 | -1.62 | 1.55 |
| Q5: (7.858, 268.884] | 79 | 39.24 | 21.52 | 2.46 | -0.58 | 1.56 |

**atr_pct** quintiles (daily ATR % of price):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (0.00401, 0.0507] | 80 | 30.00 | 0.00 | -0.70 | -0.57 | 0.38 |
| Q2: (0.0507, 0.0668] | 79 | 41.77 | 1.27 | 0.67 | -0.33 | 1.43 |
| Q3: (0.0668, 0.0841] | 79 | 36.71 | 2.53 | -0.78 | -1.34 | 0.72 |
| Q4: (0.0841, 0.109] | 79 | 45.57 | 15.19 | 1.63 | -0.63 | 1.53 |
| Q5: (0.109, 2.229] | 79 | 48.10 | 24.05 | 3.93 | -0.50 | 2.01 |

**sig_day_ret** quintiles (signal-day return to 15:55):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (0.0091, 0.117] | 80 | 30.00 | 0.00 | -0.83 | -0.52 | 0.42 |
| Q2: (0.117, 0.181] | 79 | 43.04 | 0.00 | -0.01 | -0.53 | 0.99 |
| Q3: (0.181, 0.266] | 79 | 40.51 | 3.80 | 0.98 | -0.63 | 1.49 |
| Q4: (0.266, 0.421] | 79 | 43.04 | 11.39 | 0.38 | -1.62 | 1.12 |
| Q5: (0.421, 5.334] | 79 | 45.57 | 27.85 | 4.25 | -0.66 | 1.91 |

**dvol20** quintiles (20-day avg dollar volume):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (743013.7089999999, 2888379.648] | 80 | 48.75 | 18.75 | 3.88 | -0.26 | 2.56 |
| Q2: (2888379.648, 6069425.979] | 79 | 32.91 | 2.53 | -1.00 | -1.32 | 0.64 |
| Q3: (6069425.979, 10086209.546] | 79 | 32.91 | 3.80 | -0.87 | -0.55 | 0.65 |
| Q4: (10086209.546, 20073020.296] | 79 | 43.04 | 5.06 | 0.12 | -0.56 | 1.05 |
| Q5: (20073020.296, 373337124.542] | 79 | 44.30 | 12.66 | 2.57 | -0.50 | 2.23 |

**headroom** quintiles (headroom to next resistance (13m)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (0.00082, 0.111] | 80 | 47.50 | 8.75 | 1.26 | -0.59 | 1.48 |
| Q2: (0.111, 1.021] | 79 | 40.51 | 12.66 | 2.58 | -0.75 | 2.05 |
| Q3: (1.021, 2.0] | 237 | 37.97 | 7.17 | 0.30 | -0.55 | 1.12 |

![05_quintiles](figures/05_quintiles.png)

By shares outstanding (SimFin, millions):

| size | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| <10M sh | 13 | 46.15 | 7.69 | -1.16 | -0.64 | 0.78 | -0.15 | -5.52 |
| 10-30M | 63 | 49.21 | 17.46 | 3.69 | -0.04 | 2.71 | 2.32 | 85.21 |
| 30-100M | 113 | 36.28 | 5.31 | 0.62 | -0.65 | 1.33 | 0.71 | 25.89 |
| 100-300M | 70 | 40.00 | 1.43 | -0.22 | -0.73 | 0.88 | -0.15 | -5.68 |
| >300M | 12 | 25.00 | 8.33 | 0.02 | -1.71 | 1.01 | 0.00 | 0.10 |

## 6. Link to the broader market

Daily mean trade return vs market variables (218 trading days; '*_gap' = next-session close-to-open move of the index, known only afterwards; '*_day' = the index move from the prior close to 15:55 on entry day):

| market variable | spearman_vs_daily_mean_ret | p |
|---|---|---|
| spy_gap | 0.165 | 0.015 |
| iwm_gap | 0.158 | 0.020 |
| qqq_gap | 0.194 | 0.004 |
| spy_day | -0.007 | 0.916 |
| iwm_day | 0.089 | 0.191 |
| iwm_ret5 | -0.111 | 0.102 |
| iwm_ret20 | -0.017 | 0.800 |
| spy_ret20 | 0.013 | 0.844 |

**Overnight beta.** OLS with Newey-West errors, daily mean trade return on IWM and SPY overnight gaps: intercept (alpha) +0.62% per day (t = 1.07); beta to IWM gap 0.12 (t = 0.10); beta to SPY gap 0.24 (t = 0.12); R2 = 0.001.

Trade outcomes by tercile of market variable:

| variable | b | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|---|
| IWM next-session gap | low | 134 | 29.10 | 8.96 | -0.18 | -1.30 | 0.94 | -0.25 | -6.55 |
| IWM next-session gap | mid | 131 | 41.22 | 8.40 | 1.70 | -0.55 | 1.84 | 2.23 | 59.39 |
| IWM next-session gap | high | 131 | 51.15 | 8.40 | 1.35 | 0.12 | 1.64 | 1.77 | 47.16 |
| IWM entry-day move | low | 133 | 39.85 | 6.77 | 0.54 | -0.55 | 1.22 | 0.72 | 19.11 |
| IWM entry-day move | mid | 131 | 39.69 | 8.40 | 0.53 | -0.72 | 1.19 | 0.69 | 18.37 |
| IWM entry-day move | high | 132 | 41.67 | 10.61 | 1.78 | -0.50 | 1.84 | 2.35 | 62.52 |
| SPY entry-day move | low | 133 | 39.10 | 9.77 | 0.95 | -0.56 | 1.33 | 1.27 | 33.79 |
| SPY entry-day move | mid | 133 | 47.37 | 7.52 | 1.28 | -0.24 | 1.59 | 1.70 | 45.18 |
| SPY entry-day move | high | 130 | 34.62 | 8.46 | 0.61 | -0.89 | 1.26 | 0.79 | 21.03 |
| IWM 20-day trend | low | 132 | 39.39 | 11.36 | 1.67 | -0.57 | 1.72 | 2.21 | 58.78 |
| IWM 20-day trend | mid | 132 | 47.73 | 8.33 | 1.72 | -0.39 | 1.76 | 2.27 | 60.51 |
| IWM 20-day trend | high | 132 | 34.09 | 6.06 | -0.55 | -0.86 | 0.81 | -0.72 | -19.29 |

**Breadth ('hype') indicator.** The number of candidate breakouts that day relative to its prior 20-day average: Spearman vs that day's mean trade return +0.080 (p = 0.243); vs number of big winners -0.031.

![06_market](figures/06_market.png)

### 6b. How much of the result is just the market's overnight move?

| IWM hedge ratio | mean% | median% | win% | PF | daily_Sharpe_ann | NW_t_daily |
|---|---|---|---|---|---|---|
| 0.00 | 0.95 | -0.56 | 40.40 | 1.38 | 1.14 | 1.11 |
| 0.50 | 0.93 | -0.57 | 38.89 | 1.38 | 1.10 | 1.07 |
| 1.00 | 0.91 | -0.66 | 39.90 | 1.37 | 1.06 | 1.02 |

A hedge ratio of 1 removes the IWM close-to-open move entirely. The trades hold some positions through the open only partly (several exit pre-market or after-hours), so this is an approximation; it shows whether the edge survives once index direction is removed.

## 7. Can the winners be predicted? (gradient boosting, purged walk-forward)

HistGradientBoosting (depth 3, strong regularization) on 32 point-in-time features known at 15:55; expanding monthly folds with a 5-business-day purge before each test month; final model trained on all dev and scored on the locked test period. AUC 0.5 = no skill.

| target | dev_oos_AUC | dev_oos_PR_AUC | base_rate% | dev_oos_top10%_lift | test_AUC | test_PR_AUC | test_base_rate% | test_top20%_lift | n_dev_oos | n_test |
|---|---|---|---|---|---|---|---|---|---|---|
| big | 0.769 | 0.208 | 6.977 | 3.344 | 0.883 | 0.292 | 6.250 | 5.333 | 301 | 32 |
| winner | 0.478 | 0.397 | 37.542 | 1.154 | 0.474 | 0.429 | 40.625 | 0.821 | 301 | 32 |

Permutation importance (drop in AUC when a feature is shuffled inside each walk-forward test fold; big-winner target):

| index | mean_AUC_drop | sd_across_folds | folds_positive% |
|---|---|---|---|
| range_pct | 0.0273 | 0.0254 | 60.0000 |
| atr_pct | 0.0267 | 0.0409 | 60.0000 |
| dist_ema9 | 0.0125 | 0.0528 | 40.0000 |
| dvol20 | 0.0113 | 0.0204 | 30.0000 |
| from_high | 0.0082 | 0.0255 | 30.0000 |
| price | 0.0077 | 0.0227 | 30.0000 |
| day_dvol | 0.0055 | 0.0203 | 40.0000 |
| dist_vwap | 0.0050 | 0.0091 | 30.0000 |
| rel_vol | 0.0046 | 0.0206 | 40.0000 |
| spy_ret20 | 0.0046 | 0.0091 | 20.0000 |
| dist_sma200 | 0.0039 | 0.0189 | 20.0000 |
| iwm_day | 0.0035 | 0.0069 | 30.0000 |
| news_5d | 0.0027 | 0.0081 | 10.0000 |
| spy_day | 0.0027 | 0.0081 | 10.0000 |
| iwm_ret20 | 0.0025 | 0.0060 | 20.0000 |

**Remove-and-retrain check** (SHAP and permutation importance describe what a model fitted, not what generalizes): walk-forward AUC with all features 0.769; without the top 3 (range_pct, atr_pct, dist_ema9) 0.749; with only those 3 0.866.

![07_partial_dependence](figures/07_partial_dependence.png)

A depth-3 decision tree on dev (readable splits; unstable, shown for interpretation only):

```
|--- range_pct <= 0.425
|   |--- from_high <= -0.040
|   |   |--- class: False
|   |--- from_high >  -0.040
|   |   |--- dist_ema20 <= 0.328
|   |   |   |--- class: False
|   |   |--- dist_ema20 >  0.328
|   |   |   |--- class: False
|--- range_pct >  0.425
|   |--- class: False
```

## 8. Rule mining with multiple-testing control

6740 candidate rules (single and pairs of quartile/median conditions) with at least 60 dev trades: 2631 tested for a higher big-winner rate than the 8.8% base (one-sided binomial). 514 pass BH-FDR q < 0.10 on dev.

Strongest rules by dev p-value, with their out-of-sample behaviour on the locked test period (test base big-winner rate 6.2%):

| rule | n | big_rate% | big_loss_rate% | lift | mean_ret% | PF | p | q_BH | test_n | test_big_rate% | test_mean_ret% | test_big_loss% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| atr_pct >= 0.103 AND dist_ema9 >= 0.223 | 69 | 31.9 | 20.3 | 3.63 | 4.91 | 2.15 | 5.68e-08 | 8.85e-05 | 6 | 33.3 | 7.08 | 50 |
| sig_day_ret >= 0.218 AND atr_pct >= 0.103 | 71 | 31 | 18.3 | 3.52 | 4.87 | 2.2 | 1.01e-07 | 8.85e-05 | 6 | 33.3 | 7.08 | 50 |
| atr_pct >= 0.0776 AND dist_ema9 >= 0.365 | 71 | 31 | 26.8 | 3.52 | 4.08 | 1.78 | 1.01e-07 | 8.85e-05 | 7 | 14.3 | -3.86 | 71.4 |
| sig_day_ret >= 0.367 AND range_pct >= 0.333 | 63 | 31.7 | 25.4 | 3.61 | 4.54 | 1.9 | 2.43e-07 | 0.00016 | 7 | 28.6 | 5.84 | 57.1 |
| range_pct >= 0.202 AND dist_ema9 >= 0.365 | 81 | 28.4 | 24.7 | 3.23 | 3.61 | 1.72 | 3.07e-07 | 0.000162 | 9 | 11.1 | -3.89 | 55.6 |
| atr_pct >= 0.103 AND dist_ema20 >= 0.302 | 72 | 29.2 | 19.4 | 3.32 | 4.22 | 2.02 | 6.06e-07 | 0.000221 | 5 | 20 | -2.88 | 60 |
| sig_day_ret >= 0.367 AND atr_pct >= 0.0776 | 72 | 29.2 | 25 | 3.32 | 3.37 | 1.63 | 6.06e-07 | 0.000221 | 8 | 25 | 4.42 | 50 |
| sig_day_ret >= 0.218 AND dist_ema9 >= 0.365 | 85 | 27.1 | 22.4 | 3.08 | 3.48 | 1.75 | 7.88e-07 | 0.000221 | 9 | 11.1 | -3.89 | 55.6 |
| atr_pct >= 0.0776 AND range_pct >= 0.333 | 79 | 27.8 | 22.8 | 3.17 | 3.97 | 1.87 | 7.95e-07 | 0.000221 | 8 | 25 | 3.73 | 62.5 |
| sig_day_ret >= 0.367 AND dist_ema9 >= 0.365 | 68 | 29.4 | 23.5 | 3.35 | 4.2 | 1.86 | 9.62e-07 | 0.000221 | 7 | 14.3 | -2.66 | 57.1 |
| rel_vol >= 3.47 AND range_pct >= 0.333 | 74 | 28.4 | 20.3 | 3.23 | 5.06 | 2.15 | 9.99e-07 | 0.000221 | 7 | 28.6 | 5.84 | 57.1 |
| sig_day_ret >= 0.367 AND range_pct >= 0.202 | 80 | 27.5 | 23.8 | 3.13 | 3.24 | 1.63 | 1.01e-06 | 0.000221 | 9 | 22.2 | 3.64 | 44.4 |
| atr_pct >= 0.103 AND range_pct >= 0.202 | 81 | 27.2 | 18.5 | 3.09 | 3.9 | 1.98 | 1.27e-06 | 0.000221 | 6 | 33.3 | 7.08 | 50 |
| sig_day_ret >= 0.218 AND range_pct >= 0.333 | 88 | 26.1 | 19.3 | 2.97 | 4.41 | 2.03 | 1.53e-06 | 0.000221 | 8 | 25 | 3.73 | 62.5 |
| range_pct >= 0.333 AND day_dvol >= 4.8e+07 | 70 | 28.6 | 21.4 | 3.25 | 4.36 | 2 | 1.6e-06 | 0.000221 | 8 | 25 | 3.73 | 62.5 |

## 9. Winner archetypes (clustering)

K-means on 34 big winners (>= +10%) using 10 standardized features; k = 2 chosen by silhouette (0.28).

| cluster | n | mean_ret% | sig_day_ret% | gap_open% | rel_vol | atr% | range% | price | dvol20_$M | run5% | headroom | top_sectors | examples |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 9 | 30.96 | 228.55 | 21.90 | 103.69 | 21.96 | 146.51 | 4.46 | 1.66 | 9.29 | 0.51 | Technology (6), Industrials (2) | JLHL, AZIO, ZENA, WQEY, RMSG |
| 1 | 25 | 27.14 | 46.20 | 8.46 | 5.89 | 10.80 | 46.90 | 5.48 | 10.17 | 26.37 | 2.00 | Technology (8), Industrials (5) | BMEA, BATL, GWH, SDEV, HERE |

Share of ALL trades falling in each archetype's region of feature space (nearest centroid) and how those trades fared:

| arch | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 21.00 | 42.86 | 42.86 | 7.21 | -0.75 | 2.19 | 1.51 | 40.30 |
| 1.00 | 375.00 | 40.27 | 6.67 | 0.60 | -0.55 | 1.26 | 2.24 | 59.70 |

## 10. Event study: the path around entry

Mean cumulative return versus IWM (%, winsorized 1%) relative to the close before the entry day, by final trade outcome. Day 0 is the entry day (entry at 15:55, so day 0 shows the run-up the strategy buys into), day +1 the overnight session:

| bucket | n | d-5 | d-4 | d-3 | d-2 | d-1 | d0 | d1 | d2 | d3 |
|---|---|---|---|---|---|---|---|---|---|---|
| loser <-5% | 66 | -13.07 | -9.39 | -8.73 | -7.71 | 0.00 | 73.39 | 53.47 | 45.26 | 45.25 |
| small loss | 169 | -4.93 | -4.43 | -3.43 | -2.12 | 0.00 | 21.22 | 19.59 | 18.63 | 17.81 |
| winner 0-10% | 126 | -4.70 | -5.20 | -4.63 | -2.18 | 0.00 | 22.44 | 25.81 | 26.43 | 28.03 |
| big winner >=10% | 34 | -14.12 | -15.48 | -15.46 | -8.87 | 0.00 | 100.03 | 133.25 | 108.50 | 106.11 |

![10_event_study](figures/10_event_study.png)

## 11. Statistical rigor: is any of this skill?

Probabilistic / deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) on dev daily portfolio returns. 'N' is the number of strategy variants tried; N=1 is the plain probabilistic Sharpe ratio, larger N deflates for selection bias. This project has tried on the order of 100 variants (exit rules, filters, sizing).

| strategy | days | daily_SR | annual_SR | skew | kurtosis | NW_t(mean) | P(skill) N=1 | P(skill) N=30 | P(skill) N=100 | P(skill) N=300 |
|---|---|---|---|---|---|---|---|---|---|---|
| this strategy | 432 | 0.079 | 1.247 | 3.614 | 29.661 | 1.615 | 0.969 | 0.420 | 0.255 | 0.153 |

## 12. Do the main findings hold out of sample (dev vs the locked test period)?

| group | period | n | win% | big% | big_loss% | mean% | PF |
|---|---|---|---|---|---|---|---|
| all trades | dev | 364 | 40.38 | 8.79 | 7.69 | 0.98 | 1.41 |
| all trades | test | 32 | 40.62 | 6.25 | 15.62 | 0.61 | 1.19 |
| shares outstanding < 30M | dev | 71 | 49.30 | 16.90 | 9.86 | 3.02 | 2.08 |
| shares outstanding < 30M | test | 5 | 40.00 | 0.00 | 0.00 | 0.57 | 1.44 |
| market cap < $200M | dev | 84 | 52.38 | 16.67 | 10.71 | 3.75 | 2.45 |
| market cap < $200M | test | 8 | 37.50 | 0.00 | 0.00 | -0.69 | 0.59 |
| daily ATR >= 9% | dev | 127 | 48.03 | 21.26 | 14.96 | 2.99 | 1.88 |
| daily ATR >= 9% | test | 9 | 44.44 | 22.22 | 44.44 | 4.93 | 1.78 |
| signal-day move >= +26% | dev | 152 | 43.42 | 19.08 | 14.47 | 2.11 | 1.57 |
| signal-day move >= +26% | test | 13 | 46.15 | 15.38 | 38.46 | 2.67 | 1.48 |
| rel. volume >= 5.4 | dev | 116 | 38.79 | 18.10 | 16.38 | 2.34 | 1.59 |
| rel. volume >= 5.4 | test | 13 | 46.15 | 15.38 | 30.77 | 3.17 | 1.63 |
| price below 200 SMA | dev | 5 | 20.00 | 0.00 | 40.00 | -6.85 | 0.08 |
| price below 200 SMA | test | 1 | 0.00 | 0.00 | 100.00 | -13.06 | 0.00 |
| price above 200 SMA | dev | 344 | 41.86 | 9.01 | 6.40 | 1.30 | 1.60 |
| price above 200 SMA | test | 31 | 41.94 | 6.45 | 12.90 | 1.05 | 1.37 |
| headroom >= 20% | dev | 254 | 38.19 | 8.66 | 9.06 | 0.46 | 1.18 |
| headroom >= 20% | test | 19 | 42.11 | 5.26 | 5.26 | 2.35 | 2.27 |
| market cap < $200M and headroom >= 20% | dev | 43 | 48.84 | 16.28 | 16.28 | 2.06 | 1.63 |
| market cap < $200M and headroom >= 20% | test | 6 | 50.00 | 0.00 | 0.00 | 0.30 | 1.29 |
| shares < 30M and headroom >= 20% | dev | 27 | 44.44 | 22.22 | 18.52 | 3.20 | 1.84 |
| shares < 30M and headroom >= 20% | test | 1 | 100.00 | 0.00 | 0.00 | 4.85 |  |
| technology sector | dev | 104 | 50.96 | 13.46 | 7.69 | 2.43 | 2.09 |
| technology sector | test | 9 | 44.44 | 0.00 | 22.22 | -2.53 | 0.38 |
| earnings-linked (filing 0-3d) | dev | 10 | 50.00 | 0.00 | 0.00 | 0.74 | 2.47 |
| company news in prior 24h | dev | 62 | 32.26 | 8.06 | 14.52 | -0.64 | 0.82 |
| company news in prior 24h | test | 2 | 0.00 | 0.00 | 50.00 | -6.68 | 0.00 |


## Methods and sources

- **Overnight-return literature.** Lou, Polk and Skouras ("A Tug of War: Overnight Versus Intraday Expected Returns") find that momentum returns accrue overnight; Gamm ("A Surprise That Keeps You Awake") documents attention-driven overnight buying after large earnings surprises; Gregoire (Journal of Accounting Research, 2022) shows after-hours prices move through quotes more than trades, so trade-price returns understate and delay the move. Our entries are momentum names held overnight, and our fills use trade bars, so realized fills are optimistic compared to quote-based reality.
- **Selection-bias control.** Bailey and Lopez de Prado's probabilistic / deflated Sharpe ratio (section 11), Benjamini-Hochberg FDR on every family of subgroup and rule tests (sections 4, 5, 8), and a locked chronological test period (sections 7, 8, 12).
- **Purged walk-forward machine learning.** Expanding monthly folds with an embargo, gradient boosting with permutation importance, and a remove-and-retrain check because attribution methods such as SHAP describe the fitted model, including its tails, and can rank features highly that do not generalize (see recent walk-forward TreeSHAP studies).
- **Clustering and regime tests.** Permutation tests on week-level dispersion of big winners (a simple stand-in for self-exciting / Hawkes clustering), change-point test on mean return, sector sympathy permutation test, NMF topic modelling of company descriptions for themes beyond industry codes, k-means archetypes.
- **Data.** Trades from `data/pit` (delisted names included). Sector/industry and point-in-time shares outstanding, market cap and earnings publish dates from SimFin; sector fallback, current float and short interest (snapshots, not point-in-time) and company descriptions from yfinance; headlines from Alpaca (Benzinga) which cover few small-cap press releases, so absence of news in this report means "not found in this feed".
- **Caveats.** About 40% of trades are winners but 10 trades produce more than the entire net profit; every statistic about big winners rests on 157 trades (75 monsters). Float and short interest from yfinance are today's values. Post-entry news is hindsight and cannot be used as a filter.
