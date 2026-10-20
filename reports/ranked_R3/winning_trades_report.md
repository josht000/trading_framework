# Winning-trade analysis: ranked strategy R3 (score >= 3, OTC excluded)

Generated from 399 de-duplicated trades, 2024-10-18 to 2026-10-09 (367 dev, 32 test), 303 symbols, 162 winners (40.6%), 36 big winners (>= +10%), 20 monsters (>= +25%). Returns are per trade after 0.1%/side commission and slippage (0.15% regular / 0.4% extended hours); entry = 15:55 ET, exit rules as in README.

## 1. How concentrated is the profit?

399 trades; 162 winners (40.6%); mean trade +1.00%; median -0.55%. Total of per-trade returns 4.0 (sum of fractions).

| top_k_trades | share_of_total_return% | mean_of_rest% | PF_of_rest |
|---|---|---|---|
| 1.00 | 21.15 | 0.79 | 1.32 |
| 3.00 | 50.31 | 0.50 | 1.20 |
| 5.00 | 71.12 | 0.29 | 1.12 |
| 10.00 | 107.77 | -0.08 | 0.97 |
| 20.00 | 172.01 | -0.76 | 0.71 |
| 50.00 | 275.12 | -2.00 | 0.29 |
| 100.00 | 328.95 | -3.06 | 0.07 |

Bootstrap 95% CI for the mean trade: [-0.04%, +2.06%]; Newey-West t-stat of the mean: 1.80. Kurtosis 14.5, skew 2.8 (heavy right tail).

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

**Clustering of big winners (>= +10%) across weeks.** Dispersion statistic 108 vs 95 expected if big winners were spread evenly given each week's trade count (permutation p = 0.1659); Fano factor of weekly counts 0.96 (1 = Poisson). No significant time clustering of big winners.

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

The best 8 weeks (8% of 95 weeks) hold 50% of the positive weekly return.

**Regime change.** The most likely single change point in mean trade return is 2026-04-15 (max t = 2.05, permutation p = 0.243). Mean trade before: +1.63% (hit rate 43.3%), after: -0.87% (hit rate 32.7%).

Quarterly performance and market context (hit-rate 95% Wilson interval):

| day | n | win% | win%_lo | win%_hi | big% | mean% | IWM_qtr% | SPY_qtr% |
|---|---|---|---|---|---|---|---|---|
| 2024Q4 | 40 | 55.00 | 39.83 | 69.29 | 22.50 | 4.80 | 0.03 | 2.15 |
| 2025Q1 | 23 | 52.17 | 32.96 | 70.76 | 8.70 | 2.02 | -9.72 | -4.55 |
| 2025Q2 | 47 | 42.55 | 29.51 | 56.72 | 8.51 | 0.41 | 8.17 | 10.45 |
| 2025Q3 | 58 | 34.48 | 23.56 | 47.33 | 6.90 | -0.58 | 12.13 | 7.82 |
| 2025Q4 | 65 | 38.46 | 27.59 | 50.62 | 6.15 | 1.05 | 1.74 | 2.36 |
| 2026Q1 | 58 | 43.10 | 31.18 | 55.88 | 12.07 | 2.36 | 0.75 | -4.63 |
| 2026Q2 | 81 | 33.33 | 24.03 | 44.15 | 4.94 | -0.60 | 21.15 | 14.83 |
| 2026Q3 | 22 | 40.91 | 23.26 | 61.27 | 4.55 | -0.19 | -7.51 | 2.12 |
| 2026Q4 | 5 | 40.00 | 11.76 | 76.93 | 20.00 | 4.52 | -2.68 | 0.08 |

Monthly mean trade return vs IWM monthly return: Spearman -0.12 (p = 0.58); vs SPY -0.05 (p = 0.82), n = 25 months.

Autocorrelation of weekly mean trade return, lags 1-8: +0.01, -0.02, -0.11, -0.01, +0.11, +0.09, +0.06, -0.13 (permutation 95% band about +-0.23). None exceeds the band: no detectable week-to-week persistence.

Earnings seasons (approximate small-cap windows: Jan 20-Mar 15, Apr 20-May 20, Jul 20-Aug 20, Oct 20-Nov 20) vs other weeks:

| season | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| False | 223 | 44.84 | 13.00 | 2.04 | -0.50 | 1.78 | 4.55 | 113.99 |
| True | 176 | 35.23 | 3.98 | -0.32 | -0.72 | 0.86 | -0.56 | -13.99 |

Fisher exact test on big-winner share: odds ratio 0.28, p = 0.001.

![02_time](figures/02_time.png)

## 3. What moved them: earnings, news and themes

Share of trades with a headline between entry (15:55) and the next 10:00 ET, by type (Alpaca/Benzinga; mover-list and calendar items excluded from the category counts). Analysis only: this is information the strategy does not have at entry.

| group | n | any_news_after_entry% | company_specific_after% | mover_list_after% | earnings% | contract_partnership% | clinical_fda% | offering_dilution% | m_and_a% | analyst% | theme_hype% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all trades | 399 | 14.79 | 6.52 | 11.03 | 1.50 | 1.50 | 0.25 | 0.50 | 0.00 | 1.75 | 2.76 |
| winners | 162 | 16.67 | 7.41 | 12.35 | 1.23 | 1.85 | 0.00 | 0.00 | 0.00 | 1.85 | 3.09 |
| big winners (>=+10%) | 36 | 25.00 | 16.67 | 16.67 | 0.00 | 8.33 | 0.00 | 0.00 | 0.00 | 2.78 | 11.11 |
| monsters (>=+25%) | 20 | 30.00 | 20.00 | 25.00 | 0.00 | 15.00 | 0.00 | 0.00 | 0.00 | 0.00 | 15.00 |
| losers | 237 | 13.50 | 5.91 | 10.13 | 1.69 | 1.27 | 0.42 | 0.84 | 0.00 | 1.69 | 2.53 |
| worst 5% | 20 | 50.00 | 20.00 | 50.00 | 10.00 | 5.00 | 0.00 | 10.00 | 0.00 | 5.00 | 10.00 |

Does a post-entry event raise the odds of a big winner? (Fisher exact, BH-adjusted q):

| event | share_of_trades% | big_rate_with% | big_rate_without% | mean_ret_with% | odds_ratio | p | q_BH |
|---|---|---|---|---|---|---|---|
| earnings | 1.504 | 0.000 | 9.160 | -7.473 | 0.000 | 1.000 | 1.000 |
| contract_partnership | 1.504 | 50.000 | 8.397 | 13.666 | 10.909 | 0.011 | 0.041 |
| clinical_fda | 0.251 | 0.000 | 9.045 | -0.550 | 0.000 | 1.000 | 1.000 |
| offering_dilution | 0.501 | 0.000 | 9.068 | -20.731 | 0.000 | 1.000 | 1.000 |
| analyst | 1.754 | 14.286 | 8.929 | -0.781 | 1.700 | 0.487 | 0.852 |
| theme_hype | 2.757 | 36.364 | 8.247 | 6.101 | 6.357 | 0.012 | 0.041 |
| mover_list | 11.028 | 13.636 | 8.451 | 0.790 | 1.711 | 0.264 | 0.615 |

**Earnings filings (SimFin publish dates, 172 trades with data)**: a quarterly filing published on the entry day or within the next 3 days flags earnings-linked trades (the filing date can lag the press release).

| earn_next_0_3d | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 162.00 | 41.36 | 9.26 | 1.57 | -0.50 | 1.84 | 2.54 | 97.18 |
| 1.00 | 10.00 | 50.00 | 0.00 | 0.74 | -0.04 | 2.47 | 0.07 | 2.82 |

Big-winner odds ratio for earnings-linked entries 0.00 (Fisher p = 0.603). Share of big winners that are earnings-linked: 0.0% vs 5.8% of all trades.

Pre-entry news (known at 15:55):

| pre_news | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| company news in prior 24h | 64 | 31.25 | 7.81 | -0.82 | -0.76 | 0.77 | -0.53 | -13.21 |
| none | 335 | 42.39 | 9.25 | 1.35 | -0.50 | 1.61 | 4.52 | 113.21 |

Where the move happens (median return vs the 15:55 price at each checkpoint, by final outcome bucket):

| ret | n | median % at 16:45 | median % pre-market (last bar) | median % at 09:30 open |
|---|---|---|---|---|
| <-10% | 33 | -3.41 | -10.95 | -10.86 |
| -10..-3% | 66 | -0.37 | -2.68 | -4.49 |
| -3..0 | 138 | 0.00 | -0.01 | -0.71 |
| 0..3% | 71 | 0.21 | 1.55 | 1.40 |
| 3..10% | 55 | 0.76 | 4.91 | 5.81 |
| 10..25% | 16 | 0.94 | 14.90 | 15.86 |
| >25% | 20 | 2.69 | 20.70 | 20.75 |

![03_catalysts](figures/03_catalysts.png)

## 4. Sectors, industries and themes

By sector (SimFin classification, yfinance fallback; 'Unknown' = no classification, mostly delisted names):

| sector | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| Technology | 113 | 50.44 | 12.39 | 2.03 | 0.00 | 1.86 | 2.30 | 57.51 |
| Industrials | 58 | 39.66 | 12.07 | 1.88 | -0.52 | 1.75 | 1.09 | 27.37 |
| Healthcare | 104 | 38.46 | 4.81 | 0.55 | -0.77 | 1.27 | 0.57 | 14.38 |
| Consumer Cyclical | 32 | 37.50 | 9.38 | -0.07 | -0.63 | 0.98 | -0.02 | -0.54 |
| Basic Materials | 32 | 34.38 | 6.25 | -0.72 | -0.97 | 0.75 | -0.23 | -5.73 |
| Financial Services | 22 | 31.82 | 9.09 | -1.17 | -1.71 | 0.65 | -0.26 | -6.44 |

Over/under-representation of sectors among big winners (Fisher, BH-adjusted):

| sector | n | big_rate% | base_rate% | odds_ratio | p | q_BH |
|---|---|---|---|---|---|---|
| Healthcare | 104 | 4.808 | 9.023 | 0.430 | 0.110 | 0.521 |
| Technology | 113 | 12.389 | 9.023 | 1.697 | 0.174 | 0.521 |
| Industrials | 58 | 12.069 | 9.023 | 1.477 | 0.454 | 0.909 |
| Basic Materials | 32 | 6.250 | 9.023 | 0.653 | 0.755 | 1.000 |
| Financial Services | 22 | 9.091 | 9.023 | 1.009 | 1.000 | 1.000 |
| Consumer Cyclical | 32 | 9.375 | 9.023 | 1.047 | 1.000 | 1.000 |

Top industries by summed return (min 15 trades):

| industry | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| Computer Hardware | 25 | 68.00 | 16.00 | 3.72 | 2.77 | 3.26 | 0.93 | 22.08 |
| Biotechnology | 81 | 38.27 | 4.94 | 0.92 | -0.75 | 1.54 | 0.74 | 17.61 |
| Application Software | 25 | 52.00 | 8.00 | 1.50 | 0.08 | 1.72 | 0.37 | 8.90 |
| Industrial Products | 20 | 40.00 | 10.00 | 1.80 | -0.91 | 1.95 | 0.36 | 8.53 |
| Metals & Mining | 15 | 46.67 | 6.67 | 0.42 | -0.68 | 1.17 | 0.06 | 1.49 |
| Semiconductors | 19 | 36.84 | 5.26 | 0.11 | -1.44 | 1.04 | 0.02 | 0.51 |

Worst industries by summed return:

| industry | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| Computer Hardware | 25 | 68.00 | 16.00 | 3.72 | 2.77 | 3.26 | 0.93 | 22.08 |
| Biotechnology | 81 | 38.27 | 4.94 | 0.92 | -0.75 | 1.54 | 0.74 | 17.61 |
| Application Software | 25 | 52.00 | 8.00 | 1.50 | 0.08 | 1.72 | 0.37 | 8.90 |
| Industrial Products | 20 | 40.00 | 10.00 | 1.80 | -0.91 | 1.95 | 0.36 | 8.53 |
| Metals & Mining | 15 | 46.67 | 6.67 | 0.42 | -0.68 | 1.17 | 0.06 | 1.49 |
| Semiconductors | 19 | 36.84 | 5.26 | 0.11 | -1.44 | 1.04 | 0.02 | 0.51 |

**Same-day sector sympathy.** Pairs of big winners in the same sector on the same day: 1 observed vs 0.4 expected if sectors were assigned at random among big winners on each day (permutation p = 0.3437).

Strongest sector 'waves' (weeks with the most big winners in one sector):

| week | sector | n | tickers |
|---|---|---|---|
| 2025-06-21 | Industrials | 2 | SRFM, SRFM |
| 2026-04-11 | Technology | 2 | RMSG, XNDU |
| 2026-02-28 | Energy | 2 | BATL, TMDE |
| 2024-10-12 | Industrials | 1 | LTBR |
| 2024-11-16 | Healthcare | 1 | ACRS |
| 2024-11-30 | Technology | 1 | BBAI |
| 2024-12-07 | Technology | 1 | WQEY |
| 2024-11-23 | Energy | 1 | OPTT |
| 2024-11-16 | Technology | 1 | PDYN |
| 2024-12-21 | Industrials | 1 | SRFM |
| 2024-12-28 | Technology | 1 | ARBE |
| 2025-03-29 | Consumer Cyclical | 1 | HERE |

![04_sector_heatmap](figures/04_sector_heatmap.png)

![04_sector_bars](figures/04_sector_bars.png)

## 4b. Themes found from company descriptions (TF-IDF + NMF topic model)

Each company's business description was converted to TF-IDF features and factorized into 14 topics; each company is assigned to its dominant topic (377 trades with a confident assignment). Topics ranked by summed trade return:

| topic_label | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| T13: systems, solutions, equipment, segment, industrial | 81 | 45.68 | 12.35 | 2.11 | -0.33 | 1.89 | 1.71 | 38.88 |
| T12: ai, data, cloud, software, solutions | 62 | 45.16 | 12.90 | 2.18 | -0.50 | 1.84 | 1.35 | 30.63 |
| T1: treatment, clinical, phase, therapeutics, trial | 82 | 42.68 | 6.10 | 1.02 | -0.67 | 1.52 | 0.84 | 19.04 |
| T4: gold, silver, located, canada, exploration | 31 | 41.94 | 6.45 | 0.96 | -0.63 | 1.39 | 0.30 | 6.78 |
| T9: energy, power, solar, storage, energy storage | 34 | 41.18 | 8.82 | 0.70 | -0.64 | 1.26 | 0.24 | 5.37 |

Topic enrichment among big winners (Fisher, BH-adjusted):

| topic | n | big_rate% | odds_ratio | p | q_BH |
|---|---|---|---|---|---|
| T13: systems, solutions, equipment, segment, industrial | 81 | 12.346 | 1.527 | 0.284 | 0.647 |
| T12: ai, data, cloud, software, solutions | 62 | 12.903 | 1.580 | 0.335 | 0.647 |
| T1: treatment, clinical, phase, therapeutics, trial | 82 | 6.098 | 0.574 | 0.388 | 0.647 |
| T4: gold, silver, located, canada, exploration | 31 | 6.452 | 0.654 | 0.754 | 0.943 |
| T9: energy, power, solar, storage, energy storage | 34 | 8.824 | 0.941 | 1.000 | 1.000 |

Strongest theme waves (weeks with the most big winners sharing a topic):

| week | topic_label | n | tickers |
|---|---|---|---|
| 2025-06-21 | T13: systems, solutions, equipment, segment, industrial | 2 | SRFM, SRFM |
| 2026-02-28 | T3: oil, gas, natural gas, natural, crude | 2 | BATL, TMDE |
| 2024-10-12 | T9: energy, power, solar, storage, energy storage | 1 | LTBR |
| 2024-11-23 | T12: ai, data, cloud, software, solutions | 1 | ZENA |
| 2024-11-16 | T12: ai, data, cloud, software, solutions | 1 | PDYN |
| 2024-11-30 | T12: ai, data, cloud, software, solutions | 1 | BBAI |
| 2024-12-07 | T12: ai, data, cloud, software, solutions | 1 | WQEY |
| 2024-12-14 | T12: ai, data, cloud, software, solutions | 1 | QUBT |
| 2024-11-16 | T1: treatment, clinical, phase, therapeutics, trial | 1 | ACRS |
| 2024-12-21 | T13: systems, solutions, equipment, segment, industrial | 1 | SRFM |

![04b_topic_heatmap](figures/04b_topic_heatmap.png)

## 5. Float, size, liquidity and volatility

Rank correlation of each feature with trade return and with being a big winner (BH-adjusted q on the return test):

| feature | n | spearman_vs_ret | p_ret | spearman_vs_big | p_big | q_BH(ret) |
|---|---|---|---|---|---|---|
| shares_pit | 272 | -0.044 | 0.473 | -0.187 | 0.002 | 0.765 |
| mcap_pit | 272 | -0.068 | 0.264 | -0.230 | 0.000 | 0.607 |
| float_now | 374 | 0.069 | 0.182 | -0.071 | 0.169 | 0.607 |
| short_pct_float_now | 369 | 0.145 | 0.005 | 0.043 | 0.410 | 0.074 |
| price | 399 | -0.014 | 0.787 | -0.104 | 0.039 | 0.797 |
| dvol20 | 399 | 0.028 | 0.576 | -0.027 | 0.584 | 0.765 |
| rel_vol | 399 | -0.057 | 0.255 | 0.228 | 0.000 | 0.607 |
| atr_pct | 399 | 0.032 | 0.526 | 0.343 | 0.000 | 0.765 |
| sig_day_ret | 399 | -0.026 | 0.601 | 0.334 | 0.000 | 0.765 |
| gap_open | 399 | -0.058 | 0.247 | 0.157 | 0.002 | 0.607 |
| range_pct | 399 | -0.016 | 0.748 | 0.357 | 0.000 | 0.797 |
| run5 | 399 | -0.074 | 0.139 | 0.140 | 0.005 | 0.607 |
| headroom | 399 | -0.013 | 0.797 | -0.039 | 0.441 | 0.797 |
| dist_sma200 | 384 | 0.053 | 0.304 | 0.181 | 0.000 | 0.607 |

**shares_pit** quintiles (shares outstanding (SimFin, point-in-time; float proxy)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (527096.999, 26525068.0] | 56 | 51.79 | 21.43 | 4.28 | 0.27 | 2.60 |
| Q2: (26525068.0, 53907856.6] | 53 | 26.42 | 0.00 | -2.17 | -1.77 | 0.28 |
| Q3: (53907856.6, 85722102.2] | 54 | 50.00 | 9.26 | 2.70 | -0.00 | 2.67 |
| Q4: (85722102.2, 131290808.6] | 54 | 35.19 | 1.85 | -0.21 | -0.62 | 0.85 |
| Q5: (131290808.6, 574592599.0] | 55 | 36.36 | 3.64 | 0.24 | -0.72 | 1.13 |

**mcap_pit** quintiles (market cap (SimFin, point-in-time)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (3531549.8989999997, 124740171.7] | 55 | 54.55 | 21.82 | 4.38 | 0.75 | 2.58 |
| Q2: (124740171.7, 242645380.584] | 54 | 44.44 | 9.26 | 1.89 | -0.46 | 1.82 |
| Q3: (242645380.584, 546744970.512] | 54 | 35.19 | 0.00 | -1.01 | -0.72 | 0.54 |
| Q4: (546744970.512, 924143855.728] | 54 | 24.07 | 0.00 | -1.31 | -0.84 | 0.29 |
| Q5: (924143855.728, 1946667522.96] | 55 | 41.82 | 5.45 | 1.00 | -0.58 | 1.66 |

**float_now** quintiles (float (yfinance current snapshot, not point-in-time)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (-0.001, 23610389.2] | 75 | 41.33 | 14.67 | 0.78 | -0.87 | 1.18 |
| Q2: (23610389.2, 43897885.6] | 75 | 40.00 | 9.33 | 0.94 | -0.55 | 1.38 |
| Q3: (43897885.6, 78351177.6] | 74 | 39.19 | 2.70 | 1.14 | -0.70 | 1.68 |
| Q4: (78351177.6, 125696570.4] | 75 | 48.00 | 13.33 | 2.26 | -0.13 | 2.32 |
| Q5: (125696570.4, 7475818960.0] | 75 | 41.33 | 6.67 | 0.71 | -0.72 | 1.35 |

**short_pct_float_now** quintiles (short % of float (current snapshot)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (-0.0004000000000000001, 0.0477] | 74 | 33.78 | 5.41 | -1.34 | -0.81 | 0.64 |
| Q2: (0.0477, 0.083] | 74 | 37.84 | 9.46 | 1.15 | -0.86 | 1.46 |
| Q3: (0.083, 0.127] | 75 | 41.33 | 13.33 | 2.35 | -0.50 | 2.26 |
| Q4: (0.127, 0.183] | 72 | 47.22 | 8.33 | 1.60 | -0.26 | 1.79 |
| Q5: (0.183, 0.745] | 74 | 51.35 | 9.46 | 2.07 | 0.10 | 1.99 |

**price** quintiles (price at 15:55):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (2.011, 3.737] | 80 | 48.75 | 16.25 | 2.86 | -0.26 | 2.05 |
| Q2: (3.737, 5.921] | 80 | 35.00 | 6.25 | -0.14 | -1.04 | 0.95 |
| Q3: (5.921, 8.721] | 79 | 43.04 | 10.13 | 0.64 | -0.33 | 1.20 |
| Q4: (8.721, 12.802] | 80 | 40.00 | 7.50 | 1.55 | -0.54 | 1.93 |
| Q5: (12.802, 19.85] | 80 | 36.25 | 5.00 | 0.09 | -0.55 | 1.04 |

**rel_vol** quintiles (relative volume at 15:55):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (2.001, 2.369] | 80 | 40.00 | 3.75 | 0.62 | -0.50 | 1.49 |
| Q2: (2.369, 3.008] | 80 | 36.25 | 3.75 | -0.19 | -0.50 | 0.90 |
| Q3: (3.008, 4.418] | 79 | 46.84 | 5.06 | 0.31 | -0.43 | 1.16 |
| Q4: (4.418, 7.737] | 80 | 41.25 | 11.25 | 1.84 | -1.57 | 1.65 |
| Q5: (7.737, 268.884] | 80 | 38.75 | 21.25 | 2.41 | -0.61 | 1.55 |

**atr_pct** quintiles (daily ATR % of price):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (0.00401, 0.0507] | 80 | 30.00 | 0.00 | -0.67 | -0.55 | 0.39 |
| Q2: (0.0507, 0.0669] | 80 | 41.25 | 1.25 | 0.64 | -0.33 | 1.41 |
| Q3: (0.0669, 0.0842] | 79 | 36.71 | 2.53 | -0.78 | -1.34 | 0.72 |
| Q4: (0.0842, 0.109] | 80 | 46.25 | 16.25 | 1.75 | -0.58 | 1.58 |
| Q5: (0.109, 2.229] | 80 | 48.75 | 25.00 | 4.04 | -0.30 | 2.06 |

**sig_day_ret** quintiles (signal-day return to 15:55):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (0.0091, 0.117] | 80 | 30.00 | 0.00 | -0.80 | -0.50 | 0.42 |
| Q2: (0.117, 0.181] | 80 | 42.50 | 0.00 | -0.04 | -0.55 | 0.96 |
| Q3: (0.181, 0.266] | 79 | 41.77 | 5.06 | 1.21 | -0.50 | 1.64 |
| Q4: (0.266, 0.42] | 80 | 42.50 | 12.50 | 0.42 | -1.72 | 1.13 |
| Q5: (0.42, 5.334] | 80 | 46.25 | 27.50 | 4.22 | -0.61 | 1.91 |

**dvol20** quintiles (20-day avg dollar volume):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (743013.7089999999, 2941514.112] | 80 | 48.75 | 18.75 | 3.88 | -0.26 | 2.56 |
| Q2: (2941514.112, 6115567.287] | 80 | 33.75 | 2.50 | -0.92 | -1.29 | 0.67 |
| Q3: (6115567.287, 10105306.954] | 79 | 31.65 | 3.80 | -0.94 | -0.55 | 0.63 |
| Q4: (10105306.954, 20404694.178] | 80 | 42.50 | 5.00 | 0.09 | -0.57 | 1.04 |
| Q5: (20404694.178, 373337124.542] | 80 | 46.25 | 15.00 | 2.87 | -0.38 | 2.41 |

**headroom** quintiles (headroom to next resistance (13m)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (0.00082, 0.109] | 80 | 47.50 | 8.75 | 1.30 | -0.40 | 1.51 |
| Q2: (0.109, 0.828] | 80 | 41.25 | 15.00 | 2.84 | -0.70 | 2.18 |
| Q3: (0.828, 2.0] | 239 | 38.08 | 7.11 | 0.29 | -0.55 | 1.12 |

![05_quintiles](figures/05_quintiles.png)

By shares outstanding (SimFin, millions):

| size | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| <10M sh | 13 | 46.15 | 7.69 | -1.16 | -0.64 | 0.78 | -0.15 | -5.52 |
| 10-30M | 63 | 49.21 | 17.46 | 3.69 | -0.04 | 2.71 | 2.32 | 85.25 |
| 30-100M | 113 | 36.28 | 5.31 | 0.62 | -0.65 | 1.33 | 0.71 | 25.91 |
| 100-300M | 71 | 39.44 | 1.41 | -0.22 | -0.72 | 0.88 | -0.16 | -5.73 |
| >300M | 12 | 25.00 | 8.33 | 0.02 | -1.71 | 1.01 | 0.00 | 0.10 |

## 6. Link to the broader market

Daily mean trade return vs market variables (219 trading days; '*_gap' = next-session close-to-open move of the index, known only afterwards; '*_day' = the index move from the prior close to 15:55 on entry day):

| market variable | spearman_vs_daily_mean_ret | p |
|---|---|---|
| spy_gap | 0.174 | 0.010 |
| iwm_gap | 0.167 | 0.013 |
| qqq_gap | 0.198 | 0.003 |
| spy_day | -0.014 | 0.835 |
| iwm_day | 0.096 | 0.158 |
| iwm_ret5 | -0.104 | 0.124 |
| iwm_ret20 | -0.014 | 0.837 |
| spy_ret20 | 0.019 | 0.785 |

**Overnight beta.** OLS with Newey-West errors, daily mean trade return on IWM and SPY overnight gaps: intercept (alpha) +0.72% per day (t = 1.25); beta to IWM gap 0.17 (t = 0.14); beta to SPY gap 0.25 (t = 0.12); R2 = 0.001.

Trade outcomes by tercile of market variable:

| variable | b | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|---|
| IWM next-session gap | low | 134 | 29.10 | 8.96 | -0.18 | -1.30 | 0.94 | -0.25 | -6.17 |
| IWM next-session gap | mid | 133 | 41.35 | 9.02 | 1.76 | -0.55 | 1.88 | 2.34 | 58.55 |
| IWM next-session gap | high | 132 | 51.52 | 9.09 | 1.44 | 0.13 | 1.68 | 1.90 | 47.61 |
| IWM entry-day move | low | 133 | 39.85 | 6.77 | 0.54 | -0.55 | 1.22 | 0.72 | 17.98 |
| IWM entry-day move | mid | 133 | 39.85 | 9.02 | 0.62 | -0.64 | 1.22 | 0.82 | 20.50 |
| IWM entry-day move | high | 133 | 42.11 | 11.28 | 1.85 | -0.50 | 1.88 | 2.46 | 61.53 |
| SPY entry-day move | low | 134 | 39.55 | 10.45 | 1.04 | -0.55 | 1.36 | 1.40 | 35.03 |
| SPY entry-day move | mid | 132 | 47.73 | 8.33 | 1.32 | -0.11 | 1.62 | 1.75 | 43.78 |
| SPY entry-day move | high | 133 | 34.59 | 8.27 | 0.64 | -0.91 | 1.27 | 0.85 | 21.19 |
| IWM 20-day trend | low | 133 | 39.85 | 12.03 | 1.74 | -0.56 | 1.76 | 2.32 | 58.01 |
| IWM 20-day trend | mid | 133 | 47.37 | 8.27 | 1.71 | -0.33 | 1.76 | 2.27 | 56.90 |
| IWM 20-day trend | high | 133 | 34.59 | 6.77 | -0.45 | -0.82 | 0.84 | -0.60 | -14.91 |

**Breadth ('hype') indicator.** The number of candidate breakouts that day relative to its prior 20-day average: Spearman vs that day's mean trade return +0.081 (p = 0.238); vs number of big winners -0.028.

![06_market](figures/06_market.png)

### 6b. How much of the result is just the market's overnight move?

| IWM hedge ratio | mean% | median% | win% | PF | daily_Sharpe_ann | NW_t_daily |
|---|---|---|---|---|---|---|
| 0.00 | 1.00 | -0.55 | 40.60 | 1.41 | 1.34 | 1.31 |
| 0.50 | 0.98 | -0.56 | 39.10 | 1.40 | 1.29 | 1.26 |
| 1.00 | 0.96 | -0.64 | 40.10 | 1.39 | 1.25 | 1.21 |

A hedge ratio of 1 removes the IWM close-to-open move entirely. The trades hold some positions through the open only partly (several exit pre-market or after-hours), so this is an approximation; it shows whether the edge survives once index direction is removed.

## 7. Can the winners be predicted? (gradient boosting, purged walk-forward)

HistGradientBoosting (depth 3, strong regularization) on 32 point-in-time features known at 15:55; expanding monthly folds with a 5-business-day purge before each test month; final model trained on all dev and scored on the locked test period. AUC 0.5 = no skill.

| target | dev_oos_AUC | dev_oos_PR_AUC | base_rate% | dev_oos_top10%_lift | test_AUC | test_PR_AUC | test_base_rate% | test_top20%_lift | n_dev_oos | n_test |
|---|---|---|---|---|---|---|---|---|---|---|
| big | 0.750 | 0.232 | 7.566 | 3.084 | 0.917 | 0.367 | 6.250 | 5.333 | 304 | 32 |
| winner | 0.480 | 0.382 | 37.829 | 0.881 | 0.530 | 0.484 | 40.625 | 0.821 | 304 | 32 |

Permutation importance (drop in AUC when a feature is shuffled inside each walk-forward test fold; big-winner target):

| index | mean_AUC_drop | sd_across_folds | folds_positive% |
|---|---|---|---|
| dist_ema9 | 0.0257 | 0.0222 | 80.0000 |
| atr_pct | 0.0149 | 0.0596 | 50.0000 |
| range_pct | 0.0084 | 0.0172 | 50.0000 |
| iwm_ret5 | 0.0032 | 0.0081 | 30.0000 |
| price | 0.0031 | 0.0080 | 20.0000 |
| dow | 0.0024 | 0.0157 | 30.0000 |
| iwm_day | 0.0023 | 0.0068 | 10.0000 |
| spy_day | 0.0021 | 0.0092 | 20.0000 |
| day_dvol | 0.0017 | 0.0093 | 20.0000 |
| gap_open | 0.0003 | 0.0091 | 30.0000 |
| spy_ret5 | 0.0000 | 0.0066 | 10.0000 |
| pre_specific_24h | 0.0000 | 0.0000 | 0.0000 |
| spy_sma50_up | 0.0000 | 0.0000 | 0.0000 |
| iwm_sma50_up | 0.0000 | 0.0000 | 0.0000 |
| iwm_sma200_up | 0.0000 | 0.0000 | 0.0000 |

**Remove-and-retrain check** (SHAP and permutation importance describe what a model fitted, not what generalizes): walk-forward AUC with all features 0.750; without the top 3 (dist_ema9, atr_pct, range_pct) 0.745; with only those 3 0.841.

![07_partial_dependence](figures/07_partial_dependence.png)

A depth-3 decision tree on dev (readable splits; unstable, shown for interpretation only):

```
|--- range_pct <= 0.425
|   |--- run5 <= 0.209
|   |   |--- range_pct <= 0.206
|   |   |   |--- class: False
|   |   |--- range_pct >  0.206
|   |   |   |--- class: False
|   |--- run5 >  0.209
|   |   |--- class: False
|--- range_pct >  0.425
|   |--- class: False
```

## 8. Rule mining with multiple-testing control

6740 candidate rules (single and pairs of quartile/median conditions) with at least 60 dev trades: 2657 tested for a higher big-winner rate than the 9.3% base (one-sided binomial). 506 pass BH-FDR q < 0.10 on dev.

Strongest rules by dev p-value, with their out-of-sample behaviour on the locked test period (test base big-winner rate 6.2%):

| rule | n | big_rate% | big_loss_rate% | lift | mean_ret% | PF | p | q_BH | test_n | test_big_rate% | test_mean_ret% | test_big_loss% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| atr_pct >= 0.103 AND dist_ema9 >= 0.224 | 70 | 34.3 | 18.6 | 3.7 | 5.34 | 2.32 | 7.86e-09 | 2.09e-05 | 6 | 33.3 | 7.08 | 50 |
| sig_day_ret >= 0.219 AND atr_pct >= 0.103 | 73 | 32.9 | 17.8 | 3.55 | 5.06 | 2.28 | 2.01e-08 | 2.67e-05 | 6 | 33.3 | 7.08 | 50 |
| atr_pct >= 0.103 AND day_dvol >= 4.83e+07 | 60 | 35 | 18.3 | 3.78 | 5.57 | 2.45 | 4.41e-08 | 3.9e-05 | 5 | 40 | 9.83 | 60 |
| sig_day_ret >= 0.365 AND range_pct >= 0.33 | 65 | 32.3 | 24.6 | 3.49 | 4.83 | 1.99 | 2.12e-07 | 0.000122 | 7 | 28.6 | 5.84 | 57.1 |
| range_pct >= 0.33 AND dist_ema9 >= 0.363 | 60 | 33.3 | 26.7 | 3.6 | 5.12 | 2.03 | 2.3e-07 | 0.000122 | 7 | 14.3 | -3.86 | 71.4 |
| atr_pct >= 0.103 AND dist_ema20 >= 0.301 | 72 | 30.6 | 18.1 | 3.3 | 4.52 | 2.13 | 3.3e-07 | 0.000125 | 5 | 20 | -2.88 | 60 |
| atr_pct >= 0.0778 AND dist_ema9 >= 0.363 | 72 | 30.6 | 26.4 | 3.3 | 4.07 | 1.79 | 3.3e-07 | 0.000125 | 7 | 14.3 | -3.86 | 71.4 |
| atr_pct >= 0.0778 AND range_pct >= 0.33 | 80 | 28.8 | 22.5 | 3.1 | 4.23 | 1.94 | 6.06e-07 | 0.000201 | 8 | 25 | 3.73 | 62.5 |
| atr_pct >= 0.0778 AND dist_ema9 >= 0.224 | 130 | 23.8 | 16.9 | 2.57 | 3.02 | 1.75 | 7.56e-07 | 0.000205 | 12 | 16.7 | 2.07 | 41.7 |
| atr_pct >= 0.103 AND range_pct >= 0.201 | 81 | 28.4 | 17.3 | 3.06 | 4.17 | 2.09 | 7.73e-07 | 0.000205 | 6 | 33.3 | 7.08 | 50 |
| range_pct >= 0.201 AND dist_ema9 >= 0.363 | 82 | 28 | 24.4 | 3.03 | 3.6 | 1.73 | 9.8e-07 | 0.000214 | 9 | 11.1 | -3.89 | 55.6 |
| range_pct >= 0.33 AND day_dvol >= 4.83e+07 | 71 | 29.6 | 21.1 | 3.19 | 4.65 | 2.08 | 1.1e-06 | 0.000214 | 8 | 25 | 3.73 | 62.5 |
| rel_vol >= 3.46 AND range_pct >= 0.33 | 77 | 28.6 | 20.8 | 3.08 | 4.99 | 2.13 | 1.19e-06 | 0.000214 | 7 | 28.6 | 5.84 | 57.1 |
| sig_day_ret >= 0.219 AND range_pct >= 0.33 | 89 | 27 | 19.1 | 2.91 | 4.64 | 2.09 | 1.26e-06 | 0.000214 | 8 | 25 | 3.73 | 62.5 |
| sig_day_ret >= 0.219 AND atr_pct >= 0.0778 | 127 | 23.6 | 15.7 | 2.55 | 3.08 | 1.81 | 1.39e-06 | 0.000214 | 11 | 18.2 | 2.31 | 45.5 |

## 9. Winner archetypes (clustering)

K-means on 36 big winners (>= +10%) using 10 standardized features; k = 2 chosen by silhouette (0.30).

| cluster | n | mean_ret% | sig_day_ret% | gap_open% | rel_vol | atr% | range% | price | dvol20_$M | run5% | headroom | top_sectors | examples |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 27 | 26.01 | 40.94 | 8.83 | 5.71 | 10.80 | 44.52 | 5.98 | 16.67 | 26.37 | 0.83 | Technology (8), Industrials (5) | BMEA, BATL, GWH, SDEV, HERE |
| 1 | 9 | 30.96 | 228.55 | 21.90 | 103.69 | 21.96 | 146.51 | 4.46 | 1.66 | 9.29 | 0.51 | Technology (6), Industrials (2) | JLHL, AZIO, ZENA, WQEY, RMSG |

Share of ALL trades falling in each archetype's region of feature space (nearest centroid) and how those trades fared:

| arch | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 378.00 | 40.48 | 7.14 | 0.66 | -0.55 | 1.29 | 2.48 | 62.08 |
| 1.00 | 21.00 | 42.86 | 42.86 | 7.21 | -0.75 | 2.19 | 1.51 | 37.92 |

## 10. Event study: the path around entry

Mean cumulative return versus IWM (%, winsorized 1%) relative to the close before the entry day, by final trade outcome. Day 0 is the entry day (entry at 15:55, so day 0 shows the run-up the strategy buys into), day +1 the overnight session:

| bucket | n | d-5 | d-4 | d-3 | d-2 | d-1 | d0 | d1 | d2 | d3 |
|---|---|---|---|---|---|---|---|---|---|---|
| loser <-5% | 66 | -13.07 | -9.39 | -8.73 | -7.71 | 0.00 | 73.39 | 53.47 | 45.26 | 45.25 |
| small loss | 170 | -4.96 | -4.45 | -3.45 | -2.16 | 0.00 | 21.12 | 19.50 | 18.52 | 17.74 |
| winner 0-10% | 126 | -4.70 | -5.20 | -4.63 | -2.18 | 0.00 | 22.44 | 25.81 | 26.43 | 28.03 |
| big winner >=10% | 36 | -14.61 | -16.06 | -15.42 | -8.76 | 0.00 | 95.90 | 127.93 | 104.70 | 103.64 |

![10_event_study](figures/10_event_study.png)

## 11. Statistical rigor: is any of this skill?

Probabilistic / deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) on dev daily portfolio returns. 'N' is the number of strategy variants tried; N=1 is the plain probabilistic Sharpe ratio, larger N deflates for selection bias. This project has tried on the order of 100 variants (exit rules, filters, sizing).

| strategy | days | daily_SR | annual_SR | skew | kurtosis | NW_t(mean) | P(skill) N=1 | P(skill) N=30 | P(skill) N=100 | P(skill) N=300 |
|---|---|---|---|---|---|---|---|---|---|---|
| this strategy | 432 | 0.084 | 1.332 | 3.623 | 29.742 | 1.713 | 0.978 | 0.477 | 0.303 | 0.189 |

## 12. Do the main findings hold out of sample (dev vs the locked test period)?

| group | period | n | win% | big% | big_loss% | mean% | PF |
|---|---|---|---|---|---|---|---|
| all trades | dev | 367 | 40.60 | 9.26 | 7.63 | 1.03 | 1.43 |
| all trades | test | 32 | 40.62 | 6.25 | 15.62 | 0.61 | 1.19 |
| shares outstanding < 30M | dev | 71 | 49.30 | 16.90 | 9.86 | 3.02 | 2.08 |
| shares outstanding < 30M | test | 5 | 40.00 | 0.00 | 0.00 | 0.57 | 1.44 |
| market cap < $200M | dev | 84 | 52.38 | 16.67 | 10.71 | 3.75 | 2.45 |
| market cap < $200M | test | 8 | 37.50 | 0.00 | 0.00 | -0.69 | 0.59 |
| daily ATR >= 9% | dev | 129 | 48.84 | 22.48 | 14.73 | 3.13 | 1.93 |
| daily ATR >= 9% | test | 9 | 44.44 | 22.22 | 44.44 | 4.93 | 1.78 |
| signal-day move >= +26% | dev | 153 | 43.79 | 19.61 | 14.38 | 2.18 | 1.59 |
| signal-day move >= +26% | test | 13 | 46.15 | 15.38 | 38.46 | 2.67 | 1.48 |
| rel. volume >= 5.4 | dev | 116 | 38.79 | 18.10 | 16.38 | 2.34 | 1.59 |
| rel. volume >= 5.4 | test | 13 | 46.15 | 15.38 | 30.77 | 3.17 | 1.63 |
| price below 200 SMA | dev | 5 | 20.00 | 0.00 | 40.00 | -6.85 | 0.08 |
| price below 200 SMA | test | 1 | 0.00 | 0.00 | 100.00 | -13.06 | 0.00 |
| price above 200 SMA | dev | 347 | 42.07 | 9.51 | 6.34 | 1.36 | 1.63 |
| price above 200 SMA | test | 31 | 41.94 | 6.45 | 12.90 | 1.05 | 1.37 |
| headroom >= 20% | dev | 255 | 38.43 | 9.02 | 9.02 | 0.51 | 1.20 |
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
