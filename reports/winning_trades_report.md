# Executive summary: what the winning trades have in common

Base strategy (13-month `atr_pivot` breakout at 15:55, `pm_target25_v2` exits, no entry filters), 2,584 de-duplicated trades, 2024-10-18 to 2026-10-13, 40.1% winners, mean trade +0.13%, median -0.60%. Everything below is descriptive; the models were trained on the dev period only and scored on the locked last 3 months.

**1. The profit is a handful of trades.** The top 3 trades are 75% of the net return and the top 5 are 114%; without the best 10 (0.4% of trades) the strategy has a profit factor of 0.95 and a mean of -0.12%. Bootstrap 95% CI for the mean trade is -0.20% to +0.46%.

**2. It is cyclic, but not because of earnings.**
- Big winners (>= +10%) are significantly clustered in time (permutation p = 0.0005; weekly Fano factor 1.75). The best 8 weeks (8% of weeks) hold 52% of the positive weekly return. There is no week-to-week autocorrelation, so it is bursts, not a smooth cycle.
- A regime change is detected on 2025-01-06 (p = 0.02): mean trade +1.83% (45.2% hit rate) before, -0.09% (39.4%) after. The early burst is the Nov-Dec 2024 AI and quantum small-cap mania.
- Earnings do not drive the winners: entries inside approximate earnings windows averaged -0.26% versus +0.47% outside (big-winner odds ratio 0.56, p = 0.001), and trades with a quarterly filing 0-3 days after entry averaged -1.22% (PF 0.53, n = 104). Only 3.5% of big winners were earnings-linked versus 8.8% of all trades.

**3. Sector and theme hype is the real pattern.**
- Same-day sector sympathy among big winners is significant (18 same-sector pairs vs 10.5 expected, p = 0.021). Technology is the only sector over-represented after FDR control (odds 1.73, q = 0.028).
- A topic model of company descriptions finds the themes: quantum computing (45 trades, 62% hit rate, +4.97% mean, PF 3.78, q = 0.004), AI (53% of all profit), and oil and gas (39% of profit, driven by the Feb-Mar 2026 Iran/Hormuz spike: BATL, TPET, TMDE, INDO, SKYQ). Waves: Dec 2024 (INTZ, RZLV, QUBT, QBTS, LAES), Mar 2025 quantum, Oct 2025 biotech (BMEA, RVPH, TNYA), Feb-Mar 2026 energy.
- Headlines after entry: contract/partnership news (odds ratio 7.2, q = 0.005) and theme-hype keywords (6.6, q < 0.001) separate big winners from the rest. But 87% of big winners have no company-specific headline in the Alpaca/Benzinga feed, so the feed misses most catalysts; news known at entry did not help (company news in the prior 24h: mean +0.12% vs +0.14%).

**4. Float and company size matter, and held out of sample.**
- Shares outstanding under 30M: 212 trades, big-winner rate 15-24%, mean +1.8% to +3.1% (PF 1.33 to 2.36), 147% of total profit; the median trade there is still a loss (lottery-like). Market cap under about $195M: mean +1.86%, PF 1.70. On the test period: shares < 30M, 18 trades, mean +3.8%, PF 2.96; market cap < $200M, 37 trades, mean +1.54%, PF 1.99. Large and mid float names (30M-300M shares) lost money.
- Headroom of at least 20% to the next 13-month resistance also held: dev mean +1.18% (PF 1.42, 502 trades), test +1.23% (PF 1.44, 44 trades). With market cap < $200M the dev mean is +3.0% (PF 1.83, 75 trades), test +1.8% (10 trades).

**5. Volatility predicts big moves of both signs, and that edge did not hold.**
- A purged walk-forward gradient-boosting model predicts big winners with AUC 0.775 (dev) and 0.778 (test), but it predicts plain winners (return > 0) at AUC 0.50 / 0.52: it finds volatile, extended names, not direction. The top-lift rules (e.g. ATR >= 9.2% and signal-day move >= +26%) have a 26% big-winner rate and a 27% big-loser rate; on the test period they lost money (-0.3% to -3.8% mean, 31-47% big losers). Volatility-defined subgroups (ATR >= 9%, signal-day move >= +26%, relative volume >= 5.4) flipped from PF 1.28-1.38 on dev to 0.67-0.77 on test.
- Winners and the worst losers look alike before entry: both ran about +53% to +69% on the entry day; the difference is the overnight gap (+28% for big winners, -17% for big losers).

**6. The broader market affects hit rate, not the edge.**
- The next-session IWM gap (unknown at entry) moves the hit rate from 29.6% (weakest third, mean -0.66%) to 51.8% (strongest third, mean +1.05%, PF 1.57); correlation with the daily mean trade return is about 0.22 (p < 0.001). Entry-day SPY and IWM moves and the 20-day trend do not predict anything (Spearman 0.05, not significant). Hedging with an IWM short barely changes the mean (0.13% to 0.11%), so the result is not market beta; it is concentrated in themes and small floats.

**7. Why the filtered strategies miss the biggest moves (your observation is right).** The market-regime and 200 SMA filters remove the hyped spikes: `S1` keeps 4 of the top 10 trades and 53 of 157 big winners, `H3` keeps 5 of the top 10, `H2` keeps 2 of the top 20. They still keep 82-174% of the base strategy's total return because they remove many more losers (mean trade +0.60% to +1.35% versus +0.13%), but they give up the tail that creates the burst-and-drawdown shape of the base curve.

**8. Statistical caution.** Deflated Sharpe ratio (about 100 variants tried): base 0.09, `S1` 0.67, `H3` 0.60, `H2` 0.34 probability of real skill. Every big-winner statistic rests on 157 trades (75 monsters) and the test period has only 256 trades, so treat the subgroup results as hypotheses.

**What looks most worth testing next (all on dev, then one look at test):** low float (< 30M shares) or market cap < $200M combined with headroom >= 20%; a theme/sympathy signal (several same-theme names breaking out the same day, from the topic model); and using position sizing that reflects the lottery-like payoff (small, fixed risk per name) instead of equal weight.

---



# Winning-trade analysis: base strategy (13-month atr_pivot, pm_target25_v2 exits)

Generated from 2584 de-duplicated trades, 2024-10-18 to 2026-10-13 (2273 dev, 311 test), 1074 symbols, 1037 winners (40.1%), 157 big winners (>= +10%), 75 monsters (>= +25%). Returns are per trade after 0.1%/side commission and slippage (0.15% regular / 0.4% extended hours); entry = 15:55 ET, exit rules as in README.

## 1. How concentrated is the profit?

2584 trades; 1037 winners (40.1%); mean trade +0.13%; median -0.60%. Total of per-trade returns 3.5 (sum of fractions).

| top_k_trades | share_of_total_return% | mean_of_rest% | PF_of_rest |
|---|---|---|---|
| 1.00 | 28.40 | 0.10 | 1.04 |
| 3.00 | 75.27 | 0.03 | 1.01 |
| 5.00 | 114.10 | -0.02 | 0.99 |
| 10.00 | 190.48 | -0.12 | 0.95 |
| 20.00 | 300.35 | -0.27 | 0.88 |
| 50.00 | 549.79 | -0.61 | 0.73 |
| 100.00 | 912.15 | -1.13 | 0.52 |

Bootstrap 95% CI for the mean trade: [-0.20%, +0.46%]; Newey-West t-stat of the mean: 0.72. Kurtosis 21.1, skew 3.0 (heavy right tail).

![01_concentration](figures/01_concentration.png)

Top 20 trades:

| day | symbol | ret | sector | industry | sig_day_ret | gap_open | rel_vol | exit_reason | period |
|---|---|---|---|---|---|---|---|---|---|
| 2026-02-27 | BATL | 98.2 | Energy | Oil & Gas - E&P | 34.9 | 22.4 | 6.4 | pm_target | dev |
| 2025-10-06 | BMEA | 84.4 | Healthcare | Biotechnology | 23.0 | 0.9 | 6.2 | ah_target | dev |
| 2024-12-27 | INTZ | 77.6 | Technology | Software - Infrastructure | 371.1 | 346.7 | 186.9 | pm_target | dev |
| 2025-06-05 | EYEN | 72.7 | Healthcare | Biotechnology | 52.6 | 4.8 | 2.1 | pm_target | dev |
| 2026-02-27 | INDO | 61.5 | Energy | Oil & Gas E&P | 21.8 | 9.1 | 2.4 | pm_target | dev |
| 2026-03-02 | BATL | 59.5 | Energy | Oil & Gas - E&P | 116.1 | 93.8 | 6.3 | pm_target | dev |
| 2026-07-09 | JLHL | 56.9 | Industrials | Building Products & Equipment | 325.5 | 21.9 | 126.3 | ah_target | test |
| 2024-11-07 | LICY | 50.6 | Industrials | Waste Management | 13.5 | 2.5 | 2.1 | pm_target | dev |
| 2026-01-12 | AZIO | 49.4 | Technology | Software - Infrastructure | 392.3 | 148.3 | 20.9 | ah_target | dev |
| 2025-10-27 | GPUS | 47.7 | Industrials | Conglomerates | 19.2 | 7.5 | 2.7 | pm_target | dev |
| 2026-03-05 | TURB | 45.4 | Technology | Solar | 77.6 | 14.3 | 4.4 | pm_target | dev |
| 2025-03-31 | ICCT | 42.8 | Unknown |  | 637.5 | 146.3 | 326.4 | pm_target | dev |
| 2025-04-16 | HTZ | 39.9 | Consumer Cyclical | Travel & Leisure | 52.5 | 19.3 | 4.0 | pm_target | dev |
| 2026-04-10 | SKYQ | 39.4 | Energy | Oil & Gas Integrated | 74.2 | 6.3 | 2.6 | pm_target | dev |
| 2026-08-06 | HTZ | 38.1 | Consumer Cyclical | Travel & Leisure | 31.8 | 16.0 | 5.5 | pm_target | test |
| 2024-12-23 | LAES | 36.6 | Technology | Semiconductors | 76.4 | 30.5 | 2.1 | pm_target | dev |
| 2026-02-11 | FSLY | 36.4 | Technology | Application Software | 2.9 | 2.3 | 2.3 | pm_target | dev |
| 2025-05-19 | CHAI | 34.0 | Communication Services | Internet Content & Information | 75.8 | 12.6 | 6.7 | pm_target | dev |
| 2025-10-13 | GWH | 33.7 | Industrials | Industrial Products | 125.7 | 39.4 | 8.3 | pm_target | dev |
| 2025-07-18 | OPEN | 33.4 | Real Estate | Real Estate Services | 37.6 | 11.8 | 3.7 | pm_target | dev |

## 2. Timing: clusters, cycles and regimes

**Clustering of big winners (>= +10%) across weeks.** Dispersion statistic 165 vs 104 expected if big winners were spread evenly given each week's trade count (permutation p = 0.0005); Fano factor of weekly counts 1.75 (1 = Poisson). Big winners are significantly clustered in time.

Best weeks by summed trade return:

| week | n | wins | big | monster | total | win% |
|---|---|---|---|---|---|---|
| 2026-02-28 00:00:00 | 35 | 21 | 6 | 6 | 2.13 | 60.00 |
| 2024-12-21 00:00:00 | 27 | 16 | 7 | 4 | 2.07 | 59.26 |
| 2024-12-28 00:00:00 | 21 | 11 | 6 | 5 | 1.35 | 52.38 |
| 2024-12-14 00:00:00 | 21 | 10 | 6 | 4 | 1.13 | 47.62 |
| 2026-02-21 00:00:00 | 47 | 11 | 4 | 2 | 1.12 | 23.40 |
| 2025-10-04 00:00:00 | 42 | 24 | 3 | 2 | 1.06 | 57.14 |
| 2025-07-12 00:00:00 | 42 | 24 | 6 | 2 | 1.04 | 57.14 |
| 2024-11-16 00:00:00 | 21 | 16 | 3 | 1 | 0.94 | 76.19 |

The best 8 weeks (8% of 105 weeks) hold 52% of the positive weekly return.

**Regime change.** The most likely single change point in mean trade return is 2025-01-06 (max t = 3.66, permutation p = 0.020). Mean trade before: +1.83% (hit rate 45.2%), after: -0.09% (hit rate 39.4%).

Quarterly performance and market context (hit-rate 95% Wilson interval):

| day | n | win% | win%_lo | win%_hi | big% | mean% | IWM_qtr% | SPY_qtr% |
|---|---|---|---|---|---|---|---|---|
| 2024Q4 | 283 | 43.11 | 37.47 | 48.93 | 9.89 | 1.31 | 0.03 | 2.15 |
| 2025Q1 | 214 | 38.79 | 32.51 | 45.46 | 7.94 | 0.75 | -9.72 | -4.55 |
| 2025Q2 | 317 | 38.17 | 32.99 | 43.63 | 4.73 | -0.29 | 8.17 | 10.45 |
| 2025Q3 | 443 | 44.24 | 39.69 | 48.90 | 5.87 | -0.03 | 12.13 | 7.82 |
| 2025Q4 | 374 | 42.25 | 37.34 | 47.31 | 4.01 | 0.01 | 1.74 | 2.36 |
| 2026Q1 | 318 | 37.74 | 32.58 | 43.18 | 6.60 | 0.74 | 0.75 | -4.63 |
| 2026Q2 | 362 | 36.74 | 31.94 | 41.82 | 5.25 | -0.61 | 21.15 | 14.83 |
| 2026Q3 | 251 | 37.45 | 31.69 | 43.58 | 5.58 | -0.50 | -7.51 | 2.12 |
| 2026Q4 | 22 | 45.45 | 26.92 | 65.34 | 9.09 | 1.23 | -2.68 | 0.08 |

Monthly mean trade return vs IWM monthly return: Spearman -0.32 (p = 0.12); vs SPY -0.17 (p = 0.43), n = 25 months.

Autocorrelation of weekly mean trade return, lags 1-8: +0.19, +0.01, -0.08, -0.02, +0.07, +0.18, -0.04, -0.04 (permutation 95% band about +-0.22). None exceeds the band: no detectable week-to-week persistence.

Earnings seasons (approximate small-cap windows: Jan 20-Mar 15, Apr 20-May 20, Jul 20-Aug 20, Oct 20-Nov 20) vs other weeks:

| season | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| False | 1380 | 42.83 | 7.54 | 0.47 | -0.50 | 1.20 | 6.55 | 189.54 |
| True | 1204 | 37.04 | 4.40 | -0.26 | -0.75 | 0.88 | -3.10 | -89.54 |

Fisher exact test on big-winner share: odds ratio 0.56, p = 0.001.

![02_time](figures/02_time.png)

## 3. What moved them: earnings, news and themes

Share of trades with a headline between entry (15:55) and the next 10:00 ET, by type (Alpaca/Benzinga; mover-list and calendar items excluded from the category counts). Analysis only: this is information the strategy does not have at entry.

| group | n | any_news_after_entry% | company_specific_after% | mover_list_after% | earnings% | contract_partnership% | clinical_fda% | offering_dilution% | m_and_a% | analyst% | theme_hype% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all trades | 2584 | 9.06 | 4.22 | 6.42 | 0.81 | 0.62 | 0.27 | 0.19 | 0.08 | 1.04 | 1.20 |
| winners | 1037 | 10.41 | 5.11 | 7.62 | 0.77 | 0.96 | 0.29 | 0.19 | 0.10 | 1.06 | 1.64 |
| big winners (>=+10%) | 157 | 21.66 | 12.74 | 17.20 | 1.27 | 3.18 | 0.00 | 0.00 | 0.00 | 2.55 | 5.73 |
| monsters (>=+25%) | 75 | 21.33 | 10.67 | 18.67 | 1.33 | 5.33 | 0.00 | 0.00 | 0.00 | 1.33 | 6.67 |
| losers | 1547 | 8.14 | 3.62 | 5.62 | 0.84 | 0.39 | 0.26 | 0.19 | 0.06 | 1.03 | 0.90 |
| worst 5% | 130 | 27.69 | 9.23 | 26.92 | 1.54 | 0.77 | 1.54 | 1.54 | 0.00 | 1.54 | 2.31 |

Does a post-entry event raise the odds of a big winner? (Fisher exact, BH-adjusted q):

| event | share_of_trades% | big_rate_with% | big_rate_without% | mean_ret_with% | odds_ratio | p | q_BH |
|---|---|---|---|---|---|---|---|
| earnings | 0.813 | 9.524 | 6.048 | -0.809 | 1.635 | 0.368 | 0.589 |
| contract_partnership | 0.619 | 31.250 | 5.919 | 9.074 | 7.225 | 0.002 | 0.005 |
| clinical_fda | 0.271 | 0.000 | 6.092 | -1.475 | 0.000 | 1.000 | 1.000 |
| offering_dilution | 0.193 | 0.000 | 6.088 | -6.231 | 0.000 | 1.000 | 1.000 |
| m_and_a | 0.077 | 0.000 | 6.081 | 2.022 | 0.000 | 1.000 | 1.000 |
| analyst | 1.045 | 14.815 | 5.984 | 0.929 | 2.733 | 0.077 | 0.155 |
| theme_hype | 1.200 | 29.032 | 5.797 | 5.231 | 6.648 | 0.000 | 0.000 |
| mover_list | 6.424 | 16.265 | 5.376 | 0.116 | 3.419 | 0.000 | 0.000 |

**Earnings filings (SimFin publish dates, 1179 trades with data)**: a quarterly filing published on the entry day or within the next 3 days flags earnings-linked trades (the filing date can lag the press release).

| earn_next_0_3d | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 1075.00 | 39.16 | 5.12 | 0.20 | -0.58 | 1.11 | 2.10 | 252.67 |
| 1.00 | 104.00 | 43.27 | 1.92 | -1.22 | -0.60 | 0.53 | -1.27 | -152.67 |

Big-winner odds ratio for earnings-linked entries 0.36 (Fisher p = 0.226). Share of big winners that are earnings-linked: 3.5% vs 8.8% of all trades.

Pre-entry news (known at 15:55):

| pre_news | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| company news in prior 24h | 267 | 38.20 | 8.99 | 0.12 | -0.69 | 1.04 | 0.31 | 9.10 |
| none | 2317 | 40.35 | 5.74 | 0.14 | -0.58 | 1.06 | 3.14 | 90.90 |

Where the move happens (median return vs the 15:55 price at each checkpoint, by final outcome bucket):

| ret | n | median % at 16:45 | median % pre-market (last bar) | median % at 09:30 open |
|---|---|---|---|---|
| <-10% | 162 | -4.66 | -12.05 | -13.05 |
| -10..-3% | 421 | -0.25 | -3.00 | -4.22 |
| -3..0 | 964 | -0.05 | -0.26 | -0.74 |
| 0..3% | 607 | 0.16 | 1.49 | 1.54 |
| 3..10% | 273 | 0.49 | 5.06 | 5.53 |
| 10..25% | 82 | 0.99 | 13.05 | 14.64 |
| >25% | 75 | 3.15 | 22.65 | 22.92 |

![03_catalysts](figures/03_catalysts.png)

## 4. Sectors, industries and themes

By sector (SimFin classification, yfinance fallback; 'Unknown' = no classification, mostly delisted names):

| sector | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| Technology | 592 | 45.61 | 8.78 | 0.82 | -0.35 | 1.34 | 4.83 | 139.63 |
| Energy | 105 | 34.29 | 9.52 | 2.15 | -0.63 | 1.93 | 2.26 | 65.43 |
| Industrials | 301 | 41.20 | 7.64 | 0.47 | -0.75 | 1.20 | 1.42 | 41.20 |
| Healthcare | 703 | 42.53 | 4.69 | 0.10 | -0.50 | 1.05 | 0.71 | 20.62 |
| Real Estate | 68 | 25.00 | 4.41 | 0.03 | -0.76 | 1.02 | 0.02 | 0.64 |
| Communication Services | 32 | 37.50 | 9.38 | -0.38 | -1.47 | 0.89 | -0.12 | -3.56 |
| Financial Services | 157 | 34.39 | 7.01 | -0.08 | -0.71 | 0.96 | -0.13 | -3.81 |
| Utilities | 34 | 41.18 | 2.94 | -0.91 | -1.04 | 0.65 | -0.31 | -8.95 |
| Consumer Cyclical | 244 | 31.97 | 4.92 | -0.42 | -0.86 | 0.80 | -1.02 | -29.55 |
| Consumer Defensive | 62 | 27.42 | 1.61 | -1.73 | -0.97 | 0.30 | -1.07 | -31.05 |
| Basic Materials | 177 | 41.24 | 2.82 | -0.77 | -0.68 | 0.64 | -1.36 | -39.42 |
| Unknown | 90 | 40.00 | 3.33 | -1.97 | -0.53 | 0.50 | -1.78 | -51.41 |

Over/under-representation of sectors among big winners (Fisher, BH-adjusted):

| sector | n | big_rate% | base_rate% | odds_ratio | p | q_BH |
|---|---|---|---|---|---|---|
| Technology | 592 | 8.784 | 6.076 | 1.731 | 0.002 | 0.028 |
| Basic Materials | 177 | 2.825 | 6.076 | 0.431 | 0.071 | 0.315 |
| Healthcare | 703 | 4.694 | 6.076 | 0.698 | 0.079 | 0.315 |
| Energy | 105 | 9.524 | 6.076 | 1.670 | 0.141 | 0.423 |
| Consumer Defensive | 62 | 1.613 | 6.076 | 0.249 | 0.179 | 0.429 |
| Industrials | 301 | 7.641 | 6.076 | 1.327 | 0.247 | 0.493 |
| Unknown | 90 | 3.333 | 6.076 | 0.524 | 0.369 | 0.632 |
| Communication Services | 32 | 9.375 | 6.076 | 1.611 | 0.440 | 0.645 |
| Consumer Cyclical | 244 | 4.918 | 6.076 | 0.783 | 0.484 | 0.645 |
| Financial Services | 157 | 7.006 | 6.076 | 1.177 | 0.604 | 0.725 |
| Utilities | 34 | 2.941 | 6.076 | 0.465 | 0.720 | 0.785 |
| Real Estate | 68 | 4.412 | 6.076 | 0.708 | 0.796 | 0.796 |

Top industries by summed return (min 15 trades):

| industry | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| Software - Infrastructure | 64 | 54.69 | 14.06 | 3.61 | 0.24 | 2.87 | 2.31 | 44.16 |
| Oil & Gas - E&P | 26 | 34.62 | 11.54 | 5.99 | -0.50 | 4.78 | 1.56 | 29.78 |
| Computer Hardware | 123 | 50.41 | 12.20 | 1.04 | 0.12 | 1.42 | 1.27 | 24.33 |
| Software - Application | 46 | 45.65 | 21.74 | 2.20 | -0.08 | 1.64 | 1.01 | 19.38 |
| Travel & Leisure | 42 | 38.10 | 14.29 | 2.14 | -0.72 | 1.96 | 0.90 | 17.20 |
| Biotechnology | 487 | 44.97 | 4.31 | 0.17 | -0.45 | 1.10 | 0.84 | 16.14 |
| Drug Manufacturers | 45 | 44.44 | 8.89 | 1.60 | -0.50 | 2.22 | 0.72 | 13.76 |
| Semiconductors | 75 | 50.67 | 6.67 | 0.91 | 0.00 | 1.35 | 0.68 | 13.05 |
| Application Software | 194 | 43.30 | 5.15 | 0.32 | -0.31 | 1.17 | 0.62 | 11.77 |
| Real Estate Services | 18 | 44.44 | 16.67 | 3.03 | -0.35 | 2.32 | 0.54 | 10.41 |
| Business Services | 16 | 62.50 | 18.75 | 2.81 | 0.56 | 2.58 | 0.45 | 8.60 |
| Internet Content & Information | 18 | 55.56 | 16.67 | 2.49 | 0.36 | 1.94 | 0.45 | 8.58 |

Worst industries by summed return:

| industry | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| REITs | 46 | 19.57 | 0.00 | -0.75 | -0.77 | 0.29 | -0.35 | -6.62 |
| Restaurants | 27 | 37.04 | 0.00 | -1.39 | -0.56 | 0.22 | -0.38 | -7.18 |
| Entertainment | 34 | 26.47 | 0.00 | -1.15 | -1.00 | 0.27 | -0.39 | -7.47 |
| Communication Equipment | 33 | 33.33 | 0.00 | -1.21 | -0.98 | 0.38 | -0.40 | -7.64 |
| Engineering & Construction | 19 | 21.05 | 0.00 | -2.30 | -1.22 | 0.12 | -0.44 | -8.35 |
| Asset Management | 39 | 28.21 | 5.13 | -1.24 | -0.56 | 0.59 | -0.48 | -9.27 |
| Medical Devices | 51 | 31.37 | 3.92 | -1.01 | -0.69 | 0.61 | -0.51 | -9.83 |
| Retail - Apparel & Specialty | 49 | 30.61 | 2.04 | -1.28 | -0.95 | 0.46 | -0.63 | -12.00 |

**Same-day sector sympathy.** Pairs of big winners in the same sector on the same day: 18 observed vs 10.5 expected if sectors were assigned at random among big winners on each day (permutation p = 0.0210).

Strongest sector 'waves' (weeks with the most big winners in one sector):

| week | sector | n | tickers |
|---|---|---|---|
| 2024-12-21 | Technology | 4 | INTZ, LAES, PDYN, RZLV |
| 2024-12-14 | Technology | 4 | RZLV, VRAR, QUBT, QBTS |
| 2024-12-28 | Technology | 3 | CRNC, ARBE, IRBT |
| 2025-03-08 | Technology | 3 | QUBT, QMCO, QBTS |
| 2026-05-23 | Technology | 3 | MASK, APPS, CPSH |
| 2026-02-28 | Energy | 3 | BATL, TPET, TMDE |
| 2025-10-04 | Healthcare | 3 | BMEA, RVPH, TNYA |
| 2025-05-31 | Healthcare | 2 | EYEN, MODV |
| 2025-02-08 | Technology | 2 | AIFF, AIFF |
| 2025-06-21 | Industrials | 2 | SRFM, SRFM |
| 2025-07-12 | Healthcare | 2 | IXHL, IXHL |
| 2025-01-04 | Healthcare | 2 | PBYI, CVAC |

![04_sector_heatmap](figures/04_sector_heatmap.png)

![04_sector_bars](figures/04_sector_bars.png)

## 4b. Themes found from company descriptions (TF-IDF + NMF topic model)

Each company's business description was converted to TF-IDF features and factorized into 14 topics; each company is assigned to its dominant topic (2366 trades with a confident assignment). Topics ranked by summed trade return:

| topic_label | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| T12: ai, data, cloud, software, solutions | 410 | 46.10 | 9.76 | 1.14 | -0.29 | 1.49 | 4.67 | 93.04 |
| T3: oil, gas, natural gas, natural, crude | 84 | 35.71 | 7.14 | 1.94 | -0.47 | 2.06 | 1.63 | 32.54 |
| T1: treatment, clinical, phase, therapeutics, trial | 477 | 45.07 | 4.61 | 0.16 | -0.29 | 1.08 | 0.75 | 14.93 |
| T4: gold, silver, located, canada, exploration | 140 | 43.57 | 4.29 | 0.51 | -0.50 | 1.26 | 0.71 | 14.10 |
| T0: media, content, digital, advertising, video | 65 | 46.15 | 7.69 | 0.72 | -0.29 | 1.44 | 0.47 | 9.29 |
| T2: loans, banking, insurance, credit, financial | 52 | 38.46 | 3.85 | 0.34 | -0.69 | 1.26 | 0.18 | 3.57 |
| T7: real estate, estate, real, trust, investment | 64 | 29.69 | 7.81 | 0.22 | -0.85 | 1.11 | 0.14 | 2.79 |
| T11: million, investments, debt, companies, fund | 34 | 32.35 | 0.00 | -0.18 | -0.53 | 0.87 | -0.06 | -1.23 |
| T13: systems, solutions, equipment, segment, industrial | 401 | 40.65 | 4.99 | -0.05 | -0.64 | 0.98 | -0.18 | -3.62 |
| T6: new york, york, new, york new, share | 94 | 32.98 | 11.70 | -0.37 | -1.53 | 0.88 | -0.35 | -6.91 |
| T9: energy, power, solar, storage, energy storage | 165 | 41.21 | 5.45 | -0.26 | -1.06 | 0.90 | -0.44 | -8.69 |
| T5: stores, accessories, brands, apparel, retail | 166 | 30.72 | 4.22 | -0.30 | -0.88 | 0.83 | -0.50 | -9.96 |
| T10: health, care, medical, healthcare, insurance | 118 | 38.98 | 5.93 | -0.72 | -0.69 | 0.70 | -0.85 | -16.90 |
| T8: limited, china, people republic, republic china, republic | 96 | 36.46 | 6.25 | -1.20 | -1.05 | 0.64 | -1.15 | -22.96 |

Topic enrichment among big winners (Fisher, BH-adjusted):

| topic | n | big_rate% | odds_ratio | p | q_BH |
|---|---|---|---|---|---|
| T12: ai, data, cloud, software, solutions | 410 | 9.756 | 1.887 | 0.002 | 0.021 |
| T6: new york, york, new, york new, share | 94 | 11.702 | 2.098 | 0.044 | 0.308 |
| T1: treatment, clinical, phase, therapeutics, trial | 477 | 4.612 | 0.688 | 0.135 | 0.632 |
| T11: million, investments, debt, companies, fund | 34 | 0.000 | 0.000 | 0.267 | 0.860 |
| T13: systems, solutions, equipment, segment, industrial | 401 | 4.988 | 0.766 | 0.307 | 0.860 |
| T5: stores, accessories, brands, apparel, retail | 166 | 4.217 | 0.653 | 0.400 | 0.901 |
| T4: gold, silver, located, canada, exploration | 140 | 4.286 | 0.667 | 0.467 | 0.901 |
| T7: real estate, estate, real, trust, investment | 64 | 7.812 | 1.299 | 0.593 | 0.901 |
| T0: media, content, digital, advertising, video | 65 | 7.692 | 1.277 | 0.596 | 0.901 |
| T3: oil, gas, natural gas, natural, crude | 84 | 7.143 | 1.177 | 0.644 | 0.901 |
| T2: loans, banking, insurance, credit, financial | 52 | 3.846 | 0.603 | 0.769 | 0.978 |
| T9: energy, power, solar, storage, energy storage | 165 | 5.455 | 0.869 | 0.867 | 1.000 |
| T10: health, care, medical, healthcare, insurance | 118 | 5.932 | 0.957 | 1.000 | 1.000 |
| T8: limited, china, people republic, republic china, republic | 96 | 6.250 | 1.014 | 1.000 | 1.000 |

Strongest theme waves (weeks with the most big winners sharing a topic):

| week | topic_label | n | tickers |
|---|---|---|---|
| 2024-12-21 | T12: ai, data, cloud, software, solutions | 5 | INTZ, LAES, PDYN, RZLV, RR |
| 2024-12-14 | T12: ai, data, cloud, software, solutions | 4 | SIDU, RZLV, QUBT, QBTS |
| 2026-05-23 | T13: systems, solutions, equipment, segment, industrial | 4 | MNTS, SPCE, CPSH, CRSR |
| 2025-03-08 | T12: ai, data, cloud, software, solutions | 3 | QUBT, QMCO, QBTS |
| 2025-10-04 | T1: treatment, clinical, phase, therapeutics, trial | 3 | BMEA, RVPH, TNYA |
| 2024-12-28 | T12: ai, data, cloud, software, solutions | 2 | RR, CRNC |
| 2025-07-19 | T13: systems, solutions, equipment, segment, industrial | 2 | LIDR, UAVS |
| 2025-07-12 | T1: treatment, clinical, phase, therapeutics, trial | 2 | IXHL, IXHL |
| 2024-11-23 | T13: systems, solutions, equipment, segment, industrial | 2 | OPTT, VUZI |
| 2025-02-08 | T6: new york, york, new, york new, share | 2 | AIFF, AIFF |

![04b_topic_heatmap](figures/04b_topic_heatmap.png)

## 5. Float, size, liquidity and volatility

Rank correlation of each feature with trade return and with being a big winner (BH-adjusted q on the return test):

| feature | n | spearman_vs_ret | p_ret | spearman_vs_big | p_big | q_BH(ret) |
|---|---|---|---|---|---|---|
| shares_pit | 1712 | -0.015 | 0.542 | -0.105 | 0.000 | 0.644 |
| mcap_pit | 1712 | -0.038 | 0.120 | -0.136 | 0.000 | 0.186 |
| float_now | 2318 | 0.053 | 0.010 | -0.083 | 0.000 | 0.037 |
| short_pct_float_now | 2249 | 0.052 | 0.014 | 0.029 | 0.174 | 0.038 |
| price | 2584 | -0.012 | 0.552 | -0.067 | 0.001 | 0.644 |
| dvol20 | 2584 | 0.016 | 0.403 | 0.026 | 0.192 | 0.564 |
| rel_vol | 2584 | -0.038 | 0.056 | 0.152 | 0.000 | 0.106 |
| atr_pct | 2584 | -0.044 | 0.025 | 0.251 | 0.000 | 0.058 |
| sig_day_ret | 2584 | -0.076 | 0.000 | 0.243 | 0.000 | 0.001 |
| gap_open | 2584 | -0.060 | 0.002 | 0.146 | 0.000 | 0.010 |
| range_pct | 2584 | -0.064 | 0.001 | 0.255 | 0.000 | 0.007 |
| run5 | 2584 | -0.007 | 0.741 | 0.105 | 0.000 | 0.741 |
| headroom | 2584 | 0.007 | 0.719 | 0.099 | 0.000 | 0.741 |
| dist_sma200 | 2484 | 0.038 | 0.061 | 0.062 | 0.002 | 0.106 |

**shares_pit** quintiles (shares outstanding (SimFin, point-in-time; float proxy)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (999.999, 44538420.8] | 343 | 42.27 | 12.54 | 1.23 | -0.63 | 1.39 |
| Q2: (44538420.8, 74949593.0] | 343 | 43.15 | 2.92 | 0.19 | -0.39 | 1.13 |
| Q3: (74949593.0, 108037852.8] | 341 | 39.59 | 1.76 | -0.16 | -0.50 | 0.89 |
| Q4: (108037852.8, 167784378.4] | 342 | 35.09 | 2.92 | -0.41 | -0.68 | 0.76 |
| Q5: (167784378.4, 956601911.0] | 343 | 40.52 | 5.25 | 0.27 | -0.61 | 1.17 |

**mcap_pit** quintiles (market cap (SimFin, point-in-time)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (16999.999, 195260672.072] | 343 | 46.06 | 12.83 | 1.86 | -0.50 | 1.70 |
| Q2: (195260672.072, 431531268.256] | 342 | 39.77 | 3.80 | -0.27 | -0.58 | 0.87 |
| Q3: (431531268.256, 720057743.44] | 346 | 43.35 | 1.73 | -0.08 | -0.42 | 0.95 |
| Q4: (720057743.44, 1164916655.376] | 338 | 37.57 | 4.44 | -0.09 | -0.59 | 0.95 |
| Q5: (1164916655.376, 1998805547.0] | 343 | 33.82 | 2.62 | -0.31 | -0.74 | 0.79 |

**float_now** quintiles (float (yfinance current snapshot, not point-in-time)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (-0.001, 36136153.0] | 468 | 39.32 | 12.82 | 0.18 | -1.00 | 1.04 |
| Q2: (36136153.0, 64641423.2] | 459 | 42.27 | 2.83 | -0.04 | -0.63 | 0.98 |
| Q3: (64641423.2, 101027937.0] | 471 | 42.46 | 5.73 | 0.48 | -0.42 | 1.28 |
| Q4: (101027937.0, 163760352.0] | 457 | 39.61 | 3.06 | -0.18 | -0.67 | 0.89 |
| Q5: (163760352.0, 7475818960.0] | 463 | 41.68 | 6.48 | 0.68 | -0.56 | 1.40 |

**short_pct_float_now** quintiles (short % of float (current snapshot)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (-0.0009, 0.0524] | 451 | 35.70 | 5.10 | -0.67 | -0.88 | 0.76 |
| Q2: (0.0524, 0.0938] | 453 | 36.42 | 5.52 | -0.18 | -0.82 | 0.92 |
| Q3: (0.0938, 0.145] | 445 | 44.49 | 6.97 | 0.94 | -0.42 | 1.52 |
| Q4: (0.145, 0.2] | 450 | 48.00 | 6.44 | 0.71 | -0.20 | 1.40 |
| Q5: (0.2, 0.745] | 450 | 40.44 | 7.56 | 0.47 | -0.89 | 1.21 |

**price** quintiles (price at 15:55):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (1.999, 3.168] | 517 | 42.17 | 8.51 | 0.59 | -0.60 | 1.23 |
| Q2: (3.168, 5.0] | 518 | 41.31 | 7.34 | 0.14 | -0.63 | 1.06 |
| Q3: (5.0, 7.728] | 515 | 42.91 | 5.63 | 0.18 | -0.59 | 1.08 |
| Q4: (7.728, 11.852] | 517 | 36.56 | 4.45 | -0.01 | -0.56 | 1.00 |
| Q5: (11.852, 20.0] | 517 | 37.72 | 4.45 | -0.23 | -0.65 | 0.89 |

**rel_vol** quintiles (relative volume at 15:55):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (1.999, 2.258] | 517 | 38.49 | 2.71 | -0.16 | -0.57 | 0.90 |
| Q2: (2.258, 2.657] | 517 | 41.59 | 3.87 | 0.33 | -0.50 | 1.22 |
| Q3: (2.657, 3.347] | 516 | 38.57 | 3.68 | -0.30 | -0.62 | 0.85 |
| Q4: (3.347, 5.418] | 517 | 42.36 | 5.61 | -0.26 | -0.64 | 0.88 |
| Q5: (5.418, 501.735] | 517 | 39.65 | 14.51 | 1.05 | -0.85 | 1.27 |

**atr_pct** quintiles (daily ATR % of price):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (0.00145, 0.0452] | 517 | 32.30 | 0.77 | -0.53 | -0.53 | 0.52 |
| Q2: (0.0452, 0.0612] | 517 | 40.81 | 2.32 | -0.09 | -0.50 | 0.94 |
| Q3: (0.0612, 0.0762] | 516 | 43.99 | 2.13 | -0.40 | -0.50 | 0.78 |
| Q4: (0.0762, 0.0999] | 517 | 41.20 | 5.42 | -0.06 | -0.91 | 0.97 |
| Q5: (0.0999, 2.229] | 517 | 42.36 | 19.73 | 1.75 | -1.08 | 1.39 |

**sig_day_ret** quintiles (signal-day return to 15:55):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (0.0006399999999999999, 0.087] | 517 | 35.59 | 1.93 | -0.24 | -0.50 | 0.80 |
| Q2: (0.087, 0.135] | 517 | 40.81 | 1.55 | -0.28 | -0.53 | 0.81 |
| Q3: (0.135, 0.192] | 516 | 43.99 | 1.74 | -0.36 | -0.50 | 0.78 |
| Q4: (0.192, 0.304] | 517 | 40.04 | 6.58 | 0.38 | -0.84 | 1.18 |
| Q5: (0.304, 6.375] | 517 | 40.23 | 18.57 | 1.17 | -1.74 | 1.24 |

**dvol20** quintiles (20-day avg dollar volume):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (280166.41000000003, 3214615.434] | 517 | 42.17 | 7.16 | 0.15 | -0.66 | 1.06 |
| Q2: (3214615.434, 6333639.2] | 517 | 40.23 | 4.26 | -0.39 | -0.50 | 0.82 |
| Q3: (6333639.2, 10852263.562] | 516 | 34.69 | 5.23 | -0.14 | -0.70 | 0.93 |
| Q4: (10852263.562, 22843590.848] | 517 | 41.20 | 4.26 | -0.16 | -0.50 | 0.92 |
| Q5: (22843590.848, 737407869.12] | 517 | 42.36 | 9.48 | 1.21 | -0.60 | 1.51 |

**headroom** quintiles (headroom to next resistance (13m)):

| bin | n | win% | big% | mean% | median% | PF |
|---|---|---|---|---|---|---|
| Q1: (-0.0009825, 0.0173] | 517 | 36.75 | 3.68 | -0.48 | -0.63 | 0.74 |
| Q2: (0.0173, 0.0385] | 517 | 38.49 | 2.51 | -0.42 | -0.59 | 0.77 |
| Q3: (0.0385, 0.0699] | 516 | 42.25 | 5.81 | 0.08 | -0.57 | 1.04 |
| Q4: (0.0699, 0.226] | 517 | 40.62 | 8.12 | 0.40 | -0.82 | 1.15 |
| Q5: (0.226, 2.0] | 517 | 42.55 | 10.25 | 1.08 | -0.50 | 1.39 |

![05_quintiles](figures/05_quintiles.png)

By shares outstanding (SimFin, millions):

| size | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| <10M sh | 70 | 44.29 | 24.29 | 1.82 | -1.80 | 1.33 | 1.27 | 33.20 |
| 10-30M | 142 | 45.07 | 15.49 | 3.08 | -0.24 | 2.36 | 4.37 | 114.12 |
| 30-100M | 730 | 41.92 | 2.60 | -0.08 | -0.50 | 0.95 | -0.59 | -15.34 |
| 100-300M | 662 | 37.01 | 3.63 | -0.26 | -0.68 | 0.84 | -1.73 | -45.22 |
| >300M | 108 | 37.96 | 4.63 | 0.47 | -0.82 | 1.28 | 0.51 | 13.24 |

## 6. Link to the broader market

Daily mean trade return vs market variables (476 trading days; '*_gap' = next-session close-to-open move of the index, known only afterwards; '*_day' = the index move from the prior close to 15:55 on entry day):

| market variable | spearman_vs_daily_mean_ret | p |
|---|---|---|
| spy_gap | 0.217 | 0.000 |
| iwm_gap | 0.227 | 0.000 |
| qqq_gap | 0.223 | 0.000 |
| spy_day | 0.046 | 0.317 |
| iwm_day | 0.063 | 0.172 |
| iwm_ret5 | -0.032 | 0.482 |
| iwm_ret20 | -0.014 | 0.754 |
| spy_ret20 | -0.004 | 0.925 |

**Overnight beta.** OLS with Newey-West errors, daily mean trade return on IWM and SPY overnight gaps: intercept (alpha) +0.15% per day (t = 0.59); beta to IWM gap 0.20 (t = 0.34); beta to SPY gap 0.68 (t = 0.78); R2 = 0.011.

Trade outcomes by tercile of market variable:

| variable | b | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|---|
| IWM next-session gap | low | 862 | 29.58 | 5.68 | -0.66 | -1.39 | 0.77 | -5.70 | -164.80 |
| IWM next-session gap | mid | 863 | 39.05 | 5.10 | 0.01 | -0.56 | 1.01 | 0.11 | 3.11 |
| IWM next-session gap | high | 859 | 51.80 | 7.45 | 1.05 | 0.12 | 1.57 | 9.05 | 261.69 |
| IWM entry-day move | low | 863 | 39.98 | 7.42 | 0.24 | -0.65 | 1.09 | 2.06 | 59.49 |
| IWM entry-day move | mid | 860 | 38.02 | 5.35 | -0.27 | -0.71 | 0.88 | -2.35 | -67.87 |
| IWM entry-day move | high | 861 | 42.39 | 5.46 | 0.44 | -0.50 | 1.23 | 3.75 | 108.38 |
| SPY entry-day move | low | 862 | 38.98 | 7.19 | -0.02 | -0.80 | 0.99 | -0.21 | -6.21 |
| SPY entry-day move | mid | 865 | 44.62 | 6.47 | 0.53 | -0.39 | 1.26 | 4.59 | 132.74 |
| SPY entry-day move | high | 857 | 36.76 | 4.55 | -0.11 | -0.66 | 0.95 | -0.92 | -26.53 |
| IWM 20-day trend | low | 863 | 38.47 | 6.49 | 0.23 | -0.63 | 1.11 | 1.96 | 56.83 |
| IWM 20-day trend | mid | 864 | 41.78 | 6.25 | 0.37 | -0.54 | 1.16 | 3.22 | 93.14 |
| IWM 20-day trend | high | 857 | 40.14 | 5.48 | -0.20 | -0.64 | 0.91 | -1.73 | -49.97 |

**Breadth ('hype') indicator.** The number of candidate breakouts that day relative to its prior 20-day average: Spearman vs that day's mean trade return +0.079 (p = 0.087); vs number of big winners +0.206.

![06_market](figures/06_market.png)

### 6b. How much of the result is just the market's overnight move?

| IWM hedge ratio | mean% | median% | win% | PF | daily_Sharpe_ann | NW_t_daily |
|---|---|---|---|---|---|---|
| 0.00 | 0.13 | -0.60 | 40.13 | 1.06 | 0.49 | 0.69 |
| 0.50 | 0.12 | -0.64 | 39.47 | 1.05 | 0.45 | 0.62 |
| 1.00 | 0.11 | -0.65 | 39.16 | 1.05 | 0.40 | 0.55 |

A hedge ratio of 1 removes the IWM close-to-open move entirely. The trades hold some positions through the open only partly (several exit pre-market or after-hours), so this is an approximation; it shows whether the edge survives once index direction is removed.

## 7. Can the winners be predicted? (gradient boosting, purged walk-forward)

HistGradientBoosting (depth 3, strong regularization) on 32 point-in-time features known at 15:55; expanding monthly folds with a 5-business-day purge before each test month; final model trained on all dev and scored on the locked test period. AUC 0.5 = no skill.

| target | dev_oos_AUC | dev_oos_PR_AUC | base_rate% | dev_oos_top10%_lift | test_AUC | test_PR_AUC | test_base_rate% | test_top20%_lift | n_dev_oos | n_test |
|---|---|---|---|---|---|---|---|---|---|---|
| big | 0.773 | 0.191 | 5.349 | 4.542 | 0.803 | 0.357 | 5.466 | 3.541 | 1776 | 311 |
| winner | 0.492 | 0.392 | 39.977 | 0.975 | 0.570 | 0.451 | 39.228 | 1.151 | 1776 | 311 |

Permutation importance (drop in AUC when a feature is shuffled inside each walk-forward test fold; big-winner target):

| index | mean_AUC_drop | sd_across_folds | folds_positive% |
|---|---|---|---|
| atr_pct | 0.0194 | 0.0311 | 80.0000 |
| dist_ema20 | 0.0159 | 0.0305 | 73.3333 |
| shares_pit | 0.0133 | 0.0457 | 66.6667 |
| dist_ema9 | 0.0090 | 0.0224 | 73.3333 |
| price | 0.0083 | 0.0345 | 60.0000 |
| dist_vwap | 0.0049 | 0.0340 | 46.6667 |
| range_pct | 0.0038 | 0.0286 | 46.6667 |
| day_dvol | 0.0017 | 0.0330 | 60.0000 |
| sig_day_ret | 0.0004 | 0.0185 | 60.0000 |
| qqq_day | 0.0004 | 0.0075 | 60.0000 |
| iwm_sma200_up | 0.0000 | 0.0000 | 0.0000 |
| spy_sma50_up | 0.0000 | 0.0000 | 0.0000 |
| iwm_sma50_up | 0.0000 | 0.0000 | 0.0000 |
| news_24h | -0.0009 | 0.0035 | 13.3333 |
| news_5d | -0.0010 | 0.0042 | 26.6667 |

**Remove-and-retrain check** (SHAP and permutation importance describe what a model fitted, not what generalizes): walk-forward AUC with all features 0.773; without the top 3 (atr_pct, dist_ema20, shares_pit) 0.755; with only those 3 0.802.

![07_partial_dependence](figures/07_partial_dependence.png)

A depth-3 decision tree on dev (readable splits; unstable, shown for interpretation only):

```
|--- dist_ema20 <= 0.765
|   |--- atr_pct <= 0.175
|   |   |--- range_pct <= 0.310
|   |   |   |--- class: False
|   |   |--- range_pct >  0.310
|   |   |   |--- class: False
|   |--- atr_pct >  0.175
|   |   |--- class: False
|--- dist_ema20 >  0.765
|   |--- from_high <= -0.062
|   |   |--- class: False
|   |--- from_high >  -0.062
|   |   |--- class: True
```

## 8. Rule mining with multiple-testing control

6739 candidate rules (single and pairs of quartile/median conditions) with at least 60 dev trades: 6588 tested for a higher big-winner rate than the 6.2% base (one-sided binomial). 1971 pass BH-FDR q < 0.10 on dev.

Strongest rules by dev p-value, with their out-of-sample behaviour on the locked test period (test base big-winner rate 5.5%):

| rule | n | big_rate% | big_loss_rate% | lift | mean_ret% | PF | p | q_BH | test_n | test_big_rate% | test_mean_ret% | test_big_loss% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sig_day_ret >= 0.264 AND atr_pct >= 0.092 | 318 | 26.4 | 26.7 | 4.29 | 2.61 | 1.49 | 2.44e-30 | 1.61e-26 | 55 | 20 | -1.35 | 43.6 |
| atr_pct >= 0.092 AND dist_ema9 >= 0.252 | 302 | 26.2 | 26.8 | 4.25 | 2.92 | 1.56 | 2.5e-28 | 8.24e-25 | 47 | 19.1 | -2.22 | 46.8 |
| sig_day_ret >= 0.264 AND atr_pct >= 0.0687 | 447 | 21.7 | 21.9 | 3.52 | 1.83 | 1.39 | 1.75e-27 | 3.85e-24 | 66 | 16.7 | -1.42 | 39.4 |
| sig_day_ret >= 0.264 AND range_pct >= 0.242 | 439 | 21.6 | 22.3 | 3.51 | 1.95 | 1.41 | 7.42e-27 | 1.22e-23 | 58 | 19 | -1.11 | 41.4 |
| atr_pct >= 0.092 AND dist_ema9 >= 0.16 | 425 | 21.9 | 20.9 | 3.55 | 2.42 | 1.55 | 1.05e-26 | 1.39e-23 | 56 | 21.4 | -1.15 | 42.9 |
| atr_pct >= 0.092 AND dist_ema20 >= 0.325 | 300 | 25.3 | 26.3 | 4.11 | 2.8 | 1.53 | 2.37e-26 | 2.6e-23 | 37 | 16.2 | -3.99 | 48.6 |
| range_pct >= 0.242 AND dist_ema9 >= 0.252 | 400 | 22.2 | 24 | 3.61 | 2.19 | 1.45 | 3.57e-26 | 3.36e-23 | 48 | 16.7 | -2.71 | 47.9 |
| atr_pct >= 0.092 AND range_pct >= 0.242 | 355 | 23.1 | 23.7 | 3.75 | 2.37 | 1.49 | 2.17e-25 | 1.78e-22 | 56 | 19.6 | -0.862 | 39.3 |
| atr_pct >= 0.092 AND day_dvol >= 9.05e+07 | 228 | 28.1 | 23.2 | 4.56 | 3.99 | 1.83 | 4.25e-25 | 3.11e-22 | 41 | 22 | 0.336 | 43.9 |
| sig_day_ret >= 0.264 AND dist_ema9 >= 0.252 | 431 | 21.1 | 22.3 | 3.43 | 1.9 | 1.4 | 5.46e-25 | 3.59e-22 | 56 | 14.3 | -2.52 | 42.9 |
| atr_pct >= 0.092 AND day_dvol >= 3.7e+07 | 346 | 23.1 | 22 | 3.75 | 2.33 | 1.51 | 7.65e-25 | 4.58e-22 | 58 | 19 | 0.00228 | 36.2 |
| atr_pct >= 0.092 AND dist_ema20 >= 0.198 | 397 | 21.7 | 21.4 | 3.52 | 2.44 | 1.54 | 1.66e-24 | 9.1e-22 | 51 | 15.7 | -3.33 | 43.1 |
| sig_day_ret >= 0.264 AND dist_ema20 >= 0.325 | 384 | 21.9 | 24.2 | 3.55 | 1.83 | 1.37 | 2.82e-24 | 1.36e-21 | 45 | 11.1 | -3.91 | 44.4 |
| sig_day_ret >= 0.162 AND atr_pct >= 0.092 | 474 | 20 | 19.4 | 3.25 | 1.97 | 1.46 | 2.9e-24 | 1.36e-21 | 77 | 16.9 | -0.713 | 31.2 |
| range_pct >= 0.242 AND dist_ema20 >= 0.325 | 383 | 21.7 | 25.1 | 3.52 | 1.98 | 1.39 | 1.01e-23 | 4.17e-21 | 37 | 13.5 | -4.59 | 51.4 |

## 9. Winner archetypes (clustering)

K-means on 157 big winners (>= +10%) using 10 standardized features; k = 2 chosen by silhouette (0.41).

| cluster | n | mean_ret% | sig_day_ret% | gap_open% | rel_vol | atr% | range% | price | dvol20_$M | run5% | headroom | top_sectors | examples |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 132 | 23.99 | 32.73 | 8.22 | 3.97 | 10.90 | 33.29 | 5.00 | 11.19 | 13.72 | 0.09 | Technology (37), Healthcare (30) | BATL, BMEA, EYEN, INDO, BATL |
| 1 | 25 | 30.81 | 177.74 | 36.07 | 62.48 | 15.30 | 146.36 | 4.29 | 2.45 | 3.93 | 0.14 | Technology (15), Industrials (5) | INTZ, JLHL, AZIO, ICCT, GWH |

Share of ALL trades falling in each archetype's region of feature space (nearest centroid) and how those trades fared:

| arch | n | win% | big% | mean% | median% | PF | sum_ret | share_of_total_profit% |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 2503.00 | 40.35 | 5.27 | 0.11 | -0.57 | 1.06 | 2.82 | 81.58 |
| 1.00 | 81.00 | 33.33 | 30.86 | 0.79 | -11.05 | 1.09 | 0.64 | 18.42 |

## 10. Event study: the path around entry

Mean cumulative return versus IWM (%, winsorized 1%) relative to the close before the entry day, by final trade outcome. Day 0 is the entry day (entry at 15:55, so day 0 shows the run-up the strategy buys into), day +1 the overnight session:

| bucket | n | d-5 | d-4 | d-3 | d-2 | d-1 | d0 | d1 | d2 | d3 |
|---|---|---|---|---|---|---|---|---|---|---|
| loser <-5% | 347 | -8.34 | -8.06 | -6.82 | -5.32 | 0.00 | 53.16 | 36.07 | 35.15 | 34.76 |
| small loss | 1191 | -2.12 | -1.98 | -1.66 | -1.14 | 0.00 | 16.34 | 15.06 | 15.31 | 15.21 |
| winner 0-10% | 880 | -3.52 | -3.50 | -2.98 | -1.85 | 0.00 | 17.11 | 20.09 | 20.26 | 20.42 |
| big winner >=10% | 157 | -10.52 | -11.06 | -8.87 | -4.04 | 0.00 | 68.63 | 96.86 | 88.88 | 85.65 |

![10_event_study](figures/10_event_study.png)

## 11. Statistical rigor: is any of this skill?

Probabilistic / deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) on dev daily portfolio returns. 'N' is the number of strategy variants tried; N=1 is the plain probabilistic Sharpe ratio, larger N deflates for selection bias. This project has tried on the order of 100 variants (exit rules, filters, sizing).

| strategy | days | daily_SR | annual_SR | skew | kurtosis | NW_t(mean) | P(skill) N=1 | P(skill) N=30 | P(skill) N=100 | P(skill) N=300 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 432 | 0.059 | 0.935 | 1.790 | 9.603 | 1.084 | 0.901 | 0.216 | 0.107 | 0.054 |
| S1 | 432 | 0.108 | 1.717 | 3.848 | 28.932 | 2.302 | 0.997 | 0.751 | 0.588 | 0.443 |
| S2 | 432 | 0.112 | 1.772 | 3.634 | 28.095 | 2.327 | 0.998 | 0.770 | 0.611 | 0.467 |
| H3 | 432 | 0.094 | 1.497 | 5.692 | 57.065 | 1.570 | 0.995 | 0.684 | 0.509 | 0.366 |
| H2 | 432 | 0.073 | 1.154 | 5.178 | 58.361 | 1.493 | 0.964 | 0.394 | 0.234 | 0.138 |

## 12. Do the main findings hold out of sample (dev vs the locked test period)?

| group | period | n | win% | big% | big_loss% | mean% | PF |
|---|---|---|---|---|---|---|---|
| all trades | dev | 2273 | 40.26 | 6.16 | 5.90 | 0.22 | 1.10 |
| all trades | test | 311 | 39.23 | 5.47 | 9.00 | -0.48 | 0.81 |
| shares outstanding < 30M | dev | 187 | 43.32 | 18.18 | 13.90 | 2.70 | 1.78 |
| shares outstanding < 30M | test | 25 | 56.00 | 20.00 | 4.00 | 2.41 | 2.08 |
| market cap < $200M | dev | 305 | 45.90 | 12.79 | 10.82 | 1.99 | 1.72 |
| market cap < $200M | test | 47 | 53.19 | 10.64 | 2.13 | 1.12 | 1.66 |
| daily ATR >= 9% | dev | 597 | 42.21 | 16.25 | 16.08 | 1.48 | 1.38 |
| daily ATR >= 9% | test | 87 | 36.78 | 14.94 | 29.89 | -0.99 | 0.82 |
| signal-day move >= +26% | dev | 589 | 40.58 | 16.98 | 18.00 | 1.19 | 1.29 |
| signal-day move >= +26% | test | 79 | 32.91 | 13.92 | 32.91 | -1.14 | 0.81 |
| rel. volume >= 5.4 | dev | 455 | 40.88 | 14.73 | 16.26 | 1.32 | 1.35 |
| rel. volume >= 5.4 | test | 66 | 30.30 | 12.12 | 28.79 | -1.00 | 0.81 |
| price below 200 SMA | dev | 638 | 37.30 | 5.96 | 6.11 | -0.14 | 0.94 |
| price below 200 SMA | test | 117 | 38.46 | 8.55 | 17.09 | -1.18 | 0.68 |
| price above 200 SMA | dev | 1548 | 41.80 | 6.33 | 5.68 | 0.42 | 1.20 |
| price above 200 SMA | test | 181 | 40.88 | 3.31 | 3.87 | 0.01 | 1.00 |
| headroom >= 20% | dev | 487 | 42.09 | 10.47 | 10.47 | 1.15 | 1.41 |
| headroom >= 20% | test | 59 | 40.68 | 10.17 | 10.17 | 1.46 | 1.51 |
| market cap < $200M and headroom >= 20% | dev | 72 | 45.83 | 16.67 | 16.67 | 3.42 | 1.97 |
| market cap < $200M and headroom >= 20% | test | 13 | 38.46 | 7.69 | 0.00 | -0.08 | 0.97 |
| shares < 30M and headroom >= 20% | dev | 43 | 41.86 | 25.58 | 20.93 | 5.91 | 2.31 |
| shares < 30M and headroom >= 20% | test | 7 | 28.57 | 14.29 | 0.00 | 0.74 | 1.21 |
| technology sector | dev | 547 | 45.89 | 8.96 | 6.22 | 0.99 | 1.43 |
| technology sector | test | 45 | 42.22 | 6.67 | 17.78 | -1.32 | 0.63 |
| earnings-linked (filing 0-3d) | dev | 104 | 43.27 | 1.92 | 7.69 | -1.22 | 0.53 |
| company news in prior 24h | dev | 247 | 39.68 | 9.31 | 8.50 | 0.32 | 1.12 |
| company news in prior 24h | test | 20 | 20.00 | 5.00 | 20.00 | -2.37 | 0.42 |


## Methods and sources

- **Overnight-return literature.** Lou, Polk and Skouras ("A Tug of War: Overnight Versus Intraday Expected Returns") find that momentum returns accrue overnight; Gamm ("A Surprise That Keeps You Awake") documents attention-driven overnight buying after large earnings surprises; Gregoire (Journal of Accounting Research, 2022) shows after-hours prices move through quotes more than trades, so trade-price returns understate and delay the move. Our entries are momentum names held overnight, and our fills use trade bars, so realized fills are optimistic compared to quote-based reality.
- **Selection-bias control.** Bailey and Lopez de Prado's probabilistic / deflated Sharpe ratio (section 11), Benjamini-Hochberg FDR on every family of subgroup and rule tests (sections 4, 5, 8), and a locked chronological test period (sections 7, 8, 12).
- **Purged walk-forward machine learning.** Expanding monthly folds with an embargo, gradient boosting with permutation importance, and a remove-and-retrain check because attribution methods such as SHAP describe the fitted model, including its tails, and can rank features highly that do not generalize (see recent walk-forward TreeSHAP studies).
- **Clustering and regime tests.** Permutation tests on week-level dispersion of big winners (a simple stand-in for self-exciting / Hawkes clustering), change-point test on mean return, sector sympathy permutation test, NMF topic modelling of company descriptions for themes beyond industry codes, k-means archetypes.
- **Data.** Trades from `data/pit` (delisted names included). Sector/industry and point-in-time shares outstanding, market cap and earnings publish dates from SimFin; sector fallback, current float and short interest (snapshots, not point-in-time) and company descriptions from yfinance; headlines from Alpaca (Benzinga) which cover few small-cap press releases, so absence of news in this report means "not found in this feed".
- **Caveats.** About 40% of trades are winners but 10 trades produce more than the entire net profit; every statistic about big winners rests on 157 trades (75 monsters). Float and short interest from yfinance are today's values. Post-entry news is hindsight and cannot be used as a filter.
