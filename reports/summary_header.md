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

