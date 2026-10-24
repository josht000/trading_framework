# Exit testing report — PARTIAL / STOPPED EARLY

> **Status after the fix pass.** The simulator defects in section 1b have been fixed and only the fixed-time and
> trailing families were re-run (section 8, "Re-run after fixes"). **Every number from the first run (sections 1, 3, 5 and
> the tables in 1a) is invalid, pre-fix** and is kept only for comparison. The momentum, scale-out, pm_ah, combos,
> robustness, learned and optimal-stopping families have NOT been re-run; their old results are invalid too.

This report is intentionally marked **PARTIAL**. The campaign was stopped at the
user's instruction before the six-hour wall-clock budget was exhausted. No
finalists file was written and no new locked-test exit evaluation was performed.
The only test numbers below are the already-generated Phase 0 comparison and
rank summaries, not campaign-selected exit results.

## 1. What ran (first run; results INVALID, pre-fix)

Budget start is recorded in `results/exit/budget_start.txt` (`1792501160`).
The last campaign checkpoint in `results/exit/elapsed.jsonl` is
`21007.0` seconds (350.1 minutes); the live progress document estimated about
12.9 minutes remained when the campaign was stopped. Process inspection after
the stop showed no Python campaign or progress-watcher processes.

The campaign used the cached 2,687-window NPZ and evaluated 2,375 dev windows
for search. Trial counts on disk are:

| family | completed trials | status | best dev CV J |
|---|---:|---|---:|
| fixed-time | 11 | complete | 0.005447 |
| trailing | 200 | complete | 0.001504 |
| momentum | 200 | complete | 0.002490 |
| scale-out | 150 | complete | -0.000278 |
| overnight/AH + PM | 100 | complete | 0.001745 |
| combos | 200 | complete | 0.002361 |
| learned GBM | 20 | complete | -0.007210 |
| Longstaff-style Ridge | 20 | complete | -0.007826 |
| robustness batches 00–24 | 5,000 | complete | 0.003110 (batch 06) |
| robustness batch 25 | 100 | **incomplete** | not persisted |

Thus 5,901 policy trials were completed and persisted, plus 100 trials attempted in the incomplete batch 25 (6,001 trials attempted in total). The robustness batches are 25 further independent random draws of 200 policies each from the same `combos` parameter distribution (section 1a). They are not adaptive refinements around top performers, and not independent strategies; they were run to use the available budget.

The `learned` family was a real walk-forward `HistGradientBoostingRegressor`
on causal bar features (future max return as the training target), and the
`optimal_stopping` family was a walk-forward Ridge continuation-value
regression in a Longstaff-style backward target setup. They were not selected
for a finalist. The learned table has 17/20 rows passing the script's `valid`
flag; optimal stopping has 20/20. The validity filter is only a coarse guard:
it checks aggregate fold capture and aggregate hit rate, not the minimum
capture separately for each subset, and it does not account for the large
multiple-testing burden.

## 1a. How each family was built (what varied, what was fixed)

Source: `scripts/exit_campaign.py` (`_policy_rows`, `_score`, `evaluate_model_family`) and `src/backtest/exit_sim.py` (`_events`). Parameters are drawn uniformly at random between the stated bounds, one independent draw per trial. There is no search algorithm beyond random sampling (no Optuna, TPE or CMA).

**Common to every family**
- **Trades:** the 2,375 dev entries, each with its cached 15:55 to next-day 11:00 minute window. Entries and entry fills are taken from the trade table and not re-simulated.
- **Fills:** the realistic model, where the order goes out one bar after the decision and fills at that bar's open. It takes at most 10% of each bar's volume, applies 0.15% slippage in regular hours and 0.40% outside them, and charges 0.10% commission per side. Any unfilled remainder is sold at the deadline bar.
- **Decision times:** bar closes from 08:00 on the next day through the deadline. The after-hours window is 16:00 to 16:45, and it is used only where `keep_ah` is true.
- **Scoring (`_score`):**
  - Dev days are split into 5 consecutive date blocks.
  - In each block, the mean net return is taken separately for three subsets and averaged equally: base (all trades), "S1" and "H3".
  - The campaign's S1 mask is only `above_sma200`. It does not include the IWM and SPY up condition of the real S1. H3 is `above_sma200` and headroom of at least 20%.
  - Score J is the mean of the five block values minus 0.5 times their standard deviation.
  - A trial is "valid" if every block has at least 30 trades and the average hit rate is within 2 points of the hit rate of the original v2 trade table.
  - The baseline was never scored in this same measure, so J has no reference value in this campaign.

**Families**

| Family | Trials | What varied (random range) | What was fixed |
|---|---:|---|---|
| fixed_time | 11 | Deadline time: 09:30, 09:35, 09:40, 09:45, 09:50, 09:55, 10:00, 10:05, 10:10, 10:15, 10:30. Not random. | No after-hours rule. Sell on the first regular-session bar. |
| trailing | 200 | `trail` 1% to 16% (drop from the running high that triggers a sell), `activate` 0% to 30% (profit the trade must reach first), `deadline` one of 09:45, 10:00, 10:15, 10:30 | No after-hours rule, no VWAP rule, no velocity rule |
| momentum | 200 | `velocity_floor` -8% to +1% (sell when the 5-bar return is at or below it), `giveback` 1% to 20% (sell when the drop from the running high reaches it), `below_vwap` on or off (sell when the close is below the bar's `vwap` field), `deadline` one of four values | No trailing stop, no after-hours rule |
| scale_out | 150 | `a` 3% to 20% (sell 50% at this gain), second target at a plus 3% to 25% (sell 25% of what remains), `trail` 2% to 12% starting at `a`, `deadline` one of 10:00, 10:15, 10:30 | No after-hours rule |
| pm_ah | 100 | `ah_stop` 8% to 25%, `ah_target` 20% to 70%, `pm_target` 10% to 40%, `pm_stop` 8% to 30% (pre-market, from 08:00), `deadline` one of 09:45, 10:00, 10:15 | After-hours on. No morning momentum or trailing rule, so surviving trades are held to the deadline. |
| combos | 200 | `keep_ah` on or off, `ah_stop` 8% to 20%, `ah_target` 25% to 60%, `pm_target` 15% to 35%, `trail` 2% to 12%, `activate` 0% to 15%, `giveback` 3% to 15%, `below_vwap` on or off, `deadline` one of 10:00, 10:15, 10:30 | No `pm_stop`, no velocity rule |
| robustness 00 to 24 | 5,000 | The same random draws as combos, 25 batches of 200. | As combos |
| learned (gradient boosting) | 20 | `edge_threshold` -0.2% to +3%, L2 regularization 0.01 to 2.0, random seed. | Model: `HistGradientBoostingRegressor` (80 iterations, 15 leaves, learning rate 0.08) on 7 bar features: return since entry, return since the open, peak return, drop from peak, 5-bar velocity, volume ratio, minutes since open. Target: the best future close up to 10:15 relative to the current close. Sell at the first bar where the predicted remaining upside is at or below the threshold. Deadline 10:15. |
| optimal_stopping (Ridge) | 20 | `edge_threshold` -0.2% to +3%, ridge alpha 0.1 to 20 | Same features, target and rule as the gradient-boosting family. This is a Ridge regression on the best remaining upside, not a true Longstaff-Schwartz backward induction, so the name overstates it. |

For both learned families, each block is predicted by a model trained only on trades from earlier blocks (with a 4-day gap). The first block has too little training data and is skipped, so these two families are scored on four blocks and are not strictly comparable with the others.

Trial counts overstate the variety: in the trailing family only two numbers (`trail`, `activate`) matter in practice, since the deadline barely changes anything (see 1b). The 5,000 robustness trials are further draws from one 9-parameter distribution.

## 1b. Implementation defects found after the stop (read before trusting any table)

I found these by reading `_events` and by running the simulator on 400 dev windows after the campaign stopped. **Status: items 1-5 are fixed (see the tag at the end of each item); the fixed_time and trailing families were re-run in section 8. Counts quoted below are the pre-fix measurements.**

1. **Morning rules are checked only on the 09:30 bar.** In `_events`, after the first regular-session bar the loop does `if used: if not scale_out: continue`, so trailing-stop, momentum, VWAP, give-back and model rules are never evaluated again; the trade is held to the deadline. In a 400-trade test:
   - Trailing policy (trail 2%, activate 0): 257 exits as `trailing_stop`, with a median exit 1 minute after the open, and 143 held to the deadline.
   - Velocity policy (floor -0.1%): 234 exits as `momentum_loss`, median exit 1 minute after the open.

   So none of these tested momentum or velocity exits across the 09:30 to 10:15 window. They are effectively "exit at the open unless a single first-bar condition passes, otherwise hold to the deadline". This also applies to both learned families. **FIXED**: rules are now evaluated at every regular-session decision bar close from 09:30 through the deadline (`test_trailing_fires_on_later_bar`, `test_velocity_floor_fires_on_later_bar`, `test_giveback_fires_on_later_bar`, `test_session_vwap_loss_fires_on_later_bar`, `test_no_lookahead_future_perturbation_does_not_change_decision`). The learned families still need a re-run.
2. **Scale-out remainder is not trailed.** After the first scale-out sale, the loop `continue`s on every later bar, so the trailing stop is skipped and the rest is held to the deadline. I haven't confirmed this by running it. **FIXED**: the remainder keeps being monitored on every later bar (`test_scale_out_then_trailing_on_remainder`). Also fixed: tranche fractions are now fractions of the original position (the old final tranche was computed as a fraction of the remaining shares, so it was too small), and tranches share the per-bar volume budget. The scale-out family is not re-run.
3. **Fixed-time rows measure the wrong thing.** The decision is always the first regular-session bar. The deadline changes only the fill window. At 09:30 the fill start (next bar) falls after the window end, so the whole position is sold at once at the 09:30 bar's open, with no volume limit and no lag. That explains the 09:30 row (1.06% mean, 42.6% hit rate) being far above the others (0.57%). Rows from 09:45 to 10:30 are identical because the volume-limited fill completes within a few bars. Their "J" is therefore not evidence about holding to those times. **FIXED**: the deadline now selects the decision bar, and the fixed-time exit decides at that bar's close (`test_fixed_time_decision_bars_and_prices_differ`, `test_deadline_key_is_label_of_decision_bar`). The bar-label convention is in the `exit_sim.py` module docstring (label = bar open; a bar labelled 09:30 closes at 09:31).
4. **Deadline exits price at the decision bar's open.** The same fill-window issue applies when the decision is the deadline bar itself (no lag, no volume limit for that remainder). **FIXED**: realistic fills start at bar t+1 open, are volume limited, and run through a separate fill horizon (last cached bar of the exit day, <= 11:00); residue is sold at that last bar's open (`test_realistic_exit_price_comes_from_next_bar_open`, `test_no_fill_at_or_before_decision_and_volume_limited`, `test_residue_after_window_priced_at_last_bar_open`, `test_decision_on_last_bar_is_unfilled_nan_not_lookahead`). The legacy model uses the same horizon and still equals the stored v2 trades to 5.6e-16 on all 2,375 dev windows (`test_legacy_baseline_matches_stored_trades`, 20-trade sample).
5. **Not comparable to the baseline.** The baseline was not scored with `_score`. The campaign's S1 subset omits the IWM/SPY condition. The `below_vwap` rule uses each minute's own `vwap` field rather than the session VWAP. The pm_ah family holds to the deadline instead of exiting at the open, so it is a different baseline from v2. **FIXED** (except pm_ah, not re-run): the baseline and every candidate are scored with the same `_score`; S1 is the real S1 (above_sma200 AND IWM up AND SPY up on the entry day, regime columns from `attach_regime`); `below_vwap` uses a causal cumulative session VWAP of typical price (`test_session_vwap_uses_cumulative_typical_price_not_bar_vwap_field`); `peak_from` is a policy option (`test_peak_from_open_vs_entry`).
6. **Additional defects found during the fix pass.**
   - **Duplicated bars in the window cache (fixed in code):** 364 of 2,687 cached windows contained every bar twice (overlapping day files). That doubled the volume available in fills and made the "5-bar" velocity a 2.5-bar quantity. `load_windows` now drops duplicate timestamps and `build_windows` does the same. The legacy-equivalence residual went from 3.2e-3 (4 trades) to 5.6e-16 (0 trades). Covered by `test_loaded_windows_have_unique_timestamps`. The cached NPZ was rebuilt afterwards (see "Data cleaning" below).
   - **Duplicate securities in the trade set (fixed in the re-run):** the 2,375 dev windows include 109 ticker-change duplicates that `dedupe_trades` removes everywhere else. The re-run scores the 2,267 deduplicated dev trades (after the rebuild; 2,266 before, when the trade for ticker `NA` had no window because pandas read it as NaN).
   - **Scale-out sizing (fixed):** see item 2.
   - **Velocity (fixed):** on the first bars of the session it used pre-market bars. It is now computed on regular-session bars and is undefined on the 09:30 bar.
   - **Open caveat:** under the realistic model the v2 baseline (sell at "next open") decides at the close of the 09:30 bar and fills from 09:31, not at the 09:30 open as the live rule would. This penalizes baseline (a) by about one minute of drift against a true open sale. All candidates share that convention.
   - **Open:** the old driver `scripts/exit_campaign.py:main()` is disabled (it also evaluated the locked test). The other families need the new driver.

Consequence: statements elsewhere in this report that a family or the 09:30 fixed-time exit is "above the realistic baseline" are not supported and should be disregarded. The only reliable results are the Phase 0 old-vs-new comparison and the legacy-versus-realistic fill sensitivity of the unchanged baseline.

## 2. Phase 0: 08:00 floor retest

The pre-change files were copied to
`results/_backup_pre08/20261020_090015/` before regeneration. The exact
comparison is `results/exit/old_vs_new_baseline.csv`.

| period | version | trades | mean | PF | total | max DD | Sharpe |
|---|---|---:|---:|---:|---:|---:|---:|
| dev | old | 2,273 | 0.217% | 1.10 | 216.1% | -39.9% | 1.16 |
| dev | new, 08:00 floor | 2,273 | 0.152% | 1.07 | 218.3% | -45.0% | 1.11 |
| locked test, existing Phase 0 artifact | old | 311 | -0.477% | 0.81 | -54.0% | -63.9% | -2.09 |
| locked test, existing Phase 0 artifact | new, 08:00 floor | 311 | -0.420% | 0.83 | -50.4% | -63.3% | -1.68 |

The last two rows are included because they were already present in the
Phase 0 artifact and were explicitly requested for the old-vs-new comparison;
they are not results of exit-parameter selection. The ranked Phase 0 summary
(`reports/ranking_report.md`) contains the requested existing strategy rows:

**Dev:** base 1,547 trades, 40.59% hit, 0.32% mean, PF 1.14, +97.66%,
-44.23% DD; S1 660, 43.94%, 0.90%, PF 1.49, +187.72%, -18.16%; H3 394,
44.16%, 1.38%, PF 1.58, +157.52%, -25.84%; H2 204, 43.14%, 1.20%,
PF 1.57, +54.99%, -21.44%. Tuned base: 912, 44.41%, 0.67%, PF 1.34,
+191.34%, -25.25%; tuned H3: 339, 43.95%, 0.72%, PF 1.31, +52.63%,
-17.19%; tuned S1: 613, 44.21%, 1.01%, PF 1.61, +209.91%, -17.05%.
R3 portfolio: 399 trades, 40.60% hit, 1.00% mean, PF 1.41, +100.94%,
-21.24% DD.

**Existing locked-test Phase 0 summary:** base 257 trades, -0.68% mean,
PF 0.75, -31.79%, -36.13% DD; S1 75, +0.23%, PF 1.14, +2.56%, -11.29%;
H3 34, +1.34%, PF 1.79, +8.78%, -5.98%; H2 17, +2.66%, PF 2.93,
+8.77%, -2.43%; tuned base 109, -0.13%, PF 0.93, -3.89%, -13.52%;
tuned H3 28, +2.66%, PF 3.71, +15.30%, -1.59%; tuned S1 70, +0.13%,
PF 1.07, +0.85%, -12.74%. These tiny S1/H3 samples are weak evidence.

## 3. Fill baselines and fixed-time exits (first run: INVALID, pre-fix; superseded by section 8)

`reports/exit/tables/baseline_fill_sensitivity_dev.csv`:

| fill model | trades | hit rate | mean return | PF |
|---|---:|---:|---:|---:|
| LEGACY trigger-close | 2,375 | 40.29% | 0.1510% | 1.07 |
| REALISTIC next-bar-open | 2,375 | 40.51% | 0.0655% | 1.03 |

The realistic model is the headline model for new exits. It starts execution
one bar after a close decision, at the next bar open, with 10% participation
and forced remainder at the deadline. Legacy baseline equivalence was checked
on 20 sampled trades: maximum absolute return difference was below `3e-16`.

`reports/exit/tables/fixed_time.csv` contains all tested deadlines. Its key
rows are:

| exit deadline | mean return | CV J | hit rate |
|---|---:|---:|---:|
| 09:30 close | 1.0584% | 0.005447 | 42.62% |
| 09:35 close | 0.5701% | 0.000804 | 40.58% |
| 09:40 close | 0.5698% | 0.000806 | 40.62% |
| 09:45 close | 0.5735% | 0.000247 | 40.55% |
| 09:50–10:15 close | 0.5735% | 0.000248 | 40.55% |
| 10:30 close | 0.5735% | 0.000248 | 40.55% |

The implementation's fixed-time label is a deadline bar, so the repeated
later rows are not evidence of independent minute-by-minute discoveries.
“Close of the 1st/5th/10th/15th/30th/45th minute” was represented by the
09:30/09:35/09:40/09:45/10:00/10:15 deadline grid; hold-to-10:00 and
hold-to-10:15 are therefore present as the repeated 10:00 and 10:15 rows.

## 4. Oracle and capture ratio

**Dev only (2,267 trades; S1 n=854, H3 n=395).** The oracle is a per-trade derived table, `results/exit/oracle.parquet` (+ `oracle_meta.json`, built by `scripts/build_exit_oracle.py`; all columns prefixed `hindsight_`, kept out of `data/pit` and `windows.npz`). It was also computed for the 309 test trades, but no test row enters any number below. Analysis: `scripts/analyze_exit_oracle.py`; tables `reports/exit/tables/oracle_{headroom,capture,peak_time,early_moves,random_walk}.csv`; figures `reports/exit/figures/oracle_peak_time.png`, `oracle_headroom.png`. This replaces the earlier "not computed" note; it is a post-stop diagnostic, not a selection step.

**Definitions.** *Gross peak* = best close in the window / entry - 1, no costs ("pm" = 08:00-10:15 incl. pre-market, "open" = 09:30-10:15). *Executable oracle* = for every candidate decision bar t, the realistic net return of a full sale decided at t's close and filled from t+1's open (10% participation, 0.15%/0.40% slippage, 0.10% per side), maximised over t. It equals `simulate_trade` for the same bar (test to 1e-9 on 60 real windows). Both are upper bounds that no causal rule can reach.

| mean net return per trade (dev) | all | S1 | H3 |
|---|---:|---:|---:|
| baseline (a) | 0.13% | 0.69% | 1.33% |
| fixed close 09:35 | 0.12% | 0.56% | 1.41% |
| fixed close 10:15 | -0.12% | 0.40% | 1.26% |
| executable oracle, 09:30-10:15 | 3.67% | 4.05% | 5.70% |
| executable oracle, incl. pre-market | 4.91% | 5.23% | 7.49% |
| gross peak, 09:30-10:15 | 4.06% | 4.43% | 6.05% |
| gross peak, incl. pre-market | 5.57% | 5.88% | 8.12% |
| random-walk reference (gross peak, 09:30-10:15) | 4.10% | 4.50% | 6.10% |

Medians / hit rates (all dev): baseline -0.65% / 39.7%; executable oracle (open) +1.37% / 63.3%; gross peak (open) +1.72% / 68.5%. The means are skewed by a few large gappers (gap > 10% trades have a mean peak of 33%), so the medians are the better guide.

**Capture ratio** = mean policy net return / mean executable-oracle (09:30-10:15) net return; **gain captured** = (policy - baseline) / (oracle - baseline). Trailing policies are the top 3 of `v2_trailing_top10.csv`, re-simulated on the clean cache.

| policy | capture all / S1 / H3 | gain captured all / S1 / H3 |
|---|---|---|
| baseline (a) | 3.6% / 17.0% / 23.4% | 0 |
| fixed 09:35 | 3.3% / 13.8% / 24.8% | -0.3% / -3.8% / +1.8% |
| fixed 09:45 | -0.2% / 10.8% / 24.0% | -3.9% / -7.4% / +0.8% |
| fixed 10:15 | -3.4% / 10.0% / 22.1% | -7.2% / -8.4% / -1.6% |
| top1 trailing | 3.7% / 14.1% / 28.7% | +0.2% / -3.5% / +6.9% |
| top2 trailing | 2.9% / 12.8% / 27.7% | -0.7% / -5.0% / +5.7% |
| top3 trailing | 3.4% / 13.7% / 26.0% | -0.2% / -3.9% / +3.5% |

The capture ratio can mislead: it is dominated by the denominator (the oracle is a maximum of noisy points, see below), it is low for any causal rule by construction, and a policy that is worse than the baseline can still show a similar ratio because the baseline already captures about the same small fraction. The gain-captured version measures only the improvement over the baseline and is the more useful one; it is at most a few percent here, negative on S1 and positive only on H3 (where the trailing policies were selected, so it is optimistic).

**When the peak occurs (gross peak close, 09:30-10:15; offset = bar label minus 09:30).** Share of trades whose peak is within the first k minutes:

| group | n | median min | 1 | 5 | 15 | 30 |
|---|---:|---:|---:|---:|---:|---:|
| all | 2,267 | 9 | 18% | 39% | 58% | 76% |
| gap: open < entry | 1,141 | 10 | 19% | 39% | 57% | 74% |
| gap 0-3% | 711 | 9 | 17% | 39% | 59% | 75% |
| gap 3-10% | 276 | 6 | 20% | 46% | 64% | 82% |
| gap > 10% | 139 | 9 | 17% | 38% | 58% | 78% |
| H3 / S1 | 395 / 854 | 8 / 10 | 19% / 17% | 40% / 37% | 60% / 56% | 78% / 75% |

Only 4% of peaks are at the last bar (10:15). **First minutes after the open** (close of the 5th/15th/30th minute vs the 09:30 open, 2,206 of 2,267 trades have the needed bar): up 44% / 42% / 42%, down 55% / 57% / 57% (flat 1-2%). Above the entry price at minute 15: 24% for open < entry, 57% for 0-3%, 74% for 3-10%, 96% for > 10%. Up-vs-open at minute 15 is 44% / 40% / 37% / 38% across the four gap buckets, i.e. the larger the gap, the more often the stock fades from the open.

**Random-walk reference.** For each trade I shuffled the minute-to-minute close returns of the 09:30-10:15 path (the gap and the first bar fixed), recomputed the gross peak and averaged over 200 permutations (`hindsight_rw_peak_ret_open`). Mean gross peak: 4.06% real vs 4.10% random-walk (all), 4.43% vs 4.50% (S1), 6.05% vs 6.10% (H3). **Excess hindsight (oracle minus random walk) = -0.04 pp (SE 0.04), -0.06 pp, -0.05 pp: indistinguishable from zero.** If the first bar's open-to-close move is shuffled too, the reference rises to 4.56% (sensitivity only). The executable oracle is 0.43 pp below the random-walk gross reference (all) because of costs.

**What this means.** Hindsight picking of the best exit minute is worth about 3.7% per trade net (4.9% if pre-market bars are allowed) against 0.13% for the baseline, so the "headroom" looks like about 28 times the current edge. Almost all of it is the maximum of 46 to 75 noisy points: a random walk with the same volatility and gap gives the same peak (4.1% vs 4.06%), so the excess that is specific to this strategy's actual paths is about zero. The peak is early (median minute 9; 58% within 15 minutes) but a random walk's maximum of a path that is flat to falling is also early, so the timing distribution does not tell us the peak is predictable. Realistically, the room for an exit-timing rule is whatever causal information predicts the path after the open, and the evidence from sections 3 and 8 is that simple rules (fixed times, trailing) capture 0 to 7% of the gain over the baseline, with confidence intervals that include zero. I would not read the oracle as evidence that a better exit exists; it is an upper bound whose size is mostly an artefact of maximising over noise.

## 5. Best dev results and uncertainty (first run: INVALID, pre-fix; see section 8)

**Retracted.** The best fixed-time row (09:30) is not evidence of an improvement: see 1b item 3 (it is filled at the 09:30 open with no volume limit) and 1b item 5 (the baseline was never scored in the same measure). It is also one of 6,001 completed policy trials plus an incomplete batch. The best robustness batch has mean return 0.681% and J
0.00311, while the best fixed-time row has mean 1.058% and J 0.00545.
[Invalid, pre-fix; the baseline was never scored in this measure.] Trailing, momentum, AH/PM, and combo families were claimed positive relative to the
realistic baseline in-sample/CV, but their best rows range from J 0.00150 to
0.00249. Scale-out was weakest among the hand-built families. GBM and
Longstaff-style regression families were negative.

No paired bootstrap confidence intervals were computed before the stop, and
there is no valid basis to call these differences statistically
distinguishable from noise. Parameter selection across thousands of trials
would require a deflated-Sharpe/multiple-testing correction and fresh
pre-registered evaluation; neither was completed.

## 6. Caveats

- No `reports/exit/finalists.json` exists. No campaign finalist was evaluated
  on the locked test and no test result should be inferred from this report.
- The realistic fill model is still stylized: fixed 10% participation,
  close-only decisions, next-bar open execution, fixed slippage, and forced
  deadline fills. Thin/sparse bars and the forced remainder can flatter or
  penalize policies depending on the path.
- Legacy trigger-close fills are optimistic relative to realistic execution and
  are retained only for regression compatibility.
- The oracle is missing, so achievable-bound/capture claims are unavailable.
- Conditional rules and robust random refinements have a substantial
  multiple-testing risk. S1/H3 account samples are especially small.

## 7. Code and verification state

Created/changed:

- `src/backtest/exit_sim.py`: compact NPZ windows, causal policy evaluation,
  legacy/realistic fills, partial tranche support, causal model features.
- `tests/test_exit_sim.py`: floor, no-lookahead, lag, partial, and baseline
  policy tests.
- `scripts/exit_campaign.py`: resumable family runner and report generator.
- `scripts/exit_progress.py`: atomic live progress document generator.
- `src/backtest/exit_variants.py`, `src/backtest/pit_runner.py`: `pm_start`
  support and 08:00 path floor; added `smart_exit_v1` experimental spec.
- `README.md`: simulator campaign section.
- `results/exit/` artifacts and the dated pre-change backup.

`smart_exit_v1` is a recommendation-only experimental variant, not a proven
winner or the default. It has a basic trailing/deadline implementation in
`pit_runner.py`, but it was not selected or validated as a finalist. The
default `main.py` selection remains `v2` when `EXIT_VARIANTS` is unset.

Verification after the stop (pre-fix, superseded):

```text
pytest tests/test_exit_sim.py tests/test_pit.py -q
10 passed in 0.77s
```

Verification after the fix pass:

```text
pytest tests/test_exit_sim.py tests/test_pit.py -q
24 passed in 1.26s
```

Fix-pass files: `src/backtest/exit_sim.py` (rewritten `_events`, `_fill`, `simulate_trade`, `features_at`; new `fixed_time_policy`, `close_to_label`; de-duplicated window loading), `scripts/exit_campaign.py` (`_masks`, `_score`, `score_vs_baseline`, `_paired_bootstrap`; old `main` disabled), `scripts/exit_rerun_v2.py`, `scripts/exit_sanity_v2.py`, `tests/test_exit_sim.py`.

No pip packages were installed. `optuna` was not installed or used.

## 8. Re-run after fixes (fixed_time and trailing only; dev only)

**What changed against the first run.** The simulator now evaluates rules at every regular-session decision bar, fixed-time exits decide at the close of the requested bar, and realistic fills start at the next bar's open with volume limits through the end of the cached exit-day window. The window cache is de-duplicated and the 110 duplicate-security trades are dropped (2,267 dev trades; S1 n=854, H3 n=395). S1 is now the real S1 (above_sma200 AND IWM up AND SPY up on the entry day). The baseline and every candidate are scored by the same `_score`. The locked test was not touched; selection and scoring are dev only. Driver: `scripts/exit_rerun_v2.py` (log `results/exit/rerun_v2.log`); sanity output: `results/exit/sanity_v2.txt`.

**Conventions.** Bar label = bar open time. "Close at 09:35" means the decision is taken at the close of the bar labelled 09:34 and the fill starts at the 09:35 open. "09:31" is the close of the 09:30 bar. Realistic fills: 10% participation per bar, 0.15% regular / 0.40% extended slippage, 0.10% commission per side. J = mean over 5 consecutive date blocks of (mean of the base, S1 and H3 subset mean returns) minus 0.5 x std over blocks. `dJ` is J minus the J of the baseline scored the same way. "mean" in the tables is the pooled mean net return per trade.

### Baselines and fixed-time exits (all on the identical 2,267 trades)

| policy | J | mean (base) | hit (base) | median (base) | PF (base) | mean S1 | mean H3 | dJ vs (a) | dJ vs (b) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) v2 / pm_target25, 08:00 floor | 0.00067 | 0.131% | 39.7% | -0.65% | 1.06 | 0.686% | 1.332% | 0 | +0.00050 |
| (b) close of 09:30 bar (09:31) | 0.00016 | 0.071% | 40.0% | -0.62% | 1.03 | 0.455% | 1.492% | -0.00050 | 0 |
| close 09:35 | 0.00027 | 0.120% | 41.3% | -0.69% | 1.05 | 0.558% | 1.411% | -0.00040 | +0.00011 |
| close 09:40 | -0.00042 | 0.081% | 41.6% | -0.69% | 1.03 | 0.489% | 1.362% | -0.00109 | -0.00058 |
| close 09:45 | -0.00106 | -0.008% | 41.3% | -0.78% | 1.00 | 0.439% | 1.366% | -0.00173 | -0.00123 |
| close 10:00 | -0.00355 | -0.142% | 40.1% | -0.85% | 0.95 | 0.304% | 1.164% | -0.00422 | -0.00372 |
| close 10:15 | -0.00270 | -0.125% | 41.8% | -0.78% | 0.96 | 0.405% | 1.262% | -0.00337 | -0.00286 |
| close 10:30 | -0.00408 | -0.199% | 41.2% | -0.86% | 0.94 | 0.313% | 0.676% | -0.00475 | -0.00424 |

Tables: `reports/exit/tables/v2_baselines_and_fixed_time.csv`, `v2_fixed_time.csv`. Figure: `reports/exit/figures/v2_fixed_time_bars.png`. The rows now differ across deadlines (the first run showed 09:45 to 10:30 as identical). No fixed-time exit beats the baseline (a): holding later than about 09:40 loses money monotonically in J; the best fixed row (09:35) has dJ = -0.00039 vs (a) and +0.00011 vs (b). Paired day-level bootstrap (5,000 resamples of 398 entry days): 09:35 minus (a) = -0.011% per trade, 95% CI [-0.255%, +0.222%]; 09:35 minus (b) = +0.050%, CI [-0.111%, +0.208%] (`v2_fixed_bootstrap.csv`). Both include zero. Hit rate rises by 1 to 2 points for 09:35 and later exits, but that comes with a lower or equal mean.

Real-window sanity (2,375 un-deduplicated dev windows, `results/exit/sanity_v2.txt`): legacy baseline vs the stored v2 trades max |diff| 5.6e-16; fixed-time mean returns by close time 0.068%, 0.129%, 0.088%, -0.011%, -0.152%, -0.134%, -0.198% for 09:31/09:35/09:40/09:45/10:00/10:15/10:30 (hit 40.1/41.5/41.6/41.3/40.1/41.9/41.3%). For a trailing policy (trail 3%, activate 0, deadline 10:15) the decision minute is spread over the session: 255 trades decide in minutes 1-4, 225 in 5-9, 263 in 10-19, 135 in 20-29, 195 in 30-44 and 1,302 at the deadline (minute 45), against the first run's clustering at minute 1.

### Trailing: random search (1,500 trials) and grid

Search space (as specified): trail U(0.01, 0.16), activate U(0, 0.30), deadline in {09:45, 10:00, 10:15, 10:30}, `peak_from` in {entry, open}; one independent draw per trial (seed 20261022). **The number of trials is 1,500 (plus 224 grid cells, 1,724 evaluated policies in all), all selected on dev, so the best of them is biased upward by multiple testing; none of these numbers is a test result.** Table: `v2_trailing.csv`; top 10 by dJ among valid trials: `v2_trailing_top10.csv`. The validity filter (hit rate within 2 points of the original table) passed all 1,500 trials, so it does not discriminate.

- 22 of 1,500 trials have dJ > 0 against (a): 21 use `peak_from=entry` and 1 uses `open`. 81 beat (b). Median trial dJ is about -0.0022. By deadline and `peak_from`, the mean dJ is negative in every cell (best cell mean -0.0007: entry / 09:45).
- Top 10 by dJ vs (a):

| rank | trail | activate | deadline | peak_from | J | dJ vs (a) | dJ vs (b) | d mean base | d mean S1 | d mean H3 |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.36% | 1.61% | 10:15 | entry | 0.00175 | +0.00108 | +0.00158 | +0.006pp | -0.116pp | +0.302pp |
| 2 | 1.73% | 1.87% | 10:30 | entry | 0.00132 | +0.00064 | +0.00114 | -0.025pp | -0.168pp | +0.249pp |
| 3 | 1.15% | 2.60% | 10:30 | entry | 0.00117 | +0.00050 | +0.00100 | -0.006pp | -0.131pp | +0.152pp |
| 4 | 2.51% | 6.03% | 10:15 | entry | 0.00104 | +0.00037 | +0.00088 | +0.017pp | -0.088pp | +0.061pp |
| 5 | 3.59% | 0.60% | 10:15 | entry | 0.00102 | +0.00035 | +0.00085 | -0.030pp | -0.115pp | +0.179pp |
| 6 | 2.07% | 1.99% | 10:30 | entry | 0.00101 | +0.00035 | +0.00085 | -0.028pp | -0.180pp | +0.217pp |
| 7 | 3.31% | 4.42% | 10:15 | entry | 0.00096 | +0.00029 | +0.00079 | +0.004pp | -0.133pp | +0.091pp |
| 8 | 3.66% | 1.33% | 10:15 | entry | 0.00094 | +0.00027 | +0.00078 | -0.021pp | -0.145pp | +0.186pp |
| 9 | 1.89% | 4.26% | 10:30 | entry | 0.00089 | +0.00022 | +0.00073 | +0.017pp | -0.084pp | +0.025pp |
| 10 | 4.27% | 2.09% | 10:15 | entry | 0.00088 | +0.00021 | +0.00071 | -0.025pp | -0.095pp | +0.046pp |

("pp" = percentage points of mean net return per trade.) The improvements in J come from the H3 subset and from a higher hit rate (39% to 46%, most 43% to 46%, against 39.7% for (a)), while the mean on all trades is unchanged and the mean on S1 is lower than (a) in every one of the top 10.

- **Top 3 per subset against the baselines** (`v2_top3_subset_stats.csv`; hit / mean / median / PF):

| policy | base | S1 | H3 |
|---|---|---|---|
| (a) | 39.7% / 0.131% / -0.65% / 1.06 | 41.9% / 0.686% / -0.50% / 1.38 | 44.1% / 1.332% / -0.38% / 1.55 |
| (b) 09:31 | 40.0% / 0.071% / -0.62% / 1.03 | 42.0% / 0.455% / -0.50% / 1.25 | 44.6% / 1.492% / -0.32% / 1.63 |
| top1 | 44.2% / 0.137% / -0.42% / 1.06 | 47.7% / 0.571% / -0.12% / 1.29 | 47.1% / 1.634% / -0.12% / 1.67 |
| top2 | 44.0% / 0.106% / -0.50% / 1.04 | 47.4% / 0.518% / -0.19% / 1.25 | 47.6% / 1.581% / -0.23% / 1.62 |
| top3 | 46.8% / 0.125% / -0.32% / 1.05 | 49.9% / 0.555% / -0.02% / 1.27 | 50.1% / 1.483% / +0.06% / 1.55 |

- **Paired day-level bootstrap** (5,000 resamples over 398 entry days, mean per-trade improvement, 95% CI; `v2_top3_bootstrap.csv`):

| candidate | vs (a) | vs (b) |
|---|---|---|
| top1 | +0.006% [-0.193%, +0.194%] | +0.067% [-0.032%, +0.170%] |
| top2 | -0.025% [-0.251%, +0.186%] | +0.035% [-0.086%, +0.159%] |
| top3 | -0.006% [-0.243%, +0.207%] | +0.054% [-0.079%, +0.189%] |

All six intervals include zero, and these are the best of 1,500 selected trials, so the true improvement is probably smaller than shown.

- **Stability (grid, deadline 10:15; `v2_trailing_grid.csv`, `reports/exit/figures/v2_trailing_grid_heatmap.png`).** With `peak_from=open` all 112 cells are negative (dJ -0.0001 to -0.0038); the surface is smooth (correlation of a cell with the mean of its four neighbours 0.96) and gets worse as the activation level rises, with the least bad region at the tightest trails (1% to 3%) and low activation. With `peak_from=entry` only 8 of 112 cells are positive (best +0.00096). They form one small contiguous pocket at trail 1% to 5% and activate 2% to 5%, and the pocket edge is steep: activate 0% (the neighbour) is about zero or negative, and trail 6% or more is clearly negative. The surface is smooth overall (neighbour correlation 0.95), so the best random trial is not an isolated spike, but the positive region is narrow and sits next to zero or negative cells, and it is the same tight-trail, `peak_from=entry` region in which trailing is closest to "sell at the open unless the stock is already trading well above its after-hours/pre-market range". I have not verified that mechanism; it is a hypothesis only.

**Conclusion (dev only).** Neither fixed-time exits nor trailing stops beat the v2/pm_target25 baseline in a statistically distinguishable way in this re-run. Fixed-time exits are all worse than the baseline in J. The best trailing policies raise J and the hit rate by shifting gains into the H3 subset at the cost of S1, with no improvement in mean return on all trades and CIs that straddle zero after 1,724 trials. I would not carry any of them to the locked test as an improvement without a pre-registered, much smaller candidate set. The other families have not been re-run, so no statement about them is made here.

### Data cleaning (root causes, fixes and the validate step)

- **Duplicate bars.** No minute file has a repeated (symbol, timestamp) and `daily_pit.parquet` / `candidates.parquet` have no repeated (symbol, date). The duplicates came from `build_windows` concatenating the day files of a symbol: each file holds the signal day plus the next session, so a symbol that is a candidate on consecutive days has the same bars in two files (584 symbols, 554,588 repeated bars, 0 with differing values). 364 of the 2,687 old windows were affected. Loaders now drop duplicate timestamps (`load_minute`, `build_windows`, `load_windows`); the overlap itself is by design and is checked for value conflicts by `validate`.
- **Duplicate securities.** A ticker change leaves the same company's bars under the old and new symbol (113 of 2,697 trades: 110 dev, 3 test; 102 of the 113 pairs share a CIK or composite FIGI in the reference, the rest have no reference match). `dedupe_trades` keys on (entry day, prior close, 15:55 price) and keeps the first row, i.e. the alphabetically first symbol since the tables are symbol-sorted. The same rule is now applied to candidates (`drop_daily_aliases`: same date and OHLC, volume within 2% and at least 10,000 shares, alphabetically first candidate kept; 3 twins that are the traded security in existing results are kept): `candidates.parquet` 8,496 to 8,195 rows. The daily history itself keeps both tickers (87,211 daily rows repeat an earlier ticker's bar).
- **Other.** 130 exact duplicate rows in `news.parquet` removed; 30 ELOX daily rows from Massive with zero prices removed from `daily_pit.parquet`; 300,288 minute rows (8,542,607 to 8,242,319) of twin candidates removed from 194 minute files. Originals are in `data/pit/_backup_dedupe_<timestamp>/` (`scripts/repair_pit_data.py`).
- **Cache.** `results/exit/windows.npz` rebuilt by `scripts/build_exit_windows.py` (old one kept as `windows_pre_dedupe.npz`, id map in `windows_index.parquet`): 2,687 to 2,576 windows (dev 2,267, test 309) = 2,584 de-duplicated trades minus 8 with no next-session bars; 113 twin trades skipped; 0 duplicate timestamps; every window starts at the entry bar. A ticker named `NA` was read as NaN by `pd.read_csv`, so its trade had no window before (`keep_default_na=False` now).
- **Effect on the re-run.** Dev set 2,266 to 2,267 trades; all numbers in the tables moved by at most 0.01 percentage points of mean return and 0.00002 of J (the conclusions are unchanged); only the affected lines of this section were updated.
- **Validate.** `python build_pit_dataset.py check` (or `validate_pit_data()` in `src/data/pit_integrity.py`, run automatically at the start of `exit_rerun_v2.py` and `exit_campaign.py` and after the candidates / alpaca-minute / build steps) exits 1 on duplicates, twin candidates, non-monotonic or out-of-session timestamps, bad prices, and on window-cache problems (duplicate timestamps, missing entry bar, windows for twin trades).

## Corrected partition baselines

The partition implementation now separates random draws from the hard-screen
universe and the strategy-filtered pre-ranking pool. It counts no-choice days
and uses a single pooled comparable-trial Bias-B estimate with `N_eff` and the
cross-trial standard deviation. Values below are log-return percentage points;
`skill` is the residual `total - trend_iwm - bias`.

| setup | period | total | trend IWM | trend SPY | random universe | random pool | skill | bias A | bias B | N_eff | sd | skill % | random p U/P |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | dev | 68.14 | 13.71 | 12.67 | 46.55 | 44.42 | -65.93 | n/a | 120.36 | 1,747 | 35.29 | n/a | 0.249/0.216 |
| S1 | dev | 105.68 | 3.85 | 6.72 | 39.66 | 78.24 | -46.30 | n/a | 148.13 | 1,824 | 43.28 | n/a | 0.004/0.043 |
| H3 | dev | 94.59 | 0.82 | 1.91 | 41.24 | 95.17 | -54.36 | n/a | 148.13 | 1,824 | 43.28 | n/a | 0.041/0.586 |
| H2 | dev | 43.82 | -1.36 | 1.42 | 33.85 | 44.39 | -102.94 | n/a | 148.13 | 1,824 | 43.28 | n/a | 0.303/0.586 |
| R3 | dev | 66.85 | 1.89 | 4.00 | 30.16 | 60.83 | -110.79 | n/a | 148.13 | 1,824 | 43.28 | n/a | 0.080/0.182 |
| S1+tuned | dev | 113.11 | 3.26 | 6.69 | 36.12 | 87.96 | -38.28 | 108.18 | 148.13 | 1,824 | 43.28 | n/a | 0.001/0.045 |
| H3+tuned | dev | 42.29 | 1.77 | 2.83 | 24.90 | 41.68 | -107.61 | 133.49 | 148.13 | 1,824 | 43.28 | n/a | 0.257/0.049 |
| base | test | -38.26 | 1.95 | 1.47 | -40.92 | -43.06 | -40.21 | n/a | 0 | n/a | n/a | n/a | 0.332/0.201 |
| S1 | test | 2.52 | 0.26 | 1.23 | -7.54 | 0.53 | 2.26 | n/a | 0 | n/a | n/a | 89.63 | 0.141/0.087 |
| H3 | test | 8.41 | 0.67 | 0.33 | -1.74 | 8.41 | 7.74 | n/a | 0 | n/a | n/a | 92.01 | 0.144/1.000 |
| H2 | test | 8.41 | 0.31 | 0.39 | 0.01 | 8.41 | 8.10 | n/a | 0 | n/a | n/a | 96.26 | 0.170/1.000 |
| R3 | test | 2.93 | 0.27 | 0.51 | -3.32 | 2.93 | 2.66 | n/a | 0 | n/a | n/a | 90.81 | 0.270/0.001 |
| S1+tuned | test | 0.85 | -0.05 | 0.91 | -4.64 | 1.18 | 0.90 | n/a | 0 | n/a | n/a | 105.62 | 0.293/0.577 |
| H3+tuned | test | 14.24 | 0.65 | 0.29 | 2.66 | 14.24 | 13.58 | n/a | 0 | n/a | n/a | 95.41 | 0.104/1.000 |

All Dev Bias-B rows are flagged unreliable because the pooled haircut exceeds
the observed total; the estimate is retained for transparency but should not
be interpreted as measured bias. Tuned rows also show Bias-A and Bias-B of the
same order, but the estimates remain weak. The current registry has no aligned
per-trial return series, so `N_eff` is the raw comparable-trial count rather
than a correlation-adjusted count. Skill percentages are deliberately
`n/a` whenever residual skill is negative.

## Five-fold partition stability

The chronological k=5 fold outputs are in `reports/partition/`; the combined
summary is `reports/partition/kfold_summary_dev_test.csv`. Folds are contiguous
trading-day blocks with no parameter refitting inside a fold. The same-pool
random-pick draws (2,000 per fold) determine the exact Poisson-binomial luck
tail.

| setup | period | profitable | beats random | fold t | best-fold share | luck p |
|---|---|---:|---:|---:|---:|---:|
| base | dev | 3/5 | 3/5 | 0.75 | 0.96 | 0.686 |
| S1 | dev | 4/5 | 4/5 | 1.65 | 0.63 | 0.524 |
| H3 | dev | 3/5 | 1/5 | 1.31 | 0.70 | 1.000 |
| H2 | dev | 3/5 | 2/5 | 0.98 | 0.92 | 1.000 |
| R3 | dev | 4/5 | 3/5 | 1.51 | 0.69 | 0.583 |
| S1+tuned | dev | 4/5 | 5/5 | 2.25 | 0.46 | 0.826 |
| H3+tuned | dev | 4/5 | 4/5 | 1.54 | 0.64 | 1.000 |
| base | test | 0/5 | 2/5 | -2.57 | 0.04 | 1.000 |
| S1 | test | 1/5 | 1/5 | 0.20 | 4.17 | 1.000 |
| H3 | test | 3/5 | 1/5 | 0.76 | 1.20 | 1.000 |
| H2 | test | 3/5 | 1/5 | 0.83 | 1.16 | 1.000 |
| R3 | test | 1/5 | 1/5 | 0.23 | 3.62 | 1.000 |
| S1+tuned | test | 2/5 | 0/5 | 0.06 | 12.49 | 1.000 |
| H3+tuned | test | 4/5 | 1/5 | 1.30 | 0.80 | 1.000 |

Exit-specific dev outputs are also in `reports/exit/kfold_*_dev.csv` for
baseline (3/5 profitable, p=0.316), fixed 09:30 (2/5, p=0.698), and trailing
(2/5, p=0.637). Five folds have low power; a 4/5 result is not strong evidence
by itself, and dev folds of tuned configurations are not independent evidence.
The locked-test period is only about four months, so its folds are noisy and
the entry set has been inspected historically.

## References

Longstaff and Schwartz, *Valuing American Options by Simulation* (2001);
Kaufman, *New Trading Systems and Methods* (2005); Bailey et al.,
*The Probability of Backtest Overfitting* (2014); and the Alpaca minute-bar
documentation. These references informed the causal continuation-value,
efficiency/momentum, and multiple-testing controls; they are not evidence that
the stopped campaign's results generalize.
