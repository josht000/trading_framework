# Direct visual resistance detection audit — 2026-10-09

## Scope and provenance

This file contains concise, observable evidence summaries and decision criteria. The detector is the assistant inspecting each PNG and its OHLCV data. Scripts are used only for price lookup, scoring, and output verification.

Chart and price files are under `data/resistance_dataset/{train,val}/{charts,price_data}`. The 19 available training and validation charts (11 train, 8 val) are in scope; chart 08 is absent from the supplied training inputs.

## Evaluation protocol, fixed before prediction

- Primary benchmark: absolute error <= $0.10, as specified in the dataset prompt.
- Secondary benchmark: <= 1% of the labelled level for charts whose last close is >= $50; otherwise <= $0.10, as specified in the skill.
- One-to-one matching, nearest absolute difference first, following the existing `score.py` convention. Several predictions near one label cannot all score as hits.
- Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN).
- Requested accuracy is explicitly level-set agreement TP/(TP+FP+FN), NOT conventional classification accuracy. TN and conventional accuracy are undefined: there is no specified universe of negative price levels.
- Report every chart, including a chart with no selected levels. Do not infer profitability, execution quality, or live-trading readiness from annotation agreement.
- The existing scorer iterates only labelled chart keys. Coverage will be checked separately so unlabelled/zero-label charts cannot silently hide false positives.

## Initial operational rules (A), before training-label access

A1. Only prices strictly above the final close qualify. Inspect the entire visible window, not future bars.
A2. Select visually distinct failed rallies, rejection wicks, or repeatedly tested shelves. Ground the output in a dated OHLCV high; no arbitrary gridline or round-number extrapolation.
A3. Preserve the major spike/window-top supply zone when overhead. Volume corroborates participation; a volume spike alone is not resistance or proof of institutional supply.
A4. Keep separate post-peak rebound episodes. Do not automatically output every lower high or every consecutive decline candle. A breakdown-day level needs a visible gap/ledge or repeated interaction.
A5. Consolidation touches are one zone, represented by a well-supported high. Separated episodes may remain distinct when visibly different; no fixed count quota.
A6. Reject unconfirmed highs from the last three bars unless a major rejection or historical alignment supports them. A 2% near-close distance is a warning, not sufficient evidence by itself; on large caps it is not a hard exclusion.
A7. Pre-peak highs remain eligible if visibly significant and not subsumed by a later zone. Do not select a level merely to fill a price gap.
A8. Do not promote support lows, moving averages, or volume-only signals into resistance without a visible failed retest.
A9. Conflicting skill rules (always include the top versus sometimes omit it; shelf extreme versus modal shelf; strict pivots versus decline-day highs) are resolved by observable chart structure and documented uncertainty, not by a mechanical pivot rule.

Price lookup uses the existing `highs.py` with window=0: this displays EVERY bar high above the final close, with date/open/close/volume. It does not compute candidate pivots or choose outputs. The PNG supplies the surrounding candle/low and volume context. Exact CSV rows can be inspected for ambiguous cases. Volumes in evidence are shares, usually abbreviated in millions.

## Training predictions and evidence — recorded before labels

### 01 BB — 2024-10-21, close 2.58

Selected: **2.87, 3.02, 3.2499, 3.52, 3.85**. The PNG shows a May blow-off, a descending supply ladder, and June breakdown. OHLCV anchors: Jun 12 H2.87/C2.77/V7.96M (last rebound before breakdown); May 20 H3.02/C3.01/V7.96M and May 22 H3.02 (repeated shelf); May 16 H3.2499/C3.03/V12.00M (post-spike ledge aligned with April highs); May 15 H3.52/C3.23/V30.82M (large decline-day upper ledge); May 14 H3.85/C3.47/V93.93M (dominant spike). Applied A1-A5. Excluded 2.61/2.63 as marginal near-close supply and Oct 16 H2.68 as a still-developing right-edge episode; 3.18/3.236 overlap the later 3.2499 zone. Uncertainty: decline-day ledges may be over-counted relative to a pure swing-high annotation convention.

### 02 CLOV — 2023-11-06, close 1.13

Selected: **1.24, 1.41, 1.63**. July rally stalls at Jul 20 H1.24/C1.15/V9.77M; Sep 14 H1.24/C1.13/V6.46M reinforces the later breakdown ledge. Aug 30/31 and Sep 11 repeatedly reach H1.41 (Sep 11 C1.35/V6.86M), followed by a sustained decline. Aug 9 H1.63/C1.39/V20.60M is the major upper-wick spike; Aug 14 H1.60 is the same topping region. Applied A1-A5/A7. Excluded the 1.17 September wiggle and last-bar 1.18: weak, near price, and the latest high lacks subsequent rejection confirmation. Low nominal share price makes $0.10 tolerance unusually broad; a hit need not imply an exact zone match.

### 03 CLSK — 2023-05-18, close 4.21

Selected: **4.65, 4.82, 5.16**. PNG shows April expansion on increased volume, a sharp reversal, and May retest. May 15 H4.65/C4.60/V7.82M then May 16 C4.29 identifies the recent lower rebound; Apr 19 H4.82/C4.46/V10.75M is the upper decline ledge; Apr 18 H5.16/C5.03/V16.81M is the top, retested May 10 H5.12/V13.73M. Applied A1-A5. Merged 5.12 with 5.16 as a repeated ceiling rather than independent nearby outputs. Excluded 4.29 as weak near-close supply and current-bar 4.58 as unconfirmed. The 4.82 decline-day selection is lower confidence than the two clear rally tops.

### 04 GME — 2024-05-10, close 17.46

Selected: **18.10, 18.59, 20.20**. Sep 28 H18.10/C16.84/V7.43M is a conspicuous failed rally; Dec 19 H18.59/C17.72/V5.45M caps the December range; May 10 H20.20/C17.46/V36.83M is a large same-day upper-wick rejection (high 15.7% above close), so the right-edge exception is justified. Applied A1-A3/A5-A7. Merged Dec 27 H18.37 and May 9 H18.51 into the 18.59 region; excluded Nov 29 H17.56 because it is almost at the current close. No assumption is made that the high-volume spike guarantees future rejection.

### 05 HUT — 2023-12-26, close 15.62

Selected: **16.75, 18.30, 20.00, 21.55, 22.75**. The July spike unwinds through identifiable ledges before an August collapse; December price is only approaching these older zones. Aug 9 H16.75/C15.30/V3.12M is a failed rebound; Jul 27 H18.30/C16.50/V2.25M is another clear rejection; Jul 19 H20.00/C19.10/V2.44M caps a post-spike pause; Jul 17 H21.55/C19.15/V4.52M is the immediate upper collapse ledge; Jul 14 H22.75/C20.80/V4.39M is the window top. Applied A1-A5. Excluded the last-bar 15.83, minor 17.39, and pre-peak 20.15 because they lack independent supply episodes or overlap selected zones. The 21.55 level is a decline-day exception, not a two-sided pivot.

### 06 IONQ — 2024-10-14, close 10.84

Selected: **11.72, 12.32**. Feb 29 H11.72/C10.36/V26.32M terminates the opening shelf; Mar 8 H12.32/C10.81/V17.13M is a long upper-wick failed recovery. Both remain overhead while all later mid-window rallies are below the current close. Applied A1-A3/A5-A7. Excluded Oct 14 H11.00 (unconfirmed latest-bar high) and Mar 7 H11.32 (upswing into the larger next-day rejection). Earlier context before the left image boundary is unavailable; confidence in 11.72 is based on the visible three-day shelf and subsequent fall only.

### 07 JOBY — 2024-12-23, close 7.62

Selected: **8.29, 9.14, 9.33**. Dec 17 H8.29/C8.24/V16.23M and Dec 11 H8.25 form the repeated lower shelf, distinct from the tall wick above it. Dec 9 H9.14/C8.53/V20.06M and Dec 18 H9.08/C7.94/V34.72M form the next rejection zone. Nov 29 H9.33/C8.95/V23.55M and Dec 2 H9.32/C8.11/V38.25M mark the main top. Applied A1-A5. Excluded Jul 16 H7.69 (nearly at close) and separate outputs at 8.79/8.89 (the same upswing into 9.14). Merged 9.08 with 9.14 but kept 9.33 because the intervening drawdown is substantial; shelf representation is somewhat subjective.

### 09 NIO — 2025-02-27, close 4.79

Selected: **5.36, 6.03, 6.45, 6.93, 7.38, 7.71**. Dec 9 H5.36/C5.18/V123.19M is a failed rebound aligned with October/November highs; Oct 28 H6.03/C5.81/V109.99M is a pronounced failed rally. Oct 9 H6.45/C6.29/V70.07M caps a post-gap shelf, Oct 7 H6.93/C6.79/V78.41M caps the preceding short shelf, Oct 2 H7.38/C7.21/V135.60M is the first failed recovery, and Sep 30 H7.71/C6.68/V214.41M is the peak rejection. Applied A1-A5. Excluded 4.92 as marginal near-close old resistance, Feb 27 H5.13 as unconfirmed, and pre-peak 5.70 as superseded by the clearer later ladder. Short shelves 6.45/6.93 have weaker independent-pivot evidence; retained because the PNG shows distinct ledges separated by gaps.

### 10 NOK — 2022-08-04, close 5.23

Selected: **5.34, 5.45, 5.59, 5.85, 6.00, 6.40**. May 4 H5.34/C5.33/V28.24M is the latest material failed rebound; Apr 7/8 H5.45 forms a decline shelf; Mar 30 and Apr 4 H5.59 form a repeated ceiling; Feb 16 H5.85/C5.84/V28.37M and Feb 2 H6.00/C5.97/V31.48M mark distinct earlier recoveries; Dec 28 H6.40/C6.30/V12.60M is the window high. Applied A1-A5. Excluded 5.27 at the right edge and Jan 11 H6.26 inside the old broad topping area. This is a dense staircase: 5.34/5.45 may be difficult to distinguish visually, but they belong to different dated episodes. The remote 6.40 top is not a volume climax and is lower priority.

### 11 PLTR — 2023-04-06, close 8.09

Selected: **8.62, 9.06, 9.21, 9.71, 10.31**. Mar 22 H8.62/C8.21/V34.15M represents the repeated March ceiling, supported by Mar 8 H8.58 and Apr 5 H8.65/C7.98/V54.95M. Nov 1 H9.06/C8.65/V31.05M and Oct 26 H9.05 form an autumn double top. Feb 2 H9.21/C8.99/V71.89M is a separate failed rally. Feb 17 H9.71/C9.20/V61.98M is the post-spike upper ledge; Feb 16 H10.31/C9.71/V88.64M is the top following very high participation on Feb 14/15. Applied A1-A5/A7. Excluded Jan 24 H8.76 as an isolated anomalous-looking candle rather than a separate shelf, and Aug 17 H9.62 at the truncated left edge as overlapping 9.71. The latter and the closely spaced 9.06/9.21 are annotation-sensitive decisions.

### 12 PLUG — 2023-02-01, close 17.68

Selected: **18.26, 19.75, 23.45, 30.43, 31.56**. Nov 15 H18.26/C17.56/V17.16M is reinforced by Jan 18 H18.13/C16.70/V24.29M. Oct 18 H19.75/C19.11/V17.08M is the lower rebound; Sep 26/Oct 4 both reach H23.45 (Oct 4 C23.39/V15.86M), a clear shelf after the collapse. Sep 15 H30.43/C29.30/V17.99M is the last major failed rally; Aug 25 H31.56/C30.00/V61.22M is the peak. Applied A1-A5. Excluded current-bar 18.09, pre-peak 20.02, and 25.50/26.84 during uninterrupted descent without a distinct shelf. Aug 30 H30.68 is merged conceptually into the broad upper region rather than emitted as another step; that choice is uncertain under exact-label evaluation.

Training predictions frozen in `train/predictions_vlm_audit_20261009.csv` before scoring. Training evidence above is the pre-label record; subsequent feedback is recorded separately, not rewritten into the original selections.

## Training evaluation and rule update B

Scored only after predictions were saved. Both the existing scorer and coverage-aware audit scorer agree. Actual labels: **38**, not the skill's stale count of 37. All 11 input charts have labels. Prediction SHA256: `2a707ef7cb580ddd0f611ef197cb3880317324dcff79e2749a01e0a0977faea1`.

Flat and large-cap-adjusted policies are identical on this split.

| Chart | TP | FP | FN | Set accuracy | Precision | Recall | F1 | FP levels | Missed labels |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 01 BB | 4 | 1 | 1 | .6667 | .8000 | .8000 | .8000 | 3.52 | 2.66 |
| 02 CLOV | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — | — |
| 03 CLSK | 2 | 1 | 0 | .6667 | .6667 | 1.0000 | .8000 | 4.82 | — |
| 04 GME | 2 | 1 | 0 | .6667 | .6667 | 1.0000 | .8000 | 18.10 | — |
| 05 HUT | 4 | 1 | 0 | .8000 | .8000 | 1.0000 | .8889 | 21.55 | — |
| 06 IONQ | 2 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — | — |
| 07 JOBY | 2 | 1 | 0 | .6667 | .6667 | 1.0000 | .8000 | 8.29 | — |
| 09 NIO | 6 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — | — |
| 10 NOK | 4 | 2 | 0 | .6667 | .6667 | 1.0000 | .8000 | 6.00; 6.40 | — |
| 11 PLTR | 3 | 2 | 0 | .6000 | .6000 | 1.0000 | .7500 | 9.06; 9.21 | — |
| 12 PLUG | 5 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — | — |
| TOTAL | 37 | 9 | 1 | .7872 | .8043 | .9737 | .8810 | 9 extras | 1 missed |

Error interpretation: BB/CLSK/HUT show that a large decline candle is not sufficient to justify another level. PLTR and GME show that older pre-peak pivots can be superseded. JOBY shows that a small shelf beneath a larger wick may be unlabelled. NOK shows that a remote window top is not universally labelled. BB's missed 2.66 is close to the visible Oct 16-18 rejection cluster: the exclusion was too cautious. These are annotation-agreement errors, not proof that any selected zone has no market relevance.

Rules B (A remains in force except where amended), frozen before validation:

B1. Tighten A4: an isolated next-day decline high is lower priority than a rally reversal, multi-session ledge, or clear gap boundary. Do not populate the entire decline automatically. Retain a particularly conspicuous post-spike ledge only with chart evidence; PLTR shows that banning all decline-day highs would also lose hits.
B2. Tighten A7: older pre-peak pivots require independent relevance after the dominant supply episode; avoid duplicating obsolete intermediate tops. Keep an isolated major top provisionally, but do not assume all historical maxima are labels.
B3. Amend A6: a recent multi-bar rejection with visible follow-through can count even near price. Do not impose a blanket last-three-bars or 2% exclusion when older structure or several rejecting bars corroborate it.
B4. Prefer the dominant rejection zone over a weak inner shelf lacking an independent reversal. Do not set a numeric level-count target.

No training prediction was changed after these labels were revealed. Training score is fitting feedback, not an independent estimate.

## Validation predictions and evidence — recorded before labels

### 13 PTON — 2024-08-26, close 4.90

Selected: **5.30, 6.21, 7.24**. Feb 1 H5.30/C4.21/V84.24M marks a large gap-down boundary; today's H5.26/C4.90/V38.74M tests essentially that old ledge, so this is not an unsupported last-bar prediction. Jan 12/22 H6.21 and Jan 23 H6.18 form the January shelf (Jan 22 C5.86/V14.69M). Jan 5 H7.24/C6.72/V68.00M is the visible top with a large wick. Applied A1-A5/B1/B3. Excluded Feb 12 H5.07 as a weaker nearby rebound and Jan 9 H7.05 as part of the same upper rejection episode. Left-edge history is truncated; 7.24 is the window top, not a claim about the all-time high.

### 14 RIOT — 2024-02-27, close 16.80

Selected: **18.24, 18.75, 19.20, 19.50, 20.65**. Feb 15 H18.24/C16.53/V37.29M is a clear failed rally, reinforced by today's H18.36/C16.80/V43.62M; no separate output at 18.36. Dec 27 H18.75/C18.62/V43.17M precedes the end-December collapse. Aug 2 H19.20/C17.97/V22.88M and Jul 31 H19.19 form a repeated lower top after the Jul 20 H19.50/C18.47/V23.81M recovery. Jul 14 H20.65/C19.12/V39.09M is the major peak. Applied A1-A5/B1/B3. Excluded Jan 11 H17.02 and Aug 8 H17.36 as near-close decline-boundary candidates with weaker independent rejection structure. Distinguishing the 19.20 and 19.50 zones is more subjective than the three main rally tops.

### 15 RIVN — 2024-07-30, close 16.22

Selected: **16.90, 17.56, 18.49, 18.86, 20.66, 21.94, 24.61**. Jul 25/26 H16.90 is a repeated recent rejection aligned with Jan/Feb highs 16.80-16.86; Jul 23 H17.56/C17.50/V23.46M is followed by a sharp gap lower. Jul 17 H18.49/C17.19/V46.77M is a separate failed recovery after Jul 12 H18.86/C18.11/V97.34M. Jan 3 H20.66/C20.32/V35.63M and Jan 2 H21.94/C21.10/V49.63M are conspicuous separate gap-down boundaries. Dec 26 H24.61/C23.59/V30.46M caps the December cluster. Applied A1-A5/B1/B3. Excluded Jan 9 H19.76 as a small rebound embedded in the uninterrupted decline and Jun 26 H16.35 as too near price, not an independent overhead zone. Seven outputs are justified by separate visible structures, not a quota. The two January boundaries are explicitly lower-confidence non-pivot exceptions.

### 16 RKLB — 2024-07-10, close 5.13

Selected: **5.25, 5.67, 5.83, 6.14**. Jan 24 H5.25/C5.03/V3.42M is tested again by today's H5.25/C5.13/V5.29M; included for historical alignment rather than a new confirmed pivot. Jan 9 H5.67/C5.45/V6.82M is a lower recovery top after a substantial early-January pullback. Dec 26 H5.83/C5.64/V18.72M represents the repeated December shelf (Dec 27 H5.81), distinct from Dec 28 H6.14/C5.79/V11.19M upper wick. Applied A1-A5/B3. Excluded Dec 14 H5.48 as pre-peak and superseded, Jan 12 H5.35 as an interior decline wiggle, and Jun 24 H5.16 as almost at close. The 5.25 level and splitting the December shelf from the spike are lower-confidence decisions.

### 17 SNAP — 2024-05-28, close 15.32

Selected: **15.97, 16.65, 17.29, 17.50, 17.75, 17.90**. May 23 H15.97/C15.18/V16.73M follows May 22 H15.92 and is followed by two weaker sessions: a recent small rejection, not merely the last candle high. May 20 H16.65/C16.43/V22.17M is the lower recovery top; May 7 H17.29/C17.14/V24.97M aligns with Dec 21 H17.29. Feb 5/6 H17.50 precedes the large gap down (Feb 6 V92.64M); Jan 9 H17.75/C17.05/V35.65M is a separate failed rally; Dec 18 H17.90/C17.02/V49.76M is the window's tallest wick. Applied A1-A5/B3. Excluded 16.27 inside the May rebound and 17.14 inside the older range. The three upper zones are dense and the remote 17.90 wick may be redundant, but it remains a visibly separate rejection episode rather than being deleted based on the skill's disclosed label example.

### 18 SNDL — 2024-04-23, close 2.07

Selected: **2.20, 2.41, 2.67**. Apr 10 H2.20/C2.07/V6.36M and Apr 12 H2.17/C2.00/V4.62M form the lower failed-recovery shelf. Apr 8 H2.41/C2.27/V9.00M is the post-spike rejection. Apr 4 H2.67/C2.27/V27.32M is the dominant bearish peak on the largest visible volume. Applied A1-A5/B1/B2. Merged Sep 15 H2.36 into the later 2.41 region; excluded Apr 3 H2.53 on the upswing and today's H2.08, only a cent above close. The first zone is a relatively small rebound and is lower confidence.

### 19 SOFI — 2023-12-13, close 8.94

Selected: **9.19, 10.10, 11.70**. Sep 12 H9.19/C9.00/V25.56M and Sep 14 H9.18 cap a clear failed rally. Aug 4 H10.10/C9.50/V42.56M represents the short post-spike shelf around 10.10-10.28, also near Jul 19 H10.13; the chosen price is a shelf anchor, not its highest extreme. Jul 31 H11.70/C11.45/V211.00M is the top before the immediate reversal. Applied A1-A5/B1/B2/B4. Excluded Aug 1 H11.10 as a single decline-day high without an independent shelf, Jun 14 H10.23 as overlapping the later zone, and today's H8.96 as unconfirmed/near close. Shelf-price choice is uncertain and must not be described as a literal modal high: these few bars have different highs.

### 20 SOUN — 2023-07-06, close 3.52

Selected: **4.19, 4.53, 4.97, 5.11**. Feb 10 H4.19/C3.96/V18.39M and Feb 16 H4.18 form a repeated recovery ceiling, with several intervening highs around 4.01-4.09. Feb 23 H4.53/C3.92/V33.48M is a distinct failed rally, near Jun 26 H4.59. Feb 8 H4.97/C4.63/V65.20M and Jun 29 H5.11/C4.50/V28.14M are separate major peaks months apart, with a deep intervening trough. Applied A1-A5/B1. Excluded Jul 3 H4.75 as a small fluctuation inside the top's unwind, today's H3.99 as an unconfirmed decline-bar high, and separate 4.05/4.09 outputs inside the broad lower shelf. The June rally visibly revisits February supply; this does not establish future predictive success.

Validation predictions frozen in `val/predictions_vlm_audit_20261009.csv` before scoring. Rules A+B used; no validation labels were opened during selection in this rerun. Prior skill exposure remains a limitation.

## Validation evaluation and final rule update C

Both scorers agree after fixing the audit loader to accept the validation CSV's missing header. This parsing fix changes no predictions or matching policy. Coverage: 8 charts, 35 predictions, 33 labels, no zero-label input charts. Prediction SHA256: `a91d966249d3bec42eb8da84d198d8f8cfd411785dd77932078ccebd0f65826e`.

| Chart | TP | FP | FN | Set accuracy | Precision | Recall | F1 | FP levels | Missed labels |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 13 PTON | 3 | 0 | 2 | .6000 | 1.0000 | .6000 | .7500 | — | 5.37; 5.56 |
| 14 RIOT | 4 | 1 | 0 | .8000 | .8000 | 1.0000 | .8889 | 18.24 | — |
| 15 RIVN | 4 | 3 | 0 | .5714 | .5714 | 1.0000 | .7273 | 16.90; 17.56; 18.49 | — |
| 16 RKLB | 4 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — | — |
| 17 SNAP | 4 | 2 | 1 | .5714 | .6667 | .8000 | .7273 | 15.97; 17.90 | 15.52 |
| 18 SNDL | 3 | 0 | 1 | .7500 | 1.0000 | .7500 | .8571 | — | 2.17 |
| 19 SOFI | 3 | 0 | 1 | .7500 | 1.0000 | .7500 | .8571 | — | 11.10 |
| 20 SOUN | 3 | 1 | 0 | .7500 | .7500 | 1.0000 | .8571 | 4.19 | — |
| TOTAL | 28 | 7 | 5 | .7000 | .8000 | .8485 | .8235 | 7 extras | 5 missed |

The two policies remain identical on validation. These are one-pass validation scores, NOT post-label corrected scores.

Important signal audit findings:

- PTON labels 5.37 and 5.56 are consistent with January support lows: Jan 19 L5.37; Jan 17/22/31 L5.56, before the February gap down. They are not overhead pivot highs in the visible data. The high-only assumption omitted a real class of annotation.
- SNAP 15.52 matches May 21 L15.52, with May 15 L15.53 and May 22 L15.5512 nearby, broken May 23. The selected H15.97 was the wrong feature of the breakdown. This is why all OHLC fields, not just highs, must be inspected for shelf boundaries.
- SNDL labels both 2.17 and 2.20 despite a three-cent separation. One-to-one scoring penalizes merging them even though they lie within the same $0.10 zone. Keep this annotation granularity limitation explicit; do not spray near-duplicate outputs solely to collect hits.
- SOFI 11.10 shows that the isolated post-spike decline high can count; the B1 tightening sacrificed a label. No universal visual criterion separates all labelled and unlabelled decline highs in these examples.
- RIOT/RIVN/SOUN extras are plausible smaller rejection structures skipped by the labeler. More apparent pivots do not automatically improve precision.
- The prior skill already disclosed many validation values. The relatively high score is therefore not independent evidence of learning or generalization.

Final operational rules C (A+B retained except as amended):

C1. Amend A2/A8: inspect candle lows as well as highs. A repeatedly tested support shelf decisively broken, now above the final close, is eligible as potential overhead resistance even without a completed retest. Mark it explicitly as a **broken-support hypothesis**, not an observed failed retest. Use the dated shelf low, not the breakdown candle's high.
C2. The lookup helper now displays exact four-decimal OHLC values, including lows, while keeping window=0. It still only lists data; all level selections remain visual/manual. Scoring allows low anchors only for predictions explicitly marked `support`.
C3. Prefer major failed rally zones and sustained shelves over every short pause inside one unwind. Recentness alone is not strength. A weak inner shelf beneath a dominant wick is optional and must be identified as lower confidence.
C4. An immediate post-spike decline high can remain eligible when the top and lower shelf are clearly separated; treat it as a lower-confidence ledge, not a confirmed local pivot. Do not force such a level on every spike.
C5. Keep distinct episodes, merge redundant touches of one zone, and do not emit extra variants to exploit matching tolerance. There is no automatic level quota and no automatic requirement to retain every remote maximum.
C6. Freeze rules, predictions, and tolerance before evaluating on an independent, consistently labelled dataset.

The high-only lookup was expanded after validation feedback; training and validation entries above retain their original evidence and predictions. No revised train/val score is presented as if it were out-of-sample.

## Summary table

Micro-averaged level counts, not an average of per-chart F1 values. Training and validation use different sequential rule versions, so their scores are not directly comparable as one frozen detector.

| Split / policy | Charts | Labels | Predictions | Accuracy: TP/(TP+FP+FN) | TP | FP | FN (missed) | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train, flat or skill policy | 11 | 38 | 46 | 78.72% | 37 | 9 | 1 | 80.43% | 97.37% | .8810 |
| Validation, flat or skill policy | 8 | 33 | 35 | 70.00% | 28 | 7 | 5 | 80.00% | 84.85% | .8235 |

## Verification and reproducibility

New prediction files are `data/resistance_dataset/{train,val}/predictions_vlm_audit_20261009.csv`. Old prediction files and the resistance-detector skill were not edited. The only existing helper change is adding lows and four-decimal prices to `highs.py`; window=0 still performs data display, not detection. The new `evaluate_run.py` only validates and scores manually supplied predictions; it never generates levels.

From the repository root, reproduce each report with:

```text
py -3 data/vlm/claude_sonnet_5.5/evaluate_run.py train data/resistance_dataset/train/predictions_vlm_audit_20261009.csv
py -3 data/vlm/claude_sonnet_5.5/evaluate_run.py val data/resistance_dataset/val/predictions_vlm_audit_20261009.csv
```

The audit scorer validates PNG/CSV pairing, as-of dates, no future OHLCV rows, finite prediction prices strictly above close, no duplicate prediction prices, and an OHLCV high anchor (or low for an explicitly marked support prediction). Anchor allowance is half a cent plus a small margin because the initial training/validation lookup printed prices to cents. It includes zero-label charts in the evaluation universe and warns about duplicate/non-overhead labels without silently changing them. It handles label CSVs with or without headers.

Nine synthetic scoring checks passed: flat tolerance boundary/ineligible distance, one-to-one duplicate-nearby matching, relative tolerance boundary/ineligible distance, empty predictions, empty labels, and undefined all-empty metrics. An initial import attempt used an invalid dotted-directory module path; it was corrected by adding the actual scorer directory to Python's import path. The validation loader's header issue was corrected and cross-checked against the original scorer. Neither recovery changed predictions.

Final verification passed for the training and validation splits: coverage/anchor/as-of checks, unchanged prediction SHA256 values, and exact agreement with the original scorer's TP/FP/FN totals under both policies. The audit contains 19 dated chart sections and no pending placeholders.

## Limitations and live-use assessment

All 19 training and validation PNGs were opened, with OHLCV lookup evidence and a per-chart record. This file documents observable justifications and uncertainty; it does not establish that resistance caused or will cause price reversals. Volume is corroborative only. There is no return-performance analysis, execution model, transaction-cost analysis, risk-control review, probability calibration, or repeated-run reproducibility study here.

**These results do not justify autonomous live-trading deployment.** Before that, use an untouched, chronological and non-overlapping holdout; adjudicate ambiguous or invalid labels; confirm price-adjustment conventions and suspect bars; fix the evaluation tolerance in advance; and measure repeated-run consistency plus forward/paper-trading behavior with independent risk controls. Do not select the more flattering tolerance after observing performance. No trades or brokerage actions were performed.

