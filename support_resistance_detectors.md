Here's the landscape, grouped by algorithm family — with the closest analogues to what you already have:

## Pivot / fractal + clustering (closest to your detectors)

- day0market/support_resistance — ZigZag pivots or raw highs/lows merged via AgglomerativeClustering, with level scoring. Probably the most directly comparable to your pivot_resistance + merge step.
- albertsl/support-resistance_trading-bot (support_resistance_finder.py) — a grab-bag worth mining: elementary derivative extrema, MeanShift clustering, pivot points, volume-based levels, and CryptoCred-style trendlines, each returning lines + a likelihood score.
- KMeans variants — judopro/Stock_Support_Resistance_ML (elbow method for K on highs/lows separately) and pablob206/pkg-support-resistance (vanilla clustering, KMeans, DBSCAN; emits zones + a volume-weighted score rather than single lines).

## Trendline / regression methods (diagonal S/R)

- GregoryMorse/trendln (~695★) — calc_support_resistance finds local extrema then fits support/resistance trendlines several ways; the most mature OSS option in this family.
- ednunezg/pytrendline — exhaustive O(N³) scan over candidate lines through pivots, scored by touches and body-crossing tolerance. Good for short windows.
- ryanllu/Trendline — residual-percentile-based support/resistance line fitting.

## Building blocks people roll their own from

- scipy.signal.find_peaks / argrelextrema — the generic pivot detector most of these wrap.
- ZigZag implementations (zigzag package) — filters noise by min % reversal before pivoting.
- KDE over pivot prices — density peaks become levels; common in blog implementations, less common as a package.
- Volume profile / market profile — you already have this (volume_profile.py); the OSS equivalent is volume-at-price histograms with HVN peaks, plus TPO/market-profile libs.

## Pine Script (if you're willing to port)

- LuxAlgo / open Pine indicators and the casoon/pine-scripts S/R zone engine — multi-detector zone registry (pivot clusters, order blocks, FVGs, liquidity, supply/demand) with confluence scoring. Interesting because it treats each signal family as an independent "detector" emitting evidence — similar in spirit to your sources/scoring design.

## Honest assessment

Nothing in OSS is clearly stronger than what you have — most repos are weekend projects with no labeled evaluation. The ones worth actually benchmarking on resistance_dataset are day0market's ZigZag+Agglomerative approach (zigzag pivots may fix your "interior staircase" FN problem that pure ±4-bar pivots miss) and MeanShift (bandwidth auto-selected, avoids your fixed merge_tol). If you want, I can wire one into eval_pivot_resistance.py as another --detector option and score it on the test split.