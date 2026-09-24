# FX9 bootstrap runtime improvement

Evidence: PBS184430 read unchanged CIFAR/M0 evaluated score sidecars and checked32 resampled real-score/FPR cases. The conservative threshold TPR is identical (maximum error0); the threshold-sorting reference took5.479847s and the order-statistic computation0.446288s (12.279× for this component). No attack, calibration split, chance floor, resampling draw, or public metrics function changed.

For n negative scores and FPR budget alpha, let k be the largest integer with k/n <= alpha, using exactly the reference floating comparison. The (n-k)-th ascending negative score cannot be admitted without exceeding the budget. Counting positives strictly above it gives the same best admissible threshold, including ties. Tests compare the existing reference on continuous/tied/constant scores and adjacent floating-point budget boundaries.

Only bootstrap metric calls use this helper. All318 tests passed (13.44s). Twelve completed non-M0 bootstrap outputs were kept. The three unfinished M0 jobs in184371 were replaced by184431 after moving downstream dependencies; their roughly34minutes of incomplete work were discarded to shorten the remaining critical path. Existing M0/M3/M8 score and per-combo evidence remains protected by checksum acceptance.
