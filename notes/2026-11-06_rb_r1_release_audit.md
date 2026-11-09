# 2026-11-06: Rebuttal R1, release-only audit of M2/M3 (EVIDENCE: results/rb_release_audit.csv)

Protocol: `rebuttal/R1_PROTOCOL.md` (fixed before running; amendment 1 = float32 equivalence check
only). The pipeline is paired with the paper: every `margin_repro` row equals
`a1_lira_fixedk_summary_<ds>_<method>_full.csv` exactly.

Prediction (fixed in advance): release-only TPR@1%FPR at e = 0 ≥ the paper's margin TPR@1%FPR at
e = 6 (trajectory). **It holds in 12/12 cells.** TPR@1%FPR, trajectory ablation, 3 seeds, 2,000-replicate
hierarchical bootstrap:

| dataset | method | view | release e=0 | paper margin e=0 → e=6 |
|---|---|---|---|---|
| cifar100 | M2 | broadcast | 0.244 | 0.032 → 0.042 |
| cifar100 | M2 | client | 0.159 (wide CI 0.12–0.44) | |
| cifar100 | M3 | broadcast (8-dim subspace) | 0.050 | 0.029 → 0.038 |
| cifar100 | M3 | server (task covariance) | 0.999 | |
| cub200 | M2 | broadcast | 0.995 | 0.076 → 0.329 |
| cub200 | M2 | client | 0.862 | |
| cub200 | M3 | broadcast | 0.181 | 0.062 → 0.090 |
| cub200 | M3 | server | 1.000 | |
| imagenet_r | M2 | broadcast | 0.647 | 0.095 → 0.235 |
| imagenet_r | M2 | client | 0.259 | |
| imagenet_r | M3 | broadcast | 0.102 | 0.079 → 0.090 |
| imagenet_r | M3 | server | 1.000 | |

Reading (for the authors):
- M2's per-class Gaussian broadcast and M3's task covariance leak far more at their own task than the
  paper's F1 margin attack reaches after 6 more tasks. The paper's growth curves for M2/M3 are an
  attack-strength effect, as the data-processing argument requires. The paper's F1-only audit of
  M2/M3 was therefore a loose lower bound on their leakage. That changes the reading of M2/M3, but
  no paper number changes.
- The combined arms (margin + broadcast LLR) do not improve on the broadcast release alone, and are
  sometimes slightly below it at low FPR. `combined_once` falls with e, because the cumulative margin
  trajectory adds noise. Reported as observed.
- The M2 client view drops NaN scores (OUT targets whose client holds no other example of their
  class; CUB about 11%). Its TPR is therefore conservative.
