#!/usr/bin/env python3
"""FIG17 — seed variance (`03_RESULTS_SPEC.md` appendix figures): per-seed values of the project's
headline number (TPR@1%FPR, trajectory ablation, elapsed=0, `full` view -- the standard,
most-immediate MIA setting) for every (dataset, method), so nobody has to wonder how much a single
reported point moves across seeds. Reads the already-computed post-fix
`results/a1_lira_pertask_<dataset>_<method>_seed<N>_full.csv` files (`run_lira_pertask.py`'s §7a
fixed-k-pooled output -- no new shadow generation / attack runs) and writes
`results/fig17_seed_variance.csv` (`dataset, method, metric, seed, value`), per
`03_RESULTS_SPEC.md`'s schema.

2026-09-23 (`08_FIX_PLAN.md` FX6): repointed from the pre-fix `a1_lira_<dataset>_<method>[_seed<N>].csv`
files (deleted -- superseded by FX2's per-task, fixed-k-population pipeline; see
`notes/2026-09-22_fx2_stale_shadow_dir_bug.md`) to the current `a1_lira_pertask_*_full.csv` files.
M0 has 5 seeds (0-4, its pre-existing gold-standard budget); every other method has 3 (0-2, wave V2's
budget) -- both satisfy CLAUDE.md non-negotiable #4's >=3-seed floor, missing seeds are skipped, not
padded or inferred.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from fx9_io import current  # noqa: E402
from p3fcl import provenance  # noqa: E402
from p3fcl.experiment import CHANGED_METHODS  # noqa: E402

DATASETS = ["cifar100", "cub200", "imagenet_r"]
METHODS = ["m0_fedavg", "m1_glfc", "m2_target", "m3_fot", "m4_proto", "m5_hybrid_replay", "m8_analytic"]
SEEDS = [0, 1, 2, 3, 4]
METRIC = "tpr_at_1pct_fpr_elapsed0_trajectory"


def _tpr_at_elapsed0(csv_path: Path) -> float | None:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["ablation"] == "trajectory" and r["task_k"] == "pooled" and int(r["elapsed"]) == 0:
            return float(r["tpr1"])
    return None


def main() -> int:
    out_rows = []
    for dataset in DATASETS:
        for method in METHODS:
            for seed in SEEDS if method == "m0_fedavg" else [0, 1, 2]:
                csv_path = REPO_ROOT / "results" / f"a1_lira_pertask_{dataset}_{method}_seed{seed}_full.csv"
                assert csv_path.exists(), csv_path
                if method in CHANGED_METHODS:
                    current(csv_path)
                value = _tpr_at_elapsed0(csv_path)
                assert value is not None, csv_path
                out_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": METRIC,
                        "seed": seed,
                        "value": value,
                    }
                )

    assert len(out_rows) == 69
    out_csv = REPO_ROOT / "results" / "fig17_seed_variance.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "metric", "seed", "value"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {out_csv}")

    config = {
        "seed": 0,
        "purpose": "FIG17 seed-variance panel: per-seed TPR@1%FPR at elapsed=0",
        "datasets": DATASETS,
        "methods": METHODS,
        "n_seeds": len(SEEDS),
    }
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
