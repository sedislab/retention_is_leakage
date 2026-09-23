#!/usr/bin/env python3
"""FX3 (`08_FIX_PLAN.md` §5): `results/fx3_views_summary.csv`. Reads `results/a1_lira_fixedk_summary.csv`
(already computed, CLAUDE.md non-negotiable #3 -- no recomputation), restricted to the primary
`ablation=trajectory` (H3's transcript arm) since FX3's schema has no ablation column of its own.

For F1 methods (M0, M1, M2, M3, M5), only the `full` view exists in the source data -- FX4h's own
design: the logit-margin attack only ever reads the global model, which is identical under secure
aggregation once weights are exact (Prop. 3), so there is nothing distinct to compute for `aggregate`/
`global`. This script adds explicit `aggregate` and `global` rows for F1 methods that COPY the `full`
row's numbers, tagged `identical_by_construction=True`, so FIG11 v2's grouped-bar chart can draw all
three bars for every method group without a family-conditional special case in the plotting code --
the plotting code just checks that one flag.

Columns: `dataset, method, family, view, elapsed, auc, auc_ci_lo, auc_ci_hi, tpr1, tpr1_ci_lo,
tpr1_ci_hi, tpr01, tpr01_ci_lo, tpr01_ci_hi, n_seeds, n_shadows, identical_by_construction`.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402

F1_METHODS = {"m0_fedavg", "m1_glfc", "m2_target", "m3_fot", "m5_hybrid_replay"}


def main() -> int:
    src = REPO_ROOT / "results" / "a1_lira_fixedk_summary.csv"
    with open(src, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["ablation"] == "trajectory"]

    out_rows = []
    for r in rows:
        base = {
            "dataset": r["dataset"], "method": r["method"], "family": r["family"],
            "elapsed": r["elapsed"], "auc": r["auc"], "auc_ci_lo": r["auc_ci_lo"], "auc_ci_hi": r["auc_ci_hi"],
            "tpr1": r["tpr1"], "tpr1_ci_lo": r["tpr1_ci_lo"], "tpr1_ci_hi": r["tpr1_ci_hi"],
            "tpr01": r["tpr01"], "tpr01_ci_lo": r["tpr01_ci_lo"], "tpr01_ci_hi": r["tpr01_ci_hi"],
            "n_seeds": r["n_seeds"], "n_shadows": r["n_shadows"],
        }
        if r["method"] in F1_METHODS:
            for view in ("full", "aggregate", "global"):
                out_rows.append({**base, "view": view, "identical_by_construction": "True" if view != "full" else ""})
        else:
            out_rows.append({**base, "view": r["view"], "identical_by_construction": ""})

    out_csv = REPO_ROOT / "results" / "fx3_views_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {out_csv}")

    config = {"seed": 0, "purpose": "FX3 views summary (secure-aggregation views, FIG11/FIG19)"}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
