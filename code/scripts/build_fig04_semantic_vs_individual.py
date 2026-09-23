#!/usr/bin/env python3
"""FX8 (`08_FIX_PLAN.md` §10): `results/fig04_semantic_vs_individual.csv`, per
`03_RESULTS_SPEC.md`'s FIG04 schema (`dataset, method, retention_type{semantic|individual},
knob_value, tpr1, retention_bwt, seed, ci_lo, ci_hi`). Reads `results/fig03_dose_response.csv`
(already the join of accuracy + leakage) and tags each row with `MethodSpec.retention_type` (not
hand-typed, matching `build_fig04.py`'s own established pattern) -- no new computation.

`retention_type` per method (`p3fcl.methods.*`): M5 = "individual" (raw exemplar replay); M2 =
"semantic" (a per-class distributional summary -- fixed from a latent "individual" mislabeling while
building this, see `code/src/p3fcl/methods/m2_target.py`'s own comment and
`notes/2026-09-23_fx8_dose_response_design_notes.md`). This is the actual Red Team split this figure
answers: "retention is semantic, membership is individual" (`04_METHODS_AND_ATTACKS.md` §... the
objections table) -- if the dose-response effect holds only for the individual-retention group
(M5), that supports the objection; if it holds for both, it does not.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402
from p3fcl.methods.m2_target import TARGET  # noqa: E402
from p3fcl.methods.m5_hybrid_replay import HybridReplay  # noqa: E402

_RETENTION_TYPE = {"m5_hybrid_replay": HybridReplay.spec.retention_type, "m2_target": TARGET.spec.retention_type}


def main() -> int:
    fig03_path = REPO_ROOT / "results" / "fig03_dose_response.csv"
    if not fig03_path.exists():
        print(f"{fig03_path} does not exist yet -- run build_fig03_dose_response.py first")
        return 1
    with open(fig03_path, newline="") as f:
        fig03_rows = list(csv.DictReader(f))

    rows = []
    for r in fig03_rows:
        retention_type = _RETENTION_TYPE.get(r["method"])
        if retention_type is None:
            continue
        rows.append({
            "dataset": r["dataset"], "method": r["method"], "retention_type": retention_type,
            "knob_value": r["knob_value"], "tpr1": r["tpr1"], "retention_bwt": r["retention_bwt"],
            "seed": r["seed"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
        })

    if not rows:
        print("no rows matched a known method -- nothing written")
        return 1

    out_csv = REPO_ROOT / "results" / "fig04_semantic_vs_individual.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv} "
          f"(retention_type: {sorted({r['retention_type'] for r in rows})})")

    config = {"seed": 0, "purpose": "FX8 FIG04 v2 semantic-vs-individual split (claim C2, Red Team objection)"}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
