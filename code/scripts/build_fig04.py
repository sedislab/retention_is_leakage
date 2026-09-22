#!/usr/bin/env python3
"""FIG04 — semantic vs. individual retention (claim C2, the Red Team's objection --
`00_BUILD_PLAN.md` P5, `agents/OPEN_QUESTIONS.md` H2). Merges the two dose-response pilot arms
already run separately (`run_dose_response_pilot.py` for M5/individual,
`run_dose_response_pilot_m4.py` for M4/semantic) into one CSV so they can be plotted together. No new
computation -- reads the two existing per-arm CSVs, tags each with its method's own
`MethodSpec.retention_type` (not hand-typed), writes `results/fig04_semantic_vs_individual.csv`.

**This remains a PILOT, same scope caveats as `results/fig03_dose_response_pilot.csv`** (3 levels,
3 seeds, reduced shadow budget, 1 dataset) -- see `notes/2026-09-21_p5_dose_response_pilot.md` and
its M4-arm companion note. What FIG04 adds is *not* more scale, it is the qualitative
semantic-vs-individual split `00_BUILD_PLAN.md` explicitly asks P5 to test.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl.methods.m4_proto import PrototypeFCL  # noqa: E402
from p3fcl.methods.m5_hybrid_replay import HybridReplay  # noqa: E402


def main() -> int:
    m5_csv = REPO_ROOT / "results" / "fig03_dose_response_pilot.csv"
    m4_csv = REPO_ROOT / "results" / "fig04_semantic_arm_m4.csv"
    if not m5_csv.exists() or not m4_csv.exists():
        print(f"need both {m5_csv} and {m4_csv} -- one is missing, nothing to merge")
        return 1

    with open(m5_csv, newline="") as f:
        m5_rows = list(csv.DictReader(f))
    with open(m4_csv, newline="") as f:
        m4_rows = list(csv.DictReader(f))

    for r in m5_rows:
        r["retention_type"] = HybridReplay.spec.retention_type
    # m4_rows already carry retention_type from run_dose_response_pilot_m4.py, but recompute from
    # the method spec directly so this file can never silently drift from the method's own declaration
    for r in m4_rows:
        r["retention_type"] = PrototypeFCL.spec.retention_type

    fieldnames = [
        "dataset", "method", "retention_type", "knob_value", "n_seeds",
        "retention_bwt_mean", "retention_bwt_ci_lo", "retention_bwt_ci_hi",
        "tpr1_mean", "tpr1_ci_lo", "tpr1_ci_hi", "elapsed",
    ]
    out_rows = []
    for r in m5_rows + m4_rows:
        out_rows.append({k: r.get(k, "") for k in fieldnames})

    out_csv = REPO_ROOT / "results" / "fig04_semantic_vs_individual.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {out_csv} "
          f"({len(m5_rows)} individual/M5 + {len(m4_rows)} semantic/M4)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
