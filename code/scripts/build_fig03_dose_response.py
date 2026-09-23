#!/usr/bin/env python3
"""FX8 (`08_FIX_PLAN.md` §10): `results/fig03_dose_response.csv`, per `03_RESULTS_SPEC.md`'s FIG03
schema (`dataset, method, knob_name, knob_value, retention_bwt, tpr1, tpr01, final_acc, seed, ci_lo,
ci_hi`). Pure join of `run_dose_response_accuracy.py`'s and `run_dose_response_lira.py`'s already-
computed outputs on `(dataset, method, knob_name, knob_value, seed)` -- no new computation, per
CLAUDE.md non-negotiable #3. `ci_lo`/`ci_hi` are `tpr1`'s own Clopper-Pearson interval (that row's own
CI, not a cross-seed aggregate -- the plot script does cross-seed aggregation from these raw per-seed
rows if it needs one, matching this project's established raw-CSV-carries-per-seed-CI convention).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    acc_path = REPO_ROOT / "results" / "dose_response_accuracy.csv"
    lira_path = REPO_ROOT / "results" / "dose_response_lira.csv"
    if not acc_path.exists() or not lira_path.exists():
        print(f"missing {acc_path if not acc_path.exists() else lira_path} -- run both "
              f"run_dose_response_accuracy.py and run_dose_response_lira.py first")
        return 1

    acc_by_key = {
        (r["dataset"], r["method"], r["knob_name"], r["knob_value"], r["seed"]): r
        for r in _read_csv(acc_path)
    }

    rows = []
    for lr in _read_csv(lira_path):
        key = (lr["dataset"], lr["method"], lr["knob_name"], lr["knob_value"], lr["seed"])
        acc_row = acc_by_key.get(key)
        if acc_row is None:
            print(f"(no accuracy row for {key}, skipping)")
            continue
        rows.append({
            "dataset": lr["dataset"], "method": lr["method"], "knob_name": lr["knob_name"],
            "knob_value": lr["knob_value"], "retention_bwt": acc_row["retention_bwt"],
            "tpr1": lr["tpr1"], "tpr01": lr["tpr01"], "final_acc": acc_row["final_acc"],
            "seed": lr["seed"], "ci_lo": lr["ci_lo"], "ci_hi": lr["ci_hi"],
        })

    if not rows:
        print("no matching (accuracy, leakage) rows -- nothing written")
        return 1

    out_csv = REPO_ROOT / "results" / "fig03_dose_response.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {"seed": 0, "purpose": "FX8 FIG03 v2 dose-response join (claim C2, H2)"}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
