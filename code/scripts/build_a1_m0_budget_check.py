#!/usr/bin/env python3
"""FX2 (`08_FIX_PLAN.md` §7d): `results/a1_m0_budget_check.csv`, an appendix robustness row --
M0 at the 1024-shadow/3-seed budget every other method got in wave V2, against M0's existing
4096-shadow/5-seed store (seeds 0-2 only, to compare like-for-like against the restricted run's
seeds). Confirms wave V2's smaller shadow budget doesn't systematically bias the leak measurement
relative to the "gold standard" M0 baseline it's calibrated against.

Reads only already-computed per-seed CSVs (`run_lira_pertask.py`'s output, both the full-budget and
`max_shadows=1024`-restricted runs) -- CLAUDE.md non-negotiable #3, no recomputation of the attack
itself. `run_lira_pertask.py cifar100 m0_fedavg <seed> full 1024` must have already been run for
seeds 0-2 before this script does anything useful.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402

DATASET = "cifar100"
METHOD = "m0_fedavg"
SEEDS = [0, 1, 2]
E_CHECK = [0, 6]


def _pooled_rows(seed: int, suffix: str) -> dict:
    path = REPO_ROOT / "results" / f"a1_lira_pertask_{DATASET}_{METHOD}_seed{seed}_full{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist -- run run_lira_pertask.py for this (seed, budget) first")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["task_k"] == "pooled" and r["ablation"] == "trajectory"]
    return {int(r["elapsed"]): r for r in rows}


def main() -> int:
    out_rows = []
    for e in E_CHECK:
        full_tpr1s, restricted_tpr1s = [], []
        full_aucs, restricted_aucs = [], []
        for seed in SEEDS:
            full = _pooled_rows(seed, "")
            restricted = _pooled_rows(seed, "_max1024")
            if e not in full or e not in restricted:
                continue
            full_tpr1s.append(float(full[e]["tpr1"]))
            restricted_tpr1s.append(float(restricted[e]["tpr1"]))
            full_aucs.append(float(full[e]["auc"]))
            restricted_aucs.append(float(restricted[e]["auc"]))

        out_rows.append({
            "dataset": DATASET, "method": METHOD, "elapsed": e,
            "full_budget_n_shadows": 4096, "full_budget_n_seeds": len(full_tpr1s),
            "full_budget_tpr1_mean": float(np.mean(full_tpr1s)) if full_tpr1s else "",
            "full_budget_auc_mean": float(np.mean(full_aucs)) if full_aucs else "",
            "restricted_budget_n_shadows": 1024, "restricted_budget_n_seeds": len(restricted_tpr1s),
            "restricted_budget_tpr1_mean": float(np.mean(restricted_tpr1s)) if restricted_tpr1s else "",
            "restricted_budget_auc_mean": float(np.mean(restricted_aucs)) if restricted_aucs else "",
            "tpr1_diff": (float(np.mean(restricted_tpr1s) - np.mean(full_tpr1s)) if full_tpr1s and restricted_tpr1s else ""),
            "auc_diff": (float(np.mean(restricted_aucs) - np.mean(full_aucs)) if full_aucs and restricted_aucs else ""),
        })

    out_csv = REPO_ROOT / "results" / "a1_m0_budget_check.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {out_csv}")
    for r in out_rows:
        print(f"  e={r['elapsed']}: full tpr1={r['full_budget_tpr1_mean']}, "
              f"restricted(1024) tpr1={r['restricted_budget_tpr1_mean']}, diff={r['tpr1_diff']}")

    config = {"seed": 0, "purpose": "FX2 M0 shadow-budget robustness check"}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
