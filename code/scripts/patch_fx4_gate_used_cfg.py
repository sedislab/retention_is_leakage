#!/usr/bin/env python3
"""One-off patch: job 183846 (FX4g gate) ran from a `run_fx4g_gate.py` version that predates adding
the `used_cfg_json` column (added the same day, after submission, for FX4i's wave V2 manifest builder
to consume directly instead of re-deriving "tuned" from `fx4_gate_grid.csv`'s selection rule -- see
`build/STATE.md`'s FX4g note). Re-running the whole gate would waste real compute for a column that is
fully reconstructible from files that already exist: `results/fx4_gate.csv`'s `config_id` says which
rows are "tuned", and `results/fx4_gate_grid.csv` has every candidate this run's own selection rule
(`run_fx4g_gate.py`: best `val_final_avg_acc` among rows with `val_bwt <= 0.02`) considered. This
script replays that exact rule to fill in `used_cfg_json` for the already-completed run, in place.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CFG = {
    "m0_fedavg": {"local_epochs": 30, "lr": 0.5},
    "m1_glfc": {"local_epochs": 30, "lr": 0.5, "exemplar_budget": 10, "distillation_weight": 1.0, "temperature": 2.0},
    "m2_target": {"local_epochs": 30, "lr": 0.5, "replay_ratio": 1.0, "n_synthetic_per_class": 20},
    "m3_fot": {"local_epochs": 30, "lr": 0.5, "subspace_rank": 8, "projection_strength": 1.0},
    "m5_hybrid_replay": {"local_epochs": 30, "lr": 0.5, "buffer_size_per_class": 10},
}


def _best_tuned_cfg(grid_rows: list, dataset: str, method: str) -> dict | None:
    candidates = [r for r in grid_rows if r["dataset"] == dataset and r["method"] == method]
    valid = [r for r in candidates if float(r["val_bwt"]) <= 0.02]
    if not valid:
        return None
    best = max(valid, key=lambda r: float(r["val_final_avg_acc"]))
    return {
        **DEFAULT_CFG[method],
        "lr": float(best["lr"]),
        "local_epochs": int(best["local_epochs"]),
        "distillation_weight": float(best["distillation_weight"]),
    }


def main() -> int:
    gate_csv = REPO_ROOT / "results" / "fx4_gate.csv"
    grid_csv = REPO_ROOT / "results" / "fx4_gate_grid.csv"

    with open(gate_csv, newline="") as f:
        gate_rows = list(csv.DictReader(f))
    if "used_cfg_json" in gate_rows[0]:
        print("fx4_gate.csv already has used_cfg_json -- nothing to patch")
        return 0

    with open(grid_csv, newline="") as f:
        grid_rows = list(csv.DictReader(f))

    n_tuned_filled = 0
    for row in gate_rows:
        if row["config_id"] == "tuned":
            tuned = _best_tuned_cfg(grid_rows, row["dataset"], row["method"])
            if tuned is None:
                raise RuntimeError(
                    f"{row['dataset']}/{row['method']}: config_id=='tuned' but no grid row has "
                    f"val_bwt<=0.02 -- this should be impossible (tuned only happens when `best` was found)"
                )
            row["used_cfg_json"] = json.dumps(tuned, sort_keys=True)
            n_tuned_filled += 1
        else:
            row["used_cfg_json"] = json.dumps(DEFAULT_CFG[row["method"]], sort_keys=True)

    fieldnames = list(gate_rows[0].keys())
    with open(gate_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(gate_rows)
    print(f"patched {gate_csv}: {n_tuned_filled} 'tuned' rows filled from fx4_gate_grid.csv's own selection rule; {len(gate_rows) - n_tuned_filled} default/baseline rows filled from DEFAULT_CFG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
