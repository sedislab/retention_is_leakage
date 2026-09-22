#!/usr/bin/env python3
"""H11 secure-aggregation PILOT report: calibrated TPR@1%FPR for A1 (LiRA) on M0/CIFAR-100 under the
`adversary_view=secure_agg` shadows (`shadows/cifar100/m0_fedavg/secure_agg/seed<N>/`), directly
comparable against the existing full-view TPR numbers already in
`results/a1_lira_cifar100_m0_fedavg[_seed<N>].csv` at the same elapsed values. Reuses `run_lira.py`'s
own `load_shadow_store`/`compute_log_lr_surfaces` (same calibration-split logic, same math) -- this
is the same report `run_lira.py` would produce, just pointed at a different shadow directory, so the
two are apples-to-apples.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

import run_lira  # noqa: E402
from p3fcl import metrics  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402

DATASET = "cifar100"
METHOD = "m0_fedavg"
SEEDS = [0, 1, 2]
ELAPSED_GRID = [0, 5, 9]  # match the pilot's usual mid-horizon read-out plus the endpoints


def _shadow_dir(seed: int) -> Path:
    return REPO_ROOT / "shadows" / DATASET / METHOD / "secure_agg" / f"seed{seed}"


def _report_for_seed(seed: int, calib_frac: float, seed_cfg: int) -> dict:
    store = run_lira.load_shadow_store(_shadow_dir(seed))
    targets = store["targets"]
    n_shadows = len(store["shadow_ids"])
    r = rng_mod.seeded(f"run_secure_agg_pilot_report::calib_split::{DATASET}::{METHOD}::{seed}", seed_cfg)
    perm = r.permutation(n_shadows)
    n_calib = int(round(calib_frac * n_shadows))
    calib_idx, eval_idx = perm[:n_calib], perm[n_calib:]

    surfaces = run_lira.compute_log_lr_surfaces(store, calib_idx)
    traj_eval = surfaces["trajectory"][eval_idx]
    in_out_eval = store["in_out"][eval_idx]
    n_rounds = store["n_rounds"]

    out = {}
    for elapsed in ELAPSED_GRID:
        scores_list, labels_list = [], []
        for j, t in enumerate(targets):
            T = t["task"] + elapsed
            if not (0 <= T < n_rounds):
                continue
            col = traj_eval[:, j, T]
            keep = ~np.isnan(col)
            scores_list.append(col[keep])
            labels_list.append(in_out_eval[keep, j])
        if not scores_list:
            out[elapsed] = float("nan")
            continue
        scores_arr = np.concatenate(scores_list)
        labels_arr = np.concatenate(labels_list).astype(int)
        report = metrics.membership_report(scores_arr, labels_arr)
        out[elapsed] = report["tpr_at_1pct_fpr"]
    return out


def main() -> int:
    with open(REPO_ROOT / "code" / "configs" / "attack_lira.yaml") as f:
        cfg = yaml.safe_load(f)
    calib_frac = float(cfg["shadow"]["calibration_frac"])
    seed_cfg = int(cfg["seed"])

    rows = []
    for seed in SEEDS:
        shadow_dir = _shadow_dir(seed)
        if not shadow_dir.exists():
            print(f"(skipping seed={seed}: {shadow_dir} not found)")
            continue
        per_elapsed = _report_for_seed(seed, calib_frac, seed_cfg)
        for elapsed, tpr1 in per_elapsed.items():
            rows.append({"dataset": DATASET, "method": METHOD, "adversary_view": "secure_agg",
                         "seed": seed, "elapsed": elapsed, "tpr1": tpr1})
            print(f"seed={seed} elapsed={elapsed}: tpr1={tpr1:.4f}")

    if not rows:
        print("no seeds had complete data -- nothing written")
        return 1

    out_csv = REPO_ROOT / "results" / "secure_agg_pilot_m0.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    # Print the direct comparison against the existing full-view numbers, if present.
    print("\n=== comparison against existing full-view TPR@1%FPR ===")
    for seed in SEEDS:
        suffix = f"_seed{seed}" if seed != 0 else ""
        full_csv = REPO_ROOT / "results" / f"a1_lira_{DATASET}_{METHOD}{suffix}.csv"
        if not full_csv.exists():
            continue
        with open(full_csv, newline="") as f:
            full_rows = list(csv.DictReader(f))
        for elapsed in ELAPSED_GRID:
            full_row = next(
                (r for r in full_rows if r["ablation"] == "trajectory" and int(r["elapsed"]) == elapsed), None
            )
            secure_row = next((r for r in rows if r["seed"] == seed and r["elapsed"] == elapsed), None)
            if full_row and secure_row:
                print(f"seed={seed} elapsed={elapsed}: full={float(full_row['tpr_at_1pct_fpr']):.4f} "
                      f"secure_agg={secure_row['tpr1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
