#!/usr/bin/env python3
"""FIG18 v2 report (`08_FIX_PLAN.md`'s H13 fix, matched-5-client redesign): calibrated TPR@1%FPR (A1)
+ real -BWT, for M0/Camelyon17 under both the natural (`client_field="slide"`, n_clients=5) and
Dirichlet-subpartitioned (n_clients=5, matched) shadow stores
(`shadows_v2/camelyon17/m0_fedavg/fig18_v2_<partition>/seed<N>/` -- NOT the old, pre-redesign
`shadows/camelyon17/.../fig18_<partition>/` stores, which used the confounded n_clients=1-vs-10
comparison this redesign replaces), plus the already-computed real accuracy matrix
(`results/accuracy_matrix_camelyon17.csv`, also rebuilt for the matched-5-client redesign). Reuses
`run_lira.py`'s own `load_shadow_store`/`compute_log_lr_surfaces`. Writes
`results/fig18_natural_federation.csv` (schema per `03_RESULTS_SPEC.md` FIG18: `partition, beta,
n_clients, method, family, elapsed, acc, tpr1, seed, ci_lo, ci_hi` -- long format, aggregated over
seeds with `metrics.seed_ci`; `n_clients` is new in v2, so the CSV itself shows the comparison is now
matched on client count, not just the accompanying prose).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

import run_lira  # noqa: E402
from p3fcl import metrics, provenance  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402

DATASET = "camelyon17"
METHOD = "m0_fedavg"
N_CLIENTS = 5  # matched across both arms -- the v2 redesign's whole point
PARTITIONS = {"natural": ""}
PARTITIONS["dirichlet"] = 0.5  # beta column: blank for natural (no Dirichlet draw), 0.5 for dirichlet arm
SEEDS = [0, 1, 2]
ELAPSED_GRID = [0, 1, 2, 3, 4]  # 5 hospitals -> horizon 4
CALIB_FRAC = 0.8


def _shadow_dir(partition: str, seed: int) -> Path:
    return REPO_ROOT / "shadows_v2" / DATASET / METHOD / f"fig18_v2_{partition}" / f"seed{seed}"


def _tpr1_per_elapsed(partition: str, seed: int) -> dict:
    store = run_lira.load_shadow_store(_shadow_dir(partition, seed))
    targets = store["targets"]
    n_shadows = len(store["shadow_ids"])
    r = rng_mod.seeded(f"run_fig18_report::calib_split::{DATASET}::{METHOD}::{partition}::{seed}", 0)
    perm = r.permutation(n_shadows)
    n_calib = int(round(CALIB_FRAC * n_shadows))
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
            continue
        scores_arr = np.concatenate(scores_list)
        labels_arr = np.concatenate(labels_list).astype(int)
        report = metrics.membership_report(scores_arr, labels_arr)
        out[elapsed] = report["tpr_at_1pct_fpr"]
    return out


def _acc_per_elapsed(partition: str) -> dict:
    """Reads the already-real, non-shadow accuracy matrix -- mean over (task_k, seed) pairs at each
    elapsed value, matching FIG01's own normalisation convention (acc/first-value)."""
    with open(REPO_ROOT / "results" / "accuracy_matrix_camelyon17.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["partition"] == partition]
    by_k_seed: dict = {}
    for r in rows:
        by_k_seed.setdefault((r["task_k"], r["seed"]), {})[int(r["elapsed"])] = float(r["acc"])
    normed_by_elapsed: dict = {}
    for (_, _), by_elapsed in by_k_seed.items():
        if 0 not in by_elapsed or by_elapsed[0] == 0:
            continue
        base = by_elapsed[0]
        for e, acc in by_elapsed.items():
            normed_by_elapsed.setdefault(e, []).append(acc / base)
    return {e: float(np.mean(v)) for e, v in normed_by_elapsed.items()}


def main() -> int:
    rows = []
    for partition, beta in PARTITIONS.items():
        per_seed_tpr = {e: [] for e in ELAPSED_GRID}
        for seed in SEEDS:
            shadow_dir = _shadow_dir(partition, seed)
            if not shadow_dir.exists():
                print(f"(skipping partition={partition} seed={seed}: {shadow_dir} not found)")
                continue
            tpr_by_elapsed = _tpr1_per_elapsed(partition, seed)
            for e, tpr in tpr_by_elapsed.items():
                per_seed_tpr[e].append(tpr)
            print(f"partition={partition} seed={seed}: " +
                  " ".join(f"e{e}={v:.4f}" for e, v in sorted(tpr_by_elapsed.items())))

        acc_by_elapsed = _acc_per_elapsed(partition)
        for e in ELAPSED_GRID:
            if not per_seed_tpr[e]:
                continue
            tpr_mean, tpr_lo, tpr_hi = metrics.seed_ci(per_seed_tpr[e])
            rows.append({
                "partition": partition, "beta": beta, "n_clients": N_CLIENTS,
                "method": METHOD, "family": "F1", "elapsed": e,
                "acc": acc_by_elapsed.get(e, float("nan")),
                "tpr1": tpr_mean, "ci_lo": tpr_lo, "ci_hi": tpr_hi, "n_seeds": len(per_seed_tpr[e]),
            })

    if not rows:
        print("no partitions had complete data -- nothing written")
        return 1

    out_csv = REPO_ROOT / "results" / "fig18_natural_federation.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {
        "seed": 0, "purpose": "FIG18 v2 matched-5-client report (H13 fix)",
        "dataset": DATASET, "method": METHOD, "n_clients": N_CLIENTS, "seeds": SEEDS,
    }
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
