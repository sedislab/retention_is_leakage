#!/usr/bin/env python3
"""Assembles `results/fig01_decoupling.csv` — the headline decoupling comparison (accuracy decay vs.
A1 membership leakage, both normalised and averaged with a 95%-CI band over seeds) — from already
-computed CSVs (`accuracy_matrix_<dataset>.csv`, `a1_lira_<dataset>_<method>[_seed<N>].csv`). Pure
post-processing, no training: reads results, writes results, per CLAUDE.md non-negotiable #3
("no number reaches the paper except through a CSV... plot scripts never recompute" — the
aggregation happens *here*, once, so `analysis/fig01_decoupling.py` only has to draw what's already
in this file).

**Schema note**: `03_RESULTS_SPEC.md`'s own listed FIG01 schema names a `seed` column alongside
`ci_lo`/`ci_hi`, which reads two ways (one row per seed with a within-seed CI, or one row per
aggregate with a seed-level CI) — resolved in favor of the second reading here, since "no computation
at plot time" only holds if every aggregate is already computed: this CSV has one row per
`(dataset, method, family, elapsed)`, already averaged across all available seeds, with the CI band
computed *over those seeds* (`metrics.seed_ci`) — not a per-seed row layout.

**Scope**: `DATASETS` below (H2's real bar needs >=3 -- extend the list as more land; each dataset
needs a real `accuracy_matrix_<dataset>.csv` and `a1_lira_<dataset>_<method>[_seed<N>].csv` set
already computed). All 7 A1-validated methods as of 2026-09-17. M6/M7 excluded — they need the
separate offline/GPU LiRA variant.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import metrics, provenance  # noqa: E402
from p3fcl.artifacts import Family  # noqa: E402

DATASETS = ["cifar100", "cub200", "imagenet_r"]
SEEDS = [0, 1, 2, 3, 4]
METHOD_FAMILY = {
    "m0_fedavg": Family.MODEL_DELTA,
    "m1_glfc": Family.MODEL_DELTA,
    "m2_target": Family.MODEL_DELTA,
    "m3_fot": Family.MODEL_DELTA,
    "m4_proto": Family.PROTOTYPE,
    "m5_hybrid_replay": Family.MODEL_DELTA,
    "m8_analytic": Family.GRAM,
}


def _read_csv_rows(path: Path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _accuracy_normalized_by_elapsed(acc_rows: list, method: str, seed: int) -> dict:
    """Per-elapsed accuracy, normalised to each task's own just-trained value and averaged over every
    valid task index k for this (method, seed) -- `acc(k, k+elapsed) / acc(k, k)`."""
    by_k: dict = defaultdict(dict)
    for r in acc_rows:
        if r["method"] != method or int(r["seed"]) != seed:
            continue
        by_k[int(r["task_k"])][int(r["elapsed"])] = float(r["acc"])

    by_elapsed: dict = defaultdict(list)
    for k, elapsed_map in by_k.items():
        base = elapsed_map.get(0)
        if not base:
            continue
        for elapsed, acc in elapsed_map.items():
            by_elapsed[elapsed].append(acc / base)
    return {e: float(np.mean(v)) for e, v in by_elapsed.items()}


def _lira_seed_path(dataset: str, method: str, seed: int) -> Path:
    suffix = f"_seed{seed}" if seed != 0 else ""
    return REPO_ROOT / "results" / f"a1_lira_{dataset}_{method}{suffix}.csv"


def main() -> int:
    out_rows = []
    for dataset in DATASETS:
        acc_matrix_path = REPO_ROOT / "results" / f"accuracy_matrix_{dataset}.csv"
        if not acc_matrix_path.exists():
            print(f"(skipping dataset {dataset}: {acc_matrix_path.name} not found)")
            continue
        acc_rows = _read_csv_rows(acc_matrix_path)

        for method, family in METHOD_FAMILY.items():
            acc_by_elapsed_seed: dict = defaultdict(list)
            tpr1_by_elapsed_seed: dict = defaultdict(list)
            tpr01_by_elapsed_seed: dict = defaultdict(list)
            auc_by_elapsed_seed: dict = defaultdict(list)

            available_seeds = []
            for seed in SEEDS:
                lira_path = _lira_seed_path(dataset, method, seed)
                if not lira_path.exists():
                    print(f"  (skipping {dataset}/{method} seed={seed}: {lira_path.name} not found)")
                    continue
                available_seeds.append(seed)

                acc_by_elapsed = _accuracy_normalized_by_elapsed(acc_rows, method, seed)
                for e, v in acc_by_elapsed.items():
                    acc_by_elapsed_seed[e].append(v)

                for row in _read_csv_rows(lira_path):
                    if row["ablation"] != "trajectory":
                        continue
                    e = int(row["elapsed"])
                    tpr1_by_elapsed_seed[e].append(float(row["tpr_at_1pct_fpr"]))
                    tpr01_by_elapsed_seed[e].append(float(row["tpr_at_0.1pct_fpr"]))
                    auc_by_elapsed_seed[e].append(float(row["auc"]))

            print(f"{dataset}/{method}: {len(available_seeds)} seeds available ({available_seeds})")
            all_elapsed = sorted(set(acc_by_elapsed_seed) | set(tpr1_by_elapsed_seed))
            for e in all_elapsed:
                acc_mean, acc_lo, acc_hi = metrics.seed_ci(acc_by_elapsed_seed.get(e, []))
                tpr1_mean, tpr1_lo, tpr1_hi = metrics.seed_ci(tpr1_by_elapsed_seed.get(e, []))
                tpr01_mean, tpr01_lo, tpr01_hi = metrics.seed_ci(tpr01_by_elapsed_seed.get(e, []))
                auc_mean, auc_lo, auc_hi = metrics.seed_ci(auc_by_elapsed_seed.get(e, []))
                out_rows.append({
                    "dataset": dataset, "method": method, "family": family.value, "elapsed": e,
                    "n_seeds": len(acc_by_elapsed_seed.get(e, [])),
                    "acc_mean": acc_mean, "acc_ci_lo": acc_lo, "acc_ci_hi": acc_hi,
                    "tpr1_mean": tpr1_mean, "tpr1_ci_lo": tpr1_lo, "tpr1_ci_hi": tpr1_hi,
                    "tpr01_mean": tpr01_mean, "tpr01_ci_lo": tpr01_lo, "tpr01_ci_hi": tpr01_hi,
                    "auc_mean": auc_mean, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
                })

    out_csv = REPO_ROOT / "results" / "fig01_decoupling.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {out_csv}")

    config = {"seed": 0, "purpose": "P4 FIG01 decoupling CSV assembly (claim C1)", "datasets": DATASETS}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
