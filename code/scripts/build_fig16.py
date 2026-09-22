#!/usr/bin/env python3
"""FIG16 — log-log ROC curves (CLAUDE.md non-negotiable #5: "membership results are never reported
as AUC alone... log-log ROC is mandatory in the appendix" -- this is the one piece of that rule this
project had not yet built). Recomputes the full ROC curve (many fpr/tpr points, not just the 3 summary
numbers already in `results/a1_lira_*.csv`) from the already-generated shadow stores, reusing
`run_lira.py`'s own `load_shadow_store`/`compute_log_lr_surfaces` and the *same* seeded calibration/
evaluation split it uses for the numbers already reported -- so this curve is consistent with, not a
re-derivation that could silently disagree with, the headline TPR@1%/0.1%FPR numbers.

Scope: one curve per (dataset, method), **seed=0 only** (not all 5 seeds -- a full 5-seed-per-curve
version is a reasonable follow-up if there's time, but seed=0 is the same "primary" seed every other
per-seed CSV in this project treats as the default, and one representative curve per (dataset, method)
is what the appendix figure needs). `elapsed=0` (the standard, most-immediate MIA setting), trajectory
ablation (this project's primary ablation, per H3).
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
import yaml  # noqa: E402
from p3fcl import provenance  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402
from p3fcl.shadow_runner import METHOD_REGISTRY  # noqa: E402

DATASETS = ["cifar100", "cub200", "imagenet_r"]
METHODS = ["m0_fedavg", "m1_glfc", "m2_target", "m3_fot", "m4_proto", "m5_hybrid_replay", "m8_analytic"]
N_FPR_GRID = 200


def _roc_curve(scores: np.ndarray, labels: np.ndarray, n_points: int = N_FPR_GRID):
    """Full empirical ROC curve (exact, from every unique threshold), then read off onto a
    log-spaced FPR grid (dense at low FPR, where this venue's readers look first) via monotonic
    interpolation of the step function -- standard practice for a compact, plottable ROC CSV."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return np.array([]), np.array([])
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tpr_full = np.concatenate([[0.0], np.cumsum(labels_sorted == 1) / n_pos])
    fpr_full = np.concatenate([[0.0], np.cumsum(labels_sorted == 0) / n_neg])
    fpr_grid = np.concatenate([[0.0], np.logspace(-4, 0, n_points)])
    tpr_grid = np.interp(fpr_grid, fpr_full, tpr_full)
    return fpr_grid, tpr_grid


def main() -> int:
    # Optional `dataset method` args process a single combo and write a per-combo CSV
    # (`fig16_roc_<dataset>_<method>.csv`) -- lets the 21 combos be parallelized across processes
    # (each shadow-store load is CPU-bound and independent) rather than run one after another, which
    # was impractically slow given ImageNet-R's per-combo shadow-loading cost (see
    # `notes/2026-09-21_p4_third_dataset_imagenet_r.md`). No args processes every combo sequentially
    # into the single combined `fig16_roc.csv` (fine for cifar100/cub200 alone, slow for imagenet_r).
    single_combo = len(sys.argv) == 3
    datasets = [sys.argv[1]] if single_combo else DATASETS
    methods_by_dataset = {d: ([sys.argv[2]] if single_combo else METHODS) for d in datasets}

    with open(REPO_ROOT / "code" / "configs" / "attack_lira.yaml") as f:
        cfg = yaml.safe_load(f)
    calib_frac = float(cfg["shadow"]["calibration_frac"])
    seed = int(cfg["seed"])

    out_rows = []
    for dataset in datasets:
        for method in methods_by_dataset[dataset]:
            shadow_dir = REPO_ROOT / "shadows" / dataset / method
            if not shadow_dir.exists():
                print(f"missing {shadow_dir}, skipping")
                continue
            store = run_lira.load_shadow_store(shadow_dir)
            n_shadows = len(store["shadow_ids"])
            r = rng_mod.seeded(f"run_lira::calib_split::{dataset}::{method}", seed)
            perm = r.permutation(n_shadows)
            n_calib = int(round(calib_frac * n_shadows))
            calib_idx, eval_idx = perm[:n_calib], perm[n_calib:]

            surfaces = run_lira.compute_log_lr_surfaces(store, calib_idx)
            traj_eval = surfaces["trajectory"][eval_idx]
            in_out_eval = store["in_out"][eval_idx]
            targets = store["targets"]

            scores_list, labels_list = [], []
            for j, t in enumerate(targets):
                T = t["task"]  # elapsed=0 -> observe at the target's own release round
                if not (0 <= T < store["n_rounds"]):
                    continue
                col = traj_eval[:, j, T]
                lab = in_out_eval[:, j]
                keep = ~np.isnan(col)
                scores_list.append(col[keep])
                labels_list.append(lab[keep])
            if not scores_list:
                print(f"no usable elapsed=0 scores for {dataset}/{method}, skipping")
                continue
            scores_arr = np.concatenate(scores_list)
            labels_arr = np.concatenate(labels_list).astype(int)

            fpr, tpr = _roc_curve(scores_arr, labels_arr)
            family = METHOD_REGISTRY[method][1].value
            for f_, t_ in zip(fpr, tpr):
                out_rows.append({
                    "dataset": dataset, "method": method, "family": family, "attack": "A1_LiRA",
                    "fpr": f_, "tpr": t_, "seed": 0,
                })
            print(f"{dataset}/{method}: {len(fpr)} ROC points (n_pos={int(labels_arr.sum())}, "
                  f"n_neg={int((1 - labels_arr).sum())})")

    if single_combo:
        out_csv = REPO_ROOT / "results" / f"fig16_roc_{datasets[0]}_{methods_by_dataset[datasets[0]][0]}.csv"
    else:
        out_csv = REPO_ROOT / "results" / "fig16_roc.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "family", "attack", "fpr", "tpr", "seed"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {out_csv}")

    config = {
        "seed": seed, "purpose": "FIG16 log-log ROC curves, seed=0, elapsed=0, trajectory ablation",
        "datasets": datasets, "methods": methods_by_dataset, "calibration_frac": calib_frac,
    }
    manifest = provenance.run_manifest(config, seed=seed)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
