#!/usr/bin/env python3
"""R2 projection ablation (`rebuttal/R2_PROTOCOL.md`). PBS only, one job; datasets run in parallel
processes. Arms: pca_id (paper), no_proj (P = I), pca_ood (PCA and selection on another dataset's ref),
random_proj (seeded Gaussian + QR, selected on the OOD ref). Selection reuses the paper's
`run_m9_hparam_selection` helpers; final runs reuse `run_m9_sweep._run_m9`.

Outputs: `results/rb_projection_ablation.csv`, `results/rb_projection_ablation_selection.csv`.
The job fails before writing anything if rerunning the arm-A selection does not reproduce the
paper's (p, lambda, C).
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))

from p3fcl import features, provenance  # noqa: E402
from p3fcl.rebuttal.common import manifest  # noqa: E402
from p3fcl.rebuttal.m9_select import random_orthonormal, select_first_max  # noqa: E402
from run_m9_hparam_selection import (  # noqa: E402
    LAMBDA_GRID, P_GRID_BASE, _fit_pca, _pseudo_client_norms, _ref_train_val_split, _run_and_eval,
)
from run_m9_sweep import BACKBONE, DELTA, N_CLIENTS_DEFAULT, _load_hparam_table, _run_m9  # noqa: E402

DATASETS = ["cifar100", "cub200", "imagenet_r"]
OOD_SOURCE = {"cifar100": "imagenet_r", "cub200": "cifar100", "imagenet_r": "cifar100"}
CELLS = [("U1", 1.0), ("U1", 2.0), ("U1", 4.0), ("U1", 8.0), ("U1", float("inf")), ("U2", 8.0)]
SEEDS = [0, 1, 2]
RANDOM_P = [32, 64, 128, 256]


def _load(dataset, split):
    c = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, split)
    return c["features"], c["labels"]


def select(sel_dataset: str, arm: str, basis_for_selection, p_list: list) -> tuple:
    """The paper's selection on `sel_dataset`'s ref split. Returns `({cell: winner}, grid_rows)`."""
    X, y = _load(sel_dataset, "ref")
    n_classes = int(y.max()) + 1
    tr, va = _ref_train_val_split(y, seed=0, frac=0.5)
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    grids = {cell: [] for cell in CELLS}
    rows = []
    for p in p_list:
        P = basis_for_selection(Xtr, p)
        C = float(np.percentile(_pseudo_client_norms(Xtr, ytr, P, n_classes, seed=0), 95))
        for unit, eps in CELLS:
            for lam in LAMBDA_GRID:
                acc = _run_and_eval(Xtr, ytr, Xva, yva, n_classes, P, unit, eps, lam, C, seed=0)
                g = {"p": p, "lambda": lam, "clip_C": C, "acc": acc}
                grids[(unit, eps)].append(g)
                rows.append({"arm": arm, "selection_dataset": sel_dataset, "unit": unit, "eps": eps, **g})
    winners = {cell: select_first_max(grid) for cell, grid in grids.items()}
    for r in rows:
        w = winners[(r["unit"], r["eps"])]
        r["selected"] = int(r["p"] == w["p"] and r["lambda"] == w["lambda"])
    return winners, rows


def run_dataset(dataset: str) -> tuple:
    X, y = _load(dataset, "train")
    Xte, yte = _load(dataset, "test")
    n_classes, d = int(y.max()) + 1, X.shape[1]
    ood = OOD_SOURCE[dataset]
    pca_on_train = lambda Xtr, p: _fit_pca(Xtr, p, seed=0)  # noqa: E731  (selection basis, ref-train)
    full_ref_pca = lambda src, p: _fit_pca(_load(src, "ref")[0], p, seed=0)  # noqa: E731  (final basis)
    pca_p = sorted(set(P_GRID_BASE + [d]))

    # Arm A: the paper's selection, plus a rerun with this script's code that must reproduce it.
    paper = _load_hparam_table(dataset)
    resel, sel_rows = select(dataset, "pca_id_reselect_check", lambda Xtr, p: _fit_pca(Xtr, max(pca_p), seed=0)[:, :p], pca_p)
    for cell in CELLS:
        p0, l0, c0 = paper[cell]
        w = resel[cell]
        if not (w["p"] == p0 and w["lambda"] == l0 and abs(w["clip_C"] - c0) <= 1e-6 * abs(c0)):  # R2_PROTOCOL.md amendment 1
            raise AssertionError(f"{dataset} {cell}: reselection {w} != paper {(p0, l0, c0)}")
    arms = {"pca_id": (dataset, {cell: {"p": paper[cell][0], "lambda": paper[cell][1], "clip_C": paper[cell][2]} for cell in CELLS},
                       lambda p: full_ref_pca(dataset, p))}

    eye = np.eye(d)
    w, rows = select(dataset, "no_proj", lambda Xtr, p: eye, [d])
    sel_rows += rows
    arms["no_proj"] = (dataset, w, lambda p: eye)

    w, rows = select(ood, "pca_ood", lambda Xtr, p: _fit_pca(Xtr, max(pca_p), seed=0)[:, :p], pca_p)
    sel_rows += rows
    arms["pca_ood"] = (ood, w, lambda p: full_ref_pca(ood, p))

    w, rows = select(ood, "random_proj", lambda Xtr, p: random_orthonormal(d, p, 0), RANDOM_P)
    sel_rows += rows
    arms["random_proj"] = (ood, w, lambda p: random_orthonormal(d, p, 0))

    out = []
    for arm, (src, winners, final_basis) in arms.items():
        cache = {}
        for unit, eps in CELLS:
            wv = winners[(unit, eps)]
            if wv["p"] not in cache:
                cache[wv["p"]] = final_basis(wv["p"])
            for seed in SEEDS:
                cfg = {"gamma": 1.0, "eps": eps, "delta": DELTA, "ridge_lambda": wv["lambda"], "unit": unit, "clip_C": wv["clip_C"]}
                res = _run_m9(X, y, Xte, yte, n_classes, cache[wv["p"]], cfg, dataset, N_CLIENTS_DEFAULT, seed)
                out.append({"dataset": dataset, "arm": arm, "unit": unit, "eps": eps, "selection_dataset": src,
                            "selected_p": wv["p"], "lambda": wv["lambda"], "clip_C": wv["clip_C"], "seed": seed, **res})
                print(f"R2 {dataset} {arm} {unit} eps={eps} seed={seed} p={wv['p']} lam={wv['lambda']}: "
                      f"final_avg_acc={res['final_avg_acc']:.4f}", flush=True)
    for r in sel_rows:
        r["dataset"] = dataset
    return out, sel_rows


def main() -> int:
    man = manifest({"seed": 0, "purpose": "rebuttal R2 projection ablation", "datasets": DATASETS,
                    "ood_source": OOD_SOURCE, "cells": [list(c) for c in CELLS], "seeds": SEEDS,
                    "random_p": RANDOM_P, "lambda_grid": LAMBDA_GRID, "protocol": "rebuttal/R2_PROTOCOL.md"})
    with mp.get_context("fork").Pool(len(DATASETS)) as pool:
        parts = pool.map(run_dataset, DATASETS)
    rows = [r for p in parts for r in p[0]]
    sel = [r for p in parts for r in p[1]]
    out = REPO_ROOT / "results" / "rb_projection_ablation.csv"
    out_sel = REPO_ROOT / "results" / "rb_projection_ablation_selection.csv"
    for path, data in ((out, rows), (out_sel, sel)):
        cols = list(data[0])
        if path == out_sel:
            cols = ["dataset"] + [c for c in cols if c != "dataset"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(data)
    print(f"R2 DONE rows={len(rows)} selection_rows={len(sel)} reselect_check=PASS")
    provenance.finalize(man, [out, out_sel])
    return 0


if __name__ == "__main__":
    sys.exit(main())
