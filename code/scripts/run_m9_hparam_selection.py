#!/usr/bin/env python3
"""FX5 (`08_FIX_PLAN.md` §8): M9 hyperparameter selection, `ref` split only -- never train or test
(CLAUDE.md non-negotiable #6). `ref` is split 50/50 (seeded, stratified by class) into `ref-train`
(fits the PCA basis, the per-task/client statistics, and `clip_C`) and `ref-val` (scores every grid
point); the same task/client stream structure as the real runs (10 tasks, 10 clients, beta=0.5) is
built on `ref-train`'s ids.

Grid: `p` in {32, 64, 128, 256, feature_dim}, `lambda` in {1e-2, 1e-1, 1, 10, 100}, selected
separately per (dataset, unit, eps) -- sensitivity and clipping trade off differently against noise at
each privacy budget, so the best (p, lambda) is not assumed to be the same across them. `clip_C` is
not a free grid axis: for each `p`, it is fixed to the 95th percentile of "pseudo-client" (G, H) pair
norms on `ref-train` (the same PCA+cap pipeline, no clipping yet) -- computed once per `p`, reused for
every `(unit, eps, lambda)` combination at that `p`.

`gamma` is fixed at 1.0 for this selection (not part of `08_FIX_PLAN.md`'s stated hparam grid, which
lists only p/lambda/C): `gamma` is the core sweep's own retention-strength axis, and mixing state
decay into the choice of p/lambda/C would confound a representation/regularization choice with a
retention-strength choice. This is a modeling decision this script makes explicit, not something the
plan specifies either way.

Output: `results/m9_hparam_selection.csv`, one row per (dataset, unit, eps, p, lambda) grid point plus
a `selected` column marking the winner (by ref-val final average accuracy) for each (dataset, unit,
eps). PBS only -- the full grid is 3 datasets x 2 units x 6 eps x 5 p x 5 lambda = 900 full M9 runs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, provenance, sim, streams  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402
from p3fcl.methods.m9_contractive import ContractiveDPAnalytic, client_stats, project_and_cap  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
DATASETS = ["cifar100", "cub200", "imagenet_r"]
UNITS = ["U1", "U2"]
EPS_GRID = [0.5, 1, 2, 4, 8, float("inf")]
P_GRID_BASE = [32, 64, 128, 256]  # + feature_dim itself, appended per-dataset
LAMBDA_GRID = [1e-2, 1e-1, 1.0, 10.0, 100.0]
N_TASKS = 10
N_CLIENTS = 10
BETA = 0.5
GAMMA = 1.0  # see module docstring
DELTA = 1e-5


def _ref_train_val_split(labels, seed=0, frac=0.5):
    r = rng_mod.seeded("m9_hparam::ref_split", seed)
    train_idx, val_idx = [], []
    for c in sorted(set(labels.tolist())):
        class_idx = np.where(labels == c)[0]
        perm = class_idx[r.permutation(len(class_idx))]
        k = int(round(len(perm) * frac))
        train_idx.extend(perm[:k].tolist())
        val_idx.extend(perm[k:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


def _fit_pca(X_train: np.ndarray, p_max: int, seed=0) -> np.ndarray:
    """Top-`p_max` PCA basis fit on `ref-train`, centered at 0 (M9's projection is linear, `X @ P`,
    so centering must be a no-op or baked into `P` -- kept as a plain top-singular-vectors basis, no
    mean-subtraction, so `project_and_cap` stays a pure linear map matching the method's own code)."""
    _ = seed  # PCA via SVD is deterministic given X_train; kept for signature symmetry with other helpers
    U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
    del U, S
    return Vt[:p_max].T  # (d, p_max)


def _pseudo_client_norms(X_train, y_train, P, n_classes, seed=0) -> np.ndarray:
    """Unclipped `(vech G_c, vec H_c)` Frobenius norms over the same task/client stream structure on
    `ref-train`, used to set `clip_C` at the 95th percentile -- "pseudo-client" because `ref` has no
    real client/task labels of its own, only the same partitioning procedure applied to it."""
    idx = np.arange(len(y_train))
    stream = streams.build_stream(y_train, idx, n_tasks=N_TASKS, n_clients=N_CLIENTS, beta=BETA, seed=seed)
    norms = []
    for task_shards in stream:
        for shard in task_shards:
            sidx = np.array(shard.ids, dtype=int)
            Xc_proj = project_and_cap(X_train[sidx], P, B=1.0)
            Gc, Hc = client_stats(Xc_proj, y_train[sidx], n_classes)
            norms.append(float(np.sqrt(np.sum(Gc**2) + np.sum(Hc**2))))
    return np.array(norms)


def _run_and_eval(X_train, y_train, X_val, y_val, n_classes, P, unit, eps, ridge_lambda, clip_C, seed):
    stream = streams.build_stream(y_train, np.arange(len(y_train)), n_tasks=N_TASKS, n_clients=N_CLIENTS, beta=BETA, seed=seed)
    eval_id_lists = streams.task_eval_sets(y_val, np.arange(len(y_val)), n_tasks=N_TASKS, seed=seed)
    eval_sets = [(X_val[ids], y_val[ids]) for ids in eval_id_lists]
    method = ContractiveDPAnalytic({
        "n_classes": n_classes, "pca_basis": P, "gamma": GAMMA, "eps": eps, "delta": DELTA,
        "ridge_lambda": ridge_lambda, "unit": unit, "clip_C": clip_C, "B": 1.0, "noise_seed": seed,
    })
    result = sim.run(method, X_train, y_train, stream, seed=seed, eval_sets=eval_sets)
    return result["final_avg_acc"]


def main() -> int:
    rows = []
    for dataset in DATASETS:
        cache = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, "ref")
        X, y = cache["features"], cache["labels"]
        n_classes = int(y.max()) + 1
        d = X.shape[1]
        train_idx, val_idx = _ref_train_val_split(y, seed=0, frac=0.5)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        p_grid = sorted(set(P_GRID_BASE + [d]))
        p_max = max(p_grid)
        P_full = _fit_pca(X_train, p_max, seed=0)

        for p in p_grid:
            P = P_full[:, :p]
            norms = _pseudo_client_norms(X_train, y_train, P, n_classes, seed=0)
            clip_C = float(np.percentile(norms, 95))

            for unit in UNITS:
                for eps in EPS_GRID:
                    best_acc, best_lambda = -1.0, None
                    grid_this_point = []
                    for lam in LAMBDA_GRID:
                        acc = _run_and_eval(X_train, y_train, X_val, y_val, n_classes, P, unit, eps, lam, clip_C, seed=0)
                        grid_this_point.append((lam, acc))
                        if acc > best_acc:
                            best_acc, best_lambda = acc, lam
                    for lam, acc in grid_this_point:
                        rows.append({
                            "dataset": dataset, "unit": unit, "eps": eps, "p": p, "lambda": lam,
                            "clip_C": clip_C, "ref_val_final_avg_acc": acc,
                            "selected": int(lam == best_lambda),
                        })
                    print(f"{dataset} unit={unit} eps={eps} p={p}: best lambda={best_lambda} acc={best_acc:.4f}")

    # Reduce to one "selected" row per (dataset, unit, eps) across ALL p too (the grid above fixes p
    # per outer loop iteration and marks the best lambda within it; now mark the overall best (p,
    # lambda) per (dataset, unit, eps) so downstream consumers don't have to re-derive it).
    by_key: dict = {}
    for r in rows:
        if not r["selected"]:
            continue
        key = (r["dataset"], r["unit"], r["eps"])
        if key not in by_key or r["ref_val_final_avg_acc"] > by_key[key]["ref_val_final_avg_acc"]:
            by_key[key] = r
    winners = {(r["dataset"], r["unit"], r["eps"], r["p"], r["lambda"]) for r in by_key.values()}
    for r in rows:
        r["selected"] = int((r["dataset"], r["unit"], r["eps"], r["p"], r["lambda"]) in winners)

    out_csv = REPO_ROOT / "results" / "m9_hparam_selection.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {
        "seed": 0, "purpose": "FX5 M9 hyperparameter selection (ref split only)", "datasets": DATASETS,
        "units": UNITS, "eps_grid": EPS_GRID, "p_grid_base": P_GRID_BASE, "lambda_grid": LAMBDA_GRID,
        "gamma": GAMMA, "delta": DELTA,
    }
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
