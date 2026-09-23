#!/usr/bin/env python3
"""FX5 (`08_FIX_PLAN.md` §8) core sweep: eps x unit x gamma x dataset x seed, using each
(dataset, unit, eps)'s SELECTED (p, lambda, clip_C) from `results/m9_hparam_selection.csv` (fit on
`ref` only -- never train/test, CLAUDE.md non-negotiable #6). Also runs the CIFAR-100 U2
n_clients-scaling arm ({50, 100} clients, eps in {1, 4, 8}, gamma=1.0) and non-private M0/M8
references on the same streams. PBS only -- one job per dataset (`--dataset` arg), since the full grid
across all 3 datasets is the project's largest single script by run count.

The PCA basis is fit ONCE per (dataset, p) on the FULL `ref` split (not the ref-train/ref-val split
`run_m9_hparam_selection.py` used for selection -- once p/lambda/C are locked in, using all of `ref`
for the basis itself is still a public-data-only step, just no longer split for held-out selection).

Outputs: `results/m9_sweep.csv`, one row per run.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))

from p3fcl import features, provenance, sim, streams  # noqa: E402
from p3fcl.methods.m0_fedavg import FedAvgSequential  # noqa: E402
from p3fcl.methods.m8_analytic import AnalyticFCL  # noqa: E402
from p3fcl.methods.m9_contractive import ContractiveDPAnalytic  # noqa: E402
from run_m9_hparam_selection import _fit_pca  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
N_TASKS = 10
N_CLIENTS_DEFAULT = 10
BETA = 0.5
DELTA = 1e-5

EPS_GRID = [0.5, 1, 2, 4, 8, float("inf")]
UNIT_GRID = ["U1", "U2"]
GAMMA_GRID = [1.0, 0.9, 0.7]
SEEDS = [0, 1, 2]

# CIFAR-100 U2 n_clients-scaling arm (08_FIX_PLAN.md §8: "U2 with 10 clients is expected to be
# hard... Also run U2 on CIFAR-100 with n_clients in {50, 100} and eps in {1, 4, 8}.")
NCLIENTS_ARM_EPS = [1, 4, 8]
NCLIENTS_ARM_N = [50, 100]
NCLIENTS_ARM_GAMMA = 1.0


def _load_hparam_table(dataset: str) -> dict:
    path = REPO_ROOT / "results" / "m9_hparam_selection.csv"
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] == dataset and r["selected"] == "1"]
    out = {}
    for r in rows:
        eps = float(r["eps"])
        out[(r["unit"], eps)] = (int(r["p"]), float(r["lambda"]), float(r["clip_C"]))
    return out


def _pca_cache(dataset: str, X_ref: np.ndarray) -> dict:
    """`{p: basis}` fit lazily as needed (one SVD per distinct `p` this dataset's hparam table uses)."""
    cache: dict = {}

    def get(p: int) -> np.ndarray:
        if p not in cache:
            cache[p] = _fit_pca(X_ref, p_max=p, seed=0)
        return cache[p]

    return get


def _run_m9(X, y, X_test, y_test, n_classes, P, cfg, dataset, n_clients, seed) -> dict:
    stream = streams.build_stream(y, np.arange(len(y)), n_tasks=N_TASKS, n_clients=n_clients, beta=BETA, seed=seed)
    eval_id_lists = streams.task_eval_sets(y_test, np.arange(len(y_test)), n_tasks=N_TASKS, seed=seed)
    eval_sets = [(X_test[ids], y_test[ids]) for ids in eval_id_lists]
    method = ContractiveDPAnalytic({"n_classes": n_classes, "pca_basis": P, "noise_seed": seed, **cfg})
    result = sim.run(method, X, y, stream, seed=seed, eval_sets=eval_sets)
    return {
        "final_avg_acc": result["final_avg_acc"], "bwt": result["bwt"],
        "avg_incremental_acc": result["avg_incremental_acc"],
    }


def _run_reference(method_cls, method_cfg, X, y, X_test, y_test, n_classes, feature_dim, n_clients, seed) -> dict:
    stream = streams.build_stream(y, np.arange(len(y)), n_tasks=N_TASKS, n_clients=n_clients, beta=BETA, seed=seed)
    eval_id_lists = streams.task_eval_sets(y_test, np.arange(len(y_test)), n_tasks=N_TASKS, seed=seed)
    eval_sets = [(X_test[ids], y_test[ids]) for ids in eval_id_lists]
    method = method_cls({"n_classes": n_classes, "feature_dim": feature_dim, **method_cfg})
    result = sim.run(method, X, y, stream, seed=seed, eval_sets=eval_sets)
    return {
        "final_avg_acc": result["final_avg_acc"], "bwt": result["bwt"],
        "avg_incremental_acc": result["avg_incremental_acc"],
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_m9_sweep.py <dataset>", file=sys.stderr)
        return 2
    dataset = sys.argv[1]

    train_cache = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, "train")
    X, y = train_cache["features"], train_cache["labels"]
    n_classes = int(y.max()) + 1
    feature_dim = X.shape[1]
    test_cache = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, "test")
    X_test, y_test = test_cache["features"], test_cache["labels"]
    ref_cache = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, "ref")
    X_ref = ref_cache["features"]

    hparams = _load_hparam_table(dataset)
    pca_get = _pca_cache(dataset, X_ref)

    rows = []

    # Main grid.
    for unit in UNIT_GRID:
        for eps in EPS_GRID:
            if (unit, eps) not in hparams:
                print(f"WARNING: no selected hparams for {dataset}/{unit}/eps={eps} -- skipping", file=sys.stderr)
                continue
            p, lam, clip_c = hparams[(unit, eps)]
            P = pca_get(p)
            for gamma in GAMMA_GRID:
                for seed in SEEDS:
                    cfg = {"gamma": gamma, "eps": eps, "delta": DELTA, "ridge_lambda": lam, "unit": unit, "clip_C": clip_c}
                    out = _run_m9(X, y, X_test, y_test, n_classes, P, cfg, dataset, N_CLIENTS_DEFAULT, seed)
                    rows.append({
                        "dataset": dataset, "method": "m9_contractive", "unit": unit, "eps": eps,
                        "gamma": gamma, "n_clients": N_CLIENTS_DEFAULT, "seed": seed, "p": p,
                        "lambda": lam, "clip_C": clip_c, **out,
                    })
                    print(f"{dataset} unit={unit} eps={eps} gamma={gamma} seed={seed}: "
                          f"final_avg_acc={out['final_avg_acc']:.4f} bwt={out['bwt']:.4f}")

    # Non-private references (M0, M8) on the same streams, n_clients=10 only.
    for seed in SEEDS:
        m0_out = _run_reference(FedAvgSequential, {"local_epochs": 30, "lr": 0.5}, X, y, X_test, y_test, n_classes, feature_dim, N_CLIENTS_DEFAULT, seed)
        rows.append({"dataset": dataset, "method": "m0_fedavg", "unit": "n/a", "eps": float("inf"), "gamma": "n/a", "n_clients": N_CLIENTS_DEFAULT, "seed": seed, "p": "n/a", "lambda": "n/a", "clip_C": "n/a", **m0_out})
        m8_out = _run_reference(AnalyticFCL, {"ridge_lambda": 1.0}, X, y, X_test, y_test, n_classes, feature_dim, N_CLIENTS_DEFAULT, seed)
        rows.append({"dataset": dataset, "method": "m8_analytic", "unit": "n/a", "eps": float("inf"), "gamma": "n/a", "n_clients": N_CLIENTS_DEFAULT, "seed": seed, "p": "n/a", "lambda": "n/a", "clip_C": "n/a", **m8_out})
        print(f"{dataset} reference seed={seed}: M0 final_avg_acc={m0_out['final_avg_acc']:.4f}, M8 final_avg_acc={m8_out['final_avg_acc']:.4f}")

    # CIFAR-100 U2 n_clients-scaling arm.
    if dataset == "cifar100":
        for n_clients in NCLIENTS_ARM_N:
            for eps in NCLIENTS_ARM_EPS:
                if ("U2", float(eps)) not in hparams:
                    print(f"WARNING: no selected hparams for cifar100/U2/eps={eps} -- skipping n_clients={n_clients} arm", file=sys.stderr)
                    continue
                p, lam, clip_c = hparams[("U2", float(eps))]
                P = pca_get(p)
                for seed in SEEDS:
                    cfg = {"gamma": NCLIENTS_ARM_GAMMA, "eps": float(eps), "delta": DELTA, "ridge_lambda": lam, "unit": "U2", "clip_C": clip_c}
                    out = _run_m9(X, y, X_test, y_test, n_classes, P, cfg, dataset, n_clients, seed)
                    rows.append({
                        "dataset": dataset, "method": "m9_contractive", "unit": "U2", "eps": float(eps),
                        "gamma": NCLIENTS_ARM_GAMMA, "n_clients": n_clients, "seed": seed, "p": p,
                        "lambda": lam, "clip_C": clip_c, **out,
                    })
                    print(f"cifar100 n_clients-arm unit=U2 eps={eps} n_clients={n_clients} seed={seed}: final_avg_acc={out['final_avg_acc']:.4f}")

    out_csv = REPO_ROOT / "results" / f"m9_sweep_{dataset}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {"seed": 0, "purpose": "FX5 M9 core sweep", "dataset": dataset, "eps_grid": EPS_GRID, "unit_grid": UNIT_GRID, "gamma_grid": GAMMA_GRID, "seeds": SEEDS}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
