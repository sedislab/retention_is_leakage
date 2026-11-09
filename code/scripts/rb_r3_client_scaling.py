#!/usr/bin/env python3
"""R3 client scaling (`rebuttal/R3_PROTOCOL.md`). PBS array, one task per (dataset, n), index in
0..11. Writes `rebuttal/work/r3/part_<ds>_n<n>.csv` (+ `sel_...`); `rb_r3_merge.py` checks them
against the paper and writes `results/rb_client_scaling*.csv`.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))

from p3fcl import features, provenance, sim, streams  # noqa: E402
from p3fcl.methods.m8_analytic import AnalyticFCL  # noqa: E402
from p3fcl.methods.m9_contractive import ContractiveDPAnalytic  # noqa: E402
from p3fcl.rebuttal import client_scaling as cs  # noqa: E402
from p3fcl.rebuttal.common import manifest  # noqa: E402
from p3fcl.rebuttal.m9_select import select_first_max  # noqa: E402
from run_m9_hparam_selection import LAMBDA_GRID, P_GRID_BASE, _fit_pca, _ref_train_val_split, _run_and_eval  # noqa: E402
from run_m9_sweep import BACKBONE, BETA, DELTA, N_TASKS, _load_hparam_table  # noqa: E402

N_GRID = {"cifar100": [10, 50, 100, 500, 1000, 2000], "cub200": [10, 50, 100], "imagenet_r": [10, 50, 100]}
COMBOS = [(d, n) for d in ["cifar100", "cub200", "imagenet_r"] for n in N_GRID[d]]
EPS = [1.0, 4.0, 8.0]
SEEDS = [0, 1, 2]
WORK = REPO_ROOT / "rebuttal" / "work" / "r3"


class AnalyticFCLNoLedger(AnalyticFCL):
    """M8 unchanged, but its per-client ledger records are dropped (accuracy never reads them). At
    large n they would need tens of GB."""

    def fit_task(self, *args, **kwargs):
        super().fit_task(*args, **kwargs)
        return []


def _load(dataset, split):
    c = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, split)
    return c["features"], c["labels"]


def _run(method, X, y, Xte, yte, n, seed):
    """`run_m9_sweep._run_m9` / `_run_reference`'s construction, returning the ledger too."""
    stream = streams.build_stream(y, np.arange(len(y)), n_tasks=N_TASKS, n_clients=n, beta=BETA, seed=seed)
    eval_ids = streams.task_eval_sets(yte, np.arange(len(yte)), n_tasks=N_TASKS, seed=seed)
    res = sim.run(method, X, y, stream, seed=seed, eval_sets=[(Xte[i], yte[i]) for i in eval_ids])
    return stream, res


def main() -> int:
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["PBS_ARRAY_INDEX"])
    dataset, n = COMBOS[idx]
    man = manifest({"seed": 0, "purpose": "rebuttal R3 client scaling", "dataset": dataset, "n_clients": n,
                    "eps": EPS, "seeds": SEEDS, "n_pseudo_per_task": cs.N_PSEUDO_PER_TASK,
                    "percentile": cs.PERCENTILE, "protocol": "rebuttal/R3_PROTOCOL.md"})
    X, y = _load(dataset, "train")
    Xte, yte = _load(dataset, "test")
    Xref, yref = _load(dataset, "ref")
    n_classes, d = int(y.max()) + 1, X.shape[1]
    tr, va = _ref_train_val_split(yref, seed=0, frac=0.5)
    Xtr, ytr, Xva, yva = Xref[tr], yref[tr], Xref[va], yref[va]
    m_n = cs.mean_client_size(len(y), N_TASKS, n)
    pseudo = cs.pseudo_client_ids(ytr, streams.class_incremental_tasks(ytr, N_TASKS, 0), m_n, dataset, n)
    pca_p = sorted(set(P_GRID_BASE + [d]))
    P_sel_full = _fit_pca(Xtr, max(pca_p), seed=0)

    # Size-matched selection.
    sel_rows, grids = [], {e: [] for e in EPS}
    for p in pca_p:
        P = P_sel_full[:, :p]
        C = cs.size_matched_clip(Xtr, ytr, pseudo, P, n_classes)
        for eps in EPS:
            for lam in LAMBDA_GRID:
                acc = _run_and_eval(Xtr, ytr, Xva, yva, n_classes, P, "U2", eps, lam, C, seed=0)
                grids[eps].append({"p": p, "lambda": lam, "clip_C": C, "acc": acc})
                sel_rows.append({"dataset": dataset, "n_clients": n, "m_n": m_n, "eps": eps, "p": p, "lambda": lam, "clip_C": C, "ref_val_final_avg_acc": acc})
        print(f"R3 {dataset} n={n} m={m_n} p={p}: C(n,p)={C:.4f}", flush=True)
    winners = {e: select_first_max(g) for e, g in grids.items()}
    for r in sel_rows:
        w = winners[r["eps"]]
        r["selected"] = int(r["p"] == w["p"] and r["lambda"] == w["lambda"])

    paper = _load_hparam_table(dataset)
    bases: dict = {}

    def basis(p):
        if p not in bases:
            bases[p] = _fit_pca(Xref, p, seed=0)
        return bases[p]

    rows = []
    for seed in SEEDS:
        runs = []
        for eps in EPS:
            w = winners[eps]
            runs.append(("size_matched", "U2", eps, w["p"], w["lambda"], w["clip_C"]))
            runs.append(("old_fixed_C", "U2", eps, *paper[("U2", eps)]))
            runs.append(("u1_reference", "U1", eps, *paper[("U1", eps)]))
        for calib, unit, eps, p, lam, C in runs:
            P = basis(p)
            m = ContractiveDPAnalytic({"n_classes": n_classes, "pca_basis": P, "noise_seed": seed, "gamma": 1.0,
                                       "eps": eps, "delta": DELTA, "ridge_lambda": lam, "unit": unit, "clip_C": C})
            stream, res = _run(m, X, y, Xte, yte, n, seed)
            diag = cs.u2_diagnostics(X, y, stream, P, n_classes, C, clip=(unit == "U2"))
            sigma = [rec.meta["sigma"] for rec in res["ledger"] if rec.round == 0][0]
            rows.append({"dataset": dataset, "method": "m9_contractive", "unit": unit, "calibration": calib,
                         "n_clients": n, "m_n": m_n, "eps": eps, "seed": seed, "p": p, "lambda": lam, "clip_C": C,
                         "frac_clipped": diag["frac_clipped"], "noise_std": sigma, "agg0_frob": diag["agg0_frob"],
                         "final_avg_acc": res["final_avg_acc"], "bwt": res["bwt"], "avg_incremental_acc": res["avg_incremental_acc"]})
            print(f"R3 {dataset} n={n} {calib} {unit} eps={eps} seed={seed}: acc={res['final_avg_acc']:.4f} "
                  f"clipped={diag['frac_clipped']} sigma={sigma:.3f} agg0={diag['agg0_frob']:.2f}", flush=True)
        m8 = AnalyticFCLNoLedger({"n_classes": n_classes, "feature_dim": d, "ridge_lambda": 1.0})
        _, res = _run(m8, X, y, Xte, yte, n, seed)
        rows.append({"dataset": dataset, "method": "m8_analytic", "unit": "n/a", "calibration": "m8_reference",
                     "n_clients": n, "m_n": m_n, "eps": float("inf"), "seed": seed, "p": "n/a", "lambda": 1.0, "clip_C": "n/a",
                     "frac_clipped": None, "noise_std": 0.0, "agg0_frob": None,
                     "final_avg_acc": res["final_avg_acc"], "bwt": res["bwt"], "avg_incremental_acc": res["avg_incremental_acc"]})
        print(f"R3 {dataset} n={n} M8 seed={seed}: acc={res['final_avg_acc']:.4f}", flush=True)

    WORK.mkdir(parents=True, exist_ok=True)
    out, out_sel = WORK / f"part_{dataset}_n{n}.csv", WORK / f"sel_{dataset}_n{n}.csv"
    for path, data in ((out, rows), (out_sel, sel_rows)):
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0]))
            wr.writeheader()
            wr.writerows(data)
    provenance.finalize(man, [out, out_sel])
    return 0


if __name__ == "__main__":
    sys.exit(main())
