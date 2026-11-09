#!/usr/bin/env python3
"""R4 accuracy vs. examples per class (`rebuttal/R4_PROTOCOL.md`). PBS only, one job. Datasets:
CIFAR-100 only. Writes `results/rb_examples_per_class.csv` and `_prediction.csv`, after checking that
the m = all rows equal the paper's sweep bit for bit (otherwise it writes `rebuttal/r4_FAILED.csv`).
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

from p3fcl import features, provenance, sim, streams  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402
from p3fcl.methods.m8_analytic import AnalyticFCL  # noqa: E402
from p3fcl.methods.m9_contractive import ContractiveDPAnalytic  # noqa: E402
from p3fcl.rebuttal.common import manifest  # noqa: E402
from run_m9_hparam_selection import _fit_pca  # noqa: E402
from run_m9_sweep import BACKBONE, BETA, DELTA, N_CLIENTS_DEFAULT, N_TASKS, _load_hparam_table  # noqa: E402

DATASET = "cifar100"
M_GRID = [25, 50, 120, 250, "all"]
EPS = [1.0, 4.0, 8.0, float("inf")]
SEEDS = [0, 1, 2]
METRICS = ["final_avg_acc", "bwt", "avg_incremental_acc"]


def subset_ids(y: np.ndarray, m, seed: int) -> np.ndarray:
    if m == "all":
        return np.arange(len(y))
    out = []
    for c in sorted(set(y.tolist())):
        ids = np.flatnonzero(y == c)
        out.append(ids[rng_mod.seeded(f"rb_r4::subsample::{c}", seed).permutation(len(ids))][:m])
    return np.sort(np.concatenate(out))


def _job(args):
    m, seed = args
    X, y = (lambda c: (c["features"], c["labels"]))(features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "train"))
    te = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "test")
    Xte, yte = te["features"], te["labels"]
    Xref = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "ref")["features"]
    n_classes, d = int(y.max()) + 1, X.shape[1]
    ids = subset_ids(y, m, seed)
    stream = streams.build_stream(y, ids, n_tasks=N_TASKS, n_clients=N_CLIENTS_DEFAULT, beta=BETA, seed=seed)
    eval_ids = streams.task_eval_sets(yte, np.arange(len(yte)), n_tasks=N_TASKS, seed=seed)
    eval_sets = [(Xte[i], yte[i]) for i in eval_ids]
    hp = _load_hparam_table(DATASET)
    rows = []
    for eps in EPS:
        p, lam, C = hp[("U1", eps)]
        meth = ContractiveDPAnalytic({"n_classes": n_classes, "pca_basis": _fit_pca(Xref, p, seed=0), "noise_seed": seed,
                                      "gamma": 1.0, "eps": eps, "delta": DELTA, "ridge_lambda": lam, "unit": "U1", "clip_C": C})
        res = sim.run(meth, X, y, stream, seed=seed, eval_sets=eval_sets)
        rows.append({"method": "m9_contractive", "unit": "U1", "m_per_class": m, "eps": eps, "seed": seed, "p": p, "lambda": lam,
                     **{k: res[k] for k in METRICS}})
        print(f"R4 m={m} U1 eps={eps} seed={seed}: acc={res['final_avg_acc']:.4f}", flush=True)
    res = sim.run(AnalyticFCL({"n_classes": n_classes, "feature_dim": d, "ridge_lambda": 1.0}), X, y, stream, seed=seed, eval_sets=eval_sets)
    rows.append({"method": "m8_analytic", "unit": "n/a", "m_per_class": m, "eps": float("inf"), "seed": seed, "p": "n/a", "lambda": 1.0,
                 **{k: res[k] for k in METRICS}})
    print(f"R4 m={m} M8 seed={seed}: acc={res['final_avg_acc']:.4f}", flush=True)
    return rows


def main() -> int:
    man = manifest({"seed": 0, "purpose": "rebuttal R4 examples per class", "dataset": DATASET, "m_grid": M_GRID,
                    "eps": EPS, "seeds": SEEDS, "protocol": "rebuttal/R4_PROTOCOL.md"})
    with mp.get_context("fork").Pool(4) as pool:
        rows = [r for part in pool.map(_job, [(m, s) for m in M_GRID for s in SEEDS]) for r in part]

    def paper_means(ds):
        with open(REPO_ROOT / "results" / f"m9_sweep_{ds}.csv", newline="") as f:
            return list(csv.DictReader(f))
    cif, cub = paper_means("cifar100"), paper_means("cub200")
    failures = []
    for r in rows:
        r["paper_check"] = "n/a"
        if r["m_per_class"] != "all":
            continue
        hits = [p for p in cif if p["method"] == r["method"] and int(p["seed"]) == r["seed"] and int(p["n_clients"]) == 10
                and (r["method"] == "m8_analytic" or (p["unit"] == "U1" and p["gamma"] != "n/a" and float(p["gamma"]) == 1.0 and float(p["eps"]) == r["eps"]))]
        ok = len(hits) == 1 and all(float(hits[0][k]) == float(r[k]) for k in METRICS)
        r["paper_check"] = "matches_paper" if ok else "MISMATCH"
        if not ok:
            failures.append(f"m=all {r['method']} eps={r['eps']} seed={r['seed']}")

    def mean(method, m, eps=None):
        v = [r["final_avg_acc"] for r in rows if r["method"] == method and r["m_per_class"] == m and (eps is None or r["eps"] == eps)]
        return float(np.mean(v))
    m8_25, m8_all = mean("m8_analytic", 25), mean("m8_analytic", "all")
    pred = []
    for eps in EPS[:-1]:
        cub_mean = float(np.mean([float(p["final_avg_acc"]) for p in cub if p["method"] == "m9_contractive" and p["unit"] == "U1"
                                  and float(p["gamma"]) == 1.0 and int(p["n_clients"]) == 10 and float(p["eps"]) == eps]))
        c25, call = mean("m9_contractive", 25, eps), mean("m9_contractive", "all", eps)
        pred.append({"eps": eps, "cifar_m25_mean": c25, "cifar_all_mean": call, "cub_paper_mean": cub_mean,
                     "holds_eps": int(abs(c25 - cub_mean) < abs(c25 - call)), "m8_m25_mean": m8_25, "m8_all_mean": m8_all,
                     "m8_stays_high": int(m8_25 >= m8_all - 0.10), "cub_source": "results/m9_sweep_cub200.csv (m9_contractive, U1, gamma=1, n_clients=10)"})
    holds = int(all(p["holds_eps"] for p in pred) and pred[0]["m8_stays_high"])
    for p in pred:
        p["prediction_holds"] = holds
        print(f"R4 PRED eps={p['eps']}: cifar m25={p['cifar_m25_mean']:.4f} all={p['cifar_all_mean']:.4f} cub={p['cub_paper_mean']:.4f} "
              f"holds_eps={p['holds_eps']} | M8 m25={m8_25:.4f} all={m8_all:.4f} stays_high={p['m8_stays_high']} -> prediction_holds={holds}")

    out = REPO_ROOT / "results" / "rb_examples_per_class.csv" if not failures else REPO_ROOT / "rebuttal" / "r4_FAILED.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for msg in failures:
        print("R4 FAILURE:", msg)
    print(f"R4 rows={len(rows)} failures={len(failures)} -> {out}")
    if failures:
        return 1
    out_pred = REPO_ROOT / "results" / "rb_examples_per_class_prediction.csv"
    with open(out_pred, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred[0]))
        w.writeheader()
        w.writerows(pred)
    provenance.finalize(man, [out, out_pred])
    return 0


if __name__ == "__main__":
    sys.exit(main())
