#!/usr/bin/env python3
"""Rebuttal R0 (`rebuttal/REBUTTAL_EXPERIMENTS.md` §3.4): seed-matched reproduction check.

Re-runs, with the unchanged sweep code (`run_m9_sweep._run_m9` / `_run_reference` and the selected
hyperparameters from `results/m9_hparam_selection.csv`), the following committed rows of
`results/m9_sweep_<dataset>.csv`, and compares them column by column:

- M8 non-private, all three datasets, seeds 0-2;
- M9, U1, gamma = 1, eps_0 = 4, CIFAR-100, seeds 0-2;
- M0, CIFAR-100, seeds 0-2.

Pass = every compared value within 1e-6 of the committed one, and the M9 (p, lambda, clip_C) used
equal to the committed row's. Writes `rebuttal/r0_check.csv` (never `results/`). PBS only.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))

from p3fcl import features, provenance  # noqa: E402
from p3fcl.methods.m0_fedavg import FedAvgSequential  # noqa: E402
from p3fcl.methods.m8_analytic import AnalyticFCL  # noqa: E402
from run_m9_sweep import (  # noqa: E402
    BACKBONE, DELTA, N_CLIENTS_DEFAULT, _load_hparam_table, _pca_cache, _run_m9, _run_reference,
)

TOL = 1e-6
SEEDS = [0, 1, 2]
DATASETS = ["cifar100", "cub200", "imagenet_r"]
METRICS = ["final_avg_acc", "bwt", "avg_incremental_acc"]
OUT_CSV = REPO_ROOT / "rebuttal" / "r0_check.csv"


def _committed(dataset: str) -> list[dict]:
    with open(REPO_ROOT / "results" / f"m9_sweep_{dataset}.csv", newline="") as f:
        return list(csv.DictReader(f))


def _find(rows, method, seed, unit="n/a", eps=None, gamma="n/a", n_clients=N_CLIENTS_DEFAULT):
    hits = [
        r for r in rows
        if r["method"] == method and int(r["seed"]) == seed and r["unit"] == unit
        and int(r["n_clients"]) == n_clients
        and (gamma == "n/a" and r["gamma"] == "n/a" or gamma != "n/a" and r["gamma"] != "n/a" and float(r["gamma"]) == gamma)
        and (eps is None or float(r["eps"]) == eps)
    ]
    if len(hits) != 1:
        raise RuntimeError(f"expected 1 committed row for {method} seed={seed} unit={unit} eps={eps}, found {len(hits)}")
    return hits[0]


def main() -> int:
    out_rows = []

    def record(dataset, method, seed, committed, new, extra):
        for m in METRICS:
            c, n = float(committed[m]), float(new[m])
            diff = abs(c - n)
            out_rows.append({
                "dataset": dataset, "method": method, "seed": seed, **extra, "metric": m,
                "committed": repr(c), "rerun": repr(n), "abs_diff": repr(diff),
                "pass": int(diff <= TOL and math.isfinite(n)),
            })
            print(f"R0 {dataset} {method} seed={seed} {m}: committed={c!r} rerun={n!r} diff={diff:.3e}")

    for dataset in DATASETS:
        train = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, "train")
        X, y = train["features"], train["labels"]
        test = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, "test")
        X_test, y_test = test["features"], test["labels"]
        n_classes, feature_dim = int(y.max()) + 1, X.shape[1]
        rows = _committed(dataset)
        no_hp = {"unit": "n/a", "eps": "inf", "gamma": "n/a", "p": "n/a", "lambda": "n/a", "clip_C": "n/a", "hparams_match": "n/a"}

        for seed in SEEDS:
            new = _run_reference(AnalyticFCL, {"ridge_lambda": 1.0}, X, y, X_test, y_test, n_classes, feature_dim, N_CLIENTS_DEFAULT, seed)
            record(dataset, "m8_analytic", seed, _find(rows, "m8_analytic", seed), new, no_hp)

        if dataset != "cifar100":
            continue

        for seed in SEEDS:
            new = _run_reference(FedAvgSequential, {"local_epochs": 30, "lr": 0.5}, X, y, X_test, y_test, n_classes, feature_dim, N_CLIENTS_DEFAULT, seed)
            record(dataset, "m0_fedavg", seed, _find(rows, "m0_fedavg", seed), new, no_hp)

        ref = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, "ref")
        p, lam, clip_c = _load_hparam_table(dataset)[("U1", 4.0)]
        P = _pca_cache(dataset, ref["features"])(p)
        for seed in SEEDS:
            cfg = {"gamma": 1.0, "eps": 4.0, "delta": DELTA, "ridge_lambda": lam, "unit": "U1", "clip_C": clip_c}
            new = _run_m9(X, y, X_test, y_test, n_classes, P, cfg, dataset, N_CLIENTS_DEFAULT, seed)
            com = _find(rows, "m9_contractive", seed, unit="U1", eps=4.0, gamma=1.0)
            match = int(int(com["p"]) == p and float(com["lambda"]) == lam and abs(float(com["clip_C"]) - clip_c) <= 1e-9)
            record(dataset, "m9_contractive", seed, com, new,
                   {"unit": "U1", "eps": 4.0, "gamma": 1.0, "p": p, "lambda": lam, "clip_C": repr(clip_c), "hparams_match": match})
            if not match:
                out_rows[-1]["pass"] = 0

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    n_pass = sum(r["pass"] for r in out_rows)
    max_diff = max(float(r["abs_diff"]) for r in out_rows)
    print(f"R0 SUMMARY compared={len(out_rows)} passed={n_pass} max_abs_diff={max_diff:.3e} tol={TOL}")

    config = {"seed": 0, "purpose": "rebuttal R0 seed-matched reproduction check", "datasets": DATASETS,
              "seeds": SEEDS, "tol": TOL, "compared_against": "results/m9_sweep_<dataset>.csv"}
    provenance.finalize(provenance.run_manifest(config, seed=0), [OUT_CSV])
    return 0 if n_pass == len(out_rows) else 1


if __name__ == "__main__":
    sys.exit(main())
