#!/usr/bin/env python3
"""FIG05 (H1, claim C3): eps(T) by unit of privacy, computed by `dp.accountant.account` on REAL
method ledgers (not synthetic) at matched sigma, for units U1-U5, T over a log-spaced grid up to
1000. `04_METHODS_AND_ATTACKS.md`'s own stated order: "no new science, produces a required figure
from work that is already written" -- `dp.accountant` was built and unit-tested in P0; this script
just points it at real ledgers instead of synthetic ones.

One representative real-data run per method (CIFAR-100, 10 tasks, seed 0 -- matching
`run_utility_baseline.py`'s registry so the ledgers are directly comparable to TAB05's numbers for
the same configuration), because `sim.run`'s ledger is not persisted by that script and needs
regenerating here rather than invented.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, provenance, sim, streams  # noqa: E402
from p3fcl.dp.accountant import account, check_disjointness  # noqa: E402
from p3fcl.units import Unit  # noqa: E402

# reuse the exact same method registry/config as run_utility_baseline.py for comparability
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))
from run_utility_baseline import BACKBONE, DATASET_SPEC, METHOD_REGISTRY, _standardize  # noqa: E402

SIGMA = 2.0
DELTA = 1e-5
T_GRID = [1, 5, 10, 50, 100, 500, 1000]
DATASET = "cifar100"
N_TASKS = 10
SEED = 0


def _ledger_hash(ledger) -> str:
    h = hashlib.sha256()
    for rec in ledger:
        h.update(f"{rec.round}|{rec.task}|{rec.client}|{rec.family.value}|{sorted(rec.touched)}".encode())
    return h.hexdigest()[:16]


def main() -> int:
    spec = DATASET_SPEC[DATASET]
    cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    (X_std,) = _standardize(X)
    idx = np.arange(len(y))
    stream = streams.build_stream(y, idx, n_tasks=N_TASKS, n_clients=spec["n_clients"], beta=spec["beta"], seed=SEED)

    rows = []
    for short, (cls, base_cfg) in METHOD_REGISTRY.items():
        cfg = {**base_cfg, "n_classes": spec["n_classes"], "feature_dim": X_std.shape[1]}
        method = cls(cfg)
        result = sim.run(method, X_std, y, stream, seed=SEED)
        ledger = result["ledger"]
        lhash = _ledger_hash(ledger)
        report = check_disjointness(ledger)
        print(f"{short}: ledger_hash={lhash} task_disjoint={report.task_disjoint} violations={report.violations}")

        for unit in Unit:
            lr = account(ledger, sigma=SIGMA, unit=unit, delta=DELTA)
            for T in T_GRID:
                eps = lr.eps_of_T(T)
                rows.append({
                    "method": short, "ledger_hash": lhash, "unit": unit.value, "sigma": SIGMA,
                    "delta": DELTA, "T": T, "eps": eps, "regime": lr.regime,
                    "task_disjoint": report.task_disjoint, "violations": ";".join(report.violations),
                })

    out_csv = REPO_ROOT / "results" / "fig05_eps_of_T.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {"seed": SEED, "purpose": "FIG05 eps(T) on real ledgers", "dataset": DATASET, "n_tasks": N_TASKS, "sigma": SIGMA}
    manifest = provenance.run_manifest(config, seed=SEED)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
