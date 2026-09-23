#!/usr/bin/env python3
"""FIG05 v2 (H1, claim C3; FX1, `08_FIX_PLAN.md` §6): eps(T) by unit of privacy, computed by
`dp.accountant.account`/`extrapolate_lifelong` on REAL post-fix method ledgers, for units U1-U4 (U5 is
identical to U1 under this project's one-person-one-example renewal model -- not computed separately,
see `p3fcl.units.neighbouring(Unit.INDIVIDUAL)`), extrapolated out to T=1000.

One representative real-data run per method (CIFAR-100, 10 tasks, seed 0 -- matching
`run_utility_baseline.py`'s registry so the ledgers are directly comparable to TAB05's numbers for the
same configuration), because `sim.run`'s ledger is not persisted by that script and needs regenerating
here rather than invented.

M9 is not in `METHOD_REGISTRY` yet (FX5, `08_FIX_PLAN.md` §8, not landed as of this run) -- its row is
skipped, and `analysis/fig05_eps_of_T.py` draws its FIG05 v2 panel as "pending FX5" rather than leaving
a silently-empty slot. Likewise the 50-task CIFAR ledgers FX1 asks for "if FX5 produced them" are not
generated yet.
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
from p3fcl.dp.accountant import account, check_disjointness, extrapolate_lifelong  # noqa: E402
from p3fcl.units import Unit  # noqa: E402

# reuse the exact same method registry/config as run_utility_baseline.py for comparability
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))
from run_utility_baseline import BACKBONE, DATASET_SPEC, METHOD_REGISTRY, _standardize  # noqa: E402

SIGMA = 2.0
DELTA = 1e-5
WINDOW = 3
T_MAX = 1000
DATASET = "cifar100"
N_TASKS = 10
SEED = 0

# U5 (INDIVIDUAL) is identical to U1 (EXAMPLE) under this project's renewal model -- account() itself
# routes Unit.INDIVIDUAL to the U1 computation, so there is nothing distinct to compute or plot for it.
UNITS = (Unit.EXAMPLE, Unit.TASK, Unit.CLIENT_BOUNDED, Unit.CLIENT_LIFELONG)


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

        for unit in UNITS:
            df = account(ledger, stream, unit=unit, sigma=SIGMA, delta=DELTA, window=WINDOW)
            ext = extrapolate_lifelong(df, sigma=SIGMA, delta=DELTA, T_max=T_MAX)
            for _, r in ext.iterrows():
                rows.append({
                    "method": short, "dataset": DATASET, "n_tasks": N_TASKS, "ledger_hash": lhash,
                    "unit": r["unit"], "window_W": r["window"], "sigma": SIGMA, "delta": DELTA,
                    "T": int(r["T"]), "m_T": r["m_T"], "m_T_passes": r["m_T_passes"], "eps": r["eps"],
                    "regime": r["regime"], "observed": int(r["observed"]),
                })

    out_csv = REPO_ROOT / "results" / "fig05_eps_of_T.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {
        "seed": SEED, "purpose": "FIG05 v2 eps(T) on real post-fix ledgers (FX1)", "dataset": DATASET,
        "n_tasks": N_TASKS, "sigma": SIGMA, "delta": DELTA, "window": WINDOW, "T_max": T_MAX,
    }
    manifest = provenance.run_manifest(config, seed=SEED)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
