#!/usr/bin/env python3
"""FX9 real raw-feature method ledgers -> per-method FX1 accountant curves.
M9 uses its separate actual-noise producer; merge_fx9_fig05 combines only these
per-method products. No changes to the FX1 accountant itself.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, sim, streams  # noqa: E402
from p3fcl.dp.accountant import account, check_disjointness, extrapolate_lifelong  # noqa: E402
from p3fcl.units import Unit  # noqa: E402

# reuse the exact same method registry/config as run_utility_baseline.py for comparability
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))
from fx9_io import write  # noqa: E402
from p3fcl.experiment import METHODS, attacked_method_config  # noqa: E402
from p3fcl.shadow_runner import BACKBONE, METHOD_REGISTRY  # noqa: E402

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
    spec = {"n_classes": 100, "n_clients": 10, "beta": 0.5}
    cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    idx = np.arange(len(y))
    stream = streams.build_stream(
        y, idx, n_tasks=N_TASKS, n_clients=spec["n_clients"], beta=spec["beta"], seed=SEED
    )

    for method_id in METHODS:
        rows = []
        short = method_id.split("_")[0].upper()
        cls = METHOD_REGISTRY[method_id][0]
        cfg = attacked_method_config(method_id, spec["n_classes"], X.shape[1], DATASET)
        method = cls(cfg)
        result = sim.run(method, X, y, stream, seed=SEED)
        ledger = result["ledger"]
        lhash = _ledger_hash(ledger)
        report = check_disjointness(ledger)
        print(
            f"{short}: ledger_hash={lhash} task_disjoint={report.task_disjoint} violations={report.violations}"
        )

        for unit in UNITS:
            df = account(ledger, stream, unit=unit, sigma=SIGMA, delta=DELTA, window=WINDOW)
            ext = extrapolate_lifelong(df, sigma=SIGMA, delta=DELTA, T_max=T_MAX)
            for _, r in ext.iterrows():
                rows.append(
                    {
                        "method": short,
                        "dataset": DATASET,
                        "n_tasks": N_TASKS,
                        "ledger_hash": lhash,
                        "unit": r["unit"],
                        "window_W": r["window"],
                        "sigma": SIGMA,
                        "delta": DELTA,
                        "T": int(r["T"]),
                        "m_T": r["m_T"],
                        "m_T_passes": r["m_T_passes"],
                        "eps": r["eps"],
                        "regime": r["regime"],
                        "observed": int(r["observed"]),
                    }
                )

        out_csv = REPO_ROOT / "results" / f"fig05_accountant_{method_id}.csv"
        write(
            out_csv,
            rows,
            dict(
                phase="FX9-7",
                hypothesis="H1",
                dataset=DATASET,
                method=method_id,
                n_tasks=N_TASKS,
                sigma=SIGMA,
                delta=DELTA,
                window=WINDOW,
                method_config=cfg,
            ),
        )
        print(f"wrote {len(rows)} rows to {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
