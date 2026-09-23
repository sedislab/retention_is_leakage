#!/usr/bin/env python3
"""FX5 (`08_FIX_PLAN.md` §8 Outputs): `results/fig08_pareto.csv` from the already-computed
`results/m9_sweep.csv` (CLAUDE.md non-negotiable #3 -- no recomputation, `merge_m9_sweep.py`'s output
is the only input here). Schema per `03_RESULTS_SPEC.md`'s FIG08 row:
`dataset, method, T, eps_target, eps_analytical, eps_audited_lb, eps_audited_ci_lo, eps_audited_ci_hi,
final_acc, bwt, seed, ci_lo, ci_hi` -- one row per seed, `ci_lo`/`ci_hi` are the across-seed CI on
`final_acc` for that (dataset, method, T, eps_target) group, repeated on every row of the group (the
same denormalized convention `03_RESULTS_SPEC.md` uses for FIG06/07/09).

**Reduced scope, matching `08_FIX_PLAN.md` §8's own restated FIG08 v2 (not the more ambitious
original spec)**: only M9 (unit in {U1, U2}, gamma=1.0) plus the M0/M8 non-private reference lines --
no DP-SGD-FedAvg/linear-probe baselines (P2 stretch, not built) and only `T=10` (the core sweep's
fixed horizon; a `T=50` panel needs the 50-task stretch run, also not built). `eps_audited_lb/
eps_audited_ci_lo/eps_audited_ci_hi` are left blank -- the M9 LiRA audit (FIG10 v2) hasn't run yet;
`eps_analytical` is the nominal `eps` config value passed to the noise mechanism, not a re-derived
FX1-accountant-certified epsilon at a given T (that's TAB08's own, separate "certified eps at
T=10/50" columns, not yet built either).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402

T_FIXED = 10


def _ci(values: list) -> tuple:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    if sem == 0:
        return (mean, mean)
    lo, hi = stats.t.interval(0.95, df=len(values) - 1, loc=mean, scale=sem)
    return (float(lo), float(hi))


def main() -> int:
    sweep_csv = REPO_ROOT / "results" / "m9_sweep.csv"
    with open(sweep_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    rows_out = []

    # M9: unit in {U1, U2}, gamma=1.0, n_clients=10 (main grid only, not the n_clients-scaling arm).
    m9_rows = [r for r in rows if r["method"] == "m9_contractive" and r["gamma"] == "1.0" and r["n_clients"] == "10"]
    groups: dict = {}
    for r in m9_rows:
        key = (r["dataset"], r["unit"], float(r["eps"]))
        groups.setdefault(key, []).append(r)
    for (dataset, unit, eps), group in groups.items():
        final_accs = [float(r["final_avg_acc"]) for r in group]
        ci_lo, ci_hi = _ci(final_accs)
        for r in group:
            rows_out.append({
                "dataset": dataset, "method": f"m9_contractive_{unit}", "T": T_FIXED,
                "eps_target": eps, "eps_analytical": eps, "eps_audited_lb": "",
                "eps_audited_ci_lo": "", "eps_audited_ci_hi": "",
                "final_acc": float(r["final_avg_acc"]), "bwt": float(r["bwt"]),
                "seed": int(r["seed"]), "ci_lo": ci_lo, "ci_hi": ci_hi,
            })

    # Non-private references (M0, M8), one horizontal-line value per dataset.
    for method in ("m0_fedavg", "m8_analytic"):
        ref_rows = [r for r in rows if r["method"] == method]
        by_dataset: dict = {}
        for r in ref_rows:
            by_dataset.setdefault(r["dataset"], []).append(r)
        for dataset, group in by_dataset.items():
            final_accs = [float(r["final_avg_acc"]) for r in group]
            ci_lo, ci_hi = _ci(final_accs)
            for r in group:
                rows_out.append({
                    "dataset": dataset, "method": method, "T": T_FIXED,
                    "eps_target": float("inf"), "eps_analytical": float("inf"), "eps_audited_lb": "",
                    "eps_audited_ci_lo": "", "eps_audited_ci_hi": "",
                    "final_acc": float(r["final_avg_acc"]), "bwt": float(r["bwt"]),
                    "seed": int(r["seed"]), "ci_lo": ci_lo, "ci_hi": ci_hi,
                })

    out_csv = REPO_ROOT / "results" / "fig08_pareto.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {len(rows_out)} rows to {out_csv}")

    config = {"seed": 0, "purpose": "FX5 FIG08 v2 privacy-utility Pareto (claim C4)"}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
