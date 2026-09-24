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
fixed horizon; a `T=50` panel needs a separate utility sweep). FX9 fills audited epsilon only
for the actually audited CIFAR/U1 epsilon=1,4,infinity cells, retaining explicit
not-audited labels elsewhere. eps_analytical denotes the nominal mechanism
calibration; the separately scoped FX1 composition values are in TAB08.

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

    audit_path = REPO_ROOT / "results" / "m9_audit_summary.csv"
    with audit_path.open() as f:
        audits = {(r["dataset"], r["unit"], float(r["eps"])): r for r in csv.DictReader(f)}
    rows_out = []

    # M9: unit in {U1, U2}, gamma=1.0, n_clients=10 (main grid only, not the n_clients-scaling arm).
    m9_rows = [
        r for r in rows if r["method"] == "m9_contractive" and r["gamma"] == "1.0" and r["n_clients"] == "10"
    ]
    groups: dict = {}
    for r in m9_rows:
        key = (r["dataset"], r["unit"], float(r["eps"]))
        groups.setdefault(key, []).append(r)
    for (dataset, unit, eps), group in groups.items():
        final_accs = [float(r["final_avg_acc"]) for r in group]
        ci_lo, ci_hi = _ci(final_accs)
        audit = audits.get((dataset, unit, eps))
        audited_lb = float(audit["eps_lb"]) if audit and audit.get("eps_lb") else float("nan")
        for r in group:
            rows_out.append(
                {
                    "dataset": dataset,
                    "method": f"m9_contractive_{unit}",
                    "T": T_FIXED,
                    "eps_target": eps,
                    "eps_analytical": eps,
                    "eps_audited_lb": audited_lb,
                    "eps_audited_ci_lo": audited_lb,
                    "eps_audited_ci_hi": float("inf") if audit else float("nan"),
                    "audit_status": "pointwise empirical lower bound"
                    if audit
                    else "not audited for this dataset/unit/epsilon",
                    "final_acc": float(r["final_avg_acc"]),
                    "final_acc_mean": float(np.mean(final_accs)),
                    "bwt": float(r["bwt"]),
                    "seed": int(r["seed"]),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                }
            )

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
                rows_out.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "T": T_FIXED,
                        "eps_target": float("inf"),
                        "eps_analytical": float("inf"),
                        "eps_audited_lb": "",
                        "eps_audited_ci_lo": "",
                        "eps_audited_ci_hi": "",
                        "audit_status": "non-private reference; not this audit mechanism",
                        "final_acc": float(r["final_avg_acc"]),
                        "final_acc_mean": float(np.mean(final_accs)),
                        "bwt": float(r["bwt"]),
                        "seed": int(r["seed"]),
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                    }
                )

    out_csv = REPO_ROOT / "results" / "fig08_pareto.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {len(rows_out)} rows to {out_csv}")
    matched = [r for r in rows_out if r["audit_status"] == "pointwise empirical lower bound"]
    assert len(matched) == 9 and all(np.isfinite(r["eps_audited_lb"]) for r in matched)
    print(
        f"FX9-7 FIG08 ACCEPT audited_seed_rows={len(matched)} audited_cells=3 blank_supported_audit_cells=0; unsupported cells explicitly labelled"
    )

    config = {"seed": 0, "purpose": "FX5 FIG08 v2 privacy-utility Pareto (claim C4)"}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
